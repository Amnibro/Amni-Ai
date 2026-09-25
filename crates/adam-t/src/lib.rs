pub mod lut;
use adam_ray::rng::Pcg64;
use burn::{module::{AutodiffModule, Module, Param}, nn::loss::CrossEntropyLossConfig, optim::{grad_clipping::GradientClippingConfig, AdamWConfig, GradientsParams, Optimizer}, record::{BinFileRecorder, FullPrecisionSettings, Recorder}, tensor::{activation::relu, backend::{AutodiffBackend, Backend}, module::{attention, embedding}, ops::AttentionModuleOptions, Distribution, Int, Tensor, TensorData}};
use memmap2::Mmap;
use std::{env::var, fs, io::{self, Write}, path::Path, time::Instant};
#[derive(Clone, Debug)]
pub struct Cfg { pub d: usize, pub l: usize, pub h: usize, pub t: usize, pub ff: usize, pub ns: Vec<usize>, pub rb: u32, pub tern: bool, pub mem: bool, pub bs: usize, pub steps: usize, pub lr: f64, pub lrm: f64, pub wd: f32, pub warm: usize, pub seed: u64, pub ev: usize, pub ck: usize, pub evb: usize, pub tag: String, pub home: String, pub out: String, pub thf: String, pub corpus: String, pub par: usize, pub unit: f64, pub wall: f64 }
impl Cfg {
    pub fn env() -> Cfg {
        let e = |k: &str, d: &str| var(k).unwrap_or_else(|_| d.into());
        let p = |k: &str, d: &str| e(k, d).trim().parse::<f64>().unwrap_or_else(|_| panic!("bad {k}"));
        let (home, d, tag) = (e("AMNI_HOME", "/home/anmire/ai/Amni-Ai"), p("D", "256") as usize, e("TAG", "smoke"));
        Cfg { d, l: p("L", "8") as usize, h: p("H", "4") as usize, t: p("CTX", "256") as usize, ff: p("FF", &(4 * d).to_string()) as usize, ns: e("NGRAMS", "2,3,4,5,6,8").split(',').filter(|s| !s.is_empty()).map(|s| s.trim().parse().unwrap()).collect(), rb: p("MEM_LOG2", "15") as u32, tern: e("TERN", "1") == "1", mem: e("MEM", "0") == "1", bs: p("BS", "64") as usize, steps: p("STEPS", "2000") as usize, lr: p("LR", "2e-3"), lrm: p("LR_MEM", &e("LR", "2e-3")), wd: p("WD", "0.05") as f32, warm: p("WARM", "200") as usize, seed: p("SEED", "0") as u64, ev: p("EVAL_EVERY", "500") as usize, ck: p("CKPT_EVERY", "500") as usize, evb: p("EVAL_BATCHES", "20") as usize, out: e("OUT", &format!("{home}/exports/adam_t/{tag}")), tag, thf: e("THROTTLE_FILE", &format!("{home}/throttle.txt")), corpus: e("CORPUS", "clean2"), par: p("PARITY_WINDOWS", "8") as usize, unit: p("THROTTLE_UNIT_MS", "50"), wall: p("MAX_MIN", "19"), home }
    }
    pub fn from_dir(dir: &str) -> io::Result<Cfg> {
        let l = fs::read_to_string(format!("{dir}/log.jsonl"))?.lines().next().unwrap_or("").to_string();
        let j: serde_json::Value = serde_json::from_str(&l).map_err(io::Error::other)?;
        let (g, u) = (&j["cfg"], |k: &str| j["cfg"][k].as_u64().ok_or_else(|| io::Error::other(format!("cfg.{k} missing in {dir}/log.jsonl"))).map(|v| v as usize));
        Ok(Cfg { d: u("d")?, l: u("l")?, h: u("h")?, t: u("ctx")?, ff: u("ff")?, rb: u("mem_log2")? as u32, tern: g["tern"].as_bool().unwrap_or(true), mem: g["mem"].as_bool().unwrap_or(false), ns: g["ngrams"].as_array().map(|a| a.iter().filter_map(|v| v.as_u64().map(|v| v as usize)).collect()).unwrap_or_default(), out: dir.into(), ..Cfg::env() })
    }
    pub fn k(&self) -> usize { if self.mem { self.ns.len() } else { 0 } }
    pub fn rows(&self) -> usize { self.k() << self.rb }
    pub fn mult(&self, it: usize) -> f64 { let pr = ((it as f64 - self.warm as f64) / (self.steps.saturating_sub(self.warm)).max(1) as f64).clamp(0.0, 1.0); if it < self.warm { (it as f64 + 1.0) / self.warm as f64 } else { 0.1 + 0.45 * (1.0 + (std::f64::consts::PI * pr).cos()) } }
    pub fn core_params(&self) -> usize { 256 * self.d + self.t * self.d + self.l * (4 * self.d * self.d + 2 * self.d * self.ff) + self.d * 256 }
    pub fn json(&self) -> String { format!("{{\"d\": {}, \"l\": {}, \"h\": {}, \"ctx\": {}, \"ff\": {}, \"tern\": {}, \"mem\": {}, \"ngrams\": {:?}, \"mem_log2\": {}, \"bs\": {}, \"steps\": {}, \"lr\": {}, \"lr_mem\": {}, \"wd\": {}, \"warm\": {}, \"seed\": {}, \"corpus\": \"{}\", \"core_params\": {}, \"mem_params\": {}, \"bytes_per_step\": {}}}", self.d, self.l, self.h, self.t, self.ff, self.tern, self.mem, self.ns, self.rb, self.bs, self.steps, self.lr, self.lrm, self.wd, self.warm, self.seed, self.corpus, self.core_params(), self.rows() * self.d, self.bs * self.t) }
}
#[derive(Module, Debug)]
pub struct Lin<B: Backend> { pub w: Param<Tensor<B, 2>> }
#[derive(Module, Debug)]
pub struct Blk<B: Backend> { pub qkv: Lin<B>, pub o: Lin<B>, pub f1: Lin<B>, pub f2: Lin<B> }
#[derive(Module, Debug)]
pub struct Core<B: Backend> { pub emb: Param<Tensor<B, 2>>, pub pos: Param<Tensor<B, 2>>, pub blk: Vec<Blk<B>>, pub head: Lin<B> }
#[derive(Module, Debug)]
pub struct Mem<B: Backend> { pub tab: Param<Tensor<B, 2>> }
fn rnd<B: Backend>(s: [usize; 2], sd: f64, dev: &B::Device) -> Param<Tensor<B, 2>> { Param::from_tensor(Tensor::random(s, Distribution::Normal(0.0, sd), dev)) }
pub fn nrm<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> { x.clone() / x.square().mean_dim(1).add_scalar(1e-6).sqrt() }
pub fn tq<B: Backend>(w: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) { let g = w.clone().abs().mean().reshape([1, 1]).add_scalar(1e-6); let h = g.clone().mul_scalar(0.5); (((w.clone() - h.clone()).sign() + (w + h).sign()).mul_scalar(0.5).round(), g) }
impl<B: Backend> Lin<B> {
    pub fn new(i: usize, o: usize, sd: f64, dev: &B::Device) -> Self { Lin { w: rnd([i, o], sd, dev) } }
    pub fn fwd(&self, x: Tensor<B, 2>, q: bool) -> Tensor<B, 2> {
        let (xn, w) = (nrm(x), self.w.val());
        if !q { return xn.matmul(w); }
        let m = xn.clone().abs().max_dim(1).clamp_min(1e-5).detach();
        let xs = xn * m.clone().recip().mul_scalar(127.0);
        let xi = xs.clone() + (xs.clone().round().clamp(-128.0, 127.0) - xs).detach();
        let (t, g) = tq(w.clone().detach());
        let a = w / g.clone();
        xi.matmul(a.clone() + (t - a).detach()) * (m * g.mul_scalar(1.0 / 127.0))
    }
}
impl<B: Backend> Core<B> {
    pub fn new(c: &Cfg, dev: &B::Device) -> Self { let so = 0.02 / (2.0 * c.l as f64).sqrt(); Core { emb: rnd([256, c.d], 0.02, dev), pos: rnd([c.t, c.d], 0.02, dev), blk: (0..c.l).map(|_| Blk { qkv: Lin::new(c.d, 3 * c.d, 0.02, dev), o: Lin::new(c.d, c.d, so, dev), f1: Lin::new(c.d, c.ff, 0.02, dev), f2: Lin::new(c.ff, c.d, so, dev) }).collect(), head: Lin::new(c.d, 256, 0.02, dev) } }
    pub fn lins(&self) -> Vec<&Lin<B>> { self.blk.iter().flat_map(|b| [&b.qkv, &b.o, &b.f1, &b.f2]).chain([&self.head]).collect() }
    pub fn fwd(&self, ids: Tensor<B, 2, Int>, mem: Option<Tensor<B, 3>>, h: usize, q: bool) -> Tensor<B, 2> {
        let ([b, t], d) = (ids.dims(), self.emb.val().dims()[1]);
        let x = embedding(self.emb.val(), ids) + self.pos.val().narrow(0, 0, t).unsqueeze::<3>();
        let x = mem.map_or(x.clone(), |m| x + m).reshape([b * t, d]);
        let x = self.blk.iter().fold(x, |x, k| {
            let a = k.qkv.fwd(x.clone(), q).reshape([b, t, 3 * h, d / h]).swap_dims(1, 2);
            let o = attention(a.clone().narrow(1, 0, h), a.clone().narrow(1, h, h), a.narrow(1, 2 * h, h), None, None, AttentionModuleOptions { scale: None, softcap: None, is_causal: true }).swap_dims(1, 2).reshape([b * t, d]);
            let x = x + k.o.fwd(o, q);
            x.clone() + k.f2.fwd(relu(k.f1.fwd(x, q)).square(), q)
        });
        self.head.fwd(x, q)
    }
}
impl<B: Backend> Mem<B> {
    pub fn new(c: &Cfg, dev: &B::Device) -> Self { Mem { tab: rnd([c.rows(), c.d], 0.02, dev) } }
    pub fn fwd(&self, ix: Tensor<B, 2, Int>, k: usize) -> Tensor<B, 3> { let [b, tk] = ix.dims(); let d = self.tab.val().dims()[1]; embedding(self.tab.val(), ix).reshape([b, tk / k, k, d]).sum_dim(2).reshape([b, tk / k, d]) }
}
pub fn mmap(p: &str) -> io::Result<Mmap> { unsafe { Mmap::map(&fs::File::open(p)?) } }
pub fn hsh(b: &[u8], n: usize) -> u64 { let mut h = (n as u64) << 56 ^ (b.len() as u64) << 48 ^ 0x9E37_79B9_7F4A_7C15; b.iter().for_each(|&c| h = (h ^ c as u64).wrapping_mul(0x0000_0100_0000_01B3).rotate_left(23)); h ^= h >> 33; h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD); h ^= h >> 33; h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53); h ^ h >> 33 }
pub fn hrow(x: &[u8], p: usize, k: usize, n: usize, rb: u32) -> usize { k << rb | (hsh(&x[p + 1 - n.min(p + 1)..=p], n) >> (64 - rb)) as usize }
pub fn hix(x: &[u8], b: usize, t: usize, ns: &[usize], rb: u32) -> Vec<i32> { (0..b * t).flat_map(|i| { let s = &x[i / t * t..i / t * t + t]; ns.iter().enumerate().map(move |(k, &n)| hrow(s, i % t, k, n, rb) as i32) }).collect() }
fn sm(mut z: u64) -> u64 { z = z.wrapping_add(0x9E37_79B9_7F4A_7C15); z = (z ^ z >> 30).wrapping_mul(0xBF58_476D_1CE4_E5B9); z = (z ^ z >> 27).wrapping_mul(0x94D0_49BB_1331_11EB); z ^ z >> 31 }
pub fn tb(d: &[u8], seed: u64, it: usize, bs: usize, t: usize) -> (Vec<u8>, Vec<u8>) { let ix: Vec<usize> = (0..bs).map(|j| (sm(seed << 40 ^ (it as u64) << 12 ^ j as u64) % (d.len() - t - 1) as u64) as usize).collect(); (ix.iter().flat_map(|&i| d[i..i + t].iter().copied()).collect(), ix.iter().flat_map(|&i| d[i + 1..i + 1 + t].iter().copied()).collect()) }
pub fn it2<B: Backend>(v: Vec<i32>, s: [usize; 2], dev: &B::Device) -> Tensor<B, 2, Int> { Tensor::from_data(TensorData::new(v, s), dev) }
pub fn logits<B: Backend>(c: &Cfg, core: &Core<B>, mem: Option<&Mem<B>>, x: &[u8], b: usize, dev: &B::Device) -> Tensor<B, 2> { core.fwd(it2(x.iter().map(|&v| v as i32).collect(), [b, c.t], dev), mem.map(|m| m.fwd(it2(hix(x, b, c.t, &c.ns, c.rb), [b, c.t * c.k()], dev), c.k())), c.h, c.tern) }
pub fn infer<B: Backend>(c: &Cfg, core: &Core<B>, mem: Option<&Mem<B>>, x: &[u8], b: usize, dev: &B::Device) -> Vec<f32> { let n = x.len() / b; f1(core.fwd(it2(x.iter().map(|&v| v as i32).collect(), [b, n], dev), mem.map(|m| m.fwd(it2(hix(x, b, n, &c.ns, c.rb), [b, n * c.k()], dev), c.k())), c.h, c.tern)) }
pub fn load<B: Backend>(c: &Cfg, dev: &B::Device) -> io::Result<(Core<B>, Option<Mem<B>>)> { let (rec, le) = (BinFileRecorder::<FullPrecisionSettings>::new(), |e: burn::record::RecorderError| io::Error::other(format!("{e:?}"))); Ok((Core::<B>::new(c, dev).load_file(format!("{}/ckpt/core", c.out), &rec, dev).map_err(le)?, c.mem.then(|| Mem::<B>::new(c, dev).load_file(format!("{}/ckpt/mem", c.out), &rec, dev).map_err(le)).transpose()?)) }
pub fn ce<B: Backend>(lg: Tensor<B, 2>, y: &[u8], dev: &B::Device) -> Tensor<B, 1> { CrossEntropyLossConfig::new().init(dev).forward(lg, Tensor::from_data(TensorData::new(y.iter().map(|&v| v as i32).collect::<Vec<i32>>(), [y.len()]), dev)) }
pub fn f1<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> { t.into_data().iter::<f32>().collect() }
pub fn eval<B: Backend>(c: &Cfg, core: &Core<B>, mem: Option<&Mem<B>>, hold: &[u8], nb: usize, dev: &B::Device) -> f64 {
    let (mut g, th) = (Pcg64::new(Some(1234)), thr(&c.thf));
    let v: Vec<f64> = (0..nb).map(|_| { let (x, y) = adam_train::batch(hold, &mut g, 128, c.t, None); (0..4).map(|q| { let t0 = Instant::now(); let r = f1(ce(logits(c, core, mem, &x[q * 32 * c.t..(q + 1) * 32 * c.t], 32, dev), &y[q * 32 * c.t..(q + 1) * 32 * c.t], dev))[0] as f64; nap(th, t0.elapsed().as_secs_f64() * 1e3, c.unit); r }).sum::<f64>() / 4.0 }).collect();
    adam_ray::rng::pairwise(&v) / nb as f64
}
pub fn thr(p: &str) -> f64 { fs::read_to_string(p).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0.0) }
pub fn nap(th: f64, ms: f64, unit: f64) -> f64 { let s = if th > 0.0 { th * 1e3 * (ms / unit).max(1.0) } else { 0.0 }; (s > 0.0).then(|| std::thread::sleep(std::time::Duration::from_secs_f64(s / 1e3))); s }
pub fn train<B: AutodiffBackend>(c: &Cfg, dev: B::Device, bench: usize) -> io::Result<String> {
    let d = format!("{}/data/{}", c.home, c.corpus);
    let (tr, ho) = (mmap(&format!("{d}/train.bin"))?, mmap(&format!("{d}/hold.bin"))?);
    Path::new(&c.thf).exists().then_some(()).ok_or_else(|| io::Error::other(format!("THROTTLE_FILE {} missing: refusing to train without the GPU governor", c.thf)))?;
    let (ckd, rec) = (format!("{}/ckpt", c.out), BinFileRecorder::<FullPrecisionSettings>::new());
    fs::create_dir_all(&ckd)?;
    let mut core = Core::<B>::new(c, &dev);
    let mut mem = c.mem.then(|| Mem::<B>::new(c, &dev));
    let mut oc = AdamWConfig::new().with_weight_decay(c.wd).with_beta_2(0.95).with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))).init::<B, Core<B>>();
    let mut om = AdamWConfig::new().with_weight_decay(0.0).with_beta_2(0.95).init::<B, Mem<B>>();
    let st = format!("{ckd}/state.txt");
    let start = if bench == 0 && Path::new(&st).exists() { fs::read_to_string(&st)?.trim().parse::<usize>().unwrap_or(0) } else { 0 };
    let le = |e: burn::record::RecorderError| io::Error::other(format!("{e:?}"));
    if start > 0 {
        core = core.load_file(format!("{ckd}/core"), &rec, &dev).map_err(le)?;
        oc = oc.load_record(rec.load(format!("{ckd}/oc").into(), &dev).map_err(le)?);
        if let Some(m) = mem.take() { mem = Some(m.load_file(format!("{ckd}/mem"), &rec, &dev).map_err(le)?); om = om.load_record(rec.load(format!("{ckd}/om").into(), &dev).map_err(le)?); }
        println!("resumed {} at step {start}", c.out);
    }
    let mut log = fs::OpenOptions::new().create(true).append(true).open(format!("{}/log.jsonl", c.out))?;
    let steps = if bench > 0 { bench + 5 } else { c.steps };
    println!("=== adam-t [{}] {} | {}", std::any::type_name::<B>().split("::").last().unwrap_or(""), c.tag, c.json());
    writeln!(log, "{{\"cfg\": {}, \"start\": {start}}}", c.json())?;
    let (t0, mut th, mut tm, mut du, mut lt, mut last, mut acc) = (Instant::now(), 0.0, 0.0f64, 1.0f64, 0.0f64, String::new(), (0.0f64, 0.0f64, 0usize));
    for it in start..steps {
        th = thr(&c.thf);
        let ts = Instant::now();
        let (x, y) = tb(&tr, c.seed, it, c.bs, c.t);
        let loss = ce(logits(c, &core, mem.as_ref(), &x, c.bs, &dev), &y, &dev);
        let lv = (it % 50 == 49 || it + 1 == steps || it == 4).then(|| f1(loss.clone())[0] as f64);
        let mut gr = loss.backward();
        let (gc, lr) = (GradientsParams::from_module(&mut gr, &core), c.lr * c.mult(it));
        core = oc.step(lr, core, gc);
        mem = mem.map(|m| { let gm = GradientsParams::from_module(&mut gr, &m); om.step(c.lrm * c.mult(it), m, gm) });
        B::sync(&dev).map_err(|e| io::Error::other(format!("{e:?}")))?;
        let ms = ts.elapsed().as_secs_f64() * 1e3;
        let sl = nap(th, ms, c.unit);
        acc = if it < start + 3 { acc } else { (acc.0 + ms, acc.1 + sl, acc.2 + 1) };
        if let Some(l) = lv { lt = l; (tm, du) = if acc.2 > 0 { (acc.0 / acc.2 as f64, acc.0 / (acc.0 + acc.1)) } else { (tm, du) }; acc = (0.0, 0.0, 0); }
        let (done, wall) = (it + 1 == steps, bench == 0 && c.wall > 0.0 && t0.elapsed().as_secs_f64() > c.wall * 60.0);
        if bench == 0 && ((it + 1) % c.ev == 0 || done) {
            let va = eval(c, &core.valid(), mem.as_ref().map(|m| m.valid()).as_ref(), &ho, c.evb, &dev);
            last = format!("{{\"step\": {}, \"bytes\": {}, \"train_nats\": {:.4}, \"val_nats\": {:.4}, \"val_bits_per_byte\": {:.4}, \"gpu_ms_per_step\": {:.1}, \"gpu_bytes_per_s\": {:.0}, \"self_duty\": {:.3}, \"lr\": {:.2e}, \"throttle_s\": {th}, \"elapsed_s\": {:.1}}}", it + 1, (it + 1) * c.bs * c.t, lt, va, va / std::f64::consts::LN_2, tm, (c.bs * c.t) as f64 * 1e3 / tm.max(1e-9), du, lr, t0.elapsed().as_secs_f64());
            println!("{last}");
            writeln!(log, "{last}")?;
        }
        if bench == 0 && c.ck > 0 && ((it + 1) % c.ck == 0 || done || wall) {
            core.clone().save_file(format!("{ckd}/core"), &rec).map_err(le)?;
            rec.record(oc.to_record(), format!("{ckd}/oc").into()).map_err(le)?;
            if let Some(m) = mem.as_ref() { m.clone().save_file(format!("{ckd}/mem"), &rec).map_err(le)?; rec.record(om.to_record(), format!("{ckd}/om").into()).map_err(le)?; }
            fs::write(&st, format!("{}\n", it + 1))?;
        }
        if wall && !done { let r = format!("{{\"paused_at\": {}, \"wall_min\": {}, \"elapsed_s\": {:.1}}}", it + 1, c.wall, t0.elapsed().as_secs_f64()); println!("{r}"); writeln!(log, "{r}")?; return Ok(r); }
    }
    if bench > 0 {
        let r = format!("{{\"bench\": \"{}\", \"tern\": {}, \"mem\": {}, \"steps\": {bench}, \"gpu_ms_per_step\": {:.1}, \"gpu_bytes_per_s\": {:.0}, \"self_duty\": {du:.3}, \"throttle_s\": {th}, \"loss\": {:.4}, \"core_params\": {}}}", std::any::type_name::<B>(), c.tern, c.mem, tm, (c.bs * c.t) as f64 * 1e3 / tm.max(1e-9), lt, c.core_params());
        println!("{r}");
        return Ok(r);
    }
    if c.tern { lut::finish(c, &core.valid(), mem.as_ref().map(|m| m.valid()).as_ref(), &ho, &dev, &mut log)?; }
    Ok(last)
}
