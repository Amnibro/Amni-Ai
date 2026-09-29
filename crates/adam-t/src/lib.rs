pub mod bpe;
pub mod fa;
pub mod lut;
pub mod recall;
pub mod sft;
#[cfg(test)]
mod bind_tests;
use adam_ray::rng::Pcg64;
use fa::Fa;
use burn::{module::{AutodiffModule, Module, Param}, nn::loss::CrossEntropyLossConfig, optim::{grad_clipping::GradientClippingConfig, AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer}, record::{BinFileRecorder, FullPrecisionSettings, Recorder}, tensor::{activation::{log_sigmoid, log_softmax, relu, softmax}, backend::{AutodiffBackend, Backend}, module::{attention, embedding}, ops::AttentionModuleOptions, Distribution, IndexingUpdateOp, Int, Tensor, TensorData}};
use memmap2::Mmap;
use std::{env::var, fs, io::{self, Write}, path::Path, time::Instant};
#[derive(Clone, Debug)]
pub struct Cfg { pub d: usize, pub l: usize, pub h: usize, pub t: usize, pub ff: usize, pub ns: Vec<usize>, pub rb: u32, pub tern: bool, pub mem: bool, pub bs: usize, pub steps: usize, pub lr: f64, pub lrm: f64, pub wd: f32, pub warm: usize, pub seed: u64, pub ev: usize, pub ck: usize, pub evb: usize, pub tag: String, pub home: String, pub out: String, pub thf: String, pub corpus: String, pub par: usize, pub unit: f64, pub wall: f64, pub v: usize, pub bpe: String, pub micro: usize, pub base: String, pub sw: String, pub ep: f64, pub lpe: bool, pub lphi: f64, pub lsq: u32, pub init: String, pub lazy: bool, pub copy: bool, pub rmix: f64 }
impl Cfg {
    pub fn env() -> Cfg {
        let e = |k: &str, d: &str| var(k).unwrap_or_else(|_| d.into());
        let p = |k: &str, d: &str| e(k, d).trim().parse::<f64>().unwrap_or_else(|_| panic!("bad {k}"));
        let (home, d, tag, bpe, base) = (e("AMNI_HOME", "/home/anmire/ai/Amni-Ai"), p("D", "256") as usize, e("TAG", "smoke"), e("BPE", ""), e("BASE", ""));
        let sd = |a: &'static str, b: &'static str| if base.is_empty() { b } else { a };
        Cfg { d, l: p("L", "8") as usize, h: p("H", "4") as usize, t: p("CTX", "256") as usize, ff: p("FF", &(4 * d).to_string()) as usize, ns: e("NGRAMS", "2,3,4,5,6,8").split(',').filter(|s| !s.is_empty()).map(|s| s.trim().parse().unwrap()).collect(), rb: p("MEM_LOG2", "15") as u32, tern: e("TERN", "1") == "1", mem: e("MEM", "0") == "1", bs: p("BS", "64") as usize, steps: p("STEPS", "2000") as usize, lr: p("LR", sd("3e-4", "2e-3")), lrm: p("LR_MEM", &e("LR", sd("3e-4", "2e-3"))), wd: p("WD", "0.05") as f32, warm: p("WARM", sd("100", "200")) as usize, seed: p("SEED", "0") as u64, ev: p("EVAL_EVERY", sd("250", "500")) as usize, ck: p("CKPT_EVERY", "500") as usize, evb: p("EVAL_BATCHES", "20") as usize, out: e("OUT", &format!("{home}/exports/adam_t/{tag}")), tag, thf: e("THROTTLE_FILE", &format!("{home}/throttle.txt")), corpus: e("CORPUS", sd("adam_t/sft", "clean2")), par: p("PARITY_WINDOWS", "8") as usize, unit: p("THROTTLE_UNIT_MS", "50"), wall: p("MAX_MIN", "19"), v: if bpe.is_empty() { 256 } else { serde_json::from_str::<serde_json::Value>(&fs::read_to_string(&bpe).unwrap_or_default()).ok().and_then(|j| j["vocab_size"].as_u64()).unwrap_or_else(|| panic!("bad BPE {bpe}")) as usize }, bpe, micro: p("MICRO", "1") as usize, sw: e("SRC_W", ""), ep: p("EPOCHS", "0"), lpe: e("LPE", "0") == "1", lphi: p("LPE_PHI", "4"), lsq: p("LPE_S", "64") as u32, init: e("INIT", ""), lazy: match e("MEM_OPT", "dense").as_str() { "lazy" => true, "dense" => false, o => panic!("bad MEM_OPT {o}: dense or lazy") }, copy: e("COPY", "0") == "1", rmix: Some(p("RECALL_MIX", "0")).filter(|r| (0.0..1.0).contains(r)).unwrap_or_else(|| panic!("bad RECALL_MIX: want 0 <= r < 1")), base, home }
    }
    pub fn from_dir(dir: &str) -> io::Result<Cfg> {
        let l = fs::read_to_string(format!("{dir}/log.jsonl"))?.lines().next().unwrap_or("").to_string();
        let j: serde_json::Value = serde_json::from_str(&l).map_err(io::Error::other)?;
        let (g, u) = (&j["cfg"], |k: &str| j["cfg"][k].as_u64().ok_or_else(|| io::Error::other(format!("cfg.{k} missing in {dir}/log.jsonl"))).map(|v| v as usize));
        Ok(Cfg { d: u("d")?, l: u("l")?, h: u("h")?, t: u("ctx")?, ff: u("ff")?, rb: u("mem_log2")? as u32, tern: g["tern"].as_bool().unwrap_or(true), mem: g["mem"].as_bool().unwrap_or(false), ns: g["ngrams"].as_array().map(|a| a.iter().filter_map(|v| v.as_u64().map(|v| v as usize)).collect()).unwrap_or_default(), out: dir.into(), v: g["v"].as_u64().unwrap_or(256) as usize, bpe: g["bpe"].as_str().unwrap_or("").into(), corpus: g["corpus"].as_str().unwrap_or("clean2").into(), base: g["base"].as_str().unwrap_or("").into(), sw: g["src_w"].as_str().unwrap_or("").into(), ep: g["epochs"].as_f64().unwrap_or(0.0), lpe: g["lpe"].as_bool().unwrap_or(false), lphi: g["lpe_phi"].as_f64().unwrap_or(4.0), lsq: g["lpe_s"].as_u64().unwrap_or(64) as u32, init: g["init"].as_str().unwrap_or("").into(), lazy: g["mem_opt"].as_str() == Some("lazy"), copy: g["copy"].as_bool().unwrap_or(false), rmix: g["recall_mix"].as_f64().unwrap_or(0.0), ..Cfg::env() })
    }
    pub fn sft(&self) -> bool { !self.base.is_empty() }
    pub fn k(&self) -> usize { if self.mem { self.ns.len() } else { 0 } }
    pub fn rows(&self) -> usize { self.k() << self.rb }
    pub fn dc(&self) -> usize { self.d / self.h }
    pub fn mult(&self, it: usize) -> f64 { let pr = ((it as f64 - self.warm as f64) / (self.steps.saturating_sub(self.warm)).max(1) as f64).clamp(0.0, 1.0); if it < self.warm { (it as f64 + 1.0) / self.warm as f64 } else { 0.1 + 0.45 * (1.0 + (std::f64::consts::PI * pr).cos()) } }
    pub fn core_params(&self) -> usize { self.v * self.d + self.t * self.d * !self.lpe as usize + self.l * (4 * self.d * self.d + 2 * self.d * self.ff) + self.d * self.v + self.copy as usize * (self.d * (2 * self.dc() + 1) + 1) }
    pub fn lz(&self) -> Option<Lz> { self.lpe.then(|| Lz::new(self.d / self.h, self.t, self.lphi, self.lsq)) }
    pub fn bpe(&self) -> io::Result<Option<bpe::Bpe>> { (!self.bpe.is_empty()).then(|| bpe::Bpe::open(&self.bpe)).transpose() }
    pub fn data(&self, split: &str) -> io::Result<Dat> { let d = format!("{}/data/{}", self.home, self.corpus); Ok(if self.v == 256 { Dat::B(mmap(&format!("{d}/{split}.bin"))?) } else { Dat::W(mmap(&format!("{d}/{split}.bpe{}.u16", self.v))?) }) }
    pub fn json(&self) -> String { format!("{{\"d\": {}, \"l\": {}, \"h\": {}, \"ctx\": {}, \"ff\": {}, \"tern\": {}, \"mem\": {}, \"ngrams\": {:?}, \"mem_log2\": {}, \"bs\": {}, \"steps\": {}, \"lr\": {}, \"lr_mem\": {}, \"wd\": {}, \"warm\": {}, \"seed\": {}, \"corpus\": \"{}\", \"v\": {}, \"bpe\": \"{}\", \"micro\": {}, \"base\": \"{}\", \"src_w\": \"{}\", \"epochs\": {}, \"lpe\": {}, \"lpe_phi\": {}, \"lpe_s\": {}, \"init\": \"{}\", \"mem_opt\": \"{}\", \"copy\": {}, \"recall_mix\": {}, \"core_params\": {}, \"mem_params\": {}, \"symbols_per_step\": {}}}", self.d, self.l, self.h, self.t, self.ff, self.tern, self.mem, self.ns, self.rb, self.bs, self.steps, self.lr, self.lrm, self.wd, self.warm, self.seed, self.corpus, self.v, self.bpe, self.micro, self.base, self.sw, self.ep, self.lpe, self.lphi, self.lsq, self.init, if self.lazy { "lazy" } else { "dense" }, self.copy, self.rmix, self.core_params(), self.rows() * self.d, self.bs * self.t) }
}
#[derive(Module, Debug)]
pub struct Lin<B: Backend> { pub w: Param<Tensor<B, 2>> }
#[derive(Module, Debug)]
pub struct Blk<B: Backend> { pub qkv: Lin<B>, pub o: Lin<B>, pub f1: Lin<B>, pub f2: Lin<B> }
#[derive(Module, Debug)]
pub struct Core<B: Backend> { pub emb: Param<Tensor<B, 2>>, pub pos: Param<Tensor<B, 2>>, pub blk: Vec<Blk<B>>, pub head: Lin<B> }
#[derive(Module, Debug)]
pub struct Mem<B: Backend> { pub tab: Param<Tensor<B, 2>> }
#[derive(Module, Debug)]
pub struct Cp<B: Backend> { pub w: Lin<B>, pub gb: Param<Tensor<B, 2>> }
fn rnd<B: Backend>(s: [usize; 2], sd: f64, dev: &B::Device) -> Param<Tensor<B, 2>> { Param::from_tensor(Tensor::random(s, Distribution::Normal(0.0, sd), dev)) }
pub fn nrm<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> { x.clone() / x.square().mean_dim(1).add_scalar(1e-6).sqrt() }
pub fn tq<B: Backend>(w: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) { let g = w.clone().abs().mean().reshape([1, 1]).add_scalar(1e-6); let h = g.clone().mul_scalar(0.5); (((w.clone() - h.clone()).sign() + (w + h).sign()).mul_scalar(0.5).round(), g) }
impl<B: Backend> Lin<B> {
    pub fn new(i: usize, o: usize, sd: f64, dev: &B::Device) -> Self { Lin { w: rnd([i, o], sd, dev) } }
    pub fn fwd(&self, x: Tensor<B, 2>, q: bool) -> Tensor<B, 2> where B: Fa { fa::tl(x, self.w.val(), q) }
}
impl<B: Backend> Core<B> {
    pub fn new(c: &Cfg, dev: &B::Device) -> Self { let so = 0.02 / (2.0 * c.l as f64).sqrt(); Core { emb: rnd([c.v, c.d], 0.02, dev), pos: rnd([c.t, c.d], 0.02, dev), blk: (0..c.l).map(|_| Blk { qkv: Lin::new(c.d, 3 * c.d, 0.02, dev), o: Lin::new(c.d, c.d, so, dev), f1: Lin::new(c.d, c.ff, 0.02, dev), f2: Lin::new(c.ff, c.d, so, dev) }).collect(), head: Lin::new(c.d, c.v, 0.02, dev) } }
    pub fn lins(&self) -> Vec<&Lin<B>> { self.blk.iter().flat_map(|b| [&b.qkv, &b.o, &b.f1, &b.f2]).chain([&self.head]).collect() }
    pub fn fwd(&self, ids: Tensor<B, 2, Int>, mem: Option<Tensor<B, 3>>, h: usize, q: bool, lz: Option<Tensor<B, 2>>, cp: Option<&Cp<B>>) -> Tensor<B, 2> where B: Fa {
        let ([b, t], d) = (ids.dims(), self.emb.val().dims()[1]);
        let x = embedding(self.emb.val(), ids.clone());
        let x = if lz.is_some() { x } else { x + self.pos.val().narrow(0, 0, t).unsqueeze::<3>() };
        let x = mem.map_or(x.clone(), |m| x + m).reshape([b * t, d]);
        let x = self.blk.iter().fold(x, |x, k| {
            let a = boost(k.qkv.fwd(x.clone(), q), lz.as_ref(), b, t);
            let o = if *fa::FLASH { fa::fa(a, b, t, h) } else { let [aq, ak, av] = fa::split(a, b, t, h); attention(aq, ak, av, None, None, AttentionModuleOptions { scale: None, softcap: None, is_causal: true }).swap_dims(1, 2).reshape([b * t, d]) };
            let x = x + k.o.fwd(o, q);
            x.clone() + k.f2.fwd(relu(k.f1.fwd(x, q)).square(), q)
        });
        match cp { Some(p) => cmix(self.head.fwd(x.clone(), q), p.w.fwd(x, q), p.gb.val(), ids, b, t), None => self.head.fwd(x, q) }
    }
}
impl<B: Backend> Cp<B> {
    pub fn new(c: &Cfg, dev: &B::Device) -> Self { Cp { w: Lin::new(c.d, 2 * c.dc() + 1, 0.02, dev), gb: Param::from_tensor(Tensor::full([1, 1], 2.0, dev)) } }
}
pub fn cmix<B: Backend>(lg: Tensor<B, 2>, c: Tensor<B, 2>, gb: Tensor<B, 2>, ids: Tensor<B, 2, Int>, b: usize, t: usize) -> Tensor<B, 2> {
    let (v, dc, dev) = (lg.dims()[1], (c.dims()[1] - 1) / 2, lg.device());
    let f = |w: Vec<f32>, s: [usize; 2]| Tensor::<B, 2>::from_data(TensorData::new(w, s), &dev);
    let s = c.clone().narrow(1, 0, dc).reshape([b, t, dc]).matmul(c.clone().narrow(1, dc, dc).reshape([b, t, dc]).swap_dims(1, 2)).mul_scalar(1.0 / (dc as f64).sqrt()) + f((0..t * t).map(|i| if i % t >= i / t { -1.0e30 } else { 0.0 }).collect(), [t, t]).unsqueeze::<3>();
    let sx = if t > 1 { Tensor::cat(vec![ids.clone().narrow(1, 1, t - 1), ids.narrow(1, t - 1, 1)], 1) } else { ids };
    let pc = Tensor::<B, 2>::zeros([b * t, v], &dev).scatter(1, sx.unsqueeze_dim::<3>(1).expand([b, t, t]).reshape([b * t, t]), softmax(s, 2).reshape([b * t, t]), IndexingUpdateOp::Add);
    let z = c.narrow(1, 2 * dc, 1) + gb + f((0..b * t).map(|i| if i % t == 0 { 1.0e4 } else { 0.0 }).collect(), [b * t, 1]);
    let (x, y) = (log_sigmoid(z.clone()) + log_softmax(lg, 1), log_sigmoid(z.neg()) + pc.clamp_min(1e-30).log());
    let m = x.clone().max_pair(y.clone()).detach();
    m.clone() + ((x - m.clone()).exp() + (y - m).exp()).log()
}
impl<B: Backend> Mem<B> {
    pub fn new(c: &Cfg, dev: &B::Device) -> Self { Mem { tab: rnd([c.rows(), c.d], 0.02, dev) } }
    pub fn fwd(&self, ix: Tensor<B, 2, Int>, k: usize) -> Tensor<B, 3> { mrow(self.tab.val(), ix, k) }
    pub fn grow(self, k: usize, a: u32, b: u32) -> Self { if a == b { return self; } let t = self.tab.into_value().set_require_grad(false); let d = t.dims()[1]; Mem { tab: Param::from_tensor(t.reshape([k << a, 1, d]).repeat_dim(1, 1 << (b - a)).reshape([k << b, d])) } }
}
pub fn mrow<B: Backend>(tab: Tensor<B, 2>, ix: Tensor<B, 2, Int>, k: usize) -> Tensor<B, 3> { let ([b, tk], d) = (ix.dims(), tab.dims()[1]); embedding(tab, ix).reshape([b, tk / k, k, d]).sum_dim(2).reshape([b, tk / k, d]) }
pub struct Lazy<B: Backend> { pub m: Tensor<B, 2>, pub v: Tensor<B, 2> }
impl<B: Backend> Lazy<B> {
    pub fn new(r: usize, d: usize, dev: &B::Device) -> Self { Lazy { m: Tensor::zeros([r, d], dev), v: Tensor::zeros([r, d], dev) } }
    pub fn step(self, tab: Tensor<B, 2>, gs: Vec<(Vec<i32>, Tensor<B, 3>)>, k: usize, lr: f64, t: usize, dev: &B::Device) -> (Self, Tensor<B, 2>) {
        let (ix, d, b1, b2) = (gs.iter().flat_map(|g| g.0.iter().copied()).collect::<Vec<i32>>(), tab.dims()[1], 0.9f32, 0.95f32);
        let mut u = ix.clone();
        u.sort_unstable();
        u.dedup();
        let (ui, inv, n) = (it1::<B>(u.clone(), dev), it1::<B>(ix.iter().map(|r| u.binary_search(r).unwrap_or(0) as i32).collect(), dev), u.len());
        let g = Tensor::<B, 2>::zeros([n, d], dev).select_assign(0, inv, Tensor::cat(gs.into_iter().map(|(_, g)| { let [b, w, _] = g.dims(); g.reshape([b * w, 1, d]).repeat_dim(1, k).reshape([b * w * k, d]) }).collect(), 0), IndexingUpdateOp::Add);
        let m = self.m.clone().select(0, ui.clone()).mul_scalar(b1).add(g.clone().mul_scalar(1.0 - b1));
        let v = self.v.clone().select(0, ui.clone()).mul_scalar(b2).add(g.square().mul_scalar(1.0 - b2));
        let p = tab.clone().select(0, ui.clone()).sub(m.clone().div_scalar(1.0 - b1.powi(t as i32)).div(v.clone().div_scalar(1.0 - b2.powi(t as i32)).sqrt().add_scalar(1e-5)).mul_scalar(lr));
        let (u2, w) = (ui.reshape([n, 1]), |a: Tensor<B, 2>, u: &Tensor<B, 2, Int>, b: Tensor<B, 2>| a.scatter_nd(u.clone(), b, IndexingUpdateOp::Assign));
        (Lazy { m: w(self.m, &u2, m), v: w(self.v, &u2, v) }, w(tab, &u2, p))
    }
    pub fn save(&self, ckd: &str) -> io::Result<()> { [("lm", &self.m), ("lv", &self.v)].into_iter().try_for_each(|(n, t)| Mem { tab: Param::from_tensor(t.clone()) }.save_file(format!("{ckd}/{n}"), &BinFileRecorder::<FullPrecisionSettings>::new()).map_err(|e| io::Error::other(format!("{e:?}")))) }
    pub fn load(c: &Cfg, ckd: &str, dev: &B::Device) -> io::Result<Self> { let l = |n: &str| Mem::<B> { tab: Param::from_tensor(Tensor::zeros([c.rows(), c.d], dev)) }.load_file(format!("{ckd}/{n}"), &BinFileRecorder::<FullPrecisionSettings>::new(), dev).map(|m| m.tab.into_value()).map_err(|e| io::Error::other(format!("{e:?}"))); Ok(Lazy { m: l("lm")?, v: l("lv")? }) }
}
pub const G17: [u8; 16] = { let mut t = [1u8; 16]; let mut i = 1; while i < 16 { t[i] = (t[i - 1] as u32 * 3 % 17) as u8; i += 1; } t };
pub const L17: [u8; 17] = { let mut t = [0u8; 17]; let mut i = 0; while i < 16 { t[G17[i] as usize] = i as u8; i += 1; } t };
#[derive(Clone, Debug)]
pub struct Lz { pub dh: usize, pub t: usize, pub phi: f64, pub sq: u32, pub dl: f64, pub s: Vec<i64>, pub c: Vec<i64>, pub n: i64, pub tab: Vec<f32> }
impl Lz {
    pub fn new(dh: usize, t: usize, phi: f64, sq: u32) -> Lz {
        let (m, w) = (dh / 2, t.max(2) as i64 - 1);
        let s: Vec<i64> = (0..m).map(|j| ((sq.max(1) as f64).powf(1.0 - j as f64 / (m.max(2) - 1) as f64).round() as i64).max(1)).collect();
        let c: Vec<i64> = s.iter().map(|&v| v * w / 2).collect();
        let n = (s.iter().zip(&c).map(|(&v, &c)| (v * w - c).max(c)).max().unwrap_or(0) + 15) / 16 * 16;
        let dl = phi / (sq.max(1) as f64 * w as f64);
        Lz { dh, t, phi, sq, dl, tab: (-n..n + 16).map(|k| (dl * k as f64).exp() as f32).collect(), s, c, n }
    }
    pub fn ex(&self, p: usize, m: usize) -> i64 { self.s[m] * p as i64 - self.c[m] }
    pub fn f(&self, n: i64) -> f32 { self.tab[(n + self.n) as usize] }
    pub fn gz(&self, st: (i64, u8), k: i64) -> (i64, u8) { let kl = k.rem_euclid(16) as u8; (st.0 + k.div_euclid(16) + (L17[st.1 as usize] + kl >= 16) as i64, (st.1 as u32 * G17[kl as usize] as u32 % 17) as u8) }
    pub fn inv(&self, st: (i64, u8)) -> (i64, u8) { (-st.0 - (st.1 != 1) as i64, G17[(16 - L17[st.1 as usize] as usize) % 16]) }
    pub fn gf(&self, st: (i64, u8)) -> f32 { self.tab[(st.0 * 16 + L17[st.1 as usize] as i64 + self.n) as usize] }
    pub fn st(&self, p: usize, m: usize) -> (i64, u8) { self.gz((0, 1), self.ex(p, m)) }
    pub fn uv(&self, p: usize, m: usize, gf: bool) -> (f32, f32) { if gf { let s = self.st(p, m); (self.gf(s), self.gf(self.inv(s))) } else { let n = self.ex(p, m); (self.f(n), self.f(-n)) } }
    pub fn app(&self, q: &mut [f32], k: &mut [f32], p: usize, gf: bool) { let ft: Vec<(f32, f32)> = (0..self.dh / 2).map(|m| self.uv(p, m, gf)).collect(); q.chunks_exact_mut(2).zip(k.chunks_exact_mut(2)).enumerate().for_each(|(i, (a, b))| { let (u, v) = ft[i % ft.len()]; (a[0], a[1], b[0], b[1]) = (a[0] * u, a[1] * v, b[0] * v, b[1] * u); }); }
    pub fn row(&self, d: usize, p: usize) -> Vec<f32> { let (mut q, mut k) = (vec![1f32; d], vec![1f32; d]); self.app(&mut q, &mut k, p, false); [q, k, vec![1f32; d]].concat() }
    pub fn tensor<B: Backend>(&self, d: usize, dev: &B::Device) -> Tensor<B, 2> { Tensor::from_data(TensorData::new((0..self.t).flat_map(|p| self.row(d, p)).collect::<Vec<f32>>(), [self.t, 3 * d]), dev) }
}
pub fn boost<B: Backend>(a: Tensor<B, 2>, lz: Option<&Tensor<B, 2>>, b: usize, t: usize) -> Tensor<B, 2> { match lz { Some(z) => { let e = a.dims()[1]; (a.reshape([b, t, e]) * z.clone().narrow(0, 0, t).unsqueeze::<3>()).reshape([b * t, e]) } None => a } }
pub fn mmap(p: &str) -> io::Result<Mmap> { unsafe { Mmap::map(&fs::File::open(p)?) } }
pub fn hsh(b: &[u8], n: usize) -> u64 { let mut h = (n as u64) << 56 ^ (b.len() as u64) << 48 ^ 0x9E37_79B9_7F4A_7C15; b.iter().for_each(|&c| h = (h ^ c as u64).wrapping_mul(0x0000_0100_0000_01B3).rotate_left(23)); h ^= h >> 33; h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD); h ^= h >> 33; h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53); h ^ h >> 33 }
pub fn hrow(x: &[u32], p: usize, k: usize, n: usize, rb: u32, w: bool) -> usize { let s = &x[p + 1 - n.min(p + 1)..=p]; k << rb | (hsh(&if w { s.iter().flat_map(|&v| (v as u16).to_le_bytes()).collect::<Vec<u8>>() } else { s.iter().map(|&v| v as u8).collect() }, n) >> (64 - rb)) as usize }
pub fn hix(x: &[u32], b: usize, t: usize, ns: &[usize], rb: u32, w: bool) -> Vec<i32> { (0..b * t).flat_map(|i| { let s = &x[i / t * t..i / t * t + t]; ns.iter().enumerate().map(move |(k, &n)| hrow(s, i % t, k, n, rb, w) as i32) }).collect() }
pub enum Dat { B(Mmap), W(Mmap) }
impl Dat {
    pub fn len(&self) -> usize { match self { Dat::B(m) => m.len(), Dat::W(m) => m.len() / 2 } }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn get(&self, i: usize, n: usize) -> Vec<u32> { match self { Dat::B(m) => m[i..i + n].iter().map(|&v| v as u32).collect(), Dat::W(m) => m[2 * i..2 * (i + n)].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect() } }
    pub fn win(&self, ix: &[usize], t: usize) -> (Vec<u32>, Vec<u32>) { xy(ix.iter().map(|&i| self.get(i, t + 1)).collect(), t) }
    pub fn hold(&self, g: &mut Pcg64, bs: usize, t: usize) -> (Vec<u32>, Vec<u32>) { self.win(&adam_train::bix(self.len(), g, bs, t, None).iter().map(|&i| i as usize).collect::<Vec<_>>(), t) }
}
pub fn sm(mut z: u64) -> u64 { z = z.wrapping_add(0x9E37_79B9_7F4A_7C15); z = (z ^ z >> 30).wrapping_mul(0xBF58_476D_1CE4_E5B9); z = (z ^ z >> 27).wrapping_mul(0x94D0_49BB_1331_11EB); z ^ z >> 31 }
pub fn xy(w: Vec<Vec<u32>>, t: usize) -> (Vec<u32>, Vec<u32>) { (w.iter().flat_map(|v| v[..t].to_vec()).collect(), w.iter().flat_map(|v| v[1..].to_vec()).collect()) }
pub fn tb(d: &Dat, seed: u64, it: usize, bs: usize, t: usize, rg: Option<&recall::Gen>) -> (Vec<u32>, Vec<u32>) { xy((0..bs).map(|j| match rg.filter(|g| g.hit(seed, it, j)) { Some(g) => g.window(seed, it, j, t + 1), None => d.get((sm(seed << 40 ^ (it as u64) << 12 ^ j as u64) % (d.len() - t - 1) as u64) as usize, t + 1) }).collect(), t) }
pub fn it2<B: Backend>(v: Vec<i32>, s: [usize; 2], dev: &B::Device) -> Tensor<B, 2, Int> { Tensor::from_data(TensorData::new(v, s), dev) }
pub fn it1<B: Backend>(v: Vec<i32>, dev: &B::Device) -> Tensor<B, 1, Int> { let n = v.len(); Tensor::from_data(TensorData::new(v, [n]), dev) }
pub fn logits<B: Fa>(c: &Cfg, core: &Core<B>, mem: Option<&Mem<B>>, cp: Option<&Cp<B>>, x: &[u32], b: usize, dev: &B::Device) -> Tensor<B, 2> { let n = x.len() / b; lgm(c, core, mem.map(|m| m.fwd(it2(hix(x, b, n, &c.ns, c.rb, c.v > 256), [b, n * c.k()], dev), c.k())), cp, x, b, dev) }
pub fn lgm<B: Fa>(c: &Cfg, core: &Core<B>, m: Option<Tensor<B, 3>>, cp: Option<&Cp<B>>, x: &[u32], b: usize, dev: &B::Device) -> Tensor<B, 2> { core.fwd(it2(x.iter().map(|&v| v as i32).collect(), [b, x.len() / b], dev), m, c.h, c.tern, c.lz().map(|z| z.tensor::<B>(c.d, dev)), cp) }
pub fn mleaf<B: AutodiffBackend>(c: &Cfg, m: &Mem<B>, x: &[u32], b: usize, dev: &B::Device) -> (Vec<i32>, Tensor<B, 3>) { let (n, ix) = (x.len() / b, hix(x, b, x.len() / b, &c.ns, c.rb, c.v > 256)); (ix.clone(), Tensor::from_inner(mrow(m.tab.val().inner(), it2(ix, [b, n * c.k()], dev), c.k())).require_grad()) }
pub fn infer<B: Fa>(c: &Cfg, core: &Core<B>, mem: Option<&Mem<B>>, cp: Option<&Cp<B>>, x: &[u32], b: usize, dev: &B::Device) -> Vec<f32> { f1(logits(c, core, mem, cp, x, b, dev)) }
pub fn load<B: Backend>(c: &Cfg, dev: &B::Device) -> io::Result<(Core<B>, Option<Mem<B>>, Option<Cp<B>>)> { let (rec, le) = (BinFileRecorder::<FullPrecisionSettings>::new(), |e: burn::record::RecorderError| io::Error::other(format!("{e:?}"))); Ok((Core::<B>::new(c, dev).load_file(format!("{}/ckpt/core", c.out), &rec, dev).map_err(le)?, c.mem.then(|| Mem::<B>::new(c, dev).load_file(format!("{}/ckpt/mem", c.out), &rec, dev).map_err(le)).transpose()?, c.copy.then(|| Cp::<B>::new(c, dev).load_file(format!("{}/ckpt/copy", c.out), &rec, dev).map_err(le)).transpose()?)) }
pub fn ce<B: Backend>(lg: Tensor<B, 2>, y: &[u32], dev: &B::Device) -> Tensor<B, 1> { CrossEntropyLossConfig::new().init(dev).forward(lg, Tensor::from_data(TensorData::new(y.iter().map(|&v| v as i32).collect::<Vec<i32>>(), [y.len()]), dev)) }
pub fn f1<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> { t.into_data().iter::<f32>().collect() }
pub fn eval<B: Fa>(c: &Cfg, core: &Core<B>, mem: Option<&Mem<B>>, cp: Option<&Cp<B>>, hold: &Dat, lens: &[u32], nb: usize, dev: &B::Device) -> f64 {
    let e = 1usize << (c.bs / c.micro.max(1)).clamp(1, 32).ilog2();
    let (mut g, th, q) = (Pcg64::new(Some(1234)), thr(&c.thf), e * c.t);
    let v: Vec<(f64, f64)> = (0..nb).flat_map(|_| { let (x, y) = hold.hold(&mut g, 128, c.t); (0..128 / e).map(|k| { let t0 = Instant::now(); let r = f1(ce(logits(c, core, mem, cp, &x[k * q..(k + 1) * q], e, dev), &y[k * q..(k + 1) * q], dev))[0] as f64 * q as f64; nap(th, t0.elapsed().as_secs_f64() * 1e3, c.unit); (r, y[k * q..(k + 1) * q].iter().map(|&t| lens[t as usize] as f64).sum::<f64>()) }).collect::<Vec<_>>() }).collect();
    adam_ray::rng::pairwise(&v.iter().map(|x| x.0).collect::<Vec<_>>()) / adam_ray::rng::pairwise(&v.iter().map(|x| x.1).collect::<Vec<_>>())
}
pub fn thr(p: &str) -> f64 { fs::read_to_string(p).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0.0) }
pub fn nap(th: f64, ms: f64, unit: f64) -> f64 { let s = if th > 0.0 { th * 1e3 * (ms / unit).max(1.0) } else { 0.0 }; (s > 0.0).then(|| std::thread::sleep(std::time::Duration::from_secs_f64(s / 1e3))); s }
pub fn train<B: AutodiffBackend + Fa>(c: &Cfg, dev: B::Device, bench: usize) -> io::Result<String> where B::InnerBackend: Fa {
    (c.base.is_empty() || c.init.is_empty()).then_some(()).ok_or_else(|| io::Error::other("BASE (sft) and INIT (continued pretraining) are exclusive"))?;
    let bc = [&c.base, &c.init].into_iter().find(|s| !s.is_empty()).map(|d| Cfg::from_dir(d)).transpose()?;
    bc.as_ref().is_none_or(|b| c.sft() || c.rb >= b.rb).then_some(()).ok_or_else(|| io::Error::other(format!("MEM_LOG2 {} is below the INIT checkpoint's {}: only growing is exact", c.rb, bc.as_ref().map_or(0, |b| b.rb))))?;
    let c = &bc.as_ref().map_or_else(|| c.clone(), |b| Cfg { d: b.d, l: b.l, h: b.h, t: b.t, ff: b.ff, ns: b.ns.clone(), rb: if c.sft() { b.rb } else { c.rb }, tern: b.tern, mem: b.mem, v: b.v, bpe: b.bpe.clone(), lpe: b.lpe, lphi: b.lphi, lsq: b.lsq, lazy: if c.sft() && var("MEM_OPT").is_err() { b.lazy } else { c.lazy }, copy: c.copy || (var("COPY").is_err() && b.copy), ..c.clone() });
    let sj = c.sft().then(|| sft::Job::new(c)).transpose()?;
    let c = &sj.as_ref().map_or_else(|| c.clone(), |j| Cfg { steps: j.steps, ..c.clone() });
    let (tr, ho, lens, rg) = (c.data("train")?, c.data("hold")?, c.bpe()?.map(|b| b.lens()).unwrap_or_else(|| vec![1; 256]), (c.rmix > 0.0 && !c.sft()).then(|| recall::Gen::new(c)).transpose()?);
    let bpt = { let s = ho.get(0, ho.len().min(1 << 20)); s.iter().map(|&t| lens[t as usize] as f64).sum::<f64>() / s.len() as f64 };
    Path::new(&c.thf).exists().then_some(()).ok_or_else(|| io::Error::other(format!("THROTTLE_FILE {} missing: refusing to train without the GPU governor", c.thf)))?;
    let (ckd, rec) = (format!("{}/ckpt", c.out), BinFileRecorder::<FullPrecisionSettings>::new());
    fs::create_dir_all(&ckd)?;
    B::seed(&dev, c.seed);
    let mut core = Core::<B>::new(c, &dev);
    let mut mem = c.mem.then(|| Mem::<B>::new(c, &dev));
    let mut cp = c.copy.then(|| Cp::<B>::new(c, &dev));
    let aw = || AdamWConfig::new().with_weight_decay(c.wd).with_beta_2(0.95).with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));
    let (mut oc, mut ocp) = (aw().init::<B, Core<B>>(), aw().init::<B, Cp<B>>());
    let mut om = AdamWConfig::new().with_weight_decay(0.0).with_beta_2(0.95).init::<B, Mem<B>>();
    let st = format!("{ckd}/state.txt");
    let start = if bench == 0 && Path::new(&st).exists() { fs::read_to_string(&st)?.trim().parse::<usize>().unwrap_or(0) } else { 0 };
    let le = |e: burn::record::RecorderError| io::Error::other(format!("{e:?}"));
    if start > 0 {
        core = core.load_file(format!("{ckd}/core"), &rec, &dev).map_err(le)?;
        oc = oc.load_record(rec.load(format!("{ckd}/oc").into(), &dev).map_err(le)?);
        if let Some(m) = mem.take() { mem = Some(m.load_file(format!("{ckd}/mem"), &rec, &dev).map_err(le)?); if !c.lazy { om = om.load_record(rec.load(format!("{ckd}/om").into(), &dev).map_err(le)?); } }
        if let Some(p) = cp.take() { cp = Some(p.load_file(format!("{ckd}/copy"), &rec, &dev).map_err(le)?); ocp = ocp.load_record(rec.load(format!("{ckd}/ocp").into(), &dev).map_err(le)?); }
        println!("resumed {} at step {start}", c.out);
    }
    if let (0, Some(b)) = (start, bc.as_ref()) { let (cr, m, pb) = load::<B>(b, &dev)?; (core, mem, cp) = (cr, m.map(|m| m.grow(c.k(), b.rb, c.rb)), c.copy.then(|| pb.unwrap_or_else(|| Cp::new(c, &dev)))); println!("{} init from {} step {}{}", if c.sft() { "sft" } else { "continued" }, b.out, fs::read_to_string(format!("{}/ckpt/state.txt", b.out)).unwrap_or_default().trim(), if b.rb < c.rb { format!(", PTEX memory grown 2^{} -> 2^{} rows per order", b.rb, c.rb) } else { String::new() }); }
    let mut lo = (c.lazy && c.mem).then(|| if start > 0 { Lazy::<B::InnerBackend>::load(c, &ckd, &dev) } else { Ok(Lazy::new(c.rows(), c.d, &dev)) }).transpose()?;
    if bench == 0 { fs::write(format!("{ckd}/steps.txt"), format!("{}\n", c.steps))?; }
    let mut log = fs::OpenOptions::new().create(true).append(true).open(format!("{}/log.jsonl", c.out))?;
    let steps = if bench > 0 { bench + 5 } else { c.steps };
    println!("=== adam-t [{}] {} | {}", std::any::type_name::<B>().split("::").last().unwrap_or(""), c.tag, c.json());
    writeln!(log, "{{\"cfg\": {}, \"start\": {start}}}", c.json())?;
    if let Some(j) = sj.as_ref() { let r = j.info(); println!("{r}"); writeln!(log, "{r}")?; }
    let (t0, mut th, mut tm, mut du, mut lt, mut last, mut acc) = (Instant::now(), 0.0, 0.0f64, 1.0f64, 0.0f64, String::new(), (0.0f64, 0.0f64, 0usize));
    for it in start..steps {
        th = thr(&c.thf);
        let (x, y, wm) = sj.as_ref().map_or_else(|| { let (x, y) = tb(&tr, c.seed, it, c.bs, c.t, rg.as_ref()); (x, y, None) }, |j| { let (x, y, m, _) = j.batch(it, c); (x, y, Some(m)) });
        let (mb, want) = (c.micro.max(1), it % 50 == 49 || it + 1 == steps || it == 4);
        let den = wm.as_ref().map_or(mb as f64, |m| m.iter().map(|&v| v as f64).sum::<f64>().max(1.0));
        let (q, se) = (c.bs / mb * c.t, |e: burn::tensor::backend::ExecutionError| io::Error::other(format!("{e:?}")));
        let (mut ac, mut am, mut ap, mut ls, mut ms, mut sl, mut gs) = (GradientsAccumulator::<Core<B>>::new(), GradientsAccumulator::<Mem<B>>::new(), GradientsAccumulator::<Cp<B>>::new(), 0f64, 0f64, 0f64, Vec::new());
        for k in 0..mb {
            let tk = Instant::now();
            let (xk, yk) = (&x[k * q..(k + 1) * q], &y[k * q..(k + 1) * q]);
            let ml = mem.as_ref().filter(|_| lo.is_some()).map(|m| mleaf(c, m, xk, c.bs / mb, &dev));
            let lg = match ml.as_ref() { Some((_, t)) => lgm(c, &core, Some(t.clone()), cp.as_ref(), xk, c.bs / mb, &dev), None => logits(c, &core, mem.as_ref(), cp.as_ref(), xk, c.bs / mb, &dev) };
            let loss = match wm.as_ref() { Some(m) => sft::mce(lg, yk, &m[k * q..(k + 1) * q], den, &dev), None => ce(lg, yk, &dev).div_scalar(den) };
            ls += if want { f1(loss.clone())[0] as f64 } else { 0.0 };
            let mut gr = loss.backward();
            if let Some((ix, t)) = ml { gs.push((ix, t.grad(&gr).ok_or_else(|| io::Error::other("no PTEX memory gradient"))?)); }
            ac.accumulate(&core, GradientsParams::from_module(&mut gr, &core));
            if let Some(p) = cp.as_ref() { ap.accumulate(p, GradientsParams::from_module(&mut gr, p)); }
            if let Some(m) = mem.as_ref().filter(|_| lo.is_none()) { am.accumulate(m, GradientsParams::from_module(&mut gr, m)); }
            B::sync(&dev).map_err(se)?;
            let t = tk.elapsed().as_secs_f64() * 1e3;
            (ms, sl) = (ms + t, sl + nap(th, t, c.unit));
            th = thr(&c.thf);
        }
        let (to, lr) = (Instant::now(), c.lr * c.mult(it));
        core = oc.step(lr, core, ac.grads());
        cp = cp.map(|p| ocp.step(lr, p, ap.grads()));
        mem = match (mem, lo.take()) { (Some(m), Some(o)) => { let (o, t) = o.step(m.tab.into_value().inner(), gs, c.k(), c.lrm * c.mult(it), it + 1, &dev); lo = Some(o); Some(Mem { tab: Param::from_tensor(Tensor::from_inner(t)) }) } (m, _) => m.map(|m| om.step(c.lrm * c.mult(it), m, am.grads())) };
        B::sync(&dev).map_err(se)?;
        let t = to.elapsed().as_secs_f64() * 1e3;
        (ms, sl) = (ms + t, sl + nap(th, t, c.unit));
        let lv = want.then_some(ls);
        acc = if it < start + 3 { acc } else { (acc.0 + ms, acc.1 + sl, acc.2 + 1) };
        if let Some(l) = lv { lt = l; (tm, du) = if acc.2 > 0 { (acc.0 / acc.2 as f64, acc.0 / (acc.0 + acc.1)) } else { (tm, du) }; acc = (0.0, 0.0, 0); }
        let (done, wall) = (it + 1 == steps, bench == 0 && (c.wall > 0.0 && t0.elapsed().as_secs_f64() > c.wall * 60.0 || Path::new(&format!("{}/STOP", c.out)).exists()));
        if bench == 0 && ((it + 1) % c.ev == 0 || done) {
            let (cv, mv, pv) = (core.valid(), mem.as_ref().map(|m| m.valid()), cp.as_ref().map(|p| p.valid()));
            let (va, ex) = sj.as_ref().map_or_else(|| (eval(c, &cv, mv.as_ref(), pv.as_ref(), &ho, &lens, c.evb, &dev), String::new()), |j| j.eval(c, &cv, mv.as_ref(), pv.as_ref(), &lens, &dev));
            last = format!("{{\"step\": {}, \"symbols\": {}, \"bytes\": {:.0}, \"train_nats\": {:.4}, \"val_nats\": {:.4}, \"val_bits_per_byte\": {:.4}, \"gpu_ms_per_step\": {:.1}, \"gpu_bytes_per_s\": {:.0}, \"self_duty\": {:.3}, \"lr\": {:.2e}, \"throttle_s\": {th}, \"elapsed_s\": {:.1}{ex}}}", it + 1, (it + 1) * c.bs * c.t, ((it + 1) * c.bs * c.t) as f64 * bpt, lt, va, va / std::f64::consts::LN_2, tm, (c.bs * c.t) as f64 * bpt * 1e3 / tm.max(1e-9), du, lr, t0.elapsed().as_secs_f64());
            println!("{last}");
            writeln!(log, "{last}")?;
        }
        if bench == 0 && c.ck > 0 && ((it + 1) % c.ck == 0 || done || wall) {
            core.clone().save_file(format!("{ckd}/core"), &rec).map_err(le)?;
            rec.record(oc.to_record(), format!("{ckd}/oc").into()).map_err(le)?;
            if let Some(m) = mem.as_ref() { m.clone().save_file(format!("{ckd}/mem"), &rec).map_err(le)?; match lo.as_ref() { Some(o) => o.save(&ckd)?, None => rec.record(om.to_record(), format!("{ckd}/om").into()).map_err(le)? } }
            if let Some(p) = cp.as_ref() { p.clone().save_file(format!("{ckd}/copy"), &rec).map_err(le)?; rec.record(ocp.to_record(), format!("{ckd}/ocp").into()).map_err(le)?; }
            fs::write(&st, format!("{}\n", it + 1))?;
        }
        if wall && !done { let r = format!("{{\"paused_at\": {}, \"wall_min\": {}, \"elapsed_s\": {:.1}}}", it + 1, c.wall, t0.elapsed().as_secs_f64()); println!("{r}"); writeln!(log, "{r}")?; return Ok(r); }
    }
    if bench > 0 {
        let r = format!("{{\"bench\": \"{}\", \"tern\": {}, \"mem\": {}, \"steps\": {bench}, \"gpu_ms_per_step\": {:.1}, \"gpu_bytes_per_s\": {:.0}, \"self_duty\": {du:.3}, \"throttle_s\": {th}, \"loss\": {:.4}, \"core_params\": {}}}", std::any::type_name::<B>(), c.tern, c.mem, tm, (c.bs * c.t) as f64 * bpt * 1e3 / tm.max(1e-9), lt, c.core_params());
        println!("{r}");
        return Ok(r);
    }
    if c.tern { lut::finish(c, &core.valid(), mem.as_ref().map(|m| m.valid()).as_ref(), cp.as_ref().map(|p| p.valid()).as_ref(), &ho, &lens, &dev, &mut log, &format!("{}/model.gf17", c.out))?; }
    Ok(last)
}
#[cfg(test)]
mod lpe_tests {
    use super::*;
    use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
    type Ad = Autodiff<NdArray>;
    const DEV: NdArrayDevice = NdArrayDevice::Cpu;
    fn u(i: u64) -> f64 { ((sm(i) >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 3.0 }
    fn tiny(s: &str, lpe: bool) -> Cfg { Cfg { d: 64, l: 1, h: 1, t: 64, ff: 128, ns: vec![2, 3], rb: 6, tern: true, mem: true, v: 256, bpe: String::new(), bs: 4, micro: 1, base: String::new(), sw: String::new(), ep: 0.0, seed: 3, par: 2, wall: 0.0, unit: 50.0, home: s.into(), corpus: "syn".into(), out: format!("{s}/run"), thf: format!("{s}/thr.txt"), steps: 60, warm: 5, lr: 3e-3, lrm: 3e-3, ev: 20, ck: 60, evb: 1, lpe, lphi: 4.0, lsq: 64, copy: false, rmix: 0.0, ..Cfg::env() } }
    #[test]
    fn gf17_boost_table_exact_and_additive() {
        let mut seen = G17.to_vec();
        seen.sort_unstable();
        assert!(seen == (1..17).collect::<Vec<u8>>() && (0..16).all(|i| L17[G17[i] as usize] as usize == i), "3 must be a primitive root of GF(17)");
        (0..16).for_each(|a| (0..16).for_each(|b| assert_eq!(G17[a] as u32 * G17[b] as u32 % 17, G17[(a + b) % 16] as u32)));
        let bo = |(u, v): (u32, u32), k: i64| (u * G17[k.rem_euclid(16) as usize] as u32 % 17, v * G17[(-k).rem_euclid(16) as usize] as u32 % 17);
        (1..17u32).for_each(|u0| (1..17u32).for_each(|v0| { let ks: Vec<i64> = (0..40).map(|i| (sm(u0 as u64 * 131 + v0 as u64 * 7 + i) % 97) as i64 - 48).collect(); let ch = ks.iter().fold((u0, v0), |a, &k| bo(a, k)); assert!(ch == bo((u0, v0), ks.iter().sum()) && ch.0 * ch.1 % 17 == u0 * v0 % 17, "a boost chain must collapse to one exponent sum and keep u*v"); }));
        let z = Lz::new(64, 256, 4.0, 64);
        assert!(z.s[0] == 64 && *z.s.last().unwrap() == 1 && z.s.windows(2).all(|w| w[0] >= w[1]) && z.n % 16 == 0 && z.tab.len() as i64 == 2 * z.n + 16);
        (0..z.tab.len() as i64).map(|i| i - z.n).for_each(|n| { let s = z.gz((0, 1), n); assert!(z.gf(s).to_bits() == z.f(n).to_bits() && z.f(n).to_bits() == ((z.dl * n as f64).exp() as f32).to_bits() && s.0 * 16 + L17[s.1 as usize] as i64 == n); });
        (0..32).for_each(|m| { let mut s = z.st(0, m); (0..z.t).for_each(|p| { assert!(s == z.st(p, m) && z.inv(s) == z.gz((0, 1), -z.ex(p, m))); let (a, b) = z.uv(p, m, true); assert!(a.to_bits() == z.f(z.ex(p, m)).to_bits() && b.to_bits() == z.f(-z.ex(p, m)).to_bits() && (a, b) == z.uv(p, m, false)); s = z.gz(s, z.s[m]); }); });
        (0..500u64).for_each(|i| { let ks: Vec<i64> = (0..8).map(|j| (sm(i * 8 + j) % 2001) as i64 - 1000).collect(); let n: i64 = ks.iter().sum(); assert_eq!(ks.iter().fold((0i64, 1u8), |s, &k| z.gz(s, k)), z.gz((0, 1), n)); assert!(n.abs() > z.n || z.gf(z.gz((0, 1), n)).to_bits() == z.f(n).to_bits()); });
        let rel = |i: usize, j: usize, m: usize| z.f(z.ex(i, m)) as f64 * z.f(-z.ex(j, m)) as f64;
        (0..32).for_each(|m| [(9usize, 3usize), (200, 194), (255, 0)].iter().for_each(|&(i, j)| assert!((rel(i, j, m) / (z.dl * (z.s[m] * (i - j) as i64) as f64).exp() - 1.0).abs() < 1e-6)));
        assert!((0..32).all(|m| (0..256).all(|p| (z.dl * z.ex(p, m).abs() as f64) <= 2.0 + 1e-9)), "centred rapidities stay within phi/2");
    }
    #[test]
    fn lpe_fwd_bwd_matches_minkowski_reference() {
        let (b, t, h, dh) = (2usize, 16usize, 2usize, 8usize);
        let (d, z) = (h * dh, Lz::new(dh, t, 3.0, 8));
        assert_eq!(z.s, vec![8, 4, 2, 1]);
        let (x, r): (Vec<f64>, Vec<f64>) = ((0..b * t * 3 * d).map(|i| u(i as u64)).collect(), (0..b * t * d).map(|i| u(i as u64 + (1 << 30))).collect());
        let (la, mk) = (|ph: f64, a: (f64, f64)| (ph.exp() * a.0, (-ph).exp() * a.1), |a: (f64, f64), b: (f64, f64)| a.0 * b.1 + a.1 * b.0);
        assert!((mk(la(0.7, (1.3, -0.4)), la(0.7, (0.2, 2.1))) - mk((1.3, -0.4), (0.2, 2.1))).abs() < 1e-12, "the light-cone form is boost invariant");
        let rf = |x: &[f64]| -> (Vec<f64>, f64) {
            let mut o = vec![0f64; b * t * d];
            (0..b).for_each(|bb| (0..h).for_each(|hh| (0..t).for_each(|i| {
                let qi = (bb * t + i) * 3 * d + hh * dh;
                let sc: Vec<f64> = (0..=i).map(|j| { let kj = (bb * t + j) * 3 * d + d + hh * dh; (0..dh / 2).map(|m| mk(la(z.dl * (z.s[m] * i as i64) as f64, (x[qi + 2 * m], x[qi + 2 * m + 1])), la(z.dl * (z.s[m] * j as i64) as f64, (x[kj + 2 * m + 1], x[kj + 2 * m])))).sum::<f64>() / (dh as f64).sqrt() }).collect();
                let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = sc.iter().map(|v| (v - mx).exp()).collect();
                let zs: f64 = e.iter().sum();
                (0..dh).for_each(|c| o[(bb * t + i) * d + hh * dh + c] = (0..=i).map(|j| e[j] * x[(bb * t + j) * 3 * d + 2 * d + hh * dh + c]).sum::<f64>() / zs);
            })));
            let l = o.iter().zip(&r).map(|(a, b)| a * b).sum();
            (o, l)
        };
        let (o_ref, _) = rf(&x);
        let fd: Vec<f64> = (0..x.len()).map(|i| { let (mut a, mut c) = (x.clone(), x.clone()); a[i] += 1e-6; c[i] -= 1e-6; (rf(&a).1 - rf(&c).1) / 2e-6 }).collect();
        let tf = |v: &[f64], s: [usize; 2]| Tensor::<Ad, 2>::from_data(TensorData::new(v.iter().map(|&a| a as f32).collect::<Vec<f32>>(), s), &DEV);
        let run = |lz: bool| { let xa = tf(&x, [b * t, 3 * d]).require_grad(); let zt = z.tensor::<Ad>(d, &DEV); let o = fa::fa(boost(xa.clone(), lz.then_some(&zt), b, t), b, t, h); let g = (o.clone() * tf(&r, [b * t, d])).sum().backward(); (f1(o), f1(xa.grad(&g).unwrap())) };
        let ((o, dx), (o0, _)) = (run(true), run(false));
        let md = |a: &[f32], b: &[f64]| a.iter().zip(b).map(|(&p, &q)| (p as f64 - q).abs()).fold(0f64, f64::max);
        let (eo, eg, gm) = (md(&o, &o_ref), md(&dx, &fd), fd.iter().fold(0f64, |m, v| m.max(v.abs())));
        println!("{{\"lpe_out_max_abs_diff\": {eo:.3e}, \"lpe_grad_max_abs_diff\": {eg:.3e}, \"grad_max_abs\": {gm:.3e}, \"no_lpe_out_diff\": {:.3e}}}", md(&o0, &o_ref));
        assert!(eo < 1e-5 && eg < 1e-4 * gm.max(1.0) && md(&o0, &o_ref) > 1e-2, "LPE forward/backward must match the Minkowski reference and differ from no LPE");
    }
    #[test]
    fn lpe_tiny_model_trains_and_lut_is_bit_exact() {
        let r = std::env::temp_dir().join(format!("adamt-lpe-test-{}", std::process::id()));
        let (s, _) = (r.to_string_lossy().to_string(), fs::remove_dir_all(&r));
        fs::create_dir_all(format!("{s}/data/syn")).unwrap();
        fs::write(format!("{s}/thr.txt"), b"0\n").unwrap();
        let txt: Vec<u8> = (0..40000u64).map(|i| b"the cat sat on a mat. a dog ran to the cat. "[(i % 44) as usize] ^ (sm(i) % 50 == 0) as u8).collect();
        fs::write(format!("{s}/data/syn/train.bin"), &txt).unwrap();
        fs::write(format!("{s}/data/syn/hold.bin"), &txt[..8000]).unwrap();
        let c = tiny(&s, true);
        assert!(!Cfg::env().lpe || var("LPE").is_ok(), "LPE is off unless LPE=1");
        assert_eq!(c.core_params() + c.t * c.d, tiny(&s, false).core_params());
        train::<Ad>(&c, DEV, 0).unwrap();
        let lg = fs::read_to_string(format!("{s}/run/log.jsonl")).unwrap();
        let js: Vec<serde_json::Value> = lg.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
        let v: Vec<f64> = js.iter().filter_map(|j| j["val_nats"].as_f64()).collect();
        let ex = js.iter().find(|j| j.get("export").is_some()).unwrap();
        println!("{v:?} {ex}");
        assert!(v.len() == 3 && v.iter().all(|x| x.is_finite()) && v[2] < v[0] && js.iter().filter_map(|j| j["train_nats"].as_f64()).all(|x| x.is_finite()), "{lg}");
        assert!(ex["roundtrip_bytes_exact"] == true && ex["roundtrip_trits_exact"] == true && ex["argmax_agree"].as_f64().unwrap() > 0.97, "{ex}");
        let cc = Cfg::from_dir(&format!("{s}/run")).unwrap();
        assert!(cc.lpe && cc.lphi == 4.0 && cc.lsq == 64);
        let mut lm = lut::Lm::open(&format!("{s}/run/model.gf17"), 64 << 20).unwrap();
        assert!(lm.lz.as_ref().map(|z| z.tab == cc.lz().unwrap().tab && z.s == cc.lz().unwrap().s).unwrap_or(false) && lm.pos.iter().all(|&p| p == 0.0));
        let w: Vec<u32> = txt[100..164].iter().map(|&b| b as u32).collect();
        let a = lm.fwd(&w).unwrap();
        lm.gf = false;
        let bb = lm.fwd(&w).unwrap();
        assert!(a.iter().zip(&bb).all(|(p, q)| p.to_bits() == q.to_bits()), "GF(17) discrete-log boosts must be bit-exact to the direct exponent table");
        let (_, mut kv) = lm.fwd_kv(&w[..40]).unwrap();
        let st: Vec<f32> = w[40..].iter().fold(Vec::new(), |_, &tk| lm.step(&mut kv, tk).unwrap());
        assert!(st.iter().zip(&a[63 * 256..]).all(|(p, q)| (p - q).abs() <= 1e-5 * q.abs().max(1.0)), "KV-cache decoding must agree with the full window");
        let _ = fs::remove_dir_all(&r);
    }
    #[test]
    #[ignore]
    fn lpe_lut_bench() {
        let s = std::env::temp_dir().join(format!("adamt-lpe-bench-{}", std::process::id())).to_string_lossy().to_string();
        fs::create_dir_all(&s).unwrap();
        <NdArray as Backend>::seed(&DEV, 11);
        let base = Cfg { d: 256, l: 8, h: 4, t: 256, ff: 1024, ns: vec![], rb: 15, tern: true, mem: false, v: 256, bpe: String::new(), base: String::new(), lphi: 4.0, lsq: 64, ..Cfg::env() };
        let core = Core::<NdArray>::new(&base, &DEV);
        let (w, it): (Vec<u32>, usize) = ((0..256u64).map(|i| (sm(i) % 256) as u32).collect(), var("LPE_BENCH_IT").ok().and_then(|v| v.parse().ok()).unwrap_or(5));
        [("pos", false, true), ("lpe_gf17", true, true), ("lpe_ref", true, false)].iter().for_each(|&(nm, lpe, gf)| {
            let p = format!("{s}/{nm}.gf17");
            lut::save(&p, &Cfg { lpe, ..base.clone() }, &core, None::<&Mem<NdArray>>, None, 128).unwrap();
            let mut lm = lut::Lm::open(&p, 64 << 20).unwrap();
            lm.gf = gf;
            let _ = lm.fwd(&w).unwrap();
            let t0 = Instant::now();
            (0..it).for_each(|_| { let _ = lm.fwd(&w).unwrap(); });
            let fw = t0.elapsed().as_secs_f64() / it as f64;
            let t1 = Instant::now();
            (0..it).for_each(|_| { let (_, mut kv) = lm.fwd_kv(&w[..1]).unwrap(); w[1..].iter().for_each(|&tk| { let _ = lm.step(&mut kv, tk).unwrap(); }); });
            let sp = t1.elapsed().as_secs_f64() / it as f64 / 255.0;
            let t2 = Instant::now();
            let n = (0..it * 2000).map(|i| { let (mut q, mut k) = (vec![1f32; 256], vec![1f32; 256]); if let Some(z) = &lm.lz { z.app(&mut q, &mut k, i % 256, gf); } q[i % 256] }).sum::<f32>();
            let ap = t2.elapsed().as_secs_f64() / (it * 2000) as f64;
            println!("{{\"arm\": \"{nm}\", \"threads\": {}, \"window_fwd_ms\": {:.2}, \"window_tok_per_s\": {:.1}, \"kv_step_ms\": {:.3}, \"kv_tok_per_s\": {:.1}, \"boost_us_per_pos_layer\": {:.3}, \"chk\": {n}}}", rayon::current_num_threads(), fw * 1e3, 256.0 / fw, sp * 1e3, 1.0 / sp, ap * 1e6);
        });
        let _ = fs::remove_dir_all(&s);
    }
}
#[cfg(test)]
mod mem_scale_tests {
    use super::*;
    use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
    type Ad = Autodiff<NdArray>;
    const DEV: NdArrayDevice = NdArrayDevice::Cpu;
    fn tiny(s: &str, v: usize, rb: u32) -> Cfg { Cfg { d: 64, l: 1, h: 1, t: 64, ff: 128, ns: vec![2, 3, 4], rb, tern: true, mem: true, v, bpe: String::new(), bs: 4, micro: 2, base: String::new(), init: String::new(), lazy: false, sw: String::new(), ep: 0.0, seed: 3, par: 2, wall: 0.0, unit: 50.0, home: s.into(), corpus: "syn".into(), out: format!("{s}/base"), thf: format!("{s}/thr.txt"), steps: 30, warm: 5, lr: 3e-3, lrm: 3e-3, ev: 10, ck: 10, evb: 1, lpe: false, lphi: 4.0, lsq: 64, copy: false, rmix: 0.0, ..Cfg::env() } }
    fn dir(tag: &str) -> String { let s = std::env::temp_dir().join(format!("adamt-mem-{tag}-{}", std::process::id())).to_string_lossy().to_string(); let _ = fs::remove_dir_all(&s); fs::create_dir_all(format!("{s}/data/syn")).unwrap(); fs::write(format!("{s}/thr.txt"), b"0\n").unwrap(); s }
    fn bits(v: &[f32]) -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() }
    fn rows(m: &Mem<NdArray>) -> Vec<f32> { f1(m.tab.val()) }
    #[test]
    fn hash_is_prefix_consistent_and_grow_is_bit_exact() {
        let (a, b) = (6u32, 11u32);
        let (mut hi, mut lo, mut n) = (0usize, 0usize, 0usize);
        (0..4000u64).for_each(|i| { let x: Vec<u32> = (0..9).map(|j| (sm(i * 9 + j) % 4096) as u32).collect(); [false, true].iter().for_each(|&w| [2usize, 3, 4, 6, 8].iter().enumerate().for_each(|(k, &g)| (0..9).for_each(|p| { let (ro, rn) = (hrow(&x, p, k, g, a, w), hrow(&x, p, k, g, b, w)); n += 1; hi += (rn >> (b - a) == ro) as usize; lo += ((rn & ((1 << a) - 1)) | k << a == ro) as usize; }))); });
        println!("{{\"lookups\": {n}, \"old_row_eq_new_row_shr\": {hi}, \"old_row_eq_new_row_mask\": {lo}}}");
        assert!(hi == n && (lo as f64) < 0.05 * n as f64, "top-bit addressing: the old row is new >> (b - a), not new & (2^a - 1)");
        [256usize, 512].iter().for_each(|&v| {
            let c = tiny("", v, a);
            <NdArray as Backend>::seed(&DEV, 5);
            let (core, mem) = (Core::<NdArray>::new(&c, &DEV), Mem::<NdArray>::new(&c, &DEV));
            let x: Vec<u32> = (0..2 * c.t as u64).map(|i| (sm(i + v as u64) % (v as u64 / 8)) as u32).collect();
            let (cb, mg) = (Cfg { rb: b, ..c.clone() }, mem.clone().grow(c.k(), a, b));
            let (o, g, s) = (rows(&mem), rows(&mg), (b - a) as usize);
            assert!(mg.tab.dims() == [c.k() << b, c.d] && (0..c.k() << b).all(|r| g[r * c.d..(r + 1) * c.d] == o[((r >> b) << a | (r & ((1 << b) - 1)) >> s) * c.d..][..c.d]));
            let (l0, l1) = (f1(logits(&c, &core, Some(&mem), None, &x, 2, &DEV)), f1(logits(&cb, &core, Some(&mg), None, &x, 2, &DEV)));
            let mk = Mem::<NdArray> { tab: Param::from_tensor(mem.tab.val().select(0, it1((0..c.k() << b).map(|r| ((r >> b) << a | (r & ((1 << a) - 1))) as i32).collect(), &DEV))) };
            let l2 = f1(logits(&cb, &core, Some(&mk), None, &x, 2, &DEV));
            println!("{{\"v\": {v}, \"grown_logits_bit_exact\": {}, \"mask_seeded_max_abs_dlogit\": {:.3e}}}", bits(&l0) == bits(&l1), l0.iter().zip(&l2).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max));
            assert!(bits(&l0) == bits(&l1) && bits(&l0) != bits(&l2), "grown logits must be bit-identical, and low-bit seeding must not be");
        });
    }
    #[test]
    fn grown_gf17_equals_grown_export_and_lut_is_bit_exact() {
        let s = dir("gf");
        let (a, b) = (6u32, 16u32);
        let c = tiny(&s, 256, a);
        <NdArray as Backend>::seed(&DEV, 9);
        let (core, mem) = (Core::<NdArray>::new(&c, &DEV), Mem::<NdArray>::new(&c, &DEV));
        let (p0, p1, p2) = (format!("{s}/m6.gf17"), format!("{s}/m16.gf17"), format!("{s}/m16x.gf17"));
        lut::save(&p0, &c, &core, Some(&mem), None, 128).unwrap();
        let r = lut::grow(&p0, b, &p1).unwrap();
        let (cb, mg) = (Cfg { rb: b, ..c.clone() }, mem.clone().grow(c.k(), a, b));
        let (it, _) = lut::save(&p2, &cb, &core, Some(&mg), None, 128).unwrap();
        let (n, t) = lut::msrc(&mg);
        let f = |x: usize, y: usize| f1(t.clone().slice([x..y]));
        let st = adam_gf17::Store::open(&p2, 1 << 20).unwrap();
        println!("{r} pages {}", st.tabs[st.table("mem").unwrap()].np());
        assert!(st.tabs[st.table("mem").unwrap()].np() >= 2 && lut::check(&p2, &it, Some((n, &f))).unwrap(), "streamed multi-page export must roundtrip bit-exactly");
        assert!(fs::read(&p1).unwrap() == fs::read(&p2).unwrap(), "growing the .gf17 file must equal exporting the grown model");
        assert!(lut::grow(&p1, a, &format!("{s}/bad.gf17")).is_err() && !lut::check(&p0, &it, Some((n, &f))).unwrap());
        let (l0, l1) = (lut::Lm::open(&p0, 8 << 20).unwrap(), lut::Lm::open(&p1, 8 << 20).unwrap());
        let w: Vec<u32> = (0..c.t as u64).map(|i| (sm(i * 3) % 40) as u32).collect();
        assert!(l1.rb == b && bits(&l0.fwd(&w).unwrap()) == bits(&l1.fwd(&w).unwrap()), "LUT logits of the grown file must be bit-identical");
        let ((_, mut k0), (_, mut k1)) = (l0.fwd_kv(&w[..20]).unwrap(), l1.fwd_kv(&w[..20]).unwrap());
        assert!(w[20..].iter().all(|&tk| bits(&l0.step(&mut k0, tk).unwrap()) == bits(&l1.step(&mut k1, tk).unwrap())), "KV decode on the grown file must be bit-identical");
        let (gl, cl) = (f1(logits(&cb, &core, Some(&mg), None, &w, 1, &DEV)), l1.fwd(&w).unwrap());
        let ag = gl.chunks_exact(c.v).zip(cl.chunks_exact(c.v)).filter(|(p, q)| lut::am(p) == lut::am(q)).count() as f64 / c.t as f64;
        println!("{{\"lut_vs_burn_grown_argmax_agree\": {ag}}}");
        assert!(ag > 0.9);
        let _ = fs::remove_dir_all(&s);
    }
    #[test]
    fn lazy_step_matches_adamw_on_touched_rows_and_leaves_the_rest() {
        let c = tiny("", 256, 6);
        <NdArray as Backend>::seed(&DEV, 4);
        let (core, mem) = (Core::<Ad>::new(&c, &DEV), Mem::<Ad>::new(&c, &DEV));
        let xy = |o: u64| -> (Vec<u32>, Vec<u32>) { let v: Vec<u32> = (0..2 * c.t as u64 + 1).map(|i| (sm(i + o) % 30) as u32).collect(); (v[..2 * c.t].to_vec(), v[1..].to_vec()) };
        let ((x, y), (x2, y2), lr) = (xy(0), xy(1 << 20), 1e-2);
        let mut om = AdamWConfig::new().with_weight_decay(0.0).with_beta_2(0.95).init::<Ad, Mem<Ad>>();
        fn dense<O: Optimizer<Mem<Ad>, Ad>>(c: &Cfg, core: &Core<Ad>, m: Mem<Ad>, om: &mut O, x: &[u32], y: &[u32], lr: f64) -> Mem<Ad> { let mut g = ce(logits(c, core, Some(&m), None, x, 2, &DEV), y, &DEV).backward(); let gp = GradientsParams::from_module(&mut g, &m); om.step(lr, m, gp) }
        let sparse = |m: &Mem<Ad>, o: Lazy<NdArray>, x: &[u32], y: &[u32], t: usize| { let (ix, ml) = mleaf(&c, m, x, 2, &DEV); let g = ml.grad(&ce(lgm(&c, &core, Some(ml.clone()), None, x, 2, &DEV), y, &DEV).backward()).unwrap(); let (o, tb) = o.step(m.tab.val().inner(), vec![(ix.clone(), g)], c.k(), lr, t, &DEV); (o, Mem::<Ad> { tab: Param::from_tensor(Tensor::from_inner(tb)) }, ix) };
        let t0 = f1(mem.tab.val().inner());
        let d1 = dense(&c, &core, mem.clone(), &mut om, &x, &y, lr);
        let (o, s1, ix) = sparse(&mem, Lazy::new(c.rows(), c.d, &DEV), &x, &y, 1);
        let (a, b) = (f1(d1.tab.val().inner()), f1(s1.tab.val().inner()));
        let hit: std::collections::HashSet<usize> = ix.iter().map(|&r| r as usize).collect();
        let md = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        println!("{{\"touched_rows\": {}, \"rows\": {}, \"lazy_vs_adamw_step1_max_abs\": {md:.3e}}}", hit.len(), c.rows());
        assert!(md == 0.0 && (0..c.rows()).filter(|r| !hit.contains(r)).all(|r| b[r * c.d..(r + 1) * c.d] == t0[r * c.d..(r + 1) * c.d]) && hit.iter().all(|&r| b[r * c.d..(r + 1) * c.d] != t0[r * c.d..(r + 1) * c.d]));
        let (_, s2, ix2) = sparse(&s1, o, &x2, &y2, 2);
        let (h2, b2): (std::collections::HashSet<usize>, Vec<f32>) = (ix2.iter().map(|&r| r as usize).collect(), f1(s2.tab.val().inner()));
        let d2 = f1(dense(&c, &core, d1, &mut om, &x2, &y2, lr).tab.val().inner());
        let idle: Vec<usize> = hit.iter().filter(|r| !h2.contains(r)).copied().collect();
        assert!(!idle.is_empty() && idle.iter().all(|&r| b2[r * c.d..(r + 1) * c.d] == b[r * c.d..(r + 1) * c.d]) && idle.iter().any(|&r| d2[r * c.d..(r + 1) * c.d] != a[r * c.d..(r + 1) * c.d]), "lazy rows move only when touched; dense AdamW keeps moving them by momentum");
    }
    #[test]
    fn init_grows_memory_continues_training_and_resumes() {
        let s = dir("init");
        let txt: Vec<u8> = (0..40000u64).map(|i| b"the cat sat on a mat. a dog ran to the cat. "[(i % 44) as usize] ^ (sm(i) % 50 == 0) as u8).collect();
        fs::write(format!("{s}/data/syn/train.bin"), &txt).unwrap();
        fs::write(format!("{s}/data/syn/hold.bin"), &txt[..8000]).unwrap();
        let c0 = tiny(&s, 256, 6);
        train::<Ad>(&c0, DEV, 0).unwrap();
        let run = |tag: &str, rb: u32, lr: f64, lazy: bool, steps: usize| { let c = Cfg { init: format!("{s}/base"), rb, lr, lrm: lr, lazy, steps, out: format!("{s}/{tag}"), ..c0.clone() }; train::<Ad>(&c, DEV, 0).unwrap(); Cfg::from_dir(&c.out).unwrap() };
        let lg = |c: &Cfg, x: &[u32]| { let (co, me, _) = load::<NdArray>(c, &DEV).unwrap(); f1(logits(c, &co, me.as_ref(), None, x, 2, &DEV)) };
        let js = |d: &str| fs::read_to_string(format!("{d}/log.jsonl")).unwrap().lines().filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()).collect::<Vec<_>>();
        let w: Vec<u32> = txt[300..300 + 2 * c0.t].iter().map(|&b| b as u32).collect();
        let cb = Cfg::from_dir(&format!("{s}/base")).unwrap();
        [(true, "z_lazy"), (false, "z_dense")].iter().for_each(|&(lz, tag)| {
            let cz = run(tag, 9, 0.0, lz, 12);
            assert!(cz.rb == 9 && cz.lazy == lz && cz.init == format!("{s}/base") && fs::metadata(format!("{s}/{tag}/ckpt/mem.bin")).unwrap().len() > (3u64 << 9) * 64 * 4);
            assert!(bits(&lg(&cb, &w)) == bits(&lg(&cz, &w)), "LR 0 continuation from a grown memory must reproduce the base logits bit-exactly ({tag})");
            let (l0, l1) = (lut::Lm::open(&format!("{s}/base/model.gf17"), 8 << 20).unwrap(), lut::Lm::open(&format!("{s}/{tag}/model.gf17"), 8 << 20).unwrap());
            assert!(l1.rb == 9 && bits(&l0.fwd(&w[..64]).unwrap()) == bits(&l1.fwd(&w[..64]).unwrap()), "grown export must give bit-identical LUT logits ({tag})");
            let ex = js(&format!("{s}/{tag}")).into_iter().find(|j| j.get("export").is_some()).unwrap();
            assert!(ex["roundtrip_bytes_exact"] == true && ex["roundtrip_trits_exact"] == true && ex["argmax_agree"].as_f64().unwrap() > 0.97, "{ex}");
        });
        let cl = run("lazy", 9, 3e-3, true, 30);
        fs::create_dir_all(format!("{s}/split")).unwrap();
        fs::write(format!("{s}/split/STOP"), b"").unwrap();
        let _ = run("split", 9, 3e-3, true, 30);
        assert_eq!(fs::read_to_string(format!("{s}/split/ckpt/state.txt")).unwrap().trim(), "1");
        fs::remove_file(format!("{s}/split/STOP")).unwrap();
        let cs = run("split", 9, 3e-3, true, 30);
        let (ml, ms) = (load::<NdArray>(&cl, &DEV).unwrap().1.unwrap(), load::<NdArray>(&cs, &DEV).unwrap().1.unwrap());
        assert!(bits(&rows(&ml)) == bits(&rows(&ms)), "a lazy run resumed from its checkpoint (moments included) must match the uninterrupted run");
        let (g, d, k) = (rows(&ml), cl.d, cl.k());
        let fam = (0..k << 6).filter(|&o| { let r0 = ((o >> 6) << 9 | (o & 63) << 3) * d; (1..8).any(|j| g[r0 + j * d..r0 + (j + 1) * d] != g[r0..r0 + d]) }).count();
        let v: Vec<f64> = js(&format!("{s}/lazy")).iter().filter_map(|j| j["val_nats"].as_f64()).collect();
        println!("{{\"lazy_val_nats\": {v:?}, \"old_rows_with_diverged_children\": {fam}, \"old_rows\": {}}}", k << 6);
        assert!(v.len() == 3 && v.iter().all(|x| x.is_finite()) && v[2] < v[0] && fam > 0);
        let cd = run("dense", 9, 3e-3, false, 30);
        let vd: Vec<f64> = js(&cd.out).iter().filter_map(|j| j["val_nats"].as_f64()).collect();
        assert!(vd.len() == 3 && vd.iter().all(|x| x.is_finite()) && vd[2] < vd[0]);
        let mut bad = Cfg { init: format!("{s}/base"), rb: 5, out: format!("{s}/shrink"), ..c0.clone() };
        assert!(train::<Ad>(&bad, DEV, 0).is_err());
        bad = Cfg { rb: 9, base: format!("{s}/base"), ..bad };
        assert!(train::<Ad>(&bad, DEV, 0).is_err());
        let _ = fs::remove_dir_all(&s);
    }
}
