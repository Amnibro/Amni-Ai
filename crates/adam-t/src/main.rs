use adam_t::{bpe::Bpe, infer, load, lut, mmap, nap, thr, train, Cfg, Dat};
use burn::{backend::{ndarray::NdArrayDevice, wgpu::WgpuDevice, Autodiff, NdArray, Wgpu}, tensor::{backend::Backend, Distribution, Tensor}};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{env::{args, var}, fs, io::{self, BufRead, Read, Write}, time::Instant};
use wgpu::{util::{BufferInitDescriptor, DeviceExt}, *};
const MM: &str = "@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let r0 = wg.y * 64u; let c0 = wg.x * 64u; let tid = li.y * 16u + li.x; let n = p.y; let k = p.z;
  var acc: array<f32, 16>;
  for (var kk = 0u; kk < k; kk += 16u) {
    for (var i = 0u; i < 4u; i++) { let e = tid + i * 256u; sa[e] = a[(r0 + e / 16u) * k + kk + e % 16u]; sb[e] = b[(kk + e / 64u) * n + c0 + e % 64u]; }
    workgroupBarrier();
    for (var q = 0u; q < 16u; q++) { var av: array<f32, 4>; var bv: array<f32, 4>; for (var i = 0u; i < 4u; i++) { av[i] = sa[(li.y * 4u + i) * 16u + q]; bv[i] = sb[q * 64u + li.x * 4u + i]; } for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i * 4u + j] += av[i] * bv[j]; } } }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { c[(r0 + li.y * 4u + i) * n + c0 + li.x * 4u + j] = acc[i * 4u + j]; } }
}";
fn mm_wgsl(m: usize, n: usize, k: usize, it: usize) -> Result<f64, String> {
    let (dev, q, _) = adam_ray::gpu::device("adam-t-mm", Features::empty())?;
    let md = dev.create_shader_module(ShaderModuleDescriptor { label: Some("mm"), source: ShaderSource::Wgsl(MM.into()) });
    let pipe = dev.create_compute_pipeline(&ComputePipelineDescriptor { label: Some("mm"), layout: None, module: &md, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let bi = |v: &[f32]| dev.create_buffer_init(&BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(v), usage: BufferUsages::STORAGE });
    let (a, b) = (bi(&(0..m * k).map(|i| (i % 7) as f32 * 0.1).collect::<Vec<f32>>()), bi(&(0..k * n).map(|i| (i % 5) as f32 * 0.1).collect::<Vec<f32>>()));
    let c = dev.create_buffer(&BufferDescriptor { label: None, size: (m * n * 4) as u64, usage: BufferUsages::STORAGE, mapped_at_creation: false });
    let u = dev.create_buffer_init(&BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[m as u32, n as u32, k as u32, 0]), usage: BufferUsages::UNIFORM });
    let bg = dev.create_bind_group(&BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &[a.as_entire_binding(), b.as_entire_binding(), c.as_entire_binding(), u.as_entire_binding()].into_iter().enumerate().map(|(i, r)| BindGroupEntry { binding: i as u32, resource: r }).collect::<Vec<_>>() });
    let run = |r: usize| { let mut e = dev.create_command_encoder(&Default::default()); { let mut p = e.begin_compute_pass(&Default::default()); p.set_pipeline(&pipe); p.set_bind_group(0, &bg, &[]); (0..r).for_each(|_| p.dispatch_workgroups((n / 64) as u32, (m / 64) as u32, 1)); } q.submit([e.finish()]); dev.poll(Maintain::Wait); };
    run(2);
    let t0 = Instant::now();
    run(it);
    Ok(2.0 * (m * n * k * it) as f64 / t0.elapsed().as_secs_f64() / 1e9)
}
fn mm_burn<B: Backend>(dev: B::Device, m: usize, n: usize, k: usize, it: usize) -> f64 {
    let (a, b) = (Tensor::<B, 2>::random([m, k], Distribution::Normal(0.0, 1.0), &dev), Tensor::<B, 2>::random([k, n], Distribution::Normal(0.0, 1.0), &dev));
    (0..2).for_each(|_| { let _ = a.clone().matmul(b.clone()).sum().into_data(); });
    let t0 = Instant::now();
    let s = (0..it).fold(Tensor::<B, 1>::zeros([1], &dev), |s, _| s + a.clone().matmul(b.clone()).sum());
    let _ = s.into_data();
    2.0 * (m * n * k * it) as f64 / t0.elapsed().as_secs_f64() / 1e9
}
type Fw<'a> = Box<dyn Fn(&[Vec<u32>]) -> io::Result<Vec<Vec<f32>>> + 'a>;
fn burn_fw<'a, B: burn::tensor::backend::Backend>(c: &'a Cfg, dev: B::Device) -> io::Result<Fw<'a>> {
    let (core, mem) = load::<B>(c, &dev)?;
    Ok(Box::new(move |xs: &[Vec<u32>]| { let t0 = Instant::now(); let x: Vec<u32> = xs.iter().flat_map(|w| w.iter().copied().chain(std::iter::repeat(32).take(c.t - w.len()))).collect(); let lg = infer(c, &core, mem.as_ref(), &x, xs.len(), &dev); nap(thr(&c.thf), t0.elapsed().as_secs_f64() * 1e3, c.unit); Ok(xs.iter().enumerate().map(|(i, w)| lg[i * c.t * c.v..(i * c.t + w.len()) * c.v].to_vec()).collect()) }))
}
struct Srv<'a> { t: usize, v: usize, bp: Option<Bpe>, fw: Fw<'a>, lm: Option<&'a lut::Lm>, name: String, step: usize }
impl Srv<'_> {
    fn en(&self, b: &[u8]) -> Vec<u32> { self.bp.as_ref().map(|p| p.enc(b)).unwrap_or_else(|| b.iter().map(|&x| x as u32).collect()) }
    fn de(&self, i: &[u32]) -> Vec<u8> { self.bp.as_ref().map(|p| p.dec(i)).unwrap_or_else(|| i.iter().map(|&x| x as u8).collect()) }
    fn lp(&self, it: &[(Vec<u8>, Vec<u8>)]) -> io::Result<Vec<f64>> {
        let t = self.t;
        let w: Vec<(usize, usize, usize, Vec<u32>)> = it.iter().enumerate().flat_map(|(i, (a, b))| { let (a, b) = (self.en(if a.is_empty() { b"\n" } else { a }), self.en(b)); let (s, c0) = ([a.clone(), b].concat(), a.len()); let mut o = Vec::new(); let mut st = c0; while st < s.len() { let e = s.len().min(st + (t / 2).max(1)); let ws = (e - 1).saturating_sub(t); o.push((i, st - ws - 1, e - ws - 1, s[ws..e].to_vec())); st = e; } o }).collect();
        let nb = var("LP_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
        let mut r = vec![0f64; it.len()];
        for ch in w.chunks(nb) { let lg = (self.fw)(&ch.iter().map(|x| x.3[..x.3.len() - 1].to_vec()).collect::<Vec<_>>())?; ch.iter().zip(&lg).for_each(|((i, a, b, s), l)| r[*i] -= lut::nll(&l[a * self.v..b * self.v], &s[a + 1..b + 1]).iter().sum::<f64>()); }
        Ok(r)
    }
    fn gen(&self, pr: &[u8], mx: usize, stop: &[Vec<u8>]) -> io::Result<Vec<u8>> {
        let (t, v) = (self.t, self.v);
        let x = self.en(if pr.is_empty() { b"\n" } else { pr });
        let mut x = x[x.len().saturating_sub(t - 1)..].to_vec();
        let (mut kv, mut o): (Option<lut::Kv>, Vec<u32>) = (None, Vec::new());
        let mut lg = match self.lm { Some(lm) => { let (l, k) = lm.fwd_kv(&x)?; kv = Some(k); l[l.len() - v..].to_vec() } None => { let l = (self.fw)(&[x.clone()])?.remove(0); l[l.len() - v..].to_vec() } };
        let g = loop {
            let nt = lut::am(&lg) as u32;
            o.push(nt);
            x.push(nt);
            let mut g = self.de(&o);
            let cut = stop.iter().filter(|s| !s.is_empty()).filter_map(|s| g.windows(s.len()).position(|w| w == &s[..])).min();
            if let Some(c) = cut { g.truncate(c); break g; }
            if g.len() >= mx { g.truncate(mx); break g; }
            let full = x.len() > t;
            x = if full { x[x.len() - t / 2..].to_vec() } else { x };
            lg = match (self.lm, kv.as_mut()) { (Some(lm), Some(k)) if !full => lm.step(k, nt)?, (Some(lm), _) => { let (l, k) = lm.fwd_kv(&x)?; kv = Some(k); l[l.len() - v..].to_vec() } _ => { let l = (self.fw)(&[x.clone()])?.remove(0); l[l.len() - v..].to_vec() } };
        };
        Ok(g)
    }
    fn req(&self, l: &str) -> io::Result<(Value, bool)> {
        let q: Value = serde_json::from_str(l).map_err(io::Error::other)?;
        let d = |v: &Value| B64.decode(v.as_str().unwrap_or("")).map_err(io::Error::other);
        Ok(match q["op"].as_str().unwrap_or("") {
            "hello" => (json!({"model": format!("adam-t-{}", self.name), "step": self.step, "fmt": "plain", "ctx": (self.t as f64 * if self.v > 256 { 3.25 } else { 1.0 }) as usize, "ctx_symbols": self.t, "vocab": self.v, "proto": 1}), false),
            "lp" => (json!({"lp": self.lp(&q["items"].as_array().cloned().unwrap_or_default().iter().map(|p| Ok((d(&p[0])?, d(&p[1])?))).collect::<io::Result<Vec<_>>>()?)?}), false),
            "gen" => (json!({"text": B64.encode(self.gen(&d(&q["prompt"])?, q["max"].as_u64().unwrap_or(64) as usize, &q["stop"].as_array().cloned().unwrap_or_default().iter().map(d).collect::<io::Result<Vec<_>>>()?)?)}), false),
            "bye" => (json!({"bye": true}), true),
            o => (json!({"error": format!("bad op {o}")}), false),
        })
    }
}
fn run(c: &Cfg, bench: usize) -> io::Result<String> {
    match var("BACKEND").unwrap_or_else(|_| "wgpu".into()).as_str() {
        "ndarray" => train::<Autodiff<NdArray>>(c, NdArrayDevice::Cpu, bench),
        #[cfg(feature = "candle")]
        "candle" => train::<Autodiff<burn::backend::Candle>>(c, burn::backend::candle::CandleDevice::Cpu, bench),
        _ => if var("GCKPT").map(|v| v == "1").unwrap_or(false) { train::<Autodiff<Wgpu, burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing>>(c, WgpuDevice::DiscreteGpu(0), bench) } else { train::<Autodiff<Wgpu>>(c, WgpuDevice::DiscreteGpu(0), bench) },
    }
}
fn main() -> io::Result<()> {
    let a: Vec<String> = args().collect();
    let c = Cfg::env();
    match a.get(1).map(|s| s.as_str()).unwrap_or("train") {
        "bench" => { run(&c, var("BENCH_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(50))?; }
        "mm" => {
            let it = 50;
            [(16384usize, 1024usize, 256usize), (16384, 256, 1024), (16384, 768, 256)].iter().for_each(|&(m, n, k)| {
                let t0 = Instant::now();
                let w = mm_wgsl(m, n, k, it).unwrap_or(-1.0);
                let b = mm_burn::<Wgpu>(WgpuDevice::DiscreteGpu(0), m, n, k, it);
                let h = mm_burn::<Wgpu<burn::tensor::f16>>(WgpuDevice::DiscreteGpu(0), m, n, k, it);
                let th = adam_t::thr(&c.thf);
                let s = adam_t::nap(th, t0.elapsed().as_secs_f64() * 1e3, c.unit);
                let cpu = if var("MM_CPU").is_ok() { mm_burn::<NdArray>(NdArrayDevice::Cpu, m, n, k, 3) } else { 0.0 };
                println!("{{\"m\": {m}, \"n\": {n}, \"k\": {k}, \"handwritten_wgsl_gflops\": {w:.0}, \"burn_wgpu_f32_gflops\": {b:.0}, \"burn_vulkan_f16_gflops\": {h:.0}, \"burn_ndarray_cpu_gflops\": {cpu:.0}, \"throttle_s\": {th}, \"slept_ms\": {s:.0}}}");
            });
        }
        "lut" => {
            let (p, nw) = (a.get(2).expect("adam-t lut <model.gf17> [windows]"), a.get(3).and_then(|s| s.parse().ok()).unwrap_or(8usize));
            let lm = lut::Lm::open(p, 256 << 20)?;
            let hold = if lm.v == 256 { Dat::B(mmap(&format!("{}/data/{}/hold.bin", c.home, c.corpus))?) } else { Dat::W(mmap(&format!("{}/data/{}/hold.bpe{}.u16", c.home, c.corpus, lm.v))?) };
            let (x, y) = lut::hold_windows(&hold, nw, lm.t);
            let (_, bpb, bps) = lut::cpu_eval(&lm, &x, &y, &lm.bpe.as_ref().map(|b| b.lens()).unwrap_or_else(|| vec![1; 256]))?;
            println!("{{\"model\": \"{p}\", \"windows\": {nw}, \"cpu_lut_bpb\": {bpb:.5}, \"bytes_per_s\": {bps:.1}}}");
        }
        "serve" => {
            let pth = a.get(2).ok_or_else(|| io::Error::other("adam-t serve <model.gf17 | run dir>"))?.clone();
            let gf = pth.ends_with(".gf17");
            let lm = gf.then(|| lut::Lm::open(&pth, 1 << 30)).transpose()?;
            let dir = if gf { std::path::Path::new(&pth).parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default() } else { pth.clone() };
            let cc = (!gf).then(|| Cfg::from_dir(&dir)).transpose()?;
            let meta: Value = fs::read_to_string(format!("{dir}/meta.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({}));
            let step = meta["step"].as_u64().map(|v| v as usize).or_else(|| fs::read_to_string(format!("{dir}/ckpt/state.txt")).ok().and_then(|s| s.trim().parse().ok())).unwrap_or(0usize);
            let fw: Fw = match (&lm, &cc) { (Some(lm), _) => Box::new(move |xs: &[Vec<u32>]| xs.par_iter().map(|w| lm.fwd(w)).collect()), (_, Some(cc)) => if var("BACKEND").map(|b| b == "ndarray").unwrap_or(false) { burn_fw::<NdArray>(cc, NdArrayDevice::Cpu)? } else { burn_fw::<Wgpu>(cc, WgpuDevice::DiscreteGpu(0))? }, _ => unreachable!() };
            let bp = match (&lm, &cc) { (Some(l), _) => l.bpe.as_ref().map(|b| Bpe::new(adam_t::bpe::PAT, b.merges.clone())).transpose()?, (_, Some(c)) => c.bpe()?, _ => None };
            let sv = Srv { t: lm.as_ref().map(|l| l.t).or(cc.as_ref().map(|c| c.t)).unwrap_or(256), v: lm.as_ref().map(|l| l.v).or(cc.as_ref().map(|c| c.v)).unwrap_or(256), bp, fw, lm: lm.as_ref(), name: meta["name"].as_str().map(String::from).unwrap_or_else(|| std::path::Path::new(&dir).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()), step };
            let mut so = io::stdout().lock();
            for l in io::stdin().lock().lines() { let l = l?; if l.trim().is_empty() { continue; } let (r, bye) = sv.req(&l).unwrap_or_else(|e| (json!({"error": e.to_string()}), false)); writeln!(so, "{r}")?; so.flush()?; if bye { break; } }
        }
        "export" => {
            let dir = a.get(2).ok_or_else(|| io::Error::other("adam-t export <run dir>"))?.clone();
            let mut cc = Cfg::from_dir(&dir)?;
            cc.par = var("PARITY_WINDOWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
            cc.tern.then_some(()).ok_or_else(|| io::Error::other("export needs a ternary (TERN=1) run"))?;
            let step = fs::read_to_string(format!("{dir}/ckpt/state.txt"))?.trim().parse::<usize>().map_err(io::Error::other)?;
            let od = format!("{dir}/eval/step{step}");
            fs::create_dir_all(&od)?;
            let (core, mem) = load::<NdArray>(&cc, &NdArrayDevice::Cpu)?;
            let (ho, lens) = (cc.data("hold")?, cc.bpe()?.map(|b| b.lens()).unwrap_or_else(|| vec![1; 256]));
            let mut lg = fs::OpenOptions::new().create(true).append(true).open(format!("{od}/export.jsonl"))?;
            lut::finish(&cc, &core, mem.as_ref(), &ho, &lens, &NdArrayDevice::Cpu, &mut lg, &format!("{od}/model.gf17"))?;
            fs::write(format!("{od}/meta.json"), json!({"name": std::path::Path::new(&dir).file_name().map(|s| s.to_string_lossy().to_string()), "step": step, "run": dir}).to_string())?;
        }
        "tokenize" => {
            let (bp, src, dst) = (Bpe::open(&a[2])?, mmap(&a[3])?, &a[4]);
            let t0 = Instant::now();
            let ids = bp.enc_file(&src, 4 << 20);
            fs::write(dst, ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            println!("{{\"src\": \"{}\", \"bytes\": {}, \"tokens\": {}, \"bytes_per_token\": {:.4}, \"seconds\": {:.1}, \"out\": \"{dst}\"}}", a[3], src.len(), ids.len(), src.len() as f64 / ids.len() as f64, t0.elapsed().as_secs_f64());
        }
        "bpe-enc" => { let mut b = Vec::new(); io::stdin().read_to_end(&mut b)?; println!("{}", Bpe::open(&a[2])?.enc(&b).iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")); }
        _ => { run(&c, 0)?; }
    }
    Ok(())
}
