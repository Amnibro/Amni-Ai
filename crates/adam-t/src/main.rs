use adam_ray::rng::Pcg64;
use adam_t::{bpe::Bpe, f1, fa::{self, Fa}, infer, load, lut, mmap, nap, thr, train, Cfg, Dat};
use burn::{backend::{ndarray::NdArrayDevice, wgpu::WgpuDevice, Autodiff, NdArray, Wgpu}, tensor::{backend::{AutodiffBackend, Backend}, module::attention, ops::AttentionModuleOptions, Distribution, Tensor}};
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
fn mm_h32(m: usize, n: usize, k: usize, it: usize, dt: burn::tensor::DType) -> f64 {
    type C = burn::backend::wgpu::CubeBackend<burn::backend::wgpu::WgpuRuntime, f32, i32, u32>;
    let dev = WgpuDevice::DiscreteGpu(0);
    let (a, b) = (Tensor::<C, 2>::random([m, k], Distribution::Normal(0.0, 1.0), &dev).cast(dt).into_primitive().tensor(), Tensor::<C, 2>::random([k, n], Distribution::Normal(0.0, 1.0), &dev).cast(dt).into_primitive().tensor());
    let go = || burn_cubecl::kernel::matmul::matmul(a.clone(), b.clone(), None, burn_cubecl::kernel::matmul::MatmulStrategy::Autotune, burn::tensor::DType::F32).unwrap();
    let sy = |o| { let _ = Tensor::<C, 2>::from_primitive(burn::tensor::TensorPrimitive::Float(o)).slice([0..1, 0..1]).into_data(); };
    (0..3).for_each(|_| sy(go()));
    let t0 = Instant::now();
    let mut o = go();
    (1..it).for_each(|_| o = go());
    sy(o);
    2.0 * (m * n * k * it) as f64 / t0.elapsed().as_secs_f64() / 1e9
}
type Fw<'a> = Box<dyn Fn(&[Vec<u32>]) -> io::Result<Vec<Vec<f32>>> + 'a>;
fn burn_fw<'a, B: Fa>(c: &'a Cfg, dev: B::Device) -> io::Result<Fw<'a>> {
    let (core, mem) = load::<B>(c, &dev)?;
    Ok(Box::new(move |xs: &[Vec<u32>]| { let t0 = Instant::now(); let x: Vec<u32> = xs.iter().flat_map(|w| w.iter().copied().chain(std::iter::repeat(32).take(c.t - w.len()))).collect(); let lg = infer(c, &core, mem.as_ref(), &x, xs.len(), &dev); nap(thr(&c.thf), t0.elapsed().as_secs_f64() * 1e3, c.unit); Ok(xs.iter().enumerate().map(|(i, w)| lg[i * c.t * c.v..(i * c.t + w.len()) * c.v].to_vec()).collect()) }))
}
fn fa_go<B: AutodiffBackend + Fa>(x: &Tensor<B::InnerBackend, 2>, r: &Tensor<B::InnerBackend, 2>, s: [usize; 3], fl: bool, rd: bool) -> (Vec<f32>, Vec<f32>) {
    let ([b, t, h], xa) = (s, Tensor::<B, 2>::from_inner(x.clone()).require_grad());
    let o = if fl { fa::fa(xa.clone(), b, t, h) } else { let [q, k, v] = fa::split(xa.clone(), b, t, h); attention(q, k, v, None, None, AttentionModuleOptions { scale: None, softcap: None, is_causal: true }).swap_dims(1, 2).reshape([b * t, x.dims()[1] / 3]) };
    let g = (o.clone() * Tensor::<B, 2>::from_inner(r.clone())).sum().backward();
    let dx = xa.grad(&g).unwrap();
    if rd { (f1(o.inner()), f1(dx)) } else { let _ = dx.slice([0..1, 0..1]).into_data(); (vec![], vec![]) }
}
fn fa_cmp<B: AutodiffBackend + Fa>(dev: &B::Device, s: [usize; 3], sd: f64) -> String {
    let [b, t, h] = s;
    let (x, r) = (Tensor::<B::InnerBackend, 2>::random([b * t, 3 * 64 * h], Distribution::Normal(0.0, sd), dev), Tensor::<B::InnerBackend, 2>::random([b * t, 64 * h], Distribution::Normal(0.0, 1.0), dev));
    let ((o1, g1), (o2, g2)) = (fa_go::<B>(&x, &r, s, true, true), fa_go::<B>(&x, &r, s, false, true));
    let md = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    format!("{{\"backend\": \"{}\", \"b\": {b}, \"t\": {t}, \"h\": {h}, \"sd\": {sd}, \"out_max_abs_diff\": {:.3e}, \"grad_max_abs_diff\": {:.3e}, \"grad_max_abs\": {:.3e}, \"finite\": {}}}", std::any::type_name::<B>().split('<').nth(1).unwrap_or(""), md(&o1, &o2), md(&g1, &g2), g2.iter().fold(0f32, |m, v| m.max(v.abs())), o1.iter().chain(&g1).all(|v| v.is_finite()))
}
struct Srv<'a> { t: usize, v: usize, bp: Option<Bpe>, fw: Fw<'a>, lm: Option<&'a lut::Lm>, name: String, step: usize, fmt: String, stop: Vec<Vec<u8>>, dec: lut::Dec }
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
    fn gen(&self, pr: &[u8], mx: usize, stop: &[Vec<u8>], d: &lut::Dec) -> io::Result<Vec<u8>> {
        let (t, v) = (self.t, self.v);
        let x = self.en(if pr.is_empty() { b"\n" } else { pr });
        let mut x = x[x.len().saturating_sub(t - 1)..].to_vec();
        let (mut kv, mut o, mut rg): (Option<lut::Kv>, Vec<u32>, Pcg64) = (None, Vec::new(), Pcg64::new(Some(d.seed)));
        let mut lg = match self.lm { Some(lm) => { let (l, k) = lm.fwd_kv(&x)?; kv = Some(k); l[l.len() - v..].to_vec() } None => { let l = (self.fw)(&[x.clone()])?.remove(0); l[l.len() - v..].to_vec() } };
        let g = loop {
            let nt = d.pick(&lg, &o, &mut rg) as u32;
            o.push(nt);
            x.push(nt);
            let mut g = self.de(&o);
            let cut = stop.iter().filter(|s| !s.is_empty()).filter_map(|s| g.windows(s.len()).position(|w| w == &s[..])).min();
            if let Some(c) = cut.or_else(|| d.rs.then(|| lut::rstop(&g)).flatten()) { g.truncate(c); break g; }
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
            "hello" => (json!({"model": format!("adam-t-{}", self.name), "step": self.step, "fmt": self.fmt, "ctx": (self.t as f64 * if self.v > 256 { 3.25 } else { 1.0 }) as usize, "ctx_symbols": self.t, "vocab": self.v, "proto": 1, "gen": self.dec.json()}), false),
            "lp" => (json!({"lp": self.lp(&q["items"].as_array().cloned().unwrap_or_default().iter().map(|p| Ok((d(&p[0])?, d(&p[1])?))).collect::<io::Result<Vec<_>>>()?)?}), false),
            "gen" => (json!({"text": B64.encode(self.gen(&d(&q["prompt"])?, q["max"].as_u64().unwrap_or(64) as usize, &[q["stop"].as_array().cloned().unwrap_or_default().iter().map(d).collect::<io::Result<Vec<_>>>()?, self.stop.clone()].concat(), &self.dec.with(&q)?)?)}), false),
            "bye" => (json!({"bye": true}), true),
            o => (json!({"error": format!("bad op {o}")}), false),
        })
    }
}
fn lut_mb() -> usize { var("LUT_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1024) << 20 }
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
            [(16384usize, 1024usize, 256usize), (16384, 256, 1024), (16384, 768, 256), (8192, 1536, 512), (8192, 2048, 512), (8192, 512, 2048), (8192, 4096, 512)].iter().for_each(|&(m, n, k)| {
                let t0 = Instant::now();
                let w = mm_wgsl(m, n, k, it).unwrap_or(-1.0);
                let b = mm_burn::<Wgpu>(WgpuDevice::DiscreteGpu(0), m, n, k, it);
                let h = mm_burn::<Wgpu<burn::tensor::f16>>(WgpuDevice::DiscreteGpu(0), m, n, k, it);
                let (h32, b32) = (mm_h32(m, n, k, it, burn::tensor::DType::F16), mm_h32(m, n, k, it, burn::tensor::DType::BF16));
                let th = adam_t::thr(&c.thf);
                let s = adam_t::nap(th, t0.elapsed().as_secs_f64() * 1e3, c.unit);
                let cpu = if var("MM_CPU").is_ok() { mm_burn::<NdArray>(NdArrayDevice::Cpu, m, n, k, 3) } else { 0.0 };
                println!("{{\"m\": {m}, \"n\": {n}, \"k\": {k}, \"handwritten_wgsl_gflops\": {w:.0}, \"burn_wgpu_f32_gflops\": {b:.0}, \"burn_vulkan_f16_gflops\": {h:.0}, \"burn_f16in_f32out_gflops\": {h32:.0}, \"burn_bf16in_f32out_gflops\": {b32:.0}, \"burn_ndarray_cpu_gflops\": {cpu:.0}, \"throttle_s\": {th}, \"slept_ms\": {s:.0}}}");
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
            let lm = gf.then(|| lut::Lm::open(&pth, lut_mb())).transpose()?;
            let dir = if gf { std::path::Path::new(&pth).parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default() } else { pth.clone() };
            let cc = (!gf).then(|| Cfg::from_dir(&dir)).transpose()?;
            let meta: Value = fs::read_to_string(format!("{dir}/meta.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({}));
            let step = meta["step"].as_u64().map(|v| v as usize).or_else(|| fs::read_to_string(format!("{dir}/ckpt/state.txt")).ok().and_then(|s| s.trim().parse().ok())).unwrap_or(0usize);
            let fw: Fw = match (&lm, &cc) { (Some(lm), _) => Box::new(move |xs: &[Vec<u32>]| xs.par_iter().map(|w| lm.fwd(w)).collect()), (_, Some(cc)) => if var("BACKEND").map(|b| b == "ndarray").unwrap_or(false) { burn_fw::<NdArray>(cc, NdArrayDevice::Cpu)? } else { burn_fw::<Wgpu>(cc, WgpuDevice::DiscreteGpu(0))? }, _ => unreachable!() };
            let bp = match (&lm, &cc) { (Some(l), _) => l.bpe.as_ref().map(|b| Bpe::new(adam_t::bpe::PAT, b.merges.clone())).transpose()?, (_, Some(c)) => c.bpe()?, _ => None };
            let fmt = meta["fmt"].as_str().map(String::from).unwrap_or_else(|| (if cc.as_ref().map(|c| c.sft()).unwrap_or_else(|| Cfg::from_dir(&dir).map(|c| c.sft()).unwrap_or(false)) { "chat" } else { "plain" }).into());
            let stop = if fmt == "chat" { vec![b"\n\n".to_vec(), b"\nUser:".to_vec()] } else { vec![] };
            let dec = lut::Dec::new(fmt == "chat").with(&var("GEN_DEFAULTS").ok().map(|s| serde_json::from_str::<Value>(&s)).transpose().map_err(io::Error::other)?.unwrap_or_default())?;
            let sv = Srv { t: lm.as_ref().map(|l| l.t).or(cc.as_ref().map(|c| c.t)).unwrap_or(256), v: lm.as_ref().map(|l| l.v).or(cc.as_ref().map(|c| c.v)).unwrap_or(256), bp, fw, lm: lm.as_ref(), name: meta["name"].as_str().map(String::from).unwrap_or_else(|| std::path::Path::new(&dir).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()), step, fmt, stop, dec };
            let mut so = io::stdout().lock();
            for l in io::stdin().lock().lines() { let l = l?; if l.trim().is_empty() { continue; } let (r, bye) = sv.req(&l).unwrap_or_else(|e| (json!({"error": e.to_string()}), false)); writeln!(so, "{r}")?; so.flush()?; if bye { break; } }
        }
        "grow" => { let (s, rb, o) = (a.get(2), a.get(3).and_then(|v| v.parse::<u32>().ok()), a.get(4)); println!("{}", lut::grow(s.ok_or_else(|| io::Error::other("adam-t grow <model.gf17> <MEM_LOG2> <out.gf17>"))?, rb.ok_or_else(|| io::Error::other("bad MEM_LOG2"))?, o.ok_or_else(|| io::Error::other("adam-t grow <model.gf17> <MEM_LOG2> <out.gf17>"))?)?); }
        "lutbench" => {
            let (p, np, ns) = (a.get(2).ok_or_else(|| io::Error::other("adam-t lutbench <model.gf17> [prompt tokens] [decode steps]"))?, a.get(3).and_then(|s| s.parse().ok()).unwrap_or(256usize), a.get(4).and_then(|s| s.parse().ok()).unwrap_or(256usize));
            let (r0, t0) = (adam_gf17::rss(), Instant::now());
            let lm = lut::Lm::open(p, lut_mb())?;
            let (to, r1) = (t0.elapsed().as_secs_f64(), adam_gf17::rss());
            let hold = if lm.v == 256 { Dat::B(mmap(&format!("{}/data/{}/hold.bin", c.home, c.corpus))?) } else { Dat::W(mmap(&format!("{}/data/{}/hold.bpe{}.u16", c.home, c.corpus, lm.v))?) };
            let x = hold.get(hold.len() / 3, (np + ns).min(lm.t));
            let t1 = Instant::now();
            let (lg, mut kv) = lm.fwd_kv(&x[..np])?;
            let pf = t1.elapsed().as_secs_f64() * 1e3;
            let (mut h, mut ms) = (lg.iter().fold(0u64, |h, v| (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01B3)), Vec::new());
            for &tk in &x[np..] { let t = Instant::now(); let l = lm.step(&mut kv, tk)?; ms.push(t.elapsed().as_secs_f64() * 1e3); h = l.iter().fold(h, |h, v| (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01B3)); }
            let (st, r2) = (lm.st.lock().map_err(|_| io::Error::other("store lock"))?, adam_gf17::rss());
            let mut srt = ms.clone();
            srt.sort_by(|a, b| a.total_cmp(b));
            let q = |f: f64| srt.get(((srt.len() as f64 - 1.0) * f).round() as usize).copied().unwrap_or(0.0);
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
            println!("{}", json!({"model": p, "rb": lm.rb, "k": lm.k, "d": lm.d, "file_bytes": fs::metadata(p)?.len(), "budget_mb": lut_mb() >> 20, "threads": rayon::current_num_threads(), "open_s": to, "prompt_tokens": np, "prefill_ms": pf, "decode_steps": ms.len(), "step_ms_first16": mean(&ms[..ms.len().min(16)]), "step_ms_mean": mean(&ms), "step_ms_p50": q(0.5), "step_ms_p90": q(0.9), "step_ms_last_half": mean(&ms[ms.len() / 2..]), "rss_start_mb": r0.0 >> 20, "rss_open_mb": r1.0 >> 20, "rss_end_mb": r2.0 >> 20, "hwm_mb": r2.1 >> 20, "rss_file_mb": r2.2 >> 20, "store_used_mb": st.used >> 20, "tile_hits": st.hits, "tile_miss": st.miss, "tile_evict": st.ev, "logit_hash": format!("{h:016x}")}));
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
            fs::write(format!("{od}/meta.json"), json!({"name": std::path::Path::new(&dir).file_name().map(|s| s.to_string_lossy().to_string()), "step": step, "run": dir, "fmt": if cc.sft() { "chat" } else { "plain" }}).to_string())?;
        }
        "tokenize" => {
            let (bp, src, dst) = (Bpe::open(&a[2])?, mmap(&a[3])?, &a[4]);
            let t0 = Instant::now();
            let ids = bp.enc_file(&src, 4 << 20);
            fs::write(dst, ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            println!("{{\"src\": \"{}\", \"bytes\": {}, \"tokens\": {}, \"bytes_per_token\": {:.4}, \"seconds\": {:.1}, \"out\": \"{dst}\"}}", a[3], src.len(), ids.len(), src.len() as f64 / ids.len() as f64, t0.elapsed().as_secs_f64());
        }
        "fa" => {
            let g = |k: &str, d: usize| var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
            let (s, it, md) = ([g("FA_B", 8), g("FA_T", 512), g("FA_H", 8)], g("FA_IT", 20), var("FA_MODE").unwrap_or_else(|_| "check".into()));
            let dev = WgpuDevice::DiscreteGpu(0);
            if md == "xmm" {
                type C = burn::backend::wgpu::CubeBackend<burn::backend::wgpu::WgpuRuntime, f32, i32, u32>;
                let (n, i, o) = (g("FA_N", 8192), g("FA_I", 512), g("FA_O", 2048));
                let (x, w, dy) = (Tensor::<C, 2>::random([n, i], Distribution::Normal(0.0, 1.0), &dev).round(), Tensor::<C, 2>::random([i, o], Distribution::Normal(0.0, 1.0), &dev).round().clamp(-1.0, 1.0), Tensor::<C, 2>::random([n, o], Distribution::Normal(0.0, 1.0), &dev));
                let tm = |nm: &str, f: &dyn Fn() -> Tensor<C, 2>, fl: f64| { let _ = f().slice([0..1, 0..1]).into_data(); let t0 = Instant::now(); let mut r = f(); (1..it).for_each(|_| r = f()); let _ = r.slice([0..1, 0..1]).into_data(); let ms = t0.elapsed().as_secs_f64() * 1e3 / it as f64; println!("{{\"xmm\": \"{nm}\", \"ms\": {ms:.3}, \"tflops\": {:.2}}}", fl / ms / 1e9); };
                let fl = 2.0 * (n * i * o) as f64;
                tm("f32 x.w", &|| x.clone().matmul(w.clone()), fl);
                tm("xmm11 x.w", &|| fa::xmm(x.clone(), w.clone(), 1, 1), fl);
                tm("f32 dy.wT", &|| dy.clone().matmul(w.clone().transpose()), fl);
                tm("xmm31 dy.wT", &|| fa::xmm(dy.clone(), w.clone().transpose(), 3, 1), fl);
                tm("f32 xT.dy", &|| x.clone().transpose().matmul(dy.clone()), fl);
                tm("xmm13 xT.dy", &|| fa::xmm(x.clone().transpose(), dy.clone(), 1, 3), fl);
                tm("cast bf16 dy", &|| dy.clone().cast(burn::tensor::DType::BF16).cast(burn::tensor::DType::F32), 1.0);
                tm("cast bf16 xT", &|| x.clone().transpose().cast(burn::tensor::DType::BF16).cast(burn::tensor::DType::F32), 1.0);
            } else if md == "lin" {
                let (n, i, o) = (g("FA_N", 4096), g("FA_I", 512), g("FA_O", 2048));
                [true, false].iter().for_each(|&q| {
                    let (x, w, r) = (Tensor::<Wgpu, 2>::random([n, i], Distribution::Normal(0.0, 1.0), &dev), Tensor::<Wgpu, 2>::random([i, o], Distribution::Normal(0.0, 0.02), &dev), Tensor::<Wgpu, 2>::random([n, o], Distribution::Normal(0.0, 1.0), &dev));
                    let run = |f: bool| { let (xa, wa) = (Tensor::<Autodiff<Wgpu>, 2>::from_inner(x.clone()).require_grad(), Tensor::<Autodiff<Wgpu>, 2>::from_inner(w.clone()).require_grad()); let y = if f { fa::tl(xa.clone(), wa.clone(), q) } else { fa::tl_ref(xa.clone(), wa.clone(), q) }; let gr = (y.clone() * Tensor::<Autodiff<Wgpu>, 2>::from_inner(r.clone())).sum().backward(); (f1(y.inner()), f1(xa.grad(&gr).unwrap()), f1(wa.grad(&gr).unwrap())) };
                    let bt = |f: bool| { let t0 = Instant::now(); (0..it).for_each(|_| { let (xa, wa) = (Tensor::<Autodiff<Wgpu>, 2>::from_inner(x.clone()).require_grad(), Tensor::<Autodiff<Wgpu>, 2>::from_inner(w.clone()).require_grad()); let y = if f { fa::tl(xa.clone(), wa.clone(), q) } else { fa::tl_ref(xa.clone(), wa.clone(), q) }; let gr = (y * Tensor::<Autodiff<Wgpu>, 2>::from_inner(r.clone())).sum().backward(); let _ = wa.grad(&gr).unwrap().slice([0..1, 0..1]).into_data(); let _ = xa.grad(&gr).unwrap().slice([0..1, 0..1]).into_data(); }); t0.elapsed().as_secs_f64() * 1e3 / it as f64 };
                    let (_, _) = (bt(true), bt(false));
                    println!("{{\"lin_ms\": {{\"tern\": {q}, \"fused\": {:.2}, \"ref\": {:.2}}}}}", bt(true), bt(false));
                    let ((y1, x1, w1), (y2, x2, w2)) = (run(true), run(false));
                    let md = |a: &[f32], b: &[f32]| (a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max), b.iter().fold(0f32, |m, v| m.max(v.abs())));
                    println!("{{\"lin_check\": {{\"tern\": {q}, \"n\": {n}, \"i\": {i}, \"o\": {o}, \"y_maxdiff_maxabs\": {:?}, \"dx_maxdiff_maxabs\": {:?}, \"dw_maxdiff_maxabs\": {:?}}}}}", md(&y1, &y2), md(&x1, &x2), md(&w1, &w2));
                });
            } else if md == "check" {
                [([1usize, 128usize, 2usize], 1.0f64), ([1, 128, 2], 3.0)].iter().for_each(|&(s, sd)| println!("{}", fa_cmp::<Autodiff<NdArray>>(&NdArrayDevice::Cpu, s, sd)));
                [([2usize, 256usize, 8usize], 1.0f64), ([2, 256, 8], 3.0), ([1, 1024, 8], 1.0), (s, 1.0)].iter().for_each(|&(s, sd)| { println!("{}", fa_cmp::<Autodiff<Wgpu>>(&dev, s, sd)); nap(thr(&c.thf), 200.0, c.unit); });
            } else {
                let [b, t, h] = s;
                let (x, r) = (Tensor::<Wgpu, 2>::random([b * t, 3 * 64 * h], Distribution::Normal(0.0, 1.0), &dev), Tensor::<Wgpu, 2>::random([b * t, 64 * h], Distribution::Normal(0.0, 1.0), &dev));
                let fl = md == "flash";
                let v0 = fs::read_to_string("/sys/class/drm/card1/device/mem_info_vram_used").ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(0.0);
                (0..3).for_each(|_| { fa_go::<Autodiff<Wgpu>>(&x, &r, s, fl, false); });
                let (mut ms, mut sl) = (0f64, 0f64);
                (0..it).for_each(|_| { let t0 = Instant::now(); fa_go::<Autodiff<Wgpu>>(&x, &r, s, fl, false); let m = t0.elapsed().as_secs_f64() * 1e3; ms += m; sl += nap(thr(&c.thf), m, c.unit); });
                let v1 = fs::read_to_string("/sys/class/drm/card1/device/mem_info_vram_used").ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(0.0);
                println!("{{\"mode\": \"{md}\", \"b\": {b}, \"t\": {t}, \"h\": {h}, \"fwd_bwd_ms\": {:.2}, \"tokens_per_s\": {:.0}, \"vram_delta_gb\": {:.3}, \"slept_ms\": {sl:.0}}}", ms / it as f64, (b * t) as f64 * 1e3 * it as f64 / ms, (v1 - v0) / 1e9);
            }
        }
        "bpe-enc" => { let mut b = Vec::new(); io::stdin().read_to_end(&mut b)?; println!("{}", Bpe::open(&a[2])?.enc(&b).iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")); }
        m => { (c.sft() == (m == "sft")).then_some(()).ok_or_else(|| io::Error::other("adam-t sft needs BASE=<base run dir>, and adam-t train must run without BASE"))?; run(&c, 0)?; }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use adam_t::{Core, Mem};
    use burn::{module::Module, record::{BinFileRecorder, FullPrecisionSettings}};
    use std::{collections::HashSet, sync::Mutex};
    const P: &[u8] = b"User: tell me about cows\nAdam:";
    static SEED: Mutex<()> = Mutex::new(());
    fn rep3(g: &[u8]) -> bool { g.windows(3).collect::<HashSet<_>>().len() < g.len().saturating_sub(2) }
    fn tiny(tag: &str) -> (Cfg, lut::Lm) {
        let s = std::env::temp_dir().join(format!("adamt-dec-{tag}-{}", std::process::id())).to_string_lossy().to_string();
        let _ = fs::remove_dir_all(&s);
        fs::create_dir_all(format!("{s}/ckpt")).unwrap();
        let c = Cfg { d: 64, l: 1, h: 1, t: 64, ff: 128, ns: vec![2, 3], rb: 6, tern: true, mem: false, v: 256, bpe: String::new(), base: String::new(), out: s.clone(), thf: format!("{s}/none"), ..Cfg::env() };
        let core = { let _g = SEED.lock().unwrap(); <NdArray as Backend>::seed(&NdArrayDevice::Cpu, 7); Core::<NdArray>::new(&c, &NdArrayDevice::Cpu) };
        core.clone().save_file(format!("{s}/ckpt/core"), &BinFileRecorder::<FullPrecisionSettings>::new()).unwrap();
        lut::save(&format!("{s}/m.gf17"), &c, &core, None::<&Mem<NdArray>>, 128).unwrap();
        let lm = lut::Lm::open(&format!("{s}/m.gf17"), 64 << 20).unwrap();
        (c, lm)
    }
    fn srv<'a>(c: &'a Cfg, lm: Option<&'a lut::Lm>) -> Srv<'a> { Srv { t: c.t, v: c.v, bp: None, fw: lm.map_or_else(|| burn_fw::<NdArray>(c, NdArrayDevice::Cpu).unwrap(), |l| Box::new(move |xs: &[Vec<u32>]| xs.iter().map(|w| l.fwd(w)).collect())), lm, name: "tiny".into(), step: 1, fmt: "chat".into(), stop: vec![], dec: lut::Dec::new(true) } }
    #[test]
    fn gen_no_repeat_ngram_lut_and_burn() {
        let (c, lm) = tiny("nr");
        [srv(&c, Some(&lm)), srv(&c, None)].iter().for_each(|s| {
            let (g0, g3) = (s.gen(P, 48, &[], &lut::Dec::new(false)).unwrap(), s.gen(P, 48, &[], &lut::Dec { nr: 3, ..lut::Dec::new(false) }).unwrap());
            assert!(g0.len() == 48 && rep3(&g0), "greedy on the tiny model should loop: {g0:?}");
            assert!(g3.len() == 48 && !rep3(&g3) && g3 != g0, "no_repeat_ngram 3 must leave no repeated trigram: {g3:?}");
            assert_eq!(g3, s.gen(P, 48, &[], &lut::Dec { nr: 3, ..lut::Dec::new(false) }).unwrap());
        });
        let _ = fs::remove_dir_all(&c.out);
    }
    #[test]
    fn gen_rep_penalty_lut_and_burn() {
        let (c, lm) = tiny("rp");
        [srv(&c, Some(&lm)), srv(&c, None)].iter().for_each(|s| {
            let n = |rp: f32| s.gen(P, 48, &[], &lut::Dec { rp, ..lut::Dec::new(false) }).unwrap().into_iter().collect::<HashSet<u8>>().len();
            let (a, b) = (n(1.0), n(3.0));
            assert!(b > a, "rep_penalty 3 should widen the reply vocabulary: {a} vs {b}");
        });
        let _ = fs::remove_dir_all(&c.out);
    }
    #[test]
    fn proto_gen_fields_round_trip() {
        let (c, lm) = tiny("proto");
        let s = srv(&c, Some(&lm));
        let (h, _) = s.req(r#"{"op":"hello"}"#).unwrap();
        assert!(h["proto"] == 1 && h["fmt"] == "chat" && h["gen"] == lut::Dec::new(true).json(), "{h}");
        let q = |x: Value| { let mut b = json!({"op": "gen", "prompt": B64.encode(P), "max": 40}); x.as_object().unwrap().iter().for_each(|(k, v)| b[k] = v.clone()); b.to_string() };
        let tx = |x: Value| B64.decode(s.req(&q(x)).unwrap().0["text"].as_str().unwrap()).unwrap();
        let sd = |n: u64| json!({"stop": [B64.encode(b"zz")], "no_repeat_ngram": 2, "rep_penalty": 1.3, "top_k": 5, "temperature": 0.8, "seed": n, "stop_repeat": false});
        let want = s.gen(P, 40, &[b"zz".to_vec()], &lut::Dec { nr: 2, rp: 1.3, k: 5, tp: 0.8, seed: 9, rs: false }).unwrap();
        assert!(!want.is_empty() && tx(sd(9)) == want && tx(sd(9)) == want && (10..16).any(|n| tx(sd(n)) != want));
        assert_eq!(tx(json!({})), s.gen(P, 40, &[], &lut::Dec::new(true)).unwrap());
        assert!(tx(json!({"seed": 123})) == tx(json!({})) && tx(json!({"no_repeat_ngram": 0, "rep_penalty": 1.0, "stop_repeat": false})) == s.gen(P, 40, &[], &lut::Dec::new(false)).unwrap());
        assert!(s.req(r#"{"op":"gen","prompt":"","rep_penalty":"x"}"#).is_err() && s.req(r#"{"op":"gen","prompt":"","top_k":-3}"#).is_err());
        let (b, bye) = s.req(r#"{"op":"bye"}"#).unwrap();
        assert!(bye && b["bye"] == true);
        let _ = fs::remove_dir_all(&c.out);
    }
}
