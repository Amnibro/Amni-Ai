use adam_t::{infer, load, lut, nap, thr, train, Cfg};
use burn::{backend::{ndarray::NdArrayDevice, wgpu::WgpuDevice, Autodiff, NdArray, Wgpu}, tensor::{backend::Backend, Distribution, Tensor}};
use serde_json::{json, Value};
use std::{env::{args, var}, io::{self, BufRead, Write}, time::Instant};
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
type Fw<'a> = Box<dyn Fn(&[Vec<u8>]) -> io::Result<Vec<Vec<f32>>> + 'a>;
fn opt(a: &[String], k: &str) -> Option<String> { a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned()) }
fn burn_fw<'a, B: burn::tensor::backend::Backend>(c: &'a Cfg, dev: B::Device) -> io::Result<Fw<'a>> {
    let (core, mem) = load::<B>(c, &dev)?;
    Ok(Box::new(move |xs: &[Vec<u8>]| { let t0 = Instant::now(); let x: Vec<u8> = xs.iter().flat_map(|w| w.iter().copied().chain(std::iter::repeat(b' ').take(c.t - w.len()))).collect(); let lg = infer(c, &core, mem.as_ref(), &x, xs.len(), &dev); nap(thr(&c.thf), t0.elapsed().as_secs_f64() * 1e3, c.unit); Ok(xs.iter().enumerate().map(|(i, w)| lg[i * c.t * 256..(i * c.t + w.len()) * 256].to_vec()).collect()) }))
}
fn serve(mode: &str, t: usize, f: &Fw) -> io::Result<()> {
    let (mut so, lines) = (io::stdout().lock(), io::stdin().lock().lines().collect::<io::Result<Vec<String>>>()?);
    let js: Vec<Value> = lines.iter().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).unwrap_or_else(|_| json!({"prompt": l, "context": l}))).collect();
    let nb = var("LP_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
    for ch in js.chunks(if mode == "gen" { 1 } else { nb }) {
        let out: Vec<Value> = if mode == "gen" {
            let (j, mut x) = (&ch[0], ch[0]["prompt"].as_str().unwrap_or("").as_bytes().to_vec());
            let (mx, stop, p0) = (j["max_new"].as_u64().unwrap_or(64) as usize, j["stop"].as_str().unwrap_or("\n").as_bytes().to_vec(), x.len().max(1));
            x = if x.is_empty() { vec![b'\n'] } else { x };
            for _ in 0..mx { let w = x[x.len().saturating_sub(t)..].to_vec(); let lg = f(&[w.clone()])?.remove(0); x.push(lut::am(&lg[(w.len() - 1) * 256..]) as u8); if !stop.is_empty() && x[p0..].ends_with(&stop) { break; } }
            let g = &x[p0..];
            vec![json!({"prompt": j["prompt"], "completion": String::from_utf8_lossy(if !stop.is_empty() && g.ends_with(&stop) { &g[..g.len() - stop.len()] } else { g }), "new_bytes": g.len()})]
        } else {
            let sq: Vec<(Vec<u8>, usize)> = ch.iter().map(|j| { let (cx, co) = (j["context"].as_str().unwrap_or("").as_bytes(), j["continuation"].as_str().unwrap_or("").as_bytes()); let s = [if cx.is_empty() { &b"\n"[..] } else { cx }, co].concat(); (s[s.len().saturating_sub(t + 1)..].to_vec(), co.len().min(t)) }).collect();
            let lg = f(&sq.iter().map(|(s, _)| s[..s.len() - 1].to_vec()).collect::<Vec<_>>())?;
            sq.iter().zip(&lg).zip(ch).map(|(((s, n), l), j)| { let (y, k) = (&s[1..], s.len() - 1 - n); let v = lut::nll(&l[k * 256..], &y[k..]); let gr = (k..y.len()).all(|p| lut::am(&l[p * 256..(p + 1) * 256]) == y[p] as usize); json!({"context": j["context"], "continuation": j["continuation"], "logprob": -v.iter().sum::<f64>(), "n_bytes": n, "greedy": gr, "truncated": s.len() == t + 1}) }).collect()
        };
        out.iter().try_for_each(|o| writeln!(so, "{o}"))?;
        so.flush()?;
    }
    Ok(())
}
fn run(c: &Cfg, bench: usize) -> io::Result<String> {
    match var("BACKEND").unwrap_or_else(|_| "wgpu".into()).as_str() {
        "ndarray" => train::<Autodiff<NdArray>>(c, NdArrayDevice::Cpu, bench),
        #[cfg(feature = "candle")]
        "candle" => train::<Autodiff<burn::backend::Candle>>(c, burn::backend::candle::CandleDevice::Cpu, bench),
        _ => train::<Autodiff<Wgpu>>(c, WgpuDevice::DiscreteGpu(0), bench),
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
            let hold = adam_t::mmap(&format!("{}/data/{}/hold.bin", c.home, c.corpus))?;
            let (x, y) = lut::hold_windows(&hold, nw, lm.t);
            let (_, bpb, bps) = lut::cpu_eval(&lm, &x, &y)?;
            println!("{{\"model\": \"{p}\", \"windows\": {nw}, \"cpu_lut_bpb\": {bpb:.5}, \"bytes_per_s\": {bps:.1}}}");
        }
        m @ ("gen" | "logprob") => {
            let (ck, gf) = (opt(&a, "--ckpt"), opt(&a, "--gf17"));
            let lm = gf.as_ref().map(|p| lut::Lm::open(p, 256 << 20)).transpose()?;
            let cc = ck.as_ref().map(|d| Cfg::from_dir(d)).transpose()?;
            let f: Fw = match (&lm, &cc) {
                (Some(lm), _) => Box::new(move |xs: &[Vec<u8>]| xs.iter().map(|w| lm.fwd(w)).collect()),
                (_, Some(cc)) => if var("BACKEND").map(|b| b == "ndarray").unwrap_or(false) { burn_fw::<NdArray>(cc, NdArrayDevice::Cpu)? } else { burn_fw::<Wgpu>(cc, WgpuDevice::DiscreteGpu(0))? },
                _ => return Err(io::Error::other("adam-t gen|logprob --ckpt <run dir> | --gf17 <model.gf17>")),
            };
            serve(m, lm.as_ref().map(|l| l.t).or(cc.as_ref().map(|c| c.t)).unwrap_or(256), &f)?;
        }
        _ => { run(&c, 0)?; }
    }
    Ok(())
}
