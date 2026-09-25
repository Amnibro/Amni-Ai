use adam_t::{lut, train, Cfg};
use burn::{backend::{ndarray::NdArrayDevice, wgpu::WgpuDevice, Autodiff, NdArray, Wgpu}, tensor::{backend::Backend, Distribution, Tensor}};
use std::{env::{args, var}, io, time::Instant};
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
                let w = mm_wgsl(m, n, k, it).unwrap_or(-1.0);
                let b = mm_burn::<Wgpu>(WgpuDevice::DiscreteGpu(0), m, n, k, it);
                let cpu = mm_burn::<NdArray>(NdArrayDevice::Cpu, m, n, k, 3);
                println!("{{\"m\": {m}, \"n\": {n}, \"k\": {k}, \"handwritten_wgsl_gflops\": {w:.0}, \"burn_wgpu_gflops\": {b:.0}, \"burn_ndarray_cpu_gflops\": {cpu:.0}}}");
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
        _ => { run(&c, 0)?; }
    }
    Ok(())
}
