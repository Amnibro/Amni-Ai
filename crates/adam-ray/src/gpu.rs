use crate::{cpu, pack::*};
use wgpu::*;
const OUTCAP: usize = 64 << 20;
pub struct Gpu { dev: Device, q: Queue, pipe: ComputePipeline, bgl: BindGroupLayout, views: Vec<TextureView>, st: Buffer, bufs: Option<(Buffer, Buffer, Buffer, BindGroup, usize, usize)>, pub name: String }
fn mask(f: fn(u32) -> bool) -> String { (0..8).map(|w| format!("{}u", (0..32).fold(0u32, |m, i| m | (f(w * 32 + i) as u32) << i))).collect::<Vec<_>>().join(",") }
pub fn common_wgsl(g: &Geo, state: bool) -> String { (include_str!("hash.wgsl").to_string() + if state { include_str!("state.wgsl") } else { "" }).replace("$TE", &g.t.to_string()).replace("$QT", &g.qt.to_string()).replace("$BASE", &g.base.to_string()).replace("$LVH", &(g.base + g.hops).to_string()).replace("$HB", &g.hb().to_string()).replace("$TA", &format!("{:?}f", g.ta)).replace("$TC", &format!("{:?}f", g.tc)).replace("$WORD", &mask(cpu::wordch)).replace("$PUNC", &mask(cpu::punct)) }
pub fn device(label: &str, extra: Features) -> Result<(Device, Queue, String), String> {
    let inst = Instance::new(&InstanceDescriptor { backends: Backends::PRIMARY, ..Default::default() });
    let pick = std::env::var("ADAM_RAY_ADAPTER").ok().and_then(|n| inst.enumerate_adapters(Backends::PRIMARY).into_iter().find(|a| a.get_info().name.to_lowercase().contains(&n.to_lowercase())));
    let ad = pick.or_else(|| pollster::block_on(inst.request_adapter(&RequestAdapterOptions { power_preference: PowerPreference::HighPerformance, ..Default::default() }))).filter(|a| a.get_info().device_type != DeviceType::Cpu).ok_or("no gpu adapter")?;
    let al = ad.limits();
    let lim = Limits { max_push_constant_size: al.max_push_constant_size.min(128), max_storage_buffer_binding_size: al.max_storage_buffer_binding_size, max_buffer_size: al.max_buffer_size, max_storage_buffers_per_shader_stage: al.max_storage_buffers_per_shader_stage, max_storage_textures_per_shader_stage: al.max_storage_textures_per_shader_stage, max_texture_dimension_2d: al.max_texture_dimension_2d, max_texture_array_layers: al.max_texture_array_layers, max_compute_workgroups_per_dimension: al.max_compute_workgroups_per_dimension, ..Default::default() };
    let (dev, q) = pollster::block_on(ad.request_device(&DeviceDescriptor { label: Some(label), required_features: Features::PUSH_CONSTANTS | (extra & ad.features()), required_limits: lim, memory_hints: MemoryHints::Performance }, None)).map_err(|e| e.to_string())?;
    Ok((dev, q, format!("{} ({:?})", ad.get_info().name, ad.get_info().backend)))
}
impl Gpu {
    pub fn new(tb: &Tables) -> Result<Gpu, String> {
        let (dev, q, name) = device("adam-ray", Features::empty())?;
        let mk = |w: usize, h: usize, d: usize, f: TextureFormat, bpp: usize, arr: bool, data: &dyn Fn(usize) -> Vec<u8>| {
            let t = dev.create_texture(&TextureDescriptor { label: None, size: Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: d as u32 }, mip_level_count: 1, sample_count: 1, dimension: TextureDimension::D2, format: f, usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST, view_formats: &[] });
            (0..d).for_each(|z| { let mut b = data(z); b.resize(w * h * bpp, 0); q.write_texture(TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: Origin3d { x: 0, y: 0, z: z as u32 }, aspect: TextureAspect::All }, &b, TexelCopyBufferLayout { offset: 0, bytes_per_row: Some((w * bpp) as u32), rows_per_image: Some(h as u32) }, Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 }); q.submit([]); });
            t.create_view(&TextureViewDescriptor { dimension: Some(if arr { TextureViewDimension::D2Array } else { TextureViewDimension::D2 }), ..Default::default() })
        };
        let views = vec![
            mk(TW, tb.t.div_ceil(TW), tb.l, TextureFormat::Rg32Uint, 8, true, &|z| bytemuck::cast_slice(&tb.cells[z * tb.t..(z + 1) * tb.t]).to_vec()),
            mk(TW, tb.rows.div_ceil(16), 1, TextureFormat::R16Float, 2, false, &|_| bytemuck::cast_slice(&tb.dp).to_vec()),
            mk(1024, 1024, 1, TextureFormat::Rgba32Uint, 16, false, &|_| bytemuck::cast_slice(&tb.lx.lex).to_vec()),
            mk(V, 1, 1, TextureFormat::R32Float, 4, false, &|_| bytemuck::cast_slice(&tb.bias).to_vec()),
            mk(360, 1, 1, TextureFormat::Rg32Float, 8, false, &|_| bytemuck::cast_slice(&tb.lx.trig).to_vec()),
        ];
        let src = common_wgsl(&tb.geo, true) + include_str!("ray.wgsl");
        let md = dev.create_shader_module(ShaderModuleDescriptor { label: Some("ray"), source: ShaderSource::Wgsl(src.into()) });
        let tx = |b: u32, s: TextureSampleType, d: TextureViewDimension| BindGroupLayoutEntry { binding: b, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: s, view_dimension: d, multisampled: false }, count: None };
        let sb = |b: u32, ro: bool| BindGroupLayoutEntry { binding: b, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let (fl, d2) = (TextureSampleType::Float { filterable: false }, TextureViewDimension::D2);
        let bgl = dev.create_bind_group_layout(&BindGroupLayoutDescriptor { label: None, entries: &[tx(0, TextureSampleType::Uint, TextureViewDimension::D2Array), tx(1, fl, d2), tx(2, TextureSampleType::Uint, d2), tx(3, fl, d2), tx(4, fl, d2), sb(5, false), sb(6, true), sb(7, false)] });
        let pl = dev.create_pipeline_layout(&PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[PushConstantRange { stages: ShaderStages::COMPUTE, range: 0..24 }] });
        let pipe = dev.create_compute_pipeline(&ComputePipelineDescriptor { label: Some("ray"), layout: Some(&pl), module: &md, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let st = dev.create_buffer(&BufferDescriptor { label: None, size: (2 * MAXB * NS * 4) as u64, usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC, mapped_at_creation: false });
        Ok(Gpu { dev, q, pipe, bgl, views, st, bufs: None, name })
    }
    pub fn reset(&mut self, cur: usize, nb: usize) { self.q.write_buffer(&self.st, (cur * MAXB * NS * 4) as u64, bytemuck::cast_slice(&vec![cpu::init(); nb])); }
    fn ensure(&mut self, nio: usize, nout: usize) {
        let (ci, co) = self.bufs.as_ref().map_or((0, 0), |b| (b.4, b.5));
        if nio <= ci && nout <= co { return; }
        let (ci, co) = (nio.max(ci).max(MAXB), nout.max(co).max(MAXB * V));
        let mb = |n: usize, u: BufferUsages| self.dev.create_buffer(&BufferDescriptor { label: None, size: (n * 4) as u64, usage: u, mapped_at_creation: false });
        let (io, out, rd) = (mb(ci, BufferUsages::STORAGE | BufferUsages::COPY_DST), mb(co, BufferUsages::STORAGE | BufferUsages::COPY_SRC), mb(co, BufferUsages::MAP_READ | BufferUsages::COPY_DST));
        let mut ents: Vec<BindGroupEntry> = self.views.iter().enumerate().map(|(i, v)| BindGroupEntry { binding: i as u32, resource: BindingResource::TextureView(v) }).collect();
        ents.extend([(5, &self.st), (6, &io), (7, &out)].map(|(i, b)| BindGroupEntry { binding: i, resource: b.as_entire_binding() }));
        let bg = self.dev.create_bind_group(&BindGroupDescriptor { label: None, layout: &self.bgl, entries: &ents });
        self.bufs = Some((io, out, rd, bg, ci, co));
    }
    pub fn run(&mut self, cur: &mut usize, nb: usize, par: &[u32], byt: &[u8], want: &[bool]) -> Vec<f32> {
        let (mut res, per, mut t0) = (Vec::with_capacity(want.iter().filter(|&&w| w).count() * nb * V), (OUTCAP / (nb * V * 4)).max(1), 0);
        while t0 < want.len() {
            let (mut t1, mut nw) = (t0, 0);
            while t1 < want.len() && (nw < per || !want[t1]) { nw += want[t1] as usize; t1 += 1; }
            self.ensure((t1 - t0) * nb, nw * nb * V);
            let (io, out, rd, bg, _, _) = self.bufs.as_ref().unwrap();
            let pk: Vec<u32> = (t0 * nb..t1 * nb).map(|i| par[i] | (byt[i] as u32) << 16).collect();
            self.q.write_buffer(io, 0, bytemuck::cast_slice(&pk));
            let mut enc = self.dev.create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut ps = enc.begin_compute_pass(&ComputePassDescriptor { label: None, timestamp_writes: None });
                ps.set_pipeline(&self.pipe);
                ps.set_bind_group(0, bg, &[]);
                let mut o = 0;
                for t in t0..t1 {
                    ps.set_push_constants(0, bytemuck::cast_slice(&[nb as u32, *cur as u32, 1 - *cur as u32, want[t] as u32, ((t - t0) * nb) as u32, o as u32]));
                    ps.dispatch_workgroups(nb as u32, 1, 1);
                    (*cur, o) = (1 - *cur, o + want[t] as usize * nb * V);
                }
            }
            let sz = (nw * nb * V * 4) as u64;
            (sz > 0).then(|| enc.copy_buffer_to_buffer(out, 0, rd, 0, sz));
            self.q.submit([enc.finish()]);
            if sz > 0 {
                let sl = rd.slice(..sz);
                sl.map_async(MapMode::Read, |_| ());
                self.dev.poll(Maintain::Wait);
                res.extend_from_slice(bytemuck::cast_slice(&sl.get_mapped_range()));
                rd.unmap();
            }
            t0 = t1;
        }
        res
    }
}
