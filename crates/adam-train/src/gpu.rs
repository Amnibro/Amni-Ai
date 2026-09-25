use crate::{cfg::*, cpu::sid0, v19::Drv19, Engine, Part};
use adam_ray::{gpu::{common_wgsl, device}, pack::{lexicon, Lex}};
use std::{cell::RefCell, collections::HashMap};
use wgpu::*;
const CH: u64 = 32 << 20;
enum R<'a> { T(&'a TextureView, TextureSampleType, TextureViewDimension), S(&'a TextureView, TextureFormat, TextureViewDimension), B(&'a Buffer, bool) }
struct Tx { dp: Texture, sid: Texture, sw: Texture, bi: Texture }
struct Prof { qs: QuerySet, rs: Buffer, rb: Buffer, nm: RefCell<Vec<&'static str>>, acc: RefCell<HashMap<&'static str, (f64, u32)>> }
pub struct Gpu { c: Cfg, dev: Device, q: Queue, name: String, tx: Tx, b: HashMap<&'static str, Buffer>, p: HashMap<&'static str, (ComputePipeline, BindGroup)>, rows: u32, pf: Option<Prof>, d19: Option<(Drv19, Lex, Vec<u32>, Vec<f32>, Vec<i32>)> }
fn pc(v: [u32; 12]) -> Vec<u8> { bytemuck::cast_slice(&v).to_vec() }
impl Gpu {
    pub fn new(c: &Cfg) -> Result<Gpu, String> {
        let pon = std::env::var("PROF").is_ok_and(|v| v == "1");
        let (dev, q, name) = device("adam-train", Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES | if pon { Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_PASSES } else { Features::empty() })?;
        let (lv, te, npos, m) = (c.lv, c.te, c.npos(), c.npos() * c.lv * 4);
        let rows = (te / 4096) as u32;
        let tex = |w: u32, h: u32, d: u32, f: TextureFormat, st: bool| dev.create_texture(&TextureDescriptor { label: None, size: Extent3d { width: w, height: h, depth_or_array_layers: d }, mip_level_count: 1, sample_count: 1, dimension: TextureDimension::D2, format: f, usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC | if st { TextureUsages::STORAGE_BINDING } else { TextureUsages::empty() }, view_formats: &[] });
        let tx = Tx { dp: tex(4096, (2 * DE * V / 4096) as u32, 1, TextureFormat::R32Float, true), sid: tex(4096, rows, lv as u32, TextureFormat::R32Uint, true), sw: tex(4096, rows, 3 * lv as u32, TextureFormat::R32Float, true), bi: tex(V as u32, 1, 1, TextureFormat::R32Float, true) };
        let lx = lexicon(&c.syn, c.geo.qt.then_some(c.wcnt.as_str()));
        let (lex, trg) = (tex(1024, 1024, 1, TextureFormat::Rgba32Uint, false), tex(360, 1, 1, TextureFormat::Rg32Float, false));
        let wt = |t: &Texture, z: u32, w: u32, h: u32, bpp: u32, d: &[u8]| q.write_texture(TexelCopyTextureInfo { texture: t, mip_level: 0, origin: Origin3d { x: 0, y: 0, z }, aspect: TextureAspect::All }, d, TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * bpp), rows_per_image: Some(h) }, Extent3d { width: w, height: h, depth_or_array_layers: 1 });
        wt(&lex, 0, 1024, 1024, 16, bytemuck::cast_slice(&lx.lex));
        wt(&trg, 0, 360, 1, 8, bytemuck::cast_slice(&lx.trig));
        let s0: Vec<u32> = sid0(1).chunks(3).map(|s| s[0] as u32 | (s[1] as u32) << 8 | (s[2] as u32) << 16).collect();
        let init = vec![s0[0]; te];
        (0..lv as u32).for_each(|l| wt(&tx.sid, l, 4096, rows, 4, bytemuck::cast_slice(&init)));
        let mut b = HashMap::new();
        let (st, cs, cd) = (BufferUsages::STORAGE, BufferUsages::COPY_SRC, BufferUsages::COPY_DST);
        for (k, n, u) in [("xy", npos, st | cd), ("stb", npos * 40, st), ("pi", m, st | cd), ("pw", m, st | cd), ("uk", m, st), ("ud", 2 * DE, st), ("ga", m * 6, st), ("best", m, st), ("dg", (V + DE) * V * 2, st), ("cnt", lv * te, st), ("ctr", 4, st | cd), ("ag", 6, st | BufferUsages::INDIRECT), ("dl", npos * V, st | cs), ("pl", npos, st | cs), ("cpb", npos * c.copy.max(1), st | cd), ("lq", 2 * lv, st | cs | cd), ("dm", 2 * DE * V, st | cs | cd), ("dv", 2 * DE * V, st | cs | cd), ("sm", lv * te * KS, st | cs | cd), ("sv", lv * te * KS, st | cs | cd), ("bp", npos.div_ceil(V) * V, st), ("rd", (CH / 4) as usize, BufferUsages::MAP_READ | cd)] {
            b.insert(k, dev.create_buffer(&BufferDescriptor { label: Some(k), size: (n * 4) as u64, usage: u, mapped_at_creation: false }));
        }
        let fl = |t: &Texture, d: TextureViewDimension| t.create_view(&TextureViewDescriptor { dimension: Some(d), ..Default::default() });
        let (a2, d2) = (TextureViewDimension::D2Array, TextureViewDimension::D2);
        let (vdp, vsid, vsw, vbi, vlex, vtrg) = (fl(&tx.dp, d2), fl(&tx.sid, a2), fl(&tx.sw, a2), fl(&tx.bi, d2), fl(&lex, d2), fl(&trg, d2));
        let (tf, tu) = (TextureSampleType::Float { filterable: false }, TextureSampleType::Uint);
        let f = |x: f32| format!("{x:?}f");
        let head = include_str!("wgsl/head.wgsl").replace("$LV", &lv.to_string()).replace("$HOPS", &c.geo.hops.to_string()).replace("$TERN", &c.tern.to_string()).replace("$QLEV", &c.qlev.to_string()).replace("$TT", &f(c.tern_th)).replace("$SG", &c.sgrid.to_string()).replace("$SLO", &f(c.slo)).replace("$SHI", &f(c.shi)).replace("$RTH", &f(c.rth)).replace("$RW", &f(c.rw)).replace("$WD", &f(c.wd)).replace("$COPY", &c.copy.to_string());
        let bb = |k: &str| &b[k];
        let (ro, rw) = (|k: &'static str| R::B(bb(k), true), |k: &'static str| R::B(bb(k), false));
        let specs: Vec<(&'static str, &str, &str, Vec<(u32, R)>)> = vec![
            ("state", include_str!("wgsl/drv.wgsl"), "state", vec![(0, R::T(&vlex, tu, d2)), (1, R::T(&vtrg, tf, d2)), (2, ro("xy")), (3, rw("stb"))]),
            ("plans", include_str!("wgsl/drv.wgsl"), "plans", vec![(3, rw("stb")), (4, rw("pi")), (5, rw("pw"))]),
            ("fwd", include_str!("wgsl/fwd.wgsl"), "main", vec![(0, R::T(&vdp, tf, d2)), (1, R::T(&vsid, tu, a2)), (2, R::T(&vsw, tf, a2)), (3, R::T(&vbi, tf, d2)), (4, ro("lq")), (5, ro("xy")), (6, rw("pi")), (7, rw("pw")), (8, rw("dl")), (9, rw("pl")), (10, ro("cpb"))]),
            ("hist", include_str!("wgsl/hist.wgsl"), "main", vec![(0, ro("pi")), (1, ro("pw")), (2, rw("cnt")), (3, rw("uk")), (4, rw("ud")), (5, rw("ctr"))]),
            ("args", include_str!("wgsl/args.wgsl"), "main", vec![(0, ro("ctr")), (1, rw("ag"))]),
            ("umap", include_str!("wgsl/acc.wgsl"), "umap", vec![(0, ro("ctr")), (1, ro("uk")), (2, rw("cnt"))]),
            ("slots", include_str!("wgsl/acc.wgsl"), "slots", vec![(2, rw("cnt")), (3, ro("pi")), (4, ro("pw")), (5, ro("dl")), (6, ro("xy")), (7, R::T(&vsid, tu, a2)), (8, rw("ga")), (9, rw("best")), (10, ro("cpb"))]),
            ("dacc", include_str!("wgsl/acc.wgsl"), "dense", vec![(3, ro("pi")), (4, ro("pw")), (5, ro("dl")), (8, rw("dg"))]),
            ("dense", include_str!("wgsl/dense.wgsl"), "main", vec![(0, R::S(&vdp, TextureFormat::R32Float, d2)), (1, rw("dm")), (2, rw("dv")), (3, ro("ctr")), (4, ro("ud")), (5, ro("uk")), (6, rw("dg")), (7, rw("cnt"))]),
            ("slot", include_str!("wgsl/slot.wgsl"), "main", vec![(0, R::S(&vsid, TextureFormat::R32Uint, a2)), (1, R::S(&vsw, TextureFormat::R32Float, a2)), (2, rw("sm")), (3, rw("sv")), (4, ro("ctr")), (5, ro("uk")), (6, rw("ga")), (7, rw("best")), (8, ro("xy")), (9, rw("cnt"))]),
            ("part", include_str!("wgsl/bias.wgsl"), "part", vec![(0, ro("dl")), (1, rw("bp"))]),
            ("fin", include_str!("wgsl/bias.wgsl"), "fin", vec![(1, rw("bp")), (2, R::S(&vbi, TextureFormat::R32Float, d2))]),
            ("lvl", include_str!("wgsl/lvl.wgsl"), "main", vec![(0, R::T(&vsw, tf, a2)), (1, rw("lq"))]),
        ];
        let mut p = HashMap::new();
        let mut mods: HashMap<*const u8, ShaderModule> = HashMap::new();
        for (k, src, ep, ents) in specs {
            let md = mods.entry(src.as_ptr()).or_insert_with(|| dev.create_shader_module(ShaderModuleDescriptor { label: Some(k), source: ShaderSource::Wgsl((common_wgsl(&c.geo, k == "state" || k == "plans") + &head + src).into()) }));
            let le: Vec<BindGroupLayoutEntry> = ents.iter().map(|(i, r)| BindGroupLayoutEntry { binding: *i, visibility: ShaderStages::COMPUTE, count: None, ty: match r { R::T(_, s, d) => BindingType::Texture { sample_type: *s, view_dimension: *d, multisampled: false }, R::S(_, f, d) => BindingType::StorageTexture { access: StorageTextureAccess::ReadWrite, format: *f, view_dimension: *d }, R::B(_, ro) => BindingType::Buffer { ty: BufferBindingType::Storage { read_only: *ro }, has_dynamic_offset: false, min_binding_size: None } } }).collect();
            let bgl = dev.create_bind_group_layout(&BindGroupLayoutDescriptor { label: Some(k), entries: &le });
            let pl = dev.create_pipeline_layout(&PipelineLayoutDescriptor { label: Some(k), bind_group_layouts: &[&bgl], push_constant_ranges: &[PushConstantRange { stages: ShaderStages::COMPUTE, range: 0..48 }] });
            let pipe = dev.create_compute_pipeline(&ComputePipelineDescriptor { label: Some(k), layout: Some(&pl), module: md, entry_point: Some(ep), compilation_options: Default::default(), cache: None });
            let be: Vec<BindGroupEntry> = ents.iter().map(|(i, r)| BindGroupEntry { binding: *i, resource: match r { R::T(v, _, _) | R::S(v, _, _) => BindingResource::TextureView(v), R::B(x, _) => x.as_entire_binding() } }).collect();
            let bg = dev.create_bind_group(&BindGroupDescriptor { label: Some(k), layout: &bgl, entries: &be });
            p.insert(k, (pipe, bg));
        }
        let pf = pon.then(|| Prof { qs: dev.create_query_set(&QuerySetDescriptor { label: None, ty: QueryType::Timestamp, count: 64 }), rs: dev.create_buffer(&BufferDescriptor { label: None, size: 512, usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC, mapped_at_creation: false }), rb: dev.create_buffer(&BufferDescriptor { label: None, size: 512, usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST, mapped_at_creation: false }), nm: RefCell::new(vec![]), acc: RefCell::new(HashMap::new()) });
        let d19 = (c.ver == 19).then(|| (Drv19::new(c.bs), lx, vec![], vec![], vec![]));
        Ok(Gpu { c: c.clone(), dev, q, name, tx, b, p, rows, pf, d19 })
    }
    fn mark(&self, ps: &mut ComputePass, k: &'static str) { if let Some(f) = &self.pf { let mut n = f.nm.borrow_mut(); ps.write_timestamp(&f.qs, n.len() as u32); n.push(k); } }
    fn go(&self, ps: &mut ComputePass, k: &'static str, pcs: &[u8]) { self.mark(ps, k); let (pp, bg) = &self.p[k]; ps.set_pipeline(pp); ps.set_bind_group(0, bg, &[]); ps.set_push_constants(0, pcs); }
    fn flat(&self, ps: &mut ComputePass, k: &'static str, pcs: &[u8], wg: usize) { self.go(ps, k, pcs); ps.dispatch_workgroups(wg.min(32768) as u32, wg.div_ceil(32768) as u32, 1); }
    fn read(&self, src: &dyn Fn(&mut CommandEncoder, u64, u64), n: u64) -> Vec<u8> {
        let (rd, mut out) = (&self.b["rd"], Vec::with_capacity(n as usize));
        let mut o = 0;
        while o < n {
            let k = (n - o).min(CH);
            let mut enc = self.dev.create_command_encoder(&CommandEncoderDescriptor { label: None });
            src(&mut enc, o, k);
            self.q.submit([enc.finish()]);
            let sl = rd.slice(..k);
            sl.map_async(MapMode::Read, |_| ());
            self.dev.poll(Maintain::Wait);
            out.extend_from_slice(&sl.get_mapped_range());
            rd.unmap();
            o += k;
        }
        out
    }
    fn rbuf(&self, k: &str, n: usize) -> Vec<f32> { let b = &self.b[k]; bytemuck::pod_collect_to_vec(&self.read(&|e, o, k| e.copy_buffer_to_buffer(b, o, &self.b["rd"], 0, k), (n * 4) as u64)) }
    fn rtex(&self, t: &Texture, z: u32, w: u32, h: u32) -> Vec<u8> {
        let rp = (CH / (w as u64 * 4)).min(h as u64) as u32;
        (0..h.div_ceil(rp)).flat_map(|i| { let hh = rp.min(h - i * rp); self.read(&|e, _, _| e.copy_texture_to_buffer(TexelCopyTextureInfo { texture: t, mip_level: 0, origin: Origin3d { x: 0, y: i * rp, z }, aspect: TextureAspect::All }, TexelCopyBufferInfo { buffer: &self.b["rd"], layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(hh) } }, Extent3d { width: w, height: hh, depth_or_array_layers: 1 }), (w * hh * 4) as u64) }).collect()
    }
    fn wtex(&self, t: &Texture, z: u32, w: u32, h: u32, d: &[u8]) { self.q.write_texture(TexelCopyTextureInfo { texture: t, mip_level: 0, origin: Origin3d { x: 0, y: 0, z }, aspect: TextureAspect::All }, d, TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) }, Extent3d { width: w, height: h, depth_or_array_layers: 1 }); }
    fn prof(&self, mut enc: CommandEncoder) {
        let Some(f) = &self.pf else { self.q.submit([enc.finish()]); return; };
        let n = f.nm.borrow().len() as u32;
        enc.resolve_query_set(&f.qs, 0..n, &f.rs, 0);
        enc.copy_buffer_to_buffer(&f.rs, 0, &f.rb, 0, 8 * n as u64);
        self.q.submit([enc.finish()]);
        let sl = f.rb.slice(..8 * n as u64);
        sl.map_async(MapMode::Read, |_| ());
        self.dev.poll(Maintain::Wait);
        let ts: Vec<u64> = bytemuck::pod_collect_to_vec(&sl.get_mapped_range());
        f.rb.unmap();
        let (nm, mut acc, per) = (f.nm.take(), f.acc.borrow_mut(), self.q.get_timestamp_period() as f64 * 1e-6);
        (0..nm.len().saturating_sub(1)).filter(|&i| !nm[i].is_empty()).for_each(|i| { let e = acc.entry(nm[i]).or_insert((0.0, 0)); e.0 += ts[i + 1].saturating_sub(ts[i]) as f64 * per; e.1 += 1; });
    }
    fn fwd(&mut self, x: &[u8], y: &[u8], nb: usize, t: usize, salt: u32, mode: u32) -> CommandEncoder {
        let (n, lv, cp) = (nb * t, self.c.lv, self.c.copy.max(1));
        self.q.write_buffer(&self.b["xy"], 0, bytemuck::cast_slice(&x.iter().zip(y).map(|(&a, &b)| a as u32 | (b as u32) << 8).collect::<Vec<u32>>()));
        if let Some((d, lx, pi, pw, cpb)) = self.d19.as_mut() {
            (pi.len() < n * lv * 4).then(|| { pi.resize(n * lv * 4, 0); pw.resize(n * lv * 4, 0.0); cpb.resize(n * cp, -1); });
            d.run(&self.c, lx, x, t, salt, mode == 2, &mut pi[..n * lv * 4], &mut pw[..n * lv * 4], &mut cpb[..n * cp]);
            [("pi", bytemuck::cast_slice::<u32, u8>(&pi[..n * lv * 4])), ("pw", bytemuck::cast_slice(&pw[..n * lv * 4])), ("cpb", bytemuck::cast_slice(&cpb[..n * cp]))].iter().for_each(|(k, v)| self.q.write_buffer(&self.b[k], 0, v));
        }
        let mut enc = self.dev.create_command_encoder(&CommandEncoderDescriptor { label: None });
        (self.d19.is_none()).then(|| ["pi", "pw"].iter().for_each(|k| enc.clear_buffer(&self.b[k], 0, Some((n * lv * 16) as u64))));
        {
            let mut ps = enc.begin_compute_pass(&ComputePassDescriptor { label: None, timestamp_writes: None });
            if self.d19.is_none() {
                self.go(&mut ps, "state", &pc([nb as u32, t as u32, salt, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
                ps.dispatch_workgroups(nb.div_ceil(64) as u32, 1, 1);
                self.flat(&mut ps, "plans", &pc([n as u32, t as u32, salt, 0, 0, 0, 0, 0, 0, 0, 0, 0]), n.div_ceil(256));
            }
            self.flat(&mut ps, "fwd", &pc([n as u32, t as u32, salt, mode, 0, 0, 0, 0, 0, 0, 0, 0]), n);
            self.mark(&mut ps, "");
        }
        enc
    }
}
impl Engine for Gpu {
    fn run(&mut self, x: &[u8], y: &[u8], nb: usize, t: usize, salt: u32, train: Option<(usize, f32)>) -> f64 {
        let (n, lv) = (nb * t, self.c.lv);
        let mut enc = self.fwd(x, y, nb, t, salt, if train.is_some() { 0 } else { 1 });
        let loss = |g: &Gpu| (g.rbuf("pl", n).iter().fold(0f32, |a, &b| a + b) as f64) / n as f64;
        let Some((it, lr)) = train else { self.prof(enc); return loss(self); };
        let (b1, b2) = (0.9f32, 0.99f32);
        let v = pc([n as u32, t as u32, salt, 0, it as u32, lr.to_bits(), (1.0 - b1.powf(it as f32)).to_bits(), (1.0 - b2.powf(it as f32)).to_bits(), (1.0 - b1).to_bits(), (1.0 - b2).to_bits(), 0, 0]);
        enc.clear_buffer(&self.b["ctr"], 0, None);
        {
            let mut ps = enc.begin_compute_pass(&ComputePassDescriptor { label: None, timestamp_writes: None });
            (self.c.tern > 0 && it % 100 == 1).then(|| { self.go(&mut ps, "lvl", &v); ps.dispatch_workgroups(lv as u32, 1, 1); });
            self.flat(&mut ps, "hist", &v, (n * lv * 4).div_ceil(256));
            self.flat(&mut ps, "args", &v, 1);
            self.go(&mut ps, "umap", &v);
            ps.dispatch_workgroups_indirect(&self.b["ag"], 0);
            self.flat(&mut ps, "slots", &v, (n * lv * 4).div_ceil(256));
            self.flat(&mut ps, "dacc", &v, n);
            for (k, o) in [("dense", 12), ("slot", 0)] { self.go(&mut ps, k, &v); ps.dispatch_workgroups_indirect(&self.b["ag"], o); }
            self.go(&mut ps, "part", &v);
            ps.dispatch_workgroups(n.div_ceil(256) as u32, 1, 1);
            self.flat(&mut ps, "fin", &v, 1);
            self.mark(&mut ps, "");
        }
        self.prof(enc);
        loss(self)
    }
    fn logits(&mut self, x: &[u8]) -> Vec<f32> {
        let n = x.len();
        let enc = self.fwd(x, x, 1, n, 0, 2);
        self.prof(enc);
        let d = &self.b["dl"];
        bytemuck::pod_collect_to_vec(&self.read(&|e, _, k| e.copy_buffer_to_buffer(d, ((n - 1) * V * 4) as u64, &self.b["rd"], 0, k), (V * 4) as u64))
    }
    fn refresh(&mut self) {
        let mut enc = self.dev.create_command_encoder(&CommandEncoderDescriptor { label: None });
        { let mut ps = enc.begin_compute_pass(&ComputePassDescriptor { label: None, timestamp_writes: None }); self.go(&mut ps, "lvl", &pc([0; 12])); ps.dispatch_workgroups(self.c.lv as u32, 1, 1); self.mark(&mut ps, ""); }
        self.prof(enc);
    }
    fn get(&mut self, p: Part) -> Vec<f32> {
        let (lv, te, r) = (self.c.lv, self.c.te, self.rows);
        match p {
            Part::Dp => bytemuck::pod_collect_to_vec(&self.rtex(&self.tx.dp, 0, 4096, (2 * DE * V / 4096) as u32)),
            Part::Bias => bytemuck::pod_collect_to_vec(&self.rtex(&self.tx.bi, 0, V as u32, 1)),
            Part::Sid => (0..lv as u32).flat_map(|l| bytemuck::pod_collect_to_vec::<u8, u32>(&self.rtex(&self.tx.sid, l, 4096, r)).into_iter().flat_map(|v| [v & 255, v >> 8 & 255, v >> 16 & 255].map(|x| x as f32))).collect(),
            Part::Sw => (0..lv as u32).flat_map(|l| { let s: Vec<Vec<f32>> = (0..3).map(|s| bytemuck::pod_collect_to_vec(&self.rtex(&self.tx.sw, l * 3 + s, 4096, r))).collect(); (0..te).flat_map(move |e| [s[0][e], s[1][e], s[2][e]]) }).collect(),
            Part::Dm => self.rbuf("dm", 2 * DE * V), Part::Dv => self.rbuf("dv", 2 * DE * V), Part::Sm => self.rbuf("sm", lv * te * KS), Part::Sv => self.rbuf("sv", lv * te * KS), Part::Lq => self.rbuf("lq", 2 * lv),
        }
    }
    fn put(&mut self, p: Part, v: &[f32]) {
        let (lv, te, r) = (self.c.lv, self.c.te, self.rows);
        match p {
            Part::Dp => self.wtex(&self.tx.dp, 0, 4096, (2 * DE * V / 4096) as u32, bytemuck::cast_slice(v)),
            Part::Bias => self.wtex(&self.tx.bi, 0, V as u32, 1, bytemuck::cast_slice(v)),
            Part::Sid => (0..lv).for_each(|l| { let pk: Vec<u32> = v[l * te * 3..(l + 1) * te * 3].chunks(3).map(|s| s[0] as u32 | (s[1] as u32) << 8 | (s[2] as u32) << 16).collect(); self.wtex(&self.tx.sid, l as u32, 4096, r, bytemuck::cast_slice(&pk)) }),
            Part::Sw => (0..lv).for_each(|l| (0..3).for_each(|s| { let w: Vec<f32> = (0..te).map(|e| v[(l * te + e) * 3 + s]).collect(); self.wtex(&self.tx.sw, (l * 3 + s) as u32, 4096, r, bytemuck::cast_slice(&w)) })),
            Part::Dm | Part::Dv | Part::Sm | Part::Sv | Part::Lq => { let k = [(Part::Dm, "dm"), (Part::Dv, "dv"), (Part::Sm, "sm"), (Part::Sv, "sv"), (Part::Lq, "lq")].iter().find(|x| x.0 == p).unwrap().1; bytemuck::cast_slice::<f32, u8>(v).chunks(CH as usize).enumerate().for_each(|(i, d)| { self.q.write_buffer(&self.b[k], i as u64 * CH, d); self.q.submit([]); }); }
        }
        self.q.submit([]);
    }
    fn name(&self) -> String { format!("gpu {}", self.name) }
}
impl Drop for Gpu { fn drop(&mut self) { if let Some(f) = &self.pf { let mut v: Vec<_> = f.acc.borrow().iter().map(|(k, &(t, n))| (*k, t, n)).collect(); v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); v.iter().for_each(|(k, t, n)| eprintln!("prof {k:>7}: {:9.3} ms/call x{n} total {t:.1} ms", t / *n as f64)); } } }
