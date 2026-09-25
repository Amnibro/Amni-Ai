use crate::{bpe::{Bpe, PAT}, f1, hrow, logits, tq, Cfg, Core, Dat, Mem};
use adam_gf17::{Store, Table};
use adam_ray::rng::Pcg64;
use burn::tensor::backend::Backend;
use rayon::prelude::*;
use std::{fs, io::{self, Write}, sync::Mutex, time::Instant};
pub fn pack(t: &[f32], din: usize, dout: usize) -> Vec<u16> { let nw = din.div_ceil(10); (0..dout).flat_map(|o| (0..nw).map(move |w| (0..10).rev().fold(0u32, |a, j| a * 3 + if 10 * w + j < din { (t[(10 * w + j) * dout + o] as i32 + 1) as u32 } else { 1 }) as u16)).collect() }
pub fn trits(c: &[u16], din: usize, dout: usize) -> Vec<f32> { let (nw, mut t) = (din.div_ceil(10), vec![0f32; din * dout]); (0..dout).for_each(|o| (0..nw).for_each(|w| { (0..10).fold(c[o * nw + w] as u32, |v, j| { if 10 * w + j < din { t[(10 * w + j) * dout + o] = (v % 3) as f32 - 1.0; } v / 3 }); })); t }
fn tb(n: &str, dt: usize, exact: bool, b: Vec<u8>) -> (Table, Vec<u8>) { (Table { name: n.into(), dt, ord: false, exact, shape: vec![1, b.len() / dt] }, b) }
fn by<T: Copy>(v: &[T], f: impl Fn(T) -> Vec<u8>) -> Vec<u8> { v.iter().flat_map(|&x| f(x)).collect() }
fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
pub fn tabs<B: Backend>(c: &Cfg, core: &Core<B>, mem: Option<&Mem<B>>) -> (Vec<(Table, Vec<u8>)>, Vec<(Vec<f32>, usize, usize)>) {
    let tg: Vec<(Vec<f32>, f32, usize, usize)> = core.lins().iter().map(|l| { let w = l.w.val(); let [i, o] = w.dims(); let (t, g) = tq(w); (f1(t), f1(g)[0], i, o) }).collect();
    let cf: Vec<u32> = [c.d, c.l, c.h, c.t, c.ff, c.rb as usize, c.k()].iter().chain(c.ns.iter()).map(|&v| v as u32).collect();
    let mut it = vec![tb("cfg", 4, true, by(&cf, |v: u32| v.to_le_bytes().to_vec())), tb("g", 4, true, by(&tg.iter().map(|x| x.1).collect::<Vec<f32>>(), |v: f32| v.to_le_bytes().to_vec()))];
    it.extend(tg.iter().enumerate().map(|(i, (t, _, a, b))| tb(&format!("w{i}"), 2, true, by(&pack(t, *a, *b), |v: u16| v.to_le_bytes().to_vec()))));
    it.extend(c.bpe().ok().flatten().map(|b| tb("bpe", 4, true, by(&[b.v() as u32].into_iter().chain(b.merges.iter().flatten().copied()).collect::<Vec<u32>>(), |v: u32| v.to_le_bytes().to_vec()))));
    it.extend([("emb", f1(core.emb.val())), ("pos", f1(core.pos.val()))].into_iter().chain(mem.map(|m| ("mem", f1(m.tab.val())))).map(|(n, v)| tb(n, 4, false, by(&v, |x: f32| x.to_le_bytes().to_vec()))));
    (it, tg.into_iter().map(|(t, _, a, b)| (t, a, b)).collect())
}
pub struct Lm { pub v: usize, pub bpe: Option<Bpe>, pub d: usize, pub l: usize, pub h: usize, pub t: usize, pub ff: usize, pub rb: u32, pub k: usize, pub ns: Vec<usize>, pub w: Vec<(Vec<u16>, usize, usize)>, pub g: Vec<f32>, pub emb: Vec<f32>, pub pos: Vec<f32>, pub st: Mutex<Store>, pub mt: Option<usize> }
impl Lm {
    pub fn open(p: &str, budget: usize) -> io::Result<Lm> {
        let st = Store::open(p, budget)?;
        let get = |n: &str| st.table(n).ok_or_else(|| io::Error::other(format!("missing table {n}"))).and_then(|i| st.unpack(i, 4).map(|r| r.0));
        let cf: Vec<usize> = get("cfg")?.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize).collect();
        let (d, l, ff) = (cf[0], cf[1], cf[4]);
        let bpe = st.table("bpe").map(|_| get("bpe").and_then(|b| { let u: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(); Bpe::new(PAT, u[1..].chunks_exact(2).map(|m| [m[0], m[1]]).collect()) })).transpose()?;
        let v = bpe.as_ref().map(|b| b.v()).unwrap_or(256);
        let dims: Vec<(usize, usize)> = (0..l).flat_map(|_| [(d, 3 * d), (d, d), (d, ff), (ff, d)]).chain([(d, v)]).collect();
        let w = dims.iter().enumerate().map(|(i, &(a, b))| get(&format!("w{i}")).map(|v| (v.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(), a, b))).collect::<io::Result<Vec<_>>>()?;
        let (g, emb, pos, mt) = (f32s(&get("g")?), f32s(&get("emb")?), f32s(&get("pos")?), st.table("mem"));
        Ok(Lm { v, bpe, d, l, h: cf[2], t: cf[3], ff, rb: cf[5] as u32, k: cf[6], ns: cf[7..].to_vec(), w, g, emb, pos, st: Mutex::new(st), mt })
    }
    pub fn lin(&self, i: usize, x: &[f32]) -> Vec<f32> {
        let ((cw, din, dout), g) = (&self.w[i], self.g[i]);
        let nw = din.div_ceil(10);
        let r = (x.iter().map(|v| v * v).sum::<f32>() / *din as f32 + 1e-6).sqrt();
        let xn: Vec<f32> = x.iter().map(|v| v / r).collect();
        let m = xn.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-5);
        let s = (1.0 / m) * 127.0;
        let mut xi = vec![0i32; nw * 10];
        xn.iter().zip(xi.iter_mut()).for_each(|(v, q)| *q = (v * s).round_ties_even().clamp(-128.0, 127.0) as i32);
        let lt: Vec<i32> = xi.chunks(5).flat_map(|q| q.iter().fold(vec![0i32], |l, &v| [-v, 0, v].iter().flat_map(|&a| l.iter().map(move |&b| b + a)).collect())).collect();
        let sc = m * (g * (1.0 / 127.0));
        (0..*dout).map(|o| cw[o * nw..(o + 1) * nw].iter().enumerate().fold(0i32, |a, (w, &c)| a + lt[2 * w * 243 + (c % 243) as usize] + lt[(2 * w + 1) * 243 + (c / 243) as usize]) as f32 * sc).collect()
    }
    pub fn rows(&self, x: &[u32], p0: usize) -> io::Result<Vec<f32>> {
        let (n, d2) = (x.len(), 2 * self.d);
        let Some(mt) = self.mt else { return Ok(vec![0f32; (n - p0) * self.d]) };
        let q: Vec<(usize, usize, usize)> = (p0..n).flat_map(|p| (0..self.k).flat_map(move |k| { let r = hrow(x, p, k, self.ns[k], self.rb, self.v > 256); (0..d2).map(move |j| (mt, 0, r * d2 + j)) })).collect();
        let wv = self.st.lock().map_err(|_| io::Error::other("store lock"))?.get(&q, 4)?;
        let rv: Vec<f32> = wv.chunks_exact(2).map(|c| f32::from_bits(c[0] as u32 | (c[1] as u32) << 16)).collect();
        Ok((0..n - p0).flat_map(|p| (0..self.d).map(|j| (1..self.k).fold(rv[p * self.k * self.d + j], |a, k| a + rv[(p * self.k + k) * self.d + j])).collect::<Vec<f32>>()).collect())
    }
    pub fn emb_at(&self, x: &[u32], p0: usize) -> io::Result<Vec<Vec<f32>>> { let (d, mr) = (self.d, self.rows(x, p0)?); Ok((p0..x.len()).map(|p| (0..d).map(|j| self.emb[x[p] as usize * d + j] + self.pos[p * d + j] + mr[(p - p0) * d + j]).collect()).collect()) }
    pub fn blk(&self, b: usize, x: &[f32], q: &[f32], kc: &[f32], vc: &[f32], p: usize) -> Vec<f32> {
        let (d, h) = (self.d, self.h);
        let dh = d / h;
        let a: Vec<f32> = (0..h).flat_map(|hh| { let q = &q[hh * dh..(hh + 1) * dh]; let sc: Vec<f32> = (0..=p).map(|j| q.iter().zip(&kc[j * d + hh * dh..j * d + (hh + 1) * dh]).map(|(a, b)| a * b).sum::<f32>() * (1.0 / (dh as f32).sqrt())).collect(); let mx = sc.iter().fold(f32::MIN, |a, &v| a.max(v)); let e: Vec<f32> = sc.iter().map(|v| (v - mx).exp()).collect(); let z: f32 = e.iter().sum(); (0..dh).map(|c| (0..=p).map(|j| e[j] * vc[j * d + hh * dh + c]).sum::<f32>() / z).collect::<Vec<f32>>() }).collect();
        let v: Vec<f32> = x.iter().zip(self.lin(4 * b + 1, &a)).map(|(x, y)| x + y).collect();
        let f: Vec<f32> = self.lin(4 * b + 2, &v).iter().map(|z| z.max(0.0) * z.max(0.0)).collect();
        v.iter().zip(self.lin(4 * b + 3, &f)).map(|(x, y)| x + y).collect()
    }
    pub fn kv(&self) -> Kv { Kv { x: Vec::new(), k: vec![Vec::new(); self.l], v: vec![Vec::new(); self.l] } }
    pub fn fwd_kv(&self, x: &[u32]) -> io::Result<(Vec<f32>, Kv)> {
        let (d, mut s) = (self.d, self.kv());
        s.x = x.to_vec();
        let mut xs = self.emb_at(x, 0)?;
        for b in 0..self.l {
            let qkv: Vec<Vec<f32>> = xs.par_iter().map(|v| self.lin(4 * b, v)).collect();
            (s.k[b], s.v[b]) = (qkv.iter().flat_map(|q| q[d..2 * d].to_vec()).collect(), qkv.iter().flat_map(|q| q[2 * d..].to_vec()).collect());
            xs = (0..x.len()).into_par_iter().map(|p| self.blk(b, &xs[p], &qkv[p][..d], &s.k[b], &s.v[b], p)).collect();
        }
        Ok((xs.par_iter().flat_map(|v| self.lin(self.l * 4, v)).collect(), s))
    }
    pub fn fwd(&self, x: &[u32]) -> io::Result<Vec<f32>> { self.fwd_kv(x).map(|r| r.0) }
    pub fn step(&self, s: &mut Kv, tok: u32) -> io::Result<Vec<f32>> {
        s.x.push(tok);
        let (d, p) = (self.d, s.x.len() - 1);
        let mut v = self.emb_at(&s.x, p)?.remove(0);
        for b in 0..self.l { let q = self.lin(4 * b, &v); s.k[b].extend_from_slice(&q[d..2 * d]); s.v[b].extend_from_slice(&q[2 * d..]); v = self.blk(b, &v, &q[..d], &s.k[b], &s.v[b], p); }
        Ok(self.lin(self.l * 4, &v))
    }
}
pub struct Kv { pub x: Vec<u32>, pub k: Vec<Vec<f32>>, pub v: Vec<Vec<f32>> }
pub fn nll(lg: &[f32], y: &[u32]) -> Vec<f64> { lg.chunks_exact(lg.len() / y.len().max(1)).zip(y).map(|(r, &t)| { let mx = r.iter().fold(f32::MIN, |a, &v| a.max(v)) as f64; mx + r.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln() - r[t as usize] as f64 }).collect() }
pub fn am(r: &[f32]) -> usize { r.iter().enumerate().fold(0, |b, (i, &v)| if v > r[b] { i } else { b }) }
pub fn hold_windows(hold: &Dat, n: usize, t: usize) -> (Vec<u32>, Vec<u32>) { let mut g = Pcg64::new(Some(1234)); let (x, y) = hold.hold(&mut g, 128, t); (x[..n * t].to_vec(), y[..n * t].to_vec()) }
pub fn bpb(v: &[f64], y: &[u32], lens: &[u32]) -> f64 { v.iter().sum::<f64>() / y.iter().map(|&t| lens[t as usize] as f64).sum::<f64>() / std::f64::consts::LN_2 }
pub fn cpu_eval(lm: &Lm, x: &[u32], y: &[u32], lens: &[u32]) -> io::Result<(Vec<f32>, f64, f64)> { let t0 = Instant::now(); let lg = x.chunks(lm.t).map(|w| lm.fwd(w)).collect::<io::Result<Vec<_>>>()?.concat(); let s = t0.elapsed().as_secs_f64(); Ok((lg.clone(), bpb(&nll(&lg, y), y, lens), y.iter().map(|&t| lens[t as usize] as f64).sum::<f64>() / s)) }
pub fn finish<B: Backend>(c: &Cfg, core: &Core<B>, mem: Option<&Mem<B>>, hold: &Dat, lens: &[u32], dev: &B::Device, log: &mut fs::File) -> io::Result<()> {
    let p = format!("{}/model.gf17", c.out);
    let (it, tr) = tabs(c, core, mem);
    adam_gf17::write(&p, &it.iter().map(|(t, b)| (t.clone(), b.as_slice())).collect::<Vec<_>>(), 128)?;
    let st = Store::open(&p, 64 << 20)?;
    let rt = it.iter().map(|(t, b)| st.table(&t.name).map(|i| st.unpack(i, 4).map(|r| r.0 == *b)).unwrap_or(Ok(false))).collect::<io::Result<Vec<bool>>>()?;
    let tt = tr.iter().enumerate().all(|(i, (t, a, b))| st.table(&format!("w{i}")).map(|j| st.unpack(j, 4).map(|r| trits(&r.0.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<u16>>(), *a, *b) == *t).unwrap_or(false)).unwrap_or(false));
    let (nt, nz) = tr.iter().fold((0usize, 0usize), |(n, z), (t, _, _)| (n + t.len(), z + t.iter().filter(|v| **v == 0.0).count()));
    let (raw, sz, tb_) = (it.iter().map(|x| x.1.len()).sum::<usize>(), fs::metadata(&p)?.len(), it.iter().filter(|x| x.0.name.starts_with('w')).map(|x| x.1.len()).sum::<usize>());
    drop(st);
    let lm = Lm::open(&p, 256 << 20)?;
    let (x, y) = hold_windows(hold, c.par, c.t);
    let gl = f1(logits(c, core, mem, &x, c.par, dev));
    let (cl, cb, bps) = cpu_eval(&lm, &x, &y, lens)?;
    let (mx, sm, agree) = gl.chunks_exact(c.v).zip(cl.chunks_exact(c.v)).fold((0f32, 0f64, 0usize), |(m, s, a), (g, q)| (g.iter().zip(q).fold(m, |m, (u, v)| m.max((u - v).abs())), s + g.iter().zip(q).map(|(u, v)| (u - v).abs() as f64).sum::<f64>(), a + (am(g) == am(q)) as usize));
    let gb = bpb(&nll(&gl, &y), &y, lens);
    let r = format!("{{\"export\": \"{p}\", \"tables\": {}, \"roundtrip_bytes_exact\": {}, \"roundtrip_trits_exact\": {tt}, \"trits\": {nt}, \"zero_frac\": {:.4}, \"payload_bytes\": {raw}, \"trit_code_bytes\": {tb_}, \"bits_per_trit_stored\": {:.4}, \"gf17_file_bytes\": {sz}, \"parity_windows\": {}, \"parity_positions\": {}, \"max_abs_dlogit\": {mx:.3e}, \"mean_abs_dlogit\": {:.3e}, \"argmax_agree\": {:.5}, \"gpu_bpb\": {gb:.5}, \"cpu_lut_bpb\": {cb:.5}, \"cpu_lut_bytes_per_s\": {bps:.1}}}", it.len(), rt.iter().all(|&b| b), nz as f64 / nt as f64, tb_ as f64 * 8.0 / nt as f64, c.par, gl.len() / c.v, sm / gl.len() as f64, agree as f64 / (gl.len() / c.v) as f64);
    println!("{r}");
    writeln!(log, "{r}")
}
