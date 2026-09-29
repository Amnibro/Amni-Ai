use crate::{bpe::{Bpe, PAT}, f1, hrow, logits, tq, Cfg, Core, Dat, Lz, Mem};
use adam_gf17::{Store, Table, Writer, PX};
use adam_ray::rng::Pcg64;
use burn::tensor::{backend::Backend, Tensor};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{fs, io::{self, Write}, sync::Mutex, time::Instant};
pub fn pack(t: &[f32], din: usize, dout: usize) -> Vec<u16> { let nw = din.div_ceil(10); (0..dout).flat_map(|o| (0..nw).map(move |w| (0..10).rev().fold(0u32, |a, j| a * 3 + if 10 * w + j < din { (t[(10 * w + j) * dout + o] as i32 + 1) as u32 } else { 1 }) as u16)).collect() }
pub fn trits(c: &[u16], din: usize, dout: usize) -> Vec<f32> { let (nw, mut t) = (din.div_ceil(10), vec![0f32; din * dout]); (0..dout).for_each(|o| (0..nw).for_each(|w| { (0..10).fold(c[o * nw + w] as u32, |v, j| { if 10 * w + j < din { t[(10 * w + j) * dout + o] = (v % 3) as f32 - 1.0; } v / 3 }); })); t }
fn tb(n: &str, dt: usize, exact: bool, b: Vec<u8>) -> (Table, Vec<u8>) { (Table { name: n.into(), dt, ord: false, exact, shape: vec![1, b.len() / dt] }, b) }
fn by<T: Copy>(v: &[T], f: impl Fn(T) -> Vec<u8>) -> Vec<u8> { v.iter().flat_map(|&x| f(x)).collect() }
fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
pub type Src<'a> = (usize, &'a dyn Fn(usize, usize) -> Vec<f32>);
pub fn tabs<B: Backend>(c: &Cfg, core: &Core<B>) -> (Vec<(Table, Vec<u8>)>, Vec<(Vec<f32>, usize, usize)>) {
    let tg: Vec<(Vec<f32>, f32, usize, usize)> = core.lins().iter().map(|l| { let w = l.w.val(); let [i, o] = w.dims(); let (t, g) = tq(w); (f1(t), f1(g)[0], i, o) }).collect();
    let cf: Vec<u32> = [c.d, c.l, c.h, c.t, c.ff, c.rb as usize, c.k()].iter().chain(c.ns.iter()).map(|&v| v as u32).collect();
    let mut it = vec![tb("cfg", 4, true, by(&cf, |v: u32| v.to_le_bytes().to_vec())), tb("g", 4, true, by(&tg.iter().map(|x| x.1).collect::<Vec<f32>>(), |v: f32| v.to_le_bytes().to_vec()))];
    it.extend(tg.iter().enumerate().map(|(i, (t, _, a, b))| tb(&format!("w{i}"), 2, true, by(&pack(t, *a, *b), |v: u16| v.to_le_bytes().to_vec()))));
    it.extend(c.bpe().ok().flatten().map(|b| tb("bpe", 4, true, by(&[b.v() as u32].into_iter().chain(b.merges.iter().flatten().copied()).collect::<Vec<u32>>(), |v: u32| v.to_le_bytes().to_vec()))));
    it.extend(c.lpe.then(|| tb("lpe", 4, true, by(&[c.lphi.to_bits() as u32, (c.lphi.to_bits() >> 32) as u32, c.lsq], |v: u32| v.to_le_bytes().to_vec()))));
    it.extend([("emb", f1(core.emb.val())), ("pos", if c.lpe { vec![0f32; c.t * c.d] } else { f1(core.pos.val()) })].into_iter().map(|(n, v)| tb(n, 4, false, by(&v, |x: f32| x.to_le_bytes().to_vec()))));
    (it, tg.into_iter().map(|(t, _, a, b)| (t, a, b)).collect())
}
fn fw(v: &[f32]) -> Vec<u16> { v.iter().flat_map(|x| { let b = x.to_bits(); [b as u16, (b >> 16) as u16] }).collect() }
fn pg(n: usize, p: usize) -> (usize, usize) { (p * PX / 2, ((p + 1) * PX / 2).min(n)) }
pub fn wr(p: &str, it: &[(Table, Vec<u8>)], mem: Option<Src>, tile: usize) -> io::Result<()> {
    let mut w = Writer::create(p, it.iter().map(|x| x.0.clone()).chain(mem.map(|(n, _)| Table { name: "mem".into(), dt: 4, ord: false, exact: false, shape: vec![1, n] })).collect(), false, tile)?;
    it.iter().enumerate().try_for_each(|(t, (_, b))| w.put(t, 0, &adam_gf17::words(b)))?;
    mem.map_or(Ok(()), |(n, f)| (0..(2 * n).div_ceil(PX)).try_for_each(|q| { let (a, b) = pg(n, q); w.put_at(it.len(), 0, q, &fw(&f(a, b))) }))?;
    w.finish()
}
pub fn check(p: &str, it: &[(Table, Vec<u8>)], mem: Option<Src>) -> io::Result<bool> {
    let st = Store::open(p, 64 << 20)?;
    let a = it.iter().map(|(t, b)| st.table(&t.name).map_or(Ok(false), |i| st.unpack(i, 4).map(|r| r.0 == *b))).collect::<io::Result<Vec<bool>>>()?;
    let m = mem.map_or(Ok(st.table("mem").is_none()), |(n, f)| st.table("mem").map_or(Ok(false), |i| (0..st.tabs[i].np()).try_fold(st.tabs[i].n() == n, |ok, q| st.page_words(i, 0, q).map(|w| { let (a, b) = pg(n, q); ok && w == fw(&f(a, b)) }))))?;
    Ok(a.iter().all(|&x| x) && m)
}
pub fn msrc<B: Backend>(m: &Mem<B>) -> (usize, Tensor<B, 1>) { let [r, d] = m.tab.dims(); (r * d, m.tab.val().reshape([r * d])) }
pub fn save<B: Backend>(p: &str, c: &Cfg, core: &Core<B>, mem: Option<&Mem<B>>, tile: usize) -> io::Result<(Vec<(Table, Vec<u8>)>, Vec<(Vec<f32>, usize, usize)>)> {
    let ((it, tr), ms) = (tabs(c, core), mem.map(msrc));
    let f = |a: usize, b: usize| ms.as_ref().map_or_else(Vec::new, |m| f1(m.1.clone().slice([a..b])));
    wr(p, &it, ms.as_ref().map(|m| (m.0, &f as &dyn Fn(usize, usize) -> Vec<f32>)), tile)?;
    Ok((it, tr))
}
pub fn grow(src: &str, rb: u32, out: &str) -> io::Result<String> {
    let st = Store::open(src, 256 << 20)?;
    let (ci, mi) = (st.table("cfg").ok_or_else(|| io::Error::other(format!("{src}: no cfg table")))?, st.table("mem").ok_or_else(|| io::Error::other(format!("{src}: no PTEX memory table")))?);
    let mut cf: Vec<u32> = st.unpack(ci, 4)?.0.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let (d, a, k) = (cf[0] as usize, cf[5], cf[6] as usize);
    (rb >= a && rb < 31).then_some(()).ok_or_else(|| io::Error::other(format!("MEM_LOG2 {rb} must be at least the file's {a}: only growing is exact")))?;
    cf[5] = rb;
    let it = st.tabs.iter().enumerate().filter(|&(i, _)| i != mi).map(|(i, t)| Ok((t.clone(), if i == ci { by(&cf, |v: u32| v.to_le_bytes().to_vec()) } else { st.unpack(i, 4)?.0 }))).collect::<io::Result<Vec<_>>>()?;
    let (old, s, t0) = (f32s(&st.unpack(mi, 4)?.0), rb - a, Instant::now());
    let f = |x: usize, y: usize| (x..y).map(|e| { let r = e / d; old[((r >> rb) << a | (r & ((1 << rb) - 1)) >> s) * d + e % d] }).collect::<Vec<f32>>();
    wr(out, &it, Some(((k << rb) * d, &f)), st.tile)?;
    Ok(json!({"grown": out, "from": src, "mem_log2": [a, rb], "rows_per_order": 1usize << rb, "k": k, "d": d, "mem_f32": (k << rb) * d, "bytes": fs::metadata(out)?.len(), "src_bytes": fs::metadata(src)?.len(), "seconds": t0.elapsed().as_secs_f64()}).to_string())
}
pub struct Lm { pub v: usize, pub bpe: Option<Bpe>, pub d: usize, pub l: usize, pub h: usize, pub t: usize, pub ff: usize, pub rb: u32, pub k: usize, pub ns: Vec<usize>, pub w: Vec<(Vec<u16>, usize, usize)>, pub g: Vec<f32>, pub emb: Vec<f32>, pub pos: Vec<f32>, pub st: Mutex<Store>, pub mt: Option<usize>, pub lz: Option<Lz>, pub gf: bool }
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
        let lz = st.table("lpe").map(|_| get("lpe").map(|b| { let u: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(); Lz::new(d / cf[2], cf[3], f64::from_bits(u[0] as u64 | (u[1] as u64) << 32), u[2]) })).transpose()?;
        Ok(Lm { v, bpe, d, l, h: cf[2], t: cf[3], ff, rb: cf[5] as u32, k: cf[6], ns: cf[7..].to_vec(), w, g, emb, pos, st: Mutex::new(st), mt, lz, gf: std::env::var("LPE_LUT").map(|v| v != "ref").unwrap_or(true) })
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
    pub fn qkv(&self, b: usize, x: &[f32], p: usize) -> Vec<f32> { let (mut r, d) = (self.lin(4 * b, x), self.d); if let Some(z) = &self.lz { let (q, k) = r.split_at_mut(d); z.app(q, &mut k[..d], p, self.gf); } r }
    pub fn rows(&self, x: &[u32], p0: usize) -> io::Result<Vec<f32>> {
        let (n, d2) = (x.len(), 2 * self.d);
        let Some(mt) = self.mt else { return Ok(vec![0f32; (n - p0) * self.d]) };
        let q: Vec<(usize, usize, usize)> = (p0..n).flat_map(|p| (0..self.k).flat_map(move |k| { let r = hrow(x, p, k, self.ns[k], self.rb, self.v > 256); (0..d2).map(move |j| (mt, 0, r * d2 + j)) })).collect();
        let mut st = self.st.lock().map_err(|_| io::Error::other("store lock"))?;
        let wv = q.chunks((32 * self.k * d2).max(1)).map(|c| st.get(c, 4)).collect::<io::Result<Vec<Vec<u16>>>>()?.concat();
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
            let qkv: Vec<Vec<f32>> = xs.par_iter().enumerate().map(|(p, v)| self.qkv(b, v, p)).collect();
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
        for b in 0..self.l { let q = self.qkv(b, &v, p); s.k[b].extend_from_slice(&q[d..2 * d]); s.v[b].extend_from_slice(&q[2 * d..]); v = self.blk(b, &v, &q[..d], &s.k[b], &s.v[b], p); }
        Ok(self.lin(self.l * 4, &v))
    }
}
#[derive(Clone)]
pub struct Kv { pub x: Vec<u32>, pub k: Vec<Vec<f32>>, pub v: Vec<Vec<f32>> }
pub fn nll(lg: &[f32], y: &[u32]) -> Vec<f64> { lg.chunks_exact(lg.len() / y.len().max(1)).zip(y).map(|(r, &t)| { let mx = r.iter().fold(f32::MIN, |a, &v| a.max(v)) as f64; mx + r.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln() - r[t as usize] as f64 }).collect() }
pub fn am(r: &[f32]) -> usize { r.iter().enumerate().fold(0, |b, (i, &v)| if v > r[b] { i } else { b }) }
#[derive(Clone, Debug, PartialEq)]
pub struct Dec { pub nr: usize, pub rp: f32, pub k: usize, pub tp: f32, pub seed: u64, pub rs: bool }
impl Dec {
    pub fn new(chat: bool) -> Dec { Dec { nr: if chat { 3 } else { 0 }, rp: if chat { 1.15 } else { 1.0 }, k: 0, tp: 0.0, seed: 0, rs: chat } }
    pub fn with(&self, q: &Value) -> io::Result<Dec> {
        let (bad, g) = (|k: &str| io::Error::other(format!("bad gen field {k}")), |k: &str| q.get(k).filter(|v| !v.is_null()));
        let u = |k: &str, d: u64| g(k).map_or(Ok(d), |v| v.as_u64().ok_or_else(|| bad(k)));
        let f = |k: &str, d: f32| g(k).map_or(Ok(d), |v| v.as_f64().filter(|x| x.is_finite() && *x >= 0.0).map(|x| x as f32).ok_or_else(|| bad(k)));
        let r = Dec { nr: u("no_repeat_ngram", self.nr as u64)? as usize, rp: f("rep_penalty", self.rp)?, k: u("top_k", self.k as u64)? as usize, tp: f("temperature", self.tp)?, seed: u("seed", self.seed)?, rs: g("stop_repeat").map_or(Ok(self.rs), |v| v.as_bool().ok_or_else(|| bad("stop_repeat")))? };
        (r.rp > 0.0).then_some(r).ok_or_else(|| bad("rep_penalty"))
    }
    pub fn json(&self) -> Value {
        let r = |x: f32| x.to_string().parse::<f64>().unwrap_or(x as f64);
        json!({"no_repeat_ngram": self.nr, "rep_penalty": r(self.rp), "top_k": self.k, "temperature": r(self.tp), "seed": self.seed, "stop_repeat": self.rs})
    }
    pub fn pick(&self, lg: &[f32], o: &[u32], g: &mut Pcg64) -> usize {
        let (n, mut l, mut u) = (self.nr, lg.to_vec(), o.to_vec());
        u.sort_unstable();
        u.dedup();
        (self.rp != 1.0).then(|| u.iter().for_each(|&t| { let x = &mut l[t as usize]; *x = if *x > 0.0 { *x / self.rp } else { *x * self.rp }; }));
        (n > 0 && o.len() + 1 >= n).then(|| { let p = &o[o.len() + 1 - n..]; o.windows(n).filter(|w| w[..n - 1] == *p).for_each(|w| l[w[n - 1] as usize] = f32::NEG_INFINITY); });
        let l = if l.iter().all(|v| *v == f32::NEG_INFINITY) { lg.to_vec() } else { l };
        let mut ix: Vec<usize> = if self.tp > 0.0 { (0..l.len()).filter(|&i| l[i].is_finite()).collect() } else { Vec::new() };
        ix.sort_by(|&a, &b| l[b].total_cmp(&l[a]).then(a.cmp(&b)));
        ix.truncate(if self.k > 0 { self.k } else { ix.len() });
        if ix.is_empty() { am(&l) } else { ix[g.choice(&ix.iter().map(|&i| (((l[i] - l[ix[0]]) / self.tp) as f64).exp()).collect::<Vec<f64>>())] }
    }
}
pub fn rstop(g: &[u8]) -> Option<usize> {
    let sep = |c: &u8| b".!?;,:\n".contains(c);
    let cl: Vec<(usize, Vec<Vec<u8>>)> = g.split_inclusive(sep).scan(0usize, |s, c| { *s += c.len(); Some((*s - c.len(), c)) }).filter(|(_, c)| c.last().is_some_and(sep)).map(|(a, c)| (a, c.split(|x| !x.is_ascii_alphanumeric() && *x != b'\'').filter(|w| !w.is_empty()).map(|w| w.to_ascii_lowercase()).collect())).collect();
    (1..cl.len()).find(|&j| cl[j].1.len() >= 3 && cl[..j].iter().any(|c| c.1 == cl[j].1)).map(|j| g[..cl[j].0].iter().rposition(|c| !b" \t\r\n,;:".contains(c)).map_or(0, |i| i + 1))
}
pub fn hold_windows(hold: &Dat, n: usize, t: usize) -> (Vec<u32>, Vec<u32>) { let mut g = Pcg64::new(Some(1234)); let (x, y) = hold.hold(&mut g, 128, t); (x[..n * t].to_vec(), y[..n * t].to_vec()) }
pub fn bpb(v: &[f64], y: &[u32], lens: &[u32]) -> f64 { v.iter().sum::<f64>() / y.iter().map(|&t| lens[t as usize] as f64).sum::<f64>() / std::f64::consts::LN_2 }
pub fn cpu_eval(lm: &Lm, x: &[u32], y: &[u32], lens: &[u32]) -> io::Result<(Vec<f32>, f64, f64)> { let t0 = Instant::now(); let lg = x.chunks(lm.t).map(|w| lm.fwd(w)).collect::<io::Result<Vec<_>>>()?.concat(); let s = t0.elapsed().as_secs_f64(); Ok((lg.clone(), bpb(&nll(&lg, y), y, lens), y.iter().map(|&t| lens[t as usize] as f64).sum::<f64>() / s)) }
pub fn finish<B: crate::fa::Fa>(c: &Cfg, core: &Core<B>, mem: Option<&Mem<B>>, hold: &Dat, lens: &[u32], dev: &B::Device, log: &mut fs::File, p: &str) -> io::Result<()> {
    let (it, tr) = save(p, c, core, mem, 128)?;
    let ms = mem.map(msrc);
    let f = |a: usize, b: usize| ms.as_ref().map_or_else(Vec::new, |m| f1(m.1.clone().slice([a..b])));
    let rt = check(p, &it, ms.as_ref().map(|m| (m.0, &f as &dyn Fn(usize, usize) -> Vec<f32>)))?;
    let st = Store::open(p, 64 << 20)?;
    let tt = tr.iter().enumerate().all(|(i, (t, a, b))| st.table(&format!("w{i}")).map(|j| st.unpack(j, 4).map(|r| trits(&r.0.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<u16>>(), *a, *b) == *t).unwrap_or(false)).unwrap_or(false));
    let (nt, nz) = tr.iter().fold((0usize, 0usize), |(n, z), (t, _, _)| (n + t.len(), z + t.iter().filter(|v| **v == 0.0).count()));
    let (raw, sz, tb_) = (it.iter().map(|x| x.1.len()).sum::<usize>() + 4 * ms.as_ref().map_or(0, |m| m.0), fs::metadata(p)?.len(), it.iter().filter(|x| x.0.name.starts_with('w')).map(|x| x.1.len()).sum::<usize>());
    drop(st);
    let lm = Lm::open(p, 256 << 20)?;
    let (x, y) = hold_windows(hold, c.par.max(1), c.t);
    let gl = f1(logits(c, core, mem, &x, c.par.max(1), dev));
    let (cl, cb, bps) = cpu_eval(&lm, &x, &y, lens)?;
    let (mx, sm, agree) = gl.chunks_exact(c.v).zip(cl.chunks_exact(c.v)).fold((0f32, 0f64, 0usize), |(m, s, a), (g, q)| (g.iter().zip(q).fold(m, |m, (u, v)| m.max((u - v).abs())), s + g.iter().zip(q).map(|(u, v)| (u - v).abs() as f64).sum::<f64>(), a + (am(g) == am(q)) as usize));
    let gb = bpb(&nll(&gl, &y), &y, lens);
    let r = format!("{{\"export\": \"{p}\", \"tables\": {}, \"roundtrip_bytes_exact\": {rt}, \"roundtrip_trits_exact\": {tt}, \"trits\": {nt}, \"zero_frac\": {:.4}, \"payload_bytes\": {raw}, \"trit_code_bytes\": {tb_}, \"bits_per_trit_stored\": {:.4}, \"gf17_file_bytes\": {sz}, \"parity_windows\": {}, \"parity_positions\": {}, \"max_abs_dlogit\": {mx:.3e}, \"mean_abs_dlogit\": {:.3e}, \"argmax_agree\": {:.5}, \"gpu_bpb\": {gb:.5}, \"cpu_lut_bpb\": {cb:.5}, \"cpu_lut_bytes_per_s\": {bps:.1}}}", it.len() + ms.is_some() as usize, nz as f64 / nt as f64, tb_ as f64 * 8.0 / nt as f64, c.par, gl.len() / c.v, sm / gl.len() as f64, agree as f64 / (gl.len() / c.v) as f64);
    println!("{r}");
    writeln!(log, "{r}")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_repeat_ngram_bans_completions() {
        let (d, mut g) = (Dec { nr: 3, rp: 1.0, ..Dec::new(false) }, Pcg64::new(Some(0)));
        let mut lg = vec![0f32; 8];
        (lg[5], lg[6]) = (3.0, 2.0);
        assert_eq!(d.pick(&lg, &[1, 2, 5, 1, 2], &mut g), 6);
        assert_eq!(d.pick(&lg, &[1, 2, 5, 1, 3], &mut g), 5);
        assert_eq!(Dec { nr: 1, ..d.clone() }.pick(&lg, &[5, 6], &mut g), 0);
        assert_eq!(Dec { nr: 2, ..d.clone() }.pick(&[1.0, 0.0], &[0, 0], &mut g), 1);
        assert_eq!(Dec { nr: 1, ..d.clone() }.pick(&[1.0, 0.0], &[0, 1], &mut g), 0, "all banned falls back to the raw argmax");
        assert_eq!(Dec::new(false).pick(&lg, &[1, 2, 5, 1, 2], &mut g), 5);
    }
    #[test]
    fn rep_penalty_scales_seen_logits() {
        let (d, mut g) = (Dec { rp: 1.15, ..Dec::new(false) }, Pcg64::new(Some(0)));
        let lg = [2.0f32, 1.9, -1.0, -1.1, -5.0];
        assert_eq!(d.pick(&lg, &[], &mut g), 0);
        assert_eq!(d.pick(&lg, &[0, 0, 0], &mut g), 1);
        assert_eq!(d.pick(&[-1.0, -1.1, -9.0], &[0], &mut g), 1);
        assert_eq!(Dec { rp: 1.0, ..d.clone() }.pick(&lg, &[0], &mut g), 0);
        assert_eq!(Dec { rp: 1.04, ..d.clone() }.pick(&lg, &[0, 0, 0, 0, 0], &mut g), 0, "the penalty applies once per distinct token, not per occurrence");
    }
    #[test]
    fn sampling_is_seeded_and_topk_bounded() {
        let lg: Vec<f32> = (0..50).map(|i| (i % 7) as f32 * 0.3).collect();
        let run = |d: &Dec| { let mut g = Pcg64::new(Some(d.seed)); (0..200).map(|_| d.pick(&lg, &[], &mut g)).collect::<Vec<usize>>() };
        let d = Dec { tp: 1.0, k: 3, seed: 4, ..Dec::new(false) };
        let a = run(&d);
        assert!(a == run(&d) && a != run(&Dec { seed: 5, ..d.clone() }) && a.iter().all(|i| [6, 13, 20].contains(i)) && [6, 13, 20].iter().all(|i| a.contains(i)), "{a:?}");
        assert!(run(&Dec { seed: 9, ..Dec::new(false) }).iter().all(|&i| i == 6 && am(&lg) == 6));
        assert!(run(&Dec { tp: 1.0, seed: 4, ..Dec::new(false) }).iter().collect::<std::collections::HashSet<_>>().len() > 3);
    }
    #[test]
    fn clause_repeat_stop() {
        assert_eq!(rstop(b" You should cook for dinner tonight. If you have a tonight, you should choose a dinner tonight. If you have a to"), None);
        let t = b" You should cook for dinner tonight. If you have a tonight, you should choose a dinner tonight. If you have a tonight,";
        assert_eq!(&t[..rstop(t).unwrap()], b" You should cook for dinner tonight. If you have a tonight, you should choose a dinner tonight.");
        let u = b" It is used as a liquid, it is used for cooling.\nIt is used as a LIQUID!";
        assert_eq!(&u[..rstop(u).unwrap()], b" It is used as a liquid, it is used for cooling.");
        assert_eq!(rstop(b" Oh no, oh no, that is bad. Yes, yes, yes."), None);
        assert_eq!(rstop(b""), None);
    }
    #[test]
    fn dec_fields_parse() {
        let c = Dec::new(true);
        assert!(c.nr == 3 && (c.rp - 1.15).abs() < 1e-6 && c.rs && c.tp == 0.0 && Dec::new(false) == Dec { nr: 0, rp: 1.0, k: 0, tp: 0.0, seed: 0, rs: false });
        assert_eq!(c.with(&json!({"no_repeat_ngram": 4, "rep_penalty": 1.3, "top_k": 7, "temperature": 0.7, "seed": 11, "stop_repeat": false, "other": 1})).unwrap(), Dec { nr: 4, rp: 1.3, k: 7, tp: 0.7, seed: 11, rs: false });
        assert!(c.with(&json!({"rep_penalty": null})).unwrap() == c && c.with(&c.json()).unwrap() == c && c.with(&Value::Null).unwrap() == c);
        ["{\"rep_penalty\": 0}", "{\"top_k\": -1}", "{\"temperature\": \"hot\"}", "{\"stop_repeat\": 1}", "{\"no_repeat_ngram\": 1.5}"].iter().for_each(|q| assert!(c.with(&serde_json::from_str(q).unwrap()).is_err(), "{q}"));
    }
}
