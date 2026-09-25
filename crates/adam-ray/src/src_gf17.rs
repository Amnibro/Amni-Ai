use crate::pack::Tables;
#[cfg(feature = "gf17")]
use crate::pack::{copies, mkcell, BASE, V};
#[cfg(feature = "gf17")]
use adam_gf17::Store;
#[cfg(feature = "gf17")]
use half::f16;
#[cfg(feature = "gf17")]
use rayon::prelude::*;
use std::{io, sync::atomic::{AtomicU64, Ordering::Relaxed}};
#[cfg(feature = "gf17")]
use std::sync::{atomic::AtomicBool, mpsc, Arc, Mutex, OnceLock};
pub static STAT: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
pub type Loaded = (usize, usize, usize, Vec<u16>, Vec<[u32; 2]>, Vec<f32>, Vec<u16>, Src);
pub enum Src { Npy, #[cfg(feature = "gf17")] Gf17(Box<Pages>) }
#[cfg(feature = "gf17")]
type Job = Box<dyn FnOnce(&mut Store) + Send>;
#[cfg(feature = "gf17")]
type Lv = Arc<Vec<OnceLock<Vec<[u32; 2]>>>>;
#[cfg(feature = "gf17")]
pub struct Pages { tx: mpsc::Sender<Job>, ix: [usize; 2], t: usize, pub tiers: usize, cc: Vec<Mutex<Vec<[u32; 4]>>>, b: u32, lv: Lv, sm: Arc<Vec<AtomicBool>>, pub npin: usize, pub budget: usize }
#[cfg(feature = "gf17")]
const SH: u32 = 8;
pub fn stats() -> [u64; 10] { std::array::from_fn(|i| STAT[i].load(Relaxed)) }
fn bad(m: String) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, m) }
#[cfg(feature = "gf17")]
fn env(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
#[cfg(feature = "gf17")]
fn trim() { #[cfg(target_env = "gnu")] unsafe { libc::malloc_trim(0) }; }
#[cfg(feature = "gf17")]
fn cells(st: &Store, ix: [usize; 2], t: usize, n: usize, z: usize) -> io::Result<Vec<[u32; 2]>> {
    let (mut s, mut w) = (vec![0u8; 3 * t], vec![0u16; 3 * t]);
    st.level(ix[0], z, 4, &mut s)?;
    st.level(ix[1], z, n, bytemuck::cast_slice_mut(&mut w))?;
    Ok((0..t).into_par_iter().with_min_len(1 << 12).map(|c| mkcell(&s[3 * c..3 * c + 3], &w[3 * c..3 * c + 3])).collect())
}
#[cfg(feature = "gf17")]
impl Pages {
    pub fn new(mut st: Store, budget: usize, tiers: usize) -> io::Result<Pages> {
        let ix = st.table("sid").zip(st.table("sw")).map(|(a, b)| [a, b]).ok_or_else(|| bad(format!("{}: no sid/sw tables", st.path)))?;
        let (l, t, n, (tx, rx), res) = (st.tabs[ix[0]].shape[0], st.tabs[ix[0]].shape[1], rayon::current_num_threads(), mpsc::channel::<Job>(), (budget / 8).max(2 << 20));
        let b = (usize::BITS - 1 - (res / 2 / 16).max(2 << SH).leading_zeros()).min(SH + 24);
        (st.budget, st.pcap) = ((res / 2).max(4 * st.tile * st.tile), st.ents.len());
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n).thread_name(|i| format!("gf17-{i}")).build().map_err(|e| bad(e.to_string()))?;
        std::thread::Builder::new().name("gf17-pages".into()).spawn(move || rx.into_iter().for_each(|j| pool.install(|| j(&mut st))))?;
        Ok(Pages { tx, ix, t, tiers: tiers.clamp(1, 4), cc: (0..1 << SH).map(|_| Mutex::new(vec![[0u32; 4]; 1 << (b - SH)])).collect(), b, lv: Arc::new((0..l).map(|_| OnceLock::new()).collect()), sm: Arc::new((0..l).map(|_| AtomicBool::new(false)).collect()), npin: budget.saturating_sub(res) / (8 * t), budget })
    }
    pub fn run<R: Send + 'static>(&self, f: impl FnOnce(&mut Store) -> R + Send + 'static) -> R { let (t, r) = mpsc::sync_channel(1); self.tx.send(Box::new(move |s| { let _ = t.send(f(s)); })).expect("gf17 page server stopped"); r.recv().expect("gf17 page server stopped") }
    fn slot(&self, l: usize, e: usize) -> (usize, usize) { let i = ((l * self.t + e) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - self.b); ((i >> (self.b - SH)) as usize, (i & ((1 << (self.b - SH)) - 1)) as usize) }
    fn look(&self, l: usize, e: usize) -> Option<[u32; 2]> { let (s, i) = self.slot(l, e); let x = self.cc[s].lock().unwrap()[i]; (x[0] == l as u32 + 1 && x[1] == e as u32).then_some([x[2], x[3]]) }
    fn known(&self, l: usize) -> bool { self.lv[l].get().is_some() || self.sm[l].load(Relaxed) }
    pub fn ensure(&self, mut ls: Vec<usize>) {
        ls.sort_unstable();
        ls.dedup();
        let (lv, sm, ix, t, n, np, t0) = (self.lv.clone(), self.sm.clone(), self.ix, self.t, self.tiers, self.npin, std::time::Instant::now());
        self.run(move |st| ls.iter().try_for_each(|&z| { let have = lv.iter().filter(|x| x.get().is_some()).count(); if lv[z].get().is_some() || sm[z].load(Relaxed) { return Ok(()); } if have >= np { sm[z].store(true, Relaxed); return Ok(()); } let _ = lv[z].set(cells(st, ix, t, n, z)?); STAT[8].fetch_add(1, Relaxed); Ok(()) }).inspect(|_| trim())).unwrap_or_else(|e: io::Error| panic!("gf17 page level: {e}"));
        STAT[9].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    }
    pub fn fetch(&self, m: &[(usize, usize)]) -> Vec<[u32; 2]> {
        let ([a, b], n, t0) = (self.ix, self.tiers, std::time::Instant::now());
        let q: Vec<(usize, usize, usize)> = m.iter().flat_map(|&(l, e)| [(a, l, 3 * e / 2), (a, l, 3 * e / 2 + 1), (b, l, 3 * e), (b, l, 3 * e + 1), (b, l, 3 * e + 2)]).collect();
        let w = self.run(move |st| { let w = st.get(&q, n); [(5, st.used as u64), (6, st.miss), (7, st.ev)].iter().for_each(|&(i, v)| STAT[i].store(v, Relaxed)); (STAT[2].fetch_add(1, Relaxed) % 128 == 127).then(|| { st.unmap(); trim() }); w }).unwrap_or_else(|e| panic!("gf17 page fetch: {e}"));
        [(1, m.len() as u64), (4, t0.elapsed().as_nanos() as u64)].iter().for_each(|&(i, v)| { STAT[i].fetch_add(v, Relaxed); });
        m.iter().zip(w.chunks(5)).map(|(&(l, e), w)| { let (sb, o) = ([w[0].to_le_bytes(), w[1].to_le_bytes()].concat(), 3 * e % 2); let c = mkcell(&sb[o..o + 3], &w[2..5]); let (s, i) = self.slot(l, e); self.cc[s].lock().unwrap()[i] = [l as u32 + 1, e as u32, c[0], c[1]]; c }).collect()
    }
    pub fn want(&self, ks: &[(usize, usize)]) {
        let u: Vec<usize> = ks.iter().map(|k| k.0).filter(|&l| !self.known(l)).collect();
        (!u.is_empty()).then(|| self.ensure(u));
        let mut m: Vec<(usize, usize)> = ks.iter().copied().filter(|&(l, e)| self.lv[l].get().is_none() && self.look(l, e).is_none()).collect();
        m.sort_unstable();
        m.dedup();
        (!m.is_empty()).then(|| self.fetch(&m));
    }
    pub fn at(&self, l: usize, e: usize) -> [u32; 2] {
        if let Some(v) = self.lv[l].get() { return v[e]; }
        if !self.sm[l].load(Relaxed) { self.ensure(vec![l]); return self.at(l, e); }
        STAT[3].fetch_add(1, Relaxed);
        self.look(l, e).inspect(|_| { STAT[0].fetch_add(1, Relaxed); }).unwrap_or_else(|| self.fetch(&[(l, e)])[0])
    }
    pub fn level(&self, z: usize) -> io::Result<Vec<u8>> { let (ix, t, n, t0) = (self.ix, self.t, self.tiers, std::time::Instant::now()); let r = self.lv[z].get().map_or_else(|| self.run(move |st| cells(st, ix, t, n, z).map(|v| bytemuck::cast_slice(&v).to_vec()).inspect(|_| trim())), |v| Ok(bytemuck::cast_slice(v).to_vec())); STAT[9].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed); r }
}
#[cfg(feature = "gf17")]
pub fn open(path: &str) -> io::Result<Loaded> {
    let (bud, n) = (env("AMNI_RAY_PAGE_BUDGET_MB", 256) << 20, env("AMNI_RAY_PAGE_TIERS", 4).clamp(1, 4));
    let mut st = Store::open(path, bud)?;
    let (d, bi) = st.table("dp").zip(st.table("bias")).ok_or_else(|| bad(format!("{path}: no dp/bias tables")))?;
    let (s, w) = st.table("sid").zip(st.table("sw")).ok_or_else(|| bad(format!("{path}: no sid/sw tables")))?;
    let (l, t, dr, z) = (st.tabs[s].shape[0], st.tabs[s].shape[1], st.tabs[d].shape[1], |v: u16| if n < 4 && !f16::from_bits(v).is_finite() { 0 } else { v });
    let mut dp: Vec<u16> = st.get(&(0..V * V).map(|x| (d, 0, x)).collect::<Vec<_>>(), n)?.into_iter().map(z).collect();
    let mut d1 = vec![0u16; dr * V];
    st.level(d, 1, n, bytemuck::cast_slice_mut(&mut d1))?;
    dp.extend(d1);
    let bias = st.unpack(bi, 4)?.0.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let cw = (0..copies(l).0).map(|k| { let mut v = vec![0u16; 3 * t]; st.level(w, BASE + 3 + k, n, bytemuck::cast_slice_mut(&mut v)).map(|_| v.into_iter().step_by(3).map(|x| if f16::from_bits(x).is_nan() { 0 } else { x }).collect::<Vec<u16>>()) }).collect::<io::Result<Vec<_>>>()?.concat();
    st.reset(0);
    Ok((l, t, V + dr, dp, vec![], bias, cw, Src::Gf17(Box::new(Pages::new(st, bud, n)?))))
}
#[cfg(not(feature = "gf17"))]
pub fn open(path: &str) -> io::Result<Loaded> { Err(bad(format!("{path}: adam-ray built without the gf17 feature"))) }
impl Tables {
    #[inline]
    pub fn at(&self, l: usize, e: usize) -> [u32; 2] { match &self.src { Src::Npy => self.cells[l * self.t + e], #[cfg(feature = "gf17")] Src::Gf17(p) => p.at(l, e) } }
    pub fn want(&self, pl: &[(usize, usize, u32, f32)], mt: &[Option<u32>]) { match &self.src { #[cfg(feature = "gf17")] Src::Gf17(p) => p.want(&pl.iter().map(|x| (x.0, x.2 as usize)).chain(mt.iter().enumerate().filter_map(|(i, e)| e.map(|e| (self.geo.mb() + i, e as usize)))).collect::<Vec<_>>()), _ => { let _ = (pl, mt); } } }
    pub fn level_cells(&self, z: usize) -> Vec<u8> { match &self.src { Src::Npy => bytemuck::cast_slice(&self.cells[z * self.t..(z + 1) * self.t]).to_vec(), #[cfg(feature = "gf17")] Src::Gf17(p) => p.level(z).unwrap_or_else(|e| panic!("gf17 level {z}: {e}")) } }
}
