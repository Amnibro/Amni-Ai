pub mod store;
#[cfg(feature = "py")]
mod py;
use adam_ray::{cpu::{hop, init, logits, plan, update, St}, pack::{self, Tables, V}};
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs::File, io, path::Path};
pub const PAGE: usize = 4096;
pub const CMAX: usize = 2 * PAGE + 8;
pub const TOT: u64 = 1 << 16;
const TOP: u64 = 1 << 24;
const BOT: u64 = 1 << 16;
const M32: u64 = 0xFFFF_FFFF;
pub const QK: f64 = 64.0;
pub const DMAX: usize = 64 * 24;
pub const EXPC: u64 = 1057095000;
pub const START: u8 = 10;
pub const CODERS: [(&str, &str, &str); 2] = [("ks1-v16", "exports/legacy_grok/rayhash_v16_big2_L21_e20_hop4_q5_s150k_pk_q5_i8.npz", "big2"), ("ks2-clean1", "exports/gf17_continuum/rayhash_clean1v2_base_v18_s100k_sid.npy", "clean1")];
const SEPS: [&[u8]; 4] = [b"\n\n", b"\n", b". ", b" "];
pub fn bad(m: impl Into<String>) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, m.into()) }
pub fn crc16(b: &[u8]) -> u16 { crc32fast::hash(b) as u16 }
fn name(p: &str) -> String { Path::new(p).file_name().map_or_else(|| p.to_string(), |x| x.to_string_lossy().into_owned()) }
fn rfind(h: &[u8], n: &[u8], lo: usize) -> Option<usize> { (lo..(h.len() + 1).saturating_sub(n.len())).rev().find(|&k| &h[k..k + n.len()] == n) }
pub fn paginate(b: &[u8]) -> Vec<&[u8]> {
    let (mut out, mut i) = (vec![], 0);
    while b.len() - i > PAGE {
        let j = SEPS.iter().find_map(|s| rfind(&b[..i + PAGE], s, i + PAGE / 4).map(|k| k + s.len())).unwrap_or_else(|| (i + 1..=i + PAGE).rev().find(|&t| b[t] & 0xC0 != 0x80).unwrap_or(i + PAGE));
        out.push(&b[i..j]);
        i = j;
    }
    if i < b.len() || out.is_empty() { out.push(&b[i..]); }
    out
}
pub fn digest(fs: &[String]) -> io::Result<String> {
    let mut h = Sha256::new();
    for f in fs { let mut g = Sha256::new(); io::copy(&mut File::open(f)?, &mut g)?; h.update(g.finalize()); }
    Ok(h.finalize()[..16].iter().map(|b| format!("{b:02x}")).collect())
}
pub fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() { match c { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"), '\u{8}' => o.push_str("\\b"), '\u{c}' => o.push_str("\\f"), ' '..='~' => o.push(c), _ => c.encode_utf16(&mut [0u16; 2]).iter().for_each(|u| o.push_str(&format!("\\u{u:04x}"))) } }
    o.push('"');
    o
}
pub fn pyjson(v: &Value, ind: Option<usize>, lv: usize) -> String {
    let (pad, pad0, sep) = ind.map_or((String::new(), String::new(), ", ".to_string()), |k| (format!("\n{}", " ".repeat(k * (lv + 1))), format!("\n{}", " ".repeat(k * lv)), format!(",\n{}", " ".repeat(k * (lv + 1)))));
    match v {
        Value::String(s) => jstr(s),
        Value::Array(a) if !a.is_empty() => format!("[{pad}{}{pad0}]", a.iter().map(|x| pyjson(x, ind, lv + 1)).collect::<Vec<_>>().join(&sep)),
        Value::Object(m) if !m.is_empty() => format!("{{{pad}{}{pad0}}}", m.iter().map(|(k, x)| format!("{}: {}", jstr(k), pyjson(x, ind, lv + 1))).collect::<Vec<_>>().join(&sep)),
        _ => v.to_string(),
    }
}
#[derive(Clone, Debug)]
pub struct Ident { pub coder: String, pub ver: String, pub pack: String, pub pack_id: String, pub lex: String, pub lex_id: String }
impl Ident {
    pub fn new(pack: &str, lexdir: &str, ver: Option<&str>) -> io::Result<Ident> {
        let ex = |v: Vec<String>| v.into_iter().filter(|f| Path::new(f).exists()).collect::<Vec<_>>();
        let lf = |xs: &[&str]| ex(xs.iter().map(|x| format!("{}/{x}", lexdir.trim_end_matches('/'))).collect());
        let raw = pack.ends_with("_sid.npy");
        let pre = if raw { pack[..pack.len() - 8].to_string() } else { pack.replace("_pk_q5_i8.npz", "") };
        let pf = [ex(if raw { vec![format!("{pre}_dp.npy"), pack.into(), format!("{pre}_sw.npy"), format!("{pre}_bias.npy")] } else { vec![pack.into(), format!("{pre}_bias.npy")] }), lf(&["synsets.txt", "wordcounts.txt"])].concat();
        let (pid, lid, base, lex) = (digest(&pf)?, digest(&lf(&["synsets.txt", "words.txt", "wordcounts.txt"]))?, name(pack), name(lexdir.trim_end_matches('/')));
        let ver = ver.map(String::from).unwrap_or_else(|| CODERS.iter().find(|c| name(c.1) == base && c.2 == lex).map_or("custom", |c| c.0).into());
        Ok(Ident { coder: format!("ksc1:rust-cpu:{base}:{}", &pid[..12]), ver, pack: base, pack_id: pid, lex, lex_id: lid })
    }
    pub fn json(&self) -> Value { json!({"coder": self.coder, "ver": self.ver, "pack": self.pack, "pack_id": self.pack_id, "lex": self.lex, "lex_id": self.lex_id, "impl": "rust", "arch": "cpu", "qk": QK as i64, "dmax": DMAX, "expc": EXPC, "tot": TOT, "start": START, "page": PAGE}) }
}
fn renorm(low: &mut u64, rng: &mut u64, mut f: impl FnMut(u64)) {
    loop {
        let u = (*low ^ (*low + *rng)) < TOP;
        let b = !u && *rng < BOT;
        *rng = if b { low.wrapping_neg() & (BOT - 1) } else { *rng };
        if !(u || b) { return; }
        assert!(*rng != 0, "ksc1 range collapsed to 0: this page cannot be coded (the Python coder loops forever here)");
        f(*low >> 24);
        (*low, *rng) = ((*low << 8) & M32, (*rng << 8) & M32);
    }
}
fn flush(low: u64, rng: u64) -> Vec<u8> { (0..5).find_map(|k| { let u = 1u64 << (32 - 8 * k); let v = low.div_ceil(u) * u; (v < low + rng && v <= M32).then(|| v.to_be_bytes()[4..4 + k].to_vec()) }).unwrap_or_default() }
pub fn top3(lg: &[f32; V]) -> [u32; 3] {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") { return unsafe { top3x(lg) }; }
    let (mut z, mut a) = (*lg, [0u32; 3]);
    for a in a.iter_mut() {
        let mut mv = [-1e30f32; 8];
        z.chunks_exact(8).for_each(|ch| (0..8).for_each(|j| mv[j] = if ch[j] > mv[j] { ch[j] } else { mv[j] }));
        let m = mv.iter().fold(-1e30f32, |m, &x| if x > m { x } else { m });
        if !(m > -1e30) { break; }
        let i = z.iter().position(|&x| x == m).unwrap();
        (*a, z[i]) = (i as u32, f32::NEG_INFINITY);
    }
    a
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn top3x(lg: &[f32; V]) -> [u32; 3] {
    use std::arch::x86_64::*;
    let (mut z, mut a) = (*lg, [0u32; 3]);
    for a in a.iter_mut() {
        let (p, mut mv, mut w) = (z.as_ptr(), [_mm256_set1_ps(-1e30); 4], [0f32; 8]);
        (0..32).for_each(|k| mv[k & 3] = _mm256_max_ps(_mm256_loadu_ps(p.add(k * 8)), mv[k & 3]));
        _mm256_storeu_ps(w.as_mut_ptr(), _mm256_max_ps(_mm256_max_ps(mv[0], mv[1]), _mm256_max_ps(mv[2], mv[3])));
        let m = w.iter().fold(-1e30f32, |m, &x| if x > m { x } else { m });
        if !(m > -1e30) { break; }
        let (mm, mut k) = (_mm256_set1_ps(m), 0);
        let i = loop { let e = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>(_mm256_loadu_ps(p.add(k * 8)), mm)); if e != 0 { break k * 8 + e.trailing_zeros() as usize; } k += 1; };
        (*a, z[i]) = (i as u32, f32::NEG_INFINITY);
    }
    a
}
fn lse(lg: &[f32; V]) -> f64 { let m = lg.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x as f64)); lg.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m }
struct Ln { j: usize, t: usize, n: usize, low: u64, rng: u64, c: u64, ip: usize, out: Vec<u8>, nll: f64 }
fn pf<T>(x: &T) {
    #[cfg(target_arch = "x86_64")]
    unsafe { std::arch::x86_64::_mm_prefetch(x as *const T as *const i8, std::arch::x86_64::_MM_HINT_T0) }
}
pub struct Ks { pub tb: Tables, dp: Vec<f32>, sel: Vec<[u8; 3]>, lut: Vec<i32>, pub id: Ident, pub lanes: usize, pool: rayon::ThreadPool }
impl Ks {
    pub fn open(pack: &str, lexdir: &str, ver: Option<&str>, threads: usize) -> io::Result<Ks> {
        let tb = pack::load(pack, &format!("{lexdir}/synsets.txt"), &format!("{lexdir}/wordcounts.txt"))?;
        if tb.geo.copy > 0 { return Err(bad("v19 copy-memory packs are not supported by the KS coder")); }
        let mut lut = vec![1u64 << 30; DMAX + 1];
        (1..=DMAX).for_each(|d| lut[d] = lut[d - 1] * EXPC >> 30);
        let n = if threads > 0 { threads } else { std::env::var("RAYON_NUM_THREADS").ok().and_then(|x| x.parse().ok()).unwrap_or(4) };
        Ok(Ks { sel: (0..256i32).map(|u| std::array::from_fn(|s| { let t = [u / 25, u / 5 % 5, u % 5][s] - 2; if t == 0 { 0 } else { 1 + ((t.abs() - 1) & 1) as u8 + 2 * (t < 0) as u8 } })).collect(), lanes: std::env::var("AMNI_KS_LANES").ok().and_then(|x| x.parse().ok()).unwrap_or(8), dp: tb.dp.par_iter().map(|&x| half::f16::from_bits(x).to_f32()).collect(), tb, lut: lut.iter().map(|&x| x as i32).collect(), id: Ident::new(pack, lexdir, ver)?, pool: rayon::ThreadPoolBuilder::new().num_threads(n).build().map_err(|e| bad(e.to_string()))? })
    }
    pub fn ver(home: &str, ver: &str, threads: usize) -> io::Result<Ks> { let c = CODERS.iter().find(|c| c.0 == ver).ok_or_else(|| bad(format!("unknown coder {ver}")))?; Ks::open(&format!("{home}/{}", c.1), &format!("{home}/data/{}", c.2), Some(ver), threads) }
    pub fn threads(&self) -> usize { self.pool.current_num_threads() }
    pub fn step_ref(&self, s: &mut St, lg: &mut [f32; V], b: u8) { *s = update(&self.tb.lx, &self.tb.geo, s, b as u32); logits(&self.tb, s, lg, &[], &[]); }
    fn cell(&self, lg: &mut [f32; V], l: usize, e: u32, w: f32) {
        let (c, h) = (self.tb.cells[l * self.tb.t + e as usize], |x: u32| half::f16::from_bits(x as u16).to_f32());
        let (lo, hi) = (h(c[1]), h(c[1] >> 16));
        let (v, sel) = ([0.0, lo, hi, -lo, -hi], &self.sel[(c[0] >> 24) as usize]);
        (0..3).for_each(|s| lg[(c[0] >> (8 * s) & 255) as usize] += w * v[sel[s] as usize]);
    }
    pub fn steps(&self, st: &mut [St], lg: &mut [[f32; V]], by: &[u8]) {
        let (tb, hb, row) = (&self.tb, self.tb.geo.hb(), |s: &St| (256 + s[1] as usize * 256 + s[0] as usize) * V);
        let pl: Vec<Vec<(usize, usize, u32, f32)>> = st.iter_mut().zip(by).map(|(s, &b)| { *s = update(&tb.lx, &tb.geo, s, b as u32); let p = plan(&tb.geo, s); p.iter().for_each(|x| pf(&tb.cells[x.0 * tb.t + x.2 as usize])); (0..V).step_by(16).for_each(|c| pf(&self.dp[row(s) + c])); p }).collect();
        for ((s, l), p) in st.iter().zip(lg.iter_mut()).zip(&pl) { let (r1, r2) = (&self.dp[s[0] as usize * V..][..V], &self.dp[row(s)..][..V]); (0..V).for_each(|c| l[c] = tb.bias[c] + r1[c] + r2[c]); p.iter().for_each(|&(k, _, e, w)| self.cell(l, k, e, w)); }
        for h in 0..tb.geo.hops { let es: Vec<u32> = st.iter().zip(lg.iter()).map(|(s, l)| { let e = hop(&tb.geo, top3(l), s[0], s[1], h); pf(&tb.cells[(hb + h) * tb.t + e as usize]); e }).collect(); lg.iter_mut().zip(&es).for_each(|(l, &e)| self.cell(l, hb + h, e, 1.0)); }
    }
    pub fn freqs(&self, lg: &[f32; V], f: &mut [u32; V]) -> u64 {
        let (mut d, mut x, mut t) = ([0u32; V], [0i32; V], 0u32);
        if lg.iter().fold(true, |a, x| a & (x.abs() < 1e6)) { let mut q = [0i32; V]; (0..V).for_each(|c| q[c] = unsafe { (lg[c] * 64.0).floor().to_int_unchecked::<i32>() }); let m = q.iter().fold(i32::MIN, |a, &b| a.max(b)); (0..V).for_each(|c| d[c] = ((m - q[c]) as u32).min(DMAX as u32)); } else { let q: [i64; V] = std::array::from_fn(|c| { let x = lg[c]; ((if x.is_nan() { -1e4 } else if x.is_infinite() { 1e4f32.copysign(x) } else { x }) as f64 * QK).floor() as i64 }); let m = q.iter().fold(i64::MIN, |a, &b| a.max(b)); (0..V).for_each(|c| d[c] = ((m - q[c]) as u64).min(DMAX as u64) as u32); }
        (0..V).for_each(|c| x[c] = self.lut[d[c] as usize]);
        let (e, k) = (x.iter().map(|&v| v as i64).sum::<i64>() as f64, (TOT - 256) as f64);
        let r = k / e;
        (0..V).for_each(|c| { let xc = x[c] as f64; let (n, q) = (xc * k, (xc * r).floor()); let z = q.mul_add(e, -n); f[c] = 1 + unsafe { (q + if z > 0.0 { -1.0 } else if z + e <= 0.0 { 1.0 } else { 0.0 }).to_int_unchecked::<i32>() } as u32; t += f[c]; });
        t as u64
    }
    fn run(&self, jobs: &[&[u8]], lens: &[usize], dec: bool, stats: bool) -> Vec<(Vec<u8>, f64)> {
        let (mut res, mut ln, mut st, mut lg, mut by, mut nx, mut f): (Vec<(Vec<u8>, f64)>, Vec<Ln>, Vec<St>, Vec<[f32; V]>, Vec<u8>, usize, [u32; V]) = (vec![(vec![], 0.0); jobs.len()], vec![], vec![], vec![], vec![], 0, [0; V]);
        loop {
            while ln.len() < self.lanes && nx < jobs.len() {
                if lens[nx] > 0 { let at = |i: usize| *jobs[nx].get(i).unwrap_or(&0) as u64; ln.push(Ln { j: nx, t: 0, n: lens[nx], low: 0, rng: M32, c: (0..4).fold(0, |a, i| a << 8 | at(i)), ip: 4, out: Vec::with_capacity(if dec { lens[nx] } else { lens[nx] / 3 + 8 }), nll: 0.0 }); st.push(init()); lg.push([0f32; V]); by.push(START); }
                nx += 1;
            }
            if ln.is_empty() { return res; }
            self.steps(&mut st, &mut lg, &by);
            let mut j = 0;
            while j < ln.len() {
                let (code, g) = (jobs[ln[j].j], &lg[j]);
                let l = &mut ln[j];
                let (tt, at) = (self.freqs(g, &mut f), |i: usize| *code.get(i).unwrap_or(&0) as u64);
                let rr = l.rng / tt;
                let (b, lo) = if dec { let v = ((l.c.wrapping_sub(l.low) & M32).checked_div(rr).unwrap_or(0)).min(tt - 1); let (mut b, mut lo) = (0, 0u64); while lo + f[b] as u64 <= v { lo += f[b] as u64; b += 1; } (b, lo) } else { let b = code[l.t] as usize; (b, f[..b].iter().map(|&x| x as u64).sum()) };
                l.nll += if stats { (lse(g) - g[b] as f64) / std::f64::consts::LN_2 } else { 0.0 };
                (l.low, l.rng) = ((l.low + rr * lo) & M32, rr * f[b] as u64);
                if dec { renorm(&mut l.low, &mut l.rng, |_| { l.c = ((l.c << 8) | at(l.ip)) & M32; l.ip += 1; }); l.out.push(b as u8); } else { renorm(&mut l.low, &mut l.rng, |x| l.out.push(x as u8)); }
                (l.t, by[j]) = (l.t + 1, b as u8);
                if l.t < l.n { j += 1; continue; }
                let mut l = ln.swap_remove(j);
                (st.swap_remove(j), lg.swap_remove(j), by.swap_remove(j));
                if !dec { l.out.extend(flush(l.low, l.rng)); }
                res[l.j] = (l.out, l.nll);
            }
        }
    }
    fn par_run(&self, jobs: &[&[u8]], lens: &[usize], dec: bool, stats: bool) -> Vec<(Vec<u8>, f64)> {
        let mut o: Vec<usize> = (0..lens.len()).collect();
        o.sort_by_key(|&i| std::cmp::Reverse(lens[i]));
        let r: Vec<Vec<(Vec<u8>, f64)>> = self.pool.install(|| o.par_chunks(self.lanes * 4).map(|c| self.run(&c.iter().map(|&i| jobs[i]).collect::<Vec<_>>(), &c.iter().map(|&i| lens[i]).collect::<Vec<_>>(), dec, stats)).collect());
        let mut out = vec![(vec![], 0.0); lens.len()];
        o.iter().zip(r.into_iter().flatten()).for_each(|(&i, x)| out[i] = x);
        out
    }
    pub fn encode(&self, p: &[u8], stats: bool) -> (Vec<u8>, f64) { self.run(&[p], &[p.len()], false, stats).remove(0) }
    pub fn decode(&self, code: &[u8], n: usize) -> Vec<u8> { self.run(&[code], &[n], true, false).remove(0).0 }
    pub fn encode_many(&self, pages: &[&[u8]], stats: bool) -> (Vec<Vec<u8>>, f64) { let r = self.par_run(pages, &pages.iter().map(|p| p.len()).collect::<Vec<_>>(), false, stats); let n = r.iter().map(|x| x.1).sum(); (r.into_iter().map(|x| x.0).collect(), n) }
    pub fn decode_many(&self, lens: &[usize], codes: &[&[u8]]) -> Vec<Vec<u8>> { self.par_run(codes, lens, true, false).into_iter().map(|x| x.0).collect() }
}
