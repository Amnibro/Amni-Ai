#[cfg(feature = "py")]
mod py;
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use std::{borrow::Cow, collections::HashMap, fs::{File, OpenOptions}, io, os::unix::{fs::FileExt, io::AsRawFd}, sync::Arc};
pub const K4: [u32; 4] = [1, 17, 289, 4913];
pub const PW: usize = 4096;
pub const PH: usize = 4096;
pub const PX: usize = PW * PH;
pub const HDR: usize = 4096;
pub const MID: u8 = 8;
pub const TILE: usize = 1 << 16;
pub const CH: usize = 1 << 14;
pub const MAGIC: &[u8; 8] = b"GF17PTX1";
pub const CKPT: [(&str, bool); 4] = [("sid", true), ("sw", false), ("dp", false), ("bias", true)];
pub const RB: u32 = 10;
pub const RM: u32 = 1 << RB;
pub const RL: u32 = 1 << 16;
pub const LN: usize = 32;
pub const FT: usize = 17 * 17;
#[derive(Clone, Debug)]
pub struct Table { pub name: String, pub dt: usize, pub ord: bool, pub exact: bool, pub shape: Vec<usize> }
#[derive(Clone, Copy, Debug)]
pub struct Ent { pub t: u32, pub l: u32, pub p: u32, pub k: u8, pub rows: u32, pub c0: u64, pub c1: u64, pub off: u64, pub len: u64 }
pub struct Pl { lut: Vec<u32>, pub off: Vec<u32>, pub hs: Vec<u32> }
pub struct Writer { pub f: File, pub tabs: Vec<Table>, pub ents: Vec<Ent>, loc: Vec<Vec<usize>>, drop: bool, pub tile: usize, sp: Option<(File, String)>, so: u64 }
pub struct Store { pub path: String, f: File, map: Mmap, pub tabs: Vec<Table>, pub ents: Vec<Ent>, loc: Vec<Vec<usize>>, pub budget: usize, pub used: usize, nx: Vec<u32>, pv: Vec<u32>, seen: Vec<u16>, pub hits: u64, pub miss: u64, pub ev: u64, cs: Option<(usize, usize)>, pub tile: usize, tb: Vec<Vec<usize>>, tiles: HashMap<usize, Arc<(Vec<u8>, usize)>>, pls: HashMap<usize, Arc<Pl>>, free: Vec<Vec<u8>>, pub pcap: usize }
struct Sp(*mut u8);
unsafe impl Sync for Sp {}
unsafe impl Send for Sp {}
impl Sp { fn at(&self, i: usize) -> *mut u8 { unsafe { self.0.add(i) } } }
fn bad(m: String) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, m) }
pub fn key(b: u16, ord: bool) -> u16 { if !ord { b } else if b >> 15 == 1 { !b } else { b | 0x8000 } }
pub fn unkey(k: u16, ord: bool) -> u16 { if !ord { k } else if k >> 15 == 1 { k & 0x7FFF } else { !k } }
pub fn dec(k: u16, n: usize, ord: bool) -> u16 { unkey(if ord && n < 4 { k.clamp(0x0400, 0xFBFF) } else { k }, ord) }
pub fn dig(v: u16, k: usize) -> u8 { (v as u32 / K4[k] % 17) as u8 }
pub fn val(d: [u8; 4], n: usize) -> u16 { (0..4).map(|k| K4[k] * if k + n >= 4 { d[k] } else { MID } as u32).sum::<u32>().min(65535) as u16 }
pub const PXB: usize = 17;
pub const PG17: usize = PX * PXB / 8;
pub fn pxw(d: &[u8]) -> u32 { (0..4).map(|k| K4[k] * *d.get(k).unwrap_or(&0) as u32).sum() }
pub fn lut17() -> Vec<u32> { (0..K4[3] * 17).map(|v| u32::from_le_bytes(std::array::from_fn(|k| (v / K4[k] % 17) as u8))).collect() }
#[inline(always)]
pub fn px17(b: &[u8], p: usize) -> u32 { let o = (p << 4) + p; u32::from_le_bytes(b[o >> 3..(o >> 3) + 4].try_into().unwrap()) >> (o & 7) & 0x1FFFF }
pub fn pack17(dg: &[u8]) -> Vec<u8> { let mut o = vec![0u8; (dg.len().div_ceil(4) * PXB).div_ceil(8) + 8]; o.par_chunks_mut(PXB).zip(dg.par_chunks(32)).for_each(|(b, c)| c.chunks(4).enumerate().for_each(|(i, d)| { let (s, w) = ((i << 4) + i, pxw(d) as u64); let v = w << (s & 7); (0..4).for_each(|j| { let k = (s >> 3) + j; (k < b.len()).then(|| b[k] |= (v >> (8 * j)) as u8); }) })); o }
pub fn unpack17(b: &[u8], n: usize) -> Vec<u8> { let l = lut17(); let mut o = vec![0u8; n.div_ceil(4) * 4]; o.par_chunks_mut(4).enumerate().for_each(|(p, c)| c.copy_from_slice(&l[px17(b, p) as usize].to_le_bytes())); o.truncate(n); o }
pub fn page17(b: &[u8], n: usize, pg: usize) -> Vec<u8> { let (a, z) = (pg * PX, ((pg + 1) * PX).min(n.div_ceil(4))); let l = lut17(); let mut o = vec![0u8; (z.max(a) - a) * 4]; o.par_chunks_mut(4).enumerate().for_each(|(i, c)| c.copy_from_slice(&l[px17(b, a + i) as usize].to_le_bytes())); o }
pub fn words(b: &[u8]) -> Vec<u16> { b.par_chunks(2).with_min_len(1 << 14).map(|c| c[0] as u16 | (*c.get(1).unwrap_or(&0) as u16) << 8).collect() }
pub fn rss() -> (usize, usize, usize) { let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default(); let g = |k: &str| s.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0) << 10; (g("VmRSS:"), g("VmHWM:"), g("RssFile:")) }
pub fn hash(b: &[u8]) -> u32 { let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ b.len() as u64; let mx = |h: u64, w: u64| (h ^ w).wrapping_mul(0xFF51_AFD7_ED55_8CCD).rotate_left(29); let c = b.chunks_exact(8); let r = c.remainder(); c.for_each(|x| h = mx(h, u64::from_le_bytes(x.try_into().unwrap()))); r.iter().for_each(|&x| h = mx(h, x as u64 | 0x100)); h = (h ^ h >> 31).wrapping_mul(0x94D0_49BB_1331_11EB); (h ^ h >> 32) as u32 }
static ZP: [u8; CH] = [0; CH];
pub fn asm(d: [&[u8]; 4], n: usize, tb: &Table, o: &mut [u8]) -> usize {
    let (m, base, wk) = (o.len() / 2, (0..4 - n).map(|k| MID as u32 * K4[k]).sum::<u32>(), std::array::from_fn::<u32, 4, _>(|k| if k + n >= 4 { K4[k] } else { 0 }));
    let s = |i: usize| (base + wk[0] * d[0][i] as u32 + wk[1] * d[1][i] as u32 + wk[2] * d[2][i] as u32 + wk[3] * d[3][i] as u32).min(65535) as u16;
    let f = |v: u16| { let v = dec(v, n, tb.ord); let z = n < 4 && tb.dt == 2 && !f16::from_bits(v).is_finite(); (if z { 0 } else { v }, z as usize) };
    let z = if n == 4 && !tb.ord { o.chunks_exact_mut(2).zip(d[0].iter().zip(d[1]).zip(d[2].iter().zip(d[3]))).for_each(|(c, ((&a, &b), (&e, &g)))| c.copy_from_slice(&(a as u16 + 17 * b as u16 + 289 * e as u16 + 4913 * g as u16).to_le_bytes())); 0 } else { (0..m).map(|i| { let (v, z) = f(s(i)); o[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes()); z }).sum() };
    z + (o.len() % 2 == 1).then(|| { let (v, z) = f(s(m)); o[2 * m] = v as u8; z }).unwrap_or(0)
}
pub fn rects(rows: usize, tw: usize) -> Vec<(usize, usize, usize)> { (0..rows.div_ceil(tw)).flat_map(|ty| (0..PW / tw).map(move |tx| (tx * tw, ty * tw, tw.min(rows - ty * tw)))).collect() }
fn gat(d: &[u8], (x, y, h): (usize, usize, usize), tw: usize) -> Vec<u8> { (0..h).flat_map(|r| &d[(y + r) * PW + x..(y + r) * PW + x + tw]).copied().collect() }
pub fn norm(h: &[u64]) -> Vec<u16> { h.chunks(17).flat_map(|c| { let n: u64 = c.iter().sum(); let mut f: Vec<i64> = c.iter().map(|&x| if x == 0 { 0 } else { ((x * RM as u64 + n / 2) / n).max(1) as i64 }).collect(); let j = (0..17).max_by_key(|&i| f[i]).unwrap(); f[j] += if n == 0 { 0 } else { RM as i64 - f.iter().sum::<i64>() }; f.into_iter().map(|x| x as u16) }).collect() }
fn cum(f: &[u16]) -> Vec<u16> { f.chunks(17).flat_map(|c| c.iter().scan(0u16, |a, &x| { let r = *a; *a += x; Some(r) }).collect::<Vec<_>>()).collect() }
pub fn renc(s: &[u8], c: &[u8], f: &[u16], cu: &[u16]) -> Vec<u8> {
    if s.iter().all(|&x| x == s[0]) { return vec![s[0]]; }
    let (mut x, mut o) = ([RL; LN], Vec::<u16>::with_capacity(s.len() / 2));
    (0..s.len()).rev().for_each(|i| { let j = c[i] as usize * 17 + s[i] as usize; let (fi, ci, y) = (f[j] as u32, cu[j] as u32, &mut x[i % LN]); (*y as u64 >= (fi as u64) << (32 - RB)).then(|| { o.push(*y as u16); *y >>= 16; }); *y = (*y / fi << RB) + *y % fi + ci; });
    x.iter().flat_map(|y| y.to_le_bytes()).chain(o.iter().rev().flat_map(|w| w.to_le_bytes())).collect()
}
const PERM: [[u32; 8]; 256] = { let mut t = [[0u32; 8]; 256]; let mut m = 0; while m < 256 { let (mut j, mut k) = (0, 0); while j < 8 { if m >> j & 1 == 1 { t[m][j] = k; k += 1; } j += 1; } m += 1; } t };
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rdec8<const LF: bool>(b: &[u8], u: &[u8], lut: &[u32], o: &mut [u8], x: &mut [u32; LN], p: &mut usize) {
    use std::arch::x86_64::*;
    let (mk, m31, m16, z, sh) = (_mm256_set1_epi32(RM as i32 - 1), _mm256_set1_epi32(31), _mm256_set1_epi32(16), _mm256_setzero_si256(), _mm256_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    let (mut xv, mut cv, mut pad): ([__m256i; LN / 8], [__m256i; LN / 8], [u8; 16]) = (std::array::from_fn(|q| _mm256_loadu_si256(x.as_ptr().add(8 * q) as *const __m256i)), [z; LN / 8], [0; 16]);
    for g in 0..o.len() / LN { for q in 0..LN / 8 {
        let i = g * LN + 8 * q;
        let c = if LF { cv[q] } else { _mm256_min_epu32(_mm256_cvtepu8_epi32(_mm_loadl_epi64(u.as_ptr().add(i) as *const __m128i)), m16) };
        let v = _mm256_i32gather_epi32::<4>(lut.as_ptr() as *const i32, _mm256_or_si256(_mm256_slli_epi32::<{ RB as i32 }>(c), _mm256_and_si256(xv[q], mk)));
        let xn = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32::<16>(v), _mm256_srli_epi32::<{ RB as i32 }>(xv[q])), _mm256_and_si256(_mm256_srli_epi32::<5>(v), mk));
        let need = _mm256_cmpeq_epi32(_mm256_srli_epi32::<16>(xn), z);
        let m = _mm256_movemask_ps(_mm256_castsi256_ps(need)) as usize;
        let src = if *p + 16 <= b.len() { b.as_ptr().add(*p) } else { pad.fill(0); b.get(*p..).map(|r| pad[..r.len()].copy_from_slice(r)); pad.as_ptr() };
        let w = _mm256_permutevar8x32_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(src as *const __m128i)), _mm256_loadu_si256(PERM[m].as_ptr() as *const __m256i));
        xv[q] = _mm256_blendv_epi8(xn, _mm256_or_si256(_mm256_slli_epi32::<16>(xn), w), need);
        *p += 2 * m.count_ones() as usize;
        cv[q] = _mm256_and_si256(v, m31);
        let t = _mm256_shuffle_epi8(cv[q], sh);
        std::ptr::copy_nonoverlapping([_mm_cvtsi128_si32(_mm256_castsi256_si128(t)), _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(t))].as_ptr() as *const u8, o.as_mut_ptr().add(i), 8);
    } }
    (0..LN / 8).for_each(|q| _mm256_storeu_si256(x.as_mut_ptr().add(8 * q) as *mut __m256i, xv[q]));
}
fn rdec<const LF: bool>(b: &[u8], u: &[u8], lut: &[u32], o: &mut [u8]) -> bool {
    if b.len() == 1 { o.fill(b[0]); return b[0] < 17; }
    if b.len() < 4 * LN || b.len() % 2 == 1 || (!LF && u.len() < o.len()) || lut.len() < 17 << RB { return false; }
    let (mut x, mut p, n): ([u32; LN], usize, usize) = (std::array::from_fn(|j| u32::from_le_bytes(b[4 * j..4 * j + 4].try_into().unwrap())), 4 * LN, o.len());
    #[cfg(target_arch = "x86_64")]
    let n8 = if is_x86_feature_detected!("avx2") { unsafe { rdec8::<LF>(b, u, lut, o, &mut x, &mut p) }; n / LN * LN } else { 0 };
    #[cfg(not(target_arch = "x86_64"))]
    let n8 = 0;
    (n8..n).for_each(|i| { let j = i % LN; let c = if LF { if i >= LN { o[i - LN] } else { 0 } } else { u[i] }.min(16) as usize; let v = lut[c << RB | (x[j] & (RM - 1)) as usize]; let z = (v >> 16).wrapping_mul(x[j] >> RB).wrapping_add((v >> 5) & (RM - 1)); let r = z < RL; let w = u16::from_le_bytes([*b.get(p).unwrap_or(&0), *b.get(p + 1).unwrap_or(&0)]) as u32; x[j] = if r { z << 16 | w } else { z }; p += 2 * r as usize; o[i] = (v & 31) as u8; });
    p == b.len() && x == [RL; LN]
}
pub fn encp(d: &[u8], u: Option<&[u8]>, rows: usize, tw: usize) -> Vec<u8> {
    let g: Vec<(Vec<u8>, Vec<u8>)> = rects(rows, tw).into_par_iter().map(|r| { let s = gat(d, r, tw); let c = u.map_or_else(|| (0..s.len()).map(|i| if i >= LN { s[i - LN] } else { 0 }).collect(), |u| gat(u, r, tw)); (s, c) }).collect();
    let h = g.par_iter().fold(|| vec![0u64; FT], |mut h, (s, c)| { s.iter().zip(c).for_each(|(&a, &b)| h[b as usize * 17 + a as usize] += 1); h }).reduce(|| vec![0u64; FT], |a, b| a.iter().zip(&b).map(|(x, y)| x + y).collect());
    let (f, hl) = (norm(&h), 8 + 2 * FT + 8 * g.len() + 4);
    let cu = cum(&f);
    let bs: Vec<Vec<u8>> = g.par_iter().map(|(s, c)| renc(s, c, &f, &cu)).collect();
    let mut o = (g.len() as u32).to_le_bytes().to_vec();
    o.extend([0; 4]);
    f.iter().for_each(|x| o.extend(x.to_le_bytes()));
    o.extend((hl as u32).to_le_bytes());
    bs.iter().scan(hl, |a, b| { *a += b.len(); Some(*a as u32) }).collect::<Vec<_>>().iter().chain(bs.iter().map(|b| hash(b)).collect::<Vec<_>>().iter()).for_each(|x| o.extend(x.to_le_bytes()));
    let hh = hash(&o[8..]);
    o[4..8].copy_from_slice(&hh.to_le_bytes());
    bs.iter().for_each(|b| o.extend(b));
    o
}
pub fn parse(b: &[u8], nt: usize) -> io::Result<Pl> {
    let hl = 8 + 2 * FT + 8 * nt + 4;
    let r4 = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    (b.len() >= hl && r4(0) as usize == nt && hash(&b[8..hl]) == r4(4)).then_some(()).ok_or_else(|| bad(format!("plane header corrupt ({} B, {nt} tiles)", b.len())))?;
    let f: Vec<u16> = (0..FT).map(|i| u16::from_le_bytes([b[8 + 2 * i], b[9 + 2 * i]])).collect();
    let off: Vec<u32> = (0..=nt).map(|i| r4(8 + 2 * FT + 4 * i)).collect();
    (f.chunks(17).all(|c| matches!(c.iter().map(|&x| x as u32).sum::<u32>(), 0 | RM)) && off[0] as usize == hl && off.windows(2).all(|w| w[0] < w[1]) && off[nt] as usize <= b.len()).then_some(()).ok_or_else(|| bad("plane frequency table or tile index corrupt".into()))?;
    let (cu, mut lut) = (cum(&f), vec![0u32; 17 << RB]);
    (0..FT).for_each(|j| lut[(j / 17) << RB | cu[j] as usize..][..f[j] as usize].iter_mut().enumerate().for_each(|(i, e)| *e = (j % 17) as u32 | (i as u32) << 5 | (f[j] as u32) << 16));
    Ok(Pl { lut, hs: (0..nt).map(|i| r4(8 + 2 * FT + 4 * (nt + 1) + 4 * i)).collect(), off })
}
impl Table {
    pub fn lv(&self) -> usize { if self.shape.len() > 1 { self.shape[0] } else { 1 } }
    pub fn n(&self) -> usize { self.shape.iter().product() }
    pub fn per(&self) -> usize { self.n() / self.lv().max(1) }
    pub fn cell(&self) -> usize { self.shape.iter().skip(2).product() }
    pub fn words(&self) -> usize { (self.per() * self.dt).div_ceil(2) }
    pub fn np(&self) -> usize { self.words().div_ceil(PX) }
    pub fn rows(&self, p: usize) -> usize { (self.words() - p * PX).min(PX).div_ceil(PW) }
    pub fn tiles(&self, p: usize, tw: usize) -> usize { PW / tw * self.rows(p).div_ceil(tw) }
    pub fn descr(&self) -> &'static str { match self.dt { 1 => "|u1", 2 => "<f2", _ => "<f4" } }
}
pub fn stream(tabs: &[Table], e: &Ent) -> usize { if tabs[e.t as usize].exact { 3 } else { e.k as usize } }
pub fn plan(tabs: &[Table]) -> Vec<Ent> {
    let (mut off, mut v) = (HDR as u64, vec![]);
    for s in (0..4).rev() { for (ti, t) in tabs.iter().enumerate() { for l in 0..t.lv() { for p in 0..t.np() { for k in (0..4).rev().filter(|&k| if t.exact { s == 3 } else { k == s }) {
        let (rows, w0) = (t.rows(p), p * PX);
        v.push(Ent { t: ti as u32, l: l as u32, p: p as u32, k: k as u8, rows: rows as u32, c0: (w0 * 2 / t.dt / t.cell()) as u64, c1: (((w0 + PX).min(t.words()) * 2 / t.dt).min(t.per()).div_ceil(t.cell())) as u64, off, len: (rows * PW) as u64 });
        off += (rows * PW) as u64;
    } } } } }
    v
}
fn locs(tabs: &[Table], ents: &[Ent]) -> Vec<Vec<usize>> { let mut v: Vec<Vec<usize>> = tabs.iter().map(|t| vec![usize::MAX; t.lv() * t.np() * 4]).collect(); ents.iter().enumerate().for_each(|(i, e)| v[e.t as usize][(e.l as usize * tabs[e.t as usize].np() + e.p as usize) * 4 + e.k as usize] = i); v }
fn dir(tabs: &[Table], ents: &[Ent], v2: bool) -> Vec<u8> {
    let mut b = (tabs.len() as u32).to_le_bytes().to_vec();
    tabs.iter().for_each(|t| { b.extend((t.name.len() as u16).to_le_bytes()); b.extend(t.name.as_bytes()); b.extend([t.dt as u8, t.ord as u8, t.exact as u8, t.shape.len() as u8]); t.shape.iter().for_each(|&s| b.extend((s as u64).to_le_bytes())); });
    b.extend((ents.len() as u32).to_le_bytes());
    ents.iter().for_each(|e| { [e.t, e.l, e.p, e.k as u32, e.rows].iter().for_each(|x| b.extend(x.to_le_bytes())); [e.c0, e.c1, e.off].iter().chain(v2.then_some(&e.len)).for_each(|x| b.extend(x.to_le_bytes())); });
    b
}
fn hdr(tile: usize, end: u64, d: &[u8]) -> Vec<u8> {
    let mut h = vec![0u8; HDR];
    h[..8].copy_from_slice(MAGIC);
    [1 + (tile > 0) as u32, PW as u32, PH as u32, (tile > 0) as u32].iter().enumerate().for_each(|(i, x)| h[8 + 4 * i..12 + 4 * i].copy_from_slice(&x.to_le_bytes()));
    h[24..32].copy_from_slice(&end.to_le_bytes());
    h[32..40].copy_from_slice(&(d.len() as u64).to_le_bytes());
    (tile > 0).then(|| { [tile as u32, tile as u32, hash(d)].iter().enumerate().for_each(|(i, x)| h[40 + 4 * i..44 + 4 * i].copy_from_slice(&x.to_le_bytes())); let x = hash(&h[..52]); h[52..56].copy_from_slice(&x.to_le_bytes()); });
    h
}
impl Writer {
    pub fn create(path: &str, tabs: Vec<Table>, drop: bool, tile: usize) -> io::Result<Writer> {
        (tile == 0 || (tile >= 8 && PW.is_multiple_of(tile))).then_some(()).ok_or_else(|| bad(format!("tile {tile} must divide {PW} and be >= 8")))?;
        let ents = plan(&tabs);
        let (end, d) = (ents.last().map_or(HDR as u64, |e| e.off + e.len), dir(&tabs, &ents, false));
        let f = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(path)?;
        (tile == 0).then(|| -> io::Result<()> { f.set_len(end + d.len() as u64)?; f.write_all_at(&hdr(0, end, &d), 0)?; f.write_all_at(&d, end) }).transpose()?;
        let sp = (tile > 0).then(|| { let s = format!("{path}.spill"); OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&s).map(|f| (f, s)) }).transpose()?;
        Ok(Writer { f, loc: locs(&tabs, &ents), tabs, ents, drop, tile, sp, so: 0 })
    }
    fn out(&self, f: &File, b: &[u8], o: u64) -> io::Result<()> { f.write_all_at(b, o)?; if self.drop { f.sync_data()?; unsafe { libc::posix_fadvise(f.as_raw_fd(), o as i64, b.len() as i64, libc::POSIX_FADV_DONTNEED) }; } Ok(()) }
    pub fn put(&mut self, t: usize, l: usize, w: &[u16]) -> io::Result<()> { (w.len() == self.tabs[t].words()).then_some(()).ok_or_else(|| bad(format!("{} level {l}: {} words, want {}", self.tabs[t].name, w.len(), self.tabs[t].words())))?; self.put_at(t, l, 0, w) }
    pub fn put_at(&mut self, t: usize, l: usize, p0: usize, w: &[u16]) -> io::Result<()> {
        let tb = self.tabs[t].clone();
        (p0 * PX + w.len() <= tb.words() && (w.len().is_multiple_of(PX) || p0 * PX + w.len() == tb.words())).then_some(()).ok_or_else(|| bad(format!("{} level {l}: {} words at page {p0} are not whole pages of {} words", tb.name, w.len(), tb.words())))?;
        (p0..p0 + w.len().div_ceil(PX)).try_for_each(|p| {
            let s = &w[(p - p0) * PX..((p - p0) * PX + PX).min(w.len())];
            let d: Vec<Vec<u8>> = (0..4).map(|k| { let mut pl = vec![0u8; tb.rows(p) * PW]; pl.par_iter_mut().with_min_len(1 << 16).zip(s.par_iter()).for_each(|(o, &x)| *o = dig(key(x, tb.ord), k)); pl }).collect();
            (0..4).rev().try_for_each(|k| {
                let e = self.loc[t][(l * tb.np() + p) * 4 + k];
                let b = if self.tile == 0 { Cow::Borrowed(&d[k][..]) } else { Cow::Owned(encp(&d[k], (k < 3).then(|| &d[k + 1][..]), tb.rows(p), self.tile)) };
                let (f, o) = self.sp.as_ref().map_or((&self.f, self.ents[e].off), |x| (&x.0, self.so));
                self.out(f, &b, o)?;
                (self.tile > 0).then(|| { (self.ents[e].off, self.ents[e].len) = (self.so, b.len() as u64); self.so += b.len() as u64; });
                Ok(())
            })
        })
    }
    pub fn finish(mut self) -> io::Result<()> {
        let Some((sp, sn)) = self.sp.take() else { return self.f.sync_all() };
        let src: Vec<u64> = self.ents.iter().map(|e| e.off).collect();
        let end = self.ents.iter_mut().fold(HDR as u64, |o, e| { e.off = o; o + e.len });
        let d = dir(&self.tabs, &self.ents, true);
        self.f.set_len(end + d.len() as u64)?;
        self.f.write_all_at(&hdr(self.tile, end, &d), 0)?;
        self.f.write_all_at(&d, end)?;
        self.ents.iter().zip(&src).try_for_each(|(e, &s)| { let mut b = vec![0u8; e.len as usize]; sp.read_exact_at(&mut b, s)?; self.out(&self.f, &b, e.off) })?;
        self.f.sync_all()?;
        std::fs::remove_file(sn)
    }
}
pub fn write(out: &str, items: &[(Table, &[u8])], tile: usize) -> io::Result<()> {
    let mut w = Writer::create(out, items.iter().map(|x| x.0.clone()).collect(), false, tile)?;
    items.iter().enumerate().try_for_each(|(t, (tb, b))| { let pb = tb.per() * tb.dt; (b.len() == tb.n() * tb.dt).then_some(()).ok_or_else(|| bad(format!("{}: {} bytes for shape {:?}", tb.name, b.len(), tb.shape)))?; (0..tb.lv()).try_for_each(|l| w.put(t, l, &words(&b[l * pb..(l + 1) * pb]))) })?;
    w.finish()
}
#[cfg(feature = "npy")]
pub fn npy(path: &str) -> io::Result<(String, Vec<usize>, Mmap, usize)> {
    let m = unsafe { Mmap::map(&File::open(path)?)? };
    (m.len() > 12 && &m[..6] == b"\x93NUMPY").then_some(()).ok_or_else(|| bad(format!("{path}: bad npy magic")))?;
    let (hl, hs) = if m[6] == 1 { (u16::from_le_bytes([m[8], m[9]]) as usize, 10) } else { (u32::from_le_bytes(m[8..12].try_into().unwrap()) as usize, 12) };
    let h = String::from_utf8_lossy(m.get(hs..hs + hl).ok_or_else(|| bad(format!("{path}: npy header past end")))?).to_string();
    let f = |k: &str, e: char| h.find(k).and_then(|i| h[i + k.len()..].split(e).next().map(str::to_string)).unwrap_or_default();
    (!h.contains("'fortran_order': True")).then_some(()).ok_or_else(|| bad(format!("{path}: fortran order unsupported")))?;
    Ok((f("'descr': '", '\''), f("'shape': (", ')').split(',').filter_map(|v| v.trim().parse().ok()).collect(), m, hs + hl))
}
#[cfg(feature = "npy")]
pub fn pack(prefix: &str, out: &str, ord: bool, tile: usize) -> io::Result<Vec<Table>> {
    let arrs = CKPT.iter().map(|(s, _)| npy(&format!("{prefix}_{s}.npy"))).collect::<io::Result<Vec<_>>>()?;
    let items = CKPT.iter().zip(&arrs).map(|((s, ex), (descr, shape, m, o))| { let dt: usize = descr[2..].parse().map_err(|_| bad(format!("{s}: dtype {descr}")))?; (!descr.starts_with('>') && matches!(dt, 1 | 2 | 4)).then_some(()).ok_or_else(|| bad(format!("{s}: dtype {descr} unsupported")))?; let n = shape.iter().product::<usize>() * dt; (m.len() >= o + n).then_some(()).ok_or_else(|| bad(format!("{s}: truncated npy")))?; Ok((Table { name: s.to_string(), dt, ord: ord && dt == 2, exact: *ex, shape: shape.clone() }, &m[*o..o + n])) }).collect::<io::Result<Vec<_>>>()?;
    write(out, &items, tile)?;
    Ok(items.into_iter().map(|x| x.0).collect())
}
impl Store {
    pub fn open(path: &str, budget: usize) -> io::Result<Store> {
        let f = File::open(path)?;
        let map = unsafe { Mmap::map(&f)? };
        (map.len() >= HDR && &map[..8] == MAGIC).then_some(()).ok_or_else(|| bad(format!("{path}: not a GF(17) page file")))?;
        let r = |o: usize, n: usize| { let mut b = [0u8; 8]; b[..n].copy_from_slice(&map[o..o + n]); u64::from_le_bytes(b) };
        (r(12, 4) as usize == PW && r(16, 4) as usize == PH).then_some(()).ok_or_else(|| bad(format!("{path}: page size {}x{}", r(12, 4), r(16, 4))))?;
        let (ver, tile, mut o, dl) = (r(8, 4), r(40, 4) as usize, r(24, 8) as usize, r(32, 8) as usize);
        (o.checked_add(dl).is_some_and(|x| x <= map.len()) && (ver == 1 || (ver == 2 && tile >= 8 && PW.is_multiple_of(tile) && r(44, 4) as usize == tile && hash(&map[..52]) == r(52, 4) as u32 && hash(&map[o..o + dl]) == r(48, 4) as u32))).then_some(()).ok_or_else(|| bad(format!("{path}: header or directory corrupt (version {ver})")))?;
        let tile = if ver == 2 { tile } else { 0 };
        let mut rd = |n: usize| { o += n; r(o - n, n) };
        let nt = rd(4) as usize;
        let tabs: Vec<Table> = (0..nt).map(|_| { let nl = rd(2) as usize; let name = String::from_utf8_lossy(&(0..nl).map(|_| rd(1) as u8).collect::<Vec<_>>()).to_string(); let (dt, ord, exact, nd) = (rd(1) as usize, rd(1) == 1, rd(1) == 1, rd(1) as usize); Table { name, dt, ord, exact, shape: (0..nd).map(|_| rd(8) as usize).collect() } }).collect();
        let ne = rd(4) as usize;
        let ents: Vec<Ent> = (0..ne).map(|_| { let (t, l, p, k, rows, c0, c1, off) = (rd(4) as u32, rd(4) as u32, rd(4) as u32, rd(4) as u8, rd(4) as u32, rd(8), rd(8), rd(8)); Ent { t, l, p, k, rows, c0, c1, off, len: if tile > 0 { rd(8) } else { rows as u64 * PW as u64 } } }).collect();
        ents.iter().all(|e| e.off.checked_add(e.len).is_some_and(|x| x as usize <= map.len())).then_some(()).ok_or_else(|| bad(format!("{path}: plane past end of file")))?;
        unsafe { libc::madvise(map.as_ptr() as *mut _, map.len(), libc::MADV_RANDOM) };
        let ix = |n: &str| tabs.iter().position(|t| t.name == n);
        let cs = ix("sid").zip(ix("sw"));
        let mut acc = 0usize;
        let tb: Vec<Vec<usize>> = tabs.iter().map(|t| (0..t.lv() * t.np()).map(|i| { let x = if tile > 0 { t.tiles(i % t.np(), tile) } else { 0 }; acc += x; acc - x }).collect()).collect();
        let n = if tile > 0 { acc } else { map.len().div_ceil(TILE) };
        Ok(Store { path: path.into(), f, map, loc: locs(&tabs, &ents), nx: vec![n as u32; n + 1], pv: vec![n as u32; n + 1], seen: vec![0; n], tabs, ents, budget: budget.max(if tile > 0 { 4 * tile * tile } else { TILE }), used: 0, hits: 0, miss: 0, ev: 0, cs, tile, tb, tiles: HashMap::new(), pls: HashMap::new(), free: vec![], pcap: (budget / 8 / (68 << RB)).max(64) })
    }
    pub fn table(&self, name: &str) -> Option<usize> { self.tabs.iter().position(|t| t.name == name) }
    pub fn len(&self) -> usize { self.map.len() }
    pub fn is_empty(&self) -> bool { self.map.is_empty() }
    pub fn ent(&self, t: usize, l: usize, p: usize, k: usize) -> usize { self.loc[t][(l * self.tabs[t].np() + p) * 4 + k] }
    pub fn plane(&self, e: usize) -> &[u8] { let e = &self.ents[e]; &self.map[e.off as usize..(e.off + e.len) as usize] }
    fn adv(&self, o: usize, n: usize, a: i32) { let g = if a == libc::MADV_DONTNEED { 1 << 16 } else { 4096 }; let b = o / g * g; unsafe { libc::madvise(self.map.as_ptr().add(b) as *mut _, (n + o - b).next_multiple_of(g).min(self.map.len() - b), a) }; }
    fn unlink(&mut self, e: usize) { let (a, b) = (self.pv[e] as usize, self.nx[e] as usize); self.nx[a] = b as u32; self.pv[b] = a as u32; }
    fn lru(&mut self, e: usize, sz: usize) {
        let s = self.seen.len();
        if self.seen[e] != 0 { self.hits += 1; self.unlink(e); } else { self.miss += 1; self.used += sz; self.seen[e] = (self.tile > 0) as u16; }
        let f = self.nx[s] as usize;
        (self.nx[e], self.pv[e], self.pv[f], self.nx[s]) = (f as u32, s as u32, e as u32, e as u32);
        while self.used > self.budget && self.pv[s] as usize != e { let t = self.pv[s] as usize; self.unlink(t); self.seen[t] = 0; self.used -= sz; self.ev += 1; if self.tile > 0 { let a = self.tiles.remove(&t); a.map(|a| self.recyc(a)); } else { self.adv(t * TILE, TILE, libc::MADV_DONTNEED) } }
    }
    fn recyc(&mut self, a: Arc<(Vec<u8>, usize)>) { (self.free.len() < 16).then(|| Arc::try_unwrap(a).map(|x| self.free.push(x.0))); }
    fn touch(&mut self, o: usize, pf: &mut Vec<usize>) { let (e, b) = (o / TILE, 1u16 << (o % TILE / PW)); self.lru(e, TILE); if self.seen[e] & b == 0 { self.seen[e] |= b; pf.push(o / PW * PW); } }
    pub fn reset(&mut self, lvl: usize) {
        (lvl > 0).then(|| { self.adv(0, self.map.len(), libc::MADV_DONTNEED); unsafe { libc::posix_fadvise(self.f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) }; });
        (lvl > 1).then(|| self.pls.clear());
        #[cfg(target_env = "gnu")]
        (lvl > 1).then(|| unsafe { libc::malloc_trim(0) });
        let n = self.seen.len();
        (self.nx, self.pv, self.seen, self.used) = (vec![n as u32; n + 1], vec![n as u32; n + 1], vec![0; n], 0);
        let v: Vec<Arc<(Vec<u8>, usize)>> = self.tiles.drain().map(|x| x.1).collect();
        v.into_iter().for_each(|a| self.recyc(a));
    }
    pub fn cold(&mut self) { self.reset(2) }
    pub fn unmap(&self) { self.adv(0, self.map.len(), libc::MADV_DONTNEED) }
    fn hl(&self, e: usize) -> usize { let x = &self.ents[e]; 8 + 2 * FT + 8 * self.tabs[x.t as usize].tiles(x.p as usize, self.tile) + 4 }
    pub fn pl(&self, e: usize) -> io::Result<Pl> { let x = &self.ents[e]; parse(self.plane(e), self.tabs[x.t as usize].tiles(x.p as usize, self.tile)).map_err(|er| bad(format!("{}: {} level {} page {} d{}: {er}", self.path, self.tabs[x.t as usize].name, x.l, x.p, x.k))) }
    fn span(&self, e: usize, q: &Pl, ti: usize) -> (usize, usize) { (self.ents[e].off as usize + q.off[ti] as usize, (q.off[ti + 1] - q.off[ti]) as usize) }
    pub fn dtile(&self, t: usize, l: usize, p: usize, ti: usize, n: usize, pl: &[Option<Arc<Pl>>], pr: Option<&(Vec<u8>, usize)>, mut o: Vec<u8>) -> io::Result<Vec<u8>> {
        let (ts, m) = (self.tile * self.tile, pr.map_or(0, |x| x.1));
        let npx = self.tile * self.tile.min(self.tabs[t].rows(p) - ti / (PW / self.tile) * self.tile);
        o.clear();
        o.extend_from_slice(pr.map_or(&[][..], |x| &x.0[..]));
        o.resize(4 * ts, 0);
        (4 - n.max(m)..4 - m).rev().try_for_each(|k| {
            let (e, q) = (self.ent(t, l, p, k), pl[k].as_deref().ok_or_else(|| bad("plane table not loaded".into()))?);
            let (a, z) = (q.off[ti] as usize, q.off[ti + 1] as usize);
            let tb = &self.plane(e)[a..z];
            let (lo, hi) = o.split_at_mut((k + 1) * ts);
            (hash(tb) == q.hs[ti] && if k == 3 { rdec::<true>(tb, &[], &q.lut, &mut lo[k * ts..k * ts + npx]) } else { rdec::<false>(tb, &hi[..npx], &q.lut, &mut lo[k * ts..k * ts + npx]) }).then_some(()).ok_or_else(|| bad(format!("{}: {} level {l} page {p} d{k} tile {ti}: checksum or rANS state mismatch", self.path, self.tabs[t].name)))
        })?;
        Ok(o)
    }
    fn getc(&mut self, q: &[(usize, usize, usize)], tiers: usize) -> io::Result<Vec<u16>> {
        (self.pls.len() > self.pcap).then(|| self.pls.clear());
        let (tw, ts) = (self.tile, self.tile * self.tile);
        let lc: Vec<(usize, usize, usize)> = q.iter().map(|&(t, l, w)| { let (px, n) = (w % PX, if self.tabs[t].exact { 4 } else { tiers }); ((self.tb[t][l * self.tabs[t].np() + w / PX] + px / PW / tw * (PW / tw) + px % PW / tw), px / PW % tw * tw + px % tw, n) }).collect();
        let mut need: HashMap<usize, (usize, usize, usize, usize, usize)> = HashMap::new();
        q.iter().zip(&lc).for_each(|(&(t, l, w), &(g, _, n))| { let p = w / PX; let ti = g - self.tb[t][l * self.tabs[t].np() + p]; need.entry(g).and_modify(|v| v.4 = v.4.max(n)).or_insert((t, l, p, ti, n)); });
        let (mut have, mut todo) = (HashMap::new(), vec![]);
        for (&g, &(t, l, p, ti, n)) in &need { self.lru(g, 4 * ts); let c = self.tiles.get(&g).cloned(); match c { Some(a) if a.1 >= n => { have.insert(g, a); } _ => todo.push((g, t, l, p, ti, n, c)) } }
        let es: Vec<usize> = { let mut v: Vec<usize> = todo.iter().flat_map(|&(_, t, l, p, _, n, ref c)| (4 - n..4 - c.as_ref().map_or(0, |x| x.1.min(n))).map(move |k| (t, l, p, k))).map(|(t, l, p, k)| self.ent(t, l, p, k)).filter(|e| !self.pls.contains_key(e)).collect(); v.sort_unstable(); v.dedup(); v };
        es.iter().for_each(|&e| self.adv(self.ents[e].off as usize, self.hl(e), libc::MADV_WILLNEED));
        let g = |&e: &usize| self.pl(e).map(|p| (e, Arc::new(p)));
        let ps: Vec<(usize, Arc<Pl>)> = if es.len() > 8 { es.par_iter().map(g).collect::<io::Result<_>>()? } else { es.iter().map(g).collect::<io::Result<_>>()? };
        self.pls.extend(ps);
        let bufs: Vec<Vec<u8>> = todo.iter().map(|_| self.free.pop().unwrap_or_default()).collect();
        let me = &*self;
        let pls: Vec<Vec<Option<Arc<Pl>>>> = todo.iter().map(|&(_, t, l, p, _, n, _)| (0..4).map(|k| (k + n >= 4).then(|| me.pls.get(&me.ent(t, l, p, k)).cloned()).flatten()).collect()).collect();
        let sp: Vec<(usize, usize)> = todo.iter().zip(&pls).flat_map(|(&(_, t, l, p, ti, _, _), pl)| (0..4).filter_map(move |k| pl[k].as_ref().map(|q| (k, q.clone()))).map(move |(k, q)| me.span(me.ent(t, l, p, k), &q, ti)).collect::<Vec<_>>()).collect();
        sp.iter().for_each(|&(o, n)| me.adv(o, n, libc::MADV_WILLNEED));
        let f = |((&(_, t, l, p, ti, n, ref c), pl), b): ((&(usize, usize, usize, usize, usize, usize, Option<Arc<(Vec<u8>, usize)>>), &Vec<Option<Arc<Pl>>>), Vec<u8>)| me.dtile(t, l, p, ti, n, pl, c.as_deref(), b);
        let got: Vec<Vec<u8>> = if todo.len() > 2 { todo.par_iter().zip(&pls).zip(bufs).map(f).collect::<io::Result<_>>()? } else { todo.iter().zip(&pls).zip(bufs).map(f).collect::<io::Result<_>>()? };
        todo.iter().zip(got).for_each(|(&(g, _, _, _, _, n, ref c), d)| { let a = Arc::new((d, n.max(c.as_ref().map_or(0, |x| x.1)))); (self.seen[g] != 0).then(|| self.tiles.insert(g, a.clone())); have.insert(g, a); });
        sp.iter().for_each(|&(o, n)| self.adv(o, n, libc::MADV_DONTNEED));
        Ok(q.iter().zip(&lc).map(|(&(t, _, _), &(g, i, n))| { let d = &have[&g].0; dec(val(std::array::from_fn(|k| if k + n >= 4 { d[k * ts + i] } else { MID }), n), n, self.tabs[t].ord) }).collect())
    }
    pub fn get(&mut self, q: &[(usize, usize, usize)], tiers: usize) -> io::Result<Vec<u16>> {
        if self.tile > 0 { return self.getc(q, tiers); }
        let (mut need, mut pf) = (Vec::with_capacity(q.len() * 4), vec![]);
        for &(t, l, w) in q { let (np, n) = (self.tabs[t].np(), if self.tabs[t].exact { 4 } else { tiers }); for k in 4 - n..4 { let o = self.ents[self.loc[t][(l * np + w / PX) * 4 + k]].off as usize + w % PX; self.touch(o, &mut pf); need.push(o); } }
        pf.iter().for_each(|&o| self.adv(o, PW, libc::MADV_WILLNEED));
        let mut it = need.into_iter();
        Ok(q.iter().map(|&(t, _, _)| { let (n, ord) = (if self.tabs[t].exact { 4 } else { tiers }, self.tabs[t].ord); let d = std::array::from_fn(|k| if k + n >= 4 { self.map[it.next().unwrap()] } else { MID }); dec(val(d, n), n, ord) }).collect())
    }
    pub fn fetch_many(&mut self, q: &[(usize, usize, usize)], tiers: usize) -> io::Result<Vec<(u8, u16)>> {
        let (a, b) = self.cs.ok_or_else(|| bad("store has no sid/sw tables".into()))?;
        let (ca, cb) = (self.tabs[a].cell(), self.tabs[b].cell());
        let w = self.get(&q.iter().flat_map(|&(l, c, s)| [(a, l, (c * ca + s) / 2), (b, l, c * cb + s)]).collect::<Vec<_>>(), tiers)?;
        Ok(q.iter().enumerate().map(|(i, &(_, c, s))| ((w[2 * i] >> (8 * ((c * ca + s) % 2))) as u8, w[2 * i + 1])).collect())
    }
    pub fn fetch(&mut self, l: usize, c: usize, s: usize) -> io::Result<(u8, u16)> { Ok(self.fetch_many(&[(l, c, s)], 4)?[0]) }
    fn planes(&self, t: usize, l: usize, n: usize, a: i32) { (l < self.tabs[t].lv()).then(|| (0..self.tabs[t].np()).for_each(|p| (4 - n..4).for_each(|k| { let e = &self.ents[self.ent(t, l, p, k)]; self.adv(e.off as usize, e.len as usize, a) }))); }
    pub fn pdig(&self, t: usize, l: usize, p: usize, tiers: usize) -> io::Result<Vec<Cow<'_, [u8]>>> {
        let n = if self.tabs[t].exact { 4 } else { tiers };
        if self.tile == 0 { return Ok((0..4).map(|k| Cow::Borrowed(if k + n >= 4 { self.plane(self.ent(t, l, p, k)) } else { &[][..] })).collect()); }
        let (tw, rows) = (self.tile, self.tabs[t].rows(p));
        let pl = self.pls4(t, l, p, n)?;
        let mut o: Vec<Vec<u8>> = (0..4).map(|k| vec![0u8; if k + n >= 4 { rows * PW } else { 0 }]).collect();
        let ptr: Vec<Sp> = o.iter_mut().map(|v| Sp(v.as_mut_ptr())).collect();
        rects(rows, tw).into_par_iter().enumerate().try_for_each(|(ti, (x, y, h))| { let d = self.dtile(t, l, p, ti, n, &pl, None, vec![])?; (4 - n..4).for_each(|k| (0..h).for_each(|r| unsafe { std::ptr::copy_nonoverlapping(d.as_ptr().add(k * tw * tw + r * tw), ptr[k].at((y + r) * PW + x), tw) })); Ok::<(), io::Error>(()) })?;
        Ok(o.into_iter().map(Cow::Owned).collect())
    }
    fn pls4(&self, t: usize, l: usize, p: usize, n: usize) -> io::Result<Vec<Option<Arc<Pl>>>> { (0..4).map(|k| (k + n >= 4).then(|| self.pl(self.ent(t, l, p, k)).map(Arc::new)).transpose()).collect() }
    pub fn level(&self, t: usize, l: usize, tiers: usize, out: &mut [u8]) -> io::Result<usize> {
        let tb = &self.tabs[t];
        let (n, ob, op) = (if tb.exact { 4 } else { tiers }, out.len(), Sp(out.as_mut_ptr()));
        (l == 0).then(|| self.planes(t, l, n, libc::MADV_WILLNEED));
        self.planes(t, l + 1, n, libc::MADV_WILLNEED);
        let z = (0..tb.np()).map(|p| {
            let (w0, pb) = (p * PX, (2 * (p * PX + PX)).min(ob));
            if self.tile == 0 { let d = self.pdig(t, l, p, n)?; return Ok((2 * w0..pb).into_par_iter().step_by(2 * CH).map(|c| { let (a, m) = (c / 2 - w0, (pb - c).min(2 * CH)); asm(std::array::from_fn(|k| if k + n >= 4 { &d[k][a..a + m.div_ceil(2)] } else { &ZP[..m.div_ceil(2)] }), n, tb, unsafe { std::slice::from_raw_parts_mut(op.at(c), m) }) }).sum::<usize>()); }
            let (pl, tw) = (self.pls4(t, l, p, n)?, self.tile);
            rects(tb.rows(p), tw).into_par_iter().enumerate().map(|(ti, (x, y, h))| { let d = self.dtile(t, l, p, ti, n, &pl, None, vec![])?; Ok((0..h).map(|r| (w0 + (y + r) * PW + x, r * tw)).filter(|&(wr, _)| 2 * wr < ob).map(|(wr, a)| { let m = (2 * tw).min(ob - 2 * wr); asm(std::array::from_fn(|k| &d[k * tw * tw + a..k * tw * tw + a + m.div_ceil(2)]), n, tb, unsafe { std::slice::from_raw_parts_mut(op.at(2 * wr), m) }) }).sum::<usize>()) }).sum::<io::Result<usize>>()
        }).sum::<io::Result<usize>>()?;
        self.planes(t, l, n, libc::MADV_DONTNEED);
        Ok(z)
    }
    pub fn page_words(&self, t: usize, l: usize, p: usize) -> io::Result<Vec<u16>> { let (tb, d) = (&self.tabs[t], self.pdig(t, l, p, 4)?); Ok((0..(tb.words() - p * PX).min(PX)).map(|i| dec(val([d[0][i], d[1][i], d[2][i], d[3][i]], 4), 4, tb.ord)).collect()) }
    pub fn unpack(&self, t: usize, tiers: usize) -> io::Result<(Vec<u8>, usize)> {
        let tb = &self.tabs[t];
        let mut out = vec![0u8; tb.n() * tb.dt];
        let z = out.chunks_mut((tb.per() * tb.dt).max(1)).enumerate().map(|(l, o)| self.level(t, l, tiers, o)).sum::<io::Result<usize>>()?;
        Ok((out, z))
    }
    pub fn rgba(&self, t: usize, l: usize, p: usize) -> io::Result<Vec<u8>> {
        let pl = self.pdig(t, l, p, 4)?;
        let mut o = vec![0u8; pl[0].len() * 4];
        o.par_chunks_mut(4).with_min_len(1 << 14).enumerate().for_each(|(i, c)| (0..4).for_each(|k| c[k] = pl[k][i]));
        Ok(o)
    }
}
