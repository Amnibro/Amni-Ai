use crate::*;
use std::collections::HashMap;
pub const RM: u32 = 83_521;
pub const RLO: u32 = 24_137_569;
pub const NV: u32 = 83_521;
pub const NB: u32 = 512;
pub const TEND: u32 = NV;
pub const ESC: u32 = u32::MAX;
pub trait Dg { fn dg(&self, i: u64) -> u8; }
impl Dg for Pg { #[inline(always)] fn dg(&self, i: u64) -> u8 { self.dig(i) } }
impl Dg for [u8] { #[inline(always)] fn dg(&self, i: u64) -> u8 { self[i as usize] } }
impl Dg for Vec<u8> { #[inline(always)] fn dg(&self, i: u64) -> u8 { self[i as usize] } }
pub fn ufq(n: u32, r: u32) -> (u32, u32) { let c = |i: u32| (i as u64 * RM as u64 / n as u64) as u32; (c(r), c(r + 1) - c(r)) }
#[derive(Default, Clone)]
pub struct Enc { pub s: Vec<(u32, u32)> }
impl Enc {
    #[inline(always)]
    pub fn put(&mut self, c: u32, f: u32) { self.s.push((c, f)) }
    pub fn uni(&mut self, n: u64, r: u64) { let (mut n, mut r) = (n, r); while n > 289 { self.put((r % 17) as u32 * 4913, 4913); r /= 17; n = n.div_ceil(17) } if n > 1 { let (c, f) = ufq(n as u32, r as u32); self.put(c, f) } }
    pub fn bits(&self) -> f64 { self.s.iter().map(|&(_, f)| (RM as f64 / f as f64).log2()).sum() }
    pub fn finish(&self) -> Vec<u8> { let (mut x, mut o) = (RLO, vec![]); for &(c, f) in self.s.iter().rev() { let xm = 4913 * f; while x >= xm { o.push((x % 17) as u8); x /= 17 } x = (x / f) * RM + x % f + c } (0..7).for_each(|_| { o.push((x % 17) as u8); x /= 17 }); o.reverse(); o }
}
pub struct RDec<'a, S: Dg + ?Sized> { pub g: &'a S, pub i: u64, pub x: u32 }
impl<'a, S: Dg + ?Sized> RDec<'a, S> {
    pub fn new(g: &'a S, i: u64) -> RDec<'a, S> { let mut d = RDec { g, i, x: 0 }; (0..7).for_each(|_| { d.x = d.x * 17 + d.g.dg(d.i) as u32; d.i += 1 }); d }
    #[inline(always)]
    fn adv(&mut self, c: u32, f: u32, s: u32) { self.x = f * (self.x / RM) + s - c; while self.x < RLO { self.x = self.x * 17 + self.g.dg(self.i) as u32; self.i += 1 } }
    #[inline(always)]
    pub fn get(&mut self, t: &Fq) -> u32 { let s = self.x % RM; let j = t.cum.partition_point(|&q| q <= s) - 1; self.adv(t.cum[j], t.cum[j + 1] - t.cum[j], s); t.sym[j] }
    pub fn uni(&mut self, n: u64) -> u64 {
        let (mut n, mut r, mut m) = (n, 0u64, 1u64);
        while n > 289 { let s = self.x % RM; let d = s / 4913; self.adv(d * 4913, 4913, s); r += d as u64 * m; m *= 17; n = n.div_ceil(17) }
        if n > 1 { let s = self.x % RM; let mut i = (s as u64 * n / RM as u64) as u32; while ufq(n as u32, i).0 > s { i -= 1 } while i + 1 < n as u32 && ufq(n as u32, i + 1).0 <= s { i += 1 } let (c, f) = ufq(n as u32, i); self.adv(c, f, s); r += i as u64 * m }
        r
    }
}
#[derive(Clone, Default, Debug)]
pub struct Fq { pub sym: Vec<u32>, pub cum: Vec<u32> }
impl Fq {
    pub fn new(c: &[(u32, u64)]) -> Fq {
        let mut v: Vec<(u32, u64)> = c.iter().copied().filter(|x| x.1 > 0).collect();
        v.sort_unstable();
        assert!(!v.is_empty() && v.len() <= RM as usize / 2, "fq alphabet {}", v.len());
        let tot: u64 = v.iter().map(|x| x.1).sum();
        let mut q: Vec<i64> = v.iter().map(|x| ((x.1 as u128 * RM as u128 + tot as u128 / 2) / tot as u128).max(1) as i64).collect();
        let mut d: i64 = RM as i64 - q.iter().sum::<i64>();
        while d != 0 { let j = (0..q.len()).filter(|&i| d > 0 || q[i] > 1).max_by_key(|&i| q[i]).unwrap(); let s = if d > 0 { d } else { -((-d).min(q[j] - 1)) }; q[j] += s; d -= s }
        let mut cum = vec![0u32];
        q.iter().for_each(|&f| cum.push(cum.last().unwrap() + f as u32));
        Fq { sym: v.iter().map(|x| x.0).collect(), cum }
    }
    #[inline(always)]
    pub fn at(&self, s: u32) -> Option<(u32, u32)> { self.sym.binary_search(&s).ok().map(|j| (self.cum[j], self.cum[j + 1] - self.cum[j])) }
    pub fn put(&self, e: &mut Enc, s: u32) -> bool { self.at(s).map(|(c, f)| e.put(c, f)).is_some() }
    pub fn cost(&self, s: u32) -> Option<f64> { self.at(s).map(|(_, f)| (RM as f64 / f as f64).log2()) }
    pub fn ser(&self, o: &mut Vec<u8>) { ev(o, self.sym.len() as u64 - 1); let mut v: Vec<(u64, u32)> = self.sym.iter().enumerate().map(|(j, &s)| (if s == ESC { 0 } else { s as u64 + 1 }, self.cum[j + 1] - self.cum[j])).collect(); v.sort_unstable(); let mut p = 0u64; v.iter().for_each(|&(s, f)| { ev(o, s - p); p = s; ev(o, f as u64 - 1) }) }
    pub fn de(r: &mut Rd) -> Fq { let n = r.ev() + 1; let (mut p, mut sym, mut cum) = (0u64, vec![], vec![0u32]); (0..n).for_each(|_| { p += r.ev(); sym.push(if p == 0 { ESC } else { (p - 1) as u32 }); cum.push(cum.last().unwrap() + r.ev() as u32 + 1) }); let mut o: Vec<(u32, u32)> = sym.into_iter().zip(cum.windows(2).map(|w| w[1] - w[0])).collect(); o.sort_unstable_by_key(|x| x.0); let mut c = vec![0u32]; o.iter().for_each(|x| c.push(c.last().unwrap() + x.1)); Fq { sym: o.iter().map(|x| x.0).collect(), cum: c } }
}
pub fn hbits(c: &[u64]) -> f64 { let t: u64 = c.iter().sum(); c.iter().filter(|&&x| x > 0).map(|&x| x as f64 * (t as f64 / x as f64).log2()).sum() }
pub struct Bpe { pub ids: HashMap<Vec<u8>, u32, FxB>, pub rank: HashMap<(u32, u32), (u32, u32), FxB>, pub sp: Vec<Vec<u8>>, pub fin: Vec<bool> }
impl Bpe {
    pub fn load(dir: &str) -> io::Result<Bpe> {
        let t = fs::read_to_string(format!("{dir}/pieces.txt"))?;
        let (mut ids, mut sp, mut fin): (HashMap<Vec<u8>, u32, FxB>, Vec<Vec<u8>>, Vec<bool>) = (HashMap::default(), (0..NB).map(|i| vec![(i & 255) as u8]).collect(), (0..NB).map(|i| i >= 256).collect());
        t.lines().enumerate().for_each(|(i, l)| { ids.insert(l.as_bytes().to_vec(), NB + i as u32); let f = l.ends_with("</w>"); sp.push(l.as_bytes()[..l.len() - if f { 4 } else { 0 }].to_vec()); fin.push(f) });
        let m = fs::read_to_string(format!("{dir}/merges.txt"))?;
        let rank = m.lines().enumerate().filter_map(|(r, l)| { let (a, b) = l.split_once(' ')?; let ab = [a.as_bytes(), b.as_bytes()].concat(); Some(((*ids.get(a.as_bytes())?, *ids.get(b.as_bytes())?), (r as u32, *ids.get(&ab)?))) }).collect();
        Ok(Bpe { ids, rank, sp, fin })
    }
    pub fn word(&self, w: &str) -> Vec<u32> {
        let ch: Vec<(usize, char)> = w.char_indices().collect();
        let mut s: Vec<u32> = vec![];
        ch.iter().enumerate().for_each(|(k, &(o, c))| { let last = k + 1 == ch.len(); let b = &w.as_bytes()[o..o + c.len_utf8()]; let key = if last { [b, b"</w>".as_slice()].concat() } else { b.to_vec() }; match self.ids.get(&key) { Some(&i) => s.push(i), None => b.iter().enumerate().for_each(|(j, &x)| s.push(x as u32 + if last && j + 1 == b.len() { 256 } else { 0 })) } });
        loop { let best = s.windows(2).enumerate().filter_map(|(i, p)| self.rank.get(&(p[0], p[1])).map(|&(r, m)| (r, i, m))).min(); match best { Some((_, i, m)) => { s[i] = m; s.remove(i + 1); } None => break } }
        s
    }
    pub fn bytes(&self, t: &[u8]) -> Vec<u32> { t.iter().enumerate().map(|(j, &x)| x as u32 + if j + 1 == t.len() { 256 } else { 0 }).collect() }
    pub fn text(&self, t: &str) -> Vec<u32> { t.split(' ').flat_map(|w| self.word(w)).collect() }
    pub fn spell(&self, p: &[u32]) -> Vec<u8> { let mut o = vec![]; p.iter().for_each(|&i| { o.extend_from_slice(&self.sp[i as usize]); self.fin[i as usize].then(|| o.push(b' ')); }); o.pop(); o }
}
pub fn caseop(w: &str, lw: &str) -> u8 {
    if w == lw { return 0 }
    let mut c = lw.chars();
    let cap: String = c.next().map_or(String::new(), |f| f.to_uppercase().chain(c).collect());
    if cap == w { return 1 }
    if lw.to_uppercase() == w { return 2 }
    let (a, b): (Vec<char>, Vec<char>) = (w.chars().collect(), lw.chars().collect());
    if a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x == y || x.to_lowercase().eq([*y]) && y.to_uppercase().eq([*x])) { 3 } else { 5 }
}
pub fn casemask(w: &str, lw: &str) -> Vec<u8> { w.chars().zip(lw.chars()).filter(|(_, y)| y.to_uppercase().ne([*y])).map(|(x, y)| (x != y) as u8).collect() }
pub fn segbits(w: &str, lw: &str) -> Option<Vec<u8>> { let (a, b): (Vec<char>, Vec<char>) = (w.chars().collect(), lw.chars().collect()); if a.len() != b.len() { return None } let (mut o, mut pa) = (vec![], false); for k in 0..b.len() { let al = b[k].is_alphabetic(); let st = al && !pa; pa = al; if st { o.push((a[k] != b[k]) as u8) } else if a[k] != b[k] { return None } } Some(o) }
pub fn nmask(lw: &str, op: u8) -> usize { if op == 3 { let mut pa = false; lw.chars().filter(|c| { let al = c.is_alphabetic(); let st = al && !pa; pa = al; st }).count() } else { lw.chars().filter(|y| y.to_uppercase().ne([*y])).count() } }
pub fn uncase(lw: &str, op: u8, mask: &[u8]) -> String {
    let up = |y: char| y.to_uppercase().next().unwrap();
    match op { 0 => lw.into(), 1 => { let mut c = lw.chars(); c.next().map_or(String::new(), |f| f.to_uppercase().chain(c).collect()) } 2 => lw.to_uppercase(), 3 => { let (mut k, mut pa) = (0, false); lw.chars().map(|y| { let al = y.is_alphabetic(); let st = al && !pa; pa = al; if st { k += 1; if mask[k - 1] == 1 { up(y) } else { y } } else { y } }).collect() } _ => { let mut k = 0; lw.chars().map(|y| if y.to_uppercase().ne([y]) { k += 1; if mask[k - 1] == 1 { up(y) } else { y } } else { y }).collect() } }
}
#[derive(Clone, Debug, PartialEq)]
pub enum Tw { P(Vec<u32>, u8, Vec<u8>), Raw(Vec<u32>) }
pub fn cased(b: &Bpe, t: &[u8]) -> Option<Vec<Tw>> {
    let s = std::str::from_utf8(t).ok()?;
    if s.is_empty() || s.split(' ').any(|w| w.is_empty()) { return None }
    Some(s.split(' ').map(|w| { let lw = w.to_lowercase(); let op = caseop(w, &lw); if op == 5 { return Tw::Raw(b.word(w)) } let (op, m) = if op == 3 { segbits(w, &lw).map_or_else(|| (4, casemask(w, &lw)), |m| (3, m)) } else { (op, vec![]) }; let p = b.word(&lw); if uncase(&String::from_utf8_lossy(&b.spell(&p)), op, &m) == w { Tw::P(p, op, m) } else { Tw::Raw(b.word(w)) } }).collect())
}
pub fn uncased(b: &Bpe, ws: &[Tw]) -> Vec<u8> { ws.iter().map(|x| match x { Tw::P(p, op, m) => uncase(&String::from_utf8_lossy(&b.spell(p)), *op, m).into_bytes(), Tw::Raw(p) => b.spell(p) }).collect::<Vec<_>>().join(&b' ') }
#[derive(Clone, Debug, PartialEq, Default)]
pub enum Tf { #[default] E, W(Vec<Tw>), R(Vec<u32>) }
pub fn tf(b: &Bpe, t: &[u8], case: bool) -> Tf { if t.is_empty() { return Tf::E } let s = std::str::from_utf8(t).ok().filter(|s| !s.split(' ').any(|w| w.is_empty())); match (s, case) { (Some(_), true) => cased(b, t).map_or_else(|| Tf::R(b.bytes(t)), Tf::W), (Some(s), false) => Tf::W(s.split(' ').map(|w| Tw::Raw(b.word(w))).collect()), _ => Tf::R(b.bytes(t)) } }
pub fn untf(b: &Bpe, x: &Tf) -> Vec<u8> { match x { Tf::E => vec![], Tf::W(w) => uncased(b, w), Tf::R(p) => b.spell(p) } }
pub fn tfp(x: &Tf) -> Vec<u32> { match x { Tf::E => vec![], Tf::W(w) => w.iter().flat_map(|t| match t { Tw::P(p, _, _) | Tw::Raw(p) => p.clone() }).collect(), Tf::R(p) => p.clone() } }
pub fn tfm(x: &Tf) -> u32 { match x { Tf::E => 0, Tf::W(_) => 1, Tf::R(_) => 2 } }
pub fn splitw(b: &Bpe, p: &[u32]) -> Vec<Vec<u32>> { let mut o: Vec<Vec<u32>> = vec![vec![]]; p.iter().for_each(|&x| { o.last_mut().unwrap().push(x); b.fin[x as usize].then(|| o.push(vec![])); }); o.pop(); o }
pub const M_QG: u8 = 0;
pub const M_SL: u8 = 1;
pub const M_SIGS: u8 = 2;
pub const M_SIGP: u8 = 3;
pub const M_CNT: u8 = 4;
pub const M_CNTB: u8 = 5;
pub const M_AUX: u8 = 6;
pub const M_SLOT: u8 = 7;
pub const M_NEAR: u8 = 8;
pub const M_BKT: u8 = 9;
pub const M_CLS: u8 = 10;
pub const M_TK: u8 = 11;
pub const M_LM: u8 = 12;
pub const M_P1: u8 = 13;
pub const M_P0H: u8 = 14;
pub const M_P0L: u8 = 15;
pub const M_CASE: u8 = 16;
pub const M_SEG: u8 = 17;
pub const M_CM: u8 = 18;
pub const M_QP: u8 = 19;
pub const M_QK: u8 = 20;
pub const M_QB: u8 = 21;
pub const M_NK: u8 = 22;
pub const M_ACP: u8 = 23;
pub const M_AM: u8 = 24;
pub const M_CBJ: u8 = 25;
pub const M_CBD: u8 = 26;
pub const M_CBR: u8 = 27;
pub const CB2: usize = 64;
pub const FL: u64 = 0;
pub const FT: u64 = 1;
pub const FA: u64 = 2;
pub const FS: u64 = 3;
pub const PSTART: u32 = 0xFFFFF;
pub const S_RULE: u8 = 0;
pub const S_CTX0: u8 = 9;
pub const S_NEAR: u8 = 17;
pub const S_NEWS: u8 = 18;
pub const S_CLS: u8 = 19;
pub const S_ESC: u8 = 20;
pub const MB: [u8; 9] = [0, 1, 2, 3, 3, 4, 4, 4, 4];
pub type Tabs = HashMap<(u8, u64), Fq, FxB>;
pub type Cm = HashMap<(u8, u64), HashMap<u32, u64, FxB>, FxB>;
pub trait Cd { fn cat(&mut self, _c: usize) {} fn dec(&self) -> bool; fn s(&mut self, m: u8, c: u64, x: u32) -> u32; fn u(&mut self, n: u64, x: u64) -> u64; fn has(&self, m: u8, c: u64) -> bool; fn hasx(&self, m: u8, c: u64, x: u32) -> bool; fn so(&mut self, m: u8, c: u64, x: u32) -> Option<u32> { if self.has(m, c) { let y = if self.dec() || self.hasx(m, c, x) { x } else { ESC }; Some(self.s(m, c, y)) } else { None } } }
pub struct Cnt<'a> { pub c: Cm, pub k: Option<&'a Tabs> }
impl Cd for Cnt<'_> {
    fn dec(&self) -> bool { false }
    fn s(&mut self, m: u8, c: u64, x: u32) -> u32 { if self.k.is_none() == (m == M_P1) { *self.c.entry((m, c)).or_default().entry(x).or_default() += 1 } x }
    fn u(&mut self, _n: u64, x: u64) -> u64 { x }
    fn has(&self, m: u8, c: u64) -> bool { self.k.map_or(true, |k| k.contains_key(&(m, c))) }
    fn hasx(&self, m: u8, c: u64, x: u32) -> bool { self.k.map_or(true, |k| k.get(&(m, c)).is_some_and(|t| t.at(x).is_some())) }
}
pub fn cmerge(mut a: Cm, b: Cm) -> Cm { b.into_iter().for_each(|(k, m)| { let e = a.entry(k).or_default(); m.into_iter().for_each(|(x, c)| *e.entry(x).or_default() += c) }); a }
pub struct EncC<'a> { pub t: &'a Tabs, pub e: Enc, pub c: usize, pub bits: [f64; 8] }
impl Cd for EncC<'_> {
    fn cat(&mut self, c: usize) { self.c = c }
    fn dec(&self) -> bool { false }
    fn s(&mut self, m: u8, c: u64, x: u32) -> u32 { let n0 = self.e.s.len(); assert!(self.t.get(&(m, c)).is_some_and(|t| t.put(&mut self.e, x)), "enc miss m{m} c{c} x{x}"); self.bits[self.c] += self.e.s[n0..].iter().map(|&(_, f)| (RM as f64 / f as f64).log2()).sum::<f64>(); x }
    fn u(&mut self, n: u64, x: u64) -> u64 { let n0 = self.e.s.len(); self.e.uni(n, x); self.bits[self.c] += self.e.s[n0..].iter().map(|&(_, f)| (RM as f64 / f as f64).log2()).sum::<f64>(); x }
    fn has(&self, m: u8, c: u64) -> bool { self.t.contains_key(&(m, c)) }
    fn hasx(&self, m: u8, c: u64, x: u32) -> bool { self.t.get(&(m, c)).is_some_and(|t| t.at(x).is_some()) }
}
pub struct DecC<'a, S: Dg + ?Sized> { pub t: &'a Tabs, pub d: RDec<'a, S> }
impl<S: Dg + ?Sized> Cd for DecC<'_, S> {
    fn dec(&self) -> bool { true }
    fn s(&mut self, m: u8, c: u64, _x: u32) -> u32 { self.d.get(&self.t[&(m, c)]) }
    fn u(&mut self, n: u64, _x: u64) -> u64 { self.d.uni(n) }
    fn has(&self, m: u8, c: u64) -> bool { self.t.contains_key(&(m, c)) }
    fn hasx(&self, _m: u8, _c: u64, _x: u32) -> bool { true }
    fn so(&mut self, m: u8, c: u64, _x: u32) -> Option<u32> { let t = self.t; t.get(&(m, c)).map(|q| self.d.get(q)) }
}
#[inline(always)]
pub fn bkt(r: u64) -> (u32, u64) { let b = 63 - (r + 1).leading_zeros(); (b, r + 1 - (1u64 << b)) }
pub fn cbk<C: Cd>(cd: &mut C, m: u8, c: u64, r: u64) -> u64 { let (b, o) = if cd.dec() { (0, 0) } else { bkt(r) }; let b = cd.s(m, c, b); let o = cd.u(1u64 << b, o); (1u64 << b) - 1 + o }
pub fn pc<C: Cd>(cd: &mut C, f: u64, prev: u32, x: u32) -> u32 {
    let c1 = f << 20 | prev as u64;
    if let Some(y) = cd.so(M_P1, c1, x) { if y != ESC { return y } }
    let h = cd.s(M_P0H, f, x / 289);
    h * 289 + cd.s(M_P0L, f << 12 | h as u64, x % 289)
}
pub fn pseq<C: Cd>(cd: &mut C, f: u64, xs: &mut Vec<u32>) { let (d, n) = (cd.dec(), xs.len()); let mut prev = PSTART; let mut i = 0; loop { let y = pc(cd, f, prev, if d { 0 } else if i < n { xs[i] } else { TEND }); if y == TEND { break } if d { xs.push(y) } prev = y; i += 1 } }
pub fn pword<C: Cd>(cd: &mut C, b: &Bpe, f: u64, xs: &mut Vec<u32>) { let d = cd.dec(); let mut prev = PSTART; let mut i = 0; loop { let y = pc(cd, f, prev, if d { 0 } else { xs[i] }); if d { xs.push(y) } prev = y; i += 1; if b.fin[y as usize] { break } } }
pub fn ctf<C: Cd>(cd: &mut C, b: &Bpe, pcls: &[u8], f: u64, x: &mut Tf) {
    let d = cd.dec();
    let m = cd.s(M_LM, f, if d { 0 } else { tfm(x) });
    if m == 0 { if d { *x = Tf::E } return }
    let mut p = if d { vec![] } else { tfp(x) };
    pseq(cd, f, &mut p);
    if m == 2 { if d { *x = Tf::R(p) } return }
    let ws = splitw(b, &p);
    let src: Vec<(u32, Vec<u8>)> = if d { vec![] } else if let Tf::W(w) = x { w.iter().map(|t| match t { Tw::P(_, o, m) => (*o as u32, m.clone()), Tw::Raw(_) => (5, vec![]) }).collect() } else { vec![] };
    let (mut out, mut po) = (vec![], 6u32);
    for (k, wp) in ws.into_iter().enumerate() {
        let op = cd.s(M_CASE, f << 16 | ((k == 0) as u64) << 12 | (po as u64) << 8 | pcls[wp[0] as usize] as u64, if d { 0 } else { src[k].0 });
        let mut mk = if d { vec![] } else { src[k].1.clone() };
        if op == 3 || op == 4 { let nb = nmask(&String::from_utf8_lossy(&b.spell(&wp)), op as u8); if d { mk = vec![0; nb] } let mut pb = 0u64; for q in 0..nb { mk[q] = cd.s(if op == 3 { M_SEG } else { M_CM }, f << 8 | pb << 1 | (q == 0) as u64, mk[q] as u32) as u8; pb = mk[q] as u64 } }
        out.push(if op == 5 { Tw::Raw(wp) } else { Tw::P(wp, op as u8, mk) });
        po = op;
    }
    if d { *x = Tf::W(out) }
}
pub fn calias<C: Cd>(cd: &mut C, b: &Bpe, cand: &mut Vec<Vec<u32>>, x: &mut Tf) {
    let d = cd.dec();
    let m = cd.s(M_AM, 0, if d { 0 } else { tfm(x) });
    if m == 2 { let mut p = if d { vec![] } else { tfp(x) }; pseq(cd, FA | 4, &mut p); if d { *x = Tf::R(p) } return }
    let ws: Vec<Vec<u32>> = if d { vec![] } else if let Tf::W(w) = x { w.iter().map(|t| match t { Tw::P(p, _, _) | Tw::Raw(p) => p.clone() }).collect() } else { vec![] };
    let mut out = vec![];
    loop {
        let nc = cand.len().min(16);
        let k = out.len();
        let s0 = if d { 0 } else { ws.get(k).map_or(0, |w| cand[..nc].iter().position(|c| c == w).map_or(1, |j| 2 + j as u32)) };
        let s = cd.s(M_ACP, (nc as u64) << 2 | ((k == 0) as u64) << 1, s0);
        if s == 0 { break }
        let wp = if s == 1 { let mut p = if d { vec![] } else { ws[k].clone() }; pword(cd, b, FA, &mut p); p } else { cand[(s - 2) as usize].clone() };
        (!cand.contains(&wp)).then(|| cand.push(wp.clone()));
        out.push(wp);
    }
    if d { *x = Tf::W(out.into_iter().map(Tw::Raw).collect()) }
}
pub fn lwords(b: &Bpe, lab: &[u8]) -> Vec<Vec<u32>> { let mut o: Vec<Vec<u32>> = vec![]; String::from_utf8_lossy(lab).to_lowercase().split(' ').filter(|w| !w.is_empty()).for_each(|w| { let p = b.word(w); (!o.contains(&p)).then(|| o.push(p)); }); o }
pub fn cnames<C: Cd>(cd: &mut C, b: &Bpe, mut cand: Vec<Vec<u32>>, r: &mut Vec<(u8, Tf)>) {
    let d = cd.dec();
    let mut pk = 4u32;
    for k in 0.. { let kk = cd.s(M_NK, (k.min(3) as u64) << 4 | pk as u64, if d { 0 } else { r.get(k).map_or(3, |x| x.0 as u32) }); if kk == 3 { break } if d { r.push((kk as u8, Tf::E)) } if kk == 2 { calias(cd, b, &mut cand, &mut r[k].1) } pk = kk }
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Fs { pub s: u8, pub aux: u8, pub near: i8, pub cls: u32, pub pos: u64 }
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Er { pub gap: u64, pub sl: u64, pub g: Vec<(u16, u32)>, pub f: Vec<Vec<Fs>>, pub tk: u8, pub lab: Tf, pub tx: Tf }
#[derive(Clone, Debug, Default)]
pub struct Pm { pub pnum: Vec<u16>, pub pkind: Vec<u8>, pub cls: Vec<bool>, pub rules: Vec<Vec<(u16, u16)>>, pub cpi: Vec<u16>, pub p31: u16, pub nstr: u64, pub qkind: Vec<u8> }
pub struct Bst { pub cnt: Vec<u16>, pub sigs: Vec<Vec<(u16, u32)>>, pub pg: u32, pub ps: u32 }
impl Bst { pub fn new(np: usize) -> Bst { Bst { cnt: vec![0; np], sigs: vec![], pg: 0, ps: 0 } } }
pub fn crec<C: Cd>(cd: &mut C, pm: &Pm, bs: &mut Bst, r: &mut Er) {
    let d = cd.dec();
    cd.cat(1);
    let np = bs.sigs.len().min(4);
    let sj0 = if d { 0 } else { bs.sigs.iter().rev().take(4).position(|s| s.len() == r.g.len() && s.iter().zip(&r.g).all(|(a, b)| a.0 == b.0)).map_or(4, |j| j as u32) };
    let sj = cd.s(M_SIGS, np as u64, sj0);
    let refc: Option<Vec<(u16, u32)>> = (sj < 4).then(|| bs.sigs[bs.sigs.len() - 1 - sj as usize].clone());
    if let Some(rc) = &refc { if d { r.g = rc.iter().map(|x| (x.0, 0)).collect() } } else { let mut prev = 0xFFFFu32; let mut k = 0; loop { let x0 = if d { 0 } else { r.g.get(k).map_or(0xFFFF, |x| x.0 as u32) }; let y = cd.s(M_SIGP, prev as u64, x0); if y == 0xFFFF { break } if d { r.g.push((y as u16, 0)) } prev = y; k += 1 } }
    for k in 0..r.g.len() { let pcn = refc.as_ref().map_or(0, |v| v[k].1.min(15)); let n0 = r.g[k].1; let n = cd.s(M_CNT, (r.g[k].0 as u64) << 4 | pcn as u64, n0.min(16)); r.g[k].1 = if n == 16 { 16 + cbk(cd, M_CNTB, r.g[k].0 as u64, n0.saturating_sub(16) as u64) as u32 } else { n } }
    if d { r.f = r.g.iter().map(|x| vec![Fs::default(); x.1 as usize]).collect() }
    cd.cat(2);
    let has = |g: &[(u16, u32)], pi: u16| g.iter().any(|x| x.0 == pi);
    let hc = pm.cpi.iter().any(|&c| has(&r.g, c));
    let h31 = has(&r.g, pm.p31);
    for gi in 0..r.g.len() {
        let (pi, n) = r.g[gi];
        let pu = pi as usize;
        let ra = pm.rules[pu].iter().any(|v| has(&r.g, v.0));
        let mt = (bs.cnt[pu] as usize).min(8);
        let ca = pi == pm.p31 || h31;
        let cl = pm.cls[pu] && hc;
        let mut pa = 0u32;
        for i in 0..n as usize {
            let fs = &mut r.f[gi][i];
            fs.aux = cd.s(M_AUX, (pi as u64) << 8 | ((i > 0) as u64) << 4 | pa.min(8) as u64, fs.aux as u32) as u8;
            pa = fs.aux as u32 + 1;
            let mask = ra as u64 | (MB[mt] as u64) << 1 | (ca as u64) << 4 | (cl as u64) << 5;
            fs.s = cd.s(M_SLOT, (pi as u64) << 8 | mask << 1 | (i > 0) as u64, fs.s as u32) as u8;
            match fs.s { S_NEAR => { let x = cd.s(M_NEAR, pi as u64, if fs.near < 0 { (fs.near + 8) as u32 } else { (fs.near + 7).max(0) as u32 }); fs.near = if x < 8 { x as i8 - 8 } else { x as i8 - 7 } } S_CLS => fs.pos = cbk(cd, M_BKT, (pi as u64) << 2 | 1, fs.pos), S_ESC => { if pm.pkind[pu] >= 4 { fs.pos = cd.u(pm.nstr, fs.pos) } else { if pm.cls[pu] { fs.cls = cd.s(M_CLS, pi as u64, fs.cls) } fs.pos = cbk(cd, M_BKT, (pi as u64) << 2 | if pm.cls[pu] { 2 } else { 0 }, fs.pos) } } _ => {} }
        }
    }
    r.g.iter().for_each(|x| bs.cnt[x.0 as usize] = bs.cnt[x.0 as usize].saturating_add(x.1.min(60000) as u16));
    bs.sigs.push(r.g.clone());
}
pub fn chdr<C: Cd>(cd: &mut C, bs: &mut Bst, r: &mut Er, first: bool) { cd.cat(0); if !first { r.gap = cbk(cd, M_QG, bs.pg as u64, r.gap); bs.pg = bkt(r.gap).0.min(15) } r.sl = cbk(cd, M_SL, bs.ps as u64, r.sl); bs.ps = bkt(r.sl).0.min(15) }
pub fn clab<C: Cd>(cd: &mut C, b: &Bpe, pcls: &[u8], r: &mut Er) { cd.cat(3); ctf(cd, b, pcls, FL, &mut r.lab); cd.cat(4); r.tk = cd.s(M_TK, tfm(&r.lab) as u64, r.tk as u32) as u8; if r.tk >= 2 { ctf(cd, b, pcls, FT, &mut r.tx) } else if cd.dec() { r.tx = Tf::E } }
pub type Qr = Vec<Vec<Vec<(u16, u64)>>>;
pub fn cqual<C: Cd>(cd: &mut C, pm: &Pm, r: &Er, q: &mut Qr) {
    let d = cd.dec();
    if d { *q = r.f.iter().map(|v| v.iter().map(|_| vec![]).collect()).collect() }
    for gi in 0..r.g.len() { let pi = r.g[gi].0 as u64; for i in 0..r.f[gi].len() { let nq = (r.f[gi][i].aux >> 1) as usize; let mut pq = 0xFFFFu32; for j in 0..nq { let (q0, v0) = if d { (0, 0) } else { q[gi][i][j] }; let qp = cd.s(M_QP, pi << 16 | pq as u64, q0 as u32); let v = if pm.qkind[qp as usize] >= 4 { cd.u(pm.nstr, v0) } else { cbk(cd, M_QB, qp as u64, v0) }; if d { q[gi][i].push((qp as u16, v)) } pq = qp } } }
}
pub fn kfreq(m: &HashMap<u32, u64, FxB>) -> Vec<(u32, u64)> { m.iter().map(|(&k, &v)| (k, v)).collect() }
pub fn tabs(c: &Cm, keep: Option<&Tabs>) -> Tabs { let mut t: Tabs = c.par_iter().filter(|(k, _)| k.0 != M_P1).map(|(k, m)| (*k, Fq::new(&kfreq(m)))).collect(); keep.map(|k| k.iter().filter(|(x, _)| x.0 == M_P1).for_each(|(x, f)| { t.insert(*x, f.clone()); })); t }
pub fn p1tabs(c: &Cm, t: u64, cap: usize) -> Tabs { c.par_iter().filter(|(k, _)| k.0 == M_P1).filter_map(|(k, m)| { let mut v = kfreq(m); v.sort_unstable_by(|a, b| b.1.cmp(&a.1)); let tot: u64 = v.iter().map(|x| x.1).sum(); let tt = t.max(v.get(cap).map_or(0, |x| x.1 + 1)); let kept: Vec<(u32, u64)> = v.into_iter().filter(|x| x.1 >= tt).collect(); if kept.is_empty() { return None } let ks: u64 = kept.iter().map(|x| x.1).sum(); let mut f = kept; (tot > ks).then(|| f.push((ESC, tot - ks))); Some((*k, Fq::new(&f))) }).collect() }
pub fn tser(t: &Tabs) -> Vec<u8> { let mut k: Vec<&(u8, u64)> = t.keys().collect(); k.sort_unstable(); let mut o = vec![]; ev(&mut o, k.len() as u64); let mut p = (0u8, 0u64); k.iter().for_each(|&&(m, c)| { ev(&mut o, (m - p.0) as u64); ev(&mut o, if m == p.0 { c - p.1 } else { c }); p = (m, c); t[&(m, c)].ser(&mut o) }); o }
pub fn tde(g: &Pg, o: u64) -> Tabs { let mut r = g.rd(o); let n = r.ev(); let mut t: Tabs = HashMap::with_capacity_and_hasher(n as usize, FxB::default()); let mut p = (0u8, 0u64); for _ in 0..n { let dm = r.ev() as u8; let m = p.0 + dm; let c = if dm == 0 { p.1 + r.ev() } else { r.ev() }; p = (m, c); t.insert((m, c), Fq::de(&mut r)); } t }
pub const SBK: u64 = 16;
pub const RU2: [(u16, u16, u16); 6] = [(17, 131, 17), (17, 276, 17), (17, 159, 17), (27, 19, 17), (27, 20, 17), (20, 19, 0)];
pub const CP2: [u16; 3] = [17, 27, 495];
pub fn sigk(g: &[(u16, u32)]) -> u64 { g.iter().fold(0x9E37_79B9u64, |h, x| (h.rotate_left(7) ^ (x.0 as u64) << 1 ^ (x.1 > 1) as u64).wrapping_mul(0x100_0000_01B3)) & 0xFFFF_FFFF }
pub fn kh(p: &[u32]) -> u64 { let mut h = 0x5170_2000_9E37_79B9u64 ^ p.len() as u64; p.iter().for_each(|&x| { h = (h ^ x as u64).wrapping_mul(0xFF51_AFD7_ED55_8CCD).rotate_left(31) }); h = (h ^ h >> 33).wrapping_mul(0xC4CE_B9FE_1A85_EC53); h ^ h >> 29 }
pub fn bstart(q: u64) -> bool { (q + 1).is_power_of_two() }
pub fn c64<C: Cd>(cd: &mut C, m: u8, c: u64, r: u64) -> u64 { let b = cd.s(m, c, if cd.dec() { 0 } else { 64 - r.leading_zeros() }); if b <= 1 { return b as u64 } (1u64 << (b - 1)) + cd.u(1u64 << (b - 1), if cd.dec() { 0 } else { r - (1u64 << (b - 1)) }) }
pub fn ccb<C: Cd>(cd: &mut C, c: u64, prev: &[u64], t: &mut Vec<u64>, len: usize) { let d = cd.dec(); if d { t.clear(); t.resize(len, 0) } let j = if prev.is_empty() { 0 } else { cd.s(M_CBJ, c << 2 | len as u64, if d { 0 } else { prev.iter().zip(t.iter()).take_while(|(a, b)| a == b).count() as u32 }) as usize }; (0..j).for_each(|k| t[k] = prev[k]); for k in j..len { t[k] = if k == j && !prev.is_empty() { prev[k] + 1 + c64(cd, M_CBD, c << 2 | k as u64, t[k].wrapping_sub(prev[k] + 1)) } else { c64(cd, M_CBR, c << 2 | k as u64, t[k]) } } }
pub fn cbblk<C: Cd>(cd: &mut C, l: &mut [Vec<u64>], q0: usize, c: u64, len: usize) { for q in q0..(q0 + CB2).min(l.len()) { let pv = if q == q0 || bstart(q as u64) { vec![] } else { l[q - 1].clone() }; ccb(cd, c, &pv, &mut l[q], len) } }
pub fn bpe_ser(b: &Bpe, base: u32) -> Vec<u8> { let mut o = vec![]; ev(&mut o, base as u64); ev(&mut o, (b.sp.len() as u32 - NB) as u64); (NB..NB + base).for_each(|i| { let s = &b.sp[i as usize]; ev(&mut o, s.len() as u64 * 2 + b.fin[i as usize] as u64); s.iter().for_each(|&c| ev(&mut o, c as u64)) }); let inv: HashMap<u32, (u32, u32), FxB> = b.rank.iter().map(|(k, v)| (v.1, *k)).collect(); (NB + base..b.sp.len() as u32).for_each(|i| { let (a, c) = inv[&i]; fx(&mut o, a as u64, 4); fx(&mut o, c as u64, 4) }); o }
pub fn bpe_de(g: &Pg, o: u64) -> Bpe {
    let mut r = g.rd(o);
    let (base, n) = (r.ev() as u32, r.ev() as u32);
    let (mut sp, mut fin): (Vec<Vec<u8>>, Vec<bool>) = ((0..NB).map(|i| vec![(i & 255) as u8]).collect(), (0..NB).map(|i| i >= 256).collect());
    let mut ids: HashMap<Vec<u8>, u32, FxB> = HashMap::default();
    let mut rank: HashMap<(u32, u32), (u32, u32), FxB> = HashMap::default();
    let key = |s: &[u8], f: bool| if f { [s, b"</w>".as_slice()].concat() } else { s.to_vec() };
    for i in 0..n { let id = NB + i; if i < base { let l = r.ev(); let s: Vec<u8> = (0..l / 2).map(|_| r.ev() as u8).collect(); ids.insert(key(&s, l & 1 == 1), id); sp.push(s); fin.push(l & 1 == 1) } else { let (a, c) = (r.fx(4) as u32, r.fx(4) as u32); let s = [sp[a as usize].clone(), sp[c as usize].clone()].concat(); let f = fin[c as usize]; ids.insert(key(&s, f), id); rank.insert((a, c), (i - base, id)); sp.push(s); fin.push(f) } }
    Bpe { ids, rank, sp, fin }
}
#[derive(Clone, Debug)]
pub struct Blk { pub e0: u64, pub q0: u64, pub r: Vec<Er>, pub ns: Vec<u32> }
pub struct S2 { pub hcache: std::sync::Mutex<HashMap<u64, std::sync::Arc<Vec<(u64, u64)>>, FxB>>, pub lcache: std::sync::Mutex<HashMap<u64, std::sync::Arc<Vec<Er>>, FxB>>, pub dir: String, pub g: Pg, pub ne: u64, pub nb: u64, pub bc: u64, pub pm: Pm, pub t: Tabs, pub b: Bpe, pub pcls: Vec<u8>, pub ctx: HashMap<(u16, u64), Vec<Vec<u64>>, FxB>, pub cbl: HashMap<(u16, u32), (u64, u64), FxB>, pub cid: HashMap<(u16, u32), u32, FxB>, pub qcb: HashMap<u16, (u64, u64), FxB>, pub pidx: HashMap<u16, u16, FxB>, pub w: [u64; 7], pub nm: Nm, pub nme: Option<Table<16>>, pub fd: usize, pub cache: std::sync::Mutex<HashMap<u64, std::sync::Arc<Blk>, FxB>>, pub hot: HashMap<u32, u64, FxB> }
#[derive(Clone, Debug, PartialEq)]
pub struct Fv { pub kind: u8, pub t: Vec<u64>, pub pref: u8, pub nq: u8 }
impl S2 {
    pub fn open(dir: &str, fd: usize) -> io::Result<S2> {
        let g = Pg::open(&format!("{dir}/store2.gf17pg"))?;
        let mut r = g.rd(g.sec("meta").0);
        let (ne, np, bc, nstr, p31) = (r.ev(), r.ev() as usize, r.ev(), r.ev(), r.ev() as u16);
        let mut pm = Pm { p31, nstr, ..Default::default() };
        for _ in 0..np { pm.pnum.push(r.ev() as u16); pm.pkind.push(r.ev() as u8); pm.cls.push(r.ev() == 1); pm.qkind.push(r.ev() as u8); let nr = r.ev(); pm.rules.push((0..nr).map(|_| (r.ev() as u16, r.ev() as u16)).collect()) }
        let ncp = r.ev();
        pm.cpi = (0..ncp).map(|_| r.ev() as u16).collect();
        let w: [u64; 7] = std::array::from_fn(|_| r.ev());
        let t = tde(&g, g.sec("tabs").0);
        let b = bpe_de(&g, g.sec("bpe").0);
        let mut rp = g.rd(g.sec("pcls").0);
        let pcls: Vec<u8> = (0..NV as usize + 1).map(|_| rp.d()).collect();
        let mut rc = g.rd(g.sec("ctx").0);
        let mut ctx: HashMap<(u16, u64), Vec<Vec<u64>>, FxB> = HashMap::default();
        let nc = rc.ev();
        let mut pk = (0u16, 0u64);
        for _ in 0..nc { let dp = rc.ev() as u16; let pi = pk.0 + dp; let k = if dp == 0 { pk.1 + rc.ev() } else { rc.ev() }; pk = (pi, k); let l = rc.ev(); let v: Vec<Vec<u64>> = (0..l).map(|_| { let m = rc.ev(); (0..m).map(|_| rc.ev()).collect() }).collect(); ctx.insert((pi, k), v); }
        let (mut cbl, mut cid, mut qcb): (HashMap<(u16, u32), (u64, u64), FxB>, HashMap<(u16, u32), u32, FxB>, HashMap<u16, (u64, u64), FxB>) = Default::default();
        let mut rd = g.rd(g.sec("cbd").0);
        let nl = rd.ev();
        let (mut blk, mut pp, mut ci) = (0u64, u16::MAX, 0u32);
        for _ in 0..nl { let pi = rd.ev() as u16; let ce = rd.ev(); let l = rd.ev(); if pi != pp { ci = 0; pp = pi } cbl.insert((pi, ci), (blk, l)); cid.insert((pi, if ce == 0 { u32::MAX } else { ce as u32 - 1 }), ci); ci += 1; blk += l.div_ceil(CB2 as u64) }
        let mut rq = g.rd(g.sec("qcbd").0);
        let nq = rq.ev();
        let mut qb = 0u64;
        for _ in 0..nq { let pi = rq.ev() as u16; let l = rq.ev(); qcb.insert(pi, (qb, l)); qb += l.div_ceil(CB2 as u64) }
        let mut rh = g.rd(g.sec("hotx").0);
        let nh = rh.ev();
        let (mut he, mut hot): (u64, HashMap<u32, u64, FxB>) = (0u64, HashMap::default());
        for _ in 0..nh { he += rh.ev(); let o = rh.ev(); hot.insert(he as u32, o); }
        let hot = { let mut h: HashMap<u32, u64, FxB> = HashMap::default(); let mut acc = 0u64; let mut v: Vec<(u32, u64)> = hot.into_iter().collect(); v.sort_unstable(); v.into_iter().for_each(|(e, o)| { acc += o; h.insert(e, acc - o); }); h };
        let pidx = pm.pnum.iter().enumerate().map(|(i, &p)| (p, i as u16)).collect();
        let ep = format!("{dir}/name_f{fd}_end.gf17rib");
        Ok(S2 { dir: dir.into(), nb: ne.div_ceil(bc), ne, bc, t, b, pcls, ctx, cbl, cid, qcb, pidx, w, nm: Nm::open(&format!("{dir}/name_f{fd}.gf17rib"), fd)?, nme: if fs::metadata(&ep).is_ok() { Some(Table::open(&ep)?) } else { None }, fd, cache: std::sync::Mutex::new(HashMap::default()), lcache: std::sync::Mutex::new(HashMap::default()), hcache: std::sync::Mutex::new(HashMap::default()), pm, g, hot })
    }
    fn xoff(&self, s: &str, b: u64, wr: u64) -> u64 { let c = s == "xc"; let (h, ws) = if c { (20, self.w[0] + self.w[1]) } else { (8, wr) }; let o = self.g.sec(s).0 + b / SBK * (h + (SBK - 1) * ws); let a = self.g.rd(o).fx(8); if b % SBK == 0 { a } else { a + self.g.rd(o + h + (b % SBK - 1) * ws).fx(wr as usize) } }
    fn bq0(&self, b: u64) -> (u64, u64) { let (x, _) = self.g.sec("xc"); let sb = b / SBK; let per = 20 + (SBK - 1) * (self.w[0] + self.w[1]); let o = x + sb * per; let mut r = self.g.rd(o + 8); let (q, s) = (r.fx(7), r.fx(5)); (if b % SBK == 0 { q } else { q + self.g.rd(o + 20 + (b % SBK - 1) * (self.w[0] + self.w[1]) + self.w[0]).fx(self.w[1] as usize) }, s) }
    pub fn block(&self, b: u64) -> std::sync::Arc<Blk> {
        if let Some(x) = self.cache.lock().unwrap().get(&b) { return x.clone() }
        let (q0, _) = self.bq0(b);
        let mut cd = DecC { t: &self.t, d: RDec::new(&self.g, self.g.sec("sc").0 + self.xoff("xc", b, self.w[0])) };
        let mut bs = Bst::new(self.pm.pnum.len());
        let e0 = b * self.bc;
        let r: Vec<Er> = (e0..(e0 + self.bc).min(self.ne)).map(|_| { let mut x = Er::default(); crec(&mut cd, &self.pm, &mut bs, &mut x); x }).collect();
        let ns = r.iter().scan(0u32, |a, x| { let o = *a; *a += x.f.iter().flatten().filter(|f| f.s == S_NEWS).count() as u32; Some(o) }).collect();
        let x = std::sync::Arc::new(Blk { e0, q0, r, ns });
        let mut c = self.cache.lock().unwrap();
        if c.len() > 4096 { c.clear() }
        c.insert(b, x.clone());
        x
    }
    pub fn lblock(&self, b: u64) -> std::sync::Arc<Vec<Er>> { if let Some(x) = self.lcache.lock().unwrap().get(&b) { return x.clone() } let mut cd = DecC { t: &self.t, d: RDec::new(&self.g, self.g.sec("sl").0 + self.xoff("xl", b, self.w[5])) }; let e0 = b * self.bc; let v: Vec<Er> = (e0..(e0 + self.bc).min(self.ne)).map(|_| { let mut x = Er::default(); clab(&mut cd, &self.b, &self.pcls, &mut x); x }).collect(); let x = std::sync::Arc::new(v); let mut c = self.lcache.lock().unwrap(); if c.len() > 4096 { c.clear() } c.insert(b, x.clone()); x }
    pub fn hblock(&self, b: u64) -> std::sync::Arc<Vec<(u64, u64)>> { if let Some(x) = self.hcache.lock().unwrap().get(&b) { return x.clone() } let mut cd = DecC { t: &self.t, d: RDec::new(&self.g, self.g.sec("sh").0 + self.xoff("xh", b, self.w[6])) }; let mut bs = Bst::new(0); let e0 = b * self.bc; let v: Vec<(u64, u64)> = (e0..(e0 + self.bc).min(self.ne)).map(|e| { let mut x = Er::default(); chdr(&mut cd, &mut bs, &mut x, e == e0); (x.gap, x.sl) }).collect(); let x = std::sync::Arc::new(v); let mut c = self.hcache.lock().unwrap(); if c.len() > 65536 { c.clear() } c.insert(b, x.clone()); x }
    pub fn lrec(&self, e: u64) -> (std::sync::Arc<Vec<Er>>, usize) { (self.lblock(e / self.bc), (e % self.bc) as usize) }
    pub fn rec(&self, e: u64) -> (std::sync::Arc<Blk>, usize) { let b = self.block(e / self.bc); (b, (e % self.bc) as usize) }
    pub fn qid(&self, e: u64) -> u64 { let (b, k) = (e / self.bc, (e % self.bc) as usize); let h = self.hblock(b); self.bq0(b).0 + (1..=k).map(|j| h[j].0 + 1).sum::<u64>() }
    pub fn ix(&self, q: u64) -> i64 { let (mut lo, mut hi) = (0u64, self.nb); while hi - lo > 1 { let m = (lo + hi) / 2; if self.bq0(m).0 <= q { lo = m } else { hi = m } } let h = self.hblock(lo); let mut c = self.bq0(lo).0; for k in 0..h.len() { if k > 0 { c += h[k].0 + 1 } if c == q { return (lo * self.bc + k as u64) as i64 } if c > q { break } } -1 }
    pub fn sitelinks(&self, e: u64) -> u64 { self.hblock(e / self.bc)[(e % self.bc) as usize].1 }
    pub fn groups(&self, e: u64) -> Vec<(u16, u32)> { let (b, k) = self.rec(e); b.r[k].g.iter().map(|x| (self.pm.pnum[x.0 as usize], x.1)).collect() }
    pub fn label(&self, e: u64) -> Vec<u8> { let (b, k) = self.lrec(e); untf(&self.b, &b[k].lab) }
    pub fn title(&self, e: u64) -> Vec<u8> { let (b, k) = self.lrec(e); let r = &b[k]; let (l, x) = (untf(&self.b, &r.lab), untf(&self.b, &r.tx)); match r.tk { 0 => vec![], 1 => l, 2 => [l, b" (".to_vec(), x, b")".to_vec()].concat(), _ => x } }
    pub fn hotlabel(&self, e: u64) -> Option<Vec<u8>> { let o = *self.hot.get(&(e as u32))?; let (s, _) = self.g.sec("hot"); let mut r = self.g.rd(s + o); let n = r.ev(); let ws: Vec<Tw> = (0..n).map(|_| { let np = r.ev(); let p: Vec<u32> = (0..np).map(|_| r.fx(4) as u32).collect(); let op = r.d(); let m = if op == 3 || op == 4 { let k = r.ev(); (0..k).map(|_| r.d()).collect() } else { vec![] }; if op == 5 { Tw::Raw(p) } else { Tw::P(p, op, m) } }).collect(); Some(uncased(&self.b, &ws)) }
    pub fn country(&self, e: u64) -> u32 { let (b, k) = self.rec(e); let r = &b.r[k]; self.pm.cpi.iter().find_map(|&c| r.g.iter().position(|x| x.0 == c)).map_or(u32::MAX, |gi| self.fact(e, gi, 0, 0).t[0] as u32) }
    pub fn cbent2(&self, sec: &str, blk0: u64, pos: u64, c: u64, len: usize) -> Vec<u64> { let (o, _) = self.g.sec(sec); let (x, _) = self.g.sec(&format!("{sec}x")); let b = blk0 + pos / CB2 as u64; let mut cd = DecC { t: &self.t, d: RDec::new(&self.g, o + self.g.rd(x + 8 * b).fx(8)) }; let q0 = pos - pos % CB2 as u64; let mut l: Vec<Vec<u64>> = vec![vec![]; (pos - q0 + 1) as usize]; for q in 0..l.len() { let pv = if q == 0 || bstart(q0 + q as u64) { vec![] } else { l[q - 1].clone() }; ccb(&mut cd, c, &pv, &mut l[q], len) } l.pop().unwrap() }
    pub fn fact(&self, e: u64, gi: usize, i: usize, dep: u32) -> Fv {
        assert!(dep < 4096, "resolution depth");
        let (b, k) = self.rec(e);
        let r = &b.r[k];
        let (pi, _) = r.g[gi];
        let pu = pi as usize;
        let f = &r.f[gi][i];
        let kind = self.pm.pkind[pu];
        let mk = |t: Vec<u64>| Fv { kind, t, pref: f.aux & 1, nq: f.aux >> 1 };
        let gp = |pp: u16| r.g.iter().position(|x| x.0 == pp);
        match f.s {
            S_RULE => { let &(via, src) = self.pm.rules[pu].iter().find(|v| gp(v.0).is_some()).unwrap(); let vg = gp(via).unwrap(); if src == u16::MAX { let nv = r.g[vg].1 as usize; mk(self.fact(e, vg, i.min(nv - 1), dep + 1).t) } else { let tt = self.fact(e, vg, 0, dep + 1).t[0]; let (tb, tk) = self.rec(tt); let sg = tb.r[tk].g.iter().position(|x| x.0 == src).unwrap(); let nt = tb.r[tk].g[sg].1 as usize; mk(self.fact(tt, sg, i.min(nt - 1), dep + 1).t) } }
            1..=8 => { let mut j = (f.s - 1) as usize; for kk in (0..k).rev() { if let Some(g2) = b.r[kk].g.iter().position(|x| x.0 == pi) { let n2 = b.r[kk].g[g2].1 as usize; if j < n2 { return mk(self.fact(b.e0 + kk as u64, g2, j, dep + 1).t) } j -= n2 } } unreachable!() }
            9..=16 => { let key = if pi == self.pm.p31 { sigk(&r.g) } else { self.fact(e, gp(self.pm.p31).unwrap(), 0, dep + 1).t[0] + 1 }; mk(self.ctx[&(pi, key)][(f.s - S_CTX0) as usize].clone()) }
            S_NEAR => mk(vec![(e as i64 + f.near as i64) as u64]),
            S_NEWS => { let (_, sb) = self.bq0(e / self.bc); let b0 = (e / self.bc) / SBK * SBK; let before: u64 = (b0..e / self.bc).map(|bb| { let x = self.block(bb); x.r.iter().map(|y| y.f.iter().flatten().filter(|f| f.s == S_NEWS).count() as u64).sum::<u64>() }).sum(); let mut c = b.ns[k] as u64; 'o: for g2 in 0..r.g.len() { for i2 in 0..r.f[g2].len() { if g2 == gi && i2 == i { break 'o } c += (r.f[g2][i2].s == S_NEWS) as u64 } } mk(vec![sb + before + c]) }
            S_CLS => { let ce = self.country(e); let ci = self.cid[&(pi, ce)]; let (b0, _) = self.cbl[&(pi, ci)]; mk(self.cbent2("cb", b0, f.pos, pi as u64, if kind == 3 { 2 } else { 1 })) }
            _ => if kind >= 4 { mk(vec![f.pos]) } else { let ci = if self.pm.cls[pu] { f.cls } else { 0 }; let (b0, _) = self.cbl[&(pi, ci)]; mk(self.cbent2("cb", b0, f.pos, pi as u64, if kind == 3 { 2 } else { 1 })) },
        }
    }
    pub fn values(&self, e: u64, pn: u16) -> Vec<Fv> { let Some(&pi) = self.pidx.get(&pn) else { return vec![] }; let (b, k) = self.rec(e); match b.r[k].g.iter().position(|x| x.0 == pi) { Some(gi) => (0..b.r[k].g[gi].1 as usize).map(|i| self.fact(e, gi, i, 0)).collect(), None => vec![] } }
    pub fn qblock(&self, b: u64) -> Vec<Qr> { let bl = self.block(b); let mut cd = DecC { t: &self.t, d: RDec::new(&self.g, self.g.sec("sq").0 + self.xoff("xq", b, self.w[2])) }; bl.r.iter().map(|r| { let mut q = vec![]; cqual(&mut cd, &self.pm, r, &mut q); q }).collect() }
    pub fn quals(&self, e: u64, pn: u16, i: usize) -> Vec<(u16, u8, Vec<u64>)> { let Some(&pi) = self.pidx.get(&pn) else { return vec![] }; let (b, k) = self.rec(e); let Some(gi) = b.r[k].g.iter().position(|x| x.0 == pi) else { return vec![] }; if b.r[k].f[gi][i].aux >> 1 == 0 { return vec![] } let q = self.qblock(e / self.bc); q[k][gi][i].iter().map(|&(qp, v)| { let qk = self.pm.qkind[qp as usize]; (self.pm.pnum[qp as usize], qk, if qk >= 4 { vec![v] } else { self.cbent2("qcb", self.qcb[&qp].0, v, 1024 + qp as u64, 1) }) }).collect() }
    pub fn nmenc(&self, e: u64) -> Vec<(u8, Vec<u8>)> { let b = e / self.bc; let mut cd = DecC { t: &self.t, d: RDec::new(&self.g, self.g.sec("sn").0 + self.xoff("xn", b, self.w[3])) }; let lb = self.lblock(b); let mut out = vec![]; for (k, r) in lb.iter().enumerate() { let mut n = vec![]; cnames(&mut cd, &self.b, lwords(&self.b, &untf(&self.b, &r.lab)), &mut n); if k as u64 == e % self.bc { out = n.into_iter().map(|(t, x)| (t, untf(&self.b, &x))).collect(); break } } out }
    pub fn string(&self, s: u64) -> Vec<u8> { let b = s / 16; let mut cd = DecC { t: &self.t, d: RDec::new(&self.g, self.g.sec("ss").0 + self.xoff("xs", b, self.w[4])) }; let mut x = Tf::E; for _ in 0..=s % 16 { x = Tf::E; ctf(&mut cd, &self.b, &self.pcls, FS, &mut x) } untf(&self.b, &x) }
    pub fn ents(&self, h: u64) -> Vec<(u32, u8)> { let mut o: Vec<(u32, u8)> = vec![]; for i in 0..1 << 20 { let Some(d) = self.nm.row(mixk(h, i)) else { break }; let e = d[..6].iter().rev().fold(0u32, |a, &x| a * 17 + x as u32); if e == crate::END || e as u64 >= self.ne || o.last().is_some_and(|l| l.0 >= e) || d[6] > 3 || i > 0 && self.nme.as_ref().is_some_and(|t| t.row(mixk(h, i)).is_some()) { break } o.push((e, d[6])) } o }
    pub fn key(&self, s: &str) -> u64 { kh(&self.b.text(s)) }
    pub fn clear(&self) { self.cache.lock().unwrap().clear(); self.lcache.lock().unwrap().clear(); self.hcache.lock().unwrap().clear() }
    pub fn cold(&self) { self.clear(); self.nm.cold(); self.nme.as_ref().map(|t| t.cold()); unsafe { libc::madvise(self.g.map.as_ptr() as *mut _, self.g.map.len(), libc::MADV_DONTNEED) }; if let Ok(f) = File::open(format!("{}/store2.gf17pg", self.dir)) { unsafe { libc::posix_fadvise(std::os::fd::AsRawFd::as_raw_fd(&f), 0, 0, libc::POSIX_FADV_DONTNEED) }; } }
}
impl S2 {
    pub fn nblock(&self, b: u64) -> Vec<Vec<(u8, Vec<u8>)>> { let bl = self.lblock(b); let mut cd = DecC { t: &self.t, d: RDec::new(&self.g, self.g.sec("sn").0 + self.xoff("xn", b, self.w[3])) }; bl.iter().map(|r| { let mut n = vec![]; cnames(&mut cd, &self.b, lwords(&self.b, &untf(&self.b, &r.lab)), &mut n); n.into_iter().map(|(t, x)| (t, untf(&self.b, &x))).collect() }).collect() }
    pub fn flat(&self, f: &Fv) -> (u8, u64, u32, u8, u8) { let (v, x) = match f.kind { 3 => (f.t[0], if f.t[1] == 0 { u32::MAX } else { f.t[1] as u32 - 1 }), _ => (f.t[0], u32::MAX) }; (f.kind, v, x, f.pref, f.nq) }
}
#[no_mangle]
pub unsafe extern "C" fn s2_open(dir: *const c_char, fd: u32) -> *mut S2 { CStr::from_ptr(dir).to_str().ok().and_then(|d| S2::open(d, fd as usize).ok()).map_or(std::ptr::null_mut(), |s| Box::into_raw(Box::new(s))) }
#[no_mangle]
pub unsafe extern "C" fn s2_close(s: *mut S2) { (!s.is_null()).then(|| drop(Box::from_raw(s))); }
#[no_mangle]
pub unsafe extern "C" fn s2_ne(s: *const S2) -> u64 { (*s).ne }
#[no_mangle]
pub unsafe extern "C" fn s2_ents(s: *const S2, hs: *const u64, n: usize, oi: *mut u32, oe: *mut u32, ok: *mut u8, cap: usize) -> i64 { let (st, h) = (&*s, std::slice::from_raw_parts(hs, n)); let r: Vec<Vec<(u32, u8)>> = h.iter().map(|&x| st.ents(x)).collect(); let (a, b, c) = (std::slice::from_raw_parts_mut(oi, cap), std::slice::from_raw_parts_mut(oe, cap), std::slice::from_raw_parts_mut(ok, cap)); let mut m = 0usize; for (i, v) in r.iter().enumerate() { for &(e, k) in v { if m < cap { a[m] = i as u32; b[m] = e; c[m] = k } m += 1 } } m as i64 }
#[no_mangle]
pub unsafe extern "C" fn s2_keys(s: *const S2, txt: *const u8, len: usize, out: *mut u64, cap: usize) -> i64 { let t = std::str::from_utf8(std::slice::from_raw_parts(txt, len)).unwrap_or(""); let o = std::slice::from_raw_parts_mut(out, cap); let mut n = 0usize; t.split('\n').for_each(|x| { if n < cap { o[n] = (*s).key(x) } n += 1 }); n as i64 }
#[no_mangle]
pub unsafe extern "C" fn s2_pieces(s: *const S2, txt: *const u8, len: usize, out: *mut u32, cap: usize) -> i64 { let t = std::str::from_utf8(std::slice::from_raw_parts(txt, len)).unwrap_or(""); let p = (*s).b.text(t); let o = std::slice::from_raw_parts_mut(out, cap); p.iter().take(cap).enumerate().for_each(|(i, &x)| o[i] = x); p.len() as i64 }
#[no_mangle]
pub unsafe extern "C" fn s2_groups(s: *const S2, e: u32, op: *mut u16, of: *mut u8, cap: usize) -> i64 { let st = &*s; if e as u64 >= st.ne { return 0 } let g = st.groups(e as u64); let (a, b) = (std::slice::from_raw_parts_mut(op, cap), std::slice::from_raw_parts_mut(of, cap)); g.iter().enumerate().take(cap).for_each(|(i, &(p, n))| { a[i] = p; b[i] = (n > 1) as u8 }); g.len() as i64 }
#[no_mangle]
pub unsafe extern "C" fn s2_values(s: *const S2, e: u32, pn: u16, ok: *mut u8, ov: *mut u64, ox: *mut u32, opf: *mut u8, onq: *mut u8, cap: usize) -> i64 { let st = &*s; if e as u64 >= st.ne { return 0 } let v = st.values(e as u64, pn); let (a, b, c, d, f) = (std::slice::from_raw_parts_mut(ok, cap), std::slice::from_raw_parts_mut(ov, cap), std::slice::from_raw_parts_mut(ox, cap), std::slice::from_raw_parts_mut(opf, cap), std::slice::from_raw_parts_mut(onq, cap)); v.iter().enumerate().take(cap).for_each(|(i, t)| { let x = st.flat(t); a[i] = x.0; b[i] = x.1; c[i] = x.2; d[i] = x.3; f[i] = x.4 }); v.len() as i64 }
#[no_mangle]
pub unsafe extern "C" fn s2_quals(s: *const S2, e: u32, pn: u16, i: u64, _nq: u8, op: *mut u16, ok: *mut u8, ov: *mut u64, cap: usize) -> i64 { let v = (*s).quals(e as u64, pn, i as usize); let (a, b, c) = (std::slice::from_raw_parts_mut(op, cap), std::slice::from_raw_parts_mut(ok, cap), std::slice::from_raw_parts_mut(ov, cap)); v.iter().enumerate().take(cap).for_each(|(j, t)| { a[j] = t.0; b[j] = t.1; c[j] = t.2[0] }); v.len() as i64 }
#[no_mangle]
pub unsafe extern "C" fn s2_text(s: *const S2, e: u64, w: u32, buf: *mut u8, cap: usize) -> i64 { let st = &*s; let b: Vec<u8> = match w { 0 => st.label(e), 1 => st.title(e), 2 => st.nmenc(e).into_iter().flat_map(|(t, x)| [vec![t], x, vec![0xFF]].concat()).collect(), 4 => st.hotlabel(e).unwrap_or_else(|| st.label(e)), _ => st.string(e) }; let o = std::slice::from_raw_parts_mut(buf, cap); o[..b.len().min(cap)].copy_from_slice(&b[..b.len().min(cap)]); b.len() as i64 }
#[no_mangle]
pub unsafe extern "C" fn s2_sitelinks(s: *const S2, e: u32) -> u64 { if e as u64 >= (*s).ne { 0 } else { (*s).sitelinks(e as u64) } }
#[no_mangle]
pub unsafe extern "C" fn s2_qid(s: *const S2, e: u32) -> u32 { (*s).qid(e as u64) as u32 }
#[no_mangle]
pub unsafe extern "C" fn s2_ix(s: *const S2, q: u32) -> i64 { (*s).ix(q as u64) }
#[no_mangle]
pub unsafe extern "C" fn s2_pkind(s: *const S2, pn: u16) -> i32 { let st = &*s; st.pidx.get(&pn).map_or(-1, |&p| st.pm.pkind[p as usize] as i32) }
#[no_mangle]
pub unsafe extern "C" fn s2_clear(s: *const S2) { (*s).clear() }
#[no_mangle]
pub unsafe extern "C" fn s2_cold(s: *const S2) { (*s).cold() }
