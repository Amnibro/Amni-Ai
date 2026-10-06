use adam_gf17::{hash, lut17, pack17, page17, px17, unpack17, K4, PX};
use adam_torus::{drow, keyof, layered_by, Table};
use memmap2::Mmap;
use rayon::prelude::*;
use std::{collections::{HashMap, VecDeque}, ffi::{c_char, CStr}, fs::{self, File, OpenOptions}, io, os::unix::fs::FileExt};
pub const MAGIC: &[u8; 8] = b"GF17S1PG";
pub const HDR: usize = 4096;
pub const BLK: usize = 32;
pub const TB: usize = 16;
pub const NB: usize = 32;
pub const SB: u32 = 14;
pub const RL: u32 = 1 << 23;
pub const END: u32 = 24_137_568;
pub const RULES: [(u16, u16, u16); 2] = [(17, 131, 17), (27, 19, 17)];
pub fn bad(m: String) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, m) }
pub fn ev(o: &mut Vec<u8>, v: u64) { if v < 16 { o.push(v as u8) } else { let (mut x, mut d) = (v - 16, vec![]); loop { d.push((x % 17) as u8); x /= 17; if x == 0 { break } } o.push(16); o.push(d.len() as u8 - 1); o.extend(d) } }
pub fn fx(o: &mut Vec<u8>, v: u64, n: usize) { let mut x = v; (0..n).for_each(|_| { o.push((x % 17) as u8); x /= 17 }) }
pub fn mixk(h: u64, i: u64) -> u64 { if i == 0 { h } else { let mut z = i.wrapping_mul(0x9E37_79B9_7F4A_7C15); z = (z ^ z >> 30).wrapping_mul(0xBF58_476D_1CE4_E5B9); h ^ (z ^ z >> 27) } }
pub fn seed(n: &str) -> u64 { hash(n.as_bytes()) as u64 | 0x5170_0000_0000 }
pub struct PgW { pub d: Vec<u8>, pub secs: Vec<(String, u64, u64)> }
impl PgW {
    pub fn new() -> PgW { PgW { d: vec![], secs: vec![] } }
    pub fn add(&mut self, n: &str, dg: &[u8]) { self.d.resize(self.d.len().div_ceil(4) * 4, 0); self.secs.push((n.into(), self.d.len() as u64, dg.len() as u64)); self.d.extend_from_slice(dg) }
    pub fn bytes(&mut self, n: &str, b: &[u8]) { let dg: Vec<u8> = b.chunks(2).flat_map(|c| { let w = c[0] as u32 | (*c.get(1).unwrap_or(&0) as u32) << 8; [0, 1, 2, 3].map(|k| (w / K4[k] % 17) as u8) }).collect(); self.add(n, &dg); self.secs.last_mut().unwrap().2 = b.len() as u64 }
    pub fn save(&self, path: &str) -> io::Result<(u64, bool, usize)> {
        let pk = pack17(&self.d);
        let dir: String = self.secs.iter().map(|(n, o, l)| format!("{n}\t{o}\t{l}\n")).collect();
        let mut h = vec![0u8; HDR];
        h[..8].copy_from_slice(MAGIC);
        [self.d.len() as u64, pk.len() as u64, (HDR + pk.len()) as u64, dir.len() as u64].iter().enumerate().for_each(|(i, x)| h[8 + 8 * i..16 + 8 * i].copy_from_slice(&x.to_le_bytes()));
        h[40..44].copy_from_slice(&hash(&pk).to_le_bytes());
        h[44..48].copy_from_slice(&hash(dir.as_bytes()).to_le_bytes());
        let x = hash(&h[..48]);
        h[48..52].copy_from_slice(&x.to_le_bytes());
        let f = OpenOptions::new().create(true).write(true).truncate(true).open(path)?;
        f.write_all_at(&h, 0)?;
        f.write_all_at(&pk, HDR as u64)?;
        f.write_all_at(dir.as_bytes(), (HDR + pk.len()) as u64)?;
        f.sync_all()?;
        let m = unsafe { Mmap::map(&File::open(path)?)? };
        let back = unpack17(&m[HDR..HDR + pk.len()], self.d.len());
        let pages = self.d.len().div_ceil(4).div_ceil(PX);
        let pok = (0..pages).all(|pg| { let p = page17(&m[HDR..HDR + pk.len()], self.d.len(), pg); let a = pg * PX * 4; p[..p.len().min(self.d.len() - a)] == self.d[a..(a + p.len()).min(self.d.len())] });
        Ok(((HDR + pk.len() + dir.len()) as u64, back == self.d && pok, pages))
    }
}
pub struct Pg { pub map: Mmap, nb: usize, pub nd: u64, lut: Vec<u32>, pub secs: HashMap<String, (u64, u64)> }
impl Pg {
    pub fn open(path: &str) -> io::Result<Pg> {
        let f = File::open(path)?;
        let map = unsafe { Mmap::map(&f)? };
        let q = |o: usize| u64::from_le_bytes(map[o..o + 8].try_into().unwrap()) as usize;
        (map.len() >= HDR && &map[..8] == MAGIC && hash(&map[..48]) == u32::from_le_bytes(map[48..52].try_into().unwrap())).then_some(()).ok_or_else(|| bad(format!("{path}: not a GF17S1PG container")))?;
        let (nd, nb, doff, dlen) = (q(8), q(16), q(24), q(32));
        let dir = std::str::from_utf8(&map[doff..doff + dlen]).map_err(|_| bad("dir".into()))?;
        (hash(dir.as_bytes()) == u32::from_le_bytes(map[44..48].try_into().unwrap())).then_some(()).ok_or_else(|| bad(format!("{path}: directory corrupt")))?;
        let secs = dir.lines().map(|l| { let v: Vec<&str> = l.split('\t').collect(); (v[0].to_string(), (v[1].parse().unwrap(), v[2].parse().unwrap())) }).collect();
        unsafe { libc::madvise(map.as_ptr() as *mut _, map.len(), libc::MADV_RANDOM) };
        Ok(Pg { map, nb, nd: nd as u64, lut: lut17(), secs })
    }
    #[inline(always)]
    pub fn word(&self, p: u64) -> u32 { px17(&self.map[HDR..HDR + self.nb], p as usize) }
    #[inline(always)]
    pub fn dig(&self, g: u64) -> u8 { (self.lut[self.word(g >> 2) as usize] >> (8 * (g & 3))) as u8 }
    #[inline(always)]
    pub fn byte(&self, o: u64, j: u64) -> u8 { (self.word((o >> 2) + (j >> 1)) >> (8 * (j & 1))) as u8 }
    pub fn sec(&self, n: &str) -> (u64, u64) { self.secs.get(n).copied().unwrap_or((0, 0)) }
    pub fn rd(&self, i: u64) -> Rd<'_> { Rd { g: self, i } }
    pub fn u32at(&self, o: u64, i: u64) -> u32 { (0..4).fold(0u32, |a, k| a | (self.byte(o, 4 * i + k) as u32) << (8 * k)) }
}
pub struct Rd<'a> { pub g: &'a Pg, pub i: u64 }
impl Rd<'_> {
    #[inline(always)]
    pub fn d(&mut self) -> u8 { let x = self.g.dig(self.i); self.i += 1; x }
    #[inline(always)]
    pub fn ev(&mut self) -> u64 { let a = self.d(); if a < 16 { return a as u64 } let n = self.d() as u32 + 1; (0..n).fold((0u64, 1u64), |(s, m), _| (s.wrapping_add((self.d() as u64).wrapping_mul(m)), m.wrapping_mul(17))).0.wrapping_add(16) }
    pub fn fx(&mut self, n: usize) -> u64 { (0..n).fold((0u64, 1u64), |(s, m), _| (s + self.d() as u64 * m, m * 17)).0 }
}
pub fn hlen(f: &[u64]) -> Vec<u8> {
    let n = f.len();
    if n <= 1 { return vec![1; n] }
    let m = n + (16 - (n - 1) % 16) % 16;
    let mut wt: Vec<u64> = (0..m).map(|i| if i < n { f[i] } else { 0 }).collect();
    let mut lv: Vec<usize> = (0..m).collect();
    lv.sort_by_key(|&i| wt[i]);
    let (mut par, mut a, mut q) = (vec![usize::MAX; m], 0usize, VecDeque::new());
    while (m - a) + q.len() > 1 {
        let id = wt.len();
        let mut s = 0u64;
        for _ in 0..17 { let lf = a < m && q.front().is_none_or(|&j| wt[lv[a]] <= wt[j]); let x = if lf { a += 1; lv[a - 1] } else { q.pop_front().unwrap() }; s += wt[x]; par[x] = id; }
        wt.push(s);
        par.push(usize::MAX);
        q.push_back(id);
    }
    let mut dep = vec![0u8; wt.len()];
    (0..wt.len()).rev().for_each(|i| if par[i] != usize::MAX { dep[i] = dep[par[i]] + 1 });
    dep[..n].to_vec()
}
#[derive(Clone, Default)]
pub struct Hc { pub cnt: Vec<u64>, pub first: Vec<u64>, pub start: Vec<u64>, pub blk: Vec<u64> }
impl Hc {
    pub fn new(cnt: Vec<u64>, b0: u64) -> Hc { let (mut first, mut start, mut blk, mut f, mut s, mut b) = (vec![], vec![], vec![], 0u64, 0u64, b0); cnt.iter().for_each(|&c| { first.push(f); start.push(s); blk.push(b); f = (f + c) * 17; s += c; b += c.div_ceil(BLK as u64) }); Hc { cnt, first, start, blk } }
    pub fn nblk(&self) -> u64 { self.cnt.iter().map(|c| c.div_ceil(BLK as u64)).sum() }
    pub fn code(&self, l: usize, r: u64) -> Vec<u8> { let c = self.first[l] + r; (0..=l).rev().map(|k| (c / 17u64.pow(k as u32) % 17) as u8).collect() }
}
pub fn canon(lens: &[u8]) -> (Vec<u64>, Vec<(usize, u64)>) {
    let ml = lens.iter().copied().max().unwrap_or(1) as usize;
    let mut cnt = vec![0u64; ml];
    let at: Vec<(usize, u64)> = lens.iter().map(|&l| { let r = cnt[l as usize - 1]; cnt[l as usize - 1] += 1; (l as usize - 1, r) }).collect();
    (cnt, at)
}
pub fn cbenc(o: &mut Vec<u8>, prev: &[u64], t: &[u64]) {
    let j = prev.iter().zip(t).take_while(|(a, b)| a == b).count();
    let r = t.len() - j;
    if j < 4 && r < 4 { o.push((j * 4 + r) as u8) } else { o.push(16); ev(o, j as u64); ev(o, r as u64) }
    (j..t.len()).for_each(|i| ev(o, if i == j && j < prev.len() { t[i] - prev[i] } else { t[i] }));
}
pub fn cbdec(r: &mut Rd, prev: &mut Vec<u64>) {
    let h = r.d();
    let (j, n) = if h < 16 { ((h / 4) as usize, (h % 4) as usize) } else { (r.ev() as usize, r.ev() as usize) };
    let b0 = prev.get(j).copied().unwrap_or(0);
    prev.truncate(j);
    (0..n).for_each(|i| { let v = r.ev(); prev.push(if i == 0 { b0 + v } else { v }) });
}
pub struct CasB { pub name: String, pub hc: Vec<Hc>, pub cb: Vec<u8>, pub cbx: Vec<u8>, pub lv: Vec<Vec<(u64, [u8; 1])>>, pub syms: Vec<HashMap<Vec<u64>, (usize, u64)>> }
impl CasB {
    pub fn new(name: &str, ctx: Vec<HashMap<Vec<u64>, u64>>) -> CasB {
        let (mut hc, mut cb, mut cbx, mut syms, mut b0) = (vec![], vec![], vec![], vec![], 0u64);
        for m in ctx {
            let mut ks: Vec<(Vec<u64>, u64)> = m.into_iter().collect();
            ks.sort_unstable();
            let lens = hlen(&ks.iter().map(|x| x.1).collect::<Vec<_>>());
            let (cnt, at) = canon(&lens);
            let h = Hc::new(cnt, b0);
            b0 += h.nblk();
            let mut grp: Vec<Vec<&Vec<u64>>> = vec![vec![]; h.cnt.len()];
            ks.iter().zip(&at).for_each(|((k, _), &(l, _))| grp[l].push(k));
            for g in &grp { for (i, t) in g.iter().enumerate() { if i % BLK == 0 { fx(&mut cbx, cb.len() as u64, 8) } cbenc(&mut cb, if i % BLK == 0 { &[] } else { g[i - 1] }, t) } }
            syms.push(ks.into_iter().zip(at).map(|((k, _), a)| (k, a)).collect());
            hc.push(h);
        }
        CasB { name: name.into(), hc, cb, cbx, lv: vec![], syms }
    }
    pub fn push(&mut self, key: u64, c: usize, t: &[u64]) { let (l, r) = self.syms[c][t]; let d = self.hc[c].code(l, r); while self.lv.len() < d.len() { self.lv.push(vec![]) } d.iter().enumerate().for_each(|(j, &x)| self.lv[j].push((key, [x]))) }
    pub fn hcdig(&self) -> Vec<u8> { let mut o = vec![]; ev(&mut o, self.hc.len() as u64); self.hc.iter().for_each(|h| { ev(&mut o, h.cnt.len() as u64); h.cnt.iter().for_each(|&c| ev(&mut o, c)) }); o }
    pub fn save(&mut self, dir: &str, w: &mut PgW) -> io::Result<(u64, u64)> {
        let mut by = 0u64;
        let mut keys = 0u64;
        for (j, l) in self.lv.iter().enumerate() { let p = format!("{dir}/{}_{j}.gf17rib", self.name); keys += l.len() as u64; by += layered_by::<1, [u8; 1]>(&p, l, drow::<1>, 64, 0, 65536, -0.03, seed(&p), Some((32, [0, 8, 16, 32])), 0.08, None)?.2; }
        w.add(&format!("{}_hc", self.name), &self.hcdig());
        w.add(&format!("{}_cb", self.name), &self.cb);
        w.add(&format!("{}_cbx", self.name), &self.cbx);
        fs::write(format!("{dir}/{}.lv", self.name), self.lv.len().to_string())?;
        self.lv.clear();
        Ok((by, keys))
    }
}
pub struct Cas { pub lv: Vec<Table<1>>, pub hc: Vec<Hc>, cb: u64, cbx: u64 }
impl Cas {
    pub fn open(dir: &str, name: &str, g: &Pg) -> io::Result<Cas> {
        let n: usize = fs::read_to_string(format!("{dir}/{name}.lv"))?.trim().parse().map_err(|_| bad("lv".into()))?;
        let lv = (0..n).map(|j| Table::<1>::open(&format!("{dir}/{name}_{j}.gf17rib"))).collect::<io::Result<Vec<_>>>()?;
        let mut r = g.rd(g.sec(&format!("{name}_hc")).0);
        let nc = r.ev() as usize;
        let mut b0 = 0u64;
        let hc = (0..nc).map(|_| { let l = r.ev() as usize; let h = Hc::new((0..l).map(|_| r.ev()).collect(), b0); b0 += h.nblk(); h }).collect();
        Ok(Cas { lv, hc, cb: g.sec(&format!("{name}_cb")).0, cbx: g.sec(&format!("{name}_cbx")).0 })
    }
    #[inline(always)]
    pub fn sym(&self, key: u64, c: usize) -> Option<(usize, u64)> {
        let h = self.hc.get(c)?;
        let mut v = 0u64;
        for l in 0..h.cnt.len() { v = v * 17 + self.lv.get(l)?.row(key)?[0] as u64; if v >= h.first[l] && v - h.first[l] < h.cnt[l] { return Some((l, v - h.first[l])) } }
        None
    }
    pub fn entry(&self, g: &Pg, c: usize, l: usize, r: u64) -> Vec<u64> {
        let b = self.hc[c].blk[l] + r / BLK as u64;
        let mut rd = g.rd(self.cb + g.rd(self.cbx + 8 * b).fx(8));
        let mut t = vec![];
        (0..=r % BLK as u64).for_each(|_| cbdec(&mut rd, &mut t));
        t
    }
    pub fn get(&self, g: &Pg, key: u64, c: usize) -> Option<Vec<u64>> { self.sym(key, c).map(|(l, r)| self.entry(g, c, l, r)) }
    pub fn cold(&self) { self.lv.iter().for_each(|t| t.cold()) }
    pub fn bytes(&self) -> u64 { self.lv.iter().map(|t| t.bytes).sum() }
}
pub fn ctxk(f: u64, p: &[u16], k: usize) -> u64 { p.iter().rev().take(k).fold(0u64, |a, &s| a << 9 | s as u64) | (p.len().min(k) as u64) << 52 | f << 56 }
#[derive(Default, Clone, Copy)]
pub struct Fx(u64);
impl std::hash::Hasher for Fx { fn finish(&self) -> u64 { self.0 } fn write(&mut self, b: &[u8]) { b.iter().for_each(|&x| self.0 = (self.0.rotate_left(5) ^ x as u64).wrapping_mul(0x51_7CC1_B727_220A_95)) } fn write_u64(&mut self, v: u64) { self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7CC1_B727_220A_95) } }
pub type FxB = std::hash::BuildHasherDefault<Fx>;
pub struct Tm { pub k: usize, pub ix: HashMap<u64, u32, FxB>, pub sy: Vec<u16>, pub cu: Vec<u32>, pub off: Vec<u32> }
impl Tm {
    pub fn build(rs: &[&[Vec<(u8, Vec<u16>)>]], k: usize) -> (Tm, Vec<u8>) {
        let cnt: HashMap<u64, HashMap<u16, u64>> = rs.iter().flat_map(|r| r.par_chunks(1 << 14).collect::<Vec<_>>()).collect::<Vec<_>>().into_par_iter().fold(HashMap::new, |mut m: HashMap<u64, HashMap<u16, u64>>, ch| { ch.iter().for_each(|r| r.iter().for_each(|(f, s)| s.iter().enumerate().for_each(|(i, &x)| *m.entry(ctxk(*f as u64, &s[..i], k)).or_default().entry(x).or_default() += 1))); m }).reduce(HashMap::new, |mut a, b| { b.into_iter().for_each(|(c, m)| { let e = a.entry(c).or_default(); m.into_iter().for_each(|(s, n)| *e.entry(s).or_default() += n) }); a });
        let mut cs: Vec<(u64, Vec<(u16, u64)>)> = cnt.into_iter().map(|(c, m)| { let mut v: Vec<(u16, u64)> = m.into_iter().collect(); v.sort(); (c, v) }).collect();
        cs.sort_unstable_by_key(|x| x.0);
        let mut dg = vec![];
        ev(&mut dg, k as u64);
        ev(&mut dg, cs.len() as u64);
        let (mut tm, mut pc) = (Tm { k, ix: HashMap::default(), sy: vec![], cu: vec![], off: vec![0] }, 0u64);
        for (c, v) in &cs {
            let tot: u64 = v.iter().map(|x| x.1).sum();
            let mut q: Vec<u32> = v.iter().map(|x| ((x.1 << SB) / tot).max(1) as u32).collect();
            let s: i64 = q.iter().map(|&x| x as i64).sum::<i64>() - (1i64 << SB);
            let j = (0..q.len()).max_by_key(|&i| q[i]).unwrap();
            assert!(q[j] as i64 > s, "quantize");
            q[j] = (q[j] as i64 - s) as u32;
            ev(&mut dg, c - pc);
            pc = *c;
            ev(&mut dg, v.len() as u64 - 1);
            let mut ps = 0u16;
            v.iter().zip(&q).for_each(|((sy, _), &f)| { ev(&mut dg, (sy - ps) as u64); ps = *sy; ev(&mut dg, f as u64 - 1) });
            tm.push(*c, v.iter().map(|x| x.0).zip(q.iter().copied()));
        }
        (tm, dg)
    }
    fn push(&mut self, c: u64, it: impl Iterator<Item = (u16, u32)>) { self.ix.insert(c, self.off.len() as u32 - 1); let mut a = 0u32; it.for_each(|(s, f)| { self.sy.push(s); self.cu.push(a); a += f }); self.cu.push(a); self.sy.push(u16::MAX); self.off.push(self.sy.len() as u32) }
    pub fn load(g: &Pg, o: u64) -> Tm {
        let mut r = g.rd(o);
        let (k, n) = (r.ev() as usize, r.ev());
        let (mut tm, mut c) = (Tm { k, ix: HashMap::with_capacity_and_hasher(n as usize, FxB::default()), sy: vec![], cu: vec![], off: vec![0] }, 0u64);
        for _ in 0..n { c += r.ev(); let m = r.ev() + 1; let mut s = 0u64; let v: Vec<(u16, u32)> = (0..m).map(|_| { s += r.ev(); (s as u16, r.ev() as u32 + 1) }).collect(); tm.push(c, v.into_iter()) }
        tm
    }
    #[inline(always)]
    fn ctx(&self, c: u64) -> (usize, usize) { let i = self.ix[&c] as usize; (self.off[i] as usize, self.off[i + 1] as usize - 1) }
    pub fn enc(&self, recs: &[Vec<(u8, Vec<u16>)>]) -> Vec<u8> {
        let mut st: Vec<(u32, u32)> = vec![];
        recs.iter().for_each(|r| r.iter().for_each(|(f, s)| s.iter().enumerate().for_each(|(i, &x)| { let (a, _) = self.ctx(ctxk(*f as u64, &s[..i], self.k)); let j = a + self.sy[a..].iter().position(|&y| y == x).unwrap(); st.push((self.cu[j], self.cu[j + 1] - self.cu[j])) })));
        let (mut x, mut o) = (RL, vec![]);
        for &(c, f) in st.iter().rev() { let xm = ((RL >> SB) << 8) * f; while x >= xm { o.push(x as u8); x >>= 8 } x = ((x / f) << SB) + x % f + c; }
        o.extend_from_slice(&x.to_le_bytes());
        o.reverse();
        o
    }
}
pub struct Dec<'a> { pub g: &'a Pg, pub o: u64, pub j: u64, pub x: u32, pub m: &'a Tm }
impl Dec<'_> {
    pub fn new<'a>(g: &'a Pg, o: u64, j: u64, m: &'a Tm) -> Dec<'a> { let x = (0..4).fold(0u32, |a, i| a << 8 | g.byte(o, j + i) as u32); Dec { g, o, j: j + 4, x, m } }
    #[inline(always)]
    pub fn sym(&mut self, c: u64) -> u16 {
        let (a, b) = self.m.ctx(c);
        let s = self.x & ((1 << SB) - 1);
        let i = a + self.m.cu[a..=b].partition_point(|&q| q <= s) - 1;
        self.x = (self.m.cu[i + 1] - self.m.cu[i]) * (self.x >> SB) + s - self.m.cu[i];
        while self.x < RL { self.x = self.x << 8 | self.g.byte(self.o, self.j) as u32; self.j += 1 }
        self.m.sy[i]
    }
    pub fn field(&mut self, f: u8) -> Vec<u16> { let mut s: Vec<u16> = vec![]; loop { let x = self.sym(ctxk(f as u64, &s, self.m.k)); if x >= 256 || f == 0 || f == 3 { s.push(x); return s } s.push(x) } }
}
pub fn rec_a(tk: u8, lab: &[u8], ex: &[u8]) -> Vec<(u8, Vec<u16>)> { let z = |b: &[u8]| b.iter().map(|&x| x as u16).chain([256]).collect::<Vec<u16>>(); let mut r = vec![(0u8, vec![tk as u16]), (1, z(lab))]; (tk >= 2).then(|| r.push((2, z(ex)))); r }
pub fn rec_b(nm: &[(u8, Vec<u8>)]) -> Vec<(u8, Vec<u16>)> { let mut r = vec![]; nm.iter().for_each(|(t, b)| { r.push((3, vec![*t as u16])); (*t == 2).then(|| r.push((4, b.iter().map(|&x| x as u16).chain([256]).collect()))); }); r.push((3, vec![3])); r }
pub fn rec_s(b: &[u8]) -> Vec<(u8, Vec<u16>)> { vec![(5, b.iter().map(|&x| x as u16).chain([256]).collect())] }
pub fn blocks(tm: &Tm, recs: &[Vec<(u8, Vec<u16>)>], bs: usize) -> (Vec<u8>, Vec<u8>) {
    let parts: Vec<Vec<u8>> = recs.par_chunks(bs).map(|c| tm.enc(c)).collect();
    let (mut s, mut off) = (vec![], vec![]);
    parts.iter().for_each(|p| { off.extend_from_slice(&(s.len() as u32).to_le_bytes()); s.extend_from_slice(p) });
    off.extend_from_slice(&(s.len() as u32).to_le_bytes());
    (s, off)
}
pub enum Nm { A(Table<7>), B(Table<8>), C(Table<9>), D(Table<10>) }
impl Nm {
    pub fn open(p: &str, fd: usize) -> io::Result<Nm> { Ok(match fd { 0 => Nm::A(Table::open(p)?), 1 => Nm::B(Table::open(p)?), 2 => Nm::C(Table::open(p)?), _ => Nm::D(Table::open(p)?) }) }
    #[inline(always)]
    pub fn row(&self, k: u64) -> Option<[u8; 7]> { let c = |d: &[u8]| -> [u8; 7] { d[..7].try_into().unwrap() }; match self { Nm::A(t) => t.row(k).map(|d| c(&d)), Nm::B(t) => t.row(k).map(|d| c(&d)), Nm::C(t) => t.row(k).map(|d| c(&d)), Nm::D(t) => t.row(k).map(|d| c(&d)) } }
    pub fn bytes(&self) -> u64 { match self { Nm::A(t) => t.bytes, Nm::B(t) => t.bytes, Nm::C(t) => t.bytes, Nm::D(t) => t.bytes } }
    pub fn cold(&self) { match self { Nm::A(t) => t.cold(), Nm::B(t) => t.cold(), Nm::C(t) => t.cold(), Nm::D(t) => t.cold() } }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Fact { pub kind: u8, pub v: u64, pub x: u32, pub pref: u8, pub nq: u8 }
pub struct Store { pub dir: String, pub g: Pg, pub ne: u64, pub pnum: Vec<u16>, pub pkind: Vec<u8>, pub prank: Vec<u16>, pub prk: Vec<usize>, pub pidx: HashMap<u16, usize>, pub sig: Cas, pub cnt: Cas, pub sl: Cas, pub fact: Cas, pub qual: Cas, pub sid: Option<Table<5>>, pub nm: Nm, pub nme: Option<Table<16>>, pub fd: usize, pub tm: Tm }
impl Store {
    pub fn open(dir: &str, fd: usize) -> io::Result<Store> {
        let g = Pg::open(&format!("{dir}/store1.gf17pg"))?;
        let mut r = g.rd(g.sec("meta").0);
        let (ne, np) = (r.ev(), r.ev() as usize);
        let (mut pnum, mut pkind, mut prank) = (vec![], vec![], vec![]);
        (0..np).for_each(|_| { pnum.push(r.ev() as u16); pkind.push(r.ev() as u8); prank.push(r.ev() as u16) });
        let pidx = pnum.iter().enumerate().map(|(i, &p)| (p, i)).collect();
        let mut prk = vec![0usize; np];
        prank.iter().enumerate().for_each(|(i, &r)| prk[r as usize] = i);
        let tm = Tm::load(&g, g.sec("tm").0);
        let sp = format!("{dir}/sid.gf17rib");
        let ep = format!("{dir}/name_f{fd}_end.gf17rib");
        Ok(Store { dir: dir.into(), ne, pidx, sig: Cas::open(dir, "sig", &g)?, cnt: Cas::open(dir, "cnt", &g)?, sl: Cas::open(dir, "sl", &g)?, fact: Cas::open(dir, "fact", &g)?, qual: Cas::open(dir, "qual", &g)?, sid: if fs::metadata(&sp).is_ok() { Some(Table::open(&sp)?) } else { None }, nm: Nm::open(&format!("{dir}/name_f{fd}.gf17rib"), fd)?, nme: if fs::metadata(&ep).is_ok() { Some(Table::open(&ep)?) } else { None }, fd, tm, g, pnum, pkind, prank, prk })
    }
    pub fn groups(&self, e: u32) -> Vec<(usize, u8)> {
        self.sig.get(&self.g, e as u64, 0).unwrap_or_default().chunks(2).map(|c| (self.prk[c[0] as usize], c[1] as u8)).collect()
    }
    pub fn count(&self, e: u32, p: usize, f: u8) -> u64 { if f == 1 { self.cnt.get(&self.g, keyof(e, self.pnum[p], 0), 0).map_or(0, |t| t[0]) } else { 1 } }
    pub fn fact(&self, e: u32, p: usize, i: u64) -> Option<Fact> {
        let k = keyof(e, self.pnum[p], i as u16);
        let t = self.fact.get(&self.g, k, p)?;
        let kd = self.pkind[p];
        let (v, x, a) = match kd { 1 | 2 => (t[0], u32::MAX, t[1]), 3 => (t[0], if t[1] == 0 { u32::MAX } else { t[1] as u32 - 1 }, t[2]), _ => (self.sid.as_ref().and_then(|s| s.row(k)).map_or(0, |d| d.iter().rev().fold(0u64, |a, &x| a * 17 + x as u64)), u32::MAX, t[0]) };
        Some(Fact { kind: kd, v, x, pref: (a & 1) as u8, nq: (a >> 1) as u8 })
    }
    pub fn values(&self, e: u32, pn: u16) -> Vec<Fact> { self.vals(e, pn, 0) }
    fn vals(&self, e: u32, pn: u16, dep: u32) -> Vec<Fact> {
        let Some(&p) = self.pidx.get(&pn) else { return vec![] };
        let Some(&(_, f)) = self.groups(e).iter().find(|g| g.0 == p) else { return vec![] };
        if f == 2 { let r = RULES.iter().find(|r| r.0 == pn).unwrap(); return if dep > 32 { vec![] } else { self.vals(e, r.1, dep + 1).first().map_or(vec![], |t| self.vals(t.v as u32, r.2, dep + 1).into_iter().map(|x| Fact { pref: 0, nq: 0, ..x }).collect()) } }
        (0..self.count(e, p, f)).filter_map(|i| self.fact(e, p, i)).collect()
    }
    pub fn quals(&self, e: u32, pn: u16, i: u64, nq: u8) -> Vec<(u16, u8, u64)> { let p = self.pidx[&pn]; (0..nq as u64).filter_map(|j| self.qual.get(&self.g, keyof(e, pn, i as u16) | (j + 1) << 56, p).map(|t| (self.pnum[t[0] as usize], t[1] as u8, t[2]))).collect() }
    pub fn ents(&self, h: u64) -> Vec<(u32, u8)> {
        let mut o: Vec<(u32, u8)> = vec![];
        for i in 0..1 << 20 { let Some(d) = self.nm.row(mixk(h, i)) else { break }; let e = d[..6].iter().rev().fold(0u32, |a, &x| a * 17 + x as u32); if e == END || e as u64 >= self.ne || o.last().is_some_and(|l| l.0 >= e) || d[6] > 3 || i > 0 && self.nme.as_ref().is_some_and(|t| t.row(mixk(h, i)).is_some()) { break } o.push((e, d[6])) }
        o
    }
    fn tdec(&self, s: &str, bs: u64, e: u64) -> Dec<'_> { let (o, _) = self.g.sec(&format!("{s}_x")); let b = e / bs; Dec::new(&self.g, self.g.sec(s).0, self.g.u32at(o, b) as u64, &self.tm) }
    pub fn rec_a(&self, e: u32) -> (u8, Vec<u8>, Vec<u8>) {
        let mut d = self.tdec("ta", TB as u64, e as u64);
        let b8 = |v: Vec<u16>| v.into_iter().filter(|&x| x < 256).map(|x| x as u8).collect::<Vec<u8>>();
        let mut out = (0u8, vec![], vec![]);
        for _ in 0..=(e as u64 % TB as u64) { let tk = d.field(0)[0] as u8; let l = b8(d.field(1)); let x = if tk >= 2 { b8(d.field(2)) } else { vec![] }; out = (tk, l, x) }
        out
    }
    pub fn label(&self, e: u32) -> Vec<u8> { self.rec_a(e).1 }
    pub fn title(&self, e: u32) -> Vec<u8> { let (tk, l, x) = self.rec_a(e); match tk { 0 => vec![], 1 => l, 2 => [l, b" (".to_vec(), x, b")".to_vec()].concat(), _ => x } }
    pub fn nmenc(&self, e: u32) -> Vec<(u8, Vec<u8>)> {
        let mut d = self.tdec("tb", NB as u64, e as u64);
        let mut out = vec![];
        for _ in 0..=(e as u64 % NB as u64) { out = vec![]; loop { let t = d.field(3)[0] as u8; if t == 3 { break } out.push((t, if t == 2 { d.field(4).into_iter().filter(|&x| x < 256).map(|x| x as u8).collect() } else { vec![] })) } }
        out
    }
    pub fn string(&self, s: u64) -> Vec<u8> { let mut d = self.tdec("tc", TB as u64, s); let mut o = vec![]; for _ in 0..=(s % TB as u64) { o = d.field(5).into_iter().filter(|&x| x < 256).map(|x| x as u8).collect() } o }
    pub fn sitelinks(&self, e: u32) -> u64 { self.sl.get(&self.g, e as u64, 0).map_or(0, |t| t[0]) }
    pub fn qid(&self, e: u32) -> u32 { let (o, _) = self.g.sec("qid"); let (x, _) = self.g.sec("qidx"); let b = e as u64 / 64; let mut r = self.g.rd(o + self.g.rd(x + 8 * b).fx(8)); let mut q = r.fx(7); (0..e as u64 % 64).for_each(|_| q += r.ev() + 1); q as u32 }
    pub fn ix(&self, q: u32) -> i64 {
        let (o, _) = self.g.sec("qid");
        let (x, _) = self.g.sec("qidx");
        let nb = self.ne.div_ceil(64);
        let base = |b: u64| self.g.rd(o + self.g.rd(x + 8 * b).fx(8)).fx(7);
        let (mut lo, mut hi) = (0u64, nb);
        while hi - lo > 1 { let m = (lo + hi) / 2; if base(m) <= q as u64 { lo = m } else { hi = m } }
        let mut r = self.g.rd(o + self.g.rd(x + 8 * lo).fx(8));
        let mut c = r.fx(7);
        for i in 0..64u64.min(self.ne - lo * 64) { if i > 0 { c += r.ev() + 1 } if c == q as u64 { return (lo * 64 + i) as i64 } if c > q as u64 { break } }
        -1
    }
    pub fn cold(&self) { [&self.sig, &self.cnt, &self.sl, &self.fact, &self.qual].iter().for_each(|c| c.cold()); self.nm.cold(); self.nme.as_ref().map(|t| t.cold()); self.sid.as_ref().map(|t| t.cold()); unsafe { libc::madvise(self.g.map.as_ptr() as *mut _, self.g.map.len(), libc::MADV_DONTNEED) }; }
}
unsafe fn sl<'a, T>(p: *mut T, n: usize) -> &'a mut [T] { std::slice::from_raw_parts_mut(p, n) }
#[no_mangle]
pub unsafe extern "C" fn s1_open(dir: *const c_char, fd: u32) -> *mut Store { CStr::from_ptr(dir).to_str().ok().and_then(|d| Store::open(d, fd as usize).ok()).map_or(std::ptr::null_mut(), |s| Box::into_raw(Box::new(s))) }
#[no_mangle]
pub unsafe extern "C" fn s1_close(s: *mut Store) { (!s.is_null()).then(|| drop(Box::from_raw(s))); }
#[no_mangle]
pub unsafe extern "C" fn s1_ne(s: *const Store) -> u64 { (*s).ne }
#[no_mangle]
pub unsafe extern "C" fn s1_ents(s: *const Store, hs: *const u64, n: usize, oi: *mut u32, oe: *mut u32, ok: *mut u8, cap: usize) -> i64 {
    let (st, h) = (&*s, std::slice::from_raw_parts(hs, n));
    let r: Vec<Vec<(u32, u8)>> = h.par_iter().map(|&x| st.ents(x)).collect();
    let (a, b, c) = (sl(oi, cap), sl(oe, cap), sl(ok, cap));
    let mut m = 0usize;
    for (i, v) in r.iter().enumerate() { for &(e, k) in v { if m < cap { a[m] = i as u32; b[m] = e; c[m] = k } m += 1 } }
    m as i64
}
#[no_mangle]
pub unsafe extern "C" fn s1_groups(s: *const Store, e: u32, op: *mut u16, of: *mut u8, cap: usize) -> i64 { let st = &*s; let g = st.groups(e); let (a, b) = (sl(op, cap), sl(of, cap)); g.iter().enumerate().take(cap).for_each(|(i, &(p, f))| { a[i] = st.pnum[p]; b[i] = f }); g.len() as i64 }
#[no_mangle]
pub unsafe extern "C" fn s1_values(s: *const Store, e: u32, pn: u16, ok: *mut u8, ov: *mut u64, ox: *mut u32, opf: *mut u8, onq: *mut u8, cap: usize) -> i64 { let v = (*s).values(e, pn); let (a, b, c, d, f) = (sl(ok, cap), sl(ov, cap), sl(ox, cap), sl(opf, cap), sl(onq, cap)); v.iter().enumerate().take(cap).for_each(|(i, t)| { a[i] = t.kind; b[i] = t.v; c[i] = t.x; d[i] = t.pref; f[i] = t.nq }); v.len() as i64 }
#[no_mangle]
pub unsafe extern "C" fn s1_quals(s: *const Store, e: u32, pn: u16, i: u64, nq: u8, op: *mut u16, ok: *mut u8, ov: *mut u64, cap: usize) -> i64 { let v = (*s).quals(e, pn, i, nq); let (a, b, c) = (sl(op, cap), sl(ok, cap), sl(ov, cap)); v.iter().enumerate().take(cap).for_each(|(j, t)| { a[j] = t.0; b[j] = t.1; c[j] = t.2 }); v.len() as i64 }
#[no_mangle]
pub unsafe extern "C" fn s1_text(s: *const Store, e: u64, w: u32, buf: *mut u8, cap: usize) -> i64 {
    let st = &*s;
    let b: Vec<u8> = match w { 0 => st.label(e as u32), 1 => st.title(e as u32), 2 => st.nmenc(e as u32).into_iter().flat_map(|(t, x)| [vec![t], x, vec![0xFF]].concat()).collect(), _ => st.string(e) };
    sl(buf, cap)[..b.len().min(cap)].copy_from_slice(&b[..b.len().min(cap)]);
    b.len() as i64
}
#[no_mangle]
pub unsafe extern "C" fn s1_sitelinks(s: *const Store, e: u32) -> u64 { (*s).sitelinks(e) }
#[no_mangle]
pub unsafe extern "C" fn s1_qid(s: *const Store, e: u32) -> u32 { (*s).qid(e) }
#[no_mangle]
pub unsafe extern "C" fn s1_ix(s: *const Store, q: u32) -> i64 { (*s).ix(q) }
#[no_mangle]
pub unsafe extern "C" fn s1_pkind(s: *const Store, pn: u16) -> i32 { let st = &*s; st.pidx.get(&pn).map_or(-1, |&p| st.pkind[p] as i32) }
