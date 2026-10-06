use adam_gf17::{hash, lut17, pack17, px17, PX};
use memmap2::Mmap;
use rayon::prelude::*;
use std::{fs::{File, OpenOptions}, io, os::unix::{fs::FileExt, io::AsRawFd}};
pub const MAGIC: &[u8; 8] = b"GF17RIB1";
pub const HDR: usize = 4096;
pub const VD: usize = 6;
pub const TOMB: u32 = 24_137_568;
pub const SC: usize = 1600;
pub static M17: [u8; 1100] = { let mut t = [0u8; 1100]; let mut i = 0; while i < 1100 { t[i] = (i % 17) as u8; i += 1; } t };
pub static INV: [u8; 17] = [0, 1, 9, 6, 13, 7, 3, 5, 15, 2, 12, 14, 10, 4, 11, 8, 16];
fn bad(m: String) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, m) }
pub fn sm(x: &mut u64) -> u64 { *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15); let mut z = *x; z = (z ^ z >> 30).wrapping_mul(0xBF58_476D_1CE4_E5B9); z = (z ^ z >> 27).wrapping_mul(0x94D0_49BB_1331_11EB); z ^ z >> 31 }
pub struct Tab { t: Vec<[u64; 4]> }
impl Tab {
    pub fn new(seed: u64) -> Tab { let mut s = seed; Tab { t: (0..8 * 256).map(|_| std::array::from_fn(|_| sm(&mut s))).collect() } }
    #[inline(always)]
    pub fn h(&self, k: u64) -> [u64; 4] { (0..8).fold([0u64; 4], |a, i| { let e = &self.t[i << 8 | (k >> (i << 3) & 255) as usize]; [a[0] ^ e[0], a[1] ^ e[1], a[2] ^ e[2], a[3] ^ e[3]] }) }
}
#[inline(always)]
pub fn fpd(h: &[u64; 4], k: usize) -> u8 { let f = (h[2] as u128 | (h[3] as u128) << 64) >> (10 * k) & 1023; (((f << 4) + f) >> 10) as u8 }
#[inline(always)]
pub fn fr(x: u64, n: u64) -> u64 { (x & 0xFFFF_FFFF) * n >> 32 }
pub fn vdig<const R: usize>(id: u32, h: &[u64; 4], fd: usize) -> [u8; R] { std::array::from_fn(|k| if k < VD { (id / 17u32.pow(k as u32) % 17) as u8 } else if k < VD + fd { fpd(h, k - VD) } else { 0 }) }
pub type Dig<K, const R: usize> = fn(&K, &[u64; 4], usize) -> [u8; R];
pub fn vrow<const R: usize>(id: &u32, h: &[u64; 4], fd: usize) -> [u8; R] { vdig::<R>(*id, h, fd) }
pub fn drow<const R: usize>(d: &[u8; R], h: &[u64; 4], fd: usize) -> [u8; R] { let mut o = *d; (0..fd).for_each(|k| o[R - fd + k] = fpd(h, k)); o }
#[inline(always)]
pub fn unid(d: &[u8]) -> u32 { d[..VD].iter().rev().fold(0u32, |a, &x| (a << 4) + a + x as u32) }
pub struct El<const R: usize> { pc: Vec<[u8; 64]>, pv: Vec<[u8; R]>, pub oc: Vec<bool>, pub steps: u64 }
impl<const R: usize> El<R> {
    pub fn new(m: usize) -> El<R> { El { pc: vec![[0u8; 64]; m], pv: vec![[0u8; R]; m], oc: vec![false; m], steps: 0 } }
    pub fn ins(&mut self, s: usize, cb: u64, w: usize, v0: [u8; R]) -> Result<Option<usize>, ()> {
        let (mut i, mut c, mut v) = (s, [0u8; 64], v0);
        (0..w).for_each(|j| c[j] = (cb >> j & 1) as u8);
        c[0] = 1;
        loop {
            if !self.oc[i] { let iv = INV[c[0] as usize] as u16; c.iter_mut().for_each(|x| *x = (*x as u16 * iv % 17) as u8); v.iter_mut().for_each(|x| *x = (*x as u16 * iv % 17) as u8); (self.pc[i], self.pv[i], self.oc[i]) = (c, v, true); return Ok(Some(i)); }
            self.steps += 1;
            let (f, p, q) = (17 - c[0] as u16, &self.pc[i], &self.pv[i]);
            (0..64).for_each(|j| c[j] = ((c[j] as u16 + f * p[j] as u16) % 17) as u8);
            (0..R).for_each(|j| v[j] = ((v[j] as u16 + f * q[j] as u16) % 17) as u8);
            match c.iter().position(|&x| x != 0) { Some(k) => { c.copy_within(k.., 0); c[64 - k..].fill(0); i += k; } None => return if v.iter().all(|&x| x == 0) { Ok(None) } else { Err(()) } }
        }
    }
    pub fn back(&self, seed: u64) -> Vec<[u8; R]> {
        let (m, mut rs) = (self.oc.len(), seed);
        let mut z = vec![[0u8; R]; m];
        for i in (0..m).rev() { z[i] = if self.oc[i] { let mut a = [0u32; R]; let p = &self.pc[i]; (1..64.min(m - i)).filter(|&j| p[j] != 0).for_each(|j| { let (cj, zz) = (p[j] as u32, &z[i + j]); (0..R).for_each(|k| a[k] += cj * zz[k] as u32) }); std::array::from_fn(|k| ((self.pv[i][k] as u32 + 17 - a[k] % 17) % 17) as u8) } else { let r = [sm(&mut rs), sm(&mut rs)]; std::array::from_fn(|k| (((r[k / 12] >> (5 * (k % 12)) & 31) * 17) >> 5) as u8) }; }
        z
    }
}
pub struct Solve<const R: usize> { pub z: Vec<[u8; R]>, pub steps: u64, pub codes: Vec<u8>, pub bumped: Vec<usize> }
pub fn solve<const R: usize>(rows: &[(u32, u64, [u8; R])], m: usize, w: usize, seed: u64, bump: Option<(usize, [usize; 4])>) -> Option<Solve<R>> {
    let mut e = El::<R>::new(m);
    let Some((b, th)) = bump else { for r in rows { e.ins(r.0 as usize, r.1, w, r.2).ok()?; } return Some(Solve { z: e.back(seed), steps: e.steps, codes: vec![], bumped: vec![] }) };
    let (nb, mut i0) = ((m - w + 1).div_ceil(b), 0usize);
    let (mut codes, mut bumped) = (vec![0u8; nb], vec![]);
    for bk in 0..nb {
        let i1 = i0 + rows[i0..].partition_point(|r| (r.0 as usize) < (bk + 1) * b);
        let code = (0..4).find(|&c| { let mut wr = vec![]; let ok = rows[i0..i1].iter().filter(|r| r.0 as usize % b >= th[c]).all(|r| e.ins(r.0 as usize, r.1, w, r.2).map(|s| { s.map(|x| wr.push(x)); }).is_ok()); (!ok).then(|| wr.iter().for_each(|&x| e.oc[x] = false)); ok }).unwrap();
        codes[bk] = code as u8;
        (i0..i1).filter(|&j| rows[j].0 as usize % b < th[code]).for_each(|j| bumped.push(j));
        i0 = i1;
    }
    Some(Solve { z: e.back(seed), steps: e.steps, codes, bumped })
}
pub struct Built { pub dg: Vec<u8>, pub ms: Vec<u32>, pub codes: Vec<u8>, pub bumped: Vec<usize>, pub retries: u64, pub steps: u64, pub b: usize, pub th: [usize; 4] }
pub fn build<const R: usize, K: Copy + Sync>(keys: &[(u64, K)], dg: Dig<K, R>, w: usize, fd: usize, nsh: u64, eps: f64, seed: u64, bump: Option<(usize, [usize; 4])>, old: Option<(&Table<R>, &[bool])>) -> Built {
    let tab = Tab::new(seed);
    let mut rec: Vec<(u32, u32, u64, [u8; R], u32)> = keys.par_iter().enumerate().map(|(i, &(k, id))| { let h = tab.h(k); (((h[0] >> 32).wrapping_mul(nsh) >> 32) as u32, h[0] as u32, h[1], dg(&id, &h, fd), i as u32) }).collect();
    rec.par_sort_unstable_by_key(|r| r.0);
    let mut cut = vec![0usize; nsh as usize + 1];
    rec.iter().for_each(|r| cut[r.0 as usize + 1] += 1);
    (0..nsh as usize).for_each(|i| cut[i + 1] += cut[i]);
    let wm = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
    let out: Vec<(Solve<R>, u32, u64, Vec<usize>)> = (0..nsh as usize).into_par_iter().map(|j| {
        let rs = &rec[cut[j]..cut[j + 1]];
        let n = rs.len();
        if let Some((t, _)) = old.filter(|o| !o.1[j]) { let (m, b) = (t.ms[j] as usize, t.b); let cs = t.codes[t.boff[j] as usize..t.boff[j] as usize + if b > 0 { (m - w + 1).div_ceil(b) } else { 0 }].to_vec(); let z = t.digits(t.off[j] as usize * R, m * R).chunks(R).map(|c| c.try_into().unwrap()).collect(); let bi = rs.iter().filter(|r| { let s = fr(r.1 as u64, (m - w + 1) as u64) as usize; b > 0 && s % b < t.th[cs[s / b] as usize] }).map(|r| r.4 as usize).collect(); return (Solve { z, steps: 0, codes: cs, bumped: vec![] }, m as u32, 0, bi); }
        let (mut m, mut tries) = (((n as f64 * (1.0 + eps)).ceil() as usize).max(1) + w, 0u64);
        loop {
            let mut rows: Vec<(u32, u64, [u8; R], u32)> = rs.iter().map(|r| (fr(r.1 as u64, (m - w + 1) as u64) as u32, r.2 & wm, r.3, r.4)).collect();
            rows.sort_unstable_by_key(|r| r.0);
            let rr: Vec<(u32, u64, [u8; R])> = rows.iter().map(|r| (r.0, r.1, r.2)).collect();
            match solve::<R>(&rr, m, w, seed ^ (j as u64) << 20, bump) { Some(s) => { let bi = s.bumped.iter().map(|&x| rows[x].3 as usize).collect(); return (s, m as u32, tries, bi) } None => { tries += 1; m += (n / 200).max(1); } }
        }
    }).collect();
    let ms: Vec<u32> = out.iter().map(|o| o.1).collect();
    let (retries, steps) = (out.iter().map(|o| o.2).sum(), out.iter().map(|o| o.0.steps).sum());
    let (mut dg, mut codes, mut bumped) = (Vec::with_capacity(ms.iter().map(|&m| m as usize).sum::<usize>() * R), vec![], vec![]);
    out.into_iter().for_each(|o| { o.0.z.iter().for_each(|c| dg.extend_from_slice(c)); codes.extend(o.0.codes); bumped.extend(o.3); });
    let (b, th) = bump.unwrap_or((0, [0; 4]));
    Built { dg, ms, codes, bumped, retries, steps, b, th }
}
pub fn save(path: &str, b: &Built, w: usize, r: usize, fd: usize, seed: u64, n_keys: u64) -> io::Result<u64> {
    let pk = pack17(&b.dg);
    let dir: Vec<u8> = b.ms.iter().flat_map(|m| m.to_le_bytes()).chain(b.codes.chunks(4).map(|c| c.iter().enumerate().fold(0u8, |a, (i, &x)| a | x << (2 * i)))).collect();
    let mut h = vec![0u8; HDR];
    h[..8].copy_from_slice(MAGIC);
    [1 + (b.b > 0) as u32, w as u32, r as u32, fd as u32, b.ms.len() as u32, b.b as u32, b.th[0] as u32, b.th[1] as u32, b.th[2] as u32, b.th[3] as u32].iter().enumerate().for_each(|(i, x)| h[8 + 4 * i..12 + 4 * i].copy_from_slice(&x.to_le_bytes()));
    [n_keys, b.ms.iter().map(|&m| m as u64).sum(), seed, b.dg.len() as u64, pk.len() as u64, (HDR + pk.len()) as u64, dir.len() as u64, b.codes.len() as u64].iter().enumerate().for_each(|(i, x)| h[48 + 8 * i..56 + 8 * i].copy_from_slice(&x.to_le_bytes()));
    h[112..116].copy_from_slice(&hash(&dir).to_le_bytes());
    let x = hash(&h[..116]);
    h[116..120].copy_from_slice(&x.to_le_bytes());
    let f = OpenOptions::new().create(true).write(true).truncate(true).open(path)?;
    f.write_all_at(&h, 0)?;
    f.write_all_at(&pk, HDR as u64)?;
    f.write_all_at(&dir, (HDR + pk.len()) as u64)?;
    f.sync_all()?;
    Ok((HDR + pk.len() + dir.len()) as u64)
}
pub struct Table<const R: usize> { pub map: Mmap, f: File, pub w: usize, pub fd: usize, pub nsh: u64, pub ms: Vec<u32>, pub off: Vec<u64>, pub b: usize, lb: u32, pub th: [usize; 4], pub codes: Vec<u8>, pub boff: Vec<u64>, pub seed: u64, tab: Tab, lut: Vec<u32>, pub n_keys: u64, pub cells: u64, pub nbytes: usize, pub bytes: u64, wm: u64, pub next: Option<Box<Table<R>>> }
impl<const R: usize> Table<R> {
    pub fn open(path: &str) -> io::Result<Table<R>> {
        let f = File::open(path)?;
        let map = unsafe { Mmap::map(&f)? };
        let u = |o: usize| u32::from_le_bytes(map[o..o + 4].try_into().unwrap()) as usize;
        let q = |o: usize| u64::from_le_bytes(map[o..o + 8].try_into().unwrap());
        (map.len() >= HDR && &map[..8] == MAGIC && hash(&map[..116]) as usize == u(116) && u(16) == R).then_some(()).ok_or_else(|| bad(format!("{path}: not a GF17RIB1 table with {R} digits per cell")))?;
        let (ver, w, fd, nsh, b, th) = (u(8), u(12), u(20), u(24) as u64, u(28), [u(32), u(36), u(40), u(44)]);
        let (n_keys, cells, seed, nbytes, doff, dlen, nc) = (q(48), q(56), q(64), q(80) as usize, q(88) as usize, q(96) as usize, q(104) as usize);
        let dir = &map[doff..doff + dlen];
        (hash(dir) as usize == u(112)).then_some(()).ok_or_else(|| bad(format!("{path}: shard directory corrupt")))?;
        let ms: Vec<u32> = dir[..4 * nsh as usize].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let codes: Vec<u8> = (0..nc).map(|i| dir[4 * nsh as usize + i / 4] >> (2 * (i % 4)) & 3).collect();
        let off: Vec<u64> = ms.iter().scan(0u64, |a, &m| { let r = *a; *a += m as u64; Some(r) }).collect();
        let boff: Vec<u64> = ms.iter().scan(0u64, |a, &m| { let r = *a; *a += if b > 0 { (m as usize - w + 1).div_ceil(b) as u64 } else { 0 }; Some(r) }).collect();
        unsafe { libc::madvise(map.as_ptr() as *mut _, map.len(), libc::MADV_RANDOM) };
        let next = if ver == 2 { Some(Box::new(Table::open(&format!("{path}.l1"))?)) } else { None };
        let bytes = map.len() as u64 + next.as_ref().map_or(0, |t| t.bytes);
        Ok(Table { map, f, w, fd, nsh, ms, off, b, lb: b.trailing_zeros(), th, codes, boff, seed, tab: Tab::new(seed), lut: lut17(), n_keys, cells, nbytes, bytes, wm: if w == 64 { u64::MAX } else { (1u64 << w) - 1 }, next })
    }
    #[inline(always)]
    pub fn data(&self) -> &[u8] { &self.map[HDR..HDR + self.nbytes] }
    #[inline(always)]
    pub fn band(&self, s: u64, c: u64) -> [u8; R] {
        let g = s as usize * R;
        let (p0, o, b) = (g >> 2, g & 3, self.data());
        let np = (o + R * self.w + 3) >> 2;
        let mut sc = [0u8; SC];
        (0..np).for_each(|i| sc[4 * i..4 * i + 4].copy_from_slice(&self.lut[px17(b, p0 + i) as usize].to_le_bytes()));
        let (mut a, mut cc) = ([0u16; R], c);
        while cc != 0 { let j = o + R * cc.trailing_zeros() as usize; cc &= cc - 1; (0..R).for_each(|k| a[k] += sc[j + k] as u16); }
        a.map(|x| M17[x as usize])
    }
    #[inline(always)]
    pub fn get(&self, key: u64) -> Option<u32> { self.row(key).map(|d| unid(&d)) }
    #[inline(always)]
    pub fn row(&self, key: u64) -> Option<[u8; R]> {
        let h = self.tab.h(key);
        let sh = ((h[0] >> 32).wrapping_mul(self.nsh) >> 32) as usize;
        let sl = fr(h[0], self.ms[sh] as u64 - self.w as u64 + 1);
        if self.b > 0 && ((sl & (self.b as u64 - 1)) as usize) < self.th[self.codes[(self.boff[sh] + (sl >> self.lb)) as usize] as usize] { return self.next.as_ref().and_then(|t| t.row(key)); }
        let d = self.band(self.off[sh] + sl, h[1] & self.wm | 1);
        (0..self.fd).all(|k| d[R - self.fd + k] == fpd(&h, k)).then_some(d)
    }
    pub fn digits(&self, a: usize, n: usize) -> Vec<u8> { (a..a + n).map(|g| self.lut[px17(self.data(), g >> 2) as usize].to_le_bytes()[g & 3]).collect() }
    pub fn shard_of(&self, key: u64) -> usize { ((self.tab.h(key)[0] >> 32).wrapping_mul(self.nsh) >> 32) as usize }
    pub fn pages(&self) -> usize { (self.cells as usize * R).div_ceil(4).div_ceil(PX) }
    pub fn all_cells(&self) -> u64 { self.cells + self.next.as_ref().map_or(0, |t| t.all_cells()) }
    pub fn rgba(&self, pg: usize) -> Vec<u8> { adam_gf17::page17(self.data(), self.cells as usize * R, pg) }
    pub fn cold(&self) { unsafe { libc::madvise(self.map.as_ptr() as *mut _, self.map.len(), libc::MADV_DONTNEED); libc::posix_fadvise(self.f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) }; self.next.as_ref().map(|t| t.cold()); }
}
pub fn keyof(s: u32, p: u16, idx: u16) -> u64 { s as u64 | (p as u64) << 24 | (idx as u64) << 40 }
pub fn unkey(k: u64) -> (u32, u16, u16) { ((k & 0xFF_FFFF) as u32, (k >> 24) as u16, (k >> 40) as u16) }
pub fn layered<const R: usize>(path: &str, keys: &[(u64, u32)], w: usize, fd: usize, shard: usize, eps: f64, seed: u64, bump: Option<(usize, [usize; 4])>, eps1: f64, old: Option<(&Table<R>, &[bool])>) -> io::Result<(Built, Option<Built>, u64)> { layered_by::<R, u32>(path, keys, vrow::<R>, w, fd, shard, eps, seed, bump, eps1, old) }
pub fn layered_by<const R: usize, K: Copy + Sync>(path: &str, keys: &[(u64, K)], dg: Dig<K, R>, w: usize, fd: usize, shard: usize, eps: f64, seed: u64, bump: Option<(usize, [usize; 4])>, eps1: f64, old: Option<(&Table<R>, &[bool])>) -> io::Result<(Built, Option<Built>, u64)> {
    let b0 = build::<R, K>(keys, dg, w, fd, old.map_or(keys.len().div_ceil(shard).max(1) as u64, |o| o.0.nsh), eps, seed, bump, old);
    let n0 = keys.len() as u64 - b0.bumped.len() as u64;
    let mut by = save(path, &b0, w, R, fd, seed, n0)?;
    let b1 = bump.map(|_| { let ks: Vec<(u64, K)> = b0.bumped.iter().map(|&i| keys[i]).collect(); let b1 = build::<R, K>(&ks, dg, w, fd, ks.len().div_ceil(shard).max(1) as u64, eps1, seed ^ 0x5EED, None, None); save(&format!("{path}.l1"), &b1, w, R, fd, seed ^ 0x5EED, ks.len() as u64).map(|x| { by += x; b1 }) }).transpose()?;
    Ok((b0, b1, by))
}
pub struct Live<const R: usize, const Q: usize> { pub base: Table<R>, pub ov: Option<Table<Q>> }
impl<const R: usize, const Q: usize> Live<R, Q> {
    pub fn get(&self, key: u64) -> Option<u32> { match self.ov.as_ref().and_then(|o| o.get(key)) { Some(TOMB) => None, Some(v) => Some(v), None => self.base.get(key) } }
    pub fn values(&self, s: u32, p: u16) -> Vec<u32> { let mut o = vec![]; let mut i = 0u16; loop { let k = keyof(s, p, i); match self.ov.as_ref().and_then(|t| t.get(k)) { Some(TOMB) => {} Some(v) => o.push(v), None => match self.base.get(k) { Some(v) => o.push(v), None => return o } } i += 1; } }
}
