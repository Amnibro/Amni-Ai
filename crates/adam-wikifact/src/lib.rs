pub const KEEP: [u16; 189] = [6, 17, 19, 20, 21, 22, 25, 26, 27, 30, 31, 35, 36, 37, 38, 39, 40, 50, 54, 57, 58, 61, 69, 84, 85, 86, 101, 102, 106, 108, 110, 112, 115, 118, 122, 123, 127, 131, 136, 138, 140, 144, 155, 156, 159, 161, 162, 166, 169, 170, 175, 176, 178, 179, 206, 246, 264, 272, 274, 276, 279, 286, 361, 364, 397, 403, 407, 413, 449, 451, 452, 463, 488, 495, 509, 527, 569, 570, 571, 575, 576, 577, 580, 582, 585, 610, 641, 664, 674, 676, 710, 725, 740, 742, 749, 750, 800, 840, 885, 915, 991, 1037, 1056, 1080, 1082, 1086, 1113, 1120, 1128, 1132, 1191, 1303, 1308, 1346, 1365, 1366, 1376, 1431, 1433, 1441, 1449, 1476, 1477, 1545, 1589, 1619, 1813, 1923, 2043, 2044, 2046, 2047, 2048, 2130, 2142, 2437, 3373, 3450, 4552, 4584, 4908, 619, 620, 729, 730, 171, 183, 186, 366, 425, 460, 461, 462, 516, 703, 737, 780, 828, 921, 941, 1034, 1060, 1269, 1478, 1479, 1535, 1536, 1537, 1542, 1552, 1557, 1582, 1672, 1995, 2012, 2079, 2175, 2176, 2283, 2578, 2579, 2670, 2974, 2975, 3094, 3095, 4330, 5642, 9714];
pub const QKEEP: [u16; 12] = [453, 518, 580, 582, 585, 642, 805, 1545, 1686, 2868, 3831, 4633];
pub const CAP: usize = 400;
pub const QCAP: usize = 16;
pub fn keep(p: u16, q: bool) -> bool { if q { QKEEP.contains(&p) } else { KEEP.contains(&p) } }
#[derive(Debug, Clone, PartialEq)]
pub enum Val { I(u32), T(i64, u8, u8, u8, bool), Q(f64, u32), S(String), M(String) }
pub fn ws(o: &mut Vec<u8>, s: &str) { let mut n = s.len().min(1000); while !s.is_char_boundary(n) { n -= 1 } o.extend_from_slice(&(n as u16).to_le_bytes()); o.extend_from_slice(&s.as_bytes()[..n]) }
pub fn wv(o: &mut Vec<u8>, v: &Val) {
    match v {
        Val::I(q) => { o.push(1); o.extend_from_slice(&q.to_le_bytes()) }
        Val::T(y, m, d, p, j) => { o.push(2); o.extend_from_slice(&y.to_le_bytes()); o.extend_from_slice(&[*m, *d, *p, *j as u8]) }
        Val::Q(a, u) => { o.push(3); o.extend_from_slice(&a.to_le_bytes()); o.extend_from_slice(&u.to_le_bytes()) }
        Val::S(s) => { o.push(4); ws(o, s) }
        Val::M(s) => { o.push(5); ws(o, s) }
    }
}
pub struct Rd<'a> { pub b: &'a [u8], pub i: usize }
impl<'a> Rd<'a> {
    pub fn n<const N: usize>(&mut self) -> [u8; N] { let a = self.b[self.i..self.i + N].try_into().unwrap(); self.i += N; a }
    pub fn u8(&mut self) -> u8 { self.n::<1>()[0] }
    pub fn u16(&mut self) -> u16 { u16::from_le_bytes(self.n()) }
    pub fn u32(&mut self) -> u32 { u32::from_le_bytes(self.n()) }
    pub fn i64(&mut self) -> i64 { i64::from_le_bytes(self.n()) }
    pub fn f64(&mut self) -> f64 { f64::from_le_bytes(self.n()) }
    pub fn s(&mut self) -> &'a str { let n = self.u16() as usize; let s = std::str::from_utf8(&self.b[self.i..self.i + n]).unwrap_or(""); self.i += n; s }
    pub fn v(&mut self) -> Val { match self.u8() { 1 => Val::I(self.u32()), 2 => { let y = self.i64(); let [m, d, p, j] = self.n(); Val::T(y, m, d, p, j == 1) } 3 => { let a = self.f64(); Val::Q(a, self.u32()) } 4 => Val::S(self.s().into()), _ => Val::M(self.s().into()) } }
    pub fn done(&self) -> bool { self.i >= self.b.len() }
}
pub fn tpack(y: i64, m: u8, d: u8, p: u8, j: bool) -> u64 { ((y + (1i64 << 39)) as u64) << 24 | (m as u64) << 16 | (d as u64) << 8 | (p as u64) << 1 | j as u64 }
pub fn tunpack(v: u64) -> (i64, u8, u8, u8, bool) { ((v >> 24) as i64 - (1i64 << 39), (v >> 16) as u8, (v >> 8) as u8, (v as u8) >> 1, v & 1 == 1) }
pub fn coord(w: u64) -> (u32, u32, u32, [u8; 4]) { let k = adam_gf17::K4; ((w >> 24) as u32, (w % 4096) as u32, (w >> 12 & 4095) as u32, [0, 1, 2, 3].map(|i| ((w & 0xFFFF) as u32 / k[i] % 17) as u8)) }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn codec_roundtrip() {
        let vs = [Val::I(42), Val::T(-500, 0, 0, 9, true), Val::T(1952, 3, 11, 11, false), Val::Q(2117522.0, 11573), Val::S("Zaphod".into()), Val::M("Hitchhiker's Guide".into())];
        let mut o = vec![];
        vs.iter().for_each(|v| wv(&mut o, v));
        let mut r = Rd { b: &o, i: 0 };
        vs.iter().for_each(|v| assert_eq!(&r.v(), v));
        assert!(r.done());
        assert_eq!(tunpack(tpack(-13_800_000_000, 0, 0, 3, false)), (-13_800_000_000, 0, 0, 3, false));
        assert_eq!(tunpack(tpack(1952, 3, 11, 11, true)), (1952, 3, 11, 11, true));
        let mut k = KEEP.to_vec();
        k.sort();
        k.dedup();
        assert_eq!(k.len(), KEEP.len());
        let (p, x, y, d) = coord(42 + (3 << 24) + (7 << 12));
        assert_eq!((p, x, y), (3, 42, 7));
        assert_eq!(d.iter().enumerate().map(|(i, &v)| v as u32 * adam_gf17::K4[i]).sum::<u32>(), ((42 + (7 << 12)) & 0xFFFF) as u32);
    }
}
