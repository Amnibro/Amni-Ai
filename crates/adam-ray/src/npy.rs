use memmap2::Mmap;
use std::{collections::HashMap, fs::File, io};
pub struct Npz { map: Mmap, ents: HashMap<String, usize> }
pub struct Arr<'a> { pub descr: String, pub shape: Vec<usize>, pub data: &'a [u8] }
fn bad(m: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, m.to_string()) }
fn u16a(b: &[u8], o: usize) -> usize { u16::from_le_bytes([b[o], b[o + 1]]) as usize }
fn u32a(b: &[u8], o: usize) -> usize { u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize }
fn u64a(b: &[u8], o: usize) -> usize { u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) as usize }
impl Npz {
    pub fn open(path: &str) -> io::Result<Npz> {
        let map = unsafe { Mmap::map(&File::open(path)?)? };
        let mut ents = HashMap::new();
        if path.ends_with(".npy") { ents.insert(String::new(), 0); return Ok(Npz { map, ents }); }
        let b = &map[..];
        let e = (0..b.len().saturating_sub(21)).rev().take(65600).find(|&i| u32a(b, i) == 0x06054b50).ok_or_else(|| bad("no zip eocd"))?;
        let (mut n, mut cd) = (u16a(b, e + 10), u32a(b, e + 16));
        (cd == 0xFFFFFFFF || n == 0xFFFF).then(|| { let z = u64a(b, e - 12); n = u64a(b, z + 32); cd = u64a(b, z + 48); });
        for _ in 0..n {
            let (m, nl, el, cl) = (u16a(b, cd + 10), u16a(b, cd + 28), u16a(b, cd + 30), u16a(b, cd + 32));
            let (us, cs, mut lo) = (u32a(b, cd + 24), u32a(b, cd + 20), u32a(b, cd + 42));
            let name = String::from_utf8_lossy(&b[cd + 46..cd + 46 + nl]).trim_end_matches(".npy").to_string();
            let (mut x, xe) = (cd + 46 + nl, cd + 46 + nl + el);
            while lo == 0xFFFFFFFF && x + 4 <= xe { let (id, sz) = (u16a(b, x), u16a(b, x + 2)); lo = if id == 1 { u64a(b, x + 4 + 8 * ((us == 0xFFFFFFFF) as usize + (cs == 0xFFFFFFFF) as usize)) } else { lo }; x += 4 + sz; }
            if m != 0 { return Err(bad("compressed npz entries unsupported: resave with np.savez")); }
            ents.insert(name, lo + 30 + u16a(b, lo + 26) + u16a(b, lo + 28));
            cd += 46 + nl + el + cl;
        }
        Ok(Npz { map, ents })
    }
    pub fn has(&self, k: &str) -> bool { self.ents.contains_key(k) }
    pub fn get(&self, k: &str) -> io::Result<Arr<'_>> {
        let o = *self.ents.get(k).ok_or_else(|| bad(&format!("missing array {k}")))?;
        let b = &self.map[o..];
        (&b[..6] == b"\x93NUMPY").then_some(()).ok_or_else(|| bad("bad npy magic"))?;
        let (hl, hs) = if b[6] == 1 { (u16a(b, 8), 10) } else { (u32a(b, 8), 12) };
        let h = String::from_utf8_lossy(&b[hs..hs + hl]).to_string();
        let f = |key: &str, end: char| h.find(key).map(|i| { let s = &h[i + key.len()..]; s[..s.find(end).unwrap()].to_string() }).unwrap_or_default();
        if h.contains("'fortran_order': True") { return Err(bad("fortran order unsupported")); }
        Ok(Arr { descr: f("'descr': '", '\''), shape: f("'shape': (", ')').split(',').filter_map(|v| v.trim().parse().ok()).collect(), data: &b[hs + hl..] })
    }
}
impl<'a> Arr<'a> {
    pub fn n(&self) -> usize { self.shape.iter().product() }
    pub fn vec<T: bytemuck::Pod>(&self) -> Vec<T> { bytemuck::pod_collect_to_vec(&self.data[..self.n() * std::mem::size_of::<T>()]) }
    pub fn f32s(&self) -> Vec<f32> {
        match &self.descr[1..] { "f2" => self.vec::<u16>().iter().map(|&v| half::f16::from_bits(v).to_f32()).collect(), "f8" => self.vec::<f64>().iter().map(|&v| v as f32).collect(), _ => self.vec::<f32>() }
    }
    pub fn i64s(&self) -> Vec<i64> {
        match &self.descr[1..] { "u1" => self.data[..self.n()].iter().map(|&v| v as i64).collect(), "i1" => self.data[..self.n()].iter().map(|&v| v as i8 as i64).collect(), "i4" => self.vec::<i32>().iter().map(|&v| v as i64).collect(), "f8" => self.vec::<f64>().iter().map(|&v| v as i64).collect(), _ => self.vec::<i64>() }
    }
    pub fn f64(&self) -> f64 { if &self.descr[1..] == "f8" { self.vec::<f64>()[0] } else { self.i64s()[0] as f64 } }
    pub fn objs(&self) -> io::Result<Vec<&'a [u8]>> {
        let (b, mut i, mut memo, mut last, mut inl, mut out): (&[u8], usize, Vec<Option<(usize, usize)>>, Option<(usize, usize)>, bool, Vec<&[u8]>) = (self.data, 0, vec![], None, false, Vec::with_capacity(self.n()));
        loop {
            let op = *b.get(i).ok_or_else(|| bad("pickle truncated"))?;
            i += 1;
            let mut pb = |s: usize, n: usize, i: &mut usize| { last = Some((s, n)); inl.then(|| out.push(&b[s..s + n])); *i = s + n; };
            match op {
                b'.' => break,
                0x80 => i += 1,
                0x95 => i += 8,
                0x8c => { let n = b[i] as usize; i += 1 + n; last = None; }
                b'X' => { let n = u32a(b, i); i += 4 + n; last = None; }
                0x8d => { let n = u64a(b, i); i += 8 + n; last = None; }
                b'C' => pb(i + 1, b[i] as usize, &mut i),
                b'B' => pb(i + 4, u32a(b, i), &mut i),
                0x8e => pb(i + 8, u64a(b, i), &mut i),
                0x94 => memo.push(last),
                b'q' => { let k = b[i] as usize; i += 1; memo.resize(memo.len().max(k + 1), None); memo[k] = last; }
                b'r' => { let k = u32a(b, i); i += 4; memo.resize(memo.len().max(k + 1), None); memo[k] = last; }
                b'h' | b'j' => { let k = if op == b'h' { i += 1; b[i - 1] as usize } else { i += 4; u32a(b, i - 4) }; last = memo.get(k).copied().flatten(); if let (true, Some((s, n))) = (inl, last) { out.push(&b[s..s + n]); } }
                b'K' => { i += 1; last = None; }
                b'M' => { i += 2; last = None; }
                b'J' => { i += 4; last = None; }
                0x8a => { i += 1 + b[i] as usize; last = None; }
                b']' => { inl = true; last = None; }
                0x93 | 0x85 | 0x86 | 0x87 | b't' | b')' | b'R' | b'N' | 0x88 | 0x89 | b'b' | b'e' | b'a' | b'(' | b'}' | b'u' | b's' => last = None,
                _ => return Err(bad(&format!("pickle opcode {op:#x} unsupported"))),
            }
        }
        (out.len() == self.n()).then_some(out).ok_or_else(|| bad("pickle object count mismatch"))
    }
}
