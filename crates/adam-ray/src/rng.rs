use std::{collections::hash_map::RandomState, hash::{BuildHasher, Hasher}};
const MUL: u128 = 0x2360ED051FC65DA44385DF649FCCF645;
pub struct Pcg64 { s: u128, inc: u128, half: Option<u32> }
fn hmix(v: u32, h: &mut u32) -> u32 { let mut v = v ^ *h; *h = h.wrapping_mul(0x931e8875); v = v.wrapping_mul(*h); v ^ v >> 16 }
fn mix(x: u32, y: u32) -> u32 { let r = 0xca01f9ddu32.wrapping_mul(x).wrapping_sub(0x4973f715u32.wrapping_mul(y)); r ^ r >> 16 }
pub fn pairwise<T: Copy + Default + std::ops::Add<Output = T>>(a: &[T]) -> T {
    let n = a.len();
    if n < 8 { return a.iter().fold(T::default(), |r, &x| r + x); }
    if n > 128 { let h = n / 2 - n / 2 % 8; return pairwise(&a[..h]) + pairwise(&a[h..]); }
    let mut r: [T; 8] = std::array::from_fn(|j| a[j]);
    let m = n - n % 8;
    (8..m).step_by(8).for_each(|i| (0..8).for_each(|j| r[j] = r[j] + a[i + j]));
    a[m..].iter().fold(((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7])), |s, &x| s + x)
}
impl Pcg64 {
    pub fn new(seed: Option<u64>) -> Pcg64 {
        let sd = seed.unwrap_or_else(|| { let mut h = RandomState::new().build_hasher(); h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos())); h.finish() });
        let ent: Vec<u32> = if sd >> 32 > 0 { vec![sd as u32, (sd >> 32) as u32] } else { vec![sd as u32] };
        let mut hc = 0x43b0d7e5u32;
        let mut pool: [u32; 4] = std::array::from_fn(|i| hmix(*ent.get(i).unwrap_or(&0), &mut hc));
        for s in 0..4 { for d in 0..4 { if s != d { pool[d] = mix(pool[d], hmix(pool[s], &mut hc)); } } }
        for &e in ent.iter().skip(4) { for d in 0..4 { pool[d] = mix(pool[d], hmix(e, &mut hc)); } }
        let mut hb = 0x8b51f9ddu32;
        let w: Vec<u64> = (0..8).map(|i| { let mut v = pool[i % 4] ^ hb; hb = hb.wrapping_mul(0x58f38ded); v = v.wrapping_mul(hb); (v ^ v >> 16) as u64 }).collect();
        let v: Vec<u128> = (0..4).map(|k| (w[2 * k] | w[2 * k + 1] << 32) as u128).collect();
        let mut r = Pcg64 { s: 0, inc: (v[2] << 64 | v[3]) << 1 | 1, half: None };
        r.step();
        r.s = r.s.wrapping_add(v[0] << 64 | v[1]);
        r.step();
        r
    }
    fn step(&mut self) { self.s = self.s.wrapping_mul(MUL).wrapping_add(self.inc); }
    pub fn next64(&mut self) -> u64 { self.step(); ((self.s >> 64) as u64 ^ self.s as u64).rotate_right((self.s >> 122) as u32) }
    pub fn next32(&mut self) -> u32 { self.half.take().unwrap_or_else(|| { let n = self.next64(); self.half = Some((n >> 32) as u32); n as u32 }) }
    pub fn random(&mut self) -> f64 { (self.next64() >> 11) as f64 * (1.0 / 9007199254740992.0) }
    pub fn integers(&mut self, lo: i64, hi: i64, n: usize) -> Vec<i64> {
        let rng = (hi - lo - 1) as u64;
        assert!(rng < 0xFFFFFFFF, "range beyond 32-bit lemire path");
        let ex = rng as u32 + 1;
        (0..n).map(|_| { let mut m = self.next32() as u64 * ex as u64; if (m as u32) < ex { let th = (u32::MAX - rng as u32) % ex; while (m as u32) < th { m = self.next32() as u64 * ex as u64; } } lo + (m >> 32) as i64 }).collect()
    }
    pub fn choice(&mut self, p: &[f64]) -> usize {
        let cdf: Vec<f64> = p.iter().scan(0.0, |s, &x| { *s += x; Some(*s) }).collect();
        let (last, u) = (cdf[cdf.len() - 1], self.random());
        cdf.iter().position(|&c| c / last > u).unwrap_or(cdf.len() - 1)
    }
}
