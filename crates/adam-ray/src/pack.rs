use crate::npy::Npz;
use half::f16;
use rayon::prelude::*;
use std::{io, path::Path};
pub const V: usize = 256;
pub const WM: u32 = (1 << 20) - 1;
pub const NI: usize = 37;
pub const NS: usize = 40;
pub const BASE: usize = 21;
pub const HOPS: usize = 4;
pub const MAXB: usize = 256;
pub const TW: usize = 4096;
pub const TOPIC_A: f32 = 0.99;
#[derive(Clone)]
pub struct Geo { pub base: usize, pub hops: usize, pub t: u32, pub qt: bool, pub salt: u32, pub ta: f32, pub tc: f32, pub copy: usize }
pub struct Lex { pub lex: Vec<[u32; 4]>, pub trig: Vec<[f32; 2]> }
pub struct Tables { pub l: usize, pub t: usize, pub rows: usize, pub dp: Vec<u16>, pub cells: Vec<[u32; 2]>, pub bias: Vec<f32>, pub geo: Geo, pub lx: Lex }
impl Geo {
    pub fn eng(t: usize, qt: bool) -> Geo { Geo { base: BASE, hops: HOPS, t: t as u32, qt, salt: 0, ta: TOPIC_A, tc: (1.0f64 - 0.99) as f32, copy: 0 } }
    pub fn sl(&self, u: u32, v: u32, l: usize) -> u32 { crate::cpu::hm(u, v, (l as u32 + 1).wrapping_add(self.salt.wrapping_mul(1000003))) % self.t }
    pub fn c0(&self) -> usize { self.base + 3 * self.qt as usize }
    pub fn hb(&self) -> usize { self.c0() + self.copy }
    pub fn lv(&self) -> usize { self.hb() + self.hops }
    pub fn has(&self, l: usize) -> bool { l <= 6 || (l <= 10 && self.base + self.hops >= 12) || (l <= 14 && self.base + self.hops >= 16) || (l <= 16 && self.base >= 18) || (l <= 19 && self.base >= 21) }
}
fn h16(x: f32) -> u32 { f16::from_f32(x).to_bits() as u32 }
pub fn tap(c: [u32; 2], s: usize) -> (usize, f32) { let u = (c[0] >> 24) as i32; let t = [u / 25, u / 5 % 5, u % 5][s] - 2; let m = if t == 0 { 0.0 } else { f16::from_bits((c[1] >> (16 * (t.abs() - 1))) as u16).to_f32() }; ((c[0] >> (8 * s) & 255) as usize, if t < 0 { -m } else { m }) }
pub fn scale16(lo: f64, hi: f64) -> [f32; 16] { std::array::from_fn(|s| if s > 0 { 2f32.powf(lo as f32 + ((s as f32 - 1.0) / 14.0) * (hi - lo) as f32) } else { 0.0 }) }
pub fn load_syn(path: &str, n: usize) -> Vec<Vec<u32>> {
    let mut ts = vec![vec![0u32; WM as usize + 1]; n];
    let d = std::fs::read(path).unwrap_or_default();
    for l in d.split(|&c| c == b'\n').flat_map(|l| l.split(|&c| c == b'\r')) {
        let p: Vec<&[u8]> = l.split(|c| c.is_ascii_whitespace()).filter(|x| !x.is_empty()).collect();
        if p.len() != n + 1 || !p[1..].iter().all(|x| x.iter().all(u8::is_ascii_digit)) { continue; }
        let h = p[0].iter().fold(0u32, |h, &c| h.wrapping_mul(83492791).wrapping_add(c as u32 + 1));
        (0..n).for_each(|i| ts[i][(h & WM) as usize] = p[i + 1].iter().fold(0u32, |a, &c| a.wrapping_mul(10).wrapping_add((c - b'0') as u32)) & 0x7FFFFFFF);
    }
    ts
}
pub fn load(path: &str, syn: &str, wcnt: &str) -> io::Result<Tables> {
    let raw = path.ends_with("_sid.npy");
    let (l, t, rows, dp, cells, bias) = if raw {
        let pre = &path[..path.len() - 8];
        let (dz, sz, wz) = (Npz::open(&format!("{pre}_dp.npy"))?, Npz::open(path)?, Npz::open(&format!("{pre}_sw.npy"))?);
        let (d, s, w) = (dz.get("")?, sz.get("")?, wz.get("")?);
        let (l, t, k) = (s.shape[0], s.shape[1], s.shape[2]);
        assert!(k == 3 && d.shape[2] == V);
        let dv = d.vec::<u16>();
        let dp: Vec<u16> = dv[..V * V].iter().chain(&dv[d.shape[1] * V..2 * d.shape[1] * V]).copied().collect();
        let (sid, sw) = (s.data, w.vec::<u16>());
        let cells = (0..l * t).into_par_iter().map(|c| {
            let v: [f32; 3] = std::array::from_fn(|j| { let x = f16::from_bits(sw[c * 3 + j]).to_f32(); if x.is_nan() { 0.0 } else { x.clamp(f32::MIN, f32::MAX) } });
            let mx = v.iter().fold(0f32, |m, x| m.max(x.abs()));
            let tr = v.map(|x| if x.abs() < 0.5 * mx { 0 } else { (1 + (x.abs() >= 0.75 * mx) as i32) * ((x > 0.0) as i32 - (x < 0.0) as i32) });
            [sid[c * 3] as u32 | (sid[c * 3 + 1] as u32) << 8 | (sid[c * 3 + 2] as u32) << 16 | (((tr[0] + 2) * 25 + (tr[1] + 2) * 5 + tr[2] + 2) as u32) << 24, h16(0.5 * mx) | h16(mx) << 16]
        }).collect();
        (l, t, V + d.shape[1], dp, cells, Npz::open(&format!("{pre}_bias.npy"))?.get("")?.f32s())
    } else {
        let z = Npz::open(path)?;
        let (ss, n) = (z.get("sshape")?.i64s(), z.get("trit_n")?.i64s()[0] as usize);
        let (l, t) = (ss[0] as usize, ss[1] as usize);
        assert!(ss[2] == 3 && n == l * t);
        let (dd, ds) = (z.get("dense_digits")?, z.get("dense_scale")?.f32s());
        let dp: Vec<u16> = dd.data[..dd.n()].par_iter().enumerate().map(|(i, &q)| h16((q as i16 - 128) as f32 / 127.0 * ds[i / 8]) as u16).collect();
        let (ids, tr, s4, sc) = (z.get("ids")?.data, z.get("trits")?.data, z.get("scale4")?.data, scale16(z.get("scale_lo")?.f64(), z.get("scale_hi")?.f64()));
        let cells = (0..n).into_par_iter().map(|c| {
            let (g, k) = (c / 8 * 7, c % 8);
            let x = tr[g..g + 7].iter().fold(0u64, |a, &b| a << 8 | b as u64);
            let (u, h) = (((x >> (49 - 7 * k)) & 0x7F) as u32, 0.5 * sc[((s4[c / 2] >> (4 * (1 - c % 2))) & 15) as usize]);
            [ids[c * 3] as u32 | (ids[c * 3 + 1] as u32) << 8 | (ids[c * 3 + 2] as u32) << 16 | u << 24, h16(h) | h16(2.0 * h) << 16]
        }).collect();
        let bp = path.replace("_pk_q5_i8.npz", "_bias.npy");
        (l, t, dd.n() / V, dp, cells, if Path::new(&bp).exists() { Npz::open(&bp)?.get("")?.f32s() } else { vec![0.0; V] })
    };
    let qt = l == BASE + HOPS + 3;
    Ok(Tables { l, t, rows, dp, cells, bias, geo: Geo::eng(t, qt), lx: lexicon(syn, qt.then_some(wcnt)) })
}
pub fn lexicon(syn: &str, wcnt: Option<&str>) -> Lex {
    let (sy, wc) = (load_syn(syn, 2), wcnt.map_or_else(|| vec![0; WM as usize + 1], |w| load_syn(w, 1).remove(0)));
    let a = (2.0 * std::f64::consts::PI / 360.0) as f32;
    Lex { lex: (0..=WM as usize).map(|i| [sy[0][i], sy[1][i], wc[i], 0]).collect(), trig: (0..360).map(|k| { let x = k as f32 * a; [x.cos(), x.sin()] }).collect() }
}
