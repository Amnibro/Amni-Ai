use crate::pack::*;
use half::f16;
use rayon::prelude::*;
pub type St = [u32; NS];
const TC: f32 = (1.0f64 - 0.99) as f32;
const TAU: f32 = (2.0 * std::f64::consts::PI) as f32;
pub fn hm(a: u32, b: u32, s: u32) -> u32 { let mut h = a.wrapping_mul(73856093) ^ b.wrapping_mul(19349663) ^ s.wrapping_mul(83492791); h ^= h >> 13; h = h.wrapping_mul(0x5BD1E995); h ^ h >> 15 }
pub fn wordch(c: u32) -> bool { c < 128 && ((c as u8).is_ascii_alphanumeric() || c == 95) }
pub fn punct(c: u32) -> bool { c < 128 && b".,;:!?()[]{}<>=\"'`#-".contains(&(c as u8)) }
pub fn init() -> St { let mut s = [0; NS]; s[24] = 1; s }
pub fn update(tb: &Tables, p: &St, bb: u32) -> St {
    let mut s = *p;
    let (mut tx, mut ty) = (f32::from_bits(s[37]), f32::from_bits(s[38]));
    let [mut wh, mut pw1, mut pw2, mut pw3, mut pw4, mut s1, mut s2, mut g1, mut at, mut ind, mut g2, mut col, mut inc] = std::array::from_fn(|i| s[16 + i]);
    let (mut lastp, mut tick) = (s[29] & 0xFFFF, s[29] >> 16);
    let [mut inq, mut q1, mut q2, mut c1, mut c2, mut qon, mut wlen] = std::array::from_fn(|i| s[30 + i]);
    let isw = wordch(bb);
    if !isw && wh != 0 {
        (pw4, pw3, pw2, pw1, s2, g2) = (pw3, pw2, pw1, wh, s1, g1);
        let lx = tb.lex[(pw1 & WM) as usize];
        let [c, sn] = tb.trig[(pw1 % 360) as usize];
        (s1, g1, tx, ty) = (lx[0], lx[1], TOPIC_A * tx + c * TC * 8.0, TOPIC_A * ty + sn * TC * 8.0);
        let cw = lx[2];
        if tb.qt && inq == 1 && wlen >= 3 && !(bb == 58 && s[..4] == [109, 97, 100, 65]) && cw > 0 {
            if c1 == 0 || cw < c1 { (q2, c2, q1, c1) = (q1, c1, pw1, cw) } else if pw1 != q1 && (c2 == 0 || cw < c2) { (q2, c2) = (pw1, cw) }
        }
    }
    wlen = if tb.qt { if isw { wlen + 1 } else { 0 } } else { wlen };
    wh = if isw { wh.wrapping_mul(83492791).wrapping_add(bb + 1) } else { 0 };
    let nl = bb == 10;
    col = if nl { 0 } else { col + 1 };
    (at, ind) = if nl { (1, 0) } else if bb == 32 { (at, ind + (at == 1) as u32) } else { (0, ind) };
    tick = if bb == 96 { tick + 1 } else if nl { 0 } else { tick };
    (inc, tick) = if tick == 3 { (1 - inc, 0) } else { (inc, tick) };
    lastp = if punct(bb) { bb } else if nl { 10 } else { lastp };
    s.copy_within(0..15, 1);
    s[0] = bb;
    if tb.qt {
        (q1, q2, c1, c2, inq, qon) = if s[..6].iter().rev().copied().eq(b"User: ".iter().map(|&c| c as u32)) { (0, 0, 0, 0, 1, 0) } else { (q1, q2, c1, c2, inq, qon) };
        (inq, qon) = if s[..6].iter().rev().copied().eq(b"Adam: ".iter().map(|&c| c as u32)) { (0, (q1 != 0) as u32) } else { (inq, qon) };
        qon = if s[0] == 10 && s[1] == 10 { 0 } else { qon };
    }
    s[16..37].copy_from_slice(&[wh, pw1, pw2, pw3, pw4, s1, s2, g1, at, ind, g2, col, inc, lastp | tick << 16, inq, q1, q2, c1, c2, qon, wlen]);
    (s[37], s[38]) = (tx.to_bits(), ty.to_bits());
    s
}
pub fn plan(tb: &Tables, s: &St) -> Vec<(usize, u32, f32)> {
    let t = tb.t as u32;
    let sl = |u: u32, v: u32, l: usize| hm(u, v, l as u32 + 1) % t;
    let b: [u32; 16] = std::array::from_fn(|i| s[i]);
    let (wh, pw1, pw2, pw3, pw4, s1, s2, g1, g2) = (s[16], s[17], s[18], s[19], s[20], s[21], s[22], s[23], s[26]);
    let (ind, col, inc, lastp, q1, q2, qon) = (s[25], s[27], s[28], s[29] & 0xFFFF, s[31], s[32], s[35]);
    let h5 = hm(hm(b[4], b[3], 17), hm(b[2], b[1], 19), 21);
    let h6 = hm(h5, b[5], 23);
    let h8 = hm(hm(h6, b[6], 29), b[7], 31);
    let h10 = hm(hm(h8, b[8], 41), b[9], 43);
    let h12 = hm(hm(h10, b[10], 47), b[11], 53);
    let h16 = hm(hm(hm(hm(h12, b[12], 59), b[13], 61), b[14], 67), b[15], 71);
    let mut c: Vec<(usize, u32, f32)> = [(hm(b[2], b[1], 7), b[0]), (hm(hm(b[3], b[2], 9), b[1], 11), b[0]), (wh, pw1 & 255), (pw1 & 4095, pw2 & 4095), (hm(ind.min(32), inc * 512 + lastp, 13), col.min(255)), (h5, b[0]), (h6, b[0]), (h8, b[0]), (hm(pw1, pw2, 37) & 65535, pw3 & 4095), (h10, b[0]), (h12, b[0]), (h16, b[0]), (hm(hm(pw1, pw2, 73), hm(pw3, pw4, 79), 83), wh & 65535), (wh, 0), (hm(wh, pw1, 89), 0), (s1, s2), (g1, g2), (wh, s1)].iter().enumerate().map(|(i, &(u, v))| (i + 2, sl(u, v, i + 2), 1.0)).collect();
    let (tx, ty) = (f32::from_bits(s[37]), f32::from_bits(s[38]));
    let a = ty.atan2(tx) / TAU;
    let (uu, vv) = ((a - a.floor()) * 256.0, (tx * tx + ty * ty).sqrt().min(7.99) / 8.0 * 256.0);
    let (u0, v0) = (uu.floor(), vv.floor());
    let (fu, fv) = (uu - u0, vv - v0);
    for k in 0..4 {
        let (ii, jj, w) = (u0 as i32 + (k & 1), v0 as i32 + (k >> 1), if k & 1 == 1 { fu } else { 1.0 - fu } * if k >> 1 == 1 { fv } else { 1.0 - fv });
        (w != 0.0).then(|| c.push((BASE - 1, sl((ii % 256) as u32, jj.min(255) as u32, BASE - 1), w)));
    }
    (tb.qt && qon == 1).then(|| c.extend([(BASE, sl(q1, wh, BASE), 1.0), (BASE + 1, sl(hm(q1, q2, 107), pw1 & 4095, BASE + 1), 1.0), (BASE + 2, sl(q1, b[0] + (b[1] << 8), BASE + 2), 1.0)]));
    c
}
pub fn cell(tb: &Tables, lg: &mut [f32], l: usize, e: u32, w: f32) {
    let c = tb.cells[l * tb.t + e as usize];
    (0..3).for_each(|s| { let (i, v) = tap(c, s); lg[i] += w * v; });
}
pub fn logits(tb: &Tables, s: &St, lg: &mut [f32]) {
    let (b1, b2) = (s[0], s[1]);
    let (r1, r2) = (b1 as usize * V, (256 + b2 * 256 + b1) as usize * V);
    (0..V).for_each(|c| lg[c] = tb.bias[c] + f16::from_bits(tb.dp[r1 + c]).to_f32() + f16::from_bits(tb.dp[r2 + c]).to_f32());
    plan(tb, s).into_iter().for_each(|(l, e, w)| cell(tb, lg, l, e, w));
    let hb = BASE + if tb.qt { 3 } else { 0 };
    for h in 0..HOPS as u32 {
        let (mut a, mut m) = ([0u32; 3], [-1e30f32; 3]);
        for (c, &v) in lg.iter().enumerate() {
            (a, m) = if v > m[0] { ([c as u32, a[0], a[1]], [v, m[0], m[1]]) } else if v > m[1] { ([a[0], c as u32, a[1]], [m[0], v, m[1]]) } else if v > m[2] { ([a[0], a[1], c as u32], [m[0], m[1], v]) } else { (a, m) };
        }
        let (lo, hi) = (a[0].min(a[1]).min(a[2]), a[0].max(a[1]).max(a[2]));
        let e = hm(hm(hm(lo * 65536 + (a[0] + a[1] + a[2] - lo - hi) * 256 + hi, b1, 97 + h), b2, 103 + h), h, (BASE as u32 + h) + 1) % tb.t as u32;
        cell(tb, lg, hb + h as usize, e, 1.0);
    }
}
pub struct Cpu { pub tb: Tables, st: Vec<St> }
impl Cpu {
    pub fn new(tb: Tables) -> Cpu { Cpu { tb, st: vec![init(); 2 * MAXB] } }
    pub fn reset(&mut self, cur: usize, nb: usize) { self.st[cur * MAXB..cur * MAXB + nb].fill(init()); }
    pub fn state(&self, cur: usize, b: usize) -> St { self.st[cur * MAXB + b] }
    pub fn run(&mut self, cur: &mut usize, nb: usize, par: &[u32], byt: &[u8], want: &[bool]) -> Vec<f32> {
        let mut out = vec![0f32; want.iter().filter(|&&w| w).count() * nb * V];
        let mut o = 0;
        for (t, &w) in want.iter().enumerate() {
            let (lo, hi) = self.st.split_at_mut(MAXB);
            let (src, dst) = if *cur == 0 { (&*lo, hi) } else { (&*hi, lo) };
            let (tb, pa, by) = (&self.tb, &par[t * nb..t * nb + nb], &byt[t * nb..t * nb + nb]);
            let f = |(b, (d, lg)): (usize, (&mut St, Option<&mut [f32]>))| { *d = update(tb, &src[pa[b] as usize], by[b] as u32); if let Some(lg) = lg { logits(tb, d, lg) } };
            match w {
                true => dst[..nb].par_iter_mut().zip(out[o..o + nb * V].par_chunks_mut(V).map(Some)).enumerate().with_min_len(4).for_each(f),
                false => dst[..nb].par_iter_mut().map(|d| (d, None)).enumerate().with_min_len(4).for_each(f),
            }
            o += if w { nb * V } else { 0 };
            *cur = 1 - *cur;
        }
        out
    }
}
