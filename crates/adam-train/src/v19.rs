use crate::cfg::*;
pub use adam_ray::v19::*;
use adam_ray::{cpu::plan, pack::{Geo, Lex}};
use rayon::prelude::*;
pub struct Drv19 { pub rows: Vec<W19>, pub lx: Lx19 }
impl Drv19 {
    pub fn new(bs: usize) -> Drv19 { Drv19 { rows: vec![W19::new(0); bs], lx: Lx19::new() } }
    #[allow(clippy::too_many_arguments)]
    pub fn run(&mut self, c: &Cfg, lex: &Lex, x: &[u8], t: usize, salt: u32, pi: &mut [u32], pw: &mut [f32], cpb: &mut [i32]) {
        let (lv, cp, mut g) = (c.lv, c.copy, c.geo.clone());
        g.salt = salt;
        let nb = x.len() / t;
        (self.rows.len() < nb).then(|| self.rows.resize(nb, W19::new(0)));
        let lx = &self.lx;
        self.rows[..nb].par_iter_mut().zip(pi.par_chunks_mut(t * lv * 4)).zip(pw.par_chunks_mut(t * lv * 4)).zip(cpb.par_chunks_mut(t * cp.max(1))).zip(x.par_chunks(t)).for_each(|((((w, pi), pw), cpb), xs)| drive(lv, &g, lex, lx, w, xs, pi, pw, cpb));
    }
}
#[allow(clippy::too_many_arguments)]
fn drive(lv: usize, g: &Geo, lex: &Lex, lx: &Lx19, w: &mut W19, xs: &[u8], pi: &mut [u32], pw: &mut [f32], cpb: &mut [i32]) {
    let (cp, c0) = (g.copy, g.c0());
    w.begin(cp);
    for (q, &bx) in xs.iter().enumerate() {
        w.step(g, lex, lx, bx);
        let (pi, pw, cpb) = (&mut pi[q * lv * 4..(q + 1) * lv * 4], &mut pw[q * lv * 4..(q + 1) * lv * 4], &mut cpb[q * cp.max(1)..(q + 1) * cp.max(1)]);
        pi.fill(0);
        pw.fill(0.0);
        (pi[0], pi[4], pw[0], pw[4]) = (w.s[0], w.s[1] * 256 + w.s[0], 1.0, 1.0);
        plan(g, &w.s).into_iter().for_each(|(l, k, e, v)| (pi[l * 4 + k], pw[l * 4 + k]) = (e, v));
        w.gates(g, |k, byte, e, n| { cpb[k] = byte; (0..n).for_each(|j| (pi[(c0 + k) * 4 + j], pw[(c0 + k) * 4 + j]) = (e[j], (byte >= 0) as u8 as f32)); });
    }
}
