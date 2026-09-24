use crate::{cfg::*, v19::{Drv19, CG}, Engine, Part};
use adam_ray::{cpu::{hop, init, plan, top3, update}, pack::{lexicon, Lex}};
use rayon::prelude::*;
pub struct Tab { pub dp: Vec<f32>, pub dm: Vec<f32>, pub dv: Vec<f32>, pub sid: Vec<u8>, pub sw: Vec<f32>, pub sm: Vec<f32>, pub sv: Vec<f32>, pub bias: Vec<f32>, pub lq: Vec<f32> }
pub struct Cpu { pub c: Cfg, pub lx: Lex, pub t: Tab, pi: Vec<u32>, pw: Vec<f32>, pub lg: Vec<f32>, cpb: Vec<i32>, pub d19: Option<Drv19> }
#[derive(Clone, Copy)]
struct P<T>(*mut T);
unsafe impl<T> Send for P<T> {}
unsafe impl<T> Sync for P<T> {}
impl<T> P<T> { fn at(self, i: usize) -> &'static mut T { unsafe { &mut *self.0.add(i) } } }
pub fn sid0(n: usize) -> Vec<u8> { (0..n * KS).map(|i| [32, 101, 116][i % KS]).collect() }
pub fn qw(c: &Cfg, lq: &[f32], l: usize, w: &[f32]) -> [f32; 3] {
    let mx = w.iter().fold(0f32, |m, x| m.max(x.abs()));
    let lg = mx.max(1e-30).ln() * std::f32::consts::LOG2_E;
    let mq = if !c.sgrid { mx } else if lg < c.slo { 0.0 } else { 2f32.powf(c.slo + ((lg.min(c.shi) - c.slo) / (c.shi - c.slo) * 14.0).round() * (c.shi - c.slo) / 14.0) };
    std::array::from_fn(|s| { let (x, a) = (w[s], w[s].abs()); match c.tern { 0 => x, 1 => { let m5 = if c.qlev == 3 || a >= 0.75 * mx { mq } else { 0.5 * mq }; if a < c.tern_th * mx { 0.0 } else if x > 0.0 { m5 } else { -m5 } } _ => if a < lq[l] { 0.0 } else if x > 0.0 { lq[c.lv + l] } else { -lq[c.lv + l] } } })
}
pub fn drivers(c: &Cfg, lx: &Lex, x: &[u8], t: usize, salt: u32, pi: &mut [u32], pw: &mut [f32]) {
    let (lv, mut g) = (c.lv, c.geo.clone());
    g.salt = salt;
    pi.par_chunks_mut(t * lv * 4).zip(pw.par_chunks_mut(t * lv * 4)).zip(x.par_chunks(t)).for_each(|((pi, pw), xs)| {
        let mut s = init();
        for (q, &bb) in xs.iter().enumerate() {
            s = update(lx, &g, &s, bb as u32);
            let (pi, pw) = (&mut pi[q * lv * 4..(q + 1) * lv * 4], &mut pw[q * lv * 4..(q + 1) * lv * 4]);
            pi.fill(0);
            pw.fill(0.0);
            (pi[0], pi[4], pw[0], pw[4]) = (s[0], s[1] * 256 + s[0], 1.0, 1.0);
            plan(&g, &s).into_iter().for_each(|(l, k, e, w)| (pi[l * 4 + k], pw[l * 4 + k]) = (e, w));
        }
    });
}
impl Cpu {
    pub fn new(c: &Cfg) -> Cpu {
        let (n, lv) = (c.lv * c.te, c.lv);
        Cpu { lx: lexicon(&c.syn, c.geo.qt.then_some(c.wcnt.as_str())), t: Tab { dp: vec![0.0; 2 * DE * V], dm: vec![0.0; 2 * DE * V], dv: vec![0.0; 2 * DE * V], sid: sid0(n), sw: vec![0.0; n * KS], sm: vec![0.0; n * KS], sv: vec![0.0; n * KS], bias: vec![0.0; V], lq: vec![0.0; 2 * lv] }, c: c.clone(), pi: vec![], pw: vec![], lg: vec![], cpb: vec![], d19: (c.ver == 19).then(|| Drv19::new(c.bs)) }
    }
    fn forward(&mut self, n: usize, y: &[u8], mode: u32, salt: u32) -> Vec<f32> {
        let (c, t, lv, te, hb, c0, cp, mut g) = (&self.c, &self.t, self.c.lv, self.c.te, self.c.geo.hb(), self.c.geo.c0(), self.c.copy, self.c.geo.clone());
        g.salt = salt;
        let mut pl = vec![0f32; n];
        let cpb = &self.cpb;
        self.lg.par_chunks_mut(V).zip(self.pi.par_chunks_mut(lv * 4)).zip(self.pw.par_chunks_mut(lv * 4)).zip(pl.par_iter_mut()).enumerate().for_each(|(q, (((lg, pi), pw), pl))| {
            let (b1, e1) = (pi[0] as usize, pi[4] as usize);
            (0..V).for_each(|k| lg[k] = t.bias[k] + pw[0] * t.dp[b1 * V + k] + pw[4] * t.dp[(DE + e1) * V + k]);
            let add = |lg: &mut [f32], l: usize, e: usize, w: f32| { let o = (l * te + e) * KS; let v = qw(c, &t.lq, l, &t.sw[o..o + KS]); (0..KS).for_each(|s| lg[t.sid[o + s] as usize] += w * v[s]); };
            for j in 8..c0 * 4 { if pw[j] != 0.0 { add(lg, j / 4, pi[j] as usize, pw[j]); } }
            for k in 0..cp {
                let (cb, l) = (cpb[q * cp + k], c0 + k);
                let w0 = |j: usize| t.sw[(l * te + pi[l * 4 + j] as usize) * KS];
                if cb >= 0 { lg[cb as usize] += if cp >= 13 { CG * (0..4).fold(0f32, |g, j| g + pw[l * 4 + j] * w0(j)) } else { CG * pw[l * 4] * w0(0) }; }
            }
            for j in (c.geo.mb() * 4..lv * 4).step_by(4) { if pw[j] != 0.0 { add(lg, j / 4, pi[j] as usize, pw[j]); } }
            for h in 0..c.geo.hops {
                let (j, e) = ((hb + h) * 4, hop(&g, top3(lg), b1 as u32, (e1 / 256) as u32, h));
                (pi[j], pw[j]) = (e, 1.0);
                add(lg, hb + h, e as usize, 1.0);
            }
            if mode == 2 { return; }
            let mx = (0..8).map(|g| lg[g * 32..g * 32 + 32].iter().fold(-1e30f32, |m, &x| m.max(x))).fold(-1e30f32, f32::max);
            let ss = (0..8).map(|g| lg[g * 32..g * 32 + 32].iter().fold(0f32, |z, &x| z + (x - mx).exp())).fold(0f32, |z, x| z + x);
            let yq = y[q] as usize;
            *pl = -(lg[yq] - mx - ss.ln());
            (mode == 0).then(|| (0..V).for_each(|k| lg[k] = ((lg[k] - mx).exp() / ss - if k == yq { 1.0 } else { 0.0 }) / n as f32));
        });
        pl
    }
    fn backward(&mut self, n: usize, y: &[u8], it: usize, lr: f32) {
        let (c, lv, te) = (self.c.clone(), self.c.lv, self.c.te);
        let (pi, pw, d, cpb, c0, cp) = (&self.pi, &self.pw, &self.lg, &self.cpb, self.c.geo.c0(), self.c.copy);
        let mut ks: Vec<(u32, u32)> = (0..n * lv * 4).into_par_iter().filter(|&j| pw[j] != 0.0).map(|j| (((j / 4 % lv) * te) as u32 + pi[j], j as u32)).collect();
        ks.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        let mut seg: Vec<usize> = (0..ks.len()).into_par_iter().filter(|&i| i == 0 || ks[i].0 != ks[i - 1].0).collect();
        seg.push(ks.len());
        let (b1, b2, eps) = (0.9f32, 0.99f32, 1e-8f32);
        let (c1, c2, t) = (1.0 - b1.powf(it as f32), 1.0 - b2.powf(it as f32), &mut self.t);
        let (dp, dm, dv, sid, sw, sm, sv) = (P(t.dp.as_mut_ptr()), P(t.dm.as_mut_ptr()), P(t.dv.as_mut_ptr()), P(t.sid.as_mut_ptr()), P(t.sw.as_mut_ptr()), P(t.sm.as_mut_ptr()), P(t.sv.as_mut_ptr()));
        seg.par_windows(2).for_each(|r| {
            let (key, js) = (ks[r[0]].0 as usize, &ks[r[0]..r[1]]);
            let (l, e) = (key / te, key % te);
            if l < 2 {
                let mut g = [0f32; V];
                js.iter().for_each(|&(_, j)| { let (q, w) = (j as usize / (lv * 4), pw[j as usize]); (0..V).for_each(|k| g[k] += w * d[q * V + k]); });
                (0..V).for_each(|k| {
                    let i = (l * DE + e) * V + k;
                    let m = b1 * *dm.at(i) + (1.0 - b1) * g[k];
                    let v = b2 * *dv.at(i) + (1.0 - b2) * g[k] * g[k];
                    (*dm.at(i), *dv.at(i)) = (m, v);
                    *dp.at(i) -= lr * (m / c1) / ((v / c2).sqrt() + eps);
                });
                return;
            }
            let o = key * KS;
            let ids: [u8; 3] = std::array::from_fn(|s| *sid.at(o + s));
            let mut ga = [0f32; 3];
            let cpy = (c0..c0 + cp).contains(&l);
            js.iter().for_each(|&(_, j)| { let (q, w) = (j as usize / (lv * 4), pw[j as usize]); if cpy { let cb = cpb[q * cp + l - c0]; if cb >= 0 { ga[0] += CG * w * d[q * V + cb as usize]; } } else { (0..KS).for_each(|s| ga[s] += w * d[q * V + ids[s] as usize]); } });
            (0..KS).for_each(|s| {
                let g = ga[s].min(4.0).max(-4.0);
                let m = b1 * *sm.at(o + s) + (1.0 - b1) * g;
                let v = b2 * *sv.at(o + s) + (1.0 - b2) * g * g;
                let w = *sw.at(o + s) * (1.0 - c.wd * lr) - lr * (m / c1) / ((v / c2).sqrt() + eps);
                let ok = !m.is_nan() && !v.is_nan() && !w.is_nan() && w.abs() < 1e6;
                (*sm.at(o + s), *sv.at(o + s), *sw.at(o + s)) = if ok { (m, v, w) } else { (0.0, 0.0, 0.0) };
            });
            if let Some(&(_, j)) = js.iter().find(|&&(_, j)| !cpy && !ids.contains(&y[j as usize / (lv * 4)])) {
                let (smin, vmin) = (1..KS).fold((0, sw.at(o).abs()), |(b, v), s| if sw.at(o + s).abs() < v { (s, sw.at(o + s).abs()) } else { (b, v) });
                (vmin < c.rth).then(|| { (*sid.at(o + smin), *sw.at(o + smin), *sm.at(o + smin), *sv.at(o + smin)) = (y[j as usize / (lv * 4)], c.rw, 0.0, 0.0); });
            }
        });
        let mut gb = [0f32; V];
        d[..self.c.npos() * V].chunks(V).for_each(|r| (0..V).for_each(|k| gb[k] += r[k]));
        (0..V).for_each(|k| t.bias[k] -= lr * 0.1 * gb[k]);
    }
}
impl Cpu {
    fn plan(&mut self, x: &[u8], t: usize, salt: u32, n: usize, anc: bool) {
        let (lv, cp) = (self.c.lv, self.c.copy.max(1));
        (self.pi.len() < n * lv * 4).then(|| { self.pi.resize(n * lv * 4, 0); self.pw.resize(n * lv * 4, 0.0); self.cpb.resize(n * cp, -1); });
        (self.lg.len() < n.max(self.c.npos()) * V).then(|| self.lg.resize(n.max(self.c.npos()) * V, 0.0));
        match self.d19.as_mut() { Some(d) => d.run(&self.c, &self.lx, x, t, salt, anc, &mut self.pi[..n * lv * 4], &mut self.pw[..n * lv * 4], &mut self.cpb[..n * cp]), None => drivers(&self.c, &self.lx, x, t, salt, &mut self.pi[..n * lv * 4], &mut self.pw[..n * lv * 4]) }
    }
}
impl Engine for Cpu {
    fn run(&mut self, x: &[u8], y: &[u8], nb: usize, t: usize, salt: u32, train: Option<(usize, f32)>) -> f64 {
        let n = nb * t;
        self.plan(x, t, salt, n, false);
        let pl = self.forward(n, y, if train.is_some() { 0 } else { 1 }, salt);
        let loss = pl.iter().fold(0f32, |a, &b| a + b) as f64 / n as f64;
        if let Some((it, lr)) = train { (self.c.tern > 0 && it % 100 == 1).then(|| self.refresh()); self.backward(n, y, it, lr); }
        loss
    }
    fn logits(&mut self, x: &[u8]) -> Vec<f32> {
        let n = x.len();
        self.plan(x, n, 0, n, true);
        self.forward(n, x, 2, 0);
        self.lg[(n - 1) * V..n * V].to_vec()
    }
    fn refresh(&mut self) {
        let (te, th, t) = (self.c.te, self.c.tern_th, &mut self.t);
        let r: Vec<(f32, f32)> = t.sw.par_chunks(te * KS).map(|w| { let m = w.iter().fold(0f32, |a, x| a + x.abs()) / (te * KS) as f32; let tt = th * m; let (acc, cnt) = w.iter().fold((0f32, 0i32), |(a, n), x| if x.abs() >= tt { (a + x.abs(), n + 1) } else { (a, n) }); (tt, acc / cnt.max(1) as f32) }).collect();
        let lv = r.len();
        r.into_iter().enumerate().for_each(|(l, (a, b))| (t.lq[l], t.lq[lv + l]) = (a, b));
    }
    fn get(&mut self, p: Part) -> Vec<f32> { let t = &self.t; match p { Part::Dp => t.dp.clone(), Part::Dm => t.dm.clone(), Part::Dv => t.dv.clone(), Part::Sid => t.sid.iter().map(|&v| v as f32).collect(), Part::Sw => t.sw.clone(), Part::Sm => t.sm.clone(), Part::Sv => t.sv.clone(), Part::Bias => t.bias.clone(), Part::Lq => t.lq.clone() } }
    fn put(&mut self, p: Part, v: &[f32]) { let t = &mut self.t; match p { Part::Dp => t.dp.copy_from_slice(v), Part::Sid => t.sid.iter_mut().zip(v).for_each(|(a, &b)| *a = b as u8), Part::Sw => t.sw.copy_from_slice(v), Part::Bias => t.bias.copy_from_slice(v), _ => unreachable!() } }
    fn name(&self) -> String { format!("cpu x{}", rayon::current_num_threads()) }
}
