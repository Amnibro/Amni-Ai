use crate::{cpu::{wordch, Cpu}, gpu::Gpu, pack::{self, V, MAXB}, rng::{pairwise, Pcg64}, v19::Host};
use std::io;
pub enum Back { Cpu(Cpu), Gpu(Gpu) }
pub struct RayField { pub path: String, pub syn: String, pub words_path: String, pub wcnt_path: String, pub arch: String, pub home: String, pub ready: bool, pub l: usize, pub t: usize, pub qt: bool, pub copy: usize, pub backend: String, pub ok: [bool; V], back: Option<Back>, h19: Option<Host>, cur: usize, words: Vec<Vec<u8>>, wtried: bool }
pub struct Stream { seq: Vec<u8>, lg: Vec<f32>, g: Pcg64, out: Vec<u8>, gp: usize, guide: Vec<u8>, pend: Vec<char>, ub: Vec<u8>, stop: Vec<Vec<char>>, hold: usize, left: usize, temp: f64, topk: usize, veto: bool, beta: f64, done: bool, dec: Dec, lk: Lock }
#[derive(Clone, Copy, Default)]
pub struct Dec { pub lock: bool, pub stop_sentence: bool, pub rep: f64 }
#[derive(Clone, Default)]
pub struct Lock { v: Vec<u8>, p: usize, k: usize, ban: Vec<Vec<u8>>, chain: bool, end: bool, fin: bool }
impl Lock { fn on(&self) -> bool { self.p < self.v.len() } fn push(&mut self, c: u8) { if self.on() && self.v[self.p] == c { self.p += 1; (!self.on()).then(|| { self.ban.push(self.v.to_ascii_lowercase()); (self.chain, self.end) = (self.k == 12, true); }); } } }
fn close(o: &[u8], p0: usize) -> (usize, bool) { let k = o.iter().rposition(|c| !b" ,;:-\n".contains(c)).map_or(p0, |i| (i + 1).max(p0)); (k, k > p0 && !b".!?".contains(&o[k - 1])) }
fn send(o: &[u8]) -> bool { o.len() >= 2 && matches!(o[o.len() - 1], 32 | 10) && b".!?".contains(&o[o.len() - 2]) && o[..o.len() - 2].iter().any(|&x| wordch(x as u32)) }
fn lse(r: &[f32]) -> f64 { let m = r.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x as f64)); pairwise(&r.iter().map(|&x| (x as f64 - m).exp()).collect::<Vec<_>>()).ln() + m }
fn argsort_desc(z: &[f64]) -> Vec<usize> { let mut o: Vec<usize> = (0..z.len()).collect(); o.sort_by(|&a, &b| z[b].partial_cmp(&z[a]).unwrap_or(std::cmp::Ordering::Equal).then(b.cmp(&a))); o }
fn rep4(seq: &[u8]) -> bool { seq.len() >= 4 && seq[seq.len() - 4..].iter().all(|&c| c == seq[seq.len() - 1]) }
fn find(h: &[u8], n: &[u8], from: usize) -> Option<usize> { (from..=h.len().saturating_sub(n.len())).find(|&i| i + n.len() <= h.len() && &h[i..i + n.len()] == n) }
impl RayField {
    pub fn new(path: Option<String>, syn: Option<String>, words: Option<String>, wcnt: Option<String>, arch: Option<String>, home: Option<String>) -> RayField {
        let home = home.or_else(|| std::env::var("AMNI_HOME").ok()).unwrap_or_else(|| ".".into());
        let j = |p: &str| format!("{home}/{p}");
        RayField { path: path.or_else(|| std::env::var("AMNI_RAY_PACK").ok()).unwrap_or_else(|| j("exports/gf17_continuum/rayhash_v16_big2_L21_e20_hop4_q5_s150k_pk_q5_i8.npz")), syn: syn.unwrap_or_else(|| j("data/big2/synsets.txt")), words_path: words.unwrap_or_else(|| j("data/big2/words.txt")), wcnt_path: wcnt.unwrap_or_else(|| j("data/big2/wordcounts.txt")), arch: arch.or_else(|| std::env::var("AMNI_RAY_ARCH").ok()).unwrap_or_else(|| "vulkan".into()), home, ready: false, l: 0, t: 0, qt: false, copy: 0, backend: String::new(), ok: std::array::from_fn(|c| (32..127).contains(&c) || c == 9 || c == 10), back: None, h19: None, cur: 0, words: vec![], wtried: false }
    }
    pub fn load(&mut self) -> io::Result<&mut Self> {
        if self.ready { return Ok(self); }
        let mut tb = pack::load(&self.path, &self.syn, &self.wcnt_path)?;
        (self.l, self.t, self.qt, self.copy) = (tb.l, tb.t, tb.geo.qt, tb.geo.copy);
        let g = if self.arch == "cpu" { Err("cpu requested".to_string()) } else { Gpu::new(&tb) };
        self.h19 = (self.copy > 0).then(|| Host::new(tb.geo.clone(), std::mem::replace(&mut tb.lx, pack::Lex { lex: vec![], trig: vec![] }), std::mem::take(&mut tb.cw)));
        (self.backend, self.back) = match g { Ok(g) => (format!("gpu:{}", g.name), Some(Back::Gpu(g))), Err(e) => (format!("cpu ({e})"), Some(Back::Cpu(Cpu::new(tb)))) };
        self.ready = true;
        Ok(self)
    }
    pub fn reset(&mut self, nb: usize) { match self.back.as_mut().unwrap() { Back::Cpu(c) => c.reset(self.cur, nb), Back::Gpu(g) => g.reset(self.cur, nb) } if let Some(h) = self.h19.as_mut() { h.reset(self.cur, nb) } }
    pub fn run(&mut self, nb: usize, par: &[u32], byt: &[u8], want: &[bool]) -> Vec<f32> {
        assert!(nb <= MAXB && par.len() == nb * want.len() && byt.len() == par.len());
        match self.back.as_mut().unwrap() { Back::Cpu(c) => c.run(&mut self.cur, nb, par, byt, want, self.h19.as_mut()), Back::Gpu(g) => g.run(&mut self.cur, nb, par, byt, want, self.h19.as_mut()) }
    }
    pub fn advance(&mut self, par: &[u32], byt: &[u8], want: bool) -> Option<Vec<f32>> { let r = self.run(byt.len(), par, byt, &[want]); want.then_some(r) }
    pub fn prime(&mut self, data: &[u8], nb: usize) -> Vec<f32> {
        let seq = if data.is_empty() { vec![10] } else { data.to_vec() };
        self.reset(nb);
        let want: Vec<bool> = (0..seq.len()).map(|i| i + 1 == seq.len()).collect();
        self.run(nb, &(0..seq.len()).flat_map(|_| 0..nb as u32).collect::<Vec<_>>(), &seq.iter().flat_map(|&c| std::iter::repeat_n(c, nb)).collect::<Vec<_>>(), &want)
    }
    pub fn logits_for(&mut self, data: &[u8]) -> Vec<f32> { self.reset(1); self.run(1, &vec![0; data.len()], data, &vec![true; data.len()]) }
    pub fn score(&mut self, prompts: &[Vec<u8>], conts: &[Vec<u8>]) -> Vec<f64> {
        let seqs: Vec<Vec<u8>> = prompts.iter().zip(conts).map(|(p, c)| [p.as_slice(), c].concat()).collect();
        let (nb, tt) = (seqs.len(), seqs.iter().map(Vec::len).max().unwrap_or(0));
        let mut lp = vec![0.0; nb];
        if nb == 0 || tt == 0 { return lp; }
        self.reset(nb);
        let byt: Vec<u8> = (0..tt).flat_map(|t| seqs.iter().map(move |s| *s.get(t).unwrap_or(&10))).collect();
        let lg = self.run(nb, &(0..tt).flat_map(|_| 0..nb as u32).collect::<Vec<_>>(), &byt, &(0..tt).map(|t| t + 1 < tt).collect::<Vec<_>>());
        for t in 1..tt { for b in 0..nb { if prompts[b].len() <= t && t < seqs[b].len() { let r = &lg[((t - 1) * nb + b) * V..((t - 1) * nb + b + 1) * V]; lp[b] += r[seqs[b][t] as usize] as f64 - lse(r); } } }
        lp
    }
    pub fn bpb(&mut self, hold: Option<&str>, batches: usize, ctx: usize, bs: usize, seed: u64) -> io::Result<f64> {
        let d = &std::fs::read(hold.map(String::from).unwrap_or_else(|| format!("{}/data/big2/hold.bin", self.home)))?;
        let (mut g, mut tot, mut n) = (Pcg64::new(Some(seed)), 0.0, 0usize);
        for _ in 0..batches {
            let ix = g.integers(0, (d.len() - ctx - 1) as i64, bs);
            self.reset(bs);
            let lg = self.run(bs, &(0..ctx).flat_map(|_| 0..bs as u32).collect::<Vec<_>>(), &(0..ctx).flat_map(|t| ix.iter().map(move |&i| d[i as usize + t]).collect::<Vec<_>>()).collect::<Vec<_>>(), &vec![true; ctx]);
            for t in 0..ctx { tot += pairwise(&(0..bs).map(|b| { let r = &lg[(t * bs + b) * V..(t * bs + b + 1) * V]; lse(r) - r[d[ix[b] as usize + t + 1] as usize] as f64 }).collect::<Vec<_>>()); n += bs; }
        }
        Ok(tot / n as f64 / std::f64::consts::LN_2)
    }
    fn load_words(&mut self) {
        if self.wtried { return; }
        self.wtried = true;
        let d = std::fs::read(&self.words_path).unwrap_or_default();
        self.words = d.split(|&c| c == b'\n').flat_map(|l| l.split(|&c| c == b'\r')).map(<[u8]>::to_vec).collect();
        self.words.sort();
        self.words.dedup();
        (self.words == [Vec::<u8>::new()]).then(|| self.words.clear());
    }
    pub fn word_ok(&self, seq: &[u8], c: u8) -> bool {
        if self.words.is_empty() { return true; }
        let al = |x: u8| x.is_ascii_alphabetic() || x == 39;
        let cur = &seq[seq.len() - seq.iter().rev().take_while(|&&x| al(x)).count()..];
        let has = |w: &[u8]| self.words.binary_search_by(|x| x.as_slice().cmp(w)).is_ok();
        if al(c) { let k = [cur, &[c]].concat(); let i = self.words.partition_point(|x| x.as_slice() < k.as_slice()); i < self.words.len() && self.words[i].starts_with(&k) || cur.len() >= 24 } else { cur.len() <= 1 || has(cur) || has(&cur.to_ascii_lowercase()) }
    }
    #[allow(clippy::too_many_arguments)]
    fn lockz(&self, b: usize, lk: &mut Lock, out: &[u8], z: &mut [f64], t: f64) -> Option<u8> {
        if lk.on() { return Some(lk.v[lk.p]); }
        if std::mem::take(&mut lk.end) { (0..V).filter(|&c| wordch(c as u32)).for_each(|c| z[c] = -1e9); return None; }
        let h = self.h19.as_ref()?;
        let (w, n) = (&h.w[self.cur * MAXB + b], out.iter().rev().take_while(|&&x| wordch(x as u32)).count());
        let (gv, am, sep) = (w.gv(&h.g, &h.cw), z.iter().enumerate().fold(0, |a, (c, &x)| if x > z[a] { c } else { a }), out.len() == n || !matches!(out[out.len() - n - 1], 39 | 45));
        let cs: Vec<(usize, f32, Vec<u8>, usize)> = (0..13).filter(|&k| gv[k].0 >= 0).map(|k| { let (v, p) = w.cv(k); (k, gv[k].1, v, p) }).filter(|x| x.3 < x.2.len() && x.2[x.3] as i32 == gv[x.0].0).collect();
        let ban = lk.ban.clone();
        let bn = |v: &[u8]| ban.contains(&v.to_ascii_lowercase());
        let best = cs.iter().filter(|x| x.1 > 0.0 && x.2[x.3] as usize == am && (n >= 1 && sep || lk.chain && x.0 == 12 && out.last() == Some(&32)) && !bn(&x.2)).fold(None, |a: Option<&(usize, f32, Vec<u8>, usize)>, x| if a.is_none_or(|a| x.1 > a.1) { Some(x) } else { a });
        lk.chain &= n == 0;
        if let Some(x) = best { (lk.v, lk.p, lk.k) = (x.2.clone(), x.3, x.0); return Some(lk.v[lk.p]); }
        lk.fin = n == 0 && cs.iter().any(|x| x.1 > 0.0 && x.3 == 0 && x.2[0] as usize == am && bn(&x.2)) && !cs.iter().any(|x| x.0 >= 8 && x.1 > 0.0 && !bn(&x.2));
        cs.iter().filter(|x| x.1 > 0.0 && bn(&x.2)).for_each(|x| z[x.2[x.3] as usize] -= x.1 as f64 / t);
        let cur = out[out.len() - n..].to_ascii_lowercase();
        lk.ban.iter().filter(|v| v.len() > n && v[..n] == cur[..] && n + 1 >= v.len().min(3)).for_each(|v| { z[v[n] as usize] = -1e9; z[v[n].to_ascii_uppercase() as usize] = -1e9; });
        None
    }
    #[allow(clippy::too_many_arguments)]
    pub fn stream(&mut self, prompt: &str, max_bytes: usize, temp: f64, topk: usize, stop: &[String], veto: bool, seed: Option<u64>, guide: Option<&[u8]>, beta: f64, dec: Dec) -> Stream {
        self.load().expect("pack load");
        veto.then(|| self.load_words());
        let seq = prompt.as_bytes().to_vec();
        let lg = self.prime(&seq, 1);
        Stream { seq, lg, g: Pcg64::new(seed), out: vec![], gp: 0, guide: guide.unwrap_or(&[]).to_vec(), pend: vec![], ub: vec![], stop: stop.iter().map(|s| s.chars().collect()).collect(), hold: stop.iter().map(|s| s.chars().count()).chain([1, 3 * dec.stop_sentence as usize]).max().unwrap() - 1, left: max_bytes, temp, topk, veto, beta, done: false, dec, lk: Lock::default() }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn beam(&mut self, prompt: &str, width: usize, expand: usize, maxlen: usize, rep: usize, veto: bool, dec: Dec) -> String {
        self.load().expect("pack load");
        veto.then(|| self.load_words());
        let p0 = prompt.as_bytes().to_vec();
        let lp: Vec<f64> = self.ok.iter().map(|&o| if o { 0.0 } else { 1e-300f64.ln() }).collect();
        let (md, norm) = (dec.lock || dec.stop_sentence, |t: &(f64, Vec<u8>, usize, Lock)| t.0 / ((t.1.len() - p0.len() + 1) as f64).powf(0.7));
        let (mut beams, mut lgs) = (vec![(0.0, p0.clone(), 0usize, Lock::default())], self.prime(&p0, 1));
        for _ in 0..maxlen {
            let mut cand = vec![];
            for (b, (sc, seq, _, lk0)) in beams.iter().enumerate() {
                if md && seq.len() > p0.len() && seq[seq.len() - 1] == 10 { cand.push((*sc, seq.clone(), b, lk0.clone())); continue; }
                let mut z: Vec<f64> = (0..V).map(|c| lgs[b * V + c] as f64 + lp[c]).collect();
                rep4(seq).then(|| z[seq[seq.len() - 1] as usize] -= 6.0);
                let mut lk = lk0.clone();
                let fc = if dec.lock { self.lockz(b, &mut lk, &seq[p0.len()..], &mut z, 1.0) } else { None };
                if dec.stop_sentence && lk.fin { let (k, d) = close(seq, p0.len()); cand.push((*sc, [&seq[..k], if d { &b"."[..] } else { &b""[..] }, &[10]].concat(), b, lk)); continue; }
                let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
                let s = pairwise(&e);
                let pr: Vec<f64> = e.iter().map(|x| x / s).collect();
                let order = argsort_desc(&pr);
                let top: Vec<usize> = order[..(expand * 6).min(V)].iter().copied().filter(|&c| !veto || self.word_ok(seq, c as u8)).take(expand).collect();
                for c in fc.map_or_else(|| if top.is_empty() { order[..expand.min(V)].to_vec() } else { top }, |f| vec![f as usize]) {
                    let (mut ns, mut l2) = ([seq.as_slice(), &[c as u8]].concat(), lk.clone());
                    let pen = if rep > 0 && ns.len() > 2 * rep && find(&ns[..ns.len() - rep], &ns[ns.len() - rep..], 0).is_some() { 0.6 } else { 0.0 };
                    l2.push(c as u8);
                    (dec.stop_sentence && !l2.on() && send(&ns[p0.len()..])).then(|| *ns.last_mut().unwrap() = 10);
                    cand.push((sc + (pr[c] + 1e-12).ln() - pen, ns, b, l2));
                }
            }
            cand.sort_by(|a, b| norm(b).partial_cmp(&norm(a)).unwrap_or(std::cmp::Ordering::Equal));
            cand.truncate(width);
            beams = cand;
            lgs = self.advance(&beams.iter().map(|x| x.2 as u32).collect::<Vec<_>>(), &beams.iter().map(|x| x.1[x.1.len() - 1]).collect::<Vec<_>>(), true).unwrap();
            if beams.iter().all(|x| x.1[x.1.len() - 1] == 10) { break; }
        }
        let best = beams.iter().fold(None, |a: Option<&(f64, Vec<u8>, usize, Lock)>, x| if a.is_none_or(|a| norm(x) > norm(a)) { Some(x) } else { a }).unwrap();
        let o = &best.1[p0.len()..];
        String::from_utf8_lossy(if md { &o[..o.iter().position(|&c| c == 10).unwrap_or(o.len())] } else { o }).into_owned()
    }
}
impl Stream {
    fn decode(&mut self, c: u8) {
        self.ub.push(c);
        loop {
            match std::str::from_utf8(&self.ub) {
                Ok(s) => { self.pend.extend(s.chars()); self.ub.clear(); return; }
                Err(e) => {
                    let v = e.valid_up_to();
                    self.pend.extend(std::str::from_utf8(&self.ub[..v]).unwrap().chars());
                    match e.error_len() { Some(n) => { self.pend.push('\u{FFFD}'); self.ub.drain(..v + n); } None => { self.ub.drain(..v); return; } }
                }
            }
        }
    }
    pub fn next(&mut self, f: &mut RayField) -> Option<String> {
        while !self.done {
            if self.left == 0 { self.done = true; return (!self.pend.is_empty()).then(|| self.pend.drain(..).collect()); }
            self.left -= 1;
            let mut z: Vec<f64> = (0..V).map(|c| if f.ok[c] { self.lg[c] as f64 / self.temp } else { -1e9 }).collect();
            rep4(&self.seq).then(|| z[self.seq[self.seq.len() - 1] as usize] -= 6.0);
            let gb = self.guide.get(self.gp).map(|&x| x as usize);
            if let Some(g) = gb { z[g] += self.beta; }
            let n = self.out.len();
            (self.dec.rep > 0.0 && n >= 5).then(|| (0..V).filter(|&c| find(&self.out, &[&self.out[n - 5..], &[c as u8][..]].concat(), 0).is_some()).for_each(|c| z[c] -= self.dec.rep));
            let fc = if self.dec.lock { f.lockz(0, &mut self.lk, &self.out, &mut z, self.temp) } else { None };
            if self.dec.stop_sentence && self.lk.fin {
                self.done = true;
                let (k, d) = close(&self.out, 0);
                let mut r: String = self.pend[..self.pend.len().saturating_sub(self.out.len() - k)].iter().collect();
                d.then(|| r.push('.'));
                return (!r.is_empty()).then_some(r);
            }
            let c = match fc { Some(x) => x as usize, None => {
                let order = argsort_desc(&z);
                let keep: Vec<usize> = order[..(self.topk * 6).min(V)].iter().copied().filter(|&c| !self.veto || Some(c) == gb || f.word_ok(&self.seq, c as u8)).take(self.topk).collect();
                let keep = if keep.is_empty() { order[..self.topk.min(V)].to_vec() } else { keep };
                let m = keep.iter().map(|&c| z[c]).fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = keep.iter().map(|&c| (z[c] - m).exp()).collect();
                let s = pairwise(&e);
                keep[self.g.choice(&e.iter().map(|x| x / s).collect::<Vec<_>>())]
            } };
            self.seq.push(c as u8);
            self.out.push(c as u8);
            self.lk.push(c as u8);
            self.decode(c as u8);
            let tail = &self.out[self.out.len().saturating_sub(8)..];
            self.gp = if Some(c) == gb { self.gp + 1 } else if self.guide.is_empty() { self.gp } else { find(&self.guide, tail, self.gp.saturating_sub(60)).map_or(self.gp, |j| j + tail.len()) };
            let hit = self.stop.iter().filter_map(|t| (0..=self.pend.len().saturating_sub(t.len())).find(|&i| i + t.len() <= self.pend.len() && self.pend[i..i + t.len()] == t[..])).chain((self.dec.stop_sentence && !self.lk.on() && send(&self.out)).then(|| self.pend.len().saturating_sub(1))).min();
            if hit.is_some() || (!self.guide.is_empty() && self.gp >= self.guide.len()) {
                self.done = true;
                let r: String = self.pend[..hit.unwrap_or(self.pend.len())].iter().collect();
                return (!r.is_empty()).then_some(r);
            }
            let k = self.pend.len().saturating_sub(self.hold);
            let r: String = self.pend.drain(..k).collect();
            self.lg = f.advance(&[0], &[c as u8], true).unwrap();
            if !r.is_empty() { return Some(r); }
        }
        None
    }
}
