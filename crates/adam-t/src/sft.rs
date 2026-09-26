use crate::{f1, fa::Fa, it2, logits, nap, sm, thr, Cfg, Core, Mem};
use burn::tensor::{activation::log_softmax, backend::Backend, Tensor, TensorData};
use serde_json::Value;
use std::{fs, io, time::Instant};
pub const PAD: u32 = 10;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ex { pub off: usize, pub n: usize, pub src: u8, pub loss: usize }
pub struct Set { pub tok: Vec<u16>, pub mask: Vec<u8>, pub ex: Vec<Ex> }
pub struct Job { pub tr: Set, pub ho: Set, pub plan: Vec<Vec<usize>>, pub hw: Vec<Vec<usize>>, pub names: Vec<String>, pub w: Vec<f64>, pub steps: usize, pub epw: usize, pub skip: usize }
pub fn npy(b: &[u8]) -> io::Result<Vec<Ex>> {
    let bad = |m: &str| io::Error::other(format!("idx.npy: {m}"));
    (b.len() > 12 && &b[..6] == b"\x93NUMPY").then_some(()).ok_or_else(|| bad("bad magic"))?;
    let (hl, h0) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let hd = std::str::from_utf8(b.get(h0..h0 + hl).ok_or_else(|| bad("short header"))?).map_err(|_| bad("header not utf8"))?;
    let f: Vec<(String, usize)> = regex::Regex::new(r"\('(\w+)',\s*'[<|=]?u(\d)'\)").unwrap().captures_iter(hd).map(|c| (c[1].to_string(), c[2].parse().unwrap())).collect();
    let at = |k: &str| f.iter().position(|x| x.0 == k).map(|i| (f[..i].iter().map(|x| x.1).sum::<usize>(), f[i].1)).ok_or_else(|| bad(&format!("field {k} missing or not little-endian unsigned in {hd}")));
    let ([o, n, s, l], rs) = ([at("off")?, at("n")?, at("src")?, at("loss")?], f.iter().map(|x| x.1).sum::<usize>());
    let rd = |r: &[u8], (p, w): (usize, usize)| r[p..p + w].iter().rev().fold(0u64, |a, &x| a << 8 | x as u64) as usize;
    ((b.len() - h0 - hl) % rs == 0).then_some(()).ok_or_else(|| bad("data not a whole number of records"))?;
    Ok(b[h0 + hl..].chunks_exact(rs).map(|r| Ex { off: rd(r, o), n: rd(r, n), src: rd(r, s) as u8, loss: rd(r, l) }).collect())
}
pub fn keys(j: &str, k: &str) -> Vec<String> {
    let b = j.as_bytes();
    let mut i = regex::Regex::new(&format!(r#""{}"\s*:\s*\{{"#, regex::escape(k))).unwrap().find(j).map_or(b.len(), |m| m.end());
    let (mut d, mut o, mut st) = (1usize, Vec::new(), None::<usize>);
    while i < b.len() && d > 0 {
        match (st, b[i]) {
            (Some(_), b'\\') => i += 1,
            (Some(s), b'"') => { if d == 1 && j[i + 1..].trim_start().starts_with(':') { o.push(j[s..i].to_string()) } st = None }
            (Some(_), _) => {}
            (None, b'"') => st = Some(i + 1),
            (None, b'{' | b'[') => d += 1,
            (None, b'}' | b']') => d -= 1,
            _ => {}
        }
        i += 1;
    }
    o
}
pub fn order(ex: &[Ex], w: &[f64], seed: u64, ep: usize) -> Vec<usize> {
    let h = |tag: u64, i: u64| sm(sm(seed ^ (ep as u64) << 32 ^ tag << 60) ^ i);
    let mut o: Vec<usize> = ex.iter().enumerate().flat_map(|(i, e)| { let x = w.get(e.src as usize).copied().unwrap_or(1.0).max(0.0); std::iter::repeat_n(i, x.floor() as usize + (((h(1, i as u64) >> 11) as f64 / (1u64 << 53) as f64) < x.fract()) as usize) }).collect();
    (1..o.len()).rev().for_each(|i| o.swap(i, (h(2, i as u64) % (i as u64 + 1)) as usize));
    o
}
pub fn pack(ex: &[Ex], ord: &[usize], cap: usize) -> Vec<Vec<usize>> {
    let (mut o, mut w, mut u) = (Vec::new(), Vec::new(), 0usize);
    for &i in ord.iter().filter(|&&i| ex[i].n <= cap) { if u + ex[i].n > cap { o.push(std::mem::take(&mut w)); u = 0; } w.push(i); u += ex[i].n; }
    if !w.is_empty() { o.push(w); }
    o
}
pub fn tnll<B: Backend>(lg: Tensor<B, 2>, y: &[u32], dev: &B::Device) -> Tensor<B, 1> { let n = y.len(); log_softmax(lg, 1).gather(1, it2(y.iter().map(|&v| v as i32).collect(), [n, 1], dev)).reshape([n]).neg() }
pub fn mce<B: Backend>(lg: Tensor<B, 2>, y: &[u32], m: &[f32], den: f64, dev: &B::Device) -> Tensor<B, 1> { (tnll(lg, y, dev) * Tensor::<B, 1>::from_data(TensorData::new(m.to_vec(), [m.len()]), dev)).sum().div_scalar(den) }
impl Set {
    pub fn open(dir: &str, split: &str, v: usize) -> io::Result<Set> {
        let p = format!("{dir}/{split}.bpe{v}");
        Set::new(fs::read(format!("{p}.u16"))?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(), fs::read(format!("{p}.mask.u8"))?, npy(&fs::read(format!("{p}.idx.npy"))?)?)
    }
    pub fn new(tok: Vec<u16>, mask: Vec<u8>, ex: Vec<Ex>) -> io::Result<Set> {
        let ok = tok.len() == mask.len() && mask.iter().all(|&m| m <= 1);
        let bad = ok.then(|| ex.iter().position(|e| e.n == 0 || e.off + e.n > tok.len() || mask[e.off] != 0 || mask[e.off..e.off + e.n].iter().map(|&m| m as usize).sum::<usize>() != e.loss)).flatten();
        (ok && bad.is_none()).then_some(()).ok_or_else(|| io::Error::other(format!("sft set inconsistent: {} tokens, {} mask bytes, mask 0/1 {}, first bad example {:?}", tok.len(), mask.len(), mask.iter().all(|&m| m <= 1), bad.map(|i| (i, ex[i])))))?;
        Ok(Set { tok, mask, ex })
    }
    pub fn fill(&self, ws: &[Vec<usize>], t: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<u8>) {
        let (mut x, mut y, mut m, mut s) = (Vec::with_capacity(ws.len() * t), Vec::with_capacity(ws.len() * t), Vec::with_capacity(ws.len() * t), Vec::with_capacity(ws.len() * t));
        for w in ws {
            let b: Vec<(u32, u8, u8)> = w.iter().flat_map(|&i| { let e = self.ex[i]; (e.off..e.off + e.n).map(move |p| (self.tok[p] as u32, self.mask[p], e.src)) }).chain(std::iter::repeat((PAD, 0, 255))).take(t + 1).collect();
            x.extend(b[..t].iter().map(|v| v.0));
            y.extend(b[1..].iter().map(|v| v.0));
            m.extend(b[1..].iter().map(|v| v.1 as f32));
            s.extend(b[1..].iter().map(|v| v.2));
        }
        (x, y, m, s)
    }
}
impl Job {
    pub fn new(c: &Cfg) -> io::Result<Job> {
        let d = format!("{}/data/{}", c.home, c.corpus);
        let (tr, ho, cap) = (Set::open(&d, "train", c.v)?, Set::open(&d, "hold", c.v)?, c.t + 1);
        let bp = c.bpe()?;
        let head = |s: &Set| s.ex.iter().position(|e| { let h: Vec<u32> = s.tok[e.off..e.off + e.n.min(8)].iter().map(|&v| v as u32).collect(); !bp.as_ref().map_or_else(|| h.iter().map(|&v| v as u8).collect(), |b| b.dec(&h)).starts_with(b"User:") });
        [("train", &tr), ("hold", &ho)].iter().try_for_each(|(k, s)| head(s).map_or(Ok(()), |i| Err(io::Error::other(format!("sft {k} example {i} does not start with 'User:'")))))?;
        let (mj, ns) = (fs::read_to_string(format!("{d}/meta.json")).unwrap_or_default(), tr.ex.iter().chain(&ho.ex).map(|e| e.src as usize + 1).max().unwrap_or(0));
        let names: Vec<String> = Some(keys(&mj, "sources")).filter(|k| k.len() >= ns).unwrap_or_else(|| (0..ns).map(|i| i.to_string()).collect());
        let mv: Value = serde_json::from_str(&mj).unwrap_or(Value::Null);
        let cnt = |s: &Set, i: usize| s.ex.iter().filter(|e| e.src as usize == i).count() as u64;
        names.iter().enumerate().try_for_each(|(i, n)| mv["sources"][n]["train_examples"].as_u64().map_or(Ok(()), |v| (v == cnt(&tr, i)).then_some(()).ok_or_else(|| io::Error::other(format!("meta.json source {n} has {v} train examples, idx src {i} has {}", cnt(&tr, i))))))?;
        let kv: Vec<(String, f64)> = c.sw.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.split_once('=').and_then(|(k, v)| v.trim().parse::<f64>().ok().filter(|w| *w >= 0.0).map(|w| (k.trim().to_string(), w))).ok_or_else(|| io::Error::other(format!("bad SRC_W entry {s} (want name=weight)")))).collect::<io::Result<_>>()?;
        kv.iter().find(|(k, _)| !names.iter().enumerate().any(|(i, n)| n == k || i.to_string() == *k)).map_or(Ok(()), |(k, _)| Err(io::Error::other(format!("SRC_W source {k} not in {names:?}"))))?;
        let w: Vec<f64> = names.iter().enumerate().map(|(i, n)| kv.iter().rev().find(|(k, _)| k == n || *k == i.to_string()).map_or(1.0, |x| x.1)).collect();
        let mut plan = pack(&tr.ex, &order(&tr.ex, &w, c.seed, 0), cap);
        let (epw, skip) = (plan.len(), tr.ex.iter().filter(|e| e.n > cap).count());
        (epw > 0).then_some(()).ok_or_else(|| io::Error::other("no sft windows: every source weight is 0 or every example is longer than CTX+1"))?;
        let steps = if c.ep > 0.0 { (c.ep * epw as f64 / c.bs as f64).ceil() as usize } else { c.steps };
        let mut ep = 0;
        while plan.len() < steps * c.bs { ep += 1; plan.extend(pack(&tr.ex, &order(&tr.ex, &w, c.seed, ep), cap)); }
        let e = 1usize << (c.bs / c.micro.max(1)).clamp(1, 32).ilog2();
        let hw = pack(&ho.ex, &(0..ho.ex.len()).collect::<Vec<_>>(), cap);
        let hw = if c.evb > 0 { hw[..hw.len().min(c.evb * e)].to_vec() } else { hw };
        Ok(Job { tr, ho, plan, hw, names, w, steps, epw, skip })
    }
    pub fn batch(&self, it: usize, c: &Cfg) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<u8>) { self.tr.fill(&self.plan[it * c.bs..(it + 1) * c.bs], c.t) }
    pub fn info(&self) -> String {
        let tk = |s: &Set, ws: &[Vec<usize>]| ws.iter().flatten().map(|&i| (s.ex[i].n, s.ex[i].loss)).fold((0usize, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));
        let ((et, el), (ht, hl)) = (tk(&self.tr, &self.plan[..self.epw]), tk(&self.ho, &self.hw));
        format!("{{\"sft\": {{\"train_examples\": {}, \"hold_examples\": {}, \"skipped_too_long\": {}, \"windows_per_epoch\": {}, \"epoch0_tokens\": {et}, \"epoch0_loss_tokens\": {el}, \"steps\": {}, \"epochs\": {:.3}, \"hold_windows\": {}, \"hold_tokens\": {ht}, \"hold_loss_tokens\": {hl}, \"src_w\": {{{}}}}}}}", self.tr.ex.len(), self.ho.ex.len(), self.skip, self.epw, self.steps, self.plan.len() as f64 / self.epw as f64, self.hw.len(), self.names.iter().zip(&self.w).map(|(n, w)| format!("\"{n}\": {w}")).collect::<Vec<_>>().join(", "))
    }
    pub fn eval<B: Fa>(&self, c: &Cfg, core: &Core<B>, mem: Option<&Mem<B>>, lens: &[u32], dev: &B::Device) -> (f64, String) {
        let (e, th) = (1usize << (c.bs / c.micro.max(1)).clamp(1, 32).ilog2(), thr(&c.thf));
        let mut a = vec![(0f64, 0f64, 0f64); self.names.len().max(1)];
        for ws in self.hw.chunks(e) {
            let (t0, (x, y, m, s)) = (Instant::now(), self.ho.fill(ws, c.t));
            let nl = f1(tnll(logits(c, core, mem, &x, ws.len(), dev), &y, dev));
            nap(th, t0.elapsed().as_secs_f64() * 1e3, c.unit);
            nl.iter().zip(&y).zip(m.iter().zip(&s)).filter(|(_, (m, _))| **m > 0.0).for_each(|((l, &yt), (_, &sr))| { let r = &mut a[sr as usize]; *r = (r.0 + *l as f64, r.1 + 1.0, r.2 + lens[yt as usize] as f64); });
        }
        let (l, n, b) = a.iter().fold((0.0, 0.0, 0.0), |z, r| (z.0 + r.0, z.1 + r.1, z.2 + r.2));
        (l / b.max(1.0), format!(", \"val_nats_per_tok\": {:.4}, \"val_loss_tokens\": {n}, \"val_by_src\": {{{}}}", l / n.max(1.0), self.names.iter().zip(&a).filter(|(_, r)| r.1 > 0.0).map(|(k, r)| format!("\"{k}\": {:.4}", r.0 / r.1)).collect::<Vec<_>>().join(", ")))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ce, load, lut, train};
    use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
    use std::path::PathBuf;
    type Ad = Autodiff<NdArray>;
    const DEV: NdArrayDevice = NdArrayDevice::Cpu;
    fn tiny(t: usize) -> Cfg { Cfg { d: 64, l: 1, h: 1, t, ff: 128, ns: vec![2, 3], rb: 6, tern: true, mem: true, v: 256, bpe: String::new(), bs: 4, micro: 2, base: String::new(), sw: String::new(), ep: 0.0, seed: 3, par: 1, wall: 0.0, unit: 50.0, ..Cfg::env() } }
    fn syn(ns: &[usize]) -> Set {
        let (mut tok, mut mask, mut ex) = (Vec::new(), Vec::new(), Vec::new());
        ns.iter().enumerate().for_each(|(i, &n)| { let off = tok.len(); (0..n).for_each(|j| { tok.push((off + j) as u16); mask.push((j > 0 && sm((off + j) as u64) & 3 != 0) as u8); }); ex.push(Ex { off, n, src: (i % 4) as u8, loss: mask[off..].iter().map(|&m| m as usize).sum() }); });
        Set::new(tok, mask, ex).unwrap()
    }
    fn wnpy(ex: &[Ex]) -> Vec<u8> {
        let h = format!("{{'descr': [('off', '<u8'), ('n', '<u4'), ('src', '|u1'), ('loss', '<u4')], 'fortran_order': False, 'shape': ({},), }}", ex.len());
        let h = format!("{h}{}\n", " ".repeat(63 - (10 + h.len()) % 64));
        [b"\x93NUMPY\x01\x00".to_vec(), (h.len() as u16).to_le_bytes().to_vec(), h.into_bytes(), ex.iter().flat_map(|e| [(e.off as u64).to_le_bytes().to_vec(), (e.n as u32).to_le_bytes().to_vec(), vec![e.src], (e.loss as u32).to_le_bytes().to_vec()].concat()).collect()].concat()
    }
    #[test]
    fn pack_keeps_examples_whole_and_mask_aligned() {
        let ns: Vec<usize> = (0..600).map(|i| if i == 17 { 90 } else { 1 + (sm(i as u64) % 40) as usize }).collect();
        let (s, t) = (syn(&ns), 63);
        let ord = order(&s.ex, &[1.0, 2.0, 0.5, 1.0], 7, 0);
        let ws = pack(&s.ex, &ord, t + 1);
        assert_eq!(ws.concat(), ord.iter().copied().filter(|&i| s.ex[i].n <= t + 1).collect::<Vec<_>>());
        assert!(!ws.concat().contains(&17) && ws.iter().all(|w| !w.is_empty() && w.iter().map(|&i| s.ex[i].n).sum::<usize>() <= t + 1));
        let (x, y, m, sr) = s.fill(&ws, t);
        assert!(x.len() == ws.len() * t && y.len() == x.len() && m.len() == x.len() && sr.len() == x.len());
        ws.iter().enumerate().for_each(|(k, w)| {
            let b: Vec<u32> = w.iter().flat_map(|&i| (s.ex[i].off..s.ex[i].off + s.ex[i].n).map(|p| p as u32)).collect();
            let (xw, yw, mw, sw) = (&x[k * t..(k + 1) * t], &y[k * t..(k + 1) * t], &m[k * t..(k + 1) * t], &sr[k * t..(k + 1) * t]);
            assert_eq!(xw[..b.len().min(t)], b[..b.len().min(t)]);
            assert_eq!(yw[..b.len() - 1], b[1..]);
            assert!(yw[b.len() - 1..].iter().all(|&v| v == PAD) && mw[b.len() - 1..].iter().all(|&v| v == 0.0) && sw[b.len() - 1..].iter().all(|&v| v == 255));
            (0..b.len() - 1).for_each(|j| { let p = yw[j] as usize; assert_eq!(mw[j], s.mask[p] as f32); assert_eq!(sw[j], s.ex.iter().find(|e| e.off <= p && p < e.off + e.n).unwrap().src); assert!(xw[j] + 1 == yw[j] || mw[j] == 0.0); });
            assert_eq!(mw.iter().sum::<f32>() as usize, w.iter().map(|&i| s.ex[i].loss).sum::<usize>());
        });
    }
    #[test]
    fn order_follows_source_weights() {
        let s = syn(&vec![5; 4000]);
        let o = order(&s.ex, &[1.0, 2.0, 0.5, 0.0], 11, 0);
        let c: Vec<usize> = (0..4).map(|k| o.iter().filter(|&&i| s.ex[i].src == k).count()).collect();
        assert!(c[0] == 1000 && c[1] == 2000 && c[3] == 0 && (430..570).contains(&c[2]), "{c:?}");
        assert!(o == order(&s.ex, &[1.0, 2.0, 0.5, 0.0], 11, 0) && o != order(&s.ex, &[1.0, 2.0, 0.5, 0.0], 11, 1) && o[..50] != (0..50).collect::<Vec<_>>()[..]);
    }
    #[test]
    fn npy_and_meta_keys_parse() {
        let ex = vec![Ex { off: 0, n: 304, src: 4, loss: 165 }, Ex { off: 304, n: 121, src: 16, loss: 59 }, Ex { off: 1 << 33, n: 7, src: 0, loss: 0 }];
        let b = wnpy(&ex);
        assert_eq!((b.len() - 3 * 17) % 64, 0);
        assert_eq!(npy(&b).unwrap(), ex);
        assert!(npy(&b[..40]).is_err());
        assert_eq!(keys(r#"{"a": {"sources": 1}, "sources": {"x": {"k": [1, {"z": 2}]}, "b": "s: \"q\": ", "c": 3}, "d": {"e": 1}}"#, "sources"), vec!["x", "b", "c"]);
        let d = format!("{}/data/adam_t/sft", Cfg::env().home);
        if let Ok(h) = Set::open(&d, "hold", 4096) {
            let (mj, n) = (fs::read_to_string(format!("{d}/meta.json")).unwrap(), h.ex.len());
            let (k, mv) = (keys(&mj, "sources"), serde_json::from_str::<Value>(&mj).unwrap());
            assert!(n > 0 && k.len() == mv["sources"].as_object().unwrap().len() && h.ex.iter().all(|e| e.n <= 512));
            k.iter().enumerate().for_each(|(i, s)| assert_eq!(mv["sources"][s]["hold_examples"].as_u64().unwrap() as usize, h.ex.iter().filter(|e| e.src as usize == i).count(), "{s}"));
            let hm = Cfg::env().home;
            let j = Job::new(&Cfg { corpus: "adam_t/sft".into(), v: 4096, bpe: format!("{hm}/data/clean2/bpe4096.json"), bs: 16, micro: 2, ep: 2.0, evb: 0, sw: "hotpot_rc=0.5".into(), home: hm, ..tiny(1024) }).unwrap();
            println!("{}", j.info());
            assert!(j.skip == 0 && !j.hw.is_empty() && j.w[j.names.iter().position(|n| n == "hotpot_rc").unwrap()] == 0.5 && j.plan.len() >= j.steps * 16 && j.names == k);
        }
    }
    #[test]
    fn masked_ce_matches_reference() {
        let (c, n) = (tiny(64), 4 * 64);
        <NdArray as Backend>::seed(&DEV, 5);
        let (core, mem) = (Core::<NdArray>::new(&c, &DEV), Mem::<NdArray>::new(&c, &DEV));
        let (x, y): (Vec<u32>, Vec<u32>) = (0..n).map(|i| ((sm(i as u64) % 256) as u32, (sm(i as u64 + 9999) % 256) as u32)).unzip();
        let m: Vec<f32> = (0..n).map(|i| (sm(i as u64 ^ 77) % 3 == 0) as u8 as f32).collect();
        let lg = logits(&c, &core, Some(&mem), &x, 4, &DEV);
        let (lv, den) = (f1(lg.clone()), m.iter().sum::<f32>() as f64);
        let want = lut::nll(&lv, &y).iter().zip(&m).map(|(l, &w)| l * w as f64).sum::<f64>() / den;
        let got = f1(mce(lg.clone(), &y, &m, den, &DEV))[0] as f64;
        assert!((got - want).abs() < 1e-4 * want.abs().max(1.0), "{got} vs {want}");
        let (ones, full) = (vec![1f32; n], f1(ce(lg.clone(), &y, &DEV))[0] as f64);
        assert!((f1(mce(lg.clone(), &y, &ones, n as f64, &DEV))[0] as f64 - full).abs() < 1e-4 * full, "all-ones mask must equal plain CE");
        let y2: Vec<u32> = y.iter().zip(&m).map(|(&v, &w)| if w > 0.0 { v } else { (v + 1) % 256 }).collect();
        assert!((f1(mce(lg.clone(), &y2, &m, den, &DEV))[0] as f64 - got).abs() < 1e-6, "targets outside the mask must not change the loss");
        let ca = Core::<Ad>::new(&c, &DEV);
        let g = |yy: &[u32], mm: &[f32]| { let gr = mce(logits(&c, &ca, None, &x, 4, &DEV), yy, mm, den, &DEV).backward(); (f1(ca.head.w.val().grad(&gr).unwrap()), f1(ca.emb.val().grad(&gr).unwrap())) };
        let ((h1, e1), (h2, e2), (h0, e0)) = (g(&y, &m), g(&y2, &m), g(&y, &vec![0.0; n]));
        assert!(h1 == h2 && e1 == e2 && h0.iter().chain(&e0).all(|&v| v == 0.0) && h1.iter().any(|&v| v != 0.0));
    }
    #[test]
    fn sft_mode_end_to_end_cpu() {
        let r = std::env::temp_dir().join(format!("adamt-sft-test-{}", std::process::id()));
        let (s, _) = (r.to_string_lossy().to_string(), fs::remove_dir_all(&r));
        let w = |p: &str, b: &[u8]| { let p = PathBuf::from(format!("{s}/{p}")); fs::create_dir_all(p.parent().unwrap()).unwrap(); fs::write(p, b).unwrap(); };
        w("thr.txt", b"0\n");
        let txt: Vec<u8> = (0..40000u64).map(|i| b"the cat sat on a mat. "[(i % 22) as usize] ^ (sm(i) % 50 == 0) as u8).collect();
        w("data/base/train.bin", &txt);
        w("data/base/hold.bin", &txt[..6000]);
        let base = Cfg { home: s.clone(), corpus: "base".into(), out: format!("{s}/base_run"), thf: format!("{s}/thr.txt"), steps: 3, warm: 1, lr: 3e-3, lrm: 3e-3, ev: 3, ck: 3, evb: 1, ..tiny(128) };
        train::<Ad>(&base, DEV, 0).unwrap();
        let qa = [("color of the sky?", "blue"), ("two plus two?", "four"), ("capital of france?", "paris"), ("opposite of hot?", "cold")];
        let mk = |k: usize| { let (mut tok, mut mask, mut ex) = (Vec::new(), Vec::new(), Vec::new()); (0..k).for_each(|i| { let (q, a) = qa[i % 4]; let (u, v) = (format!("User: {q}\nAdam:"), format!(" {a}\n\n")); let off = tok.len(); tok.extend(u.bytes().chain(v.bytes()).map(|b| b as u16)); mask.extend(std::iter::repeat_n(0u8, u.len()).chain(std::iter::repeat_n(1u8, v.len()))); ex.push(Ex { off, n: u.len() + v.len(), src: (i % 2) as u8, loss: v.len() }); }); (tok, mask, ex) };
        [("train", 96), ("hold", 16)].iter().for_each(|&(sp, k)| { let (tok, mask, ex) = mk(k); w(&format!("data/sft/{sp}.bpe256.u16"), &tok.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()); w(&format!("data/sft/{sp}.bpe256.mask.u8"), &mask); w(&format!("data/sft/{sp}.bpe256.idx.npy"), &wnpy(&ex)); w(&format!("data/sft/{sp}.bin"), &tok.iter().map(|&v| v as u8).collect::<Vec<u8>>()); });
        w("data/sft/meta.json", br#"{"name": "t", "sources": {"chat": {"train_examples": 48}, "qa": {"train_examples": 48}}}"#);
        let sc = |out: &str, lr: f64, sw: &str| Cfg { home: s.clone(), corpus: "sft".into(), out: format!("{s}/{out}"), thf: format!("{s}/thr.txt"), base: format!("{s}/base_run"), sw: sw.into(), ep: 3.0, steps: 1, warm: 2, lr, lrm: lr, ev: 1, ck: 4, evb: 0, ..tiny(128) };
        assert!(Job::new(&sc("x", 0.0, "nope=2")).is_err());
        let c0 = sc("sft0", 0.0, "qa=0");
        let j = Job::new(&c0).unwrap();
        assert!(j.plan.concat().iter().all(|&i| j.tr.ex[i].src == 0) && j.epw == 16 && j.steps == 12, "{}", j.info());
        train::<Ad>(&c0, DEV, 0).unwrap();
        let (bc, oc) = (Cfg::from_dir(&format!("{s}/base_run")).unwrap(), Cfg::from_dir(&format!("{s}/sft0")).unwrap());
        let ((b0, bm), (o0, om)) = (load::<NdArray>(&bc, &DEV).unwrap(), load::<NdArray>(&oc, &DEV).unwrap());
        assert!(oc.sft() && oc.d == 64 && oc.t == 128 && oc.corpus == "sft");
        assert!(f1(b0.emb.val()) == f1(o0.emb.val()) && f1(b0.head.w.val()) == f1(o0.head.w.val()) && f1(bm.unwrap().tab.val()) == f1(om.unwrap().tab.val()), "sft must start from the base weights (lr 0 keeps them)");
        train::<Ad>(&sc("sft1", 2e-2, ""), DEV, 0).unwrap();
        let lg = fs::read_to_string(format!("{s}/sft1/log.jsonl")).unwrap();
        let v: Vec<f64> = lg.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()?["val_nats_per_tok"].as_f64()).collect();
        let st: usize = fs::read_to_string(format!("{s}/sft1/ckpt/state.txt")).unwrap().trim().parse().unwrap();
        assert!(lg.contains("\"sft\": {") && st >= 20 && fs::read_to_string(format!("{s}/sft1/ckpt/steps.txt")).unwrap().trim() == st.to_string() && v.len() == st, "{lg}");
        assert!(v[v.len() - 1] < 0.5 * v[0], "masked hold loss must fall: {v:?}");
        assert!(PathBuf::from(format!("{s}/sft1/model.gf17")).exists());
        assert!(load::<NdArray>(&Cfg::from_dir(&format!("{s}/sft1")).unwrap(), &DEV).is_ok());
        let _ = fs::remove_dir_all(&r);
    }
}
