use adam_ray::{npy::Npz, rng::pairwise, RayField};
use fancy_regex::Regex as FRe;
use regex::{bytes::Regex as BRe, Regex};
use std::{collections::{HashMap, HashSet}, io, str::FromStr, sync::LazyLock};
pub static STOP: LazyLock<HashSet<&str>> = LazyLock::new(|| adam_ray::v19::STOPW.split(' ').collect());
static TOK: LazyLock<BRe> = LazyLock::new(|| BRe::new(r"[a-z][a-z']{1,}").unwrap());
static PROPER: LazyLock<FRe> = LazyLock::new(|| FRe::new(r"(?<=[\s\S])(?<![.?!] )\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)").unwrap());
static WHO: LazyLock<FRe> = LazyLock::new(|| FRe::new(r"(?<![.!?] )(?<=[\s\S])\b[A-Z][a-z]+").unwrap());
static HARD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\b(\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|last|latest|earliest|oldest|newest|youngest|most|least|largest|smallest|biggest|tallest|highest|lowest|longest|shortest|best|worst|fastest|slowest)\b|\b(?:war|part|chapter|phase|stage|henry|louis|george|edward|elizabeth|charles|richard|james|william)\s+([ivx]{1,4})\b").unwrap());
static RX: LazyLock<[Regex; 6]> = LazyLock::new(|| [r"^(?:when|what year|which year|what date|in what year|in which year)\b", r"(?i)\d{3,4}|\b(?:january|february|march|april|may|june|july|august|september|october|november|december|century|decade|bc|ad|ago)\b", r"^(?:how (?:many|much|old|long|far|tall|big|high|deep|fast)\b|what (?:percentage|percent|number)\b)", r"(?i)\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|million|billion|dozen|half|several|few)\b", r"^who(?:m|se)?\b", r"$^"].map(|r| Regex::new(r).unwrap()));
pub struct Qa { vocab: HashMap<Vec<u8>, u32>, ptr: [Vec<i64>; 3], cols: [Vec<i32>; 3], idf: Vec<f32>, widf: Vec<f32>, qlen: Vec<f32>, ans: Vec<u8>, off: Vec<i64>, qs: Vec<Vec<u8>>, src: Vec<i64> }
#[derive(Clone, Debug)]
pub struct Hit { pub score: f64, pub cov: f64, pub q: String, pub a: Vec<u8>, pub src: i64, pub id: usize }
fn env<T: FromStr>(k: &str, d: T) -> T { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
pub fn stem(w: &[u8]) -> Vec<u8> { if w.ends_with(b"oes") { w[..w.len() - 2].to_vec() } else if w.ends_with(b"ies") && w.len() > 4 { [&w[..w.len() - 3], b"y"].concat() } else if w.ends_with(b"s") && !(w.ends_with(b"ss") || w.ends_with(b"us") || w.ends_with(b"is")) && w.len() > 3 { w[..w.len() - 1].to_vec() } else { w.to_vec() } }
pub fn words(b: &[u8]) -> Vec<Vec<u8>> { let l = b.to_ascii_lowercase(); TOK.find_iter(&l).map(|m| m.as_bytes()).filter(|w| !STOP.contains(std::str::from_utf8(w).unwrap_or(""))).map(stem).collect() }
fn uniq(v: Vec<Vec<u8>>) -> Vec<Vec<u8>> { let mut s = HashSet::new(); v.into_iter().filter(|w| s.insert(w.clone())).collect() }
static GEN: LazyLock<HashSet<&str>> = LazyLock::new(|| "who mean meaning definition define defined explain know called thing term word stand".split(' ').collect());
static FRAME: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^(?:(?:what|who) (?:is|are|was|were) (?:the |an? )?(.+?)|what does (?:the |an? )?(.+?) mean)\?$").unwrap());
static TK: LazyLock<BRe> = LazyLock::new(|| BRe::new(r"[a-z0-9']+").unwrap());
pub fn toks(t: &str) -> Vec<Vec<u8>> { let l = t.to_lowercase(); TK.find_iter(l.as_bytes()).map(|m| stem(m.as_bytes())).collect() }
pub fn wgate(q: &str, c: &str, a: &[u8]) -> bool {
    let t = FRAME.captures(c.trim()).and_then(|m| m.get(1).or(m.get(2))).map_or(vec![], |x| toks(x.as_str()));
    let u = toks(q);
    let (cw, aw): (HashSet<Vec<u8>>, HashSet<Vec<u8>>) = (words(c.as_bytes()).into_iter().collect(), words(a).into_iter().collect());
    !t.is_empty() && u.len() >= t.len() && (0..=u.len() - t.len()).any(|i| u[i..i + t.len()] == t[..]) && uniq(words(q.as_bytes())).into_iter().filter(|w| !cw.contains(w) && !GEN.contains(std::str::from_utf8(w).unwrap_or(""))).all(|w| aw.contains(&w) || (w.len() >= 5 && aw.iter().any(|x| x.len() >= 5 && x[..5] == w[..5])))
}
pub fn hard(q: &str, c: &str, a: &[u8]) -> bool {
    let n = |t: &str| HARD.captures_iter(t).map(|m| m.get(1).or(m.get(2)).map_or(String::new(), |x| x.as_str().to_lowercase())).collect::<HashSet<_>>();
    let (t, ql) = (String::from_utf8_lossy(a), q.trim().to_lowercase());
    let need = if RX[0].is_match(&ql) { RX[1].is_match(&t) } else if RX[2].is_match(&ql) { RX[3].is_match(&t) } else if RX[4].is_match(&ql) { WHO.is_match(&t).unwrap_or(false) } else { true };
    n(q) == n(c) && need
}
impl Qa {
    pub fn load(path: Option<&str>) -> io::Result<Qa> {
        let p = path.map(String::from).or_else(|| std::env::var("AMNI_RAY_QA").ok()).unwrap_or_else(|| format!("{}/exports/gf17_continuum/ray_qa_index.npz", std::env::var("AMNI_HOME").unwrap_or_else(|_| ".".into())));
        let z = Npz::open(&p)?;
        let g = |k: &str| z.get(k);
        let vocab = g("vocab")?.objs()?.into_iter().enumerate().map(|(i, w)| (w.to_vec(), i as u32)).collect();
        let qlen = g("qlen")?.f32s();
        Ok(Qa { vocab, ptr: [g("ptr")?.i64s(), g("aptr")?.i64s(), g("bptr")?.i64s()], cols: [g("cols")?.vec(), g("acols")?.vec(), g("bcols")?.vec()], idf: g("idf")?.f32s(), widf: if z.has("widf") { g("widf")?.f32s() } else { g("idf")?.f32s() }, ans: g("ans")?.vec(), off: g("off")?.i64s(), qs: g("qs")?.objs()?.into_iter().map(<[u8]>::to_vec).collect(), src: if z.has("src") { g("src")?.i64s() } else { vec![0; qlen.len()] }, qlen })
    }
    pub fn len(&self) -> usize { self.qlen.len() }
    pub fn is_empty(&self) -> bool { self.qlen.is_empty() }
    pub fn answer(&self, i: usize) -> Vec<u8> { self.ans[self.off[i] as usize..self.off[i + 1] as usize].to_vec() }
    fn post(&self, k: usize, i: usize) -> &[i32] { &self.cols[k][self.ptr[k][i] as usize..self.ptr[k][i + 1] as usize] }
    pub fn overlap(&self, q: &str, c: &str, wk: bool) -> (f64, f64, f64) {
        let (a, b): (HashSet<Vec<u8>>, HashSet<Vec<u8>>) = (words(q.as_bytes()).into_iter().collect(), words(c.as_bytes()).into_iter().collect());
        let w = |x: &Vec<u8>| self.vocab.get(x).map_or(1.0, |&i| (if wk { &self.widf } else { &self.idf })[i as usize] as f64);
        let sm = |it: &mut dyn Iterator<Item = &Vec<u8>>| { let mut v: Vec<&Vec<u8>> = it.collect(); v.sort(); v.into_iter().map(w).sum::<f64>() };
        let (i, ql) = (sm(&mut a.intersection(&b)), q.to_lowercase());
        let pn = if PROPER.captures_iter(c).filter_map(Result::ok).any(|m| !ql.contains(&m[1].to_lowercase())) { 0.6 } else { 1.0 };
        (pn * i / sm(&mut a.union(&b)).max(1e-9), i / sm(&mut a.iter()).max(1e-9), pn * i / sm(&mut b.iter()).max(1e-9))
    }
    pub fn search(&self, q: &str, k: usize) -> Vec<Hit> {
        let qw = words(q.as_bytes());
        let ids: Vec<usize> = uniq(qw.clone()).iter().filter_map(|w| self.vocab.get(w).map(|&i| i as usize)).collect();
        let bi: Vec<usize> = qw.windows(2).filter_map(|p| self.vocab.get(&[p[0].as_slice(), b" ", &p[1]].concat()).map(|&i| i as usize)).collect();
        if ids.is_empty() || qw.iter().any(|w| !self.vocab.contains_key(w) && w.len() >= 4) { return vec![]; }
        let (n, mut sc, mut hit) = (self.len(), vec![0f32; self.len()], vec![]);
        let tot = ids.iter().fold(0f32, |s, &i| s + self.idf[i]);
        let mut add = |d: &[i32], v: f32, sc: &mut Vec<f32>| d.iter().for_each(|&j| { hit.push(j as usize); sc[j as usize] += v; });
        for &i in &ids { add(self.post(0, i), self.idf[i], &mut sc); add(self.post(1, i), 0.4 * self.idf[i], &mut sc); }
        let mv = (1.5 * (pairwise(&ids.iter().map(|&i| self.idf[i]).collect::<Vec<f32>>()) / ids.len() as f32) as f64) as f32;
        for &b in &bi { add(self.post(2, b), mv, &mut sc); }
        let r = ids.iter().copied().fold(ids[0], |r, i| if self.idf[i] > self.idf[r] { i } else { r });
        let mut must = vec![false; n];
        self.post(0, r).iter().chain(self.post(1, r)).for_each(|&j| must[j as usize] = true);
        let any = must.iter().any(|&m| m);
        hit.sort_unstable();
        hit.dedup();
        let nl = ids.len() as f32;
        let mut c: Vec<(f32, usize)> = hit.into_iter().map(|j| { let s = sc[j] / (self.qlen[j] * nl).sqrt(); (if any && !must[j] { s * 0.6 } else { s }, j) }).filter(|x| x.0 > 0.0).collect();
        c.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        let (mut na, mut nb) = (0usize, 0usize);
        c.retain(|x| if self.src[x.1] == 2 { nb += 1; nb <= k } else { na += 1; na <= k });
        c.into_iter().map(|(s, j)| Hit { score: s as f64, cov: ((s * ((self.qlen[j] * nl) as f64).sqrt() as f32) / (tot as f64).max(1e-9) as f32) as f64, q: String::from_utf8_lossy(&self.qs[j]).into_owned(), a: self.answer(j), src: self.src[j], id: j }).collect()
    }
}
pub fn pick_all(e: &mut RayField, qa: &Qa, q: &str) -> Option<(Hit, Vec<Hit>, Vec<f64>)> {
    let c = qa.search(q, env("QA_K", 20));
    if c.is_empty() { return None; }
    let a: Vec<Vec<u8>> = c.iter().map(|h| h.a[..h.a.len().min(env("QA_SPAN", 160))].to_vec()).collect();
    let (pc, pu) = (e.score(&vec![format!("User: {q}\nAdam: ").into_bytes(); c.len()], &a), e.score(&vec![b"User: \nAdam: ".to_vec(); c.len()], &a));
    let (rec, prec, lam, yh) = (env("QA_REC", 0.8), env("QA_PREC", 0.55), env("QA_LAM", 0.5), env("QA_YAHOO", 0.15));
    let f: Vec<(f64, f64)> = c.iter().enumerate().map(|(i, h)| { let ov = qa.overlap(q, &h.q, h.src == 2); let ok = ov.1 >= rec && ov.2 >= prec && hard(q, &h.q, &h.a[..h.a.len().min(300)]) && (h.src != 2 || wgate(q, &h.q, &h.a)); (if ok { ov.0 + lam * ((pc[i] - pu[i]) / a[i].len().max(1) as f64) - yh * h.src as f64 } else { -9.0 }, if ok { ov.0 } else { 0.0 }) }).collect();
    let i = (0..f.len()).fold(0, |m, i| if f[i].0 > f[m].0 { i } else { m });
    Some((Hit { cov: f[i].1, ..c[i].clone() }, c, f.iter().map(|x| x.0).collect()))
}
pub fn pick(e: &mut RayField, qa: &Qa, q: &str) -> Option<Hit> { pick_all(e, qa, q).map(|x| x.0) }
pub fn guide(e: &mut RayField, qa: &Qa, q: &str, cov: Option<f64>) -> Option<Vec<u8>> { pick(e, qa, q).filter(|h| h.cov >= cov.unwrap_or_else(|| env("QA_COV", 0.3))).map(|h| h.a) }
