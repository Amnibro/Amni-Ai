use crate::v19::bh;
use std::collections::{HashMap, HashSet};
const K: u32 = 83492791;
const GK: u32 = 0x9E37_79B1;
const SELFK: u32 = 1;
const ISINK: u32 = 3;
const NT: usize = 48;
const FM: u32 = 0b11_0011_1110;
macro_rules! hc { ($($n:ident = $w:literal),*) => { $(const $n: u32 = bh($w);)* } }
hc!(MY = b"my", OUR = b"our", HIS = b"his", HER = b"her", THEIR = b"their", ITS = b"its", I = b"i", ME = b"me", MYSELF = b"myself", MINE = b"mine", HE = b"he", SHE = b"she", THEY = b"they", HIM = b"him", THEM = b"them", IT = b"it", IS = b"is", AM = b"am", ARE = b"are", WOULD = b"would", HAVE = b"have", WILL = b"will", THE = b"the", AND = b"and", BUT = b"but", SO = b"so", WHILE = b"while", WHEREAS = b"whereas", SAME = b"same", SHARE = b"share", SHARES = b"shares", SHARED = b"shared", SHARING = b"sharing", AS = b"as", OLD = b"old", WITH = b"with", IF = b"if", THAT = b"that", THERE = b"there", HERE = b"here", WHAT = b"what", WHERE = b"where", WHO = b"who", WHEN = b"when", HOW = b"how", REMIND = b"remind", TELL = b"tell", LET = b"let", ASK = b"ask", GIVE = b"give", SHOW = b"show", IN = b"in", ON = b"on", AT = b"at", TO = b"to", OF = b"of", FROM = b"from", ABOUT = b"about", FOR = b"for", BY = b"by", INTO = b"into", ALLERGY = b"allergy", ALLERGIC = b"allergic", ALLERGIES = b"allergies", EPIPEN = b"epipen", HIVES = b"hives", THING = b"thing", ONE = b"one", HOME = b"home", DO = b"do", DOES = b"does", DID = b"did", DOING = b"doing", AGE = b"age", A = b"a", AN = b"an", SAYS = b"says", SAID = b"said", THINKS = b"thinks", RECKONS = b"reckons", TOLD = b"told", FALLS = b"falls", COMES = b"comes", LANDS = b"lands", AROUND = b"around", EVERY = b"every", BEEN = b"been", HAS = b"has", HAD = b"had", CITY = b"city", TOWN = b"town", VILLAGE = b"village", LOCATED = b"located", OVER = b"over", PART = b"part", SITUATED = b"situated", RIGHT = b"right", JUST = b"just", UP = b"up", DOWN = b"down", OUT = b"out", ACTUALLY = b"actually", ALSO = b"also", STATE = b"state", COUNTRY = b"country", REGION = b"region", PROVINCE = b"province", PREFECTURE = b"prefecture", COUNTY = b"county", WHICH = b"which", STUDY = b"study", STUDIED = b"studied", DEGREE = b"degree", SCHOOL = b"school", COLLEGE = b"college", UNIVERSITY = b"university", ENROLLED = b"enrolled", OTHER = b"other", BEST = b"best");
const MON: [u32; 12] = [bh(b"january"), bh(b"february"), bh(b"march"), bh(b"april"), bh(b"may"), bh(b"june"), bh(b"july"), bh(b"august"), bh(b"september"), bh(b"october"), bh(b"november"), bh(b"december")];
const FAV: [u32; 7] = [bh(b"favourite"), bh(b"favorite"), bh(b"fave"), bh(b"fav"), bh(b"go-to"), bh(b"all-time"), bh(b"top")];
const ADV: [u32; 22] = [bh(b"really"), bh(b"just"), bh(b"also"), bh(b"always"), bh(b"still"), bh(b"truly"), bh(b"totally"), bh(b"absolutely"), bh(b"seriously"), bh(b"pretty"), bh(b"super"), bh(b"so"), bh(b"very"), bh(b"actually"), bh(b"even"), bh(b"currently"), bh(b"usually"), bh(b"sometimes"), bh(b"often"), bh(b"mostly"), bh(b"definitely"), bh(b"honestly")];
const ADV2: [u32; 2] = [bh(b"literally"), bh(b"basically")];
const DIT: [u32; 17] = [bh(b"give"), bh(b"gave"), bh(b"feed"), bh(b"fed"), bh(b"buy"), bh(b"bought"), bh(b"get"), bh(b"got"), bh(b"bring"), bh(b"brought"), bh(b"make"), bh(b"made"), bh(b"send"), bh(b"sent"), bh(b"offer"), bh(b"hand"), bh(b"cook")];
const DIT2: [u32; 1] = [bh(b"cooked")];
pub fn pk(hs: &[u32]) -> u32 { hs[1..].iter().fold(hs[0], |k, &h| k.wrapping_mul(GK) ^ h) }
pub fn fold4(v: &[u8]) -> u32 { v.iter().take(4).filter(|&&c| c != 32).fold(0u32, |h, &c| h.wrapping_mul(K).wrapping_add(c as u32 + 1)) }
#[derive(Default)]
pub struct G2Lx { cue: HashMap<u32, Vec<(u32, u32)>>, rn: HashMap<u32, u32>, ty: HashMap<u32, (u32, u32)>, head: HashSet<u32>, verb: HashSet<u32>, md: HashSet<u32>, stop: HashSet<u32>, meta: HashSet<u32>, hypo: HashSet<u32>, skip: HashSet<u32>, join: HashSet<u32>, fnc: HashMap<u32, u32>, cf: HashSet<u32>, vany: HashSet<u32> }
impl G2Lx {
    pub fn new() -> G2Lx {
        let mut x = G2Lx::default();
        for l in include_str!("v19_gen2.tsv").lines() {
            let f: Vec<&str> = l.split('\t').collect();
            if f.len() < 5 { continue; }
            let (c, r, fl) = (f[2].parse::<u32>().unwrap_or(0), f[3].parse::<u32>().unwrap_or(0), f[4].parse::<u32>().unwrap_or(0));
            let ws: Vec<u32> = f[1].split(' ').map(|w| bh(w.as_bytes())).collect();
            let k = pk(&ws);
            match c { 1 => { x.cue.entry(k).or_default().push((r, fl)); x.cf.insert(ws[0]); } 2 => { x.rn.entry(k).or_insert(r); } 3 => { x.ty.entry(k).or_insert((r, fl)); } 12 => { *x.fnc.entry(k).or_insert(0) |= 1 << r; } 4 => { x.head.insert(k); } 5 => { x.verb.insert(k); } 6 => { x.md.insert(k); } 7 => { x.stop.insert(k); } 8 => { x.meta.insert(k); } 9 => { x.hypo.insert(k); } 10 => { x.skip.insert(k); } 11 => { x.join.insert(k); } 13 => { x.vany.insert(k); } _ => {} }
        }
        x
    }
    fn f(&self, h: u32) -> u32 { self.fnc.get(&h).copied().unwrap_or(0) }
    fn fb(&self, h: u32, b: u32) -> bool { self.f(h) >> b & 1 != 0 }
}
pub trait GMem { fn put(&mut self, e: u32, r: u32, v: &[u8], vs: u32); fn latest(&self, e: u32, r: u32) -> Option<(Vec<u8>, u32)>; fn lrel(&self, e: u32, rs: &[u32]) -> u32; fn findval(&self, v: &[u8]) -> Option<(u32, u32)>; }
#[derive(Clone, Copy, Default)]
pub struct Tk { pub h: u32, pub r: [u8; 24], pub l: u8, pub cap: bool, pub num: bool, pub pos: bool, pub neg: bool, pub bd: bool }
impl Tk { fn raw(&self) -> &[u8] { &self.r[..self.l as usize] } fn app(&mut self, x: &[u8]) { for &c in x { if (self.l as usize) < 24 { self.r[self.l as usize] = c; self.l += 1; } } } fn mk(h: u32, w: &[u8]) -> Tk { let mut t = Tk { h, ..Tk::default() }; t.app(w); t } }
type Vb = ([u8; 24], u8);
#[derive(Clone)]
pub struct G2 { tk: [Tk; NT], n: usize, pub ante: u32, links: Vec<(u32, u32, u32)>, lastp: Vb, pend: Option<(u32, u32)>, lastv: Option<Vb>, tf: Vec<(u32, u32, Vb, u32)>, pub qe: u32, pub qr: u32 }
impl Default for G2 { fn default() -> Self { G2 { tk: [Tk::default(); NT], n: 0, ante: 0, links: Vec::new(), lastp: ([0; 24], 0), pend: None, lastv: None, tf: Vec::new(), qe: 0, qr: 0 } } }
fn vb(v: &[u8]) -> Vb { let mut o = [0u8; 24]; let l = v.len().min(24); o[..l].copy_from_slice(&v[..l]); (o, l as u8) }
fn vs(v: &Vb) -> &[u8] { &v.0[..v.1 as usize] }
#[derive(Clone, Copy)]
struct Mn { a: usize, b: usize, k: u32, kd: u8 }
#[derive(Clone, Copy)]
struct Rh { j0: usize, j1: usize, rel: u32, own: u32, fav: bool, st: usize }
enum Sp { No, Cut, S(usize, usize) }
impl G2 {
    pub fn newturn(&mut self) { (self.pend, self.lastv, self.qe, self.qr) = (None, None, 0, 0); self.tf.clear(); }
    pub fn word(&mut self, w: &[u8], pbd: i32) {
        let w = &w[..w.len().min(16)];
        let lw = w.to_ascii_lowercase();
        let h = bh(&lw);
        let p = (self.n > 0 && !self.tk[self.n - 1].bd).then(|| self.n - 1);
        let push = |s: &mut G2, t: Tk| if s.n < NT { s.tk[s.n] = t; s.n += 1; };
        if let (2, Some(p)) = (pbd, p) {
            let ph = self.tk[p].h;
            if lw == b"s" { if [HE, SHE, IT, THAT, THERE, WHAT, WHERE, WHO, WHEN, HOW, HERE].contains(&ph) { push(self, Tk::mk(IS, b"s")) } else { self.tk[p].pos = true; self.tk[p].app(b"'s"); } return; }
            if lw == b"t" { self.tk[p].neg = true; return; }
            if lw == b"m" || lw == b"re" { push(self, Tk::mk(if lw == b"m" { AM } else { ARE }, &lw)); return; }
            if lw == b"d" { push(self, Tk::mk(WOULD, &lw)); return; }
            if lw == b"ve" || lw == b"ll" { push(self, Tk::mk(if lw == b"ve" { HAVE } else { WILL }, &lw)); return; }
            self.tk[p].app(b"'"); self.tk[p].app(w); self.tk[p].h = bh(&self.tk[p].raw().to_ascii_lowercase()); return;
        }
        if let (1, Some(p)) = (pbd, p) { self.tk[p].app(b"-"); self.tk[p].app(w); self.tk[p].h = bh(&self.tk[p].raw().to_ascii_lowercase()); self.tk[p].num = self.tk[p].num && !w.is_empty() && w.iter().all(u8::is_ascii_digit); return; }
        let t = Tk { cap: w.first().is_some_and(u8::is_ascii_uppercase), num: !w.is_empty() && w.iter().all(u8::is_ascii_digit), ..Tk::mk(h, w) };
        push(self, t);
    }
    pub fn comma(&mut self) { if self.n > 0 && self.n < NT { self.tk[self.n] = Tk { bd: true, ..Tk::mk(0, b",") }; self.n += 1; } }
    pub fn end(&mut self, nq: bool, lx: &G2Lx, mem: &mut impl GMem) { let n = self.n; self.n = 0; if n > 0 { let t: Vec<Tk> = self.tk[..n].to_vec(); self.run(&t, nq, lx, mem); } }
    fn fire(&mut self, e: u32, r: u32, v: &[u8], mem: &mut impl GMem, d: u32) { let l = [7, 10, 11]; for (t, rel, s) in self.links.clone() { if s == e && (rel == r || (l.contains(&rel) && l.contains(&r))) && d < 2 { mem.put(t, r, v, 0); self.fire(t, r, v, mem, d + 1); } } }
    #[allow(clippy::cognitive_complexity)]
    fn run(&mut self, t: &[Tk], nq: bool, x: &G2Lx, mem: &mut impl GMem) {
        let n = t.len();
        let hs: Vec<u32> = t.iter().map(|z| z.h).collect();
        let (cap, num) = (t.iter().map(|z| z.cap).collect::<Vec<_>>(), t.iter().map(|z| z.num).collect::<Vec<_>>());
        let capx: Vec<bool> = t.iter().map(|z| z.cap || (!z.bd && z.l > 0 && z.r[0].is_ascii_digit())).collect();
        let wd = |i: usize| i < n && !t[i].bd;
        let mut cl = vec![0usize; n];
        let mut c = 0;
        for i in 0..n { if t[i].bd || ([AND, BUT, SO, WHILE, WHEREAS].contains(&hs[i]) && i + 1 < n && [MY, OUR, I, HE, SHE, THEY, HIS, HER, THEIR].contains(&hs[i + 1])) { c += 1; } cl[i] = c; }
        let neg: Vec<bool> = t.iter().map(|z| z.neg || x.fb(z.h, 10)).collect();
        let (mut ifr, mut hyp) = (vec![false; n], vec![false; n]);
        let mut inif = false;
        for i in 0..n { inif = (inif && !t[i].bd) || hs[i] == IF; ifr[i] = inif; }
        for i in 0..n { hyp[i] = (i > 0 && hyp[i - 1] && cl[i] == cl[i - 1]) || x.hypo.contains(&hs[i]); }
        let (mut cov, mut men, mut rnh, mut mself) = (vec![0u8; n], Vec::<Mn>::new(), Vec::<Rh>::new(), false);
        let rnmatch = |j: usize| -> Option<(usize, u32, bool)> {
            let (mut k, mut fav) = (j, false);
            loop {
                for l in [4, 3, 2, 1] { if k + l <= n && (k..k + l).all(wd) { if let Some(&r) = x.rn.get(&pk(&hs[k..k + l])) { let tt = hs[k] == THING || hs[k] == ONE; if !(r == 0 && !fav && tt) { return Some((k + l, if r == 0 && fav && tt { 7 } else { r }, fav)); } } } }
                if k < n && wd(k) && x.md.contains(&hs[k]) && k - j < 3 { fav |= FAV.contains(&hs[k]); k += 1; } else { break; }
            }
            (fav && (k >= n || !wd(k) || x.fb(hs[k], 5) || x.stop.contains(&hs[k]))).then_some((k, 0, fav))
        };
        let npend = |j: usize, inv: bool| -> usize {
            let (mut k, mut hd) = (j, false);
            while k < n && k - j < 4 && wd(k) {
                let (h, r) = (hs[k], t[k].raw());
                if cap[k] || num[k] || x.f(h) != 0 || x.stop.contains(&h) || neg[k] || (hd && !x.head.contains(&h)) { break; }
                if k == j && (x.verb.contains(&h) || x.cf.contains(&h)) && !t[k].pos && !(k + 1 < n && wd(k + 1) && x.f(hs[k + 1]) == 0 && !x.stop.contains(&hs[k + 1]) && !cap[k + 1]) { break; }
                if k > j {
                    let isv = x.verb.contains(&h) || x.cf.contains(&h) || (!x.head.contains(&h) && ((r.ends_with(b"s") && !r.ends_with(b"ss")) || r.ends_with(b"ed") || r.ends_with(b"ing"))) || (inv && x.vany.contains(&h) && (k + 1 >= n || !wd(k + 1) || ((x.f(hs[k + 1]) != 0 || x.stop.contains(&hs[k + 1])) && !x.fb(hs[k + 1], 6) && !x.fb(hs[k + 1], 5))));
                    if inv && hd && x.vany.contains(&h) && !x.head.contains(&h) { break; }
                    if isv && !(k + 1 < n && wd(k + 1) && x.head.contains(&hs[k + 1]) && x.f(hs[k + 1]) == 0 && !cap[k + 1] && !t[k].pos && !(inv && (k + 2 >= n || !wd(k + 2)))) { break; }
                }
                hd = hd || x.head.contains(&h);
                k += 1;
                if t[k - 1].pos { break; }
            }
            k
        };
        let ante = self.ante;
        let resolve = |men: &Vec<Mn>, i: usize, kind: u32| -> u32 {
            let ks: Vec<u32> = men.iter().filter(|m| m.k != SELFK).map(|m| m.k).collect();
            if kind == 2 { let sub = men.iter().find(|m| cl[m.a] == cl[i] && m.kd != 3).map(|m| m.k); let ks2: Vec<u32> = ks.iter().copied().filter(|&z| Some(z) != sub).collect(); if let Some(&z) = ks2.last() { return z; } }
            ks.last().copied().unwrap_or(ante)
        };
        let ocue = |i: usize| i > 0 && x.cue.get(&hs[i - 1]).is_some_and(|v| v.iter().any(|y| y.1 & 1 != 0));
        let mut i = 0;
        while i < n {
            let h = hs[i];
            if !wd(i) { i += 1; continue; }
            let oc = ocue(i);
            if h == MY || h == OUR || ([HIS, HER, THEIR, ITS].contains(&h) && wd(i + 1) && !x.fb(hs[i + 1], 1) && !(h == HER && oc) && !(h == HER && x.f(hs[i + 1]) != 0)) {
                let own = if h == MY || h == OUR { SELFK } else { resolve(&men, i, 1) };
                let (mut j, mut key, inv, dit) = (i + 1, None::<u32>, i > 0 && x.fb(hs[i - 1], 6), i > 0 && (DIT.contains(&hs[i - 1]) || DIT2.contains(&hs[i - 1])));
                while j < n {
                    let r = rnmatch(j).filter(|r| !(r.1 == 9 && t[r.0 - 1].pos));
                    if let Some(r) = r {
                        rnh.push(Rh { j0: j, j1: r.0, rel: r.1, own: key.unwrap_or(own), fav: r.2, st: i });
                        (j..r.0).for_each(|z| cov[z] = 2);
                        if let Some(m) = men.last_mut() { if m.b == j && m.kd == 3 { m.kd = 0; } }
                        j = r.0;
                        break;
                    }
                    let k = npend(j, inv);
                    let k = if dit && k > j + 1 && !(k < n && wd(k) && x.fb(hs[k], 1)) { j + 1 } else { k };
                    if k == j { break; }
                    for z in j..k { key = Some(match key { None => hs[z], Some(q) => q.wrapping_mul(K) ^ hs[z] }); }
                    men.push(Mn { a: j, b: k, k: key.unwrap_or(0), kd: 0 });
                    (j..k).for_each(|z| cov[z] = 1);
                    j = k;
                    if !t[k - 1].pos { break; }
                    men.last_mut().unwrap().kd = 3;
                }
                if let Some(m) = men.last_mut() { if m.kd == 3 && m.b == j { m.kd = 0; } }
                if cov[i] == 0 { cov[i] = 3; }
                i = j.max(i + 1);
                continue;
            }
            if [I, ME, MYSELF, MINE].contains(&h) {
                let meta = [i + 1, i + 2].iter().any(|&z| z < n && !(z == i + 2 && !x.fb(hs[i + 1], 6)) && wd(z) && x.meta.contains(&hs[z]));
                let objimp = h == ME && i > 0 && [REMIND, TELL, LET, ASK, GIVE, SHOW].contains(&hs[i - 1]);
                if !meta && !objimp { men.push(Mn { a: i, b: i + 1, k: SELFK, kd: 1 }); cov[i] = 1; } else if meta { mself = true; }
                i += 1;
                continue;
            }
            if [HE, SHE, THEY, HIM, THEM].contains(&h) || (h == HER && !(wd(i + 1) && x.f(hs[i + 1]) == 0 && !oc)) || (h == IT && (oc || (wd(i + 1) && (x.cf.contains(&hs[i + 1]) || x.fb(hs[i + 1], 5))))) {
                let l = men.len();
                let rp = i >= 2 && [SAYS, SAID, THINKS, RECKONS, TOLD].contains(&hs[i - 1]) && l >= 2 && men[l - 1].kd == 0 && men[l - 2].kd == 3 && men[l - 2].b == men[l - 1].a;
                let e = if rp { men[l - 2].k } else { resolve(&men, i, if h == HIM || h == THEM || h == HER || (h == IT && oc) { 2 } else { 1 }) };
                if e != 0 { men.push(Mn { a: i, b: i + 1, k: e, kd: if h == IT { 4 } else { 2 } }); cov[i] = 1; }
                i += 1;
                continue;
            }
            i += 1;
        }
        let mut cues: Vec<(usize, usize, Vec<(u32, u32)>)> = Vec::new();
        let mut i = 0;
        while i < n {
            if !wd(i) || cov[i] == 1 || cov[i] == 3 { i += 1; continue; }
            let mut ix = vec![i];
            while ix.len() < 6 && *ix.last().unwrap() + 1 < n && wd(ix.last().unwrap() + 1) && matches!(cov[ix.last().unwrap() + 1], 0 | 2) {
                let lst = *ix.last().unwrap();
                let mut q = lst + 1;
                while q < n && wd(q) && (ADV.contains(&hs[q]) || ADV2.contains(&hs[q])) && q - lst < 3 { q += 1; }
                if q < n && wd(q) && matches!(cov[q], 0 | 2) { ix.push(q); } else { break; }
            }
            let mut hit = None;
            for l in (1..=ix.len()).rev() {
                if ix[..l].iter().all(|&z| cov[z] == 2) { continue; }
                let rs: Vec<(u32, u32)> = x.cue.get(&pk(&ix[..l].iter().map(|&z| hs[z]).collect::<Vec<_>>())).map(|v| v.iter().copied().filter(|y| nq || y.1 & 16 == 0).collect()).unwrap_or_default();
                if !rs.is_empty() && !(l == 1 && i > 0 && x.fb(hs[i - 1], 1) && hs[i - 1] != THAT) { hit = Some((i, ix[l - 1] + 1, rs)); break; }
            }
            if let Some(hh) = hit { i = hh.1; cues.push(hh); } else { i += 1; }
        }
        let mut free: Vec<(usize, usize, u32)> = Vec::new();
        for i in 0..n { if wd(i) && cov[i] == 0 && (i == 0 || !wd(i - 1) || x.fb(hs[i - 1], 1) || x.fb(hs[i - 1], 9) || x.fb(hs[i - 1], 7) || [SAME, OTHER, BEST].contains(&hs[i - 1]) || x.md.contains(&hs[i - 1])) { if let Some(r) = rnmatch(i) { if r.1 != 0 { free.push((i, r.0, r.1)); } } } }
        let alg = hs.iter().any(|h| [ALLERGY, ALLERGIC, ALLERGIES, EPIPEN, HIVES].contains(h));
        let content = |k: usize| wd(k) && x.f(hs[k]) == 0 && !x.stop.contains(&hs[k]) && !neg[k] && cov[k] == 0;
        let cst: Vec<usize> = cues.iter().filter(|c| !c.2.iter().all(|y| y.1 & 2 != 0)).map(|c| c.0).collect();
        let prepset = [ON, IN, AT, WITH, AS, TO, OF, FROM, ABOUT, FOR, BY, INTO];
        let vspan = |j: usize, rel: u32, cut: bool| -> Sp {
            let mut k = j;
            while k < n && wd(k) && cov[k] == 0 {
                if cut && cst.contains(&k) { return Sp::Cut; }
                if let Some(r) = rnmatch(k) { if r.1 != 0 && k == j && !cap[k] && r.0 < n && wd(r.0) && (capx[r.0] || x.ty.get(&hs[r.0]).map_or(0, |y| y.0) == r.1) { k = r.0; continue; } }
                if hs[k] == THE && k + 1 < n && wd(k + 1) && cov[k + 1] == 0 { if let Some(r) = rnmatch(k + 1) { if r.1 != 0 { k = r.0; continue; } } }
                if !capx[k] && ((x.skip.contains(&hs[k]) && !(hs[k] == THE && [2, 7, 8, 11].contains(&rel) && k + 1 < n && content(k + 1) && !cap[k + 1])) || x.stop.contains(&hs[k]) || x.md.contains(&hs[k]) || (prepset.contains(&hs[k]) && k == j)) { k += 1; continue; }
                break;
            }
            if k >= n || !wd(k) || cov[k] != 0 { return Sp::No; }
            let k0 = k;
            if cap[k] && x.f(hs[k]) & FM != 0 { return Sp::No; }
            if capx[k] || num[k] {
                while k < n && wd(k) && cov[k] == 0 && k - k0 < 4 && (capx[k] || num[k] || (x.join.contains(&hs[k]) && k + 1 < n && wd(k + 1) && cap[k + 1])) { k += 1; }
                while k < n && k - k0 < 4 && content(k) && !cap[k] && x.ty.get(&hs[k]).is_some_and(|y| [11, 5, 7, 9].contains(&y.0)) { k += 1; }
            } else {
                for l in [6, 5, 4, 3, 2] { if k0 + l <= n && (k0..k0 + l).all(wd) && x.ty.contains_key(&pk(&hs[k0..k0 + l])) { return Sp::S(k0, k0 + l); } }
                if hs[k] == THE { k += 1; }
                while k < n && k - k0 < 4 {
                    if content(k) || (wd(k) && cap[k] && cov[k] == 0) { k += 1; continue; }
                    if wd(k) && [IN, ON, AT, FROM, WITH].contains(&hs[k]) && (k0..k).any(|z| t[z].raw().ends_with(b"ing")) && content(k + 1) { k += 2; continue; }
                    if wd(k) && (hs[k] == AND || hs[k] == OF) && content(k + 1) && x.ty.contains_key(&pk(&hs[k0..k + 2])) { k += 2; continue; }
                    break;
                }
            }
            if k > k0 && !(k - k0 == 1 && hs[k0] == THE) { Sp::S(k0, k) } else { Sp::No }
        };
        let vspo = |j: usize, rel: u32| match vspan(j, rel, false) { Sp::S(a, b) => Some((a, b)), _ => None };
        let vback = |i: usize, used: &[bool]| -> Option<(usize, usize)> {
            let (mut k, mut g) = (i as isize - 1, 0);
            while k >= 0 && g < 7 && !(content(k as usize) || (wd(k as usize) && capx[k as usize] && cov[k as usize] == 0)) { k -= 1; g += 1; }
            if k < 0 || g >= 7 || used[k as usize] { return None; }
            let k1 = k + 1;
            if cap[k as usize] && x.f(hs[k as usize]) & FM != 0 { return None; }
            if capx[k as usize] || num[k as usize] { while k >= 0 && wd(k as usize) && cov[k as usize] == 0 && k1 - k <= 4 && (capx[k as usize] || num[k as usize] || x.join.contains(&hs[k as usize])) && !(cap[k as usize] && x.f(hs[k as usize]) & FM != 0) { k -= 1; } } else { while k >= 0 && k1 - k <= 4 && (content(k as usize) || (wd(k as usize) && cap[k as usize] && cov[k as usize] == 0)) { k -= 1; } }
            let mut k0 = k + 1;
            while k0 < k1 && x.join.contains(&hs[k0 as usize]) && !cap[k0 as usize] { k0 += 1; }
            (k1 > k0).then_some((k0 as usize, k1 as usize))
        };
        let vtype = |a: usize, b: usize| -> u32 {
            if num[a] { return 14; }
            if MON.contains(&hs[a]) { return 15; }
            if cap[a] {
                for l in (1..=b - a).rev() { if let Some(y) = x.ty.get(&pk(&hs[a..a + l])) { if y.1 & 1 != 0 { return y.0; } } }
                if (a..b).any(|z| x.ty.get(&hs[z]).is_some_and(|y| y.1 == 3)) { return 13; }
            }
            if let Some(y) = x.ty.get(&pk(&hs[a..b])).or_else(|| x.ty.get(&hs[b - 1])) { if !(y.1 & 1 != 0 && !cap[a]) { return y.0; } }
            if t[b - 1].raw().ends_with(b"ing") && !cap[a] && t[b - 1].l >= 5 { return 7; }
            x.ty.get(&hs[a]).filter(|y| !(y.1 & 1 != 0 && !cap[a])).map_or(0, |y| y.0)
        };
        let txt = |a: usize, b: usize| -> Vb { let mut o: Vec<u8> = Vec::new(); for z in a..b { if z > a { o.push(32); } o.extend_from_slice(t[z].raw()); } vb(&o) };
        let near = |men: &Vec<Mn>, i: usize, j: usize| -> u32 {
            let ms: Vec<&Mn> = men.iter().filter(|m| !(m.a <= i && i < m.b) && m.kd != 3).collect();
            ms.iter().rev().find(|m| m.b <= i && cl[m.a] == cl[i]).or_else(|| ms.iter().find(|m| m.a >= j && cl[m.a] == cl[i])).or_else(|| ms.iter().rev().find(|m| m.b <= i)).map_or(0, |m| m.k)
        };
        let after_men = |men: &Vec<Mn>, j: usize| -> (u32, usize) {
            let mut q = j;
            if q < n && cov[q] == 3 { q += 1; }
            let mut m = men.iter().find(|m| m.a == q && m.kd != 3).copied();
            if m.is_none() { if let Some(z) = rnh.iter().find(|z| z.j0 == q && z.rel == 9) { return (pk(&(z.j0..z.j1).filter(|&y| !x.md.contains(&hs[y])).map(|y| hs[y]).collect::<Vec<_>>()), z.j1); } }
            if m.is_none() { m = men.iter().find(|m| m.a == q).copied(); while let Some(mm) = m { if mm.kd != 3 { break; } match men.iter().find(|z| z.a == mm.b) { Some(&m2) => m = Some(m2), None => break } } }
            m.map_or((0, q), |m| (m.k, m.b))
        };
        let (mut used, mut out) = (vec![false; n], Vec::<(u32, u32, Vb, u32)>::new());
        let put = |used: &mut Vec<bool>, out: &mut Vec<(u32, u32, Vb, u32)>, e: u32, r: u32, a: usize, b: usize, vt: u32| {
            if e == 0 || !(2..=15).contains(&r) { return; }
            if ([4, 6, 13].contains(&r) && !capx[a]) || (r == 14 && !num[a]) || (r == 15 && !(MON.contains(&hs[a]) || cap[a])) || (r == 9 && !(cap[a] || vt == 9)) { return; }
            (a..b).for_each(|z| used[z] = true);
            out.push((e, r, txt(a, b), vt));
        };
        let same: Vec<usize> = (0..n).filter(|&i| wd(i) && ([SAME, SHARE, SHARES, SHARED, SHARING].contains(&hs[i]) || (hs[i] == AS && i > 0 && hs[i - 1] == OLD))).collect();
        let scl: Vec<usize> = same.iter().map(|&i| cl[i]).collect();
        let like3 = [7, 10, 11];
        if !nq {
            for i in 0..n.saturating_sub(3) {
                if wd(i) && hs[i] == IT && x.fb(hs[i + 1], 5) {
                    if let Some((a, b)) = vspo(i + 2, 0) { if b + 2 < n && !wd(b) && neg[b + 1] { if let Some((a2, b2)) = vspo(b + 2, 0) { if let Some((e, r)) = mem.findval(vs(&txt(a2, b2))) { put(&mut used, &mut out, e, r, a, b, vtype(a, b)); } } } }
                }
            }
            if let Some((pe, pr)) = self.pend { if n >= 3 && [IT, THAT, HE, SHE, THEY].contains(&hs[0]) && x.fb(hs[1], 5) { if let Some((a, b)) = vspo(2, pr) { put(&mut used, &mut out, pe, pr, a, b, vtype(a, b)); } } }
        }
        self.pend = None;
        if !nq {
            let fr_all: Vec<Rh> = rnh.iter().copied().chain(free.iter().map(|&(a, b, r)| Rh { j0: a, j1: b, rel: r, own: 0, fav: false, st: a })).collect();
            for rh in fr_all {
                let Rh { j0, j1, rel, own, fav, st } = rh;
                let mut own = own;
                if scl.contains(&cl[j0]) || cues.iter().any(|c| j0 <= c.0 && c.0 < j1) { continue; }
                let (mut k, mut sk) = (j1, 0);
                while k < n && wd(k) && cov[k] == 0 && !cap[k] && (x.fb(hs[k], 5) || x.fb(hs[k], 6) || x.stop.contains(&hs[k]) || [IN, ON, AT, FALLS, COMES, LANDS, AROUND, EVERY, BEEN].contains(&hs[k])) && k - j1 < 5 { k += 1; sk += 1; }
                let mut sp = if sk > 0 && (st != j0 || (j1..k).any(|z| x.fb(hs[z], 5) || [HAS, HAVE, HAD, FALLS, COMES, LANDS].contains(&hs[z]))) { vspo(k, rel) } else { None };
                if sp.is_none() && k < n && !wd(k) && k + 1 < n && [IF, FOR, IN, AS, JUST, BY].contains(&hs[k + 1]) { let mut q = k + 1; while q < n && wd(q) && q - k < 7 { q += 1; } if q + 2 < n && hs[q + 1] == IS { sp = vspo(q + 2, rel); } }
                if sp.is_none() && (k >= n || !wd(k)) && k + 2 < n && hs[k + 1] == IT && (x.fb(hs[k + 2], 5) || hs[k + 2] == WOULD) { sp = vspo(k + 3, rel); }
                let mut q = st as isize - 1;
                while q >= 0 && wd(q as usize) && [THE, BEST, A, AN].contains(&hs[q as usize]) { q -= 1; }
                if sp.is_none() && q >= 0 && x.fb(hs[q as usize], 5) { sp = vback(q as usize, &used); }
                if sp.is_some() && own == 0 && men.is_empty() && st >= 2 && hs[st - 1] == BEST { own = SELFK; }
                if sp.is_none() && rel != 0 && st >= 2 && (hs[st - 2] == IT || hs[st - 2] == THAT) && x.fb(hs[st - 1], 5) && own != 0 {
                    if let Some(tv) = self.tf.iter().rev().find(|z| z.0 == own) { if tv.1 != rel { out.push((own, rel, tv.2, tv.3)); } }
                    continue;
                }
                let Some((a, b)) = sp else { continue };
                if (a..b).any(|z| used[z]) { continue; }
                let vt = vtype(a, b);
                let mut r2 = if rel != 0 { rel } else { vt };
                if r2 == 0 && fav { r2 = mem.lrel(if own != 0 { own } else { near(&men, j0, j1) }, &like3); }
                if rel == 9 && (vt == 10 || vt == 14) { own = pk(&(j0..j1).filter(|&z| !x.md.contains(&hs[z])).map(|z| hs[z]).collect::<Vec<_>>()); r2 = vt; }
                if ifr[a] || hyp[j0] || (0..n).any(|z| neg[z] && cl[z] == cl[j0]) { continue; }
                let e = if own != 0 { own } else { near(&men, j0, j1) };
                put(&mut used, &mut out, e, r2, a, b, vt);
            }
            for (i0, i1, rs) in cues.iter() {
                let (i0, i1) = (*i0, *i1);
                if scl.contains(&cl[i0]) { continue; }
                let ng = (i0..i1).any(|z| neg[z]) || (0..i0).any(|z| neg[z] && cl[z] == cl[i0]);
                let rows: Vec<(u32, u32)> = { let a: Vec<(u32, u32)> = rs.iter().copied().filter(|y| y.1 & 16 == 0 && (y.1 & (4 | 128) != 0) == ng).collect(); if a.is_empty() { rs.iter().copied().filter(|y| y.1 & 16 == 0 && y.1 & (4 | 128) == 0).collect() } else { a } };
                let Some(&(mut rel, f)) = rows.first() else { continue };
                let mut e0 = 0;
                let sp = if f & 64 != 0 { vback(i0, &used) } else {
                    let (ee, j) = after_men(&men, i1);
                    e0 = ee;
                    let mut j2 = j;
                    while j2 < n && wd(j2) && (x.fb(hs[j2], 5) || x.stop.contains(&hs[j2])) && e0 != 0 { j2 += 1; }
                    let mut sp = match vspan(if e0 != 0 { j2 } else { j }, rel, true) { Sp::Cut => continue, Sp::No => None, Sp::S(a, b) => Some((a, b)) };
                    let mut jb = j;
                    while jb < n && wd(jb) && (x.skip.contains(&hs[jb]) || x.stop.contains(&hs[jb]) || x.md.contains(&hs[jb])) { jb += 1; }
                    if sp.is_none() && (jb >= n || !wd(jb)) {
                        let q = jb + 1;
                        if q + 1 < n && hs[q] == IT && (x.fb(hs[q + 1], 5) || hs[q + 1] == WOULD) { sp = vspo(q + 2, rel); } else if q + 1 < n && hs[q] == THAT && x.fb(hs[q + 1], 5) { sp = vspo(q + 2, rel); } else if f & 1 == 0 && (j >= n || !wd(j)) { sp = vback(i0, &used); }
                    }
                    if sp.is_none() && rel == 8 && (j >= n || !wd(j)) {
                        let e = if e0 != 0 { e0 } else { near(&men, i0, i1) };
                        let tv = self.tf.iter().rev().find(|z| z.0 == e && z.1 == 2).map(|z| (z.2, z.3)).or(self.lastv.map(|lv| (lv, 0)));
                        if e != 0 { if let Some(tv) = tv { out.push((e, 8, tv.0, tv.1)); } }
                        continue;
                    }
                    sp
                };
                if i1 < n && wd(i1) && (hs[i1] == THERE || hs[i1] == HERE) && self.lastp.1 > 0 && rel == 4 {
                    let e = if e0 != 0 { e0 } else { near(&men, i0, i1) };
                    if e != 0 { out.push((e, 4, self.lastp, 4)); }
                    used[i1] = true;
                    continue;
                }
                let Some((a, b)) = sp else { continue };
                if (a..b).any(|z| used[z]) { continue; }
                let vt = vtype(a, b);
                if f & 2 != 0 && !(vt == rel || (rel == 14 && num[a]) || (rel == 15 && MON.contains(&hs[a])) || (rel == 12 && (vt == 12 || vt == 13)) || (rel == 9 && cap[a] && a > 0 && (hs[a - 1] == A || hs[a - 1] == AN) && !MON.contains(&hs[a]))) { continue; }
                if f & 2 != 0 && rel == 12 && vt == 13 { rel = 13; }
                if f & 8 != 0 { let fr: Vec<u32> = free.iter().map(|z| z.2).filter(|z| *z == 10 || *z == 11).collect(); rel = if let Some(&z) = fr.first() { z } else if vt == 10 { 10 } else if vt == 11 { 11 } else if vt == 7 { 7 } else { rel }; }
                if f & 32 != 0 && alg { rel = 8; }
                if rel == 4 && vt == 15 { continue; }
                if rel == 15 && vt != 15 && vt != 0 && cap[a] { rel = 4; }
                if rel == 6 && b < n && wd(b) && hs[b] == HOME { rel = 4; }
                if ng && f & (4 | 128) == 0 { if like3.contains(&rel) { rel = 2; } else { continue; } }
                if (ifr[a] && ifr[i0]) || hyp[i0] { continue; }
                let e = if e0 != 0 { e0 } else { near(&men, i0, i1) };
                put(&mut used, &mut out, e, rel, a, b, vt);
            }
            let mut cls: Vec<usize> = cl.clone();
            cls.dedup();
            for c0 in cls {
                if scl.contains(&c0) || (0..n).any(|z| cl[z] == c0 && used[z]) { continue; }
                let idx: Vec<usize> = (0..n).filter(|&z| cl[z] == c0).collect();
                for m in men.iter().filter(|m| cl[m.a] == c0 && m.kd != 3).copied().collect::<Vec<_>>() {
                    let mut k = m.b;
                    while k < n && wd(k) && x.fb(hs[k], 6) && k - m.b < 2 { k += 1; }
                    if k < n && wd(k) && x.fb(hs[k], 5) && !idx.iter().any(|&z| neg[z]) && !idx.iter().any(|&z| hyp[z] || ifr[z]) {
                        let art = k + 1 < n && (hs[k + 1] == A || hs[k + 1] == AN);
                        if let Some((a, b)) = vspo(k + 1, 0) {
                            let vt = vtype(a, b);
                            let r = if [10, 14, 15, 5].contains(&vt) { vt } else if art && !cap[a] && (vt == 0 || vt == 7) { 5 } else if vt == 0 && cap[a] && b - a <= 2 && !art { 6 } else { 0 };
                            let r = if m.kd != 4 || [10, 14, 15].contains(&r) { r } else { 0 };
                            if r != 0 { put(&mut used, &mut out, m.k, r, a, b, vt); }
                        }
                        break;
                    }
                }
            }
            for p0 in 0..n {
                if !(capx[p0] && cov[p0] == 0 && (p0 == 0 || !wd(p0 - 1) || hs[p0 - 1] == AND)) { continue; }
                let mut k = p0;
                while k < n && wd(k) && (capx[k] || (x.join.contains(&hs[k]) && k + 1 < n && wd(k + 1) && cap[k + 1] && k > p0)) && k - p0 < 4 && !(k > p0 && t[k - 1].pos) { k += 1; }
                let imp = t[k - 1].pos && k < n && wd(k) && [IN, A, PART, LOCATED, OVER].contains(&hs[k]);
                if !(k < n && wd(k) && (x.fb(hs[k], 5) || imp)) { continue; }
                let mut q = if imp { k } else { k + 1 };
                while q < n && wd(q) && [AN, A, CITY, TOWN, VILLAGE, LOCATED, OVER, PART, SITUATED, RIGHT, JUST, UP, DOWN, OUT, ACTUALLY, ALSO].contains(&hs[q]) { q += 1; }
                if !(q < n && (hs[q] == IN || hs[q] == OF)) { continue; }
                q += 1;
                while q < n && wd(q) && [THE, STATE, COUNTRY, REGION, PROVINCE, PREFECTURE, COUNTY, OF].contains(&hs[q]) { q += 1; }
                if q < n && wd(q) && cap[q] { if let Some((a, b)) = vspo(q, 4) { let s = txt(p0, k); let sv = vs(&s); mem.put(fold4(if imp { &sv[..sv.len().saturating_sub(2)] } else { sv }), ISINK, vs(&txt(a, b)), 4); } }
            }
            for (e, r, v, vt) in out.iter() { mem.put(*e, *r, vs(v), *vt); self.fire(*e, *r, vs(v), mem, 0); self.tf.push((*e, *r, *v, *vt)); if self.tf.len() > 16 { self.tf.remove(0); } (self.qe, self.qr) = (*e, *r); }
            if let Some(&l) = (0..n).filter(|&z| content(z) && !used[z]).collect::<Vec<_>>().last() { let mut q = l; while q > 0 && content(q - 1) && !used[q - 1] && l - q < 3 { q -= 1; } self.lastv = Some(txt(q, l + 1)); }
            for i in 0..n.saturating_sub(2) {
                if t[i].bd && wd(i + 1) {
                    let Some(pv) = out.iter().find(|z| z.1 == 4).map(|z| z.2) else { continue };
                    let mut q = i + 1;
                    if [A, WHICH, IN].contains(&hs[q]) {
                        if hs[q] == IN && q + 1 < n && cap[q + 1] { if let Some((a, b)) = vspo(q + 1, 4) { mem.put(fold4(vs(&pv)), ISINK, vs(&txt(a, b)), 4); } continue; }
                        while q < n && wd(q) && [A, CITY, TOWN, WHICH, IS, LOCATED, PART].contains(&hs[q]) { q += 1; }
                        if q < n && (hs[q] == IN || hs[q] == OF) && q + 1 < n && cap[q + 1] { if let Some((a, b)) = vspo(q + 1, 4) { mem.put(fold4(vs(&pv)), ISINK, vs(&txt(a, b)), 4); } }
                    }
                }
            }
            let mut sc: Vec<usize> = scl.clone();
            sc.sort();
            sc.dedup();
            for c0 in sc {
                let ms: Vec<Mn> = men.iter().filter(|m| cl[m.a] == c0 && m.kd != 3).copied().collect();
                let sm = *same.iter().find(|&&z| cl[z] == c0).unwrap();
                if ms.is_empty() { continue; }
                let rr: Vec<u32> = rnh.iter().map(|z| (z.j0, z.rel)).chain(free.iter().map(|z| (z.0, z.2))).filter(|z| cl[z.0] == c0 && z.1 != 0).map(|z| z.1).collect();
                let cr: Vec<u32> = cues.iter().filter(|c| cl[c.0] == c0 && c.2[0].1 & (16 | 2) == 0).map(|c| c.2[0].0).collect();
                let rel = if hs[sm] == AS { 14 } else if let Some(&z) = rr.first() { z } else if let Some(&z) = cr.first() { z } else { 0 };
                let mut src: Vec<u32> = ms.iter().filter(|m| m.a > sm).map(|m| m.k).collect();
                let mut tg: Vec<u32> = ms.iter().filter(|m| m.a < sm).map(|m| m.k).collect();
                let own: Vec<u32> = rnh.iter().filter(|z| cl[z.j0] == c0 && z.own != 0).map(|z| z.own).collect();
                if src.is_empty() && !own.is_empty() && !tg.is_empty() && *own.last().unwrap() != tg[0] { src = vec![*own.last().unwrap()]; }
                if tg.is_empty() && !own.is_empty() && !src.is_empty() { tg = vec![own[0]]; }
                if tg.is_empty() || src.is_empty() || rel == 0 { continue; }
                let mut pairs = vec![(tg[0], *src.last().unwrap())];
                if tg.len() >= 2 && tg[1] != tg[0] { pairs = vec![(tg[0], tg[1]), (tg[1], tg[0])]; } else if [SHARE, SHARES, SHARED, SHARING].contains(&hs[sm]) && hs.contains(&WITH) { pairs.push((*src.last().unwrap(), tg[0])); }
                for (ta, sa) in pairs {
                    if ta == sa { continue; }
                    self.links.push((ta, rel, sa));
                    if self.links.len() > 8 { self.links.remove(0); }
                    let v = mem.latest(sa, rel).or_else(|| if like3.contains(&rel) { [7, 11, 10].iter().filter(|&&z| z != rel).find_map(|&z| mem.latest(sa, z)) } else { None });
                    if let Some((v, vsm)) = v { mem.put(ta, rel, &v, vsm); self.fire(ta, rel, &v, mem, 0); }
                }
            }
        } else {
            let mut owners: Vec<(usize, u32)> = men.iter().filter(|m| m.kd != 3).map(|m| (m.a, m.k)).chain(rnh.iter().filter(|z| z.own != 0).map(|z| (z.j0, z.own))).collect();
            owners.sort_by_key(|z| z.0);
            let mut e = owners.iter().rev().find(|z| z.1 != SELFK).map_or(if !owners.is_empty() || mself { SELFK } else { 0 }, |z| z.1);
            let mut rnq: Vec<Rh> = rnh.clone();
            if let Some(v9) = rnh.iter().rev().find(|z| z.rel == 9) { if cues.iter().any(|c| c.2.first().is_some_and(|y| y.0 != 9 && y.1 & 2 == 0)) || free.iter().any(|z| z.2 != 9) { e = pk(&(v9.j0..v9.j1).filter(|&z| !x.md.contains(&hs[z])).map(|z| hs[z]).collect::<Vec<_>>()); rnq.retain(|z| z.rel != 9); } }
            let r1: Vec<u32> = rnq.iter().filter(|z| z.own == e && z.rel != 0).map(|z| z.rel).collect();
            let ng = neg.iter().any(|&z| z);
            let ep = owners.iter().filter(|z| z.1 == e).map(|z| z.0 as isize).max().unwrap_or(-1);
            let mut cs: Vec<&(usize, usize, Vec<(u32, u32)>)> = cues.iter().collect();
            cs.sort_by_key(|c| ((c.0 as isize) < ep, c.0));
            let mut cr: Vec<u32> = Vec::new();
            for c in cs {
                let a: Vec<(u32, u32)> = c.2.iter().copied().filter(|y| y.1 & 2 == 0 && (y.1 & (4 | 128) != 0) == ng).collect();
                let rows = if a.is_empty() { c.2.iter().copied().filter(|y| y.1 & 2 == 0 && y.1 & (4 | 128) == 0).collect() } else { a };
                if let Some(&(r, f)) = rows.first() {
                    let fr: Vec<u32> = free.iter().map(|z| z.2).filter(|z| *z == 10 || *z == 11).collect();
                    let r = if f & 8 != 0 && !fr.is_empty() && (r == 7 || r == 11) { fr[0] } else { r };
                    let r = if ng && (r == 7 || r == 11) && f & (4 | 128) == 0 { 2 } else { r };
                    cr.push(if f & 32 != 0 && (alg || (mem.latest(e, 8).is_some() && mem.latest(e, 2).is_none())) { 8 } else { r });
                }
            }
            let fr: Vec<u32> = free.iter().map(|z| z.2).collect();
            let wh: Vec<u32> = hs.iter().copied().filter(|&h| x.fb(h, 9)).collect();
            let mut r4 = if wh.contains(&WHERE) { 4 } else if wh.contains(&WHEN) { 15 } else if (wh.contains(&HOW) && hs.contains(&OLD)) || hs.contains(&AGE) { 14 } else if wh.contains(&WHO) { 6 } else { 0 };
            if r4 == 0 && r1.is_empty() && cr.is_empty() && fr.is_empty() && hs[n.saturating_sub(2)..].iter().any(|h| [DO, DOES, DID, DOING].contains(h)) && wh.contains(&WHAT) { r4 = 5; }
            let mut rel = r1.last().copied().or(cr.first().copied()).or(fr.first().copied()).unwrap_or(r4);
            if wh.contains(&WHERE) && [12, 0, 13].contains(&rel) { rel = if rel == 12 || rel == 13 || hs.iter().any(|h| [STUDY, STUDIED, DEGREE, SCHOOL, COLLEGE, UNIVERSITY, ENROLLED].contains(h)) { 13 } else { 4 }; }
            if let Some(&z) = r1.last() { if e != 0 { self.pend = Some((e, z)); } }
            (self.qe, self.qr) = (e, rel);
        }
        let lp: Vec<usize> = (1..n).filter(|&z| capx[z] && cov[z] == 0 && wd(z - 1) && [IN, TO, AT, FROM].contains(&hs[z - 1])).collect();
        if let Some(&l) = lp.last() { let mut q = l; while q < n && wd(q) && (capx[q] || (x.join.contains(&hs[q]) && q + 1 < n && wd(q + 1) && cap[q + 1])) && q - l < 4 { q += 1; } self.lastp = txt(l, q); }
        if let Some(m) = men.iter().rev().find(|m| m.k != SELFK && m.kd != 3) { self.ante = m.k; }
    }
}
