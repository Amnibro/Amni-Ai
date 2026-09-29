use crate::{bpe::Bpe, sm, Cfg};
use std::io;
pub const FRUIT: &[&str] = &["apple", "pear", "plum", "fig", "lime", "lemon", "mango", "peach", "cherry", "grape", "melon", "kiwi", "apricot", "guava", "papaya", "quince", "date", "berry", "orange", "banana"];
pub const COLOR: &[&str] = &["red", "blue", "green", "yellow", "purple", "black", "white", "gray", "brown", "pink", "violet", "teal", "amber", "silver", "golden", "maroon", "navy", "cream", "olive", "coral"];
pub const ANIMAL: &[&str] = &["otter", "badger", "heron", "falcon", "beaver", "moose", "lynx", "walrus", "gecko", "hamster", "pelican", "raven", "salmon", "tiger", "zebra", "koala", "panda", "camel", "donkey", "rabbit"];
pub const TREE: &[&str] = &["oak", "maple", "birch", "cedar", "willow", "pine", "elm", "ash", "spruce", "poplar", "alder", "beech", "cypress", "hazel", "juniper", "larch", "linden", "aspen", "hemlock", "redwood"];
pub const FLOWER: &[&str] = &["tulip", "daisy", "lily", "orchid", "poppy", "iris", "rose", "peony", "lotus", "aster", "dahlia", "jasmine", "lilac", "marigold", "clover", "heather", "zinnia", "crocus", "primrose", "bluebell"];
pub const NOUN: &[&str] = &["kettle", "lantern", "ladder", "compass", "blanket", "bucket", "mirror", "pillow", "basket", "helmet", "whistle", "hammer", "candle", "anchor", "saddle", "teapot", "trumpet", "notebook", "umbrella", "wagon"];
pub const ADJ: &[&str] = &["quiet", "bright", "rusty", "tiny", "heavy", "narrow", "wooden", "broken", "velvet", "crooked", "dusty", "shiny", "hollow", "striped", "frozen", "gentle", "noisy", "smooth", "patched", "sturdy"];
pub const SYL: &[&str] = &["bal", "ter", "mun", "sop", "rid", "kev", "lan", "dor", "fim", "gat", "hul", "jev", "nos", "pav", "quil", "rost", "tem", "vel", "wob", "yar", "zil", "cav", "ebb", "ond", "bri", "tuv", "mek", "sarn", "plo", "gwen"];
pub const SUR: &[&str] = &["son", "ley", "ard", "ette", "ov", "ez", "berg", "wick", "ridge", "more"];
pub const PLACE: &[&str] = &["Valley", "Harbor", "Works", "Institute", "Falls", "County", "Bay", "Ridge", "Crossing", "Point"];
pub const ATTR: &[(&str, u8)] = &[("favorite fruit", 0), ("favorite color", 1), ("lucky number", 7), ("locker code", 8), ("shoe size", 9), ("favorite animal", 2), ("garden tree", 3), ("favorite flower", 4), ("secret word", 10), ("room number", 11), ("bike color", 1), ("train platform", 12), ("middle name", 13), ("travel bag", 14), ("tea flavor", 0), ("desk plant", 4)];
pub const PROP: &[(&str, u8)] = &[("mascot", 2), ("founding year", 15), ("population", 16), ("official color", 1), ("area code", 11), ("motto", 10), ("main export", 0), ("tallest tree", 3), ("river", 17), ("landmark", 14)];
pub const FACT: &[&str] = &["{E}'s {A} is {V}.", "The {A} of {E} is {V}.", "For {E}, the {A} is {V}.", "{E} picked {V} as the {A}.", "{E} -- {A}: {V}."];
pub const ASK: &[&str] = &["Question: What is {E}'s {A}? Answer: {V}.", "What is the {A} of {E}? It is {V}.", "{E}'s {A} is {V}.", "Q: Which {A} does {E} have? A: {V}.", "Asked about {E}, the {A} is {V}."];
pub const INTRO: &[&str] = &["", "Some facts:", "Notes from the meeting.", "Here is what we know.", "Record:", "From the notebook:"];
pub const KVH: &[&str] = &["", "Keys and values:", "Lookup table:", "Code list:", "Inventory:", "Settings:"];
pub const KVS: &[&str] = &[": ", " = ", " -> ", " is "];
pub const KVJ: &[&str] = &["; ", ", ", "\n", " | "];
pub const KVQ: &[&str] = &["Lookup {K}: {V}.", "The value for {K} is {V}.", "{K}{S}{V}", "Question: What is the value of {K}? Answer: {V}."];
pub const WLQ: &[&str] = &["Question: What is the {P} of {L}? Answer: {V}.", "The {P} of {L} is {V}.", "{L}, {P}: {V}."];
pub const RPS: &[&str] = &["\nAgain: ", " | ", "\nRepeat: ", "\n", " -- ", "\nOnce more: "];
pub const RPH: &[&str] = &["", "List: ", "Sequence: ", "Words: "];
pub struct R(pub u64);
impl R {
    pub fn u(&mut self) -> u64 { self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15); sm(self.0) }
    pub fn n(&mut self, k: usize) -> usize { (self.u() % k.max(1) as u64) as usize }
    pub fn f(&mut self) -> f64 { (self.u() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn p<'a>(&mut self, v: &[&'a str]) -> &'a str { v[self.n(v.len())] }
    pub fn r(&mut self, a: usize, b: usize) -> usize { a + self.n(b + 1 - a) }
    pub fn w(&mut self, a: usize, b: usize) -> String { let k = self.r(a, b); (0..k).map(|_| self.p(SYL)).collect() }
    pub fn perm(&mut self, n: usize) -> Vec<usize> { let mut v: Vec<usize> = (0..n).collect(); (1..n).rev().for_each(|i| { let j = self.n(i + 1); v.swap(i, j); }); v }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Q { pub kind: u8, pub pos: usize, pub n: usize, pub key: String, pub val: String, pub fact: String }
pub struct Gen { pub bpe: Option<Bpe>, pub mix: f64, pub tw: Vec<u32> }
fn cap(s: &str) -> String { let mut c = s.chars(); c.next().map_or(String::new(), |f| f.to_uppercase().chain(c).collect()) }
fn fill(t: &str, kv: &[(&str, &str)]) -> String { kv.iter().fold(t.to_string(), |s, (k, v)| s.replace(k, v)) }
impl Gen {
    pub fn new(c: &Cfg) -> io::Result<Gen> { let bpe = c.bpe()?; let tw = bpe.as_ref().map(|b| (0..b.v() as u32).filter(|&i| { let w = &b.voc[i as usize]; let s = w.strip_prefix(b" ").unwrap_or(w); s.len() >= 2 && s.iter().all(|c| c.is_ascii_alphabetic()) }).collect()).unwrap_or_default(); Ok(Gen { bpe, mix: c.rmix, tw }) }
    pub fn hit(&self, seed: u64, it: usize, j: usize) -> bool { self.mix > 0.0 && (((sm(seed ^ 0xB1D5_0000_0000_0000 ^ (it as u64) << 20 ^ j as u64) >> 11) as f64) / (1u64 << 53) as f64) < self.mix }
    pub fn name(r: &mut R) -> String { let f = cap(&r.w(2, 3)); if r.f() < 0.6 { format!("{f} {}", cap(&(r.w(1, 2) + r.p(SUR)))) } else { f } }
    pub fn val(r: &mut R, k: u8) -> String {
        match k { 0 => r.p(FRUIT).into(), 1 => r.p(COLOR).into(), 2 => r.p(ANIMAL).into(), 3 => r.p(TREE).into(), 4 => r.p(FLOWER).into(), 7 => r.r(1, 99).to_string(), 8 => r.r(100, 9999).to_string(), 9 => r.r(4, 15).to_string(), 10 => r.w(2, 3), 11 => r.r(100, 999).to_string(), 12 => r.r(1, 20).to_string(), 13 => cap(&r.w(2, 3)), 14 => format!("the {} {} {}", r.p(ADJ), r.p(COLOR), r.p(NOUN)), 15 => r.r(1500, 2020).to_string(), 16 => { let v = r.r(1200, 9_900_000); let s = v.to_string(); s.as_bytes().rchunks(3).rev().map(|c| std::str::from_utf8(c).unwrap_or("")).collect::<Vec<_>>().join(",") } _ => format!("the {} River", cap(&r.w(2, 3))) }
    }
    pub fn uniq(r: &mut R, n: usize, f: &mut dyn FnMut(&mut R) -> String) -> Vec<String> { let mut o: Vec<String> = Vec::new(); let mut g = 0; while o.len() < n && g < 64 * n { let v = f(r); g += 1; if !o.contains(&v) { o.push(v); } } o }
    pub fn ep(&self, r: &mut R) -> (String, Vec<Q>) {
        let u = r.f();
        match (u < 0.35, u < 0.6, u < 0.75, u < 0.87 || self.tw.is_empty()) {
            (true, _, _, _) => {
                let (n, two) = (r.r(2, 7), r.f() < 0.3);
                let at: Vec<(&str, u8)> = r.perm(ATTR.len()).into_iter().take(1 + two as usize).map(|i| ATTR[i]).collect();
                let es = Gen::uniq(r, n, &mut |r| Gen::name(r));
                let vs: Vec<Vec<String>> = at.iter().map(|&(_, k)| Gen::uniq(r, es.len(), &mut |r| Gen::val(r, k))).collect();
                let fs: Vec<(usize, usize)> = (0..es.len()).flat_map(|e| (0..at.len()).map(move |a| (e, a))).filter(|&(e, a)| e < vs[a].len()).collect();
                let (ord, ft, j) = (r.perm(fs.len()), r.p(FACT), [" ", "\n"][r.n(2)]);
                let body: Vec<String> = ord.iter().map(|&i| { let (e, a) = fs[i]; fill(ft, &[("{E}", &es[e]), ("{A}", at[a].0), ("{V}", &vs[a][e])]) }).collect();
                let qs: Vec<Q> = r.perm(fs.len()).into_iter().take(r.r(1, 3).min(fs.len())).map(|i| { let (e, a) = fs[i]; let p = ord.iter().position(|&o| o == i).unwrap_or(0); Q { kind: 0, pos: p, n: fs.len(), key: format!("{} {}", es[e], at[a].0), val: vs[a][e].clone(), fact: body[p].clone() } }).collect();
                let at_: Vec<String> = qs.iter().map(|q| { let (e, a) = fs[ord[q.pos]]; fill(r.p(ASK), &[("{E}", &es[e]), ("{A}", at[a].0), ("{V}", &q.val)]) }).collect();
                (format!("{}{}{}{}{}", r.p(INTRO), if r.n(2) == 0 { " " } else { "\n" }, body.join(j), ["\n", " "][r.n(2)], at_.join(j)).trim_start().to_string(), qs)
            }
            (_, true, _, _) => {
                let (n, s, jn, kk, vk) = (r.r(2, 12), r.p(KVS), r.p(KVJ), r.n(4), [8u8, 1, 10, 14, 7][r.n(5)]);
                let ks = Gen::uniq(r, n, &mut |r| match kk { 0 => r.w(1, 3), 1 => r.p(NOUN).to_string(), 2 => format!("item {}", r.r(1, 99)), _ => format!("{} {}", r.p(ADJ), r.p(NOUN)) });
                let vs = Gen::uniq(r, ks.len(), &mut |r| Gen::val(r, vk));
                let m = ks.len().min(vs.len());
                let qs: Vec<Q> = r.perm(m).into_iter().take(r.r(1, 3).min(m)).map(|i| Q { kind: 1, pos: i, n: m, key: ks[i].clone(), val: vs[i].clone(), fact: format!("{}{s}{}", ks[i], vs[i]) }).collect();
                let qt: Vec<String> = qs.iter().map(|q| fill(r.p(KVQ), &[("{K}", &q.key), ("{S}", s), ("{V}", &q.val)])).collect();
                let h = r.p(KVH);
                (format!("{h}{}{}\n{}", if h.is_empty() { "" } else { "\n" }, (0..m).map(|i| format!("{}{s}{}", ks[i], vs[i])).collect::<Vec<_>>().join(jn), qt.join(" ")), qs)
            }
            (_, _, true, _) => {
                let (n, (p, k)) = (r.r(2, 7), PROP[r.n(PROP.len())]);
                let ls = Gen::uniq(r, n, &mut |r| { let b = cap(&r.w(2, 3)); if r.f() < 0.5 { format!("{b} {}", r.p(PLACE)) } else { b } });
                let vs = Gen::uniq(r, ls.len(), &mut |r| Gen::val(r, k));
                let m = ls.len().min(vs.len());
                let qs: Vec<Q> = r.perm(m).into_iter().take(r.r(1, 2).min(m)).map(|i| Q { kind: 2, pos: i, n: m, key: ls[i].clone(), val: vs[i].clone(), fact: format!("{}, {p}: {}.", ls[i], vs[i]) }).collect();
                let qt: Vec<String> = qs.iter().map(|q| fill(r.p(WLQ), &[("{P}", p), ("{L}", &q.key), ("{V}", &q.val)])).collect();
                (format!("{}\n{}", (0..m).map(|i| format!("{}, {p}: {}.", ls[i], vs[i])).collect::<Vec<_>>().join("\n"), qt.join("\n")), qs)
            }
            (_, _, _, w) => {
                let (l, sp, rep) = (r.r(4, 48), r.p(RPS), r.r(1, 2));
                let s: String = if w { let pool: Vec<String> = (0..l).map(|_| match r.n(8) { 0 => r.p(FRUIT).into(), 1 => r.p(COLOR).into(), 2 => r.p(ANIMAL).into(), 3 => r.p(TREE).into(), 4 => r.p(FLOWER).into(), 5 => r.p(NOUN).into(), 6 => r.p(ADJ).into(), _ => r.w(1, 3) }).collect(); pool.join([" ", ", "][r.n(2)]) } else { let b = self.bpe.as_ref().map(|b| b.dec(&(0..l.min(40)).map(|_| self.tw[r.n(self.tw.len())]).collect::<Vec<u32>>())).unwrap_or_default(); String::from_utf8_lossy(&b).trim().to_string() };
                (format!("{}{s}{}", r.p(RPH), (0..rep).map(|_| format!("{sp}{s}")).collect::<String>()), vec![Q { kind: 3 + !w as u8, pos: 0, n: rep + 1, key: String::new(), val: s.clone(), fact: s.clone() }])
            }
        }
    }
    pub fn enc(&self, s: &str) -> Vec<u32> { self.bpe.as_ref().map_or_else(|| s.bytes().map(|b| b as u32).collect(), |b| b.enc(s.as_bytes())) }
    pub fn window(&self, seed: u64, it: usize, j: usize, n: usize) -> Vec<u32> {
        let mut r = R(sm(seed ^ 0x5EC0_11AB_0000_0000 ^ (it as u64) << 20 ^ j as u64));
        let mut o: Vec<u32> = Vec::with_capacity(n + 64);
        while o.len() < n { let (t, _) = self.ep(&mut r); o.extend(self.enc(&if o.is_empty() { t } else { format!("\n\n{t}") })); }
        o.truncate(n);
        o
    }
}
