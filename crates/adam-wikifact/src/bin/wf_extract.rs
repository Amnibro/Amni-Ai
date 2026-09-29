use adam_wikifact::{keep, ws, wv, Val, CAP, QCAP};
use crossbeam_channel::bounded;
use flate2::read::MultiGzDecoder;
use md5::{Digest, Md5};
use serde::de::{Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json::{value::RawValue, Value};
use std::borrow::Cow;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::marker::PhantomData;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
#[derive(Deserialize)]
struct Lv<'a> { #[serde(borrow)] value: Cow<'a, str> }
#[derive(Deserialize, Default)]
#[serde(default)]
struct Langs<'a> { #[serde(borrow)] en: Option<Lv<'a>>, #[serde(borrow)] mul: Option<Lv<'a>> }
#[derive(Deserialize, Default)]
#[serde(default)]
struct Alis<'a> { #[serde(borrow)] en: Option<Vec<Lv<'a>>>, #[serde(borrow)] mul: Option<Vec<Lv<'a>>> }
#[derive(Deserialize)]
struct St<'a> { #[serde(borrow)] title: Cow<'a, str> }
#[derive(Default)]
struct Sl { n: u16, t: Option<String> }
struct SlV;
impl<'de> Visitor<'de> for SlV {
    type Value = Sl;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("sitelinks") }
    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Sl, A::Error> { while a.next_element::<IgnoredAny>()?.is_some() {} Ok(Sl::default()) }
    fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<Sl, A::Error> { let mut s = Sl::default(); while let Some(k) = m.next_key::<Cow<str>>()? { s.n = s.n.saturating_add(1); if k == "enwiki" { s.t = Some(m.next_value::<St>()?.title.into_owned()) } else { m.next_value::<IgnoredAny>()?; } } Ok(s) }
}
impl<'de> Deserialize<'de> for Sl { fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Sl, D::Error> { d.deserialize_any(SlV) } }
struct Pm<T, const Q: bool>(Vec<(u16, Vec<T>)>);
struct PmV<T, const Q: bool>(PhantomData<T>);
impl<'de, T: Deserialize<'de>, const Q: bool> Visitor<'de> for PmV<T, Q> {
    type Value = Pm<T, Q>;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("property map") }
    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Pm<T, Q>, A::Error> { while a.next_element::<IgnoredAny>()?.is_some() {} Ok(Pm(vec![])) }
    fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<Pm<T, Q>, A::Error> { let mut o = vec![]; while let Some(k) = m.next_key::<Cow<str>>()? { let p = k.strip_prefix('P').and_then(|x| x.parse::<u16>().ok()); match p.filter(|&p| keep(p, Q)) { Some(p) => { let v = m.next_value::<Vec<T>>()?; (!Q).then(|| pc(p, v.len() as u64)); o.push((p, v)) } None => { let n = m.next_value::<Cn>()?.0; if let (false, Some(p)) = (Q, p) { pc(p, n) } } } } Ok(Pm(o)) }
}
thread_local! { static PC: std::cell::RefCell<Vec<(u64, u64)>> = std::cell::RefCell::new(vec![(0, 0); 1 << 16]) }
fn pc(p: u16, n: u64) { PC.with(|c| { let c = &mut c.borrow_mut()[p as usize]; c.0 += 1; c.1 += n }) }
struct Cn(u64);
struct CnV;
impl<'de> Visitor<'de> for CnV {
    type Value = Cn;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("statement list") }
    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Cn, A::Error> { let mut n = 0; while a.next_element::<IgnoredAny>()?.is_some() { n += 1 } Ok(Cn(n)) }
    fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<Cn, A::Error> { while m.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {} Ok(Cn(0)) }
}
impl<'de> Deserialize<'de> for Cn { fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Cn, D::Error> { d.deserialize_any(CnV) } }
impl<'de, T: Deserialize<'de>, const Q: bool> Deserialize<'de> for Pm<T, Q> { fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Pm<T, Q>, D::Error> { d.deserialize_any(PmV(PhantomData)) } }
#[derive(Deserialize)]
struct Dv { #[serde(rename = "type")] ty: String, value: Value }
#[derive(Deserialize)]
struct Snak { datavalue: Option<Dv> }
#[derive(Deserialize)]
struct Stmt { mainsnak: Snak, rank: String, qualifiers: Option<Pm<Snak, true>> }
#[derive(Deserialize)]
struct Ent<'a> { #[serde(borrow)] id: Cow<'a, str>, #[serde(default, borrow)] labels: Langs<'a>, #[serde(default, borrow)] aliases: Alis<'a>, #[serde(default, borrow)] claims: Option<&'a RawValue>, #[serde(default)] sitelinks: Sl }
fn qn(s: &str) -> Option<u32> { s.rsplit('/').next()?.strip_prefix('Q')?.parse().ok() }
fn val(s: &Snak) -> Option<Val> {
    let d = s.datavalue.as_ref()?;
    let v = &d.value;
    match d.ty.as_str() {
        "wikibase-entityid" => qn(v.get("id")?.as_str()?).map(Val::I),
        "time" => { let t = v.get("time")?.as_str()?; let (neg, t) = (t.starts_with('-'), t.trim_start_matches(['+', '-'])); let mut it = t.split(['-', 'T']); let y: i64 = it.next()?.parse().ok()?; let m: u8 = it.next()?.parse().ok()?; let dd: u8 = it.next()?.parse().ok()?; Some(Val::T(if neg { -y } else { y }, m, dd, v.get("precision")?.as_u64()? as u8, v.get("calendarmodel").and_then(|c| c.as_str()).is_some_and(|c| c.ends_with("Q1985786")))) }
        "quantity" => Some(Val::Q(v.get("amount")?.as_str()?.parse().ok()?, v.get("unit").and_then(|u| u.as_str()).and_then(qn).unwrap_or(0))),
        "string" => Some(Val::S(v.as_str()?.into())),
        "monolingualtext" => matches!(v.get("language")?.as_str()?, "en" | "mul" | "en-us" | "en-gb").then(|| Some(Val::M(v.get("text")?.as_str()?.into())))?,
        _ => None,
    }
}
struct Pf { urls: Vec<String>, total: u64, cs: u64, la: u64, next: AtomicU64, done: Mutex<(u64, std::collections::HashMap<u64, Vec<u8>>)>, cv: std::sync::Condvar, rate: String }
impl Pf {
    fn get(&self, u: &str, a: u64, b: u64) -> Vec<u8> {
        (1..=200u32).find_map(|t| { let o = Command::new("curl").args(["-sS", "-f", "-L", "--connect-timeout", "30", "--speed-limit", "50000", "--speed-time", "60", "--max-time", "900", "--limit-rate", &self.rate, "-A", "Amni-wikifact/0.2 (research; resumable range reads)", "-r", &format!("{a}-{b}"), u]).output().ok(); match o { Some(o) if o.status.success() && o.stdout.len() as u64 == b - a + 1 => Some(o.stdout), o => { eprintln!("chunk {a}-{b} from {u} try {t}: {:?}", o.map(|o| (o.status, o.stdout.len()))); std::thread::sleep(Duration::from_secs((3u64 << t.min(5)).min(90))); None } } }).expect("chunk fetch failed 200 times")
    }
    fn run(self: Arc<Self>, i: usize) {
        loop {
            let k = self.next.fetch_add(1, Relaxed);
            if k * self.cs >= self.total { return }
            { let mut g = self.done.lock().unwrap(); while k >= g.0 + self.la { g = self.cv.wait(g).unwrap() } }
            let v = self.get(&self.urls[i % self.urls.len()], k * self.cs, ((k + 1) * self.cs).min(self.total) - 1);
            self.done.lock().unwrap().1.insert(k, v);
            self.cv.notify_all();
        }
    }
}
struct Rs { pf: Arc<Pf>, cur: Vec<u8>, off: usize, k: u64, md: Arc<Mutex<Md5>>, prog: Arc<AtomicU64> }
impl Read for Rs {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        if self.off >= self.cur.len() {
            if self.k * self.pf.cs >= self.pf.total { return Ok(0) }
            let mut g = self.pf.done.lock().unwrap();
            let v = loop { match g.1.remove(&self.k) { Some(v) => break v, None => g = self.pf.cv.wait(g).unwrap() } };
            g.0 = self.k + 1;
            drop(g);
            self.pf.cv.notify_all();
            self.md.lock().unwrap().update(&v);
            self.prog.fetch_add(v.len() as u64, Relaxed);
            (self.cur, self.off, self.k) = (v, 0, self.k + 1);
        }
        let n = b.len().min(self.cur.len() - self.off);
        b[..n].copy_from_slice(&self.cur[self.off..self.off + n]);
        self.off += n;
        Ok(n)
    }
}
fn work(b: &[u8], sub: &mut Vec<u8>, lab: &mut Vec<u8>, st: &[AtomicU64; 6]) {
    for l in b.split(|&c| c == b'\n') {
        let l = l.trim_ascii_end();
        let l = l.strip_suffix(b",").unwrap_or(l);
        if l.len() < 3 { continue }
        let Ok(s) = std::str::from_utf8(l) else { st[4].fetch_add(1, Relaxed); continue };
        let e: Ent = match serde_json::from_str(s) { Ok(e) => e, Err(_) => { st[4].fetch_add(1, Relaxed); continue } };
        let Some(q) = e.id.strip_prefix('Q').and_then(|x| x.parse::<u32>().ok()) else { continue };
        st[0].fetch_add(1, Relaxed);
        let lb = e.labels.en.or(e.labels.mul).map(|x| x.value).unwrap_or_default();
        let al: Vec<Cow<str>> = e.aliases.en.into_iter().chain(e.aliases.mul).flatten().map(|x| x.value).filter(|a| a != &lb).take(32).collect();
        if !lb.is_empty() || !al.is_empty() { lab.extend_from_slice(&q.to_le_bytes()); ws(lab, &lb); lab.push(al.len() as u8); al.iter().for_each(|a| ws(lab, a)); st[5].fetch_add(1, Relaxed); }
        let Some(t) = e.sitelinks.t else { continue };
        let cl: Pm<Stmt, false> = e.claims.and_then(|r| serde_json::from_str(r.get()).ok()).unwrap_or(Pm(vec![]));
        st[1].fetch_add(1, Relaxed);
        sub.extend_from_slice(&q.to_le_bytes());
        sub.extend_from_slice(&e.sitelinks.n.to_le_bytes());
        ws(sub, &t);
        ws(sub, &lb);
        sub.push(al.len() as u8);
        al.iter().for_each(|a| ws(sub, a));
        let fs: Vec<(u16, u8, Val, Vec<(u16, Val)>)> = cl.0.iter().flat_map(|(p, v)| v.iter().filter(|s| s.rank != "deprecated").filter_map(|s| val(&s.mainsnak).map(|x| (*p, (s.rank == "preferred") as u8, x, s.qualifiers.as_ref().map(|q| q.0.iter().flat_map(|(qp, qs)| qs.iter().filter_map(|z| val(z).map(|y| (*qp, y)))).take(QCAP).collect()).unwrap_or_default()))).take(CAP)).collect();
        st[2].fetch_add(fs.len() as u64, Relaxed);
        sub.extend_from_slice(&(fs.len() as u32).to_le_bytes());
        for (p, r, v, qs) in &fs { sub.extend_from_slice(&p.to_le_bytes()); sub.push(*r); wv(sub, v); sub.push(qs.len() as u8); st[3].fetch_add(qs.len() as u64, Relaxed); qs.iter().for_each(|(qp, qv)| { sub.extend_from_slice(&qp.to_le_bytes()); wv(sub, qv) }) }
    }
}
fn main() -> io::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = |k: &str, d: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned().unwrap_or(d.into());
    let (url, out, total, nw, rate, file) = (g("--url", "").split(',').map(String::from).collect::<Vec<_>>(), g("--out", "."), g("--total", "0").parse::<u64>().unwrap(), g("--workers", "4").parse::<usize>().unwrap(), g("--rate", "90M"), g("--file", ""));
    let maxb = g("--max-bytes", "0").parse::<u64>().unwrap();
    std::fs::create_dir_all(&out)?;
    let md = Arc::new(Mutex::new(Md5::new()));
    let prog = Arc::new(AtomicU64::new(0));
    let total = if maxb > 0 { maxb } else { total };
    let nc = g("--conns", "4").parse::<usize>().unwrap();
    let pf = Arc::new(Pf { urls: url, total, cs: g("--chunk-mb", "64").parse::<u64>().unwrap() << 20, la: 3 * nc as u64, next: AtomicU64::new(0), done: Mutex::new((0, Default::default())), cv: Default::default(), rate });
    let fts: Vec<_> = if file.is_empty() { (0..nc).map(|i| { let p = pf.clone(); std::thread::spawn(move || p.run(i)) }).collect() } else { vec![] };
    let src: Box<dyn Read + Send> = if file.is_empty() { Box::new(Rs { pf: pf.clone(), cur: vec![], off: 0, k: 0, md: md.clone(), prog: prog.clone() }) } else { Box::new(File::open(&file)?.take(if maxb > 0 { maxb } else { u64::MAX })) };
    let st: Arc<[AtomicU64; 6]> = Arc::new(Default::default());
    let (tx, rx) = bounded::<Vec<u8>>(2 * nw);
    let (ox, orx) = bounded::<(Vec<u8>, Vec<u8>)>(2 * nw);
    let rd = std::thread::spawn(move || -> io::Result<u64> {
        let mut r = BufReader::with_capacity(4 << 20, MultiGzDecoder::new(BufReader::with_capacity(4 << 20, src)));
        let mut n = 0u64;
        loop {
            let mut b = Vec::with_capacity(9 << 20);
            while b.len() < 8 << 20 { match r.read_until(b'\n', &mut b) { Ok(0) => break, Ok(_) => {} Err(e) => { eprintln!("decode end: {e}"); break } } }
            n += b.len() as u64;
            if b.is_empty() { break }
            tx.send(b).map_err(io::Error::other)?;
        }
        Ok(n)
    });
    let ws_: Vec<_> = (0..nw).map(|_| { let (rx, ox, st) = (rx.clone(), ox.clone(), st.clone()); std::thread::spawn(move || { for b in rx { let (mut s, mut l) = (Vec::with_capacity(2 << 20), Vec::with_capacity(4 << 20)); work(&b, &mut s, &mut l, &st); if ox.send((s, l)).is_err() { break } } PC.with(|c| c.borrow().clone()) }) }).collect();
    drop(ox);
    let (mut fs, mut fl) = (BufWriter::with_capacity(8 << 20, File::create(format!("{out}/sub.bin"))?), BufWriter::with_capacity(8 << 20, File::create(format!("{out}/lab.bin"))?));
    let t0 = Instant::now();
    let mut last = Instant::now();
    for (s, l) in orx {
        fs.write_all(&s)?;
        fl.write_all(&l)?;
        if last.elapsed().as_secs() >= 60 { last = Instant::now(); let p = prog.load(Relaxed); eprintln!("{:.0}s items {} sub {} facts {} quals {} bad {} lab {} | gz {:.1}/{:.1} GB {:.1} MB/s", t0.elapsed().as_secs_f64(), st[0].load(Relaxed), st[1].load(Relaxed), st[2].load(Relaxed), st[3].load(Relaxed), st[4].load(Relaxed), st[5].load(Relaxed), p as f64 / 1e9, total as f64 / 1e9, p as f64 / 1e6 / t0.elapsed().as_secs_f64()) }
    }
    fs.flush()?;
    fl.flush()?;
    let raw = rd.join().unwrap()?;
    let pcs = ws_.into_iter().map(|w| w.join().unwrap()).fold(vec![(0u64, 0u64); 1 << 16], |mut t, v| { t.iter_mut().zip(v).for_each(|(a, b)| { a.0 += b.0; a.1 += b.1 }); t });
    std::fs::write(format!("{out}/prop_counts.json"), format!("{{{}}}", pcs.iter().enumerate().filter(|x| x.1 .0 > 0).map(|(p, c)| format!("\"P{p}\":[{},{}]", c.0, c.1)).collect::<Vec<_>>().join(",")))?;
    fts.into_iter().for_each(|w| w.join().unwrap());
    let h = md.lock().unwrap().clone().finalize();
    let js = format!("{{\"items\":{},\"subjects\":{},\"facts\":{},\"quals\":{},\"bad\":{},\"labelled\":{},\"json_bytes\":{},\"gz_bytes\":{},\"md5\":\"{}\",\"sec\":{:.0}}}", st[0].load(Relaxed), st[1].load(Relaxed), st[2].load(Relaxed), st[3].load(Relaxed), st[4].load(Relaxed), st[5].load(Relaxed), raw, prog.load(Relaxed), h.iter().map(|x| format!("{x:02x}")).collect::<String>(), t0.elapsed().as_secs_f64());
    std::fs::write(format!("{out}/extract.json"), &js)?;
    println!("{js}");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use adam_wikifact::Rd;
    #[test]
    fn parses_item_line() {
        let l = r#"{"type":"item","id":"Q42","labels":{"en":{"language":"en","value":"Douglas Adams"},"fr":{"language":"fr","value":"x"}},"descriptions":[],"aliases":{"en":[{"language":"en","value":"Douglas Noel Adams"}]},"claims":{"P569":[{"mainsnak":{"snaktype":"value","property":"P569","datavalue":{"value":{"time":"+1952-03-11T00:00:00Z","precision":11,"calendarmodel":"http://www.wikidata.org/entity/Q1985727"},"type":"time"}},"rank":"normal"}],"P161":[{"mainsnak":{"snaktype":"value","property":"P161","datavalue":{"value":{"entity-type":"item","numeric-id":7,"id":"Q7"},"type":"wikibase-entityid"}},"qualifiers":{"P453":[{"snaktype":"value","property":"P453","datavalue":{"value":{"entity-type":"item","id":"Q8"},"type":"wikibase-entityid"}}],"P1810":[]},"rank":"normal"},{"mainsnak":{"snaktype":"value","property":"P161","datavalue":{"value":{"id":"Q9"},"type":"wikibase-entityid"}},"rank":"deprecated"}],"P968":[{"mainsnak":{"snaktype":"value","property":"P968","datavalue":{"value":"mailto:x@y.z","type":"string"}},"rank":"normal"}],"P1082":[{"mainsnak":{"snaktype":"value","property":"P1082","datavalue":{"value":{"amount":"+2117522","unit":"1"},"type":"quantity"}},"rank":"preferred"}]},"sitelinks":{"enwiki":{"site":"enwiki","title":"Douglas Adams","badges":[]},"frwiki":{"site":"frwiki","title":"Douglas Adams","badges":[]}}},"#;
        let st: [AtomicU64; 6] = Default::default();
        let (mut s, mut b) = (vec![], vec![]);
        work(format!("[\n{l}\n{{\"id\":\"Q43\",\"labels\":[],\"sitelinks\":[]}},\n]").as_bytes(), &mut s, &mut b, &st);
        let mut r = Rd { b: &s, i: 0 };
        assert_eq!((r.u32(), r.u16(), r.s(), r.s(), r.u8(), r.s()), (42, 2, "Douglas Adams", "Douglas Adams", 1, "Douglas Noel Adams"));
        assert_eq!(r.u32(), 3);
        let mut got = vec![];
        for _ in 0..3 { let (p, rk, v, nq) = (r.u16(), r.u8(), r.v(), r.u8()); got.push((p, rk, v, (0..nq).map(|_| (r.u16(), r.v())).collect::<Vec<_>>())) }
        assert!(r.done());
        assert!(got.contains(&(569, 0, Val::T(1952, 3, 11, 11, false), vec![])));
        assert!(got.contains(&(161, 0, Val::I(7), vec![(453, Val::I(8))])));
        assert!(got.contains(&(1082, 1, Val::Q(2117522.0, 0), vec![])));
        assert!(got.iter().all(|x| x.0 != 968 && x.2 != Val::I(9)));
        let mut r = Rd { b: &b, i: 0 };
        assert_eq!((r.u32(), r.s(), r.u8(), r.s()), (42, "Douglas Adams", 1, "Douglas Noel Adams"));
        assert!(r.done());
        assert_eq!((st[0].load(Relaxed), st[1].load(Relaxed), st[4].load(Relaxed)), (2, 1, 0));
        PC.with(|c| { let c = c.borrow(); assert_eq!((c[968], c[161], c[569], c[453]), ((1, 1), (1, 2), (1, 1), (0, 0))) });
    }
}
