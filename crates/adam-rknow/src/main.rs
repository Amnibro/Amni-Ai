use memmap2::Mmap;
use rayon::prelude::*;
use serde_json::json;
use std::{cmp::Reverse, collections::{BinaryHeap, HashMap, HashSet}, env, fs, io::{self, BufWriter, Write}, path::{Path, PathBuf}, time::Instant};
use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};
const STOP: &[&str] = &["a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for", "by", "with", "from", "as", "is", "was", "are", "were", "be", "been", "it", "its", "that", "this", "which", "who", "he", "she", "his", "her", "they", "their", "has", "had", "have", "not", "but", "also", "into", "than", "then", "there"];
fn fnv(b: &[u8]) -> u64 { b.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3)) }
fn stops() -> HashSet<u64> { STOP.iter().map(|s| fnv(s.as_bytes())).collect() }
fn flush(w: &mut String, out: &mut Vec<u64>, st: &HashSet<u64>) {
    if w.is_empty() { return; }
    let one = w.chars().count() == 1 && !w.chars().all(|c| c.is_numeric());
    if !one {
        let n = w.len();
        if n > 4 && w.ends_with("ies") { w.truncate(n - 3); w.push('y'); } else if n > 3 && w.ends_with('s') && !(w.ends_with("ss") || w.ends_with("us") || w.ends_with("is")) { w.pop(); }
        let h = fnv(w.as_bytes());
        if !st.contains(&h) { out.push(h); }
    }
    w.clear();
}
fn tokens(t: &str, out: &mut Vec<u64>, st: &HashSet<u64>) {
    let mut w = String::new();
    for c in t.chars() {
        if c.is_ascii() { if c.is_ascii_alphanumeric() { w.push(c.to_ascii_lowercase()); } else { flush(&mut w, out, st); } continue; }
        for d in c.nfkd().filter(|d| !is_combining_mark(*d)) { if d.is_alphanumeric() { w.extend(d.to_lowercase()); } else { flush(&mut w, out, st); } }
    }
    flush(&mut w, out, st);
}
fn npy_i64(p: &Path) -> io::Result<Vec<i64>> {
    let b = fs::read(p)?;
    let s = if b[6] == 1 { 10 + u16::from_le_bytes([b[8], b[9]]) as usize } else { 12 + u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize };
    Ok(b[s..].chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
}
fn varint(o: &mut Vec<u8>, mut v: u64) { while v >= 128 { o.push((v as u8 & 127) | 128); v >>= 7; } o.push(v as u8); }
fn rvar(b: &[u8]) -> (u64, usize) { let (mut v, mut i) = (0u64, 0usize); loop { v |= ((b[i] & 127) as u64) << (7 * i); i += 1; if b[i - 1] < 128 { return (v, i); } } }
fn wcount(s: &str) -> usize { s.split_whitespace().count() }
fn chunks(t: &str) -> Vec<(usize, usize)> {
    let (mut out, mut cur, mut cw) = (vec![], None::<(usize, usize)>, 0usize);
    let mut pos = 0usize;
    for p in t.split('\n') {
        let (a, b) = (pos, pos + p.len());
        pos = b + 1;
        let wc = wcount(p);
        if wc == 0 { continue; }
        if wc > 150 {
            if let Some(c) = cur.take() { out.push(c); cw = 0; }
            let ws: Vec<usize> = p.char_indices().filter(|&(i, c)| !c.is_whitespace() && (i == 0 || p[..i].chars().next_back().map_or(true, |x| x.is_whitespace()))).map(|(i, _)| i).collect();
            let mut k = 0;
            while k < ws.len() { let e = if ws.len() - k < 130 { ws.len() } else { k + 100 }; out.push((a + ws[k], if e >= ws.len() { b } else { a + ws[e] })); k = e; }
        } else {
            cur = Some(cur.map_or((a, b), |(s, _)| (s, b)));
            cw += wc;
            if cw >= 60 { out.push(cur.take().unwrap()); cw = 0; }
        }
    }
    if let Some(c) = cur { out.push(c); }
    out
}
struct SegOut { docs: u32, pass: u32, ntok: u64, vocab: usize, bytes: usize }
fn wr<T: Copy>(p: &Path, v: &[T]) -> io::Result<()> { let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }; fs::write(p, b) }
fn rd<T: Copy + Default>(p: &Path) -> io::Result<Vec<T>> { let b = fs::read(p)?; let n = b.len() / std::mem::size_of::<T>(); let mut v = vec![T::default(); n]; unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr() as *mut u8, b.len()) }; Ok(v) }
fn seg(dir: &Path, tmp: &Path, st: &HashSet<u64>) -> io::Result<SegOut> {
    let (off, raw) = (npy_i64(&dir.join("off.npy"))?, npy_i64(&dir.join("raw.npy"))?);
    let blob = unsafe { Mmap::map(&fs::File::open(dir.join("blob.bin"))?)? };
    let mut post: HashMap<u64, (Vec<u8>, u32, u32)> = HashMap::new();
    let (mut pd, mut po, mut pl, mut pn) = (Vec::<u32>::new(), Vec::<u32>::new(), Vec::<u32>::new(), Vec::<u16>::new());
    let (mut tt, mut tk, mut ntok) = (vec![], vec![], 0u64);
    for d in 0..raw.len() {
        let x = zstd::bulk::decompress(&blob[off[d] as usize..off[d + 1] as usize], raw[d] as usize)?;
        let s = String::from_utf8_lossy(&x);
        let (title, text) = s.split_once('\0').unwrap_or(("", &s));
        tt.clear(); tokens(title, &mut tt, st);
        let tb = title.len() + 1;
        for (a, b) in chunks(text) {
            tk.clear(); tk.extend_from_slice(&tt); tokens(&text[a..b], &mut tk, st);
            if tk.len() <= tt.len() { continue; }
            let pid = pd.len() as u32;
            pd.push(d as u32); po.push((tb + a) as u32); pl.push((b - a) as u32); pn.push(tk.len().min(65535) as u16); ntok += tk.len() as u64;
            tk.sort_unstable();
            let mut i = 0;
            while i < tk.len() {
                let mut j = i + 1; while j < tk.len() && tk[j] == tk[i] { j += 1; }
                let e = post.entry(tk[i]).or_insert_with(|| (Vec::new(), 0, 0));
                let delta = if e.2 == 0 { pid } else { pid - e.1 } as u64;
                varint(&mut e.0, (delta << 3) | ((j - i).min(7) as u64)); e.1 = pid; e.2 += 1;
                i = j;
            }
        }
    }
    let mut keys: Vec<u64> = post.keys().copied().collect(); keys.sort_unstable();
    let (mut ptr, mut last, mut df, mut body) = (vec![0u64], vec![], vec![], Vec::<u8>::new());
    for k in &keys { let (v, l, n) = &post[k]; body.extend_from_slice(v); ptr.push(body.len() as u64); last.push(*l); df.push(*n); }
    fs::create_dir_all(tmp)?;
    wr(&tmp.join("terms.u64"), &keys)?; wr(&tmp.join("ptr.u64"), &ptr)?; wr(&tmp.join("last.u32"), &last)?; wr(&tmp.join("df.u32"), &df)?; fs::write(tmp.join("post.bin"), &body)?;
    wr(&tmp.join("p_doc.u32"), &pd)?; wr(&tmp.join("p_off.u32"), &po)?; wr(&tmp.join("p_len.u32"), &pl)?; wr(&tmp.join("p_ntok.u16"), &pn)?;
    Ok(SegOut { docs: raw.len() as u32, pass: pd.len() as u32, ntok, vocab: keys.len(), bytes: body.len() })
}
struct S { terms: Vec<u64>, ptr: Vec<u64>, df: Vec<u32>, post: Mmap }
fn merge(tmps: &[PathBuf], pbase: &[u64], dbase: &[u64], out: &Path) -> io::Result<(u64, u64)> {
    let ss: Vec<S> = tmps.iter().map(|t| Ok(S { terms: rd(&t.join("terms.u64"))?, ptr: rd(&t.join("ptr.u64"))?, df: rd(&t.join("df.u32"))?, post: unsafe { Mmap::map(&fs::File::open(t.join("post.bin"))?)? } })).collect::<io::Result<_>>()?;
    let mut h: BinaryHeap<Reverse<(u64, usize)>> = ss.iter().enumerate().filter(|(_, s)| !s.terms.is_empty()).map(|(i, s)| Reverse((s.terms[0], i))).collect();
    let mut cur = vec![0usize; ss.len()];
    let (mut tw, mut pw, mut dw, mut bw, mut fw) = (BufWriter::new(fs::File::create(out.join("terms.u64"))?), BufWriter::new(fs::File::create(out.join("ptr.u64"))?), BufWriter::new(fs::File::create(out.join("df.u32"))?), BufWriter::new(fs::File::create(out.join("pid.u32"))?), BufWriter::new(fs::File::create(out.join("tf.u8"))?));
    let (mut nb, mut v) = (0u64, 0u64);
    pw.write_all(&0u64.to_le_bytes())?;
    while let Some(Reverse((k, _))) = h.peek().copied() {
        let mut segs = vec![];
        while let Some(Reverse((k2, i))) = h.peek().copied() { if k2 != k { break; } h.pop(); segs.push(i); }
        segs.sort_unstable();
        let mut df = 0u64;
        for &i in &segs {
            let s = &ss[i]; let j = cur[i];
            let mut ch = &s.post[s.ptr[j] as usize..s.ptr[j + 1] as usize];
            let mut pid = 0u64;
            while !ch.is_empty() { let (x, n) = rvar(ch); ch = &ch[n..]; pid += x >> 3; bw.write_all(&((pid + pbase[i]) as u32).to_le_bytes())?; fw.write_all(&[(x & 7) as u8])?; nb += 1; }
            df += s.df[j] as u64;
            cur[i] += 1; if cur[i] < s.terms.len() { h.push(Reverse((s.terms[cur[i]], i))); }
        }
        tw.write_all(&k.to_le_bytes())?; pw.write_all(&nb.to_le_bytes())?; dw.write_all(&(df.min(u32::MAX as u64) as u32).to_le_bytes())?; v += 1;
    }
    tw.flush()?; pw.flush()?; dw.flush()?; bw.flush()?; fw.flush()?;
    let mut w = [BufWriter::new(fs::File::create(out.join("p_doc.u32"))?), BufWriter::new(fs::File::create(out.join("p_off.u32"))?), BufWriter::new(fs::File::create(out.join("p_len.u32"))?), BufWriter::new(fs::File::create(out.join("p_ntok.u16"))?)];
    for (i, t) in tmps.iter().enumerate() {
        for x in rd::<u32>(&t.join("p_doc.u32"))? { w[0].write_all(&((x as u64 + dbase[i]) as u32).to_le_bytes())?; }
        w[1].write_all(&fs::read(t.join("p_off.u32"))?)?; w[2].write_all(&fs::read(t.join("p_len.u32"))?)?; w[3].write_all(&fs::read(t.join("p_ntok.u16"))?)?;
    }
    for x in w.iter_mut() { x.flush()?; }
    Ok((v, nb))
}
fn rss() -> f64 { fs::read_to_string("/proc/self/status").unwrap_or_default().lines().find(|l| l.starts_with("VmHWM")).and_then(|l| l.split_whitespace().nth(1)).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0) / 1048576.0 }
fn main() -> io::Result<()> {
    let a: Vec<String> = env::args().skip(1).collect();
    if a.len() < 3 || a[0] != "build" { eprintln!("adam-rknow build <fetch/wikipedia> <out_dir> [threads] [limit_segs]\nadam-rknow tok <text>"); if a.first().map(String::as_str) == Some("tok") { let st = stops(); let mut o = vec![]; tokens(&a[1..].join(" "), &mut o, &st); println!("{}", json!(o.iter().map(|x| x.to_string()).collect::<Vec<_>>())); return Ok(()); } std::process::exit(2); }
    let (src, out, th, lim) = (Path::new(&a[1]), PathBuf::from(&a[2]), a.get(3).and_then(|x| x.parse().ok()).unwrap_or(6usize), a.get(4).and_then(|x| x.parse().ok()).unwrap_or(usize::MAX));
    let t0 = Instant::now(); let st = stops();
    let mut segs: Vec<String> = fs::read_dir(src)?.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).filter(|n| n.len() == 5 && n.bytes().all(|b| b.is_ascii_digit())).collect();
    segs.sort(); segs.truncate(lim);
    let tmp = out.join("tmp"); fs::create_dir_all(&tmp)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(th).build().unwrap();
    let res: Vec<(String, SegOut)> = pool.install(|| segs.par_iter().map(|s| { let t = Instant::now(); let r = seg(&src.join(s), &tmp.join(s), &st).unwrap(); eprintln!("{:.0}s seg {s} docs {} pass {} vocab {} post {}MB {:.1}s rss {:.2}GB", t0.elapsed().as_secs_f64(), r.docs, r.pass, r.vocab, r.bytes >> 20, t.elapsed().as_secs_f64(), rss()); (s.clone(), r) }).collect());
    let (mut pb, mut db) = (vec![0u64], vec![0u64]);
    for (_, r) in &res { pb.push(pb.last().unwrap() + r.pass as u64); db.push(db.last().unwrap() + r.docs as u64); }
    let (v, nb) = merge(&segs.iter().map(|s| tmp.join(s)).collect::<Vec<_>>(), &pb, &db, &out)?;
    let ntok: u64 = res.iter().map(|(_, r)| r.ntok).sum();
    let m = json!({"version": "rk2", "tokenizer": "rk1: NFKD strip marks, lowercase, alnum runs, drop 1-char non-digit, s/ies stem, stop list, fnv1a64", "stop": STOP, "segments": segs, "doc_base": db, "pass_base": pb, "n_pass": pb.last(), "n_doc": db.last(), "vocab": v, "postings": nb, "format": "pid.u32 absolute passage ids + tf.u8 (cap 7), ptr.u64 in postings", "ntok": ntok, "avgdl": ntok as f64 / (*pb.last().unwrap()).max(1) as f64, "tf_cap": 7, "chunk": "paragraphs merged to >=60 words, split >150 at 100 words", "title_in_passage": true, "build_s": t0.elapsed().as_secs_f64(), "peak_rss_gb": rss(), "threads": th, "source": "data/ks/fetch/wikipedia (Wikipedia, CC BY-SA 4.0)"});
    fs::write(out.join("meta.json"), serde_json::to_string_pretty(&m)?)?;
    fs::remove_dir_all(&tmp)?;
    println!("{}", m);
    Ok(())
}
