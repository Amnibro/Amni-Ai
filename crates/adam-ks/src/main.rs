use adam_ks::{paginate, pyjson, store::{read_docs, write_store, Store}, Ks};
use serde_json::json;
use std::{collections::HashMap, env, fs, io::{self, Write}, path::Path, time::Instant};
fn mem() -> (f64, f64) { let s = fs::read_to_string("/proc/self/status").unwrap_or_default(); let g = |k: &str| s.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0) / 1048576.0; (g("VmRSS:"), g("VmHWM:")) }
fn cpu() -> f64 { let s = fs::read_to_string("/proc/self/stat").unwrap_or_default(); let f: Vec<&str> = s.rsplit(')').next().unwrap_or("").split_whitespace().collect(); f.get(11).zip(f.get(12)).map_or(0.0, |(u, s)| (u.parse::<f64>().unwrap_or(0.0) + s.parse::<f64>().unwrap_or(0.0)) / 100.0) }
fn main() -> io::Result<()> {
    let a: Vec<String> = env::args().skip(1).collect();
    let (mut pos, mut kv, mut i) = (vec![], HashMap::new(), 0);
    while i < a.len() { if let Some(k) = a[i].strip_prefix("--") { kv.insert(k.to_string(), a.get(i + 1).cloned().unwrap_or_default()); i += 2; } else { pos.push(a[i].clone()); i += 1; } }
    let usage = "adam-ks encode <clean.parquet> <out_store> [--src wikipedia] [--chunk 4096] [--verify-stride 100]\n       adam-ks decode <store> [--ids 0,5,9] [--out file]\n       adam-ks verify <store> [--stride 1] [--against clean.parquet]\n       adam-ks ident\n       adam-ks bench <clean.parquet> [--n 100]\ncommon: [--ver ks2-clean1 | --pack <pack> --lex <lexdir>] [--home <AMNI_HOME>] [--threads N]";
    if pos.is_empty() { eprintln!("{usage}"); std::process::exit(2); }
    let g = |k: &str, d: &str| kv.get(k).cloned().unwrap_or_else(|| d.to_string());
    let n = |k: &str, d: usize| kv.get(k).and_then(|x| x.parse().ok()).unwrap_or(d);
    let home = kv.get("home").cloned().or_else(|| env::var("AMNI_HOME").ok()).unwrap_or_else(|| ".".into());
    let t0 = Instant::now();
    let ks = match kv.get("pack") { Some(p) => Ks::open(p, &g("lex", &format!("{home}/data/clean1")), kv.get("ver").map(String::as_str), n("threads", 0))?, None => Ks::ver(&home, &g("ver", "ks2-clean1"), n("threads", 0))? };
    let (tl, c0) = (t0.elapsed().as_secs_f64(), cpu());
    let arg = |i: usize| pos.get(i).cloned().ok_or_else(|| adam_ks::bad(usage));
    let r = match pos[0].as_str() {
        "ident" => ks.id.json(),
        "encode" => {
            let (src, out, t) = (arg(1)?, arg(2)?, Instant::now());
            let m = write_store(Path::new(&out), &g("src", "wikipedia"), read_docs(Path::new(&src), 256)?, &ks, n("chunk", 4096), &mut |p, sb, cb| eprintln!("{:.0}s pages {p} src {sb} code {cb} ({:.3} bpb, {:.0} kB/s)", t.elapsed().as_secs_f64(), 8.0 * cb as f64 / sb.max(1) as f64, sb as f64 / t.elapsed().as_secs_f64() / 1e3))?;
            let (te, ce) = (t.elapsed().as_secs_f64(), cpu() - c0);
            let st = Store::open(Path::new(&out))?;
            let (vi, tv): (Vec<usize>, Instant) = ((0..st.len()).step_by(n("verify-stride", 100).max(1)).collect(), Instant::now());
            st.get(&vi, &[&ks], true)?;
            let (sb, cb) = (m["src_bytes"].as_f64().unwrap_or(0.0), m["code_bytes"].as_f64().unwrap_or(0.0));
            json!({"store": out, "meta": m, "code_bpb": 8.0 * cb / sb.max(1.0), "encode_s": te, "enc_Bps": sb / te, "enc_Bps_per_cpu_s": sb / ce.max(1e-9), "cpu_s": ce, "threads": ks.threads(), "verified_pages": vi.len(), "verify_s": tv.elapsed().as_secs_f64(), "load_s": tl, "rss_gb": mem().0, "peak_rss_gb": mem().1})
        }
        "decode" => {
            let st = Store::open(Path::new(&arg(1)?))?;
            let ids: Vec<usize> = kv.get("ids").map_or_else(|| (0..st.len()).collect(), |s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
            let mut o: Box<dyn Write> = match kv.get("out") { Some(f) => Box::new(io::BufWriter::new(fs::File::create(f)?)), None => Box::new(io::stdout().lock()) };
            let (t, mut nb) = (Instant::now(), 0usize);
            for c in ids.chunks(4096) { for x in st.get(c, &[&ks], true)? { nb += x.len(); o.write_all(&x)?; } }
            o.flush()?;
            json!({"pages": ids.len(), "bytes": nb, "decode_s": t.elapsed().as_secs_f64(), "dec_Bps": nb as f64 / t.elapsed().as_secs_f64()})
        }
        "verify" => {
            let st = Store::open(Path::new(&arg(1)?))?;
            let ids: Vec<usize> = (0..st.len()).step_by(n("stride", 1).max(1)).collect();
            let mut src = kv.get("against").map(|p| read_docs(Path::new(p), 256)).transpose()?.map(|d| d.flat_map(|x| x.map_or_else(|e| vec![Err(e)], |d| paginate(d.text.as_bytes()).into_iter().map(|p| Ok(p.to_vec())).collect())).enumerate());
            let (t, mut nb, mut ex, mut ck) = (Instant::now(), 0usize, 0usize, 0usize);
            for c in ids.chunks(4096) {
                let dec = st.get(c, &[&ks], true)?;
                for (&i, x) in c.iter().zip(&dec) { nb += x.len(); ck += 1; if let Some(it) = src.as_mut() { ex += it.find(|(j, _)| *j == i).map_or(0, |(_, p)| p.is_ok_and(|p| &p == x) as usize); } }
            }
            let td = t.elapsed().as_secs_f64();
            json!({"store": arg(1)?, "pages": ids.len(), "crc_ok": ck, "exact_vs_source": kv.contains_key("against").then(|| ex as f64 / ids.len().max(1) as f64), "bytes": nb, "decode_s": td, "dec_Bps": nb as f64 / td, "dec_Bps_per_cpu_s": nb as f64 / (cpu() - c0).max(1e-9), "threads": ks.threads(), "coder": st.meta["coders"][0]["coder"], "rss_gb": mem().0, "peak_rss_gb": mem().1})
        }
        "bench" => {
            let pg: Vec<Vec<u8>> = read_docs(Path::new(&arg(1)?), 256)?.filter_map(Result::ok).flat_map(|d| paginate(d.text.as_bytes()).into_iter().map(<[u8]>::to_vec).collect::<Vec<_>>()).take(n("n", 100)).collect();
            let (nb, init) = (pg.iter().map(Vec::len).sum::<usize>(), adam_ray::cpu::init);
            let (t, mut lg, mut l2, mut par) = (Instant::now(), [0f32; 256], [0f32; 256], 0usize);
            for p in &pg { let mut s = init(); ks.step_ref(&mut s, &mut lg, 10); for &b in p { ks.step_ref(&mut s, &mut lg, b); } }
            let tr = t.elapsed().as_secs_f64();
            for p in &pg { let (mut s, mut s2) = (init(), init()); ks.steps(std::slice::from_mut(&mut s), std::slice::from_mut(&mut lg), &[10]); ks.step_ref(&mut s2, &mut l2, 10); for &b in p { par += (lg.iter().zip(&l2).all(|(x, y)| x.to_bits() == y.to_bits()) && s == s2) as usize; ks.steps(std::slice::from_mut(&mut s), std::slice::from_mut(&mut lg), &[b]); ks.step_ref(&mut s2, &mut l2, b); } }
            let (mut rs, mut bad) = (12345u64, 0usize);
            for it in 0..100000 { let mut x = [0f32; 256]; for v in x.iter_mut() { rs = rs.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); let r = (rs >> 33) as u32; *v = match it % 4 { 0 => (r % 7) as f32 - 3.0, 1 => (r % 1000) as f32 / 7.0, 2 => if r % 5 == 0 { -1e30 } else if r % 3 == 0 { -2e30 } else { -0.0 }, _ => if r % 50 == 0 { f32::NAN } else if r % 7 == 0 { 0.0 } else { -((r % 3) as f32) } }; } bad += (adam_ks::top3(&x) != adam_ray::cpu::top3(&x)) as usize; }
            let t = Instant::now();
            let cs = ks.encode_many(&pg.iter().map(Vec::as_slice).collect::<Vec<_>>(), false).0;
            let te = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let ok = ks.decode_many(&pg.iter().map(Vec::len).collect::<Vec<_>>(), &cs.iter().map(Vec::as_slice).collect::<Vec<_>>()).iter().zip(&pg).filter(|(d, p)| d == p).count();
            let td = t.elapsed().as_secs_f64();
            json!({"pages": pg.len(), "bytes": nb, "threads": ks.threads(), "lanes": ks.lanes, "ref_model_ns_per_byte": 1e9 * tr / nb as f64, "fast_vs_ref_bit_exact_steps": par as f64 / nb as f64, "top3_fuzz_mismatches": bad, "encode_ns_per_byte": 1e9 * te / nb as f64, "decode_ns_per_byte": 1e9 * td / nb as f64, "enc_Bps": nb as f64 / te, "dec_Bps": nb as f64 / td, "roundtrip": ok as f64 / pg.len() as f64, "code_bpb": 8.0 * cs.iter().map(Vec::len).sum::<usize>() as f64 / nb as f64})
        }
        _ => { eprintln!("{usage}"); std::process::exit(2) }
    };
    println!("{}", pyjson(&r, Some(1), 0));
    Ok(())
}
