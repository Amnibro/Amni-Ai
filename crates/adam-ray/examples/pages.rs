use adam_ray::{src_gf17, Dec, RayField};
use std::{f64::consts::LN_2, fs, time::Instant};
fn rss() -> (f64, f64, f64) { let s = fs::read_to_string("/proc/self/status").unwrap_or_default(); let g = |k: &str| s.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) / 1024.0; (g("VmRSS:"), g("VmHWM:"), g("RssFile:")) }
fn fnv(v: &[f32]) -> u64 { v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100_0000_01b3)) }
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: &str| a.get(i).cloned().unwrap_or_else(|| d.to_string());
    if a.get(1).map(String::as_str) == Some("pack") {
        let (out, t) = (arg(3, &format!("{}.gf17", a[2])), Instant::now());
        let tabs = adam_gf17::pack(&a[2], &out, false, arg(4, "128").parse().unwrap()).unwrap_or_else(|e| panic!("pack {}: {e}", a[2]));
        let raw: u64 = tabs.iter().map(|t| fs::metadata(format!("{}_{}.npy", a[2], t.name)).map_or(0, |m| m.len())).sum();
        let sz = fs::metadata(&out).map_or(0, |m| m.len());
        return println!("packed {out}: {sz} B ({:.3}x of {raw} B npy) in {:.1}s | tables {:?}", sz as f64 / raw as f64, t.elapsed().as_secs_f64(), tabs.iter().map(|t| (t.name.clone(), t.shape.clone())).collect::<Vec<_>>());
    }
    let (pk, lex, hold, nb, nt, arch, mode) = (arg(1, ""), arg(2, ""), arg(3, ""), arg(4, "2000").parse::<usize>().unwrap(), arg(5, "200").parse::<usize>().unwrap(), arg(6, "cpu"), arg(7, "both"));
    if mode == "upload" {
        let t = Instant::now();
        let tb = adam_ray::pack::load(&pk, &format!("{lex}/synsets.txt"), &format!("{lex}/wordcounts.txt")).unwrap();
        let (lt, r0, t) = (t.elapsed().as_secs_f64(), rss(), Instant::now());
        let g = adam_ray::gpu::Gpu::new(&tb).unwrap();
        let (ut, r1, s) = (t.elapsed().as_secs_f64(), rss(), src_gf17::stats());
        return println!("upload {pk}: tables {lt:.3}s (rss {:.1} MB) | gpu {} {ut:.3}s, of which level decode {:.3}s | rss {:.1} MB hwm {:.1} MB", r0.0, g.name, s[9] as f64 * 1e-9, r1.0, r1.1);
    }
    let prompt = arg(8, "User: What is the capital of France?\nAdam:").replace("\\n", "\n");
    let mut e = RayField::new(Some(pk.clone()), Some(format!("{lex}/synsets.txt")), Some(format!("{lex}/words.txt")), Some(format!("{lex}/wordcounts.txt")), Some(arch), None);
    let t = Instant::now();
    e.load().unwrap();
    let (lt, r0) = (t.elapsed().as_secs_f64(), rss());
    println!("pack {pk}\nbackend {} | L={} T={} copy={} | load {lt:.3}s | rss after load {:.1} MB (hwm {:.1}, file {:.1})", e.backend, e.l, e.t, e.copy, r0.0, r0.1, r0.2);
    if mode != "gen" {
        let d = fs::read(&hold).unwrap();
        let off = 1_234_567 % (d.len() - nb - 1);
        let x = &d[off..off + nb + 1];
        let t = Instant::now();
        let lg = e.logits_for(&x[..nb]);
        let et = t.elapsed().as_secs_f64();
        let bits = (0..nb).map(|i| { let r = &lg[i * 256..(i + 1) * 256]; let m = r.iter().fold(f32::MIN, |m, &v| m.max(v)) as f64; r.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m - r[x[i + 1] as usize] as f64 }).sum::<f64>() / nb as f64 / LN_2;
        let t = Instant::now();
        let bb = e.bpb(Some(&hold), 1, 128, 16, 1234).unwrap();
        let bt = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let w2 = fnv(&e.logits_for(&x[..nb])) == fnv(&lg);
        let wt = t.elapsed().as_secs_f64();
        let r1 = rss();
        println!("eval {nb} B @ {off}: bpb {bits:.17} | logits fnv {:016x} | {:.0} B/s ({et:.3}s) cold, warm {:.0} B/s same {w2} | batched bpb(16x128) {bb:.17} {:.0} B/s | rss {:.1} MB hwm {:.1} MB file {:.1} MB", fnv(&lg), nb as f64 / et, nb as f64 / wt, 2048.0 / bt, r1.0, r1.1, r1.2);
    }
    if mode != "bpb" {
        let t = Instant::now();
        let mut s = e.stream(&prompt, nt, 1.0, 1, &[], false, Some(7), None, 0.0, Dec::default());
        let (mut out, mut ft) = (String::new(), None);
        while let Some(c) = s.next(&mut e) { ft.get_or_insert(t.elapsed().as_secs_f64()); out.push_str(&c); }
        let dt = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut s = e.stream(&prompt, nt, 1.0, 1, &[], false, Some(7), None, 0.0, Dec::default());
        let mut o2 = String::new();
        while let Some(c) = s.next(&mut e) { o2.push_str(&c); }
        let (d2, r2) = (t.elapsed().as_secs_f64(), rss());
        println!("decode {nt} tok: first token {:.4}s (cold ttft incl. load {:.3}s) | {:.1} tok/s cold, warm {:.1} tok/s same {} | rss {:.1} MB hwm {:.1} MB\n  text {out:?}", ft.unwrap_or(dt), lt + ft.unwrap_or(dt), nt as f64 / dt, nt as f64 / d2, o2 == out, r2.0, r2.1);
    }
    let s = src_gf17::stats();
    println!("pages: {} levels pinned, {:.3}s in level decodes | streamed-level cell reads {} cache hits {} | cells fetched {} in {} batches ({:.3}s) | tiles hot {:.1} MB decoded {} evicted {}", s[8], s[9] as f64 * 1e-9, s[3], s[0], s[1], s[2], s[4] as f64 * 1e-9, s[5] as f64 / 1048576.0, s[6], s[7]);
}
