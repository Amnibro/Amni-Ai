use adam_gf17::{rss, Store, Table, Writer, PW};
use adam_ray::{npy::Npz, RayField};
use adam_train::npy;
use half::f16;
use rayon::prelude::*;
use std::{env, fs, io, time::Instant};
const TEXTS: [&str; 5] = ["User: Hi! My name is Anna and I live in Denver.\nAdam: Nice to meet you, Anna.\nUser: Where do I live?\nAdam:", "User: My lab partner is 63.\nAdam: Got it, your lab partner is 63.\n\nUser: How old did I say my lab partner was?\nAdam:", "User: What is the capital of France?\nAdam: The capital of France is Paris.", "The quick brown fox jumps over the lazy dog, and water boils at 100 degrees Celsius at sea level.", "User: Give me some cuisines I should try\nAdam: Here are some cuisines you should try - Indian, Mediterranean, Italian, French and Chinese\n"];
fn sm(x: u64) -> u64 { let mut z = x.wrapping_add(0x9E3779B97F4A7C15); z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^ (z >> 31) }
fn gen(t: usize, l: usize, w: usize) -> u16 { let b = sm((t as u64) << 56 ^ (l as u64) << 40 ^ w as u64) as u16; if t == 1 && b & 0x7C00 == 0x7C00 { b ^ 0x4000 } else { b } }
fn pct(v: &[f64], q: f64) -> f64 { let mut v = v.to_vec(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[((v.len() - 1) as f64 * q).round() as usize] }
fn st(v: &[f64]) -> String { format!("p50 {:8.1} us  p99 {:8.1} us  max {:8.1} us", pct(v, 0.5), pct(v, 0.99), pct(v, 1.0)) }
fn gb(b: usize) -> f64 { b as f64 / 1e9 }
fn a(i: usize) -> String { env::args().nth(i).unwrap_or_else(|| panic!("missing arg {i}; usage: gf17 pack|tiers|bench|synth|density <args>")) }
fn ent_h(s: &Store, t: usize, k: usize) -> (f64, usize) {
    let tb = &s.tabs[t];
    let h = (0..tb.lv()).flat_map(|l| (0..tb.np()).map(move |p| (l, p))).map(|(l, p)| { let n = (tb.words() - p * adam_gf17::PX).min(adam_gf17::PX); s.pdig(t, l, p, 4).unwrap()[k][..n].par_iter().fold(|| [0u64; 17], |mut h, &d| { h[d as usize] += 1; h }).reduce(|| [0u64; 17], |a, b| std::array::from_fn(|i| a[i] + b[i])) }).fold([0u64; 17], |a, b| std::array::from_fn(|i| a[i] + b[i]));
    let n: u64 = h.iter().sum();
    (h.iter().filter(|&&c| c > 0).map(|&c| { let p = c as f64 / n as f64; -p * p.log2() }).sum(), n as usize)
}
fn pbytes(s: &Store, f: impl Fn(&adam_gf17::Ent) -> bool) -> (usize, usize) { s.ents.iter().filter(|e| f(e)).fold((0, 0), |a, e| (a.0 + e.len as usize, a.1 + e.rows as usize * PW)) }
fn f16s(b: &[u8]) -> Vec<f64> { b.chunks(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64()).collect() }
fn cmd_pack() -> io::Result<()> {
    let (pre, out, ord, tile) = (a(2), a(3), a(4) == "ord", env::args().nth(5).map_or(0, |x| x.parse::<usize>().unwrap()));
    let t0 = Instant::now();
    let tabs = adam_gf17::pack(&pre, &out, ord, tile)?;
    let tp = t0.elapsed().as_secs_f64();
    let s = Store::open(&out, 256 << 20)?;
    let (mut raw_tot, mut hb) = (0usize, 0f64);
    println!("packed {pre} -> {out} (key={}, tile={tile}) in {tp:.2}s", if ord { "ord" } else { "raw" });
    println!("{:<5} {:<4} {:<20} {:>12} {:>12} {:>12} {:>6} {:>6} {:>10} {:>8} {:>30}", "table", "dt", "shape", "raw npy B", "uncoded B", "file B", "ratio", "exact", "cos", "out GB/s", "H(d3,d2,d1,d0) bits/digit");
    for (t, tb) in tabs.iter().enumerate() {
        let f = format!("{pre}_{}.npy", tb.name);
        let (z, rb) = (Npz::open(&f)?, fs::metadata(&f)?.len() as usize);
        let ar = z.get("")?;
        let raw = &ar.data[..tb.n() * tb.dt];
        let t1 = Instant::now();
        let (b, _) = s.unpack(t, 4)?;
        let td = t1.elapsed().as_secs_f64();
        let (pb, ub) = pbytes(&s, |e| e.t as usize == t);
        let cos = if tb.dt == 2 { let (x, y) = (f16s(raw), f16s(&b)); let (d, nx, ny) = x.par_iter().zip(&y).filter(|(p, q)| p.is_finite() && q.is_finite()).map(|(p, q)| (p * q, p * p, q * q)).reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2)); format!("{:.12}", d / (nx.sqrt() * ny.sqrt())) } else { "-".into() };
        let hs: Vec<(f64, usize)> = (0..4).rev().map(|k| ent_h(&s, t, k)).collect();
        hb += hs.iter().map(|(h, n)| h * *n as f64 / 8.0).sum::<f64>();
        raw_tot += rb;
        println!("{:<5} {:<4} {:<20} {:>12} {:>12} {:>12} {:>6.3} {:>6} {:>10} {:>8.2} {:>30}", tb.name, tb.descr(), format!("{:?}", tb.shape), rb, ub, pb, pb as f64 / rb as f64, b == raw, cos, gb(raw.len()) / td, hs.iter().map(|x| format!("{:.3}", x.0)).collect::<Vec<_>>().join(","));
    }
    let (fl, un) = (s.len(), pbytes(&s, |_| true).1 + adam_gf17::HDR);
    let ss: Vec<(usize, usize)> = (0..4).rev().map(|k| pbytes(&s, |e| adam_gf17::stream(&s.tabs, e) == k)).collect();
    println!("total raw npy {} B; uncoded page file {} B ({:.3}x); this file {} B ({:.3}x raw, {:.3}x uncoded); per-plane digit-entropy bound {:.0} B ({:.3}x raw)", raw_tot, un, un as f64 / raw_tot as f64, fl, fl as f64 / raw_tot as f64, fl as f64 / un as f64, hb, hb / raw_tot as f64);
    ss.iter().enumerate().for_each(|(i, (c, u))| println!("stream {} {:<34} {:>12} B (uncoded {:>12} B, {:.3}x)", 3 - i, ["(exact tables + d3 of value tables)", "(d2)", "(d1)", "(d0)"][i], c, u, *c as f64 / *u as f64));
    Ok(())
}
fn cmd_tiers() -> io::Result<()> {
    let (pre, sp, lex, scr) = (a(2), a(3), a(4), a(5));
    let s = Store::open(&sp, 256 << 20)?;
    let (syn, wd, wc) = (format!("{lex}/synsets.txt"), format!("{lex}/words.txt"), format!("{lex}/wordcounts.txt"));
    let eng = |p: &str| -> io::Result<Vec<Vec<f32>>> { let mut r = RayField::new(Some(p.into()), Some(syn.clone()), Some(wd.clone()), Some(wc.clone()), Some("cpu".into()), None); r.load()?; Ok(TEXTS.iter().map(|x| r.logits_for(x.as_bytes())).collect()) };
    let rf = eng(&format!("{pre}_sid.npy"))?;
    fs::create_dir_all(&scr)?;
    let x = format!("{scr}/gf17tier");
    for n in ["sid", "bias"] { let t = s.table(n).unwrap(); npy(&format!("{x}_{n}.npy"), s.tabs[t].descr(), &s.tabs[t].shape, &s.unpack(t, 4)?.0)?; }
    let orig: Vec<Vec<f64>> = ["sw", "dp"].iter().map(|n| Npz::open(&format!("{pre}_{n}.npy")).and_then(|z| z.get("").map(|a| f16s(&a.data[..a.n() * 2])))).collect::<io::Result<_>>()?;
    println!("store {sp} key={} texts={} positions={}", if s.tabs[s.table("sw").unwrap()].ord { "ord" } else { "raw" }, TEXTS.len(), rf.iter().map(|v| v.len() / 256).sum::<usize>());
    println!("{:<6} {:>5} {:>12} {:>12} {:>9} {:>10} {:>12} {:>12} {:>12}", "tables", "tiers", "max|dlogit|", "mean|dlogit|", "argmax=", "zeroed", "sw rel p50", "sw rel max", "dp rel p50");
    for (tag, set) in [("sw+dp", vec!["sw", "dp"]), ("sw", vec!["sw"]), ("dp", vec!["dp"])] {
        for tiers in (1..=4).filter(|&k| k < 4 || tag == "sw+dp") {
            let (mut zs, mut rel) = (0usize, vec![]);
            for (i, n) in ["sw", "dp"].iter().enumerate() {
                let t = s.table(n).unwrap();
                let (b, z) = s.unpack(t, if set.contains(n) { tiers } else { 4 })?;
                zs += z;
                let y = f16s(&b);
                let r: Vec<f64> = orig[i].par_iter().zip(&y).filter(|(p, _)| p.is_finite() && **p != 0.0).map(|(p, q)| ((q - p) / p).abs()).collect();
                rel.push((pct(&r, 0.5), r.iter().cloned().fold(0.0, f64::max)));
                npy(&format!("{x}_{n}.npy"), "<f2", &s.tabs[t].shape, &b)?;
            }
            let lg = eng(&format!("{x}_sid.npy"))?;
            let (mut mx, mut sum, mut cnt, mut agree, mut pos) = (0f64, 0f64, 0usize, 0usize, 0usize);
            for (r, g) in rf.iter().zip(&lg) { r.iter().zip(g).for_each(|(p, q)| { let d = (*p as f64 - *q as f64).abs(); mx = mx.max(d); sum += d; cnt += 1; }); for (rp, gp) in r.chunks(256).zip(g.chunks(256)) { let am = |v: &[f32]| v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b }); agree += (am(rp) == am(gp)) as usize; pos += 1; } }
            println!("{:<6} {:>5} {:>12.6} {:>12.6} {:>8.1}% {:>10} {:>12.3e} {:>12.3e} {:>12.3e}", tag, tiers, mx, sum / cnt as f64, 100.0 * agree as f64 / pos as f64, zs, rel[0].0, rel[0].1, rel[1].0);
        }
    }
    ["sid", "sw", "dp", "bias"].iter().for_each(|n| { let _ = fs::remove_file(format!("{x}_{n}.npy")); });
    Ok(())
}
fn seq(s: &mut Store, cap: usize, tiers: usize, cold: bool) -> (usize, usize, f64) {
    cold.then(|| s.cold());
    let (t0, mut rd, mut ob) = (Instant::now(), 0usize, 0usize);
    let mut buf = vec![];
    for t in (0..s.tabs.len()).rev() { for l in 0..s.tabs[t].lv() {
        if ob >= cap { break; }
        let tb = s.tabs[t].clone();
        buf.resize(tb.per() * tb.dt, 0u8);
        s.level(t, l, tiers, &mut buf).unwrap();
        rd += (0..tb.np()).flat_map(|p| (0..4).map(move |k| (p, k))).filter(|&(_, k)| tb.exact || k + tiers >= 4).map(|(p, k)| s.ents[s.ent(t, l, p, k)].len as usize).sum::<usize>();
        ob += buf.len();
    } }
    (rd, ob, t0.elapsed().as_secs_f64())
}
fn cmd_bench() -> io::Result<()> {
    let (sp, n, bud) = (a(2), a(3).parse::<usize>().unwrap(), a(4).parse::<usize>().unwrap());
    let (pre, cap) = (env::args().nth(5).filter(|x| x != "synth"), env::args().nth(6).map_or(usize::MAX, |x| x.parse::<usize>().unwrap() << 30));
    let mut s = Store::open(&sp, bud << 20)?;
    let sw = s.tabs[s.table("sw").unwrap()].clone();
    let (swi, lv, tt, sl) = (s.table("sw").unwrap(), sw.lv(), sw.shape[1], sw.shape[2]);
    let q: Vec<(usize, usize, usize)> = (0..n as u64).map(|i| { let h = sm(i ^ 0xABCDEF); ((h % lv as u64) as usize, ((h >> 8) % tt as u64) as usize, ((h >> 48) % sl as u64) as usize) }).collect();
    println!("store {sp}: {:.2} GB, tile {}, sw {:?}, budget {} MB, {} random fetches, rayon threads {}", gb(s.len()), s.tile, sw.shape, bud, n, rayon::current_num_threads());
    let (r0, _, _) = rss();
    let (mut lc, mut lw, mut ld, mut l1, mut lk) = (vec![], vec![], vec![], vec![], vec![]);
    let mut got = vec![];
    for &x in &q { s.cold(); let t0 = Instant::now(); let v = s.fetch(x.0, x.1, x.2)?; lc.push(t0.elapsed().as_secs_f64() * 1e6); got.push(v); }
    for &x in &q { std::hint::black_box(s.fetch(x.0, x.1, x.2)?); }
    for &x in &q { let t0 = Instant::now(); std::hint::black_box(s.fetch(x.0, x.1, x.2)?); lw.push(t0.elapsed().as_secs_f64() * 1e6); }
    println!("single fetch  cold  {}", st(&lc));
    println!("single fetch  warm  {}", st(&lw));
    (s.tile > 0).then(|| -> io::Result<()> {
        for &x in &q { s.reset(0); let t0 = Instant::now(); std::hint::black_box(s.fetch(x.0, x.1, x.2)?); ld.push(t0.elapsed().as_secs_f64() * 1e6); }
        for &x in &q { s.reset(0); let t0 = Instant::now(); std::hint::black_box(s.get(&[(swi, x.0, x.1 * sl + x.2)], 4)?); l1.push(t0.elapsed().as_secs_f64() * 1e6); }
        println!("single fetch  warm, tiles decoded (sid+sw: 2 tiles)  {}", st(&ld));
        println!("sw word       warm, 1 tile decoded                   {}", st(&l1));
        for &x in &q { s.reset(1); let t0 = Instant::now(); std::hint::black_box(s.fetch(x.0, x.1, x.2)?); lk.push(t0.elapsed().as_secs_f64() * 1e6); }
        println!("single fetch  cold data, plane index resident        {}", st(&lk));
        Ok(())
    }).transpose()?;
    let (mut bc, mut bw) = (vec![], vec![]);
    for c in q.chunks(64) { s.cold(); let t0 = Instant::now(); std::hint::black_box(s.fetch_many(c, 4)?); bc.push(t0.elapsed().as_secs_f64() * 1e6); }
    for c in q.chunks(64) { std::hint::black_box(s.fetch_many(c, 4)?); let t0 = Instant::now(); std::hint::black_box(s.fetch_many(c, 4)?); bw.push(t0.elapsed().as_secs_f64() * 1e6); }
    println!("batch of 64   cold  {}", st(&bc));
    (s.tile > 0).then(|| -> io::Result<()> { let mut bk = vec![]; for c in q.chunks(64) { s.reset(1); let t0 = Instant::now(); std::hint::black_box(s.fetch_many(c, 4)?); bk.push(t0.elapsed().as_secs_f64() * 1e6); } println!("batch of 64   cold data, plane index resident  {}", st(&bk)); Ok(()) }).transpose()?;
    println!("batch of 64   warm  {}", st(&bw));
    let (r1, h1, f1) = rss();
    let exact = match &pre { Some(p) => { let (zi, zw) = (Npz::open(&format!("{p}_sid.npy"))?, Npz::open(&format!("{p}_sw.npy"))?); let (ai, aw) = (zi.get("")?, zw.get("")?); q.iter().zip(&got).all(|(&(l, c, k), &(i, w))| { let e = (l * tt + c) * sl + k; ai.data[e] == i && u16::from_le_bytes([aw.data[2 * e], aw.data[2 * e + 1]]) == w }) } None => q.iter().zip(&got).all(|(&(l, c, k), &(i, w))| { let e = c * sl + k; (gen(0, l, e / 2) >> (8 * (e % 2))) as u8 == i && gen(1, l, e) == w }) };
    println!("fetch values bit-exact vs {}: {exact}; LRU hits {} misses {} evictions {} hot {} MB", pre.as_deref().unwrap_or("generator"), s.hits, s.miss, s.ev, s.used >> 20);
    println!("RSS after random fetches: VmRSS {} MB (file-backed {} MB), VmHWM {} MB, RSS at open {} MB, budget {} MB", r1 >> 20, f1 >> 20, h1 >> 20, r0 >> 20, bud);
    for th in [1usize, rayon::current_num_threads()] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(th).build().unwrap();
        for (nm, tiers) in [("full tiers d3..d0", 4), ("coarse d3 + exact tables", 1)] {
            let ((b, o, t), (_, o2, t2)) = pool.install(|| (seq(&mut s, cap, tiers, true), seq(&mut s, cap, tiers, false)));
            println!("sequential {th:>2} thr {nm:<26} cold {:6.3} GB file -> {:6.3} GB tables in {:6.2}s = {:5.2} GB/s file, {:5.2} GB/s tables | warm {:5.2} GB/s tables", gb(b), gb(o), t, gb(b) / t, gb(o) / t, gb(o2) / t2);
        }
    }
    Ok(())
}
fn cmd_synth() -> io::Result<()> {
    let (out, lv, lt, tile) = (a(2), a(3).parse::<usize>().unwrap(), a(4).parse::<u32>().unwrap(), env::args().nth(5).map_or(0, |x| x.parse::<usize>().unwrap()));
    let tabs = vec![Table { name: "sid".into(), dt: 1, ord: false, exact: true, shape: vec![lv, 1 << lt, 3] }, Table { name: "sw".into(), dt: 2, ord: true, exact: false, shape: vec![lv, 1 << lt, 3] }];
    let mut w = Writer::create(&out, tabs, true, tile)?;
    let t0 = Instant::now();
    for t in 0..2 { for l in 0..lv { let v: Vec<u16> = (0..w.tabs[t].words()).into_par_iter().map(|i| gen(t, l, i)).collect(); w.put(t, l, &v)?; } eprint!("."); }
    w.finish()?;
    let sz = fs::metadata(&out)?.len() as usize;
    println!("\nsynthetic {out}: {lv} levels x 2^{lt} cells x 3 slots (sid u8 + sw f16) = {:.2} GB page file in {:.1}s ({:.2} GB/s)", gb(sz), t0.elapsed().as_secs_f64(), gb(sz) / t0.elapsed().as_secs_f64());
    Ok(())
}
fn cmd_density() -> io::Result<()> {
    let (sp, pre) = (a(2), a(3));
    let s = Store::open(&sp, 64 << 20)?;
    let raw: usize = s.tabs.iter().map(|t| fs::metadata(format!("{pre}_{}.npy", t.name)).map(|m| m.len() as usize)).sum::<io::Result<usize>>()?;
    let (dp, dw): (Vec<(usize, bool)>, usize) = (s.ents.iter().map(|e| (e.rows as usize * PW, e.k == 3)).collect(), s.tabs.iter().map(|t| t.lv() * t.words()).sum());
    let pk = |g: usize, b: usize, g3: usize, b3: usize| dp.iter().map(|&(d, top)| if top { d.div_ceil(g3) * b3 } else { d.div_ceil(g) * b }).sum::<usize>().div_ceil(8) + adam_gf17::HDR;
    println!("store {sp}: {} words, {} padded digits in {} planes; raw npy {raw} B; this file {} B ({:.3}x raw)", dw, dp.iter().map(|x| x.0).sum::<usize>(), dp.len(), s.len(), s.len() as f64 / raw as f64);
    for (nm, g, b, g3, b3) in [("rule 5: 1 digit / 8-bit channel", 1, 8, 1, 8), ("2 digits / 9 bits", 2, 9, 2, 9), ("3 digits / 13 bits", 3, 13, 3, 13), ("7 digits / 29 bits", 7, 29, 7, 29), ("7/29 + d3 base-14 8 digits / 31 bits", 7, 29, 8, 31), ("41 digits / 168 bits (21 B)", 41, 168, 41, 168)] {
        let z = pk(g, b, g3, b3);
        println!("{nm:<40} {:>12} B  {:.4}x raw  {:.4}x rule-5  {:.3} bits/digit", z, z as f64 / raw as f64, z as f64 / pk(1, 8, 1, 8) as f64, b as f64 / g as f64);
    }
    println!("{:<40} {:>12.0} B  {:.4}x raw  (log2 17 = {:.4} bits/digit, no packing can go below)", "information limit, planar digits", dp.iter().map(|x| x.0).sum::<usize>() as f64 * 17f64.log2() / 8.0, dp.iter().map(|x| x.0).sum::<usize>() as f64 * 17f64.log2() / 8.0 / raw as f64, 17f64.log2());
    println!("{:<40} {:>12} B  {:.4}x raw  (4 digits of one word = the 16-bit key itself; not planar)", "whole-word limit (17^4 > 2^16)", dw * 2, (dw * 2) as f64 / raw as f64);
    Ok(())
}
fn main() -> io::Result<()> { match a(1).as_str() { "pack" => cmd_pack(), "tiers" => cmd_tiers(), "bench" => cmd_bench(), "synth" => cmd_synth(), "density" => cmd_density(), c => panic!("unknown command {c}") } }
