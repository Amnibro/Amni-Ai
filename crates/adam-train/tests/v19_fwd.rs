use adam_ray::{npy::Npz, pack::copies, RayField};
use adam_train::{cfg::Cfg, cpu::Cpu, gpu::Gpu, load_pages, Engine};
use std::{env::var, time::Instant};
const TEXTS: [&[u8]; 10] = [b"The weather today is sunny, and the kaolinite clay mineral forms in warm, humid soils.", b"def add(a, b):\n    return a + b\n\n```python\nprint(add(2, 3))\n```\nDone.", b"  Indented line; with (punct) {braces} and \"quotes\" -- plus a\ttab.\nNext line!", b"Photosynthesis converts light energy into chemical energy stored in glucose molecules within the plant.", b"User: Why do volcanoes erupt near tectonic boundaries?\nAdam: Volcanoes erupt where magma rises through cracks in the crust.\n\nUser: Who is Adam?\nAdam: I am Adam.", b"User: My sister lives in Denver.\nAdam: Noted.\n\nUser: Where does my sister live?\nAdam:", b"User: My sister lives in San Diego.\nAdam: Noted.\n\nUser: Where does my sister live?\nAdam: Your sister lives in San ", b"User: My sister lives in Denver.\nAdam: Noted.\n\nUser: Denver is in Colorado.\nAdam: Noted.\n\nUser: Which state is my sister in?\nAdam:", b"User: My sister lives in Denver.\nAdam: Nice.\n\nUser: Actually she moved to Austin.\nAdam: Got it.\n\nUser: Where does my sister live?\nAdam:", b"User: My sister lives in Denver.\nAdam: Noted.\n\n\x1eUser: My brother lives in Austin.\nAdam: Noted.\n\nUser: Where does my brother live?\nAdam:"];
fn am(v: &[f32]) -> usize { v.iter().enumerate().fold(0, |m, (i, &x)| if x > v[m] { i } else { m }) }
fn cmp(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f64, usize, usize, usize) { a.iter().zip(b).fold((0f64, 0, 0, 0), |(w, t, x, n), (a, b)| (a.iter().zip(b).fold(w, |w, (p, q)| w.max((p - q).abs() as f64)), t + a.chunks(256).zip(b.chunks(256)).filter(|(p, q)| am(p) == am(q)).count(), x + a.chunks(256).zip(b.chunks(256)).filter(|(p, q)| p == q).count(), n + a.len() / 256)) }
#[test]
fn v19_forward() {
    let Ok(pk) = var("ADAM_V19_PACK") else { return eprintln!("skip: set ADAM_V19_PACK=<..._sid.npy> (optional ADAM_V19_TEXTS=<\\0-separated file>, ADAM_V19_REF=<taichi t{{i}} npz>, ADAM_RAY_ARCHS)") };
    let home = var("AMNI_HOME").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../..").to_string());
    let texts: Vec<Vec<u8>> = var("ADAM_V19_TEXTS").map_or_else(|_| TEXTS.iter().map(|t| t.to_vec()).collect(), |f| std::fs::read(f).unwrap().split(|&c| c == 0).map(<[u8]>::to_vec).collect());
    let sh = Npz::open(&pk).unwrap().get("").unwrap().shape.clone();
    let (syn, l2, (cp, mr)) = (format!("{home}/data/big2/synsets.txt"), sh[1].trailing_zeros().to_string(), copies(sh[0]));
    let (cp, mr) = (cp.to_string(), mr.to_string());
    let c = Cfg::with(&[("VER", "19"), ("COPY", &cp), ("MR", &mr), ("LEVELS", "21"), ("HOPS", "4"), ("QT", "1"), ("TERN", "1"), ("TERN_SCOPE", "cell"), ("QLEV", "5"), ("LOG2_ENTRIES", &l2), ("BS", "1"), ("CTX", "8192"), ("ARCH", "cpu"), ("PAGES", &pk), ("AMNI_HOME", &home), ("SYNSETS", &syn)]);
    let t0 = Instant::now();
    let gn: usize = var("ADAM_V19_GREEDY").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let greedy = |f: &mut dyn FnMut(&[u8], Option<u8>) -> Vec<f32>, p: &[u8]| { let (mut o, mut lg) = (vec![], f(p, None)); for _ in 0..gn { let c = am(&lg) as u8; if c == 10 || c == 30 { break; } o.push(c); lg = f(&[p, &o].concat(), Some(c)); } o };
    let (tr, tg): (Vec<Vec<f32>>, Vec<Vec<u8>>) = { let mut e = Cpu::new(&c); load_pages(&c, &mut e).unwrap(); (texts.iter().map(|t| { e.logits(t); e.lg[..t.len() * 256].to_vec() }).collect(), texts.iter().map(|t| greedy(&mut |x, _| e.logits(x), t)).collect()) };
    println!("pack {pk} L={} T={} copy={cp} mr={mr} | {} texts {} positions | adam-train cpu forward {:.1}s", sh[0], sh[1], texts.len(), texts.iter().map(Vec::len).sum::<usize>(), t0.elapsed().as_secs_f64());
    let ti: Option<Vec<Vec<f32>>> = var("ADAM_V19_REF").ok().map(|r| { let z = Npz::open(&r).unwrap(); (0..texts.len()).map(|i| z.get(&format!("t{i}")).unwrap().f32s()).collect() });
    let row = |nm: &str, a: &[Vec<f32>], b: &[Vec<f32>]| { let (w, t, x, n) = cmp(a, b); println!("| {nm:<34} | {w:.3e} | {t}/{n} | {x}/{n} |"); (w, t == n) };
    println!("| comparison | max abs logit diff | top-1 agree | bit-identical rows |\n|---|---|---|---|");
    if let Some(ti) = ti.as_ref() { row("adam-train cpu vs taichi trainer", &tr, ti); }
    let mut ok = true;
    if var("ADAM_RAY_ARCHS").map_or(true, |a| a.contains("vulkan")) { if let Ok(mut gt) = Gpu::new(&c) { load_pages(&c, &mut gt).unwrap(); let (gl, cl): (Vec<Vec<f32>>, Vec<Vec<f32>>) = texts.iter().zip(&tr).map(|(t, r)| (gt.logits(t), r[(t.len() - 1) * 256..].to_vec())).unzip(); let (w, _) = row("adam-train gpu vs cpu (last byte)", &gl, &cl); ok &= w < 1e-4; } }
    for arch in var("ADAM_RAY_ARCHS").unwrap_or_else(|_| "cpu,vulkan".into()).split(',') {
        let t1 = Instant::now();
        let mut e = RayField::new(Some(pk.clone()), Some(syn.clone()), None, None, Some(arch.into()), Some(home.clone()));
        e.load().unwrap();
        let lt = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let rr: Vec<Vec<f32>> = texts.iter().map(|t| e.logits_for(t)).collect();
        let rt = t2.elapsed().as_secs_f64();
        let (w, t) = row(&format!("adam_ray {arch} vs adam-train"), &rr, &tr);
        (var("ADAM_V19_DIFF").is_ok() && arch == "cpu").then(|| rr.iter().zip(&tr).enumerate().for_each(|(i, (a, b))| a.chunks(256).zip(b.chunks(256)).enumerate().filter(|(_, (p, q))| p != q).for_each(|(q, (p, r))| println!("  diff text {i} pos {q} {:?} max {:e}", String::from_utf8_lossy(&texts[i][q.saturating_sub(30)..=q]), p.iter().zip(r).fold(0f32, |m, (x, y)| m.max((x - y).abs()))))));
        let gs = (gn > 0).then(|| texts.iter().zip(&tg).filter(|(p, g)| &greedy(&mut |x, c| match c { None => e.prime(x, 1), Some(c) => e.advance(&[0], &[c], true).unwrap() }, p) == *g).count());
        gs.map(|g| println!("  {arch} greedy {gn}-byte decode (prime/advance) identical to adam-train full-recompute greedy: {g}/{}", texts.len()));
        if let Some(ti) = ti.as_ref() { row(&format!("adam_ray {arch} vs taichi trainer"), &rr, ti); }
        let (pp, cc): (Vec<Vec<u8>>, Vec<Vec<u8>>) = texts.iter().filter(|t| t.len() > 8).map(|t| (t[..t.len() - 8].to_vec(), t[t.len() - 8..].to_vec())).unzip();
        let sc = e.score(&pp, &cc);
        let sr: Vec<f64> = pp.iter().zip(&cc).map(|(p, q)| { let full = [p.as_slice(), q].concat(); let lg = e.logits_for(&full); (p.len()..full.len()).map(|i| { let r = &lg[(i - 1) * 256..i * 256]; let m = r.iter().fold(f32::MIN, |m, &x| m.max(x)) as f64; r[full[i] as usize] as f64 - (r.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m) }).sum() }).collect();
        let sd = sc.iter().zip(&sr).fold(0f64, |w, (a, b)| w.max((a - b).abs()));
        let bm = e.beam("User: My sister lives in Denver.\nAdam: Noted.\n\nUser: Where does my sister live?\nAdam:", 4, 3, 40, 16, false);
        let mut s = e.stream("User: My sister lives in Denver.\nAdam: Noted.\n\nUser: Where does my sister live?\nAdam:", 60, 0.6, 8, &["\n".to_string()], false, Some(3), None, 6.0);
        let mut st = String::new();
        while let Some(x) = s.next(&mut e) { st += &x; }
        println!("  {arch} [{}] load {lt:.1}s logits {rt:.2}s | batched score vs sequential max diff {sd:.2e} | beam {bm:?} | stream {st:?}", e.backend);
        ok &= t && if arch == "cpu" { w == 0.0 && sd < 1e-9 } else { w < 1e-4 };
    }
    assert!(ok, "v19 parity failed");
}
