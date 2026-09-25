use adam_qa::{pick, Qa};
use adam_ray::RayField;
use std::path::Path;
fn home() -> String { std::env::var("AMNI_HOME").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../..").to_string()) }
fn hex(s: &str) -> Vec<u8> { (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect() }
#[test]
fn qa_parity() {
    let (rp, ix) = (format!("{}/qa.tsv", std::env::var("ADAM_RAY_REF").unwrap_or_default()), format!("{}/exports/gf17_continuum/ray_qa_index.npz", home()));
    if !Path::new(&rp).exists() || !Path::new(&ix).exists() { eprintln!("skip qa: reference or index missing (set AMNI_HOME, ADAM_RAY_REF)"); return; }
    let t0 = std::time::Instant::now();
    let qa = Qa::load(Some(&ix)).unwrap();
    let lt = t0.elapsed().as_secs_f64();
    let mut e = RayField::new(Some(format!("{}/exports/gf17_continuum/rayhash_clean1v2_base_v18_s100k_sid.npy", home())), Some(format!("{}/data/clean1/synsets.txt", home())), Some(format!("{}/data/clean1/words.txt", home())), Some(format!("{}/data/clean1/wordcounts.txt", home())), Some(std::env::var("ADAM_RAY_ARCH").unwrap_or_else(|_| "cpu".into())), Some(home()));
    e.load().unwrap();
    let (mut n, mut ss, mut pk, t1) = (0, 0, 0, std::time::Instant::now());
    for l in std::fs::read_to_string(&rp).unwrap().lines() {
        let f: Vec<&str> = l.split('\x1f').collect();
        let want: Vec<f64> = f[1].split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect();
        let got: Vec<f64> = qa.search(f[0], 20).iter().map(|h| h.score).collect();
        ss += (want.len() == got.len() && want.iter().zip(&got).all(|(a, b)| (a - b).abs() <= 1e-5 * a.abs().max(1.0))) as usize;
        let p = pick(&mut e, &qa, f[0]);
        pk += (p.as_ref().map(|h| (h.a.clone(), h.cov)).map_or(f[2].is_empty(), |(a, c)| a == hex(f[2]) && (c - f[3].parse::<f64>().unwrap()).abs() <= 1e-9)) as usize;
        n += 1;
    }
    println!("qa [{}] index load {lt:.1}s | search top-20 scores equal {ss}/{n} | pick answer+coverage equal {pk}/{n} | {:.2}s", e.backend, t1.elapsed().as_secs_f64());
    assert!(ss == n && pk == n);
}
