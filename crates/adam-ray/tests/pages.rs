use adam_ray::{Dec, RayField};
use std::path::Path;
fn run(pk: &str, home: &str, d: &[u8], prompt: &str) -> (Vec<u32>, String) {
    let lx = format!("{home}/data/clean1");
    let mut e = RayField::new(Some(pk.into()), Some(format!("{lx}/synsets.txt")), Some(format!("{lx}/words.txt")), Some(format!("{lx}/wordcounts.txt")), Some("cpu".into()), Some(home.into()));
    e.load().unwrap();
    let lg = e.logits_for(d).iter().map(|x| x.to_bits()).collect();
    let mut s = e.stream(prompt, 200, 1.0, 1, &[], false, Some(7), None, 0.0, Dec::default());
    let mut o = String::new();
    while let Some(c) = s.next(&mut e) { o.push_str(&c); }
    (lg, o)
}
#[test]
fn gf17_pages_bit_exact() {
    let home = std::env::var("AMNI_HOME").unwrap_or_default();
    let hold = std::fs::read(format!("{home}/data/clean1/hold.bin")).unwrap_or_default();
    let packs: Vec<String> = std::env::var("ADAM_RAY_PAGES_PACKS").unwrap_or_else(|_| "rayhash_clean1v2_base_v18_s100k,rayhash_v19e_clean_chat4e_r4_s30k,rayhash_v19d_clean_chat6c_s60k".into()).split(',').map(|p| format!("{home}/exports/gf17_continuum/{p}")).filter(|p| Path::new(&format!("{p}_sid.npy")).exists() && Path::new(&format!("{p}.gf17")).exists()).collect();
    if hold.len() < 1 << 21 || packs.is_empty() { return eprintln!("skip: set AMNI_HOME with data/clean1/hold.bin and exports/gf17_continuum/<pack>_*.npy + <pack>.gf17 (pack: target/release/examples/pages pack <prefix>)"); }
    let d = &hold[1 << 20..(1 << 20) + 2000];
    for p in &packs {
        let q = if p.contains("v19") { "User: What is the capital of France?\nAdam:" } else { "The history of the" };
        let a = run(&format!("{p}_sid.npy"), &home, d, q);
        for mb in ["256", "48"] {
            std::env::set_var("AMNI_RAY_PAGE_BUDGET_MB", mb);
            let b = run(&format!("{p}.gf17"), &home, d, q);
            let bad = a.0.iter().zip(&b.0).filter(|(x, y)| x != y).count();
            println!("{p} budget {mb} MB: {} logits, {bad} differ | decode equal {} {:?}", a.0.len(), a.1 == b.1, &b.1[..b.1.len().min(60)]);
            assert!(bad == 0 && a.0.len() == b.0.len() && a.1 == b.1, "{p} budget {mb}: gf17 pages diverge from npy");
        }
        std::env::remove_var("AMNI_RAY_PAGE_BUDGET_MB");
    }
}
