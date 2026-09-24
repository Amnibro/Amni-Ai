use adam_ray::{npy::Npz, RayField};
use std::{path::Path, time::Instant};
const TEXTS: [&[u8]; 5] = [b"The weather today is sunny, and the kaolinite clay mineral forms in warm, humid soils.", b"def add(a, b):\n    return a + b\n\n```python\nprint(add(2, 3))\n```\nDone.", b"  Indented line; with (punct) {braces} and \"quotes\" -- plus a\ttab.\nNext line!", b"Photosynthesis converts light energy into chemical energy stored in glucose molecules within the plant.", b"User: Why do volcanoes erupt near tectonic boundaries?\nAdam: Volcanoes erupt where magma rises through cracks in the crust.\n\nUser: Who is Adam?\nAdam: I am Adam."];
const PROMPTS: [&[u8]; 3] = [b"User: What is the capital of France?\nAdam: ", b"User: \nAdam: ", b"User: How do bees make honey?\nAdam: "];
const CONTS: [&[u8]; 3] = [b"Paris is the capital of France.", b"Bees collect nectar from flowers and store it in the hive.", b"I am not sure."];
fn home() -> String { std::env::var("AMNI_HOME").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../..").to_string()) }
fn pack(tag: &str) -> String { format!("{}/exports/gf17_continuum/{}", home(), if tag == "v16" { "rayhash_v16_big2_L21_e20_hop4_q5_s150k_pk_q5_i8.npz" } else { "rayhash_v18_chat2_qt_s20k_sid.npy" }) }
fn open(tag: &str, arch: &str) -> RayField { let h = home(); let mut e = RayField::new(Some(pack(tag)), None, None, None, Some(arch.into()), Some(h)); e.load().unwrap(); e }
fn stop() -> Vec<String> { ["\n\n", "\nUser:", "\nAdam:"].map(String::from).to_vec() }
fn drain(e: &mut RayField, prompt: &str, guide: Option<&[u8]>) -> String { let mut s = e.stream(prompt, 160, 0.6, 8, &stop(), true, Some(11), guide, 6.0); let mut o = String::new(); while let Some(c) = s.next(e) { o += &c; } o }
fn check(tag: &str) {
    let rp = format!("{}/{tag}.npz", std::env::var("ADAM_RAY_REF").unwrap_or_default());
    if !Path::new(&pack(tag)).exists() || !Path::new(&rp).exists() { eprintln!("skip {tag}: pack or reference missing (set AMNI_HOME, ADAM_RAY_REF)"); return; }
    let z = Npz::open(&rp).unwrap();
    for arch in std::env::var("ADAM_RAY_ARCHS").unwrap_or_else(|_| "cpu,vulkan".into()).split(',') {
        let t0 = Instant::now();
        let mut e = open(tag, arch);
        let lt = t0.elapsed().as_secs_f64();
        let (mut worst, mut agree, mut n) = (0f64, 0usize, 0usize);
        for (i, t) in TEXTS.iter().enumerate() {
            let (a, b) = (e.logits_for(t), z.get(&format!("t{i}")).unwrap().f32s());
            worst = a.iter().zip(&b).fold(worst, |w, (x, y)| w.max((x - y).abs() as f64));
            let am = |v: &[f32]| v.iter().enumerate().fold(0, |m, (i, &x)| if x > v[m] { i } else { m });
            agree += a.chunks(256).zip(b.chunks(256)).filter(|(x, y)| am(x) == am(y)).count();
            n += t.len();
        }
        let (pp, cc): (Vec<Vec<u8>>, Vec<Vec<u8>>) = PROMPTS.iter().flat_map(|p| CONTS.iter().map(move |c| (p.to_vec(), c.to_vec()))).unzip();
        let sd = e.score(&pp, &cc).iter().zip(z.get("score").unwrap().vec::<f64>()).fold(0f64, |w, (a, b)| w.max((a - b).abs()));
        let (st, gd, bm) = (drain(&mut e, "User: What is the capital of France?\nAdam:", None), drain(&mut e, "User: How do bees make honey?\nAdam:", Some(b" Bees collect nectar from flowers and store it in the hive.")), e.beam("User: What is gravity?\nAdam:", 4, 3, 40, 16, true));
        let txt = |k: &str| String::from_utf8(z.get(k).unwrap().data[..z.get(k).unwrap().n()].to_vec()).unwrap();
        let bp = e.bpb(Some(&format!("{}/data/big2/hold.bin", home())), 1, 32, 16, 1234).unwrap() - z.get("bpb").unwrap().vec::<f64>()[0];
        println!("{tag} {arch:>6} [{}] load {lt:.1}s | max|dlogit| {worst:.3e} | top1 {agree}/{n} | max|dscore| {sd:.3e} | stream {} | guided {} | beam {} | dbpb {bp:.2e}", e.backend, st == txt("stream"), gd == txt("guided"), bm == txt("beam"));
        assert!(worst < 1e-3 && agree == n, "{tag} {arch} logit parity failed");
    }
}
#[test]
fn parity_v16() { check("v16"); }
#[test]
fn parity_v18() { check("v18"); }
