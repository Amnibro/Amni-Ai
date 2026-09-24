use adam_ray::npy::Npz;
use adam_train::{cfg::Cfg, engine, train};
fn steps(p: &str) -> Vec<f64> { std::fs::read_to_string(p).unwrap().lines().filter(|l| !l.starts_with("eval")).map(|l| l.split_whitespace().nth(1).unwrap().parse().unwrap()).collect() }
fn arr(d: &str, k: &str) -> Vec<f32> { Npz::open(&format!("{d}/{k}.npy")).unwrap().get("").unwrap().f32s() }
#[test]
fn parity() {
    let Ok(rf) = std::env::var("ADAM_TRAIN_REF") else { return eprintln!("skip: set ADAM_TRAIN_REF to a tests/taichi_ref.py output dir and the same trainer env (AMNI_HOME, CORPUS, LEVELS, ...)") };
    let rs = steps(&format!("{rf}/steps.txt"));
    for arch in std::env::var("ADAM_TRAIN_ARCHS").unwrap_or_else(|_| "cpu,vulkan".into()).split(',') {
        let mut c = Cfg::env();
        let d = std::env::temp_dir().join(format!("adam_train_parity_{arch}")).to_string_lossy().into_owned();
        (c.arch, c.step_log, c.dump, c.out, c.thf) = (arch.into(), format!("{d}/steps.txt"), d.clone(), format!("{d}/rayhash_parity.json"), std::env::var("THROTTLE_FILE").unwrap_or_default());
        std::fs::create_dir_all(&d).unwrap();
        let mut e = engine(&c);
        let ls = train(&c, e.as_mut()).unwrap();
        let rel = ls.iter().zip(&rs).map(|(a, b)| ((a - b) / b).abs()).fold(0f64, f64::max);
        let ex = ls.iter().zip(&rs).filter(|(a, b)| a == b).count();
        let td: Vec<(&str, usize)> = ["dp", "dm", "dv", "sid", "sw", "sm", "sv", "bias", "lq"].iter().map(|k| (*k, arr(&d, k).iter().zip(arr(&rf, k)).filter(|(a, b)| **a != *b).count())).collect();
        println!("{arch} [{}]: {ex}/{} steps exact, max rel loss diff {rel:.3e}, differing table values {td:?}", e.name(), rs.len());
        let _ = std::fs::remove_dir_all(&d);
        assert!(if arch == "cpu" { ex == rs.len() && td.iter().all(|t| t.1 == 0) } else { rel < 1e-2 }, "{arch} parity failed");
    }
}
