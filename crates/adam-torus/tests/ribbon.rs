use adam_torus::{keyof, layered, sm, Live, Table, TOMB};
fn tmp(n: &str) -> String { format!("{}/adam_torus_{}_{n}", std::env::var("TMPDIR").unwrap_or_else(|_| std::env::temp_dir().display().to_string()), std::process::id()) }
fn keys(n: usize, s: u64) -> Vec<(u64, u32)> { let mut r = s; (0..n).map(|i| (keyof(i as u32 % 50_000, (i / 50_000) as u16, (sm(&mut r) % 3) as u16 + 4 * (i as u16 % 7)), (sm(&mut r) % 11_900_000) as u32)).collect::<std::collections::HashMap<u64, u32>>().into_iter().collect() }
#[test]
fn layered_exact_fp_overlay() {
    let (ks, p, o) = (keys(300_000, 1), tmp("base"), tmp("ov"));
    let (b0, b1, by) = layered::<9>(&p, &ks, 64, 3, 65536, -0.03, 17, Some((32, [0, 8, 16, 32])), 0.08).unwrap();
    assert!(!b0.bumped.is_empty() && b1.is_some());
    let t = Table::<9>::open(&p).unwrap();
    assert_eq!(t.bytes, by);
    assert!(ks.iter().all(|&(k, v)| t.get(k) == Some(v)));
    let mut r = 5u64;
    let fp = (0..200_000).filter(|_| t.get(keyof((sm(&mut r) % 50_000) as u32, 900 + (sm(&mut r) % 50) as u16, 0)).is_some()).count();
    assert!(fp < 200_000 / 4913 * 2 + 20, "fp {fp}");
    assert!((t.all_cells() as f64) < ks.len() as f64 * 1.03);
    let ov: Vec<(u64, u32)> = vec![(ks[0].0, TOMB), (ks[1].0, 77), (keyof(16_000_000, 31, 0), 5), (keyof(16_000_000, 31, 1), 6)];
    layered::<18>(&o, &ov, 64, 12, 65536, 0.08, 23, None, 0.0).unwrap();
    let lv = Live::<9, 18> { base: t, ov: Some(Table::<18>::open(&o).unwrap()) };
    assert_eq!((lv.get(ks[0].0), lv.get(ks[1].0), lv.values(16_000_000, 31)), (None, Some(77), vec![5, 6]));
    assert!(ks[2..].iter().all(|&(k, v)| lv.get(k) == Some(v)));
    [p.clone(), format!("{p}.l1"), o].iter().for_each(|f| { let _ = std::fs::remove_file(f); });
}
