use adam_ray::{pack::scale16, rng::{pairwise, Pcg64}};
#[test]
fn pcg64_matches_numpy_default_rng() {
    let r: Vec<f64> = { let mut g = Pcg64::new(Some(11)); (0..3).map(|_| g.random()).collect() };
    assert!(r.iter().zip([0.1285702, 0.49927786, 0.60149836]).all(|(a, b)| (a - b).abs() < 5e-8), "{r:?}");
    assert_eq!(Pcg64::new(Some(1234)).integers(0, 35995592 - 129, 3), vec![35252185, 35156760, 35561378]);
    assert_eq!(Pcg64::new(Some(0)).integers(0, 35995592 - 129, 3), vec![30618612, 22927730, 18398594]);
}
#[test]
fn v16_scale_levels_match_numpy_after_f16() {
    let want: [u32; 16] = [0x00000000, 0x3C042000, 0x3C50C9E3, 0x3CA4F7B4, 0x3D025805, 0x3D4DF94F, 0x3DA2BE5E, 0x3E00962C, 0x3E4B3276, 0x3EA08CB5, 0x3EFDB4CD, 0x3F487533, 0x3F9E629D, 0x3FFA4933, 0x4045C15B, 0x409C4001];
    let h = |v: f32| [half::f16::from_f32(0.5 * v).to_bits(), half::f16::from_f32(v).to_bits()];
    assert_eq!(scale16(-6.954240322113037, 2.287712335586548).map(h), want.map(|b| h(f32::from_bits(b))));
}
#[test]
fn pairwise_matches_numpy_blocking() { let a: Vec<f64> = (0..300).map(|i| 1.0 / (i as f64 + 1.0)).collect(); assert!((pairwise(&a) - a.iter().sum::<f64>()).abs() < 1e-12); }
