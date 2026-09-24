use adam_ray::{npy::Npz, pack};
use half::f16;
use rayon::prelude::*;
fn check(tag: &str) {
    let (Ok(pre), Ok(home)) = (std::env::var("ADAM_RAY_TABLES"), std::env::var("AMNI_HOME")) else { eprintln!("skip {tag}: set ADAM_RAY_TABLES (dump prefix from crates/adam-ray-py/tests/ray_ref.py tables_<tag>) and AMNI_HOME"); return; };
    let pk = format!("{home}/exports/gf17_continuum/{}", if tag == "v16" { "rayhash_v16_big2_L21_e20_hop4_q5_s150k_pk_q5_i8.npz" } else { "rayhash_v18_chat2_qt_s20k_sid.npy" });
    let tb = pack::load(&pk, &format!("{home}/data/big2/synsets.txt"), &format!("{home}/data/big2/wordcounts.txt")).unwrap();
    let z = |k: &str| Npz::open(&format!("{pre}_{tag}_{k}.npy")).unwrap();
    let (sid, sw, dp, bias) = (z("sid"), z("sw"), z("dp"), z("bias"));
    let (sid, sw, dp, bias) = (sid.get("").unwrap().vec::<u8>(), sw.get("").unwrap().vec::<u16>(), dp.get("").unwrap().f32s(), bias.get("").unwrap().f32s());
    let bad = tb.cells.par_iter().enumerate().filter(|&(c, &x)| (0..3).any(|s| { let (i, v) = pack::tap(x, s, tb.geo.copy > 0); i as u8 != sid[c * 3 + s] || v != f16::from_bits(sw[c * 3 + s]).to_f32() })).count();
    let dbad = (0..dp.len()).filter(|&i| f16::from_bits(tb.dp[i]).to_f32() != dp[i]).count();
    println!("{tag}: cells {} mismatched {bad} | dp {} mismatched {dbad} | bias equal {}", tb.cells.len(), dp.len(), tb.bias == bias);
    assert!(bad == 0 && dbad == 0 && tb.bias == bias);
}
#[test]
fn tables_v16_exact() { check("v16"); }
#[test]
fn tables_v18_exact() { check("v18"); }
