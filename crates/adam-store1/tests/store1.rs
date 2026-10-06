use adam_store1::*;
use adam_torus::sm;
use std::collections::HashMap;
fn tmp(n: &str) -> String { let d = format!("{}/adam_store1_{}_{n}", std::env::var("TMPDIR").unwrap_or_else(|_| std::env::temp_dir().display().to_string()), std::process::id()); std::fs::create_dir_all(&d).unwrap(); d }
#[test]
fn huffman_kraft_and_canon() {
    let mut r = 3u64;
    let f: Vec<u64> = (0..5000).map(|i| 1 + (sm(&mut r) % 1000) * (i % 7) as u64).collect();
    let l = hlen(&f);
    let k: f64 = l.iter().map(|&x| 17f64.powi(-(x as i32))).sum();
    assert!(k <= 1.0 + 1e-12);
    let (cnt, at) = canon(&l);
    let h = Hc::new(cnt, 0);
    let mut seen = std::collections::HashSet::new();
    at.iter().for_each(|&(li, rr)| { let c = h.code(li, rr); assert_eq!(c.len(), li + 1); assert!(seen.insert(c)) });
}
#[test]
fn codebook_tuple_and_list_roundtrip() {
    let mut r = 9u64;
    let mut ts: Vec<Vec<u64>> = (0..700).map(|i| if i % 3 == 0 { vec![sm(&mut r) % 50, sm(&mut r)] } else { (0..(sm(&mut r) % 9) as usize).map(|_| sm(&mut r) % 300).collect() }).collect();
    ts.sort();
    ts.dedup();
    let mut o = vec![];
    let mut st = vec![];
    ts.iter().enumerate().for_each(|(i, t)| { if i % BLK == 0 { st.push(o.len()) } cbenc(&mut o, if i % BLK == 0 { &[] } else { &ts[i - 1] }, t) });
    let mut w = PgW::new();
    w.add("cb", &o);
    let p = format!("{}/cb.gf17pg", tmp("cb"));
    let (_, ok, _) = w.save(&p).unwrap();
    assert!(ok);
    let g = Pg::open(&p).unwrap();
    let base = g.sec("cb").0;
    ts.iter().enumerate().for_each(|(i, t)| { let mut rd = g.rd(base + st[i / BLK] as u64); let mut pv = vec![]; (0..=i % BLK).for_each(|_| cbdec(&mut rd, &mut pv)); assert_eq!(&pv, t) });
}
#[test]
fn cascade_and_text_roundtrip() {
    let d = tmp("cas");
    let mut r = 5u64;
    let rows: Vec<(u64, usize, Vec<u64>)> = (0..40_000u64).map(|i| { let c = (i % 3) as usize; let v = (sm(&mut r) % 1000).pow(2) / 997; (i * 7919 + 13, c, vec![v, i % 2]) }).collect();
    let mut m: Vec<HashMap<Vec<u64>, u64>> = vec![HashMap::new(); 3];
    rows.iter().for_each(|(_, c, t)| *m[*c].entry(t.clone()).or_default() += 1);
    let mut b = CasB::new("t", m);
    rows.iter().for_each(|(k, c, t)| b.push(*k, *c, t));
    let mut w = PgW::new();
    b.save(&d, &mut w).unwrap();
    let recs: Vec<Vec<(u8, Vec<u16>)>> = (0..500).map(|i| rec_a((i % 4) as u8, format!("Label {i} é").as_bytes(), format!("x{i}").as_bytes())).collect();
    let rb: Vec<Vec<(u8, Vec<u16>)>> = (0..500).map(|i| rec_b(&(0..i % 3).map(|j| ((j % 3) as u8, format!("alias {j} {i}").into_bytes())).collect::<Vec<_>>())).collect();
    let (tm, tmd) = Tm::build(&[&recs, &rb], 3);
    w.add("tm", &tmd);
    let (s, o) = blocks(&tm, &recs, TB);
    w.bytes("ta", &s);
    w.bytes("ta_x", &o);
    let p = format!("{d}/store1.gf17pg");
    assert!(w.save(&p).unwrap().1);
    let g = Pg::open(&p).unwrap();
    let c = Cas::open(&d, "t", &g).unwrap();
    assert!(rows.iter().all(|(k, cc, t)| c.get(&g, *k, *cc).as_ref() == Some(t)));
    let tm2 = Tm::load(&g, g.sec("tm").0);
    (0..500u64).for_each(|e| { let b = e / TB as u64; let mut dd = Dec::new(&g, g.sec("ta").0, g.u32at(g.sec("ta_x").0, b) as u64, &tm2); let mut last = (0u16, vec![], vec![]); for _ in 0..=e % TB as u64 { let tk = dd.field(0)[0]; let l = dd.field(1); let x = if tk >= 2 { dd.field(2) } else { vec![] }; last = (tk, l, x) } let want = &recs[e as usize]; assert_eq!(last.0, want[0].1[0]); assert_eq!(last.1, want[1].1); if last.0 >= 2 { assert_eq!(last.2, want[2].1) } });
}
#[test]
fn rans17_symbols_uniform_roundtrip() {
    use adam_store1::s2::*;
    let mut r = 11u64;
    let f = Fq::new(&(0..300u32).map(|i| (i, 1 + (sm(&mut r) % 1000) * (i % 5) as u64)).chain([(ESC, 7)]).collect::<Vec<_>>());
    let xs: Vec<(u32, u64, u64)> = (0..20000).map(|_| { let s = f.sym[(sm(&mut r) % f.sym.len() as u64) as usize]; let n = 1 + sm(&mut r) % (1u64 << (sm(&mut r) % 40)); (s, n, sm(&mut r) % n) }).collect();
    let mut e = Enc::default();
    xs.iter().for_each(|&(s, n, u)| { assert!(f.put(&mut e, s)); e.uni(n, u) });
    let d = e.finish();
    assert!(d.iter().all(|&x| x < 17));
    let mut dc = RDec::new(&d, 0);
    xs.iter().for_each(|&(s, n, u)| { assert_eq!(dc.get(&f), s); assert_eq!(dc.uni(n), u) });
    assert!((d.len() as f64) < e.bits() / 4.087 * 1.01 + 8.0);
}
