use adam_gf17::{dec, dig, key, unkey, val, write, Store, Table, K4, MID, PW, PX};
use half::f16;
fn sm(x: u64) -> u64 { let mut z = x.wrapping_add(0x9E3779B97F4A7C15); z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^ (z >> 31) }
fn tmp(n: &str) -> String { format!("{}/adam_gf17_{}_{n}.gf17", std::env::var("TMPDIR").unwrap_or_else(|_| std::env::temp_dir().display().to_string()), std::process::id()) }
fn bytes(n: usize, s: u64) -> Vec<u8> { (0..n).map(|i| sm(i as u64 ^ s << 40) as u8).collect() }
fn skew(n: usize, s: u64) -> Vec<u8> { (0..n / 2).flat_map(|i| { let h = sm(i as u64 ^ s << 40); (if h % 7 < 5 { 0u16 } else { f16::from_f32(((h >> 8) % 64) as f32 / 256.0 - 0.125).to_bits() }).to_le_bytes() }).collect() }
fn tabs() -> Vec<Table> { vec![Table { name: "sid".into(), dt: 1, ord: false, exact: true, shape: vec![3, 5001, 3] }, Table { name: "sw".into(), dt: 2, ord: false, exact: false, shape: vec![3, 5001, 3] }, Table { name: "dp".into(), dt: 2, ord: true, exact: false, shape: vec![1, 6_000_000, 3] }, Table { name: "bias".into(), dt: 4, ord: false, exact: true, shape: vec![257] }, Table { name: "odd".into(), dt: 1, ord: false, exact: false, shape: vec![2, 7] }, Table { name: "sk".into(), dt: 2, ord: false, exact: false, shape: vec![2, 300_000] }, Table { name: "zero".into(), dt: 2, ord: true, exact: false, shape: vec![2, 5000] }] }
fn data(tabs: &[Table]) -> Vec<Vec<u8>> { tabs.iter().enumerate().map(|(i, t)| match t.name.as_str() { "sk" => skew(t.n() * t.dt, i as u64), "zero" => vec![0; t.n() * t.dt], _ => bytes(t.n() * t.dt, i as u64) }).collect() }
fn wr(p: &str, tabs: &[Table], data: &[Vec<u8>], tile: usize) { write(p, &tabs.iter().cloned().zip(data.iter().map(|d| d.as_slice())).collect::<Vec<_>>(), tile).unwrap(); }
#[test]
fn codec_exhaustive() {
    for ord in [false, true] { for b in 0..=u16::MAX { let k = key(b, ord); let d: [u8; 4] = std::array::from_fn(|i| dig(k, i)); assert!(d.iter().all(|&x| x < 17) && d[3] <= 13); assert_eq!(unkey(val(d, 4), ord), b); assert_eq!(d.iter().zip(K4).map(|(&x, k)| x as u32 * k).sum::<u32>(), k as u32); } }
    let ks: Vec<u16> = (0..=u16::MAX).map(|b| key(b, true)).collect();
    let fin: Vec<(f32, u16)> = (0..=u16::MAX).map(f16::from_bits).filter(|x| x.is_finite() && x.to_bits() != 0x8000).map(|x| (x.to_f32(), ks[x.to_bits() as usize])).collect();
    let mut s = fin.clone();
    s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    assert!(s.windows(2).all(|w| w[0].1 < w[1].1));
    for n in 1..4 { assert!((0..=u16::MAX).all(|b| { let k = key(b, true); let d: [u8; 4] = std::array::from_fn(|i| dig(k, i)); let a = val(d, n) as i64; (a - k as i64).abs() <= (0..4 - n).map(|i| 8 * K4[i] as i64).sum::<i64>() + 1 })); }
    assert!((0..=u16::MAX).all(|k| (1..4).all(|n| f16::from_bits(dec(k, n, true)).is_finite())));
    assert_eq!(val([0; 4], 0), (0..4).map(|i| MID as u32 * K4[i]).sum::<u32>() as u16);
}
#[test]
fn store_roundtrip_multi_page() {
    let (tabs, data) = (tabs(), data(&tabs()));
    let r = tmp("rt_raw");
    wr(&r, &tabs, &data, 0);
    let raw = Store::open(&r, 1 << 20).unwrap();
    for tile in [0, 64, 256] {
        let p = tmp(&format!("rt{tile}"));
        wr(&p, &tabs, &data, tile);
        let mut s = Store::open(&p, 1 << 20).unwrap();
        assert_eq!((s.tile, s.tabs[2].np()), (tile, 2));
        for (t, d) in data.iter().enumerate() { assert_eq!(&s.unpack(t, 4).unwrap().0, d, "table {} tile {tile}", s.tabs[t].name); }
        for t in 0..tabs.len() { for n in 1..4 { assert!(s.unpack(t, n).unwrap() == raw.unpack(t, n).unwrap(), "tiers {n} table {t} tile {tile}"); } for l in 0..tabs[t].lv() { for p in 0..tabs[t].np() { assert!(s.rgba(t, l, p).unwrap() == raw.rgba(t, l, p).unwrap()); } } }
        for i in 0..4000u64 { let h = sm(i); let (l, c, k) = ((h % 3) as usize, ((h >> 8) % 5001) as usize, ((h >> 40) % 3) as usize); let e = (l * 5001 + c) * 3 + k; assert_eq!(s.fetch(l, c, k).unwrap(), (data[0][e], u16::from_le_bytes([data[1][2 * e], data[1][2 * e + 1]]))); assert!(s.used <= s.budget); }
        let w = s.get(&[(2, 0, PX - 1), (2, 0, PX), (2, 0, 18_000_000 - 1), (5, 1, 299_999), (6, 0, 17)], 4).unwrap();
        assert_eq!(w, [(2, PX - 1), (2, PX), (2, 18_000_000 - 1), (5, 300_000 + 299_999), (6, 17)].map(|(t, i)| u16::from_le_bytes([data[t][2 * i], data[t][2 * i + 1]])));
        assert!(s.ents.iter().filter(|e| e.t == 2 && e.p == 1).all(|e| e.c0 == (PX / 3) as u64 && e.c1 == 6_000_000));
        s.cold();
        assert_eq!(s.used, 0);
        let (c, z) = s.unpack(2, 1).unwrap();
        assert_eq!(z, 0);
        let (lo, hi) = (data[2].chunks(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect::<Vec<_>>(), c.chunks(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect::<Vec<_>>());
        assert!(lo.iter().zip(&hi).filter(|(a, _)| a.is_finite() && a.abs() > 1e-3).all(|(a, b)| a.signum() == b.signum()));
        (tile > 0).then(|| assert!(s.len() * 10 < raw.len() * 6, "coded {} vs raw {}", s.len(), raw.len()));
        std::fs::remove_file(&p).unwrap();
    }
    std::fs::remove_file(&r).unwrap();
}
#[test]
fn coded_tile_random_access() {
    let (tabs, data) = (tabs(), data(&tabs()));
    let p = tmp("ra");
    wr(&p, &tabs, &data, 256);
    let mut s = Store::open(&p, 64 << 20).unwrap();
    let w16 = |t: usize, i: usize| u16::from_le_bytes([data[t][2 * i], data[t][2 * i + 1]]);
    let tr = |t: usize, v: u16, n: usize| if tabs[t].exact { v } else { dec(val(std::array::from_fn(|k| dig(key(v, tabs[t].ord), k)), n), n, tabs[t].ord) };
    for (t, l, w) in [(1usize, 0usize, 0usize), (1, 2, 15002), (2, 0, PX + 7 * PW + 4095), (2, 0, 18_000_000 - 1), (5, 1, 12345), (0, 0, 7000), (6, 1, 4999)] {
        let x = w16(t, l * tabs[t].words() + w);
        s.reset(0);
        let (m0, h0) = (s.miss, s.hits);
        assert_eq!((s.get(&[(t, l, w)], 4).unwrap()[0], s.miss - m0, s.used), (x, 1, 4 * 256 * 256));
        assert_eq!((s.get(&[(t, l, w)], 1).unwrap()[0], s.hits - h0, s.miss - m0), (tr(t, x, 1), 1, 1));
        s.reset(0);
        let m1 = s.miss;
        assert_eq!((s.get(&[(t, l, w)], 1).unwrap()[0], s.get(&[(t, l, w)], 2).unwrap()[0], s.get(&[(t, l, w)], 4).unwrap()[0], s.miss - m1, s.used), (tr(t, x, 1), tr(t, x, 2), x, 1, 4 * 256 * 256));
    }
    let q: Vec<(usize, usize, usize)> = (0..2000u64).map(|i| { let h = sm(i ^ 77); (2usize, 0usize, (h % 18_000_000) as usize) }).collect();
    assert!(s.get(&q, 4).unwrap().iter().zip(&q).all(|(&v, &(t, _, w))| v == w16(t, w)));
    std::fs::remove_file(&p).unwrap();
}
#[test]
fn coded_corruption_detected() {
    let (tabs, data) = (tabs(), data(&tabs()));
    let p = tmp("cx");
    wr(&p, &tabs, &data, 128);
    let orig = std::fs::read(&p).unwrap();
    let (ent, tile_at, ft_at, dir_at) = { let s = Store::open(&p, 1 << 20).unwrap(); let e = s.ent(1, 1, 0, 2); let q = s.pl(e).unwrap(); let o = s.ents[e].off as usize; (e, o + (q.off[1] + q.off[2]) as usize / 2, o + 8 + 2 * 17 * 3, u64::from_le_bytes(orig[24..32].try_into().unwrap()) as usize + 9) };
    let flip = |at: usize| { let mut b = orig.clone(); b[at] ^= 0x10; std::fs::write(&p, &b).unwrap(); };
    flip(tile_at);
    let mut s = Store::open(&p, 1 << 20).unwrap();
    assert!(s.get(&[(1, 1, 128)], 4).is_err() && s.get(&[(1, 1, 128)], 1).is_ok() && s.get(&[(1, 1, 0)], 4).is_ok() && s.get(&[(1, 1, 256)], 4).is_ok());
    assert!(s.unpack(1, 4).is_err() && s.unpack(1, 1).is_ok() && s.unpack(0, 4).is_ok() && s.rgba(1, 1, 0).is_err());
    drop(s);
    flip(ft_at);
    let mut s = Store::open(&p, 1 << 20).unwrap();
    assert!(s.pl(ent).is_err() && s.get(&[(1, 1, 0)], 4).is_err() && s.get(&[(1, 1, 0)], 1).is_ok());
    drop(s);
    flip(dir_at);
    assert!(Store::open(&p, 1 << 20).is_err());
    flip(20);
    assert!(Store::open(&p, 1 << 20).is_err());
    std::fs::write(&p, &orig).unwrap();
    let s = Store::open(&p, 1 << 20).unwrap();
    assert!((0..tabs.len()).all(|t| s.unpack(t, 4).unwrap().0 == data[t]));
    std::fs::remove_file(&p).unwrap();
}
#[test]
fn checkpoint_roundtrip() {
    let Ok(pre) = std::env::var("ADAM_GF17_PREFIX") else { return eprintln!("ADAM_GF17_PREFIX unset, skipping"); };
    for tile in [0, 256] {
        let p = tmp(&format!("ckpt{tile}"));
        let tabs = adam_gf17::pack(&pre, &p, false, tile).unwrap();
        let s = Store::open(&p, 256 << 20).unwrap();
        for (t, tb) in tabs.iter().enumerate() { let z = adam_ray::npy::Npz::open(&format!("{pre}_{}.npy", tb.name)).unwrap(); let a = z.get("").unwrap(); assert!(s.unpack(t, 4).unwrap().0 == a.data[..a.n() * tb.dt], "{} not bit-exact (tile {tile})", tb.name); }
        std::fs::remove_file(&p).unwrap();
    }
}
