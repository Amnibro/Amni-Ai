use adam_ray::{npy::Npz, pack::lexicon};
use adam_train::{cfg::Cfg, npy, v19::Drv19};
fn crc32(b: &[u8]) -> i64 { !b.iter().fold(!0u32, |c, &x| (0..8).fold(c ^ x as u32, |c, _| if c & 1 != 0 { (c >> 1) ^ 0xEDB88320 } else { c >> 1 })) as i64 }
#[test]
fn v19_driver() {
    let Ok(rf) = std::env::var("ADAM_V19_DRV") else { return eprintln!("skip: set ADAM_V19_DRV to a tests/v19_drv_ref.py npz and VER=19 COPY=13 LEVELS=21 HOPS=4 QT=1 LOG2_ENTRIES=16 CTX=8192 BS=8 SYNSETS=...") };
    let c = Cfg::env();
    assert!(c.ver == 19 && c.bs == 8);
    let z = Npz::open(&rf).unwrap();
    let (x, ls, nbs, crc) = (z.get("x").unwrap(), z.get("L").unwrap().i64s(), z.get("nb").unwrap().i64s(), z.get("crc").unwrap().i64s());
    let tn = x.shape[2];
    let mem: Vec<(&str, Vec<i64>)> = ["ent", "rel", "vlen", "stamp", "val", "cand", "qent", "qrel", "qtyp"].iter().map(|k| (*k, z.get(k).unwrap().i64s())).collect();
    let (lex, mut d, lv, cp) = (lexicon(&c.syn, Some(&c.wcnt)), Drv19::new(c.bs), c.lv, c.copy);
    println!("stopword buckets {}", d.lx.stop.iter().filter(|&&v| v > 0).count());
    let (mut bad, mut mbad, xs, mut dump) = (0, 0, &x.data[..x.n()], [vec![], vec![], vec![]]);
    for (b, (&l, &nb)) in ls.iter().zip(&nbs).enumerate() {
        let (l, nb) = (l as usize, nb as usize);
        let xb: Vec<u8> = (0..nb).flat_map(|r| xs[(b * 8 + r) * tn..(b * 8 + r) * tn + l].iter().copied()).collect();
        let (mut pi, mut pw, mut cpb) = (vec![0u32; nb * l * lv * 4], vec![0f32; nb * l * lv * 4], vec![0i32; nb * l * cp]);
        d.run(&c, &lex, &xb, l, 0, true, &mut pi, &mut pw, &mut cpb);
        let idx: Vec<u8> = pi.iter().zip(&pw).flat_map(|(&i, &w)| (if w != 0.0 { i as i32 } else { 0 }).to_le_bytes()).collect();
        let got = [crc32(&idx), crc32(bytemuck::cast_slice(&pw)), crc32(bytemuck::cast_slice(&cpb))];
        let want = &crc[b * 3..b * 3 + 3];
        bad += (got != want) as usize;
        let rows: Vec<Vec<i64>> = d.rows[..8].iter().map(|r| &r.m).map(|m| [m.ent.iter().map(|&v| v as i64).collect::<Vec<_>>(), m.rel.iter().map(|&v| v as i64).collect(), m.vlen.iter().map(|&v| v as i64).collect(), m.stamp.iter().map(|&v| v as i64).collect(), m.val.iter().flatten().map(|&v| v as i64).collect(), m.cand.iter().map(|&v| v as i64).collect(), vec![m.qent as i64], vec![m.qrel as i64], vec![m.qtyp as i64]].concat()).collect();
        let exp: Vec<Vec<i64>> = (0..8).map(|r| mem.iter().flat_map(|(_, v)| { let per = v.len() / (ls.len() * 8); v[(b * 8 + r) * per..(b * 8 + r + 1) * per].to_vec() }).collect()).collect();
        let mb = (0..8).filter(|&r| rows[r] != exp[r]).count();
        d.rows[..8].iter().map(|r| &r.m).for_each(|m| { dump[0].extend(m.cand.iter().map(|&v| v as f32)); dump[1].extend(m.val.iter().flatten().map(|&v| v as f32)); dump[2].extend(m.vlen.iter().map(|&v| v as f32)); });
        mbad += mb;
        if got != want || mb > 0 { println!("batch {b} L={l}: plan crc {got:?} vs {want:?}, memory rows differing {mb}"); }
    }
    println!("{} batches: plan mismatches {bad}, memory row mismatches {mbad}", ls.len());
    if let Ok(o) = std::env::var("ADAM_V19_DUMP") { std::fs::create_dir_all(&o).unwrap(); let nb = ls.len(); for (k, v, sh) in [("cand", &dump[0], vec![nb, 8, 12]), ("val", &dump[1], vec![nb, 8, 64, 24]), ("vlen", &dump[2], vec![nb, 8, 64])] { npy(&format!("{o}/{k}.npy"), "<f4", &sh, bytemuck::cast_slice(v)).unwrap(); } }
    assert!(bad == 0 && mbad == 0);
}
