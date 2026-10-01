use adam_gf17::{npy, rss};
use adam_torus::{keyof, layered, sm, solve, unkey, Live, Table, Tab, TOMB, VD};
use rayon::prelude::*;
use std::{collections::HashSet, fs, io, time::Instant};
const R: usize = 9;
const Q: usize = 18;
fn arg(a: &[String], k: &str, d: &str) -> String { a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned().unwrap_or(d.into()) }
fn col<T: Send>(dir: &str, n: &str, f: fn(&[u8]) -> T) -> io::Result<Vec<T>> { let (d, _, m, o) = npy(&format!("{dir}/{n}.npy"))?; let b: usize = d[2..].parse().unwrap(); Ok(m[o..].par_chunks_exact(b).map(f).collect()) }
fn facts(dir: &str) -> io::Result<Vec<(u64, u32)>> {
    let (s, p, k, v) = (col(dir, "f_s", |c| u32::from_le_bytes(c.try_into().unwrap()))?, col(dir, "f_p", |c| u16::from_le_bytes(c.try_into().unwrap()))?, col(dir, "f_k", |c| c[0])?, col(dir, "f_v", |c| u64::from_le_bytes(c.try_into().unwrap()))?);
    let (mut o, mut cur, mut cnt) = (Vec::with_capacity(46_000_000), u32::MAX, Vec::<(u16, u16)>::new());
    for i in 0..s.len() {
        (s[i] != cur).then(|| { cur = s[i]; cnt.clear(); });
        if k[i] != 1 { continue; }
        let j = cnt.iter().position(|x| x.0 == p[i]).unwrap_or_else(|| { cnt.push((p[i], 0)); cnt.len() - 1 });
        assert!(s[i] < 1 << 24 && v[i] < TOMB as u64 && cnt[j].1 < u16::MAX);
        o.push((keyof(s[i], p[i], cnt[j].1), v[i] as u32));
        cnt[j].1 += 1;
    }
    Ok(o)
}
fn js(kv: &[(&str, String)]) -> String { format!("{{{}}}", kv.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect::<Vec<_>>().join(",")) }
fn bench<const N: usize>(t: &Table<N>, ks: &[u64]) -> (f64, u64) { let t0 = Instant::now(); let x = ks.iter().fold(0u64, |a, &k| a.wrapping_add(t.get(k).unwrap_or(7) as u64)); (ks.len() as f64 / t0.elapsed().as_secs_f64(), x) }
fn absent(f: &[(u64, u32)], n: usize, seed: u64) -> (Vec<u64>, Vec<u64>) {
    let g = |k: u64| (k & 0xFF_FFFF_FFFF, k >> 40);
    let mut ks: Vec<u64> = f.iter().map(|x| x.0).collect();
    ks.par_sort_unstable_by_key(|&k| g(k));
    let mut nx: Vec<u64> = ks.par_windows(2).filter(|w| g(w[0]).0 != g(w[1]).0).map(|w| w[0] + (1 << 40)).chain(rayon::iter::once(ks[ks.len() - 1] + (1 << 40))).collect();
    let props: Vec<u16> = { let mut v: Vec<u16> = f.iter().map(|x| unkey(x.0).1).collect::<HashSet<_>>().into_iter().collect(); v.sort(); v };
    let mut r = seed;
    (0..n.min(nx.len())).for_each(|i| { let j = i + (sm(&mut r) as usize % (nx.len() - i)); nx.swap(i, j) });
    nx.truncate(n);
    let mut np = Vec::with_capacity(n);
    while np.len() < n { let (s, p) = (unkey(f[sm(&mut r) as usize % f.len()].0).0, props[sm(&mut r) as usize % props.len()]); let k = keyof(s, p, 0); ks.binary_search_by_key(&g(k), |&x| g(x)).is_err().then(|| np.push(k)); }
    (nx, np)
}
fn a1(a: &[String]) -> io::Result<()> {
    let (st, out, w, eps, shard, seed, eps1) = (arg(a, "--store", "data/wikifact/v3/store"), arg(a, "--out", "data/torus/a1_wikidata.gf17rib"), arg(a, "--w", "64").parse::<usize>().unwrap(), arg(a, "--eps", "0.0").parse::<f64>().unwrap(), arg(a, "--shard", "65536").parse::<usize>().unwrap(), arg(a, "--seed", "17").parse::<u64>().unwrap(), arg(a, "--eps1", "0.08").parse::<f64>().unwrap());
    let bump = bumpcfg(a);
    let t0 = Instant::now();
    let f = facts(&st)?;
    let tl = t0.elapsed().as_secs_f64();
    let (b, b1, bytes) = layered::<R>(&out, &f, w, R - VD, shard, eps, seed, bump, eps1, None)?;
    let tb = t0.elapsed().as_secs_f64() - tl;
    let ts = 0.0;
    let peak = rss().1;
    let t = Table::<R>::open(&out)?;
    let (bf, r1) = (b.bumped.len() as f64 / f.len() as f64, b1.as_ref().map_or(0, |x| x.retries));
    let l2 = 17f64.log2();
    let (bound, bound6) = (f.len() as f64 * R as f64 * l2 / 8.0, f.len() as f64 * VD as f64 * l2 / 8.0);
    let t1 = Instant::now();
    let bad: usize = f.par_iter().filter(|&&(k, v)| t.get(k) != Some(v)).count();
    let tv = t1.elapsed().as_secs_f64();
    let (nx, np) = absent(&f, 5_000_000, seed ^ 99);
    let (fx, fp) = (nx.par_iter().filter(|&&k| t.get(k).is_some()).count(), np.par_iter().filter(|&&k| t.get(k).is_some()).count());
    let mut r = seed ^ 7;
    let ks: Vec<u64> = (0..10_000_000).map(|_| f[sm(&mut r) as usize % f.len()].0).collect();
    bench(&t, &ks[..1_000_000]);
    let (lps, x) = std::thread::scope(|s| s.spawn(|| bench(&t, &ks)).join().unwrap());
    let (ak, _) = std::thread::scope(|s| s.spawn(|| bench(&t, &nx[..1_000_000])).join().unwrap());
    t.cold();
    let (cps, _) = std::thread::scope(|s| s.spawn(|| bench(&t, &ks[..200_000])).join().unwrap());
    let o = js(&[("n_keys", f.len().to_string()), ("w", w.to_string()), ("digits_per_cell", R.to_string()), ("fp_digits", (R - VD).to_string()), ("shards", b.ms.len().to_string()), ("bucket", b.b.to_string()), ("thresholds", format!("{:?}", b.th)), ("bumped_frac", format!("{bf:.5}")), ("cells_l0", t.cells.to_string()), ("cells", t.all_cells().to_string()), ("eps_cells", format!("{:.5}", t.all_cells() as f64 / f.len() as f64 - 1.0)), ("retries", (b.retries + r1).to_string()), ("bytes_l0", t.map.len().to_string()), ("elim_steps_per_key", format!("{:.3}", b.steps as f64 / f.len() as f64)), ("bytes", bytes.to_string()), ("bound_bytes", format!("{bound:.0}")), ("ratio_vs_bound", format!("{:.5}", bytes as f64 / bound)), ("ratio_vs_obj_only_bound", format!("{:.5}", bytes as f64 / bound6)), ("bits_per_key", format!("{:.4}", bytes as f64 * 8.0 / f.len() as f64)), ("pages", t.pages().to_string()), ("exact_wrong", bad.to_string()), ("exact_recall", format!("{:.9}", 1.0 - bad as f64 / f.len() as f64)), ("verify_s", format!("{tv:.1}")), ("fpr_next_index", format!("{:.3e}", fx as f64 / nx.len() as f64)), ("fpr_absent_prop", format!("{:.3e}", fp as f64 / np.len() as f64)), ("fpr_expected", format!("{:.3e}", 17f64.powi(-((R - VD) as i32)))), ("absent_probes", (nx.len() + np.len()).to_string()), ("lookups_per_s_core_warm", format!("{lps:.0}")), ("absent_lookups_per_s_core", format!("{ak:.0}")), ("lookups_per_s_core_cold", format!("{cps:.0}")), ("load_s", format!("{tl:.1}")), ("build_s", format!("{tb:.1}")), ("save_s", format!("{ts:.1}")), ("peak_rss_bytes", peak.to_string()), ("chk", x.to_string())]);
    println!("{o}");
    fs::write(format!("{out}.json"), &o)
}
fn bumpcfg(a: &[String]) -> Option<(usize, [usize; 4])> { let b = arg(a, "--b", "0").parse::<usize>().unwrap(); let t: Vec<usize> = arg(a, "--th", "0,0,0,0").split(',').map(|x| x.parse().unwrap()).collect(); (b > 0).then(|| (b, [t[0], t[1], t[2], b])) }
fn eps(a: &[String]) {
    let (n, w, trials, bump) = (arg(a, "--n", "65536").parse::<usize>().unwrap(), arg(a, "--w", "64").parse::<usize>().unwrap(), arg(a, "--trials", "8").parse::<u64>().unwrap(), bumpcfg(a));
    let es: Vec<f64> = arg(a, "--es", "0.005,0.01,0.015,0.02,0.03,0.04,0.06,0.08").split(',').map(|x| x.trim().parse().unwrap()).collect();
    for e in es {
        let m = ((n as f64 * (1.0 + e)).ceil() as usize) + w;
        let t0 = Instant::now();
        let ok: Vec<(u64, usize)> = (0..trials).into_par_iter().filter_map(|t| { let tab = Tab::new(t * 1000 + 1); let mut rows: Vec<(u32, u64, [u8; R])> = (0..n as u64).map(|i| { let h = tab.h(i); (((h[0] & 0xFFFF_FFFF) * (m - w + 1) as u64 >> 32) as u32, if w == 64 { h[1] } else { h[1] & ((1 << w) - 1) }, std::array::from_fn(|k| (h[2] >> (4 * k) & 15) as u8)) }).collect(); rows.sort_unstable_by_key(|r| r.0); solve::<R>(&rows, m, w, t, bump).map(|s| (s.steps, s.bumped.len())) }).collect();
        let bf = ok.iter().map(|x| x.1).sum::<usize>() as f64 / (ok.len().max(1) * n) as f64;
        println!("n={n} w={w} bump={bump:?} eps={e} ok={}/{trials} bumped={bf:.4} est_total_overhead(l1 at 1.10)={:.4} steps/key={:.2} s={:.2}", ok.len(), (m as f64 + 1.10 * bf * n as f64) / n as f64 - 1.0, ok.iter().map(|x| x.0).sum::<u64>() as f64 / (ok.len().max(1) * n) as f64, t0.elapsed().as_secs_f64());
    }
}
fn a5(a: &[String]) -> io::Result<()> {
    let (st, base, tsv, out, seed) = (arg(a, "--store", "data/wikifact/v3/store"), arg(a, "--base", "data/torus/a1_wikidata.gf17rib"), arg(a, "--facts", "crates/adam-torus/facts/a5_cc0_facts.tsv"), arg(a, "--out", "data/torus/a5"), arg(a, "--seed", "5").parse::<u64>().unwrap());
    let (f, t0) = (facts(&st)?, Instant::now());
    let qid = col(&st, "e_qid", |c| u32::from_le_bytes(c.try_into().unwrap()))?;
    let txt = fs::read_to_string(&tsv)?;
    let rows: Vec<Vec<&str>> = txt.lines().filter(|l| !l.starts_with('#')).map(|l| l.split('\t').collect()).collect();
    let mut nov: Vec<&str> = vec![];
    rows.iter().for_each(|r| { (!nov.contains(&r[0])).then(|| nov.push(r[0])); });
    let id = |x: &str| -> u32 { nov.iter().position(|n| *n == x).map(|i| 12_000_000 + i as u32).unwrap_or_else(|| { let q: u32 = x[1..].parse().unwrap(); qid.binary_search(&q).unwrap_or_else(|_| panic!("{x} not in store")) as u32 }) };
    let mut cnt: std::collections::HashMap<(u32, u16), u16> = Default::default();
    let new: Vec<(u64, u32)> = rows.iter().map(|r| { let (s, p) = (id(r[0]), r[1][1..].parse::<u16>().unwrap()); let c = cnt.entry((s, p)).or_insert(0); *c += 1; (keyof(s, p, *c - 1), id(r[2])) }).collect();
    let bt = Table::<R>::open(&base)?;
    let pre = new.iter().filter(|x| bt.get(x.0).is_some()).count();
    let mut r = seed;
    let (dbi, dni): (Vec<usize>, Vec<usize>) = { let mut b: Vec<usize> = (0..50).map(|_| sm(&mut r) as usize % f.len()).collect(); b.sort(); b.dedup(); let mut n: Vec<usize> = (0..new.len()).collect(); (0..50).for_each(|i| { let j = i + sm(&mut r) as usize % (n.len() - i); n.swap(i, j) }); n.truncate(50); n.sort(); (b, n) };
    let (db, dn): (HashSet<u64>, HashSet<u64>) = (dbi.iter().map(|&i| f[i].0).collect(), dni.iter().map(|&i| new[i].0).collect());
    let ovp = format!("{out}_overlay.gf17rib");
    let t1 = Instant::now();
    layered::<Q>(&ovp, &new, 64, Q - VD, 65536, 0.08, seed ^ 0x0BE1, None, 0.0, None)?;
    let (tl, lv) = (t1.elapsed().as_secs_f64(), Live::<R, Q> { base: Table::<R>::open(&base)?, ov: Some(Table::<Q>::open(&ovp)?) });
    let (n1, o1) = (new.iter().filter(|x| lv.get(x.0) == Some(x.1)).count(), f.par_iter().filter(|x| lv.get(x.0) == Some(x.1)).count());
    let ov2: Vec<(u64, u32)> = new.iter().map(|&(k, v)| (k, if dn.contains(&k) { TOMB } else { v })).chain(db.iter().map(|&k| (k, TOMB))).collect();
    let t2 = Instant::now();
    layered::<Q>(&ovp, &ov2, 64, Q - VD, 65536, 0.08, seed ^ 0x0BE2, None, 0.0, None)?;
    let (tf, lv) = (t2.elapsed().as_secs_f64(), Live::<R, Q> { base: Table::<R>::open(&base)?, ov: Some(Table::<Q>::open(&ovp)?) });
    let (dg2, n2, o2) = (db.iter().chain(dn.iter()).filter(|&&k| lv.get(k).is_none()).count(), new.iter().filter(|x| !dn.contains(&x.0) && lv.get(x.0) == Some(x.1)).count(), f.par_iter().filter(|x| !db.contains(&x.0) && lv.get(x.0) == Some(x.1)).count());
    let grp = |k: u64| k & 0xFF_FFFF_FFFF;
    let tg: HashSet<u64> = db.iter().chain(new.iter().map(|x| &x.0)).map(|&k| grp(k)).collect();
    let mut want: std::collections::HashMap<u64, Vec<(u16, u32)>> = Default::default();
    f.iter().chain(new.iter()).filter(|x| tg.contains(&grp(x.0)) && !db.contains(&x.0) && !dn.contains(&x.0)).for_each(|&(k, v)| want.entry(grp(k)).or_default().push((unkey(k).2, v)));
    want.values_mut().for_each(|v| v.sort());
    let fin: Vec<(u64, u32)> = f.iter().filter(|x| !tg.contains(&grp(x.0))).copied().chain(want.iter().flat_map(|(&g, v)| v.iter().enumerate().map(move |(i, &(_, x))| (g | (i as u64) << 40, x)))).collect();
    let vals = |l: &Live<R, Q>, g: u64| l.values((g & 0xFF_FFFF) as u32, (g >> 24) as u16);
    let g2 = want.iter().filter(|(&g, v)| vals(&lv, g) == v.iter().map(|x| x.1).collect::<Vec<_>>()).count();
    let mut aff = vec![false; bt.nsh as usize];
    f.iter().chain(new.iter()).filter(|x| tg.contains(&grp(x.0))).map(|x| x.0).chain(fin.iter().filter(|x| tg.contains(&grp(x.0))).map(|x| x.0)).for_each(|k| aff[bt.shard_of(k)] = true);
    let cp = format!("{out}_compacted.gf17rib");
    let t3 = Instant::now();
    layered::<R>(&cp, &fin, bt.w, bt.fd, 65536, -0.03, bt.seed, Some((bt.b, bt.th)), 0.08, Some((&bt, &aff)))?;
    let tc = t3.elapsed().as_secs_f64();
    let lc = Live::<R, Q> { base: Table::<R>::open(&cp)?, ov: None };
    let (c_all, c_grp) = (fin.par_iter().filter(|x| lc.get(x.0) == Some(x.1)).count(), want.iter().filter(|(&g, v)| vals(&lc, g) == v.iter().map(|x| x.1).collect::<Vec<_>>()).count());
    let gone = dbi.iter().map(|&i| f[i]).chain(dni.iter().map(|&i| new[i])).filter(|&(k, v)| { let n = want.get(&grp(k)).map_or(0, |w| w.iter().filter(|x| x.1 == v).count()); vals(&lc, grp(k)).iter().filter(|&&x| x == v).count() == n }).count();
    let fp = format!("{out}_fullrebuild.gf17rib");
    let t4 = Instant::now();
    layered::<R>(&fp, &fin, bt.w, bt.fd, 65536, -0.03, bt.seed, Some((bt.b, bt.th)), 0.08, None)?;
    let tfr = t4.elapsed().as_secs_f64();
    let same = fs::read(&fp)? == fs::read(&cp)?;
    [fp.clone(), format!("{fp}.l1")].iter().for_each(|p| { let _ = fs::remove_file(p); });
    let o = js(&[("new_facts", new.len().to_string()), ("novel_entities", nov.len().to_string()), ("new_keys_present_in_base_before", pre.to_string()), ("overlay_learn_s", format!("{tl:.4}")), ("learn_new_exact", format!("{n1}/{}", new.len())), ("learn_old_exact", format!("{o1}/{}", f.len())), ("deleted_base", db.len().to_string()), ("deleted_new", dn.len().to_string()), ("overlay_forget_s", format!("{tf:.4}")), ("forget_deleted_absent", format!("{dg2}/{}", db.len() + dn.len())), ("forget_new_exact", format!("{n2}/{}", new.len() - dn.len())), ("forget_old_exact", format!("{o2}/{}", f.len() - db.len())), ("forget_groups_values_ok", format!("{g2}/{}", want.len())), ("touched_groups", tg.len().to_string()), ("compact_affected_shards", format!("{}/{}", aff.iter().filter(|&&x| x).count(), aff.len())), ("compact_s", format!("{tc:.2}")), ("full_rebuild_s", format!("{tfr:.2}")), ("compacted_equals_full_rebuild_bytes", same.to_string()), ("compact_all_exact", format!("{c_all}/{}", fin.len())), ("compact_groups_values_ok", format!("{c_grp}/{}", want.len())), ("compact_deleted_gone", format!("{gone}/{}", db.len() + dn.len())), ("total_s", format!("{:.1}", t0.elapsed().as_secs_f64())), ("peak_rss_bytes", rss().1.to_string())]);
    println!("{o}");
    fs::write(format!("{out}.json"), &o)
}
fn main() -> io::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) { Some("a1") => a1(&a), Some("a5") => a5(&a), Some("eps") => { eps(&a); Ok(()) } _ => { eprintln!("usage: torus_a a1|a5|eps ..."); Ok(()) } }
}
