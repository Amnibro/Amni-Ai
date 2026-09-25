use adam_ray::{cpu::{cell, hop, plan, top3}, pack::{load, V}, v19::{Lx19, W19, MG}};
use half::f16;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (tb, lx, all) = (load(&a[1], &format!("{}/synsets.txt", a[2]), &format!("{}/wordcounts.txt", a[2])).unwrap(), Lx19::new(), std::fs::read(&a[3]).unwrap());
    for (si, txt) in all.split(|&b| b == 0).enumerate() {
        println!("== seg {si}");
        let (g, mut w) = (tb.geo.clone(), W19::new(tb.geo.copy));
        txt.iter().for_each(|&b| w.step(&g, &tb.lx, &lx, b));
        let (m, s, st) = (&w.m, w.s, |b: &[u8]| String::from_utf8_lossy(b).into_owned());
        (0..64).filter(|&i| m.stamp[i] > 0).for_each(|i| println!("ring {i:2} ent {:10} rel {:2} stamp {:3} val {:?}", m.ent[i], m.rel[i], m.stamp[i], st(&m.val[i][..m.vlen[i] as usize])));
        println!("qent {} qrel {} qtyp {} cand {:?} mstate {} ap {} vp {} mv {} tail {:?} state {:?}", m.qent, m.qrel, m.qtyp, m.cand, w.mstate(), w.q.ap, w.q.vp, g.mv, st(&txt[txt.len().saturating_sub(40)..]), st(&s[..16].iter().rev().map(|&c| c as u8).collect::<Vec<_>>()));
        let gv = w.gv(&g, &tb.cw);
        gv.iter().enumerate().filter(|x| x.1 .0 >= 0).for_each(|(k, &(b, v))| { let (cv, p) = w.cv(k); println!("gate {k:2} byte {:?} val {v:+.3} cand {:?}@{p}", b as u8 as char, st(&cv)) });
        let (b1, b2) = (s[0], s[1]);
        let (r1, r2) = (b1 as usize * V, (256 + b2 * 256 + b1) as usize * V);
        let mut lg: Vec<f32> = (0..V).map(|c| tb.bias[c] + f16::from_bits(tb.dp[r1 + c]).to_f32() + f16::from_bits(tb.dp[r2 + c]).to_f32()).collect();
        let mut stages = vec![("dense", lg.clone())];
        let pl = plan(&g, &s);
        for grp in [(2, 20, "byte/word tori"), (20, 21, "topic"), (21, 24, "question tori")] { pl.iter().filter(|x| x.0 >= grp.0 && x.0 < grp.1).for_each(|&(l, _, e, wt)| cell(&tb, &mut lg, l, e, wt)); stages.push((grp.2, lg.clone())); }
        gv.iter().filter(|x| x.0 >= 0).for_each(|&(b, v)| lg[b as usize] += v);
        stages.push(("gates", lg.clone()));
        w.mtaps(&g).iter().enumerate().for_each(|(i, e)| if let Some(e) = *e { cell(&tb, &mut lg, g.mb() + i, e, MG) });
        stages.push(("mr", lg.clone()));
        (0..g.hops).for_each(|h| { let e = hop(&g, top3(&lg), b1, b2, h); cell(&tb, &mut lg, g.hb() + h, e, 1.0) });
        stages.push(("hops", lg.clone()));
        let mut o: Vec<usize> = (0..V).collect();
        o.sort_by(|&x, &y| lg[y].partial_cmp(&lg[x]).unwrap());
        let extra: Vec<usize> = a.get(4).map(|x| x.bytes().map(|b| b as usize).collect()).unwrap_or_default();
        for &c in o[..6].iter().chain(extra.iter()) { let mut p = 0f32; println!("byte {:?} final {:+.3} | {}", c as u8 as char, lg[c], stages.iter().map(|(n, v)| { let d = v[c] - p; p = v[c]; format!("{n} {d:+.2}") }).collect::<Vec<_>>().join(", ")); }
    }
}
