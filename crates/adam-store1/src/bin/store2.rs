use adam_gf17::rss;
use adam_store1::{s2::*, *};
use adam_torus::{drow, layered_by};
use rayon::prelude::*;
use std::{collections::{HashMap, HashSet}, fs, io, time::Instant};
const L17: f64 = 4.087462841250339;
fn by(bits: f64) -> f64 { bits / L17 * 4.25 / 8.0 }
fn dby(d: usize) -> f64 { d as f64 * 4.25 / 8.0 }
fn strs(w: &W3) -> Vec<&[u8]> { let mut seen: HashSet<&[u8]> = HashSet::new(); let mut o = vec![]; (0..w.fv.n).filter(|&i| w.fk.s()[i] >= 4).map(|i| w.fv.s()[i]).chain((0..w.qv.n).filter(|&j| w.qk.s()[j] >= 4).map(|j| w.qv.s()[j])).for_each(|v| { let s = sbytes(w, v); if seen.insert(s) { o.push(s) } }); o }
struct Acc { seq: [Vec<u32>; 4], ops: Vec<u32>, mbits: u64, raw: [u64; 4], words: [u64; 4], bad: u64 }
fn acc() -> Acc { Acc { seq: Default::default(), ops: vec![], mbits: 0, raw: [0; 4], words: [0; 4], bad: 0 } }
fn put(a: &mut Acc, b: &Bpe, f: usize, x: &Tf, orig: &[u8]) {
    if x == &Tf::E { return }
    a.bad += (untf(b, x) != orig) as u64;
    a.seq[f].extend(tfp(x));
    a.seq[f].push(TEND);
    match x { Tf::W(ws) if f != 2 => { let mut pv = 5u32; ws.iter().enumerate().for_each(|(i, t)| { let op = match t { Tw::P(_, o, m) => { a.mbits += m.len() as u64; *o as u32 } Tw::Raw(_) => 4 }; a.ops.push((f as u32) << 16 | ((i == 0) as u32) << 8 | pv << 4 | op); pv = op }); a.words[f] += ws.len() as u64 } Tf::W(ws) => a.words[f] += ws.len() as u64, Tf::R(_) => a.raw[f] += 1, Tf::E => {} }
}
fn tfield(name: &str, s: &[u32], ts: &[u64]) -> String {
    let n = s.len();
    let mut c0: HashMap<u32, u64, FxB> = HashMap::default();
    s.iter().for_each(|&x| *c0.entry(x).or_default() += 1);
    let h0 = hbits(&c0.values().copied().collect::<Vec<_>>());
    let hl = hlen(&c0.values().copied().collect::<Vec<_>>());
    let hd: u64 = c0.values().zip(&hl).map(|(&c, &l)| c * l as u64).sum();
    let ends = c0.get(&TEND).copied().unwrap_or(0);
    let mut k: Vec<u64> = (0..n).into_par_iter().map(|i| (if i == 0 || s[i - 1] == TEND { TEND as u64 + 1 } else { s[i - 1] as u64 }) << 20 | s[i] as u64).collect();
    k.par_sort_unstable();
    let mut pr: Vec<(u64, u64)> = vec![];
    k.iter().for_each(|&x| if pr.last().is_some_and(|l| l.0 == x) { pr.last_mut().unwrap().1 += 1 } else { pr.push((x, 1)) });
    let mut gs: Vec<(usize, usize)> = vec![];
    let mut a = 0;
    (1..=pr.len()).for_each(|i| if i == pr.len() || pr[i].0 >> 20 != pr[a].0 >> 20 { gs.push((a, i)); a = i });
    let mut out = format!("\"{name}\":{{\"symbols\":{n},\"fields\":{ends},\"distinct\":{},\"h0_bits_per_sym\":{:.3},\"order0_rans_mb\":{:.3},\"huffman17_digits_per_sym\":{:.3},\"huffman17_mb\":{:.3},\"fixed4_mb\":{:.3},\"pairs\":{},\"order1\":[", c0.len(), h0 / n as f64, by(h0) / 1e6, hd as f64 / n as f64, dby(hd as usize) / 1e6, dby(4 * (n - ends as usize) + ends as usize) / 1e6, pr.len());
    for (ti, &t) in ts.iter().enumerate() {
        let r: Vec<(f64, HashMap<u32, u64, FxB>, usize, u64)> = gs.par_chunks(4096).map(|ch| { let mut esc: HashMap<u32, u64, FxB> = HashMap::default(); let (mut bits, mut dg, mut kc) = (0f64, 0usize, 0u64); let mut pk = 0u64; for &(a, z) in ch { let tot: u64 = pr[a..z].iter().map(|x| x.1).sum(); let mut cs: Vec<u64> = pr[a..z].iter().map(|x| x.1).collect(); cs.sort_unstable_by(|a, b| b.cmp(a)); let tt = t.max(*cs.get(8191).unwrap_or(&0) + 1); let kept: Vec<(u32, u64)> = pr[a..z].iter().filter(|x| x.1 >= tt).map(|x| ((x.0 & 0xFFFFF) as u32, x.1)).collect(); let ks: u64 = kept.iter().map(|x| x.1).sum(); if kept.is_empty() { pr[a..z].iter().for_each(|x| *esc.entry((x.0 & 0xFFFFF) as u32).or_default() += x.1); continue } let e = tot - ks; bits += kept.iter().map(|x| x.1 as f64 * (tot as f64 / x.1 as f64).log2()).sum::<f64>() + if e > 0 { e as f64 * (tot as f64 / e as f64).log2() } else { 0.0 }; pr[a..z].iter().filter(|x| x.1 < tt).for_each(|x| *esc.entry((x.0 & 0xFFFFF) as u32).or_default() += x.1); let mut v = kept.clone(); (e > 0).then(|| v.push((ESC, e))); let mut o = vec![]; let ctx = pr[a].0 >> 20; ev(&mut o, ctx - pk.min(ctx)); pk = ctx; Fq::new(&v).ser(&mut o); dg += o.len(); kc += kept.len() as u64 } (bits, esc, dg, kc) }).collect();
        let mut esc: HashMap<u32, u64, FxB> = HashMap::default();
        r.iter().for_each(|x| x.1.iter().for_each(|(&k, &v)| *esc.entry(k).or_default() += v));
        let b1: f64 = r.iter().map(|x| x.0).sum();
        let b0 = hbits(&esc.values().copied().collect::<Vec<_>>());
        let tdg: usize = r.iter().map(|x| x.2).sum();
        let kc: u64 = r.iter().map(|x| x.3).sum();
        let escn: u64 = esc.values().sum();
        let hi: HashMap<u32, Vec<(u32, u64)>> = esc.iter().fold(HashMap::new(), |mut m, (&k, &v)| { m.entry(k / 289).or_default().push((k % 289, v)); m });
        let mut o0 = vec![];
        Fq::new(&hi.iter().map(|(&h, v)| (h, v.iter().map(|x| x.1).sum())).collect::<Vec<_>>()).ser(&mut o0);
        hi.values().for_each(|v| Fq::new(v).ser(&mut o0));
        out += &format!("{}{{\"t\":{t},\"kept_pairs\":{kc},\"escaped\":{escn},\"bits_per_sym\":{:.3},\"stream_mb\":{:.3},\"order1_table_mb\":{:.3},\"order0_table_mb\":{:.3},\"total_mb\":{:.3}}}", if ti > 0 { "," } else { "" }, (b1 + b0) / n as f64, by(b1 + b0) / 1e6, dby(tdg) / 1e6, dby(o0.len()) / 1e6, (by(b1 + b0) + dby(tdg) + dby(o0.len())) / 1e6);
    }
    out + "]}"
}
fn tstat(a: &[String]) -> io::Result<()> {
    let (src, pd, bd) = (arg(a, "--v3", "data/wikifact/v3/store"), arg(a, "--pieces", "data/store2/pieces"), arg(a, "--build", "data/store1/build"));
    let t0 = Instant::now();
    let w = w3(&src)?;
    let ne = w.qid.n;
    let b = Bpe::load(&pd)?;
    let nme = nmenc(&format!("{bd}/nmenc.bin"), ne)?;
    let st = strs(&w);
    eprintln!("load {:.1}s strings {}", t0.elapsed().as_secs_f64(), st.len());
    let lim: usize = arg(a, "--n", &ne.to_string()).parse().unwrap();
    let r = (0..lim.min(ne)).into_par_iter().fold(acc, |mut x, e| { let (l, t) = (lab(&w, e), tit(&w, e)); let (tk, ex) = tkind(l, t); put(&mut x, &b, 0, &tf(&b, l, true), l); (tk >= 2).then(|| put(&mut x, &b, 1, &tf(&b, &ex, true), &ex)); nme[e].iter().filter(|n| n.0 == 2).for_each(|n| put(&mut x, &b, 2, &tf(&b, &n.1, false), &n.1)); x }).reduce(acc, |mut x, y| { (0..4).for_each(|f| { x.seq[f].extend_from_slice(&y.seq[f]); x.raw[f] += y.raw[f]; x.words[f] += y.words[f] }); x.ops.extend(y.ops); x.mbits += y.mbits; x.bad += y.bad; x });
    let rs = st.par_iter().fold(acc, |mut x, s| { put(&mut x, &b, 3, &tf(&b, s, true), s); x }).reduce(acc, |mut x, y| { x.seq[3].extend_from_slice(&y.seq[3]); x.raw[3] += y.raw[3]; x.words[3] += y.words[3]; x.ops.extend(y.ops); x.mbits += y.mbits; x.bad += y.bad; x });
    eprintln!("tokenized {:.1}s", t0.elapsed().as_secs_f64());
    let ts: Vec<u64> = arg(a, "--t", "2,4,8,16,64").split(',').map(|x| x.parse().unwrap()).collect();
    let mut ops: HashMap<u32, Vec<(u32, u64)>> = HashMap::new();
    r.ops.iter().chain(&rs.ops).for_each(|&o| { let v = ops.entry(o >> 4).or_default(); match v.iter_mut().find(|x| x.0 == o & 15) { Some(x) => x.1 += 1, None => v.push((o & 15, 1)) } });
    let ob: f64 = ops.values().map(|v| hbits(&v.iter().map(|x| x.1).collect::<Vec<_>>())).sum();
    let opc: HashMap<u32, u64> = r.ops.iter().chain(&rs.ops).fold(HashMap::new(), |mut m, &o| { *m.entry(o & 15).or_default() += 1; m });
    let names = ["label", "title_extra", "alias", "string"];
    let fs_: Vec<String> = (0..4).map(|f| tfield(names[f], if f == 3 { &rs.seq[3] } else { &r.seq[f] }, &ts)).collect();
    let o = format!("{{\"entities\":{},\"roundtrip_bad\":{},\"raw_fields\":{:?},\"words\":{:?},\"raw_strings\":{},\"case_ops\":{},\"case_bits\":{:.0},\"case_mb\":{:.3},\"mask_bits\":{},{},\"sec\":{:.1}}}", lim.min(ne), r.bad + rs.bad, r.raw, r.words, rs.raw[3], format!("[{}]", { let mut v: Vec<_> = opc.iter().collect(); v.sort(); v.iter().map(|(k, c)| format!("[{k},{c}]")).collect::<Vec<_>>().join(",") }), ob, (by(ob) + dby(0) + r.mbits as f64 / 8.0) / 1e6, r.mbits + rs.mbits, fs_.join(","), t0.elapsed().as_secs_f64());
    println!("{o}");
    fs::write(arg(a, "--out", "data/store2/results/tstat.json"), o)?;
    eprintln!("rss {:?}", rss());
    Ok(())
}
fn rk(w: &W3, f: usize, sid: &[u64]) -> u128 { let k = w.fk.s()[f]; (k as u128) << 104 | (if k == 3 { w.fx.s()[f] as u128 } else { 0 }) << 64 | if k >= 4 { sid[f] } else { w.fv.s()[f] } as u128 }
fn sidmap(w: &W3) -> (Vec<u64>, Vec<bool>) { let mut m: HashMap<&[u8], u64> = HashMap::new(); let (mut sid, mut new) = (vec![0u64; w.fv.n], vec![false; w.fv.n]); (0..w.fv.n).filter(|&i| w.fk.s()[i] >= 4).for_each(|i| { let s = sbytes(w, w.fv.s()[i]); let n = m.len() as u64; let x = *m.entry(s).or_insert(n); sid[i] = x; new[i] = x == n }); (sid, new) }
fn vids(w: &W3, sid: &[u64]) -> (Vec<u32>, HashMap<u16, u32>) { let mut k: Vec<(u128, u32)> = (0..w.fv.n).into_par_iter().map(|f| ((w.fp.s()[f] as u128) << 112 | rk(w, f, sid), f as u32)).collect(); k.par_sort_unstable(); let mut v = vec![0u32; w.fv.n]; let mut np: HashMap<u16, u32> = HashMap::new(); let mut c = 0u32; (0..k.len()).for_each(|i| { if i > 0 && k[i].0 >> 112 != k[i - 1].0 >> 112 { np.insert((k[i - 1].0 >> 112) as u16, c + 1); c = 0 } else if i > 0 && k[i].0 != k[i - 1].0 { c += 1 } v[k[i].1 as usize] = c }); k.last().map(|l| np.insert((l.0 >> 112) as u16, c + 1)); (v, np) }
fn sigh(g: &[(u16, usize, usize)]) -> u64 { g.iter().fold(0x9E37_79B9u64, |h, x| (h.rotate_left(7) ^ (x.0 as u64) << 1 ^ (x.2 > 1) as u64).wrapping_mul(0x100_0000_01B3)) }
const RU: [(u16, u16, u16); 6] = [(17, 131, 17), (17, 276, 17), (17, 159, 17), (27, 19, 17), (27, 20, 17), (20, 19, 0)];
const CP: [u16; 3] = [17, 27, 495];
fn bh(c: &mut Vec<u64>) -> f64 { c.sort_unstable_by(|a, b| b.cmp(a)); let t: u64 = c.iter().sum(); let mut bk: HashMap<u32, (u64, u64)> = HashMap::new(); c.iter().enumerate().for_each(|(r, &x)| { let e = bk.entry(64 - (r as u64 + 1).leading_zeros()).or_default(); e.0 += x; e.1 += 1 }); bk.values().map(|&(x, n)| x as f64 * ((t as f64 / x as f64).log2() + (n as f64).log2())).sum() }
#[derive(Default)]
struct St { cty: [u64; 6], ctye: HashMap<u32, u64>, sl: [HashMap<(u16, u8, u8), Vec<u64>>; 2], pay: [HashMap<(u16, u32), HashMap<u32, u64>>; 2], ccls: HashMap<u16, HashMap<u32, u64>>, near: HashMap<u16, [u64; 16]>, aux: HashMap<(u16, u8, u8), HashMap<u8, u64>>, sesc: u64, nsym: Vec<u64> }
fn mrg(mut x: St, y: St) -> St { (0..6).for_each(|i| x.cty[i] += y.cty[i]); y.ctye.iter().for_each(|(k, c)| *x.ctye.entry(*k).or_default() += c); for v in 0..2 { y.sl[v].iter().for_each(|(k, c)| { let e = x.sl[v].entry(*k).or_insert(vec![0; c.len()]); e.iter_mut().zip(c).for_each(|(a, b)| *a += b) }); y.pay[v].iter().for_each(|(k, m)| { let e = x.pay[v].entry(*k).or_default(); m.iter().for_each(|(a, c)| *e.entry(*a).or_default() += c) }) } y.ccls.iter().for_each(|(k, m)| { let e = x.ccls.entry(*k).or_default(); m.iter().for_each(|(a, c)| *e.entry(*a).or_default() += c) }); y.near.iter().for_each(|(k, v)| { let e = x.near.entry(*k).or_insert([0; 16]); (0..16).for_each(|i| e[i] += v[i]) }); y.aux.iter().for_each(|(k, m)| { let e = x.aux.entry(*k).or_default(); m.iter().for_each(|(a, c)| *e.entry(*a).or_default() += c) }); x.sesc += y.sesc; if x.nsym.len() < y.nsym.len() { x.nsym.resize(y.nsym.len(), 0) } y.nsym.iter().enumerate().for_each(|(i, c)| x.nsym[i] += c); x }
fn fstat(a: &[String]) -> io::Result<()> {
    let (src, bs, tc, rr, kk, ck) = (arg(a, "--v3", "data/wikifact/v3/store"), arg(a, "--b", "32").parse::<usize>().unwrap(), arg(a, "--tc", "4").parse::<u64>().unwrap(), arg(a, "--r", "8").parse::<usize>().unwrap(), arg(a, "--k", "2").parse::<usize>().unwrap(), arg(a, "--ctxk", "p31"));
    let pr: HashSet<String> = arg(a, "--pred", "rule,mtf,ctx,near,str,cls,auxc").split(',').map(String::from).collect();
    let on = |k: &str| pr.contains(k);
    let t0 = Instant::now();
    let w = w3(&src)?;
    let ne = w.qid.n;
    let (sid, snew) = sidmap(&w);
    let (vid, np) = vids(&w, &sid);
    let ctry: Vec<u32> = (0..ne).into_par_iter().map(|e| { let g = groups(&w, e); CP.iter().find_map(|&p| g.iter().find(|x| x.0 == p && w.fk.s()[x.1] == 1).map(|x| w.fv.s()[x.1] as u32)).unwrap_or(u32::MAX) }).collect();
    eprintln!("vids+countries {:.1}s", t0.elapsed().as_secs_f64());
    let gv = |e: usize, p: u16| groups(&w, e).into_iter().find(|g| g.0 == p);
    let cv = |g: &[(u16, usize, usize)], p: u16| -> u64 { let pv = g.iter().find(|x| x.0 == 31).map_or(0, |x| vid[x.1] as u64 + 1); match ck.as_str() { "sig" => sigh(g), "sigp" => if p == 31 { sigh(g) } else { sigh(g) ^ pv.wrapping_mul(0x9E37_79B9_7F4A_7C15) }, _ => if p == 31 { sigh(g) } else { pv } } };
    let mut ctab: HashMap<(u16, u64), Vec<u128>> = HashMap::new();
    let mut tdg = 0f64;
    if on("ctx") {
        let mut k: Vec<(u16, u64, u32, u32)> = (0..ne).into_par_iter().flat_map_iter(|e| { let g = groups(&w, e); g.iter().flat_map(|x| (x.1..x.1 + x.2).map(|f| (x.0, f)).collect::<Vec<_>>()).map(|(p, f)| (p, cv(&g, p), vid[f], f as u32)).collect::<Vec<_>>() }).filter(|x| x.0 == 31 || x.1 != 0).collect();
        k.par_sort_unstable();
        let mut i = 0;
        while i < k.len() { let mut j = i; while j < k.len() && (k[j].0, k[j].1) == (k[i].0, k[i].1) { j += 1 } if (j - i) as u64 >= tc { let mut c: Vec<(u64, u32)> = vec![]; let mut s = i; while s < j { let mut z = s; while z < j && k[z].2 == k[s].2 { z += 1 } c.push(((z - s) as u64, k[s].3)); s = z } c.sort_unstable_by(|a, b| b.cmp(a)); let l: Vec<u128> = c.iter().take(kk).filter(|x| x.0 >= 2).map(|x| rk(&w, x.1 as usize, &sid)).collect(); if !l.is_empty() { tdg += l.len() as f64 * (np[&k[i].0] as f64).log(17.0).ceil().max(1.0) + 1.0; ctab.insert((k[i].0, k[i].1), l); } } i = j }
        eprintln!("ctx tables {} entries {:.1}s", ctab.len(), t0.elapsed().as_secs_f64());
    }
    let nsl = rr + kk + 5;
    let nb = ne.div_ceil(bs);
    let st: St = (0..nb).into_par_iter().fold(St::default, |mut s: St, b| {
        let mut rec: HashMap<u16, Vec<u128>> = HashMap::new();
        let mut cm: Vec<u32> = vec![];
        for e in b * bs..((b + 1) * bs).min(ne) {
            let g = groups(&w, e);
            if on("ectry") && ctry[e] != u32::MAX { let j = cm.iter().position(|&c| c == ctry[e]); s.cty[j.unwrap_or(4)] += 1; if j.is_none() { *s.ctye.entry(ctry[e]).or_default() += 1 } cm.retain(|&c| c != ctry[e]); cm.insert(0, ctry[e]); cm.truncate(4) } else if on("ectry") { s.cty[5] += 1 }
            let rmatch = |p: u16, i: usize| -> Option<u16> { RU.iter().filter(|r| r.0 == p).find_map(|r| { let vg = g.iter().find(|x| x.0 == r.1)?; let gp = g.iter().find(|x| x.0 == p)?; let tv = rk(&w, gp.1 + i, &sid); let c = if r.2 == 0 { Some(rk(&w, vg.1 + i.min(vg.2 - 1), &sid)) } else { (w.fk.s()[vg.1] == 1).then(|| gv(w.fv.s()[vg.1] as usize, r.2)).flatten().map(|t| rk(&w, t.1 + i.min(t.2 - 1), &sid)) }; (c == Some(tv)).then_some(r.1) }) };
            let mut nocls: Vec<u16> = vec![];
            if on("nocyc") { if let Some(cp) = CP.iter().find(|&&p| g.iter().any(|x| x.0 == p && w.fk.s()[x.1] == 1)) { let mut q = *cp; while let Some(v) = rmatch(q, 0) { if nocls.contains(&v) { break } nocls.push(v); q = v } } }
            for &(p, a0, n) in &g {
                let rules: Vec<&(u16, u16, u16)> = if on("rule") { RU.iter().filter(|r| r.0 == p && g.iter().any(|x| x.0 == r.1) && !(on("ectry") && r.2 != 0)).collect() } else { vec![] };
                let ecp = on("ectry") && CP.contains(&p);
                let ml = if on("mtf") { rec.get(&p).map_or(0, |v| v.len().min(rr)) } else { 0 };
                let cx = on("ctx") && (p == 31 || g.iter().any(|x| x.0 == 31));
                let sc = ctry[e];
                let mut pa = 0u8;
                for i in 0..n {
                    let f = a0 + i;
                    let tv = rk(&w, f, &sid);
                    let k = w.fk.s()[f];
                    let cl = on("cls") && k == 1 && sc != u32::MAX && !CP.contains(&p) && p != 31 && !(i == 0 && nocls.contains(&p));
                    let mask = (!rules.is_empty() || ecp) as u8 | ([0u8, 1, 2, 3, 3, 4, 4, 4, 4][ml.min(8)]) << 1 | (cx as u8) << 4 | (cl as u8) << 5;
                    let rc = rules.first().and_then(|r| { let vg = g.iter().find(|x| x.0 == r.1).unwrap(); if r.2 == 0 { Some(rk(&w, vg.1 + i.min(vg.2 - 1), &sid)) } else { (w.fk.s()[vg.1] == 1).then(|| gv(w.fv.s()[vg.1] as usize, r.2)).flatten().map(|t| rk(&w, t.1 + i.min(t.2 - 1), &sid)) } });
                    let mi = (0..ml).find(|&j| rec[&p][j] == tv);
                    let ct = if cx { ctab.get(&(p, cv(&g, p))) } else { None };
                    let d = w.fv.s()[f] as i64 - e as i64;
                    let rc = if ecp { Some(1u128 << 104 | ctry[e] as u128) } else { rc };
                    let sym = if rc == Some(tv) { 0 } else if let Some(j) = mi { 1 + j } else if let Some(j) = ct.and_then(|c| c.iter().position(|&x| x == tv)) { rr + 1 + j } else if on("near") && k == 1 && d != 0 && d.abs() <= 8 { rr + kk + 1 } else if on("str") && k >= 4 && snew[f] { rr + kk + 2 } else { rr + kk + 4 };
                    let vcl = if k == 1 { ctry[w.fv.s()[f] as usize] } else { u32::MAX };
                    let sym1 = if sym == rr + kk + 4 && cl && vcl == sc { rr + kk + 3 } else { sym };
                    for (v, sy) in [(0usize, sym), (1, sym1)] { s.sl[v].entry((p, mask, (i > 0) as u8)).or_insert(vec![0; nsl])[sy] += 1; if sy >= rr + kk + 3 && k < 4 { *s.pay[v].entry((p, if v == 1 { vcl } else { 0 })).or_default().entry(vid[f]).or_default() += 1 } }
                    if sym1 == rr + kk + 4 && k < 4 { *s.ccls.entry(p).or_default().entry(vcl).or_default() += 1 }
                    if s.nsym.len() < nsl { s.nsym.resize(nsl, 0) }
                    s.nsym[sym1] += 1;
                    if sym == rr + kk + 1 { s.near.entry(p).or_insert([0; 16])[(if d < 0 { d + 8 } else { d + 7 }) as usize] += 1 }
                    if sym == rr + kk + 4 && k >= 4 { s.sesc += 1 }
                    let ax = w.fr.s()[f] | (qc(&w, f).min(100) as u8) << 1;
                    *s.aux.entry((p, (i > 0) as u8, if on("auxc") { pa.min(7) } else { 0 })).or_default().entry(ax).or_default() += 1;
                    pa = ax + 1;
                }
            }
            g.iter().for_each(|&(p, a0, n)| { let v = rec.entry(p).or_default(); (0..n).rev().for_each(|i| v.insert(0, rk(&w, a0 + i, &sid))); v.truncate(rr) });
        }
        s
    }).reduce(St::default, mrg);
    eprintln!("pass {:.1}s", t0.elapsed().as_secs_f64());
    let mut pb: HashMap<u16, [f64; 10]> = HashMap::new();
    for v in 0..2 { st.sl[v].iter().for_each(|(k, c)| pb.entry(k.0).or_default()[v] += hbits(c)); st.pay[v].iter().for_each(|(k, m)| { let mut c: Vec<u64> = m.values().copied().collect(); let e = pb.entry(k.0).or_default(); e[2 + v] += hbits(&c); e[4 + v] += bh(&mut c) }) }
    st.ccls.iter().for_each(|(k, m)| pb.entry(*k).or_default()[6] += hbits(&m.values().copied().collect::<Vec<_>>()));
    st.near.iter().for_each(|(k, v)| pb.entry(*k).or_default()[7] += hbits(v));
    st.aux.iter().for_each(|(k, m)| pb.entry(k.0).or_default()[8] += hbits(&m.values().copied().collect::<Vec<_>>()));
    let nstr = sid.iter().max().copied().unwrap_or(0) + 1;
    let sbits = st.sesc as f64 * (nstr as f64).log2() + hbits(&st.cty) + hbits(&st.ctye.values().copied().collect::<Vec<_>>());
    eprintln!("country symbol bits {:.0} ({:?}) per-entity-with-country {:.3}", hbits(&st.cty) + hbits(&st.ctye.values().copied().collect::<Vec<_>>()), st.cty, (hbits(&st.cty) + hbits(&st.ctye.values().copied().collect::<Vec<_>>())) / (st.cty[..5].iter().sum::<u64>().max(1)) as f64);
    let pf: HashMap<u16, u64> = (0..w.fv.n).fold(HashMap::new(), |mut m, f| { *m.entry(w.fp.s()[f]).or_default() += 1; m });
    let best = |x: &[f64; 10], ex: bool| -> (f64, bool) { let pa = if ex { 2 } else { 4 }; let a0 = x[0] + x[pa] + x[7] + x[8]; let a1 = x[1] + x[pa + 1] + x[6] + x[7] + x[8]; if on("cls") && a1 < a0 { (a1, true) } else { (a0, false) } };
    let nf = w.fv.n as f64;
    let blk = nb as f64 * 11.0 * 4.25;
    let res: Vec<String> = [true, false].iter().map(|&ex| { let tot: f64 = pb.values().map(|x| best(x, ex).0).sum::<f64>() + sbits; format!("{{\"payload\":\"{}\",\"ideal_bits_per_fact\":{:.4},\"stored_bits_per_fact\":{:.4},\"stored_with_blocks\":{:.4},\"class_props\":{}}}", if ex { "exact_entropy" } else { "rank_buckets" }, tot / nf, tot / nf * 4.25 / L17, (tot * 4.25 / L17 + blk) / nf, pb.values().filter(|x| best(x, ex).1).count()) }).collect();
    let mut top: Vec<(u16, [f64; 10])> = pb.iter().map(|(k, v)| (*k, *v)).collect();
    top.sort_by(|a, b| best(&b.1, false).0.partial_cmp(&best(&a.1, false).0).unwrap());
    let o = format!("{{\"pred\":\"{}\",\"block\":{bs},\"tc\":{tc},\"r\":{rr},\"facts\":{},\"block_overhead_bits_per_fact\":{:.4},\"variants\":[{}],\"sym_counts\":{:?},\"k\":{kk},\"ctxk\":\"{ck}\",\"ctx_table_mb\":{:.3},\"ctx_table_entries\":{},\"string_esc\":{},\"per_prop\":[{}],\"sec\":{:.1}}}", arg(a, "--pred", "rule,mtf,ctx,near,str,cls,auxc"), nf, blk / nf, res.join(","), st.nsym, tdg * 4.25 * 1.03 / 8e6, ctab.len(), st.sesc, top.iter().take(30).map(|(p, v)| { let n = pf[p] as f64; let (bb, c) = best(v, false); format!("{{\"p\":{p},\"facts\":{},\"bits_per_fact\":{:.3},\"cls\":{c},\"slot\":{:.3},\"pay\":{:.3},\"aux\":{:.3},\"nocls_bits\":{:.3}}}", pf[p], bb / n, v[if c { 1 } else { 0 }] / n, v[if c { 5 } else { 4 }] / n + if c { v[6] / n } else { 0.0 }, v[8] / n, (v[0] + v[4] + v[7] + v[8]) / n) }).collect::<Vec<_>>().join(","), t0.elapsed().as_secs_f64());
    println!("{o}");
    fs::write(arg(a, "--out", "data/store2/results/fstat.json"), o)
}
fn tup(w: &W3, f: usize, sid: &[u64]) -> Vec<u64> { match w.fk.s()[f] { 1 | 2 => vec![w.fv.s()[f]], 3 => vec![w.fv.s()[f], if w.fx.s()[f] == u32::MAX { 0 } else { w.fx.s()[f] as u64 + 1 }], _ => vec![sid[f]] } }
fn rk2t(r: u128) -> Vec<u64> { let (k, x, v) = ((r >> 104) as u8, ((r >> 64) & 0xFFFF_FFFF) as u32, r as u64); match k { 1 | 2 => vec![v], 3 => vec![v, if x == u32::MAX { 0 } else { x as u64 + 1 }], _ => vec![v] } }
fn wid(m: u64) -> u64 { let mut d = 1; while 17u64.pow(d as u32) <= m { d += 1 } d }
fn lists(m: HashMap<(u16, u32), HashMap<Vec<u64>, u64>>) -> (Vec<(u16, u32, Vec<Vec<u64>>)>, HashMap<(u16, u32), HashMap<Vec<u64>, u64>>, HashMap<(u16, u32), u32>) {
    let mut tot: HashMap<(u16, u32), u64> = HashMap::new();
    m.iter().for_each(|(k, v)| *tot.entry(*k).or_default() += v.values().sum::<u64>());
    let mut keys: Vec<(u16, u32)> = m.keys().copied().collect();
    keys.sort_by_key(|k| (k.0, std::cmp::Reverse(tot[k]), k.1));
    let (mut out, mut pos, mut cid) = (vec![], HashMap::new(), HashMap::new());
    let mut pp = u16::MAX;
    let mut ci = 0u32;
    for k in keys { if k.0 != pp { pp = k.0; ci = 0 } let mut v: Vec<(Vec<u64>, u64)> = m[&k].iter().map(|(a, b)| (a.clone(), *b)).collect(); v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0))); let mut o: Vec<Vec<u64>> = vec![]; let mut r = 0usize; while r < v.len() { let (bb, _) = bkt(r as u64); let z = ((1usize << (bb + 1)) - 1).min(v.len()); let mut c: Vec<Vec<u64>> = v[r..z].iter().map(|x| x.0.clone()).collect(); c.sort(); o.extend(c); r = z } let pm: HashMap<Vec<u64>, u64> = o.iter().enumerate().map(|(i, t)| (t.clone(), i as u64)).collect(); pos.insert(k, pm); cid.insert(k, ci); out.push((k.0, k.1, o)); ci += 1 }
    (out, pos, cid)
}
fn build2(a: &[String]) -> io::Result<()> {
    let (src, out, pd, bd, bc, kk, tc) = (arg(a, "--v3", "data/wikifact/v3/store"), arg(a, "--out", "data/store2/drive"), arg(a, "--pieces", "data/store2/pieces"), arg(a, "--build", "data/store1/build"), arg(a, "--bc", "32").parse::<usize>().unwrap(), arg(a, "--k", "8").parse::<usize>().unwrap(), arg(a, "--tc", "4").parse::<usize>().unwrap());
    let nmk = arg(a, "--namekeys", "data/store2/build/namekeys.bin");
    fs::create_dir_all(&out)?;
    let t0 = Instant::now();
    let lg = |m: &str| eprintln!("[{:.1}s] {m} rss={:.2}G", t0.elapsed().as_secs_f64(), rss().0 as f64 / 1e9);
    let w = w3(&src)?;
    let ne = w.qid.n;
    let b = Bpe::load(&pd)?;
    let nme = nmenc(&format!("{bd}/nmenc.bin"), ne)?;
    let st = strs(&w);
    let smap: HashMap<&[u8], u64> = st.iter().enumerate().map(|(i, s)| (*s, i as u64)).collect();
    let (sid, snew) = sidmap(&w);
    let mut pn: Vec<u16> = w.fp.s().iter().copied().chain(w.qp.s().iter().copied()).collect::<HashSet<u16>>().into_iter().collect();
    pn.sort();
    let pidx: HashMap<u16, u16> = pn.iter().enumerate().map(|(i, &p)| (p, i as u16)).collect();
    let np = pn.len();
    let mut pm = Pm { pnum: pn.clone(), pkind: vec![0; np], cls: vec![false; np], rules: vec![vec![]; np], cpi: CP2.iter().filter_map(|p| pidx.get(p).copied()).collect(), p31: pidx[&31], nstr: st.len() as u64, qkind: vec![0; np] };
    w.fp.s().iter().zip(w.fk.s()).for_each(|(&p, &k)| pm.pkind[pidx[&p] as usize] = k);
    w.qp.s().iter().zip(w.qk.s()).for_each(|(&p, &k)| pm.qkind[pidx[&p] as usize] = k);
    (0..np).for_each(|i| pm.cls[i] = pm.pkind[i] == 1 && pn[i] != 31 && !CP2.contains(&pn[i]));
    RU2.iter().for_each(|&(t, v, s)| if let (Some(&ti), Some(&vi)) = (pidx.get(&t), pidx.get(&v)) { pm.rules[ti as usize].push((vi, if s == 0 { u16::MAX } else { pidx[&s] })) });
    assert!((0..w.fv.n).all(|f| qc(&w, f) < 128), "qualifier count >= 128");
    lg("load");
    let gp = |e: usize| -> Vec<(u16, usize, usize)> { groups(&w, e).into_iter().map(|(p, s, n)| (pidx[&p], s, n)).collect() };
    let ctry: Vec<u32> = (0..ne).into_par_iter().map(|e| { let g = gp(e); pm.cpi.iter().find_map(|&c| g.iter().find(|x| x.0 == c && w.fk.s()[x.1] == 1).map(|x| w.fv.s()[x.1] as u32)).unwrap_or(u32::MAX) }).collect();
    let ckey = |g: &[(u16, usize, usize)], pi: u16| -> u64 { if pi == pm.p31 { sigk(&g.iter().map(|x| (x.0, x.2 as u32)).collect::<Vec<_>>()) } else { g.iter().find(|x| x.0 == pm.p31).map_or(0, |x| w.fv.s()[x.1] + 1) } };
    let mut ck: Vec<(u16, u64, u128)> = (0..ne).into_par_iter().flat_map_iter(|e| { let g = gp(e); g.iter().flat_map(|x| (x.1..x.1 + x.2).map(move |f| (x.0, f))).map(|(pi, f)| (pi, ckey(&g, pi), rk(&w, f, &sid))).filter(|x| x.0 == pm.p31 || x.1 != 0).collect::<Vec<_>>() }).collect();
    ck.par_sort_unstable();
    let mut ctab: HashMap<(u16, u64), Vec<Vec<u64>>> = HashMap::new();
    let mut i = 0;
    while i < ck.len() { let mut j = i; while j < ck.len() && (ck[j].0, ck[j].1) == (ck[i].0, ck[i].1) { j += 1 } if j - i >= tc { let mut c: Vec<(usize, u128)> = vec![]; let mut s = i; while s < j { let mut z = s; while z < j && ck[z].2 == ck[s].2 { z += 1 } c.push((z - s, ck[s].2)); s = z } c.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1))); let l: Vec<Vec<u64>> = c.iter().take(kk).filter(|x| x.0 >= 2).map(|x| rk2t(x.1)).collect(); if !l.is_empty() { ctab.insert((ck[i].0, ck[i].1), l); } } i = j }
    drop(ck);
    lg(&format!("ctx tables {}", ctab.len()));
    let nf = w.fv.n;
    let nbk = ne.div_ceil(bc);
    let plan: Vec<Vec<(u32, u8, i8, u32, u32)>> = (0..nbk).into_par_iter().map(|bb| {
        let mut o = vec![];
        let mut rec: HashMap<u16, Vec<u32>> = HashMap::new();
        for e in bb * bc..((bb + 1) * bc).min(ne) {
            let g = gp(e);
            let rk2 = |f: usize| rk(&w, f, &sid);
            let rmatch = |pi: u16, i: usize| -> Option<(u16, u32, u32)> { let r = pm.rules[pi as usize].iter().find(|v| g.iter().any(|x| x.0 == v.0))?; let vg = g.iter().find(|x| x.0 == r.0)?; let gg = g.iter().find(|x| x.0 == pi)?; let tv = rk2(gg.1 + i); if r.1 == u16::MAX { let f2 = vg.1 + i.min(vg.2 - 1); (rk2(f2) == tv).then_some((r.0, f2 as u32, u32::MAX)) } else { (w.fk.s()[vg.1] == 1).then_some(())?; let t = w.fv.s()[vg.1] as usize; let tg = gp(t).into_iter().find(|x| x.0 == r.1)?; let f2 = tg.1 + i.min(tg.2 - 1); (rk2(f2) == tv).then_some((r.0, vg.1 as u32, f2 as u32)) } };
            let mut nocls: Vec<u16> = vec![];
            if let Some(cp) = pm.cpi.iter().find(|&&c| g.iter().any(|x| x.0 == c && w.fk.s()[x.1] == 1)) { let mut q = *cp; while let Some((v, _, _)) = rmatch(q, 0) { if nocls.contains(&v) { break } nocls.push(v); q = v } }
            let cfact = pm.cpi.iter().find_map(|&c| g.iter().find(|x| x.0 == c && w.fk.s()[x.1] == 1).map(|x| x.1 as u32)).unwrap_or(u32::MAX);
            let p31f = g.iter().find(|x| x.0 == pm.p31).map_or(u32::MAX, |x| x.1 as u32);
            for &(pi, a0, n) in &g {
                let pu = pi as usize;
                for i in 0..n {
                    let f = a0 + i;
                    let tv = rk2(f);
                    let k = w.fk.s()[f];
                    let ml = rec.get(&pi).map_or(0, |v| v.len().min(8));
                    let ct = if pi == pm.p31 || p31f != u32::MAX { ctab.get(&(pi, ckey(&g, pi))) } else { None };
                    let d = w.fv.s()[f] as i64 - e as i64;
                    let tt = tup(&w, f, &sid);
                    let s = if k >= 4 && snew[f] { (S_NEWS, 0i8, u32::MAX, u32::MAX) } else if let Some((_, d0, d1)) = rmatch(pi, i) { (S_RULE, 0, d0, d1) } else if let Some(j) = (0..ml).find(|&j| rk2(rec[&pi][j] as usize) == tv) { (1 + j as u8, 0, rec[&pi][j], u32::MAX) } else if let Some(j) = ct.and_then(|c| c.iter().position(|x| *x == tt)) { (S_CTX0 + j as u8, 0, if pi == pm.p31 { u32::MAX } else { p31f }, u32::MAX) } else if k == 1 && d != 0 && d.abs() <= 8 { (S_NEAR, d as i8, u32::MAX, u32::MAX) } else if pm.cls[pu] && ctry[e] != u32::MAX && ctry[w.fv.s()[f] as usize] == ctry[e] && !(i == 0 && nocls.contains(&pi)) { (S_CLS, 0, cfact, u32::MAX) } else { (S_ESC, 0, u32::MAX, u32::MAX) };
                    o.push((f as u32, s.0, s.1, s.2, s.3));
                }
            }
            g.iter().for_each(|&(pi, a0, n)| { let v = rec.entry(pi).or_default(); (0..n).rev().for_each(|i| v.insert(0, (a0 + i) as u32)); v.truncate(8) });
        }
        o
    }).collect();
    let (mut ps, mut near, mut dep) = (vec![S_ESC; nf], vec![0i8; nf], vec![[u32::MAX; 2]; nf]);
    plan.into_iter().flatten().for_each(|(f, s, n, d0, d1)| { ps[f as usize] = s; near[f as usize] = n; dep[f as usize] = [d0, d1] });
    lg("plan");
    let mut col = vec![0u8; nf];
    let mut down = 0u64;
    for r0 in 0..nf { if col[r0] != 0 { continue } let mut stk: Vec<(usize, usize)> = vec![(r0, 0)]; col[r0] = 1; loop { let Some(top) = stk.last_mut() else { break }; let u = top.0; if top.1 >= 2 { col[u] = 2; stk.pop(); continue } let v = dep[u][top.1]; top.1 += 1; if v == u32::MAX { continue } let v = v as usize; match col[v] { 0 => { col[v] = 1; stk.push((v, 0)) } 1 => { ps[u] = S_ESC; dep[u] = [u32::MAX; 2]; down += 1 } _ => {} } } }
    drop(col);
    lg(&format!("cycles broken {down}"));
    let mut lm: HashMap<(u16, u32), HashMap<Vec<u64>, u64>> = HashMap::new();
    let vcl = |f: usize| -> u32 { let pi = pidx[&w.fp.s()[f]]; if pm.cls[pi as usize] { ctry[w.fv.s()[f] as usize] } else { u32::MAX } };
    (0..nf).filter(|&f| (ps[f] == S_CLS || ps[f] == S_ESC) && w.fk.s()[f] < 4).for_each(|f| { *lm.entry((pidx[&w.fp.s()[f]], vcl(f))).or_default().entry(tup(&w, f, &sid)).or_default() += 1 });
    let (cbl, cpos, ccid) = lists(lm);
    let mut ql: HashMap<(u16, u32), HashMap<Vec<u64>, u64>> = HashMap::new();
    (0..w.qv.n).filter(|&j| w.qk.s()[j] < 4).for_each(|j| *ql.entry((pidx[&w.qp.s()[j]], u32::MAX)).or_default().entry(vec![w.qv.s()[j]]).or_default() += 1);
    let (qbl, qpos, _) = lists(ql);
    lg(&format!("codebooks lists {} entries {} qual lists {}", cbl.len(), cbl.iter().map(|x| x.2.len()).sum::<usize>(), qbl.len()));
    let pcls: Vec<u8> = { let c: Vec<[u64; 6]> = (0..ne).into_par_iter().fold(|| vec![[0u64; 6]; NV as usize + 1], |mut c, e| { let (l, t) = (lab(&w, e), tit(&w, e)); let (tk, ex) = tkind(l, t); for x in [tf(&b, l, true), if tk >= 2 { tf(&b, &ex, true) } else { Tf::E }] { if let Tf::W(ws) = x { ws.iter().for_each(|t| match t { Tw::P(p, o, _) => c[p[0] as usize][*o as usize] += 1, Tw::Raw(p) => c[p[0] as usize][5] += 1 }) } } c }).reduce(|| vec![[0u64; 6]; NV as usize + 1], |mut a, b| { a.iter_mut().zip(&b).for_each(|(x, y)| (0..6).for_each(|i| x[i] += y[i])); a }); c.iter().map(|x| (0..6).max_by_key(|&i| (x[i], 6 - i)).unwrap() as u8).collect() };
    lg("pcls");
    let mker = |e: usize| -> (Er, Vec<(u8, Tf)>, Qr) {
        let g = gp(e);
        let (l, t) = (lab(&w, e), tit(&w, e));
        let (tk, ex) = tkind(l, t);
        let f: Vec<Vec<Fs>> = g.iter().map(|&(pi, a0, n)| (a0..a0 + n).map(|f| { let s = ps[f]; let kd = w.fk.s()[f]; let (cls, pos) = if (s == S_CLS || s == S_ESC) && kd < 4 { let c = vcl(f); (if s == S_ESC && pm.cls[pi as usize] { ccid[&(pi, c)] } else { 0 }, cpos[&(pi, c)][&tup(&w, f, &sid)]) } else if s == S_ESC { (0, sid[f]) } else { (0, 0) }; Fs { s, aux: w.fr.s()[f] | (qc(&w, f) as u8) << 1, near: near[f], cls, pos } }).collect()).collect();
        let q: Qr = g.iter().map(|&(_, a0, n)| (a0..a0 + n).map(|f| (w.fqo.s()[f] as usize..w.fqo.s()[f + 1] as usize).map(|j| { let qp = pidx[&w.qp.s()[j]]; (qp, if w.qk.s()[j] >= 4 { smap[sbytes(&w, w.qv.s()[j])] } else { qpos[&(qp, u32::MAX)][&vec![w.qv.s()[j]]] }) }).collect()).collect()).collect();
        let r = Er { gap: if e % bc == 0 { 0 } else { (w.qid.s()[e] - w.qid.s()[e - 1] - 1) as u64 }, sl: w.sl.s()[e] as u64, g: g.iter().map(|x| (x.0, x.2 as u32)).collect(), f, tk, lab: tf(&b, l, true), tx: if tk >= 2 { tf(&b, &ex, true) } else { Tf::E } };
        (r, nme[e].iter().map(|(k, x)| (*k, if *k == 2 { tf(&b, x, false) } else { Tf::E })).collect(), q)
    };
    let pass = |k: Option<&Tabs>| -> Cm { let c1 = (0..nbk).into_par_iter().fold(|| Cnt { c: HashMap::default(), k }, |mut cd, bb| { let mut bs = Bst::new(np); for e in bb * bc..((bb + 1) * bc).min(ne) { let (mut r, mut n, mut q) = mker(e); chdr(&mut cd, &mut bs, &mut r, e == bb * bc); crec(&mut cd, &pm, &mut bs, &mut r); clab(&mut cd, &b, &pcls, &mut r); cnames(&mut cd, &b, lwords(&b, lab(&w, e)), &mut n); cqual(&mut cd, &pm, &r, &mut q) } cd }).map(|c| c.c).reduce(HashMap::default, cmerge); let c2 = st.par_iter().fold(|| Cnt { c: HashMap::default(), k }, |mut cd, s| { let mut x = tf(&b, s, true); ctf(&mut cd, &b, &pcls, FS, &mut x); cd }).map(|c| c.c).reduce(HashMap::default, cmerge); cmerge(c1, c2) };
    let p1 = p1tabs(&pass(None), arg(a, "--t", "8").parse().unwrap(), 8191);
    lg(&format!("p1 tables {}", p1.len()));
    let (mut cbl, mut qbl) = (cbl, qbl);
    let mut cbc = |cd: &mut Cnt| { cbl.iter_mut().for_each(|(pi, _, l)| { let len = if pm.pkind[*pi as usize] == 3 { 2 } else { 1 }; (0..l.len()).step_by(CB2).for_each(|q| cbblk(cd, l, q, *pi as u64, len)) }); qbl.iter_mut().for_each(|(pi, _, l)| (0..l.len()).step_by(CB2).for_each(|q| cbblk(cd, l, q, 1024 + *pi as u64, 1))) };
    let mut c3 = Cnt { c: HashMap::default(), k: Some(&p1) };
    cbc(&mut c3);
    let tb = tabs(&cmerge(pass(Some(&p1)), c3.c), Some(&p1));
    drop(p1);
    lg(&format!("tables {}", tb.len()));
    let enc: Vec<(Vec<u8>, Vec<u8>, Vec<u8>, u32, [f64; 8], Vec<u8>, Vec<u8>)> = (0..nbk).into_par_iter().map(|bb| { let (mut c, mut n, mut q, mut l, mut hh) = (EncC { t: &tb, e: Enc::default(), c: 0, bits: [0.0; 8] }, EncC { t: &tb, e: Enc::default(), c: 0, bits: [0.0; 8] }, EncC { t: &tb, e: Enc::default(), c: 0, bits: [0.0; 8] }, EncC { t: &tb, e: Enc::default(), c: 0, bits: [0.0; 8] }, EncC { t: &tb, e: Enc::default(), c: 0, bits: [0.0; 8] }); let mut bs = Bst::new(np); let mut news = 0u32; for e in bb * bc..((bb + 1) * bc).min(ne) { let (mut r, mut nn, mut qq) = mker(e); news += r.f.iter().flatten().filter(|f| f.s == S_NEWS).count() as u32; chdr(&mut hh, &mut bs, &mut r, e == bb * bc); crec(&mut c, &pm, &mut bs, &mut r); clab(&mut l, &b, &pcls, &mut r); cnames(&mut n, &b, lwords(&b, lab(&w, e)), &mut nn); cqual(&mut q, &pm, &r, &mut qq) } let mut bits = c.bits; (3..5).for_each(|i| bits[i] = l.bits[i]); bits[0] = hh.bits[0]; (c.e.finish(), n.e.finish(), q.e.finish(), news, bits, l.e.finish(), hh.e.finish()) }).collect();
    let senc: Vec<Vec<u8>> = st.par_chunks(16).map(|ch| { let mut c = EncC { t: &tb, e: Enc::default(), c: 0, bits: [0.0; 8] }; ch.iter().for_each(|s| { let mut x = tf(&b, s, true); ctf(&mut c, &b, &pcls, FS, &mut x) }); c.e.finish() }).collect();
    lg("encoded");
    let cat = |v: Vec<&Vec<u8>>| -> (Vec<u8>, Vec<u64>) { let mut o = vec![]; let mut off = vec![]; v.into_iter().for_each(|x| { off.push(o.len() as u64); o.extend_from_slice(x) }); (o, off) };
    let (sc, oc) = cat(enc.iter().map(|x| &x.0).collect());
    let (sn, on) = cat(enc.iter().map(|x| &x.1).collect());
    let (sq, oq) = cat(enc.iter().map(|x| &x.2).collect());
    let (ss, os) = cat(senc.iter().collect());
    let (sl, ol) = cat(enc.iter().map(|x| &x.5).collect());
    let (sh, oh) = cat(enc.iter().map(|x| &x.6).collect());
    let rw = |off: &[u64]| -> u64 { wid(off.iter().enumerate().map(|(i, &o)| o - off[i / SBK as usize * SBK as usize]).max().unwrap_or(0)) };
    let fq0: Vec<u64> = (0..nbk).map(|bb| w.qid.s()[bb * bc] as u64).collect();
    let wq = wid((0..nbk).map(|bb| fq0[bb] - fq0[bb / SBK as usize * SBK as usize]).max().unwrap_or(0));
    let (wc, wn, wqq, wss, wl, wh) = (rw(&oc), rw(&on), rw(&oq), rw(&os), rw(&ol), rw(&oh));
    let mut sbase = vec![0u64; nbk];
    (1..nbk).for_each(|bb| sbase[bb] = sbase[bb - 1] + enc[bb - 1].3 as u64);
    let mut xc = vec![];
    (0..nbk).for_each(|bb| { let s0 = bb / SBK as usize * SBK as usize; if bb == s0 { fx(&mut xc, oc[bb], 8); fx(&mut xc, fq0[bb], 7); fx(&mut xc, sbase[bb], 5) } else { fx(&mut xc, oc[bb] - oc[s0], wc as usize); fx(&mut xc, fq0[bb] - fq0[s0], wq as usize) } });
    let xo = |off: &[u64], wr: u64| -> Vec<u8> { let mut x = vec![]; (0..off.len()).for_each(|bb| { let s0 = bb / SBK as usize * SBK as usize; if bb == s0 { fx(&mut x, off[bb], 8) } else { fx(&mut x, off[bb] - off[s0], wr as usize) } }); x };
    let mut meta = vec![];
    [ne as u64, np as u64, bc as u64, pm.nstr, pm.p31 as u64].iter().for_each(|&v| ev(&mut meta, v));
    (0..np).for_each(|i| { ev(&mut meta, pm.pnum[i] as u64); ev(&mut meta, pm.pkind[i] as u64); ev(&mut meta, pm.cls[i] as u64); ev(&mut meta, pm.qkind[i] as u64); ev(&mut meta, pm.rules[i].len() as u64); pm.rules[i].iter().for_each(|r| { ev(&mut meta, r.0 as u64); ev(&mut meta, r.1 as u64) }) });
    ev(&mut meta, pm.cpi.len() as u64);
    pm.cpi.iter().for_each(|&c| ev(&mut meta, c as u64));
    [wc, wq, wqq, wn, wss, wl, wh].iter().for_each(|&v| ev(&mut meta, v));
    let mut ctxd = vec![];
    let mut ck: Vec<&(u16, u64)> = ctab.keys().collect();
    ck.sort();
    ev(&mut ctxd, ck.len() as u64);
    let mut pk = (0u16, 0u64);
    ck.iter().for_each(|&&(pi, k)| { ev(&mut ctxd, (pi - pk.0) as u64); ev(&mut ctxd, if pi == pk.0 { k - pk.1 } else { k }); pk = (pi, k); let l = &ctab[&(pi, k)]; ev(&mut ctxd, l.len() as u64); l.iter().for_each(|t| { ev(&mut ctxd, t.len() as u64); t.iter().for_each(|&v| ev(&mut ctxd, v)) }) });
    let (mut cbd, mut cb, mut cbx) = (vec![], vec![], vec![]);
    ev(&mut cbd, cbl.len() as u64);
    let cbe = |l: &mut Vec<Vec<u64>>, c: u64, len: usize, o: &mut Vec<u8>, x: &mut Vec<u8>| (0..l.len()).step_by(CB2).for_each(|q| { let mut cd = EncC { t: &tb, e: Enc::default(), c: 0, bits: [0.0; 8] }; cbblk(&mut cd, l, q, c, len); fx(x, o.len() as u64, 8); o.extend(cd.e.finish()) });
    cbl.iter_mut().for_each(|(pi, c, l)| { ev(&mut cbd, *pi as u64); ev(&mut cbd, if *c == u32::MAX { 0 } else { *c as u64 + 1 }); ev(&mut cbd, l.len() as u64); cbe(l, *pi as u64, if pm.pkind[*pi as usize] == 3 { 2 } else { 1 }, &mut cb, &mut cbx) });
    let (mut qcbd, mut qcb, mut qcbx) = (vec![], vec![], vec![]);
    ev(&mut qcbd, qbl.len() as u64);
    qbl.iter_mut().for_each(|(pi, _, l)| { ev(&mut qcbd, *pi as u64); ev(&mut qcbd, l.len() as u64); cbe(l, 1024 + *pi as u64, 1, &mut qcb, &mut qcbx) });
    let hmin: u16 = arg(a, "--hot", "40").parse().unwrap();
    let mut hot = vec![];
    let mut hotx = vec![];
    let he: Vec<usize> = (0..ne).filter(|&e| w.sl.s()[e] >= hmin).collect();
    ev(&mut hotx, he.len() as u64);
    let mut pe = 0usize;
    he.iter().for_each(|&e| { let l0 = hot.len(); let ws = match tf(&b, lab(&w, e), true) { Tf::W(ws) => ws, x => vec![Tw::Raw(tfp(&x))] }; ev(&mut hot, ws.len() as u64); ws.iter().for_each(|t| { let (p, op, m) = match t { Tw::P(p, o, m) => (p, *o, m.clone()), Tw::Raw(p) => (p, 5, vec![]) }; ev(&mut hot, p.len() as u64); p.iter().for_each(|&x| fx(&mut hot, x as u64, 4)); hot.push(op); if op == 3 || op == 4 { ev(&mut hot, m.len() as u64); hot.extend_from_slice(&m) } }); ev(&mut hotx, (e - pe) as u64); ev(&mut hotx, (hot.len() - l0) as u64); pe = e });
    let mut pg = PgW::new();
    let secs: Vec<(&str, Vec<u8>)> = vec![("meta", meta), ("tabs", tser(&tb)), ("bpe", bpe_ser(&b, 1975)), ("pcls", pcls.clone()), ("ctx", ctxd), ("cbd", cbd), ("cb", cb), ("cbx", cbx), ("qcbd", qcbd), ("qcb", qcb), ("qcbx", qcbx), ("sc", sc), ("xc", xc), ("sn", sn), ("xn", xo(&on, wn)), ("sq", sq), ("xq", xo(&oq, wqq)), ("ss", ss), ("xs", xo(&os, wss)), ("sl", sl), ("xl", xo(&ol, wl)), ("sh", sh), ("xh", xo(&oh, wh)), ("hot", hot), ("hotx", hotx)];
    let mut sizes: Vec<(String, u64)> = vec![];
    secs.iter().for_each(|(n, d)| { sizes.push((n.to_string(), (d.len() as f64 * 4.25 / 8.0).ceil() as u64)); pg.add(n, d) });
    let (pb, ok, pages) = pg.save(&format!("{out}/store2.gf17pg"))?;
    lg(&format!("container {pb} bytes ok {ok} pages {pages}"));
    let nkb = fs::read(&nmk)?;
    let mut km: HashMap<u64, u64> = HashMap::new();
    let mut o = 0usize;
    while o < nkb.len() { let h = u64::from_le_bytes(nkb[o..o + 8].try_into().unwrap()); let l = u16::from_le_bytes([nkb[o + 8], nkb[o + 9]]) as usize; let s = std::str::from_utf8(&nkb[o + 10..o + 10 + l]).unwrap(); km.insert(h, kh(&b.text(s))); o += 10 + l }
    let mut kk2: Vec<u64> = km.values().copied().collect();
    kk2.par_sort_unstable();
    let kdup = kk2.windows(2).filter(|x| x[0] == x[1]).count();
    let (ah, ae, ak) = (w.ah.s(), w.ae.s(), w.ak.s());
    let mut stt: Vec<usize> = (0..ah.len()).filter(|&i| i == 0 || ah[i] != ah[i - 1]).collect();
    stt.push(ah.len());
    let nk: Vec<(u64, [u8; 7])> = stt.par_windows(2).flat_map_iter(|x| { let key = km[&ah[x[0]]]; (x[0]..x[1]).map(move |i| { let mut d = vec![]; fx(&mut d, ae[i] as u64, 6); d.push(ak[i]); (mixk(key, (i - x[0]) as u64), d.try_into().unwrap()) }) }).collect();
    lg(&format!("name keys {} piece-key dups {kdup}", nk.len()));
    let mut nsz = vec![];
    for f in arg(a, "--fd", "2,3").split(',').map(|x| x.parse::<usize>().unwrap()) {
        let p = format!("{out}/name_f{f}.gf17rib");
        let by = match f { 0 => nb::<7>(&p, &nk, 0)?, 1 => nb::<8>(&p, &nk, 1)?, 2 => nb::<9>(&p, &nk, 2)?, _ => nb::<10>(&p, &nk, 3)? };
        let t = Nm::open(&p, f)?;
        let add: Vec<u64> = stt.par_windows(2).filter_map(|x| { let (k, n, last) = (km[&ah[x[0]]], (x[1] - x[0]) as u64, ae[x[1] - 1]); let k2 = mixk(k, n); t.row(k2).filter(|d| { let e = d[..6].iter().rev().fold(0u32, |a, &v| a * 17 + v as u32); e != END && (e as usize) < ne && e > last && d[6] <= 3 }).map(|_| k2) }).collect();
        let ep = format!("{out}/name_f{f}_end.gf17rib");
        let rows: Vec<(u64, [u8; 16])> = add.iter().map(|&k| (k, [16u8; 16])).collect();
        let eb = if f > 0 { layered_by::<16, [u8; 16]>(&ep, &rows, drow::<16>, 64, 9, 65536, 0.08, seed(&ep), None, 0.0, None)?.2 } else { 0 };
        nsz.push(format!("{{\"fd\":{f},\"bytes\":{},\"end_markers\":{}}}", by + eb, add.len()));
        sizes.push((format!("name_f{f}"), by + eb));
    }
    let fs_sz = |p: &str| fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let j = format!("{{\"entities\":{ne},\"facts\":{nf},\"props\":{np},\"block\":{bc},\"k\":{kk},\"tc\":{tc},\"ctx_entries\":{},\"cycles_broken\":{down},\"codebook_lists\":{},\"codebook_entries\":{},\"tables\":{},\"strings\":{},\"hot_entities\":{},\"container_bytes\":{pb},\"container_file_bytes\":{},\"roundtrip_bitexact\":{ok},\"container_pages\":{pages},\"name_keys\":{},\"piece_key_dups\":{kdup},\"name_tables\":[{}],\"widths\":[{wc},{wq},{wqq},{wn},{wss}],\"components\":{{{}}},\"core_bits\":{:?},\"core_blocks\":{nbk},\"slot_counts\":{:?},\"build_s\":{:.1},\"peak_rss\":{}}}", ctab.len(), cbl.len(), cbl.iter().map(|x| x.2.len()).sum::<usize>(), tb.len(), st.len(), he.len(), fs_sz(&format!("{out}/store2.gf17pg")), nk.len(), nsz.join(","), sizes.iter().map(|(n, b)| format!("\"{n}\":{b}")).collect::<Vec<_>>().join(","), (0..5).map(|i| enc.iter().map(|x| x.4[i]).sum::<f64>().round() as u64).collect::<Vec<_>>(), (0..21u8).map(|s| ps.iter().filter(|&&x| x == s).count()).collect::<Vec<_>>(), t0.elapsed().as_secs_f64(), rss().1);
    println!("{j}");
    fs::write(format!("{out}/build.json"), j)
}
fn parity2(a: &[String]) -> io::Result<()> {
    let (src, dir, fd, bd) = (arg(a, "--v3", "data/wikifact/v3/store"), arg(a, "--dir", "data/store2/drive"), arg(a, "--fd", "2").parse::<usize>().unwrap(), arg(a, "--build", "data/store1/build"));
    let t0 = Instant::now();
    let w = w3(&src)?;
    let s = S2::open(&dir, fd)?;
    eprintln!("open {:.2}s", t0.elapsed().as_secs_f64());
    let ne = w.qid.n;
    let nme = nmenc(&format!("{bd}/nmenc.bin"), ne)?;
    let txt = |k: u8, v: u64| if k >= 4 { sbytes(&w, v).to_vec() } else { vec![] };
    let nbk = ne.div_ceil(s.bc as usize);
    let lim: usize = arg(a, "--nb", &nbk.to_string()).parse().unwrap();
    let r: Vec<[u64; 12]> = (0..lim.min(nbk)).into_par_iter().map(|bb| {
        let mut c = [0u64; 12];
        let qb = s.qblock(bb as u64);
        let nbl = s.nblock(bb as u64);
        for e in bb * s.bc as usize..((bb + 1) * s.bc as usize).min(ne) {
            let k = e % s.bc as usize;
            let g = groups(&w, e);
            let sg = s.groups(e as u64);
            c[0] += (g.len() != sg.len() || g.iter().zip(&sg).any(|(x, y)| x.0 != y.0 || x.2 as u32 != y.1)) as u64;
            for (gi, x) in g.iter().enumerate() {
                let v = s.values(e as u64, x.0);
                c[10] += x.2 as u64;
                for i in 0..x.2 {
                    let f = x.1 + i;
                    let kd = w.fk.s()[f];
                    let ok = v.len() == x.2 && v.get(i).is_some_and(|t| { let fl = s.flat(t); fl.0 == kd && fl.3 == w.fr.s()[f] && fl.4 as u64 == qc(&w, f) && fl.2 == if kd == 3 { w.fx.s()[f] } else { u32::MAX } && if kd >= 4 { s.string(fl.1) == txt(kd, w.fv.s()[f]) } else { fl.1 == w.fv.s()[f] } });
                    c[1] += !ok as u64;
                    let (qa, qz) = (w.fqo.s()[f] as usize, w.fqo.s()[f + 1] as usize);
                    let q = qb[k].get(gi).and_then(|x| x.get(i)).cloned().unwrap_or_default();
                    let qok = q.len() == qz - qa && q.iter().zip(qa..qz).all(|(&(qp, v), j)| { let qk = s.pm.qkind[qp as usize]; s.pm.pnum[qp as usize] == w.qp.s()[j] && qk == w.qk.s()[j] && if qk >= 4 { s.string(v) == txt(qk, w.qv.s()[j]) } else { s.cbent2("qcb", s.qcb[&qp].0, v, 1024 + qp as u64, 1) == vec![w.qv.s()[j]] } });
                    c[2] += !qok as u64;
                    c[11] += (qz - qa) as u64;
                }
            }
            c[3] += (s.label(e as u64) != lab(&w, e) || s.title(e as u64) != tit(&w, e)) as u64;
            c[4] += (nbl[k] != nme[e]) as u64;
            c[5] += (s.sitelinks(e as u64) != w.sl.s()[e] as u64) as u64;
            c[6] += (s.qid(e as u64) != w.qid.s()[e] as u64) as u64;
            c[7] += (s.ix(w.qid.s()[e] as u64) != e as i64) as u64;
            if let Some(h) = s.hotlabel(e as u64) { c[8] += 1; c[9] += (h != lab(&w, e)) as u64 }
        }
        if bb % 50000 == 0 { s.clear() }
        c
    }).collect();
    let t = r.iter().fold([0u64; 12], |a, b| std::array::from_fn(|i| a[i] + b[i]));
    eprintln!("entities done {:.1}s", t0.elapsed().as_secs_f64());
    let nkb = fs::read(arg(a, "--namekeys", "data/store2/build/namekeys.bin"))?;
    let mut km: HashMap<u64, String> = HashMap::new();
    let mut o = 0usize;
    while o < nkb.len() { let h = u64::from_le_bytes(nkb[o..o + 8].try_into().unwrap()); let l = u16::from_le_bytes([nkb[o + 8], nkb[o + 9]]) as usize; km.insert(h, String::from_utf8(nkb[o + 10..o + 10 + l].to_vec()).unwrap()); o += 10 + l }
    let (ah, ae, ak) = (w.ah.s(), w.ae.s(), w.ak.s());
    let mut stt: Vec<usize> = (0..ah.len()).filter(|&i| i == 0 || ah[i] != ah[i - 1]).collect();
    stt.push(ah.len());
    let em = stt.par_windows(2).filter(|x| s.ents(s.key(&km[&ah[x[0]]])) != (x[0]..x[1]).map(|i| (ae[i], ak[i])).collect::<Vec<_>>()).count();
    let ks: HashSet<u64> = km.values().map(|x| s.key(x)).collect();
    let mut rr = 99u64;
    let probes: Vec<u64> = (0..2_000_000).map(|_| adam_torus::sm(&mut rr)).filter(|h| !ks.contains(h)).collect();
    let fp = probes.par_iter().filter(|&&h| !s.ents(h).is_empty()).count();
    let full = lim >= nbk;
    let o = format!("{{\"fd\":{fd},\"full\":{full},\"entities\":{},\"facts_checked\":{},\"quals_checked\":{},\"parity_groups_mismatch\":{},\"parity_facts_mismatch\":{},\"parity_quals_mismatch\":{},\"parity_labels_mismatch\":{},\"parity_names_mismatch\":{},\"parity_strings_mismatch\":{},\"parity_sitelinks_mismatch\":{},\"parity_qid_mismatch\":{},\"parity_ix_mismatch\":{},\"hot_labels\":{},\"parity_hot_mismatch\":{},\"name_groups\":{},\"parity_ents_mismatch\":{em},\"absent_name_probes\":{},\"absent_name_fp\":{fp},\"absent_name_fpr\":\"{:.3e}\",\"sec\":{:.1}}}", (lim.min(nbk) * s.bc as usize).min(ne), t[10], t[11], t[0], t[1], t[2], t[3], t[4], t[1] + t[2], t[5], t[6], t[7], t[8], t[9], stt.len() - 1, probes.len(), fp as f64 / probes.len() as f64, t0.elapsed().as_secs_f64());
    println!("{o}");
    fs::write(arg(a, "--out", &format!("{dir}/parity_f{fd}.json")), o)
}
fn bench2(a: &[String]) -> io::Result<()> {
    let (src, dir, fd, n) = (arg(a, "--v3", "data/wikifact/v3/store"), arg(a, "--dir", "data/store2/drive"), arg(a, "--fd", "2").parse::<usize>().unwrap(), arg(a, "--n", "200000").parse::<usize>().unwrap());
    let w = w3(&src)?;
    let t0 = Instant::now();
    let s = S2::open(&dir, fd)?;
    let open_ms = t0.elapsed().as_secs_f64() * 1e3;
    let ne = w.qid.n;
    let mut r = 7u64;
    let fk: Vec<(u64, u16)> = (0..n).map(|_| { let f = adam_torus::sm(&mut r) as usize % w.fv.n; (w.fsub.s()[f] as u64, w.fp.s()[f]) }).collect();
    let ents: Vec<u64> = (0..n).map(|_| adam_torus::sm(&mut r) % ne as u64).collect();
    let tm = |f: &dyn Fn() -> u64, k: usize| { let t = Instant::now(); let x = f(); (t.elapsed().as_secs_f64() * 1e9 / k as f64, x) };
    let run = |k: usize, clr: bool| { if clr { s.clear() } let v = tm(&|| fk[..k].iter().map(|&(e, p)| { if clr { s.clear() } s.values(e, p).len() as u64 }).sum(), k); let l = tm(&|| ents[..k].iter().map(|&e| { if clr { s.clear() } s.label(e).len() as u64 }).sum(), k); let h = tm(&|| ents[..k].iter().map(|&e| s.hotlabel(e).map_or(0, |x| x.len() as u64)).sum(), k); let g = tm(&|| ents[..k].iter().map(|&e| { if clr { s.clear() } s.groups(e).len() as u64 }).sum(), k); (v.0, l.0, h.0, g.0) };
    let warm = run(n, false);
    let unc = run(n.min(20000), true);
    s.cold();
    let cold = run(2000, true);
    let o = format!("{{\"fd\":{fd},\"open_ms\":{open_ms:.1},\"values_ns_cached\":{:.0},\"label_ns_cached\":{:.0},\"hotlabel_ns\":{:.0},\"groups_ns_cached\":{:.0},\"values_ns_uncached_block\":{:.0},\"label_ns_uncached_block\":{:.0},\"groups_ns_uncached_block\":{:.0},\"values_ns_cold\":{:.0},\"label_ns_cold\":{:.0},\"rss\":\"{:?}\"}}", warm.0, warm.1, warm.2, warm.3, unc.0, unc.1, unc.3, cold.0, cold.1, rss());
    println!("{o}");
    fs::write(format!("{dir}/bench_f{fd}.json"), o)
}
fn bblk(a: &[String]) -> io::Result<()> {
    let s = S2::open(&arg(a, "--dir", "data/store2/drive"), 2)?;
    let mut r = 5u64;
    let bs: Vec<u64> = (0..2000).map(|_| adam_torus::sm(&mut r) % s.nb).collect();
    bs.iter().for_each(|&b| { s.block(b); });
    s.clear();
    let t = Instant::now();
    let mut ns = 0usize;
    bs.iter().for_each(|&b| { let x = s.block(b); ns += x.r.iter().map(|e| e.g.len() + e.f.iter().map(|v| v.len()).sum::<usize>()).sum::<usize>() });
    let d = t.elapsed().as_secs_f64() * 1e6 / bs.len() as f64;
    let th = Instant::now();
    (0..2000).for_each(|_| { s.clear(); s.block(bs[0]); });
    eprintln!("same block us {:.1}", th.elapsed().as_secs_f64() * 1e6 / 2000.0);
    let tb = Instant::now();
    let mut k = 0u64;
    bs.iter().for_each(|&b| { let mut cd = DecC { t: &s.t, d: RDec::new(&s.g, 0) }; for _ in 0..1000 { k += cd.s(M_SL, 0, 0) as u64 } let _ = b; });
    let d2 = tb.elapsed().as_secs_f64() * 1e9 / (bs.len() * 1000) as f64;
    println!("block_us {d:.1} groups_facts_per_block {:.1} sym_ns {d2:.1} k {k}", ns as f64 / bs.len() as f64);
    Ok(())
}
fn main() -> io::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) { Some("tstat") => tstat(&a), Some("fstat") => fstat(&a), Some("build") => build2(&a), Some("parity") => parity2(&a), Some("bench") => bench2(&a), Some("bblk") => bblk(&a), _ => { eprintln!("usage: store2 tstat|fstat ..."); Ok(()) } }
}
