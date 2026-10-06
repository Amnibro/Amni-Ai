use adam_gf17::{npy, rss};
use adam_store1::*;
use adam_torus::{drow, keyof, layered_by, sm};
use memmap2::Mmap;
use rayon::prelude::*;
use std::{collections::{HashMap, HashSet}, fs, io, time::Instant};
struct V<T> { m: Mmap, o: usize, n: usize, _t: std::marker::PhantomData<T> }
impl<T: Copy> V<T> { fn s(&self) -> &[T] { unsafe { std::slice::from_raw_parts(self.m[self.o..].as_ptr() as *const T, self.n) } } }
fn ld<T: Copy>(d: &str, n: &str) -> io::Result<V<T>> { let (_, sh, m, o) = npy(&format!("{d}/{n}.npy"))?; Ok(V { n: sh.iter().product(), m, o, _t: std::marker::PhantomData }) }
fn raw(p: &str) -> io::Result<Mmap> { unsafe { Mmap::map(&fs::File::open(p)?) } }
fn arg(a: &[String], k: &str, d: &str) -> String { a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned().unwrap_or(d.into()) }
fn js(kv: &[(&str, String)]) -> String { format!("{{{}}}", kv.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect::<Vec<_>>().join(",")) }
struct W3 { fsub: V<u32>, qid: V<u32>, sl: V<u16>, fs: V<u32>, fnn: V<u32>, lo: V<u64>, to: V<u64>, fp: V<u16>, fr: V<u8>, fk: V<u8>, fv: V<u64>, fx: V<u32>, fqo: V<u32>, qp: V<u16>, qk: V<u8>, qv: V<u64>, ah: V<u64>, ae: V<u32>, ak: V<u8>, lab: Mmap, tit: Mmap, st: Mmap }
fn w3(d: &str) -> io::Result<W3> { Ok(W3 { fsub: ld(d, "f_s")?, qid: ld(d, "e_qid")?, sl: ld(d, "e_sl")?, fs: ld(d, "e_fs")?, fnn: ld(d, "e_fn")?, lo: ld(d, "e_lo")?, to: ld(d, "e_to")?, fp: ld(d, "f_p")?, fr: ld(d, "f_r")?, fk: ld(d, "f_k")?, fv: ld(d, "f_v")?, fx: ld(d, "f_x")?, fqo: ld(d, "f_qo")?, qp: ld(d, "q_p")?, qk: ld(d, "q_k")?, qv: ld(d, "q_v")?, ah: ld(d, "a_h")?, ae: ld(d, "a_e")?, ak: ld(d, "a_k")?, lab: raw(&format!("{d}/e_lab.bin"))?, tit: raw(&format!("{d}/e_tit.bin"))?, st: raw(&format!("{d}/str.bin"))? }) }
fn sbytes(w: &W3, v: u64) -> &[u8] { let (o, l) = ((v >> 16) as usize, (v & 0xFFFF) as usize); &w.st[o..o + l] }
fn lab(w: &W3, e: usize) -> &[u8] { &w.lab[w.lo.s()[e] as usize..w.lo.s()[e + 1] as usize] }
fn tit(w: &W3, e: usize) -> &[u8] { &w.tit[w.to.s()[e] as usize..w.to.s()[e + 1] as usize] }
fn tkind(l: &[u8], t: &[u8]) -> (u8, Vec<u8>) { if t.is_empty() { (0, vec![]) } else if t == l { (1, vec![]) } else if t.len() > l.len() + 3 && t.starts_with(l) && &t[l.len()..l.len() + 2] == b" (" && t.ends_with(b")") { (2, t[l.len() + 2..t.len() - 1].to_vec()) } else { (3, t.to_vec()) } }
fn nmenc(p: &str, ne: usize) -> io::Result<Vec<Vec<(u8, Vec<u8>)>>> { let b = fs::read(p)?; let mut i = 0usize; Ok((0..ne).map(|_| { let n = u16::from_le_bytes([b[i], b[i + 1]]) as usize; i += 2; (0..n).map(|_| { let t = b[i]; i += 1; if t == 2 { let l = u16::from_le_bytes([b[i], b[i + 1]]) as usize; i += 2; i += l; (t, b[i - l..i].to_vec()) } else { (t, vec![]) } }).collect() }).collect()) }
fn groups(w: &W3, e: usize) -> Vec<(u16, usize, usize)> { let (a, n) = (w.fs.s()[e] as usize, w.fnn.s()[e] as usize); let p = w.fp.s(); let mut o: Vec<(u16, usize, usize)> = vec![]; (a..a + n).for_each(|i| if o.last().is_some_and(|g| g.0 == p[i]) { o.last_mut().unwrap().2 += 1 } else { o.push((p[i], i, 1)) }); o }
fn qc(w: &W3, i: usize) -> u64 { (w.fqo.s()[i + 1] - w.fqo.s()[i]) as u64 }
fn derive(w: &W3, ne: usize, on: bool) -> HashSet<(u32, u16)> {
    if !on { return HashSet::new() }
    let gv = |e: usize, p: u16| groups(w, e).into_iter().find(|g| g.0 == p).map(|g| (g.1..g.1 + g.2).collect::<Vec<usize>>());
    let mut d: HashSet<(u32, u16)> = RULES.par_iter().flat_map(|&(tp, via, src)| (0..ne).into_par_iter().filter_map(move |e| { let g = gv(e, tp)?; let v = gv(e, via)?; (w.fk.s()[v[0]] == 1 && g.iter().all(|&i| w.fk.s()[i] == 1 && w.fr.s()[i] == 0 && qc(w, i) == 0)).then_some(())?; let t = w.fv.s()[v[0]] as usize; let s = gv(t, src)?; (s.iter().map(|&i| w.fv.s()[i]).collect::<Vec<_>>() == g.iter().map(|&i| w.fv.s()[i]).collect::<Vec<_>>() && s.iter().all(|&i| w.fk.s()[i] == 1)).then_some((e as u32, tp)) }).collect::<Vec<_>>()).collect();
    let cyc: Vec<(u32, u16)> = d.par_iter().filter(|&&(e, p)| { let (mut x, mut q, mut n) = (e as usize, p, 0); while d.contains(&(x as u32, q)) { let r = RULES.iter().find(|r| r.0 == q).unwrap(); let Some(v) = gv(x, r.1) else { return true }; x = w.fv.s()[v[0]] as usize; q = r.2; n += 1; if n > 30 { return true } } false }).copied().collect();
    cyc.iter().for_each(|k| { d.remove(k); });
    d
}
fn build(a: &[String]) -> io::Result<()> {
    let (src, out, fds, tk, dv) = (arg(a, "--v3", "data/wikifact/v3/store"), arg(a, "--out", "data/store1/drive"), arg(a, "--fd", "0,1,2,3"), arg(a, "--tk", "3").parse::<usize>().unwrap(), arg(a, "--derive", "1") == "1");
    fs::create_dir_all(&out)?;
    let t0 = Instant::now();
    let w = w3(&src)?;
    let ne = w.qid.n;
    let mut pn: Vec<u16> = w.fp.s().par_iter().copied().chain(w.qp.s().par_iter().copied()).collect::<HashSet<u16>>().into_iter().collect();
    pn.sort();
    let pidx: HashMap<u16, usize> = pn.iter().enumerate().map(|(i, &p)| (p, i)).collect();
    let mut pkind = vec![0u8; pn.len()];
    w.fp.s().iter().zip(w.fk.s()).for_each(|(&p, &k)| pkind[pidx[&p]] = k);
    let gcount: Vec<u64> = (0..ne).into_par_iter().fold(|| vec![0u64; pn.len()], |mut c, e| { groups(&w, e).iter().for_each(|g| c[pidx[&g.0]] += 1); c }).reduce(|| vec![0u64; pn.len()], |a, b| a.iter().zip(&b).map(|(x, y)| x + y).collect());
    let mut ord: Vec<usize> = (0..pn.len()).collect();
    ord.sort_by_key(|&i| (std::cmp::Reverse(gcount[i]), pn[i]));
    let mut prank = vec![0u16; pn.len()];
    ord.iter().enumerate().for_each(|(r, &i)| prank[i] = r as u16);
    let mut strs: Vec<&[u8]> = vec![];
    let mut sid: HashMap<&[u8], u64> = HashMap::new();
    let mut put = |b: &'static [u8]| -> u64 { *sid.entry(b).or_insert_with(|| { strs.push(b); strs.len() as u64 - 1 }) };
    let wst: &'static W3 = Box::leak(Box::new(w));
    let w = wst;
    let fsid: Vec<u64> = (0..w.fv.n).map(|i| if w.fk.s()[i] >= 4 { put(sbytes(w, w.fv.s()[i])) } else { 0 }).collect();
    let qsid: Vec<u64> = (0..w.qv.n).map(|i| if w.qk.s()[i] >= 4 { put(sbytes(w, w.qv.s()[i])) } else { 0 }).collect();
    eprintln!("load+strings {:.1}s strings {}", t0.elapsed().as_secs_f64(), strs.len());
    if arg(a, "--only", "") == "text" {
        let nme = nmenc(&format!("{}/nmenc.bin", arg(a, "--build", "data/store1/build")), ne)?;
        let ra: Vec<Vec<(u8, Vec<u16>)>> = (0..ne).into_par_iter().map(|e| { let (k, x) = tkind(lab(w, e), tit(w, e)); rec_a(k, lab(w, e), &x) }).collect();
        let rb: Vec<Vec<(u8, Vec<u16>)>> = nme.par_iter().map(|n| rec_b(n)).collect();
        let rc: Vec<Vec<(u8, Vec<u16>)>> = strs.par_iter().map(|s| rec_s(s)).collect();
        for k in arg(a, "--tks", "2,3,4").split(',').map(|x| x.parse::<usize>().unwrap()) {
            let t = Instant::now();
            let (tm, tmd) = Tm::build(&[&ra, &rb, &rc], k);
            let z: Vec<(usize, usize)> = [(&ra, TB), (&rb, NB), (&rc, TB)].iter().map(|(r, bs)| { let (s, o) = blocks(&tm, r, *bs); (s.len(), o.len()) }).collect();
            let tot = (tmd.len() as f64 * 17.0 / 32.0) as usize + z.iter().map(|x| ((x.0 + x.1) as f64 * 17.0 / 16.0) as usize).sum::<usize>();
            println!("{{\"tk\":{k},\"contexts\":{},\"model_bytes\":{},\"streams\":{:?},\"total_bytes\":{tot},\"s\":{:.1}}}", tm.ix.len(), (tmd.len() as f64 * 17.0 / 32.0) as usize, z, t.elapsed().as_secs_f64());
        }
        return Ok(());
    }
    let der = derive(w, ne, dv);
    let der_facts: u64 = der.par_iter().map(|&(e, p)| groups(w, e as usize).iter().find(|g| g.0 == p).map_or(0, |g| g.2 as u64)).sum();
    eprintln!("derive {:.1}s groups {} facts {}", t0.elapsed().as_secs_f64(), der.len(), der_facts);
    let sigs: Vec<Vec<u64>> = (0..ne).into_par_iter().map(|e| groups(w, e).iter().flat_map(|g| [prank[pidx[&g.0]] as u64, if der.contains(&(e as u32, g.0)) { 2 } else if g.2 > 1 { 1 } else { 0 }]).collect()).collect();
    let fsym = |i: usize| -> Vec<u64> { let a = w.fr.s()[i] as u64 | qc(w, i) << 1; match w.fk.s()[i] { 1 | 2 => vec![w.fv.s()[i], a], 3 => vec![w.fv.s()[i], if w.fx.s()[i] == u32::MAX { 0 } else { w.fx.s()[i] as u64 + 1 }, a], _ => vec![a] } };
    let qsym = |j: usize| -> Vec<u64> { vec![pidx[&w.qp.s()[j]] as u64, w.qk.s()[j] as u64, if w.qk.s()[j] >= 4 { qsid[j] } else { w.qv.s()[j] }] };
    let live = |e: usize, g: &(u16, usize, usize)| !der.contains(&(e as u32, g.0));
    let cnt_maps = |np: usize, f: &(dyn Fn(usize, &mut Vec<HashMap<Vec<u64>, u64>>) + Sync)| -> Vec<HashMap<Vec<u64>, u64>> { (0..ne).into_par_iter().fold(|| vec![HashMap::new(); np], |mut m, e| { f(e, &mut m); m }).reduce(|| vec![HashMap::new(); np], |mut a, b| { a.iter_mut().zip(b).for_each(|(x, y)| y.into_iter().for_each(|(k, v)| *x.entry(k).or_default() += v)); a }) };
    let mut sizes: Vec<(String, u64)> = vec![];
    let mut pg = PgW::new();
    let cas = |name: &str, np: usize, cf: &(dyn Fn(usize, &mut Vec<HashMap<Vec<u64>, u64>>) + Sync), kf: &(dyn Fn(usize, &mut Vec<(u64, usize, Vec<u64>)>) + Sync), pg: &mut PgW, sizes: &mut Vec<(String, u64)>| -> io::Result<()> {
        let t = Instant::now();
        let mut b = CasB::new(name, cnt_maps(np, cf));
        let mut n = 0usize;
        for c0 in (0..ne).step_by(1 << 20) { let rows: Vec<(u64, usize, Vec<u64>)> = (c0..(c0 + (1 << 20)).min(ne)).into_par_iter().fold(Vec::new, |mut v, e| { kf(e, &mut v); v }).reduce(Vec::new, |mut a, mut b| { a.append(&mut b); a }); n += rows.len(); rows.iter().for_each(|(k, c, t)| b.push(*k, *c, t)); }
        let (cb, cbx, hc) = (b.cb.len(), b.cbx.len(), b.hcdig().len());
        let (by, keys) = b.save(&out, pg)?;
        eprintln!("{name}: keys {n} level-keys {keys} levels-bytes {by} cb-digits {cb} cbx {cbx} hc {hc} entries {} {:.1}s", b.syms.iter().map(|m| m.len()).sum::<usize>(), t.elapsed().as_secs_f64());
        sizes.push((format!("{name}_levels"), by));
        sizes.push((format!("{name}_codebook"), ((cb + cbx + hc) as f64 * 17.0 / 32.0) as u64));
        Ok(())
    };
    cas("sig", 1, &|e, m| { *m[0].entry(sigs[e].clone()).or_default() += 1 }, &|e, v| v.push((e as u64, 0, sigs[e].clone())), &mut pg, &mut sizes)?;
    cas("sl", 1, &|e, m| { *m[0].entry(vec![w.sl.s()[e] as u64]).or_default() += 1 }, &|e, v| v.push((e as u64, 0, vec![w.sl.s()[e] as u64])), &mut pg, &mut sizes)?;
    cas("cnt", 1, &|e, m| groups(w, e).iter().filter(|g| g.2 > 1 && live(e, g)).for_each(|g| *m[0].entry(vec![g.2 as u64]).or_default() += 1), &|e, v| groups(w, e).iter().filter(|g| g.2 > 1 && live(e, g)).for_each(|g| v.push((keyof(e as u32, g.0, 0), 0, vec![g.2 as u64]))), &mut pg, &mut sizes)?;
    cas("fact", pn.len(), &|e, m| groups(w, e).iter().filter(|g| live(e, g)).for_each(|g| (g.1..g.1 + g.2).for_each(|i| *m[pidx[&g.0]].entry(fsym(i)).or_default() += 1)), &|e, v| groups(w, e).iter().filter(|g| live(e, g)).for_each(|g| (g.1..g.1 + g.2).for_each(|i| v.push((keyof(e as u32, g.0, (i - g.1) as u16), pidx[&g.0], fsym(i))))), &mut pg, &mut sizes)?;
    cas("qual", pn.len(), &|e, m| groups(w, e).iter().filter(|g| live(e, g)).for_each(|g| (g.1..g.1 + g.2).for_each(|i| (w.fqo.s()[i] as usize..w.fqo.s()[i + 1] as usize).for_each(|j| *m[pidx[&g.0]].entry(qsym(j)).or_default() += 1))), &|e, v| groups(w, e).iter().filter(|g| live(e, g)).for_each(|g| (g.1..g.1 + g.2).for_each(|i| (w.fqo.s()[i] as usize..w.fqo.s()[i + 1] as usize).for_each(|j| v.push((keyof(e as u32, g.0, (i - g.1) as u16) | ((j - w.fqo.s()[i] as usize + 1) as u64) << 56, pidx[&g.0], qsym(j)))))), &mut pg, &mut sizes)?;
    let fsr = &fsid;
    let sk: Vec<(u64, [u8; 5])> = (0..ne).into_par_iter().flat_map_iter(|e| groups(w, e).into_iter().filter(|g| w.fk.s()[g.1] >= 4).flat_map(move |g| (g.1..g.1 + g.2).map(move |i| (keyof(e as u32, g.0, (i - g.1) as u16), fsr[i])))).map(|(k, s)| { let mut d = vec![]; fx(&mut d, s, 5); (k, d.try_into().unwrap()) }).collect();
    let sp = format!("{out}/sid.gf17rib");
    let by = layered_by::<5, [u8; 5]>(&sp, &sk, drow::<5>, 64, 0, 65536, -0.03, seed(&sp), Some((32, [0, 8, 16, 32])), 0.08, None)?.2;
    sizes.push(("sid".into(), by));
    eprintln!("sid keys {} bytes {by} {:.1}s", sk.len(), t0.elapsed().as_secs_f64());
    let mut meta = vec![];
    ev(&mut meta, ne as u64);
    ev(&mut meta, pn.len() as u64);
    (0..pn.len()).for_each(|i| { ev(&mut meta, pn[i] as u64); ev(&mut meta, pkind[i] as u64); ev(&mut meta, prank[i] as u64) });
    pg.add("meta", &meta);
    let (mut qd, mut qx) = (vec![], vec![]);
    (0..ne).for_each(|e| { if e % 64 == 0 { fx(&mut qx, qd.len() as u64, 8); fx(&mut qd, w.qid.s()[e] as u64, 7) } else { ev(&mut qd, (w.qid.s()[e] - w.qid.s()[e - 1] - 1) as u64) } });
    sizes.push(("qid".into(), ((qd.len() + qx.len()) as f64 * 17.0 / 32.0) as u64));
    pg.add("qid", &qd);
    pg.add("qidx", &qx);
    let tt = Instant::now();
    let nme = nmenc(&format!("{}/nmenc.bin", arg(a, "--build", "data/store1/build")), ne)?;
    let ra: Vec<Vec<(u8, Vec<u16>)>> = (0..ne).into_par_iter().map(|e| { let (k, x) = tkind(lab(w, e), tit(w, e)); rec_a(k, lab(w, e), &x) }).collect();
    let rb: Vec<Vec<(u8, Vec<u16>)>> = nme.par_iter().map(|n| rec_b(n)).collect();
    let rc: Vec<Vec<(u8, Vec<u16>)>> = strs.par_iter().map(|s| rec_s(s)).collect();
    drop(nme);
    let (tm, tmd) = Tm::build(&[&ra, &rb, &rc], tk);
    let syms: usize = ra.iter().chain(&rb).chain(&rc).map(|r| r.iter().map(|f| f.1.len()).sum::<usize>()).sum();
    pg.add("tm", &tmd);
    sizes.push(("text_model".into(), (tmd.len() as f64 * 17.0 / 32.0) as u64));
    for (n, r, bs) in [("ta", &ra, TB), ("tb", &rb, NB), ("tc", &rc, TB)] { let (s, o) = blocks(&tm, r, bs); sizes.push((format!("text_{n}"), (s.len() as f64 * 17.0 / 16.0) as u64)); sizes.push((format!("text_{n}_x"), (o.len() as f64 * 17.0 / 16.0) as u64)); pg.bytes(n, &s); pg.bytes(&format!("{n}_x"), &o) }
    eprintln!("text {:.1}s contexts {} symbols {syms} model digits {}", tt.elapsed().as_secs_f64(), tm.ix.len(), tmd.len());
    let (pb, ok, pages) = pg.save(&format!("{out}/store1.gf17pg"))?;
    eprintln!("container {pb} bytes, roundtrip {ok}, pages {pages}");
    let (ah, ae, ak) = (w.ah.s(), w.ae.s(), w.ak.s());
    let mut st: Vec<usize> = (0..ah.len()).filter(|&i| i == 0 || ah[i] != ah[i - 1]).collect();
    st.push(ah.len());
    let nk: Vec<(u64, [u8; 7])> = st.par_windows(2).flat_map_iter(|x| (x[0]..x[1]).map(move |i| { let mut d = vec![]; fx(&mut d, ae[i] as u64, 6); d.push(ak[i]); (mixk(ah[i], (i - x[0]) as u64), d.try_into().unwrap()) })).collect();
    let mut kk: Vec<u64> = nk.par_iter().map(|x| x.0).collect();
    kk.par_sort_unstable();
    let dup = kk.par_windows(2).filter(|x| x[0] == x[1]).count();
    eprintln!("name keys {} dup {dup}", nk.len());
    let mut nsz = vec![];
    for f in fds.split(',').map(|x| x.parse::<usize>().unwrap()) {
        let p = format!("{out}/name_f{f}.gf17rib");
        let by = match f { 0 => nb::<7>(&p, &nk, 0)?, 1 => nb::<8>(&p, &nk, 1)?, 2 => nb::<9>(&p, &nk, 2)?, _ => nb::<10>(&p, &nk, 3)? };
        let (mut eb, mut ne_) = (0u64, 0usize);
        if f > 0 {
            let t = Nm::open(&p, f)?;
            let add: Vec<u64> = st.par_windows(2).filter_map(|x| { let (h, n, last) = (ah[x[0]], (x[1] - x[0]) as u64, ae[x[1] - 1]); let k = mixk(h, n); t.row(k).filter(|d| { let e = d[..6].iter().rev().fold(0u32, |a, &v| a * 17 + v as u32); e != END && (e as usize) < ne && e > last && d[6] <= 3 }).map(|_| k) }).collect();
            let ep = format!("{out}/name_f{f}_end.gf17rib");
            let rows: Vec<(u64, [u8; 16])> = add.iter().map(|&k| (k, [16u8; 16])).collect();
            eb = layered_by::<16, [u8; 16]>(&ep, &rows, drow::<16>, 64, 9, 65536, 0.08, seed(&ep), None, 0.0, None)?.2;
            ne_ = add.len();
        }
        eprintln!("name f{f} bytes {by} end markers {ne_} end bytes {eb}");
        nsz.push((f, by + eb, ne_));
        sizes.push((format!("name_f{f}"), by + eb));
    }
    let o = js(&[("entities", ne.to_string()), ("facts", w.fv.n.to_string()), ("quals", w.qv.n.to_string()), ("strings", strs.len().to_string()), ("props", pn.len().to_string()), ("derived_groups", der.len().to_string()), ("derived_facts", der_facts.to_string()), ("text_order", tk.to_string()), ("text_contexts", tm.ix.len().to_string()), ("text_symbols", syms.to_string()), ("container_bytes", pb.to_string()), ("roundtrip_bitexact", ok.to_string()), ("container_pages", pages.to_string()), ("name_keys", nk.len().to_string()), ("name_key_dups", dup.to_string()), ("name_tables", format!("[{}]", nsz.iter().map(|(f, b, e)| format!("{{\"fd\":{f},\"bytes\":{b},\"end_markers\":{e}}}")).collect::<Vec<_>>().join(","))), ("components", format!("{{{}}}", sizes.iter().map(|(n, b)| format!("\"{n}\":{b}")).collect::<Vec<_>>().join(","))), ("build_s", format!("{:.1}", t0.elapsed().as_secs_f64())), ("peak_rss", rss().1.to_string())]);
    println!("{o}");
    fs::write(format!("{out}/build.json"), o)
}
fn nb<const R: usize>(p: &str, rows: &[(u64, [u8; 7])], fd: usize) -> io::Result<u64> { let r: Vec<(u64, [u8; R])> = rows.iter().map(|(k, d)| { let mut x = [0u8; R]; x[..7].copy_from_slice(d); (*k, x) }).collect(); Ok(layered_by::<R, [u8; R]>(p, &r, drow::<R>, 64, fd, 65536, -0.03, seed(p), Some((32, [0, 8, 16, 32])), 0.08, None)?.2) }
fn parity(a: &[String]) -> io::Result<()> {
    let (src, dir, fd) = (arg(a, "--v3", "data/wikifact/v3/store"), arg(a, "--dir", "data/store1/drive"), arg(a, "--fd", "3").parse::<usize>().unwrap());
    let t0 = Instant::now();
    let w = w3(&src)?;
    let s = Store::open(&dir, fd)?;
    let ne = w.qid.n;
    let nme = nmenc(&format!("{}/nmenc.bin", arg(a, "--build", "data/store1/build")), ne)?;
    let txt = |k: u8, v: u64| if k >= 4 { sbytes(&w, v).to_vec() } else { vec![] };
    let r: Vec<[u64; 9]> = (0..ne).into_par_iter().map(|e| {
        let mut c = [0u64; 9];
        let g = groups(&w, e);
        let sg = s.groups(e as u32);
        c[0] += (g.len() != sg.len() || g.iter().zip(&sg).any(|(a, b)| a.0 != s.pnum[b.0])) as u64;
        for x in &g {
            let v = s.values(e as u32, x.0);
            c[8] += x.2 as u64;
            for i in 0..x.2 {
                let f = x.1 + i;
                let ok = v.get(i).is_some_and(|t| t.kind == w.fk.s()[f] && t.pref == w.fr.s()[f] && t.nq as u64 == qc(&w, f) && t.x == if w.fk.s()[f] == 3 { w.fx.s()[f] } else { u32::MAX } && if t.kind >= 4 { s.string(t.v) == txt(t.kind, w.fv.s()[f]) } else { t.v == w.fv.s()[f] }) && v.len() == x.2;
                let qok = ok && { let q = s.quals(e as u32, x.0, i as u64, v[i].nq); let (a, b) = (w.fqo.s()[f] as usize, w.fqo.s()[f + 1] as usize); q.len() == b - a && q.iter().zip(a..b).all(|(t, j)| t.0 == w.qp.s()[j] && t.1 == w.qk.s()[j] && if t.1 >= 4 { s.string(t.2) == txt(t.1, w.qv.s()[j]) } else { t.2 == w.qv.s()[j] }) };
                c[1] += !ok as u64;
                c[2] += !qok as u64;
            }
        }
        c[3] += (s.label(e as u32) != lab(&w, e) || s.title(e as u32) != tit(&w, e)) as u64;
        c[4] += (s.nmenc(e as u32) != nme[e]) as u64;
        c[5] += (s.sitelinks(e as u32) != w.sl.s()[e] as u64) as u64;
        c[6] += (s.qid(e as u32) != w.qid.s()[e]) as u64;
        c[7] += (s.ix(w.qid.s()[e]) != e as i64) as u64;
        c
    }).collect();
    let t = r.iter().fold([0u64; 9], |a, b| std::array::from_fn(|i| a[i] + b[i]));
    let (ah, ae, ak) = (w.ah.s(), w.ae.s(), w.ak.s());
    let mut st: Vec<usize> = (0..ah.len()).filter(|&i| i == 0 || ah[i] != ah[i - 1]).collect();
    st.push(ah.len());
    let em = st.par_windows(2).filter(|x| s.ents(ah[x[0]]) != (x[0]..x[1]).map(|i| (ae[i], ak[i])).collect::<Vec<_>>()).count();
    let hs: HashSet<u64> = ah.iter().copied().collect();
    let mut rr = 99u64;
    let probes: Vec<u64> = (0..2_000_000).map(|_| sm(&mut rr)).filter(|h| !hs.contains(h)).collect();
    let fp = probes.par_iter().filter(|&&h| !s.ents(h).is_empty()).count();
    let o = js(&[("fd", fd.to_string()), ("entities", ne.to_string()), ("facts_checked", t[8].to_string()), ("parity_groups_mismatch", t[0].to_string()), ("parity_facts_mismatch", t[1].to_string()), ("parity_quals_mismatch", t[2].to_string()), ("parity_labels_mismatch", t[3].to_string()), ("parity_names_mismatch", t[4].to_string()), ("parity_sitelinks_mismatch", t[5].to_string()), ("parity_qid_mismatch", t[6].to_string()), ("parity_ix_mismatch", t[7].to_string()), ("name_groups", (st.len() - 1).to_string()), ("parity_ents_mismatch", em.to_string()), ("absent_name_probes", probes.len().to_string()), ("absent_name_fp", fp.to_string()), ("absent_name_fpr", format!("{:.3e}", fp as f64 / probes.len() as f64)), ("sec", format!("{:.1}", t0.elapsed().as_secs_f64()))]);
    println!("{o}");
    fs::write(format!("{dir}/parity_f{fd}.json"), o)
}
fn bench(a: &[String]) -> io::Result<()> {
    let (src, dir, fd, n) = (arg(a, "--v3", "data/wikifact/v3/store"), arg(a, "--dir", "data/store1/drive"), arg(a, "--fd", "3").parse::<usize>().unwrap(), arg(a, "--n", "1000000").parse::<usize>().unwrap());
    let w = w3(&src)?;
    let t0 = Instant::now();
    let s = Store::open(&dir, fd)?;
    let open_ms = t0.elapsed().as_secs_f64() * 1e3;
    let ne = w.qid.n;
    let mut r = 7u64;
    let fk: Vec<(u32, u16)> = (0..n).map(|_| { let f = sm(&mut r) as usize % w.fv.n; (w.fsub.s()[f], w.fp.s()[f]) }).filter(|&(e, p)| groups(&w, e as usize).iter().any(|g| g.0 == p)).collect();
    let ents: Vec<u32> = (0..n).map(|_| (sm(&mut r) % ne as u64) as u32).collect();
    let hs: Vec<u64> = (0..n).map(|_| w.ah.s()[sm(&mut r) as usize % w.ah.n]).collect();
    let tm = |f: &dyn Fn() -> u64, k: usize| { let t = Instant::now(); let x = f(); (t.elapsed().as_secs_f64() * 1e9 / k as f64, x) };
    let one = |k: usize| std::thread::scope(|sc| sc.spawn(|| (tm(&|| fk[..k].iter().map(|&(e, p)| s.values(e, p).len() as u64).sum(), k), tm(&|| hs[..k].iter().map(|&h| s.ents(h).len() as u64).sum(), k), tm(&|| ents[..k / 10].iter().map(|&e| s.label(e).len() as u64).sum(), k / 10), tm(&|| ents[..k / 10].iter().map(|&e| s.nmenc(e).len() as u64).sum(), k / 10), tm(&|| ents[..k].iter().map(|&e| s.groups(e).len() as u64).sum(), k))).join().unwrap());
    one(fk.len().min(100_000));
    let wm = one(fk.len());
    s.cold();
    let cd = one(20_000);
    let o = js(&[("fd", fd.to_string()), ("open_ms", format!("{open_ms:.1}")), ("values_ns_warm", format!("{:.0}", wm.0 .0)), ("ents_ns_warm", format!("{:.0}", wm.1 .0)), ("label_ns_warm", format!("{:.0}", wm.2 .0)), ("nmenc_ns_warm", format!("{:.0}", wm.3 .0)), ("groups_ns_warm", format!("{:.0}", wm.4 .0)), ("values_per_s_core_warm", format!("{:.0}", 1e9 / wm.0 .0)), ("values_ns_cold", format!("{:.0}", cd.0 .0)), ("ents_ns_cold", format!("{:.0}", cd.1 .0)), ("label_ns_cold", format!("{:.0}", cd.2 .0)), ("rss", format!("{:?}", rss())), ("n_values", fk.len().to_string())]);
    println!("{o}");
    fs::write(format!("{dir}/bench_f{fd}.json"), o)
}
fn main() -> io::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) { Some("build") => build(&a), Some("parity") => parity(&a), Some("bench") => bench(&a), _ => { eprintln!("usage: store1 build|parity|bench ..."); Ok(()) } }
}
