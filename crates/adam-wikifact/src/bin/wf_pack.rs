use adam_gf17::{Store, Table};
use adam_wikifact::{tpack, Rd, Val};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::time::Instant;
struct Npy { w: BufWriter<File>, n: u64, want: u64, path: String }
impl Npy {
    fn new(dir: &str, name: &str, descr: &str, n: u64) -> io::Result<Npy> {
        let path = format!("{dir}/{name}.npy");
        let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({n},), }}");
        while (h.len() + 11) % 64 != 0 { h.push(' ') }
        h.push('\n');
        let mut w = BufWriter::with_capacity(4 << 20, File::create(&path)?);
        w.write_all(b"\x93NUMPY\x01\x00")?;
        w.write_all(&(h.len() as u16).to_le_bytes())?;
        w.write_all(h.as_bytes())?;
        Ok(Npy { w, n: 0, want: n, path })
    }
    fn put(&mut self, b: &[u8]) { self.w.write_all(b).unwrap(); self.n += 1 }
    fn end(mut self) -> io::Result<String> { self.w.flush()?; if self.n == self.want { Ok(self.path) } else { Err(io::Error::other(format!("count mismatch {} {} {}", self.path, self.n, self.want))) } }
}
struct Rec<'a> { q: u32, _sl: u16, t: &'a str, _lb: &'a str, _al: Vec<&'a str>, fs: Vec<(u16, u8, Val, Vec<(u16, Val)>)> }
fn rec<'a>(r: &mut Rd<'a>) -> Rec<'a> {
    let (q, sl, t, lb) = (r.u32(), r.u16(), r.s(), r.s());
    let na = r.u8();
    let al = (0..na).map(|_| r.s()).collect();
    let nf = r.u32();
    let fs = (0..nf).map(|_| { let (p, rk, v) = (r.u16(), r.u8(), r.v()); let nq = r.u8(); (p, rk, v, (0..nq).map(|_| (r.u16(), r.v())).collect()) }).collect();
    Rec { q, _sl: sl, t, _lb: lb, _al: al, fs }
}
fn map(p: &str) -> io::Result<memmap2::Mmap> { unsafe { memmap2::Mmap::map(&File::open(p)?) } }
fn main() -> io::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = |k: &str, d: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned().unwrap_or(d.into());
    let (bd, out, tile) = (g("--build", "."), g("--out", "."), g("--tile", "128").parse::<usize>().unwrap());
    std::fs::create_dir_all(&out)?;
    let t0 = Instant::now();
    let (sb, lbm) = (map(&format!("{bd}/sub.bin"))?, map(&format!("{bd}/lab.bin"))?);
    let mut refd = vec![0u64; 1 << 22];
    let set = |q: u32, v: &mut Vec<u64>| { let i = q as usize >> 6; if i >= v.len() { v.resize(i + 1, 0) } v[i] |= 1 << (q & 63) };
    let (mut subs, mut nf, mut nq, mut sbytes) = (vec![], 0u64, 0u64, 0u64);
    let mut r = Rd { b: &sb, i: 0 };
    while !r.done() {
        let o = r.i;
        let x = rec(&mut r);
        subs.push((x.q, o as u64));
        set(x.q, &mut refd);
        nf += x.fs.len() as u64;
        for (_, _, v, qs) in &x.fs { for v in std::iter::once(v).chain(qs.iter().map(|z| &z.1)) { match v { Val::I(q) => set(*q, &mut refd), Val::Q(_, u) if *u > 0 => set(*u, &mut refd), Val::S(s) | Val::M(s) => sbytes += s.len() as u64, _ => {} } } nq += qs.len() as u64 }
    }
    subs.sort_unstable();
    subs.dedup_by_key(|x| x.0);
    eprintln!("pass A {:.0}s subjects {} facts {} quals {} strbytes {}", t0.elapsed().as_secs_f64(), subs.len(), nf, nq, sbytes);
    let has = |q: u32| refd.get(q as usize >> 6).is_some_and(|w| w >> (q & 63) & 1 == 1);
    let mut labs: Vec<(u32, u64)> = vec![];
    let mut r = Rd { b: &lbm, i: 0 };
    while !r.done() { let o = r.i; let q = r.u32(); r.s(); let n = r.u8(); (0..n).for_each(|_| { r.s(); }); if has(q) { labs.push((q, o as u64)) } }
    labs.sort_unstable();
    labs.dedup_by_key(|x| x.0);
    let mut ents: Vec<u32> = refd.iter().enumerate().flat_map(|(i, &w)| (0..64).filter(move |b| w >> b & 1 == 1).map(move |b| (i * 64 + b) as u32)).collect();
    ents.sort_unstable();
    let ne = ents.len() as u64;
    let ix = |q: u32| ents.binary_search(&q).map(|i| i as u32).unwrap_or(u32::MAX);
    eprintln!("pass B {:.0}s entities {} labelled {}", t0.elapsed().as_secs_f64(), ne, labs.len());
    let mut fsn = vec![(0u32, 0u32); ne as usize];
    let (mut fs_, mut fp, mut fr, mut fk, mut fv, mut fx, mut fqo) = (Npy::new(&out, "f_s", "<u4", nf)?, Npy::new(&out, "f_p", "<u2", nf)?, Npy::new(&out, "f_r", "|u1", nf)?, Npy::new(&out, "f_k", "|u1", nf)?, Npy::new(&out, "f_v", "<u8", nf)?, Npy::new(&out, "f_x", "<u4", nf)?, Npy::new(&out, "f_qo", "<u4", nf + 1)?);
    let (mut qp, mut qk, mut qv) = (Npy::new(&out, "q_p", "<u2", nq)?, Npy::new(&out, "q_k", "|u1", nq)?, Npy::new(&out, "q_v", "<u8", nq)?);
    let mut sh = BufWriter::with_capacity(4 << 20, File::create(format!("{out}/str.bin"))?);
    let mut so = 0u64;
    let mut enc = |v: &Val, sh: &mut BufWriter<File>| -> (u8, u64, u32) { match v { Val::I(q) => (1, ix(*q) as u64, u32::MAX), Val::T(y, m, d, p, j) => (2, tpack(*y, *m, *d, *p, *j), u32::MAX), Val::Q(x, u) => (3, x.to_bits(), if *u > 0 { ix(*u) } else { u32::MAX }), Val::S(s) | Val::M(s) => { sh.write_all(s.as_bytes()).unwrap(); let o = so << 16 | s.len() as u64; so += s.len() as u64; (if matches!(v, Val::S(_)) { 4 } else { 5 }, o, u32::MAX) } } };
    let (mut fi, mut qi) = (0u32, 0u32);
    let mut names = BufWriter::with_capacity(4 << 20, File::create(format!("{out}/names.tsv"))?);
    let mut tit: Vec<(u32, String)> = vec![];
    let clean = |s: &str| s.replace(['\t', '\n', '\r'], " ");
    for &(q, o) in &subs {
        let x = rec(&mut Rd { b: &sb, i: o as usize });
        let e = ix(q);
        fsn[e as usize] = (fi, x.fs.len() as u32);
        tit.push((e, x.t.to_string()));
        for (p, rk, v, qs) in &x.fs {
            let (k, vv, xx) = enc(v, &mut sh);
            fs_.put(&e.to_le_bytes()); fp.put(&p.to_le_bytes()); fr.put(&[*rk]); fk.put(&[k]); fv.put(&vv.to_le_bytes()); fx.put(&xx.to_le_bytes()); fqo.put(&qi.to_le_bytes());
            for (pp, qq) in qs { let (k, vv, _) = enc(qq, &mut sh); qp.put(&pp.to_le_bytes()); qk.put(&[k]); qv.put(&vv.to_le_bytes()); qi += 1 }
            fi += 1;
        }
    }
    fqo.put(&qi.to_le_bytes());
    sh.flush()?;
    let mut paths = [fs_.end()?, fp.end()?, fr.end()?, fk.end()?, fv.end()?, fx.end()?, fqo.end()?, qp.end()?, qk.end()?, qv.end()?].to_vec();
    tit.sort_unstable();
    let sl: std::collections::HashMap<u32, u16> = subs.iter().map(|&(q, o)| (q, Rd { b: &sb, i: o as usize + 4 }.u16())).collect();
    let (mut eq, mut es, mut efs, mut efn, mut elo, mut eto) = (Npy::new(&out, "e_qid", "<u4", ne)?, Npy::new(&out, "e_sl", "<u2", ne)?, Npy::new(&out, "e_fs", "<u4", ne)?, Npy::new(&out, "e_fn", "<u4", ne)?, Npy::new(&out, "e_lo", "<u8", ne + 1)?, Npy::new(&out, "e_to", "<u8", ne + 1)?);
    let (mut lh, mut th) = (BufWriter::new(File::create(format!("{out}/e_lab.bin"))?), BufWriter::new(File::create(format!("{out}/e_tit.bin"))?));
    let (mut lo, mut to, mut ti, mut li) = (0u64, 0u64, 0usize, 0usize);
    for (i, &q) in ents.iter().enumerate() {
        eq.put(&q.to_le_bytes()); es.put(&sl.get(&q).copied().unwrap_or(0).to_le_bytes()); efs.put(&fsn[i].0.to_le_bytes()); efn.put(&fsn[i].1.to_le_bytes()); elo.put(&lo.to_le_bytes()); eto.put(&to.to_le_bytes());
        while li < labs.len() && labs[li].0 < q { li += 1 }
        let mut l = "";
        if li < labs.len() && labs[li].0 == q { let mut r = Rd { b: &lbm, i: labs[li].1 as usize + 4 }; l = r.s(); lh.write_all(l.as_bytes())?; lo += l.len() as u64; (!l.is_empty()).then(|| writeln!(names, "{i}\t0\t{}", clean(l)).unwrap()); let n = r.u8(); (0..n).for_each(|_| { let a = r.s(); writeln!(names, "{i}\t1\t{}", clean(a)).unwrap() }) }
        while ti < tit.len() && (tit[ti].0 as usize) < i { ti += 1 }
        if ti < tit.len() && tit[ti].0 as usize == i { let t = &tit[ti].1; th.write_all(t.as_bytes())?; to += t.len() as u64; (t != l).then(|| writeln!(names, "{i}\t2\t{}", clean(t)).unwrap()); }
    }
    elo.put(&lo.to_le_bytes());
    eto.put(&to.to_le_bytes());
    lh.flush()?;
    th.flush()?;
    names.flush()?;
    paths.extend([eq.end()?, es.end()?, efs.end()?, efn.end()?, elo.end()?, eto.end()?]);
    eprintln!("pass C {:.0}s facts {} quals {} strheap {} labels {} titles {}", t0.elapsed().as_secs_f64(), fi, qi, so, lo, to);
    let ms: Vec<(String, memmap2::Mmap)> = paths.iter().map(|p| Ok((p.rsplit('/').next().unwrap().trim_end_matches(".npy").to_string(), map(p)?))).collect::<io::Result<_>>()?;
    let body = |m: &memmap2::Mmap| { let hl = u16::from_le_bytes([m[8], m[9]]) as usize; 10 + hl };
    let items: Vec<(Table, &[u8])> = ms.iter().filter(|(n, _)| !n.starts_with("e_lo") && !n.starts_with("e_to")).map(|(n, m)| { let b = &m[body(m)..]; let dt = if b.len() % 4 == 0 { 4 } else if b.len() % 2 == 0 { 2 } else { 1 }; (Table { name: n.clone(), dt, ord: false, exact: true, shape: vec![b.len() / dt] }, b) }).collect();
    let gp = format!("{out}/facts.gf17");
    adam_gf17::write(&gp, &items, tile)?;
    let st = Store::open(&gp, 64 << 20)?;
    let ok = items.iter().all(|(t, b)| st.table(&t.name).map(|i| st.unpack(i, 4).map(|(v, _)| v == *b).unwrap_or(false)).unwrap_or(false));
    let npyb: u64 = items.iter().map(|x| x.1.len() as u64).sum();
    let js = format!("{{\"subjects\":{},\"entities\":{},\"labelled\":{},\"facts\":{},\"quals\":{},\"str_bytes\":{},\"gf17\":\"facts.gf17\",\"gf17_bytes\":{},\"coord_bytes\":{},\"gf17_exact\":{},\"tile\":{},\"sec\":{:.0}}}", subs.len(), ne, labs.len(), fi, qi, so, st.len(), npyb, ok, tile, t0.elapsed().as_secs_f64());
    std::fs::write(format!("{out}/pack.json"), &js)?;
    println!("{js}");
    ok.then_some(()).ok_or_else(|| io::Error::other("gf17 roundtrip mismatch"))
}
