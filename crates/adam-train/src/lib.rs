pub mod cfg;
pub mod cpu;
pub mod gpu;
pub mod v19;
use adam_ray::{npy::Npz, pack::copies, rng::{pairwise, Pcg64}};
use cfg::*;
use cpu::sid0;
use half::f16;
use memmap2::Mmap;
use std::{fs::{self, File}, io::{self, Write}, time::Instant};
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Part { Dp, Dm, Dv, Sid, Sw, Sm, Sv, Bias, Lq }
pub trait Engine {
    fn run(&mut self, x: &[u8], y: &[u8], nb: usize, t: usize, salt: u32, train: Option<(usize, f32)>) -> f64;
    fn logits(&mut self, x: &[u8]) -> Vec<f32>;
    fn refresh(&mut self);
    fn get(&mut self, p: Part) -> Vec<f32>;
    fn put(&mut self, p: Part, v: &[f32]);
    fn name(&self) -> String;
}
pub fn engine(c: &Cfg) -> Box<dyn Engine> { if c.arch == "cpu" { Box::new(cpu::Cpu::new(c)) } else { match gpu::Gpu::new(c) { Ok(g) => Box::new(g), Err(e) => { eprintln!("gpu unavailable ({e}), using cpu"); Box::new(cpu::Cpu::new(c)) } } } }
pub fn npy(path: &str, descr: &str, shape: &[usize], data: &[u8]) -> io::Result<()> {
    let sh = shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ") + if shape.len() == 1 { "," } else { "" };
    let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({sh}), }}");
    h += &" ".repeat(63 - (10 + h.len()) % 64);
    let mut f = io::BufWriter::new(File::create(path)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&((h.len() + 1) as u16).to_le_bytes())?;
    f.write_all(h.as_bytes())?;
    f.write_all(b"\n")?;
    f.write_all(data)
}
pub fn f16b(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|&x| f16::from_f32(x).to_le_bytes()).collect() }
pub fn f32b(v: &[f32]) -> Vec<u8> { bytemuck::cast_slice(v).to_vec() }
fn mmap(p: &str) -> io::Result<Mmap> { unsafe { Mmap::map(&File::open(p).map_err(|e| io::Error::new(e.kind(), format!("{p}: {e}")))?) } }
pub fn corpus(c: &Cfg, name: &str) -> io::Result<(Mmap, Mmap)> { let d = format!("{}/data/{name}", c.home); Ok((mmap(&format!("{d}/train.bin"))?, mmap(&format!("{d}/hold.bin"))?)) }
pub fn seps(d: &[u8], t: usize) -> Vec<i64> { d.iter().enumerate().filter(|&(i, &b)| b == 30 && i < d.len() - t - 1).map(|(i, _)| i as i64).collect() }
pub fn bix(n: usize, g: &mut Pcg64, bs: usize, t: usize, sp: Option<&[i64]>) -> Vec<i64> {
    let mut ix = g.integers(0, (n - t - 1) as i64, bs);
    if let Some(sp) = sp.filter(|s| !s.is_empty()) { g.integers(0, sp.len() as i64, bs / 2).iter().enumerate().for_each(|(k, &r)| ix[k] = sp[r as usize]); }
    ix
}
pub fn batch(d: &[u8], g: &mut Pcg64, bs: usize, t: usize, sp: Option<&[i64]>) -> (Vec<u8>, Vec<u8>) {
    let ix = bix(d.len(), g, bs, t, sp);
    (ix.iter().flat_map(|&i| d[i as usize..i as usize + t].iter().copied()).collect(), ix.iter().flat_map(|&i| d[i as usize + 1..i as usize + 1 + t].iter().copied()).collect())
}
fn npz_f32(p: &str) -> io::Result<(Vec<usize>, Vec<f32>)> { let z = Npz::open(p)?; let a = z.get("")?; Ok((a.shape.clone(), if a.descr.ends_with("u1") || a.descr.ends_with("i4") || a.descr.ends_with("i8") { a.i64s().iter().map(|&v| v as f32).collect() } else { a.f32s() })) }
pub fn load_pages(c: &Cfg, e: &mut dyn Engine) -> io::Result<()> {
    let (p, mode) = [(&c.pages, 0), (&c.pages_v19b, 3), (&c.pages_v18, 2), (&c.pages_v16, 1)].into_iter().find(|(p, _)| !p.is_empty()).unwrap_or((&c.pages, 0));
    if p.is_empty() { return Ok(()); }
    let pre = p.trim_end_matches("_sid.npy");
    e.put(Part::Dp, &npz_f32(&format!("{pre}_dp.npy"))?.1);
    e.put(Part::Bias, &npz_f32(&format!("{pre}_bias.npy"))?.1);
    let ((sh, sid), (_, sw)) = (npz_f32(&format!("{pre}_sid.npy"))?, npz_f32(&format!("{pre}_sw.npy"))?);
    assert!(sh[1] == c.te && sh[2] == KS, "pages shape {sh:?}");
    if mode == 0 {
        let (n, oc, l) = (c.te * KS, copies(sh[0]).0, sh[0]);
        let k = if oc > 0 && oc < c.copy { c.geo.c0() + oc } else { l };
        let (mut si, mut ww): (Vec<f32>, Vec<f32>) = (sid0(c.lv * c.te).iter().map(|&v| v as f32).collect(), vec![0f32; c.lv * n]);
        [(0, 0, k), (k, c.geo.hb(), l - k)].iter().for_each(|&(a, b, m)| { si[b * n..(b + m) * n].copy_from_slice(&sid[a * n..(a + m) * n]); ww[b * n..(b + m) * n].copy_from_slice(&sw[a * n..(a + m) * n]); });
        e.put(Part::Sid, &si);
        e.put(Part::Sw, &ww);
        println!("loaded slots {p} levels {} of {}{}", l, c.lv, if k < l { format!(" (copy {oc} -> {}: 0..{} kept, {k}..{} -> {}.., fresh {k}..{})", c.copy, k - 1, l - 1, c.geo.hb(), c.geo.hb() - 1) } else { String::new() });
        return Ok(());
    }
    let (tl, c0, hb, n, hops, l) = (c.tl(), c.geo.c0(), c.geo.hb(), c.te * KS, c.geo.hops, sh[0]);
    let (k, hs) = [(0, 0), ((tl + 1).min(l), tl + 1), (c0.min(l), c0), (24.min(c0).min(l), l.saturating_sub(hops))][mode];
    let nh = hops.min(l.saturating_sub(hs));
    let (mut si, mut ww) = (e.get(Part::Sid), e.get(Part::Sw));
    si[..k * n].copy_from_slice(&sid[..k * n]);
    ww[..k * n].copy_from_slice(&sw[..k * n]);
    si[hb * n..(hb + nh) * n].copy_from_slice(&sid[hs * n..(hs + nh) * n]);
    ww[hb * n..(hb + nh) * n].copy_from_slice(&sw[hs * n..(hs + nh) * n]);
    e.put(Part::Sid, &si);
    e.put(Part::Sw, &ww);
    println!("warm {} {p} prefix 0..{} hops {hs}..{} -> {hb}..{}", ["", "v16", "v18", "v19b"][mode], k as i64 - 1, (hs + nh) as i64 - 1, (hb + nh) as i64 - 1);
    Ok(())
}
pub fn dense_used(dp: &[f32]) -> Vec<f32> { [&dp[..V * V], &dp[DE * V..]].concat() }
pub fn dump(c: &Cfg, e: &mut dyn Engine, dir: &str) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let (lv, te) = (c.lv, c.te);
    for (p, nm) in [(Part::Dp, "dp"), (Part::Dm, "dm"), (Part::Dv, "dv")] { npy(&format!("{dir}/{nm}.npy"), "<f4", &[V + DE, V], &f32b(&dense_used(&e.get(p))))?; }
    for (p, nm) in [(Part::Sid, "sid"), (Part::Sw, "sw"), (Part::Sm, "sm"), (Part::Sv, "sv")] { npy(&format!("{dir}/{nm}.npy"), "<f4", &[lv, te, KS], &f32b(&e.get(p)))?; }
    npy(&format!("{dir}/bias.npy"), "<f4", &[V], &f32b(&e.get(Part::Bias)))?;
    npy(&format!("{dir}/lq.npy"), "<f4", &[2 * lv], &f32b(&e.get(Part::Lq)))
}
pub fn ckpt(c: &Cfg, e: &mut dyn Engine, it: usize, best: f64, hist: &[String]) -> io::Result<()> {
    let tmp = format!("{}.tmp", c.ckpt);
    let _ = fs::remove_dir_all(&tmp);
    dump(c, e, &tmp)?;
    fs::write(format!("{tmp}/state.txt"), format!("{it} {best:.17e}\n{}\n", hist.join("\n")))?;
    let _ = fs::remove_dir_all(&c.ckpt);
    fs::rename(&tmp, &c.ckpt)
}
pub fn resume(c: &Cfg, e: &mut dyn Engine) -> io::Result<Option<(usize, f64, Vec<String>)>> {
    let Some((d, s)) = [c.ckpt.clone(), format!("{}.tmp", c.ckpt)].into_iter().find_map(|d| fs::read_to_string(format!("{d}/state.txt")).ok().map(|s| (d, s))) else { return Ok(None) };
    let (mut ln, full) = (s.lines(), |u: Vec<f32>| { let mut f = vec![0f32; 2 * DE * V]; f[..V * V].copy_from_slice(&u[..V * V]); f[DE * V..].copy_from_slice(&u[V * V..]); f });
    let h: Vec<String> = ln.next().unwrap_or("").split(' ').map(String::from).collect();
    for (p, nm) in [(Part::Dp, "dp"), (Part::Dm, "dm"), (Part::Dv, "dv"), (Part::Sid, "sid"), (Part::Sw, "sw"), (Part::Sm, "sm"), (Part::Sv, "sv"), (Part::Bias, "bias"), (Part::Lq, "lq")] { let v = npz_f32(&format!("{d}/{nm}.npy"))?.1; e.put(p, &if nm.starts_with('d') { full(v) } else { v }); }
    println!("resumed {d} at step {}", h[0]);
    Ok(Some((h[0].parse().unwrap(), h[1].parse().unwrap(), ln.filter(|l| !l.is_empty()).map(String::from).collect())))
}
fn js(s: &str) -> String { s.chars().fold(String::from("\""), |mut o, ch| { match ch { '"' => o += "\\\"", '\\' => o += "\\\\", '\n' => o += "\\n", '\r' => o += "\\r", '\t' => o += "\\t", c if (c as u32) < 32 => o += &format!("\\u{:04x}", c as u32), c => o.push(c) } o }) + "\"" }
fn r4(x: f64) -> f64 { (x * 1e4).round_ties_even() / 1e4 }
fn thr(p: &str) -> f64 { fs::read_to_string(p).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0.0) }
pub fn sample(c: &Cfg, e: &mut dyn Engine, pfx: &str) -> String {
    let (mut buf, mut g, ok): (Vec<u8>, Pcg64, Vec<bool>) = (pfx.as_bytes().to_vec(), Pcg64::new(Some(7)), (0..V).map(|k| (32..127).contains(&k) || k == 9 || k == 10).collect());
    for _ in 0..300 {
        let lg = e.logits(&buf[buf.len().saturating_sub(c.ctx)..]);
        let mut z: Vec<f64> = (0..V).map(|k| if ok[k] { lg[k] as f64 / c.gen_t } else { -1e9 }).collect();
        let mut o: Vec<usize> = (0..V).collect();
        o.sort_by(|&a, &b| z[a].partial_cmp(&z[b]).unwrap());
        o[..V - c.gen_k].iter().for_each(|&k| z[k] = -1e9);
        let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let p: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
        let s = pairwise(&p);
        buf.push(g.choice(&p.iter().map(|x| x / s).collect::<Vec<_>>()) as u8);
    }
    String::from_utf8_lossy(&buf).into_owned()
}
pub fn train(c: &Cfg, e: &mut dyn Engine) -> io::Result<Vec<f64>> {
    let t0 = Instant::now();
    let names = if c.multi.is_empty() { vec![c.corpus.clone()] } else { c.multi.clone() };
    let corp = names.iter().map(|n| corpus(c, n)).collect::<io::Result<Vec<_>>>()?;
    load_pages(c, e)?;
    let (lv, te) = (c.lv, c.te);
    let cells = lv * te;
    let mem = format!("{{\"dense_values\": {}, \"dense_tori\": \"byte1 (256 used) + byte2 (65536)\", \"cells\": {cells}, \"slots_per_cell\": {KS}, \"stored_values\": {}, \"inference_bytes_id8_sign1\": {}, \"inference_bytes_id8_ternary\": {}, \"fetches_per_byte\": {}, \"tori\": {lv}}}", 2 * DE * V, cells * KS, ((cells * KS * 9) as f64 / 8.0).round_ties_even(), ((cells * KS) as f64 * 9.585 / 8.0).round_ties_even(), lv + 3);
    println!("=== adam-train [{}] v{} copy={} mr={} levels={} hops={} qt={} tern={} | corpus={} train {} B hold {} B | entries/level={te} ctx={} bs={} steps={} lr={}->{} rth={} rw={} wd={} | {mem}", e.name(), if c.ver == 19 { 19 } else if c.geo.qt { 18 } else { 16 }, c.copy, c.geo.mr, c.geo.base, c.geo.hops, c.geo.qt as u8, ["off", "cell", "level"][c.tern as usize], names.join(","), corp[0].0.len(), corp[0].1.len(), c.ctx, c.bs, c.steps, c.lr, c.lr_end, c.rth, c.rw, c.wd);
    let (mut g, mut hist, mut best, mut ts, mut th, mut losses) = (Pcg64::new(Some(c.seed)), vec![], 9e9f64, vec![], 0.0, vec![]);
    let st0 = if c.ck_every > 0 { resume(c, e)?.map_or(0, |(k, b, h)| { best = b; hist = h; k }) } else { 0 };
    let mut slog = (!c.step_log.is_empty()).then(|| fs::OpenOptions::new().create(true).append(st0 > 0).write(true).truncate(st0 == 0).open(&c.step_log)).transpose()?;
    let salt = |ci: usize| if c.salted { ci as u32 } else { 0 };
    let sps: Vec<[Vec<i64>; 2]> = corp.iter().map(|(a, b)| if c.ver == 19 { [seps(a, c.ctx), seps(b, c.ctx)] } else { [vec![], vec![]] }).collect();
    let sp = |ci: usize, h: usize| (c.ver == 19).then(|| sps[ci][h].as_slice());
    let evalv = |e: &mut dyn Engine, nb: usize, ci: usize| { let mut ge = Pcg64::new(Some(1234)); let v: Vec<f64> = (0..nb).map(|_| { let (x, y) = batch(&corp[ci].1, &mut ge, c.bs, c.ctx, sp(ci, 1)); e.run(&x, &y, c.bs, c.ctx, salt(ci), None) }).collect(); pairwise(&v) / nb as f64 };
    (1..=st0).for_each(|it| { let ci = it % corp.len(); bix(corp[ci].0.len(), &mut g, c.bs, c.ctx, sp(ci, 0)); });
    for it in st0 + 1..=c.steps {
        let ci = it % corp.len();
        let (x, y) = batch(&corp[ci].0, &mut g, c.bs, c.ctx, sp(ci, 0));
        let s0 = Instant::now();
        let loss = e.run(&x, &y, c.bs, c.ctx, salt(ci), Some((it, c.lr_at(it))));
        ts.push(s0.elapsed().as_secs_f64());
        losses.push(loss);
        if let Some(f) = slog.as_mut() { writeln!(f, "{it} {loss:.17e}")?; }
        th = if it % 50 == 0 || it == 1 { thr(&c.thf) } else { th };
        (th > 0.0).then(|| std::thread::sleep(std::time::Duration::from_secs_f64(th)));
        if it % c.ev_every == 0 || it == 1 || it == c.steps {
            let va = evalv(e, 20, 0);
            best = best.min(va);
            let sw = e.get(Part::Sw);
            let mt = ts[ts.len().saturating_sub(200)..].iter().sum::<f64>() / ts[ts.len().saturating_sub(200)..].len() as f64;
            let rec = format!("{{\"step\": {it}, \"train_nats\": {}, \"val_nats\": {}, \"val_bits_per_byte\": {}, \"ms_per_step\": {}, \"positions_per_s\": {}, \"live_slots_frac\": {}, \"w_abs_mean\": {}, \"elapsed_s\": {}, \"throttle_s\": {th}}}", r4(loss), r4(va), r4(va / std::f64::consts::LN_2), (mt * 1e4).round() / 10.0, (c.npos() as f64 / mt).round(), r4(sw.iter().filter(|w| w.abs() >= c.rth).count() as f64 / sw.len() as f64), r4(sw.iter().map(|w| w.abs() as f64).sum::<f64>() / sw.len() as f64), (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0);
            if let Some(f) = slog.as_mut() { writeln!(f, "eval {it} {va:.17e}")?; }
            println!("{rec}");
            hist.push(rec);
        }
        (c.ck_every > 0 && it % c.ck_every == 0 && it < c.steps).then(|| ckpt(c, e, it, best, &hist)).transpose()?;
    }
    (c.tern > 0).then(|| e.refresh());
    if c.steps == 0 { best = evalv(e, 60, 0); println!("eval-only {}: {:.4} bits/byte", c.corpus, best / std::f64::consts::LN_2); }
    let per: Vec<String> = c.multi.iter().enumerate().map(|(i, n)| format!("{}: {}", js(n), r4(evalv(e, 60, i) / std::f64::consts::LN_2))).collect();
    let pre = c.out.trim_end_matches(".json").to_string();
    if c.steps > 0 {
        fs::create_dir_all(std::path::Path::new(&c.out).parent().unwrap())?;
        npy(&format!("{pre}_dp.npy"), "<f2", &[2, DE, V], &f16b(&e.get(Part::Dp)))?;
        npy(&format!("{pre}_sid.npy"), "|u1", &[lv, te, KS], &e.get(Part::Sid).iter().map(|&v| v as u8).collect::<Vec<_>>())?;
        let (mut sw, lq) = (e.get(Part::Sw), e.get(Part::Lq));
        (c.tern == 2).then(|| sw.iter_mut().enumerate().filter(|(i, _)| !(c.geo.c0()..c.geo.hb()).contains(&(i / (te * KS)))).for_each(|(i, w)| { let l = i / (te * KS); *w = if w.abs() < lq[l] { 0.0 } else { w.signum() * (*w != 0.0) as u8 as f32 * lq[lv + l] } }));
        npy(&format!("{pre}_sw.npy"), "<f2", &[lv, te, KS], &f16b(&sw))?;
        npy(&format!("{pre}_bias.npy"), "<f4", &[V], &f32b(&e.get(Part::Bias)))?;
    }
    (!c.dump.is_empty()).then(|| dump(c, e, &c.dump)).transpose()?;
    let samples: Vec<String> = ["The weather today is", "def add(a, b):\n    return", "Kaolinite is a clay mineral that"].iter().map(|p| { let s = sample(c, e, p); println!("\n==== {:?}\n{s}", &p[..16]); format!("{}: {}", js(&p[..16]), js(&s)) }).collect();
    let cfg = format!("{{\"entries\": {te}, \"ctx\": {}, \"bs\": {}, \"steps\": {}, \"lr\": {}, \"lr_end\": {}, \"corpus\": {}, \"slots\": {KS}, \"rth\": {}, \"rw\": {}, \"wd\": {}, \"gate\": 0.0, \"tern\": {}, \"tern_th\": {}, \"tern_scope\": \"{}\", \"qt\": {}, \"copy\": {}, \"mr\": {}, \"topic\": {}, \"levels\": {lv}{}{}{}, \"trainer\": \"adam-train {}\"}}", c.ctx, c.bs, c.steps, c.lr, c.lr_end, js(&c.corpus), c.rth, c.rw, c.wd, c.tern > 0, c.tern_th, if c.tern == 1 { "cell" } else { "level" }, c.geo.qt as u8, c.copy, c.geo.mr, c.tl(), if c.geo.mv { ", \"maskv\": 1" } else { "" }, if c.geo.lt { ", \"latest\": 1" } else { "" }, if c.geo.gn { ", \"gen\": 1" } else { "" }, e.name());
    let bv = r4(best / std::f64::consts::LN_2);
    fs::write(&c.out, format!("{{\n \"cfg\": {cfg},\n \"memory\": {mem},\n \"hist\": [\n  {}\n ],\n \"best_val_bits_per_byte\": {bv},\n \"per_corpus\": {{{}}},\n \"salted\": {},\n \"multi\": [{}],\n \"samples\": {{{}}}\n}}\n", hist.join(",\n  "), per.join(", "), c.salted, c.multi.iter().map(|m| js(m)).collect::<Vec<_>>().join(", "), samples.join(", ")))?;
    println!("best val {bv} bits/byte | total {:.1}s -> {}", t0.elapsed().as_secs_f64(), c.out);
    Ok(losses)
}
