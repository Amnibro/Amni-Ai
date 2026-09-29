use super::*;
use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use std::collections::HashSet;
type Ad = Autodiff<NdArray>;
const DEV: NdArrayDevice = NdArrayDevice::Cpu;
fn u(i: u64) -> f64 { ((sm(i) >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 3.0 }
fn bits(v: &[f32]) -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() }
fn md(a: &[f32], b: &[f64]) -> f64 { a.iter().zip(b).map(|(&p, &q)| (p as f64 - q).abs()).fold(0f64, f64::max) }
#[allow(clippy::too_many_arguments)]
fn rmix(lg: &[f64], c: &[f64], gb: f64, ids: &[u32], b: usize, t: usize, v: usize, dc: usize, cp: bool) -> Vec<f64> {
    let w = 2 * dc + 1;
    (0..b * t).flat_map(|r| {
        let (bb, i, l) = (r / t, r % t, &lg[r * v..(r + 1) * v]);
        let mx = l.iter().cloned().fold(f64::MIN, f64::max);
        let zs: f64 = l.iter().map(|x| (x - mx).exp()).sum();
        let pv: Vec<f64> = l.iter().map(|x| (x - mx).exp() / zs).collect();
        let g = 1.0 / (1.0 + (-(c[r * w + 2 * dc] + gb)).exp());
        let sc: Vec<f64> = (0..i).map(|j| (0..dc).map(|k| c[r * w + k] * c[(bb * t + j) * w + dc + k]).sum::<f64>() / (dc as f64).sqrt()).collect();
        let m2 = sc.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = sc.iter().map(|x| (x - m2).exp()).collect();
        let z2: f64 = e.iter().sum();
        let mut pc = vec![0f64; v];
        (0..i).for_each(|j| pc[ids[bb * t + j + 1] as usize] += e[j] / z2);
        (0..v).map(move |k| if i == 0 || !cp { pv[k].ln() } else { (g * pv[k] + (1.0 - g) * pc[k]).ln() }).collect::<Vec<f64>>()
    }).collect()
}
#[test]
fn copy_mix_fwd_bwd_matches_reference_and_lut() {
    let (b, t, v, dc) = (2usize, 7usize, 11usize, 3usize);
    let w = 2 * dc + 1;
    let (lg, c, r): (Vec<f64>, Vec<f64>, Vec<f64>) = ((0..b * t * v).map(|i| u(i as u64)).collect(), (0..b * t * w).map(|i| u(i as u64 + 1000)).collect(), (0..b * t * v).map(|i| u(i as u64 + 5000)).collect());
    let (ids, gb): (Vec<u32>, f64) = ((0..b * t).map(|i| (sm(i as u64 + 77) % 5) as u32).collect(), 0.3);
    let loss = |lg: &[f64], c: &[f64], gb: f64| rmix(lg, c, gb, &ids, b, t, v, dc, true).iter().zip(&r).map(|(a, b)| a * b).sum::<f64>();
    let fd = |x: &[f64], f: &dyn Fn(&[f64]) -> f64| -> Vec<f64> { (0..x.len()).map(|i| { let (mut a, mut z) = (x.to_vec(), x.to_vec()); a[i] += 1e-6; z[i] -= 1e-6; (f(&a) - f(&z)) / 2e-6 }).collect() };
    let (o_ref, glg, gc, ggb) = (rmix(&lg, &c, gb, &ids, b, t, v, dc, true), fd(&lg, &|x| loss(x, &c, gb)), fd(&c, &|x| loss(&lg, x, gb)), (loss(&lg, &c, gb + 1e-6) - loss(&lg, &c, gb - 1e-6)) / 2e-6);
    let tf = |x: &[f64], s: [usize; 2]| Tensor::<Ad, 2>::from_data(TensorData::new(x.iter().map(|&a| a as f32).collect::<Vec<f32>>(), s), &DEV);
    let (la, ca, ga) = (tf(&lg, [b * t, v]).require_grad(), tf(&c, [b * t, w]).require_grad(), tf(&[gb], [1, 1]).require_grad());
    let o = cmix(la.clone(), ca.clone(), ga.clone(), it2(ids.iter().map(|&x| x as i32).collect(), [b, t], &DEV), b, t);
    let g = (o.clone() * tf(&r, [b * t, v])).sum().backward();
    let (ov, dl, dcg, dg) = (f1(o), f1(la.grad(&g).unwrap()), f1(ca.grad(&g).unwrap()), f1(ga.grad(&g).unwrap()));
    let lt: Vec<f32> = (0..b).flat_map(|bb| { let cs: Vec<Vec<f32>> = (0..t).map(|i| c[(bb * t + i) * w..(bb * t + i + 1) * w].iter().map(|&a| a as f32).collect()).collect(); let kc: Vec<f32> = cs.iter().flat_map(|q| q[dc..2 * dc].to_vec()).collect(); let x = ids[bb * t..(bb + 1) * t].to_vec(); (0..t).flat_map(|i| lut::cmix(dc, gb as f32, &lg[(bb * t + i) * v..(bb * t + i + 1) * v].iter().map(|&a| a as f32).collect::<Vec<f32>>(), &cs[i], &kc, &x, i)).collect::<Vec<f32>>() }).collect();
    let gm = glg.iter().chain(&gc).fold(ggb.abs(), |m, v| m.max(v.abs()));
    let (eo, el, ec, eg, elut, nv) = (md(&ov, &o_ref), md(&dl, &glg), md(&dcg, &gc), (dg[0] as f64 - ggb).abs(), md(&lt, &o_ref), md(&ov, &rmix(&lg, &c, gb, &ids, b, t, v, dc, false)));
    let norm = ov.chunks_exact(v).map(|q| (q.iter().map(|&x| (x as f64).exp()).sum::<f64>() - 1.0).abs()).fold(0f64, f64::max);
    println!("{{\"copy_out_max_abs_diff\": {eo:.3e}, \"grad_lg\": {el:.3e}, \"grad_c\": {ec:.3e}, \"grad_gb\": {eg:.3e}, \"grad_max_abs\": {gm:.3e}, \"lut_vs_ref\": {elut:.3e}, \"row_norm_err\": {norm:.3e}, \"no_copy_out_diff\": {nv:.3e}}}");
    assert!(eo < 1e-5 && elut < 1e-5 && norm < 1e-5 && nv > 1e-2, "copy mixture must match the f64 reference on the trainer and the LUT path, be normalised, and differ from the vocab-only distribution");
    assert!(el < 1e-4 * gm.max(1.0) && ec < 1e-4 * gm.max(1.0) && eg < 1e-4 * gm.max(1.0), "copy mixture gradients must match central differences");
    assert!((0..b).all(|bb| { let (q, l) = (&ov[bb * t * v..(bb * t + 1) * v], &lg[bb * t * v..(bb * t + 1) * v]); let m = l.iter().cloned().fold(f64::MIN, f64::max); let z = m + l.iter().map(|x| (x - m).exp()).sum::<f64>().ln(); q.iter().zip(l).all(|(&a, &b)| (a as f64 - (b - z)).abs() < 1e-5) }), "position 0 has no context to copy from and must be the vocab distribution");
}
fn kv_ep(r: &mut recall::R) -> Vec<u8> { let (ks, vs): (Vec<usize>, Vec<u8>) = (r.perm(16)[..6].to_vec(), (0..6).map(|_| b'a' + r.n(16) as u8).collect()); (0..6).flat_map(|i| [b'A' + ks[i] as u8, vs[i]]).chain(r.perm(6).into_iter().flat_map(|i| [b'?', b'A' + ks[i] as u8, vs[i]])).chain([b'\n']).collect() }
fn kv_cfg(s: &str, tag: &str, copy: bool, steps: usize) -> Cfg { Cfg { d: 64, l: 1, h: 1, t: 64, ff: 128, ns: vec![2, 3], rb: 6, tern: true, mem: true, lazy: true, lpe: true, lphi: 4.0, lsq: 64, v: 256, bpe: String::new(), bs: 16, micro: 2, base: String::new(), init: String::new(), sw: String::new(), ep: 0.0, seed: 3, par: 2, wall: 0.0, unit: 50.0, home: s.into(), corpus: "kv".into(), out: format!("{s}/{tag}"), thf: format!("{s}/thr.txt"), steps, warm: 20, lr: 3e-3, lrm: 3e-3, ev: steps.max(2) / 2, ck: steps, evb: 1, copy, rmix: 0.0, ..Cfg::env() } }
#[test]
fn copy_head_learns_toy_kv_and_lut_is_exact() {
    let s = std::env::temp_dir().join(format!("adamt-copy-{}", std::process::id())).to_string_lossy().to_string();
    let _ = fs::remove_dir_all(&s);
    fs::create_dir_all(format!("{s}/data/kv")).unwrap();
    fs::write(format!("{s}/thr.txt"), b"0\n").unwrap();
    let mut r = recall::R(11);
    fs::write(format!("{s}/data/kv/train.bin"), (0..4000).flat_map(|_| kv_ep(&mut r)).collect::<Vec<u8>>()).unwrap();
    fs::write(format!("{s}/data/kv/hold.bin"), (0..400).flat_map(|_| kv_ep(&mut r)).collect::<Vec<u8>>()).unwrap();
    let mut rh = recall::R(999);
    let ws: Vec<Vec<u8>> = (0..32).map(|_| { let mut w = [kv_ep(&mut rh), kv_ep(&mut rh)].concat(); w.resize(65, b'\n'); w }).collect();
    let (x, y): (Vec<u32>, Vec<u32>) = (ws.iter().flat_map(|w| w[..64].iter().map(|&b| b as u32)).collect(), ws.iter().flat_map(|w| w[1..].iter().map(|&b| b as u32)).collect());
    let ans: Vec<usize> = (0..x.len()).filter(|&i| i % 64 > 0 && x[i - 1] == b'?' as u32).collect();
    let acc = |c: &Cfg| { let (co, me, cp) = load::<NdArray>(c, &DEV).unwrap(); let lg = f1(logits(c, &co, me.as_ref(), cp.as_ref(), &x, 32, &DEV)); ans.iter().filter(|&&i| lut::am(&lg[i * 256..(i + 1) * 256]) == y[i] as usize).count() as f64 / ans.len() as f64 };
    let (cb, cc) = (kv_cfg(&s, "base", false, 1000), kv_cfg(&s, "copy", true, 1000));
    let t0 = Instant::now();
    train::<Ad>(&cb, DEV, 0).unwrap();
    train::<Ad>(&cc, DEV, 0).unwrap();
    let (fb, fc) = (Cfg::from_dir(&cb.out).unwrap(), Cfg::from_dir(&cc.out).unwrap());
    let (ab, ac) = (acc(&fb), acc(&fc));
    let lg = fs::read_to_string(format!("{}/log.jsonl", cc.out)).unwrap();
    let ex = lg.lines().filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()).find(|j| j.get("export").is_some()).unwrap();
    println!("{{\"answers\": {}, \"acc_base\": {ab:.3}, \"acc_copy\": {ac:.3}, \"train_s\": {:.1}, \"export\": {ex}}}", ans.len(), t0.elapsed().as_secs_f64());
    assert!(fc.copy && !fb.copy && fc.lpe && fc.lazy && fc.core_params() == fb.core_params() + 64 * 129 + 1);
    assert!(ac >= 0.6 && ac >= ab + 0.3, "the copy head must learn the toy key-value lookup that a 1-layer core cannot: base {ab:.3} copy {ac:.3}");
    assert!(ex["roundtrip_bytes_exact"] == true && ex["roundtrip_trits_exact"] == true && ex["argmax_agree"].as_f64().unwrap() > 0.97 && ex["max_abs_dlogit"].as_f64().unwrap() < 1e-3, "{ex}");
    let lm = lut::Lm::open(&format!("{}/model.gf17", cc.out), 64 << 20).unwrap();
    let (co, me, cp) = load::<NdArray>(&fc, &DEV).unwrap();
    assert!(lm.cp.map(|(d, g)| d == 64 && g.to_bits() == f1(cp.as_ref().unwrap().gb.val())[0].to_bits()).unwrap_or(false) && lut::Lm::open(&format!("{}/model.gf17", cb.out), 64 << 20).unwrap().cp.is_none());
    let (w, gl) = (&x[..64], f1(logits(&fc, &co, me.as_ref(), cp.as_ref(), &x[..64], 1, &DEV)));
    let full = lm.fwd(w).unwrap();
    let dl = full.iter().zip(&gl).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
    let ag = full.chunks_exact(256).zip(gl.chunks_exact(256)).filter(|(p, q)| lut::am(p) == lut::am(q)).count();
    let (_, mut kv) = lm.fwd_kv(&w[..20]).unwrap();
    let st: Vec<Vec<f32>> = w[20..].iter().map(|&tk| lm.step(&mut kv, tk).unwrap()).collect();
    println!("{{\"lut_vs_burn_max_abs\": {dl:.3e}, \"argmax_agree\": {ag}}}");
    assert!(dl < 1e-3 && ag == 64, "LUT copy mixture must match the trainer");
    assert!(st.iter().enumerate().all(|(k, v)| bits(v) == bits(&full[(20 + k) * 256..(21 + k) * 256])), "KV-cached copy decoding must be bit-identical to the full window");
    fs::create_dir_all(format!("{s}/split")).unwrap();
    fs::write(format!("{s}/split/STOP"), b"").unwrap();
    let (c1, c2) = (kv_cfg(&s, "whole", true, 12), kv_cfg(&s, "split", true, 12));
    train::<Ad>(&c1, DEV, 0).unwrap();
    train::<Ad>(&c2, DEV, 0).unwrap();
    assert_eq!(fs::read_to_string(format!("{s}/split/ckpt/state.txt")).unwrap().trim(), "1");
    fs::remove_file(format!("{s}/split/STOP")).unwrap();
    train::<Ad>(&c2, DEV, 0).unwrap();
    let ((a1, _, p1), (a2, _, p2)) = (load::<NdArray>(&c1, &DEV).unwrap(), load::<NdArray>(&c2, &DEV).unwrap());
    assert!(bits(&f1(p1.unwrap().w.w.val())) == bits(&f1(p2.unwrap().w.w.val())) && bits(&f1(a1.head.w.val())) == bits(&f1(a2.head.w.val())), "a paused and resumed COPY run must match the uninterrupted run");
    let ci = Cfg { init: cc.out.clone(), lr: 0.0, lrm: 0.0, steps: 2, ev: 2, ck: 2, copy: false, out: format!("{s}/cont"), ..cc.clone() };
    train::<Ad>(&ci, DEV, 0).unwrap();
    let fi = Cfg::from_dir(&ci.out).unwrap();
    assert!(fi.copy == var("COPY").map_or(true, |v| v == "1") && (!fi.copy || bits(&lut::Lm::open(&format!("{}/model.gf17", ci.out), 64 << 20).unwrap().fwd(w).unwrap()) == bits(&full)), "continued pretraining inherits the copy head unless COPY is set");
    let _ = fs::remove_dir_all(&s);
}
fn grams(t: &str, n: usize) -> HashSet<u64> { let ws: Vec<String> = regex::Regex::new(r"[^\W_]+").unwrap().find_iter(&t.to_lowercase()).map(|m| m.as_str().to_string()).collect(); ws.windows(n).map(|g| g.iter().fold(n as u64, |h, w| sm(h ^ hsh(w.as_bytes(), 1)))).collect() }
#[test]
fn recall_generator_format_mix_and_decontam() {
    let h = Cfg::env().home;
    let bp = format!("{h}/data/clean2/bpe4096.json");
    let c = Cfg { v: if Path::new(&bp).exists() { 4096 } else { 256 }, bpe: if Path::new(&bp).exists() { bp } else { String::new() }, rmix: 0.04, ..Cfg::env() };
    let g = recall::Gen::new(&c).unwrap();
    let es: Vec<(String, Vec<recall::Q>)> = (0..4000).map(|i| g.ep(&mut recall::R(sm(0xDEC0 ^ i as u64)))).collect();
    let lk: Vec<(&String, &recall::Q)> = es.iter().flat_map(|e| e.1.iter().map(move |q| (&e.0, q))).filter(|(_, q)| q.kind < 3).collect();
    let kinds: Vec<usize> = (0..5u8).map(|k| es.iter().filter(|e| e.1.iter().any(|q| q.kind == k)).count()).collect();
    assert!(kinds.iter().enumerate().all(|(i, &k)| k > 200 || (c.v == 256 && i == 4 && k == 0)), "{kinds:?}");
    assert!(lk.iter().all(|(t, q)| q.pos < q.n && q.fact.contains(q.val.as_str()) && t.contains(q.fact.as_str()) && t.rfind(q.val.as_str()) > t.find(q.fact.as_str()).map(|p| p + q.fact.find(q.val.as_str()).unwrap_or(0))), "every lookup answer is bound by a fact in the context and repeated after it");
    assert!(lk.iter().all(|(_, q)| (2..=14).contains(&q.n)) && lk.iter().filter(|(_, q)| q.kind > 0).all(|(_, q)| (2..=12).contains(&q.n)) && (1..=6).all(|d| lk.iter().any(|(_, q)| q.kind == 0 && q.n == d + 1)), "1-6 same-type distractors per binding query");
    let rp: Vec<f64> = lk.iter().map(|(_, q)| q.pos as f64 / (q.n - 1) as f64).collect();
    let (mr, f0, fl) = (rp.iter().sum::<f64>() / rp.len() as f64, rp.iter().filter(|&&x| x == 0.0).count() as f64 / rp.len() as f64, rp.iter().filter(|&&x| x == 1.0).count() as f64 / rp.len() as f64);
    assert!((mr - 0.5).abs() < 0.03 && (f0 - fl).abs() < 0.03, "answer positions must be balanced: mean {mr:.3}, first {f0:.3}, last {fl:.3}");
    es.iter().filter(|e| e.1[0].kind >= 3).for_each(|e| assert!(e.0.matches(e.1[0].val.as_str()).count() >= e.1[0].n && !e.0.contains(" . "), "repeat spans repeat verbatim, never in the induct probe's ' .' format: {:?}", e.0));
    let mt = |k: fn(&recall::Q) -> &String| lk.iter().filter(|(_, q)| g.enc(&format!(" {}", k(q))).len() >= 2).count() as f64 / lk.len() as f64;
    let (mk, mv) = (mt(|q| &q.key), mt(|q| &q.val));
    let hits = (0..3000usize).flat_map(|it| (0..16usize).map(move |j| (it, j))).filter(|&(it, j)| g.hit(5, it, j)).count() as f64 / 48000.0;
    println!("{{\"v\": {}, \"episodes_by_kind\": {kinds:?}, \"lookup_queries\": {}, \"mean_rel_pos\": {mr:.3}, \"first\": {f0:.3}, \"last\": {fl:.3}, \"multi_token_keys\": {mk:.3}, \"multi_token_values\": {mv:.3}, \"hit_rate\": {hits:.4}}}", c.v, lk.len());
    assert!(c.v == 256 || (mk > 0.5 && mv > 0.25), "BPE-4096 keys and values must often be multi-token");
    assert!((hits - 0.04).abs() < 0.004 && !recall::Gen { mix: 0.0, ..recall::Gen::new(&c).unwrap() }.hit(5, 1, 1));
    let n = 257;
    let mut mm = memmap2::MmapMut::map_anon(2 * 20000).unwrap();
    mm.chunks_exact_mut(2).enumerate().for_each(|(i, b)| b.copy_from_slice(&((sm(i as u64) % c.v as u64) as u16).to_le_bytes()));
    let d = Dat::W(mm.make_read_only().unwrap());
    let w0 = g.window(5, 7, 3, n);
    assert!(w0.len() == n && w0 == g.window(5, 7, 3, n) && w0 != g.window(5, 7, 4, n) && w0.iter().all(|&v| (v as usize) < c.v));
    assert!(tb(&d, 5, 9, 64, n - 1, None) == d.win(&(0..64u64).map(|j| (sm(5 << 40 ^ 9 << 12 ^ j) % (d.len() - n) as u64) as usize).collect::<Vec<_>>(), n - 1), "RECALL_MIX unset keeps the old batch sampler bit for bit");
    let (hx, hy) = (0..400).fold((0usize, 0usize), |(a, b), it| { let ((x, _), (z, _)) = (tb(&d, 5, it, 64, n - 1, Some(&g)), tb(&d, 5, it, 64, n - 1, None)); (a + (0..64).filter(|&j| g.hit(5, it, j)).count(), b + (0..64).filter(|&j| x[j * (n - 1)..(j + 1) * (n - 1)] != z[j * (n - 1)..(j + 1) * (n - 1)]).count()) });
    assert!(hx == hy && hx > 0, "only the RECALL_MIX windows change: {hx} hits, {hy} changed");
    let pr = format!("{h}/data/adam_t/probe/probes.jsonl");
    let ls = |p: &Path| fs::read_dir(p).map(|r| r.filter_map(|e| e.ok()).map(|e| e.path()).collect::<Vec<_>>()).unwrap_or_default();
    let ev: Vec<String> = ls(Path::new(&format!("{h}/data/adam_t/eval"))).into_iter().flat_map(|p| if p.is_dir() { ls(&p) } else { vec![p] }).filter(|p| { let n = p.file_name().unwrap_or_default().to_string_lossy().to_string(); (n.ends_with(".json") || n.ends_with(".jsonl")) && !n.starts_with("report") && !n.contains("meta") && !n.contains("prereg_lock") }).map(|p| p.to_string_lossy().to_string()).collect();
    let (txt, src): (String, Vec<String>) = (es.iter().map(|e| e.0.clone() + "\n\n").collect(), ev.into_iter().chain(Path::new(&pr).exists().then_some(pr)).collect());
    let gen8 = grams(&txt, 8);
    let bad: Vec<(String, usize)> = src.iter().map(|f| (f.clone(), grams(&fs::read_to_string(f).unwrap_or_default(), 8).intersection(&gen8).count())).filter(|x| x.1 > 0).collect();
    println!("{{\"decontam_8gram_sources\": {}, \"generated_8grams\": {}, \"hits\": {bad:?}}}", src.len(), gen8.len());
    assert!(bad.is_empty() && (src.is_empty() || src.len() >= 5), "the generator must share no word 8-gram with the RCA probe set or the ADAM-T eval sets: {bad:?}");
}
#[test]
#[ignore]
fn copy_lut_bench() {
    let s = std::env::temp_dir().join(format!("adamt-copy-bench-{}", std::process::id())).to_string_lossy().to_string();
    fs::create_dir_all(&s).unwrap();
    <NdArray as Backend>::seed(&DEV, 11);
    let (h, it) = (Cfg::env().home, var("COPY_BENCH_IT").ok().and_then(|v| v.parse().ok()).unwrap_or(3usize));
    let base = Cfg { d: 256, l: 8, h: 4, t: 256, ff: 1024, ns: vec![], rb: 15, tern: true, mem: false, v: 4096, bpe: format!("{h}/data/clean2/bpe4096.json"), base: String::new(), lpe: true, lphi: 4.0, lsq: 64, ..Cfg::env() };
    let (core, cp) = (Core::<NdArray>::new(&base, &DEV), Cp::<NdArray>::new(&Cfg { copy: true, ..base.clone() }, &DEV));
    let w: Vec<u32> = (0..256u64).map(|i| (sm(i) % 4096) as u32).collect();
    let lms: Vec<(&str, lut::Lm)> = [("base", false), ("copy", true)].iter().map(|&(nm, c)| { let p = format!("{s}/{nm}.gf17"); lut::save(&p, &Cfg { copy: c, ..base.clone() }, &core, None::<&Mem<NdArray>>, c.then_some(&cp), 128).unwrap(); (nm, lut::Lm::open(&p, 64 << 20).unwrap()) }).collect();
    (0..2).for_each(|rep| lms.iter().for_each(|(nm, lm)| {
        let t0 = Instant::now();
        (0..it).for_each(|_| { let (_, mut kv) = lm.fwd_kv(&w[..1]).unwrap(); w[1..].iter().for_each(|&tk| { let _ = lm.step(&mut kv, tk).unwrap(); }); });
        println!("{{\"arm\": \"{nm}\", \"rep\": {rep}, \"threads\": {}, \"kv_step_ms\": {:.3}}}", rayon::current_num_threads(), t0.elapsed().as_secs_f64() * 1e3 / (it * 255) as f64);
    }));
    let _ = fs::remove_dir_all(&s);
}
