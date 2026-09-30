use crate::*;
use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
type Ad = Autodiff<NdArray>;
const DEV: NdArrayDevice = NdArrayDevice::Cpu;
fn tiny(s: &str, rb: u32, shb: usize) -> Cfg { Cfg { d: 64, l: 1, h: 1, t: 64, ff: 128, ns: vec![2, 3, 4], rb, tern: true, mem: true, v: 256, bpe: String::new(), bs: 4, micro: 2, base: String::new(), init: String::new(), lazy: true, sw: String::new(), ep: 0.0, seed: 3, par: 2, wall: 0.0, unit: 50.0, home: s.into(), corpus: "syn".into(), out: format!("{s}/base"), thf: format!("{s}/thr.txt"), steps: 30, warm: 5, lr: 3e-3, lrm: 3e-3, ev: 10, ck: 10, evb: 1, lpe: false, lphi: 4.0, lsq: 64, copy: false, rmix: 0.0, shb, ..Cfg::env() } }
fn dir(tag: &str) -> String { let s = std::env::temp_dir().join(format!("adamt-shard-{tag}-{}", std::process::id())).to_string_lossy().to_string(); let _ = fs::remove_dir_all(&s); fs::create_dir_all(format!("{s}/data/syn")).unwrap(); fs::write(format!("{s}/thr.txt"), b"0\n").unwrap(); let txt: Vec<u8> = (0..40000u64).map(|i| b"the cat sat on a mat. a dog ran to the cat. "[(i % 44) as usize] ^ (sm(i) % 50 == 0) as u8).collect(); fs::write(format!("{s}/data/syn/train.bin"), &txt).unwrap(); fs::write(format!("{s}/data/syn/hold.bin"), &txt[..8000]).unwrap(); s }
fn bits(v: &[f32]) -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() }
const SHB: [usize; 4] = [1 << 30, 512 * 256, 64 * 256, 5 * 256];
#[test]
fn sharded_memory_is_bit_exact_to_unsharded() {
    let s = dir("unit");
    let xy = |o: u64| -> (Vec<u32>, Vec<u32>) { let v: Vec<u32> = (0..2 * 64 + 1u64).map(|i| (sm(i + o) % 40) as u32).collect(); (v[..128].to_vec(), v[1..].to_vec()) };
    <NdArray as Backend>::seed(&DEV, 7);
    let core = Core::<Ad>::new(&tiny(&s, 8, SHB[0]), &DEV);
    Mem::<NdArray>::of(Mem::<NdArray>::new(&tiny(&s, 8, SHB[0]), &DEV).ts(), ParamId::from(42)).save(&format!("{s}/ref")).unwrap();
    let run = |shb: usize| -> Vec<Vec<u32>> {
        let c = tiny(&s, 8, shb);
        let m = Mem::<Ad>::load(&c, &format!("{s}/ref"), &DEV).unwrap();
        assert!(m.tab.iter().map(|p| p.dims()[0]).eq(c.cut()) && c.cut().iter().sum::<usize>() == c.rows() && m.n() == c.rows() * c.d);
        let (x, y) = xy(0);
        let lg = logits(&c, &core, Some(&m), None, &x, 2, &DEV);
        let mut gr = ce(lg.clone(), &y, &DEV).backward();
        let gd: Vec<f32> = m.tab.iter().flat_map(|p| p.val().grad(&gr).map_or_else(|| vec![0f32; p.dims()[0] * p.dims()[1]], f1)).collect();
        let gp = GradientsParams::from_module(&mut gr, &m);
        let md = AdamWConfig::new().with_weight_decay(0.0).with_beta_2(0.95).init::<Ad, Mem<Ad>>().step(1e-2, m.clone(), gp);
        let (mut lo, mut ml) = (Lazy::<NdArray>::new(&c, &DEV), m.clone());
        for t in 1..=3 {
            let (x, y) = xy(t as u64 * 977);
            let (ix, lf) = mleaf(&c, &ml, &x, 2, &DEV);
            let g = lf.grad(&ce(lgm(&c, &core, Some(lf.clone()), None, &x, 2, &DEV), &y, &DEV).backward()).unwrap();
            let (o, tb) = lo.step(ml.tab.iter().map(|p| p.val().inner()).collect(), vec![(ix, g)], c.k(), 1e-2, t, &DEV);
            (lo, ml) = (o, Mem::of(tb.into_iter().map(Tensor::from_inner).collect(), ml.tab[0].id));
        }
        let cg = Cfg { rb: 10, ..c.clone() };
        let mg = ml.valid().grow(&cg, 8, &DEV);
        assert!(mg.tab.iter().map(|p| p.dims()[0]).eq(cg.cut()));
        let fx = Mem::<NdArray>::of(ml.valid().ts(), ParamId::from(42));
        fx.save(&format!("{s}/m{shb}")).unwrap();
        lo.save(&format!("{s}/o{shb}")).unwrap();
        let (cv, p) = (core.valid(), format!("{s}/g{shb}.gf17"));
        lut::save(&p, &cg, &cv, Some(&mg), None, 128).unwrap();
        let sp: Vec<usize> = (0..9).map(|i| i * ml.n() / 8).collect();
        vec![bits(&m.flat()), bits(&f1(lg)), bits(&gd), bits(&md.flat()), bits(&ml.flat()), bits(&lo.m.iter().chain(&lo.v).flat_map(|t| f1(t.clone())).collect::<Vec<f32>>()), bits(&mg.flat()), bits(&f1(logits(&cg, &cv, Some(&mg), None, &x, 2, &DEV))), bits(&sp.windows(2).flat_map(|w| ml.span(w[0] + 7, w[1] + 3)).collect::<Vec<f32>>()), fs::read(format!("{s}/m{shb}.bin")).unwrap().iter().map(|&b| b as u32).collect(), fs::read(&p).unwrap().iter().map(|&b| b as u32).collect()]
    };
    let r0 = run(SHB[0]);
    SHB[1..].iter().for_each(|&shb| { let c = tiny(&s, 8, shb); let r = run(shb); println!("{{\"shb\": {shb}, \"shards\": {}, \"equal\": {:?}}}", c.cut().len(), r.iter().zip(&r0).map(|(a, b)| a == b).collect::<Vec<_>>()); assert!(c.cut().len() > 1 && r == r0, "sharded memory must be bit-identical to the unsharded table (init, logits, grads, dense AdamW, lazy Adam, growth, span, checkpoint bytes, .gf17 bytes)"); });
    let (c0, c3) = (tiny(&s, 8, SHB[0]), tiny(&s, 8, SHB[3]));
    let (a, b) = (Mem::<NdArray>::load(&c3, &format!("{s}/m{}", SHB[0]), &DEV).unwrap(), Mem::<NdArray>::load(&c0, &format!("{s}/m{}", SHB[3]), &DEV).unwrap());
    assert!(a.tab.len() == c3.cut().len() && b.tab.len() == 1 && a.flat() == b.flat() && a.tab[0].id == ParamId::from(42) && b.tab[0].id == ParamId::from(42), "checkpoints cross-load between shard layouts");
    assert!(Lazy::<NdArray>::load(&c3, &format!("{s}/o{}", SHB[0]), &DEV).unwrap().m.iter().flat_map(|t| f1(t.clone())).eq(Lazy::<NdArray>::load(&c0, &format!("{s}/o{}", SHB[3]), &DEV).unwrap().m.iter().flat_map(|t| f1(t.clone()))));
    assert!(Mem::<NdArray>::load(&tiny(&s, 7, SHB[2]), &format!("{s}/m{}", SHB[0]), &DEV).is_err(), "a table of the wrong shape must not load");
    let _ = fs::remove_dir_all(&s);
}
#[test]
fn sharded_training_resume_and_growth_match_unsharded() {
    let s = dir("e2e");
    let c0 = tiny(&s, 6, SHB[0]);
    train::<Ad>(&c0, DEV, 0).unwrap();
    let run = |tag: &str, shb: usize, lazy: bool, split: bool| { let c = Cfg { init: format!("{s}/base"), rb: 9, lazy, out: format!("{s}/{tag}"), ..tiny(&s, 6, shb) }; if split { fs::create_dir_all(&c.out).unwrap(); fs::write(format!("{}/STOP", c.out), b"").unwrap(); train::<Ad>(&c, DEV, 0).unwrap(); assert_eq!(fs::read_to_string(format!("{}/ckpt/state.txt", c.out)).unwrap().trim(), "1"); fs::remove_file(format!("{}/STOP", c.out)).unwrap(); } train::<Ad>(&c, DEV, 0).unwrap(); c };
    let js = |d: &str| fs::read_to_string(format!("{d}/log.jsonl")).unwrap().lines().filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()).filter_map(|j| j["val_nats"].as_f64()).map(f64::to_bits).collect::<Vec<u64>>();
    let get = |c: &Cfg| { let cc = Cfg { shb: SHB[0], ..Cfg::from_dir(&c.out).unwrap() }; let m = load::<NdArray>(&cc, &DEV).unwrap().1.unwrap(); let o = c.lazy.then(|| Lazy::<NdArray>::load(&cc, &format!("{}/ckpt", c.out), &DEV).unwrap()).map(|o| o.m.iter().chain(&o.v).flat_map(|t| f1(t.clone())).collect::<Vec<f32>>()).unwrap_or_default(); (bits(&m.flat()), bits(&o), js(&c.out), fs::read(format!("{}/model.gf17", c.out)).unwrap()) };
    [true, false].iter().for_each(|&lz| {
        let (u, h) = (get(&run(&format!("u{lz}"), SHB[0], lz, false)), get(&run(&format!("s{lz}"), SHB[2], lz, true)));
        println!("{{\"lazy\": {lz}, \"shards\": {}, \"mem_eq\": {}, \"moments_eq\": {}, \"val_eq\": {}, \"gf17_eq\": {}, \"val\": {:?}}}", tiny(&s, 9, SHB[2]).cut().len(), u.0 == h.0, u.1 == h.1, u.2 == h.2, u.3 == h.3, u.2.iter().map(|&b| f64::from_bits(b)).collect::<Vec<_>>());
        assert!(u.2.len() == 3 && u == h, "a sharded, split and resumed continuation from a grown memory must match the unsharded run bit for bit (lazy {lz})");
    });
    let _ = fs::remove_dir_all(&s);
}
#[test]
#[ignore]
fn legacy_checkpoint_reshards_and_saves_byte_exact() {
    let (d, s) = (std::env::var("SHARD_COMPAT_DIR").expect("SHARD_COMPAT_DIR=<run dir>"), dir("compat"));
    let c = Cfg::from_dir(&d).unwrap();
    let cc = Cfg { shb: (c.rows() * c.d * 4 / 5).max(4 * c.d), ..c.clone() };
    let n = ["mem", "lm", "lv"].iter().filter(|n| Path::new(&format!("{d}/ckpt/{n}.bin")).exists()).map(|n| { let (ts, id) = tload::<NdArray>(&cc, &format!("{d}/ckpt/{n}"), &DEV).unwrap(); tsave(&ts, id, &format!("{s}/{n}")).unwrap(); let eq = fs::read(format!("{d}/ckpt/{n}.bin")).unwrap() == fs::read(format!("{s}/{n}.bin")).unwrap(); println!("{{\"table\": \"{n}\", \"shards\": {}, \"bytes_equal\": {eq}}}", ts.len()); assert!(ts.len() > 1 && eq, "{n}: a legacy checkpoint must reload into shards and save back byte for byte"); }).count();
    assert!(n > 0);
    let _ = fs::remove_dir_all(&s);
}
