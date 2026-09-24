use adam_ray::pack::Geo;
use std::env::var;
pub const V: usize = 256;
pub const DE: usize = 65536;
pub const KS: usize = 3;
#[derive(Clone)]
pub struct Cfg { pub geo: Geo, pub lv: usize, pub te: usize, pub ctx: usize, pub bs: usize, pub steps: usize, pub lr: f64, pub lr_end: f64, pub tag: String, pub corpus: String, pub multi: Vec<String>, pub salted: bool, pub rth: f32, pub rw: f32, pub wd: f32, pub tern: u32, pub tern_th: f32, pub qlev: u32, pub sgrid: bool, pub slo: f32, pub shi: f32, pub seed: u64, pub arch: String, pub pages: String, pub pages_v16: String, pub syn: String, pub wcnt: String, pub home: String, pub out: String, pub thf: String, pub gen_t: f64, pub gen_k: usize, pub step_log: String, pub dump: String }
fn e(k: &str, d: &str) -> String { var(k).unwrap_or_else(|_| d.to_string()) }
fn n<T: std::str::FromStr>(k: &str, d: &str) -> T { e(k, d).trim().parse().unwrap_or_else(|_| panic!("bad {k}")) }
fn f(k: &str, d: &str) -> f32 { n::<f64>(k, d) as f32 }
impl Cfg {
    pub fn env() -> Cfg {
        let (hops, base, qt, l2): (usize, usize, bool, u32) = (n("HOPS", "2"), n("LEVELS", "8"), e("QT", "1") == "1", n("LOG2_ENTRIES", "16"));
        assert!(l2 >= 16 && n::<usize>("SLOTS", "3") == KS && f("GATE", "0") == 0.0, "LOG2_ENTRIES>=16, SLOTS=3 and GATE=0 are required");
        let (te, ta, bs, home) = (1usize << l2, f("TOPIC_ALPHA", "0.99"), n::<usize>("BS", "256"), e("AMNI_HOME", "."));
        let corpus = e("CORPUS", "big");
        let geo = Geo { base, hops, t: te as u32, qt, salt: 0, ta, tc: 1.0 - ta };
        let tag = e("TAG", &format!("v16_e{l2}_bs{bs}"));
        let (tern, cell) = (e("TERN", "0") == "1", e("TERN_SCOPE", "level") == "cell");
        Cfg { lv: geo.lv(), geo, te, ctx: n("CTX", "128"), bs, steps: n("STEPS", "10000"), lr: n("LR", "5e-3"), lr_end: n("LR_END", "5e-4"), out: format!("{}/rayhash_{tag}.json", e("EXPORTS", &format!("{home}/exports/gf17_continuum"))), tag, multi: e("MULTI", "").split(',').filter(|c| !c.is_empty()).map(String::from).collect(), salted: e("SALTED", "1") == "1", rth: f("RTH", "0.1"), rw: f("RW", "0.05"), wd: f("WD", "1e-3"), tern: if !tern { 0 } else if cell { 1 } else { 2 }, tern_th: f("TERN_TH", "0.5"), qlev: n("QLEV", "3"), sgrid: e("SGRID", "0") == "1", slo: f("SLO", "-6.5"), shi: f("SHI", "2.0"), seed: n("SEED", "0"), arch: e("ARCH", "vulkan"), pages: e("PAGES", ""), pages_v16: e("PAGES_V16", ""), syn: e("SYNSETS", &format!("{home}/data/{corpus}/synsets.txt")), wcnt: format!("{home}/data/big2/wordcounts.txt"), thf: e("THROTTLE_FILE", &format!("{home}/throttle.txt")), gen_t: n("GEN_T", "0.7"), gen_k: n("GEN_K", "12"), step_log: e("STEP_LOG", ""), dump: e("DUMP_F32", ""), corpus, home }
    }
    pub fn npos(&self) -> usize { self.bs * self.ctx }
    pub fn lr_at(&self, it: usize) -> f32 { (self.lr + (self.lr_end - self.lr) * (it as f64 / self.steps.max(1) as f64).min(1.0)) as f32 }
    pub fn tl(&self) -> usize { self.geo.base - 1 }
}
