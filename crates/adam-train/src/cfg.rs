use adam_ray::pack::Geo;
use std::env::var;
pub const V: usize = 256;
pub const DE: usize = 65536;
pub const KS: usize = 3;
#[derive(Clone)]
pub struct Cfg { pub geo: Geo, pub ver: u32, pub copy: usize, pub lv: usize, pub te: usize, pub ctx: usize, pub bs: usize, pub steps: usize, pub lr: f64, pub lr_end: f64, pub tag: String, pub corpus: String, pub multi: Vec<String>, pub salted: bool, pub rth: f32, pub rw: f32, pub wd: f32, pub tern: u32, pub tern_th: f32, pub qlev: u32, pub sgrid: bool, pub slo: f32, pub shi: f32, pub seed: u64, pub arch: String, pub pages: String, pub pages_v16: String, pub pages_v18: String, pub pages_v19b: String, pub syn: String, pub wcnt: String, pub home: String, pub out: String, pub thf: String, pub gen_t: f64, pub gen_k: usize, pub step_log: String, pub dump: String, pub ck_every: usize, pub ckpt: String, pub ev_every: usize }
fn g<T: std::str::FromStr>(src: &dyn Fn(&str) -> Option<String>, k: &str, d: &str) -> T { src(k).unwrap_or_else(|| d.into()).trim().parse().unwrap_or_else(|_| panic!("bad {k}")) }
impl Cfg {
    pub fn env() -> Cfg { Cfg::from(&|k| var(k).ok()) }
    pub fn with(kv: &[(&str, &str)]) -> Cfg { Cfg::from(&|k| kv.iter().find(|(a, _)| *a == k).map(|(_, v)| v.to_string()).or_else(|| (k == "AMNI_HOME" || k == "SYNSETS").then(|| var(k).ok()).flatten())) }
    pub fn from(src: &dyn Fn(&str) -> Option<String>) -> Cfg {
        let e = |k: &str, d: &str| src(k).unwrap_or_else(|| d.to_string());
        let f = |k: &str, d: &str| g::<f64>(src, k, d) as f32;
        let (hops, base, qt, l2): (usize, usize, bool, u32) = (g(src, "HOPS", "2"), g(src, "LEVELS", "8"), e("QT", "1") == "1", g(src, "LOG2_ENTRIES", "16"));
        assert!(l2 >= 16 && g::<usize>(src, "SLOTS", "3") == KS && f("GATE", "0") == 0.0, "LOG2_ENTRIES>=16, SLOTS=3 and GATE=0 are required");
        let (te, ta, bs, home, corpus) = (1usize << l2, f("TOPIC_ALPHA", "0.99"), g::<usize>(src, "BS", "256"), e("AMNI_HOME", "."), e("CORPUS", "big"));
        let (ver, cp) = (g::<u32>(src, "VER", "18"), g::<i64>(src, "COPY", "4"));
        let copy = if ver != 19 { 0 } else if cp >= 16 { 16 } else if cp >= 13 { 13 } else if cp >= 9 { 9 } else if cp <= 0 { 0 } else { 4 };
        let geo = Geo { base, hops, t: te as u32, qt, salt: 0, ta, tc: 1.0 - ta, copy, mr: if copy >= 13 { g::<usize>(src, "MR", "0").min(2) } else { 0 }, mv: copy >= 16 && e("MASKV", "0") == "1", lt: copy >= 16 && e("LATEST", "0") == "1" };
        let tag = e("TAG", &format!("v16_e{l2}_bs{bs}"));
        let (tern, cell, out) = (e("TERN", "0") == "1", e("TERN_SCOPE", "level") == "cell", format!("{}/rayhash_{tag}.json", e("EXPORTS", &format!("{home}/exports/gf17_continuum"))));
        Cfg { ck_every: g(src, "CKPT_EVERY", "0"), ckpt: e("CKPT", &format!("{}_ckpt", out.trim_end_matches(".json"))), ev_every: g(src, "EVAL_EVERY", "200"), lv: geo.lv(), geo, ver, copy, te, ctx: g(src, "CTX", "128"), bs, steps: g(src, "STEPS", "10000"), lr: g(src, "LR", "5e-3"), lr_end: g(src, "LR_END", "5e-4"), out, tag, multi: e("MULTI", "").split(',').filter(|c| !c.is_empty()).map(String::from).collect(), salted: e("SALTED", "1") == "1", rth: f("RTH", "0.1"), rw: f("RW", "0.05"), wd: f("WD", "1e-3"), tern: if !tern { 0 } else if cell { 1 } else { 2 }, tern_th: f("TERN_TH", "0.5"), qlev: g(src, "QLEV", "3"), sgrid: e("SGRID", "0") == "1", slo: f("SLO", "-6.5"), shi: f("SHI", "2.0"), seed: g(src, "SEED", "0"), arch: e("ARCH", "vulkan"), pages: e("PAGES", ""), pages_v16: e("PAGES_V16", ""), pages_v18: e("PAGES_V18", ""), pages_v19b: e("PAGES_V19B", ""), syn: e("SYNSETS", &format!("{home}/data/{corpus}/synsets.txt")), wcnt: e("WORDCOUNTS", &std::path::Path::new(&e("SYNSETS", &format!("{home}/data/{corpus}/synsets.txt"))).with_file_name("wordcounts.txt").to_string_lossy()), thf: e("THROTTLE_FILE", &format!("{home}/throttle.txt")), gen_t: g(src, "GEN_T", "0.7"), gen_k: g(src, "GEN_K", "12"), step_log: e("STEP_LOG", ""), dump: e("DUMP_F32", ""), corpus, home }
    }
    pub fn npos(&self) -> usize { self.bs * self.ctx }
    pub fn lr_at(&self, it: usize) -> f32 { (self.lr + (self.lr_end - self.lr) * (it as f64 / self.steps.max(1) as f64).min(1.0)) as f32 }
    pub fn tl(&self) -> usize { self.geo.base - 1 }
}
