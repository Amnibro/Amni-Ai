use adam_ray::pack::lexicon;
use adam_train::{cfg::Cfg, v19::{Lx19, W19}};
use std::collections::BTreeMap;
#[test]
fn v19_mstate() {
    let Ok(f) = std::env::var("ADAM_V19_MSTATE") else { return eprintln!("skip: set ADAM_V19_MSTATE=<\\0-separated kind\\x01text file>") };
    let c = Cfg::with(&[("VER", "19"), ("COPY", "13"), ("MR", "2"), ("LEVELS", "21"), ("HOPS", "4"), ("QT", "1"), ("LOG2_ENTRIES", "16")]);
    let (lex, lx, d) = (lexicon(&c.syn, Some(&c.wcnt)), Lx19::new(), std::fs::read(&f).unwrap());
    let (mut tab, mut ks): (BTreeMap<String, [usize; 7]>, Vec<String>) = (BTreeMap::new(), vec![]);
    for r in d.split(|&b| b == 0) {
        let i = r.iter().position(|&b| b == 1).unwrap();
        let mut w = W19::new(c.copy);
        r[i + 1..].iter().for_each(|&b| w.step(&c.geo, &lex, &lx, b));
        let (k, t) = (w.mstate(), tab.entry(String::from_utf8_lossy(&r[..i]).into_owned()).or_default());
        t[(k / 4 % 3) as usize] += 1;
        t[3] += (k / 12 % 2) as usize;
        t[4] += (k / 24 % 2) as usize;
        t[6] += (k / 48 == 2) as usize;
        t[5] += w.mtaps(&c.geo)[0].is_some() as usize;
        ks.push(k.to_string());
    }
    std::fs::write(format!("{f}.k"), ks.join("\n")).unwrap();
    println!("| kind | none | one | several | hop | qrel | on | qent other |\n|---|---|---|---|---|---|---|---|");
    tab.iter().for_each(|(k, t)| println!("| {k} | {} | {} | {} | {} | {} | {} | {} |", t[0], t[1], t[2], t[3], t[4], t[5], t[6]));
}
