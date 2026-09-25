use adam_ray::{pack::{lexicon, Geo}, v19::{Lx19, W19}};
use rayon::prelude::*;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let fl = |k: &str| a.get(3).is_none_or(|s| s.contains(k));
    let g = Geo { copy: 16, mr: 2, mv: true, lt: fl("lt"), gn: fl("gn"), g2: fl("g2"), ..Geo::eng(1 << 20, true) };
    let (lex, lx, all) = (lexicon(&format!("{}/synsets.txt", a[1]), Some(&format!("{}/wordcounts.txt", a[1]))), Lx19::new(), std::fs::read(&a[2]).unwrap());
    let segs: Vec<&[u8]> = all.split(|&b| b == 0).collect();
    let out: Vec<String> = segs.par_iter().map(|txt| {
        let mut w = W19::new(g.copy);
        txt.iter().for_each(|&b| w.step(&g, &lex, &lx, b));
        let m = &w.m;
        let mut ix: Vec<usize> = (0..64).filter(|&i| m.stamp[i] > 0).collect();
        ix.sort_by_key(|&i| m.stamp[i]);
        let fs: Vec<String> = ix.iter().map(|&i| format!("[{},{},\"{}\"]", m.ent[i], m.rel[i], m.val[i][..m.vlen[i].clamp(0, 24) as usize].iter().map(|&c| if c == b'"' || c == b'\\' || c < 32 || c > 126 { format!("\\u{:04x}", c) } else { (c as char).to_string() }).collect::<String>())).collect();
        format!("{{\"f\":[{}],\"q\":[{},{}],\"c\":[{}]}}", fs.join(","), m.qent, m.qrel, m.cand.iter().map(|&c| if c >= 0 { format!("\"{}\"", String::from_utf8_lossy(&m.val[c as usize][..m.vlen[c as usize].clamp(0, 24) as usize]).replace(['"', '\\'], "'")) } else { "null".into() }).collect::<Vec<_>>().join(","))
    }).collect();
    println!("{}", out.join("\n"));
}
