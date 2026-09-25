use rayon::prelude::*;
use regex::bytes::Regex;
use std::{collections::HashMap, fs, io};
pub const PAT: &str = r" ?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+";
pub struct Bpe { pub re: Regex, pub rank: HashMap<(u32, u32), (u32, u32)>, pub voc: Vec<Vec<u8>>, pub merges: Vec<[u32; 2]> }
impl Bpe {
    pub fn new(pat: &str, merges: Vec<[u32; 2]>) -> io::Result<Bpe> {
        let mut voc: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        merges.iter().for_each(|&[a, b]| { let v = [voc[a as usize].clone(), voc[b as usize].clone()].concat(); voc.push(v); });
        Ok(Bpe { re: Regex::new(pat).map_err(io::Error::other)?, rank: merges.iter().enumerate().map(|(i, &[a, b])| ((a, b), (i as u32, 256 + i as u32))).collect(), voc, merges })
    }
    pub fn open(p: &str) -> io::Result<Bpe> {
        let j: serde_json::Value = serde_json::from_str(&fs::read_to_string(p)?).map_err(io::Error::other)?;
        Bpe::new(j["pattern"].as_str().unwrap_or(PAT), j["merges"].as_array().ok_or_else(|| io::Error::other("bpe: merges missing"))?.iter().map(|m| [m[0].as_u64().unwrap_or(0) as u32, m[1].as_u64().unwrap_or(0) as u32]).collect())
    }
    pub fn v(&self) -> usize { self.voc.len() }
    pub fn lens(&self) -> Vec<u32> { self.voc.iter().map(|v| v.len() as u32).collect() }
    pub fn word(&self, w: &[u8]) -> Vec<u32> {
        let mut v: Vec<u32> = w.iter().map(|&b| b as u32).collect();
        while let Some((_, i, id)) = v.windows(2).enumerate().filter_map(|(i, p)| self.rank.get(&(p[0], p[1])).map(|&(r, id)| (r, i, id))).min() { v[i] = id; v.remove(i + 1); }
        v
    }
    pub fn enc_with(&self, x: &[u8], cache: &mut HashMap<Vec<u8>, Vec<u32>>) -> Vec<u32> {
        let (mut o, mut e) = (Vec::with_capacity(x.len() / 3), 0usize);
        for m in self.re.find_iter(x) {
            o.extend(x[e..m.start()].iter().map(|&b| b as u32));
            let w = m.as_bytes();
            o.extend(cache.get(w).cloned().unwrap_or_else(|| { let r = self.word(w); (cache.len() < 1 << 20).then(|| cache.insert(w.to_vec(), r.clone())); r }));
            e = m.end();
        }
        o.extend(x[e..].iter().map(|&b| b as u32));
        o
    }
    pub fn enc(&self, x: &[u8]) -> Vec<u32> { self.enc_with(x, &mut HashMap::new()) }
    pub fn dec(&self, ids: &[u32]) -> Vec<u8> { ids.iter().flat_map(|&i| self.voc[i as usize].iter().copied()).collect() }
    pub fn enc_file(&self, src: &[u8], chunk: usize) -> Vec<u16> {
        let mut cuts: Vec<usize> = std::iter::once(0).chain((1..src.len().div_ceil(chunk)).map(|k| { let p = k * chunk; p + src[p..].windows(2).position(|w| w == b"\n\n").map(|q| q + 2).unwrap_or(src.len() - p) })).chain(std::iter::once(src.len())).map(|p| p.min(src.len())).collect();
        cuts.sort_unstable();
        cuts.dedup();
        cuts.par_windows(2).map_init(HashMap::new, |c, w| self.enc_with(&src[w[0]..w[1]], c).into_iter().map(|v| v as u16).collect::<Vec<u16>>()).flatten_iter().collect()
    }
}
