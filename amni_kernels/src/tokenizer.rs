use pyo3::prelude::*;
use rayon::prelude::*;
use ahash::AHashSet;
static SUFFIXES: &[&str] = &["s", "es", "ed", "ing", "ly", "tion", "ment"];
static STOP_WORDS: &[&str] = &[
    "the","a","an","is","are","was","were","be","been","being",
    "what","who","whom","whose","which","where","when","why","how",
    "do","does","did","can","could","will","would","shall","should",
    "may","might","must","have","has","had","having",
    "in","on","at","to","for","of","by","with","from","about",
    "and","or","but","not","no","nor","so","yet","also","very",
    "it","its","this","that","these","those","they","them","their",
    "there","here","then","than","such","each","some","many",
];
fn build_stop_set() -> AHashSet<String> {
    STOP_WORDS.iter().map(|w| w.to_string()).collect()
}
fn tokenize_inner(text: &str, stop: &AHashSet<String>) -> Vec<String> {
    let mut tokens = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    let len = bytes.len();
    while i < len {
        if bytes[i].is_ascii_alphabetic() {
            let start = i;
            i += 1;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') { i += 1; }
            if i - start > 2 {
                let w: String = bytes[start..i].iter().map(|b| b.to_ascii_lowercase() as char).collect();
                if !stop.contains(&w) { tokens.push(w); }
            }
        } else {
            i += 1;
        }
    }
    tokens
}
fn stem_lookup(word: &str, vocab: &AHashSet<String>) -> Option<String> {
    if vocab.contains(word) { return Some(word.to_string()); }
    for sfx in SUFFIXES {
        if word.len() > sfx.len() + 2 && word.ends_with(sfx) {
            let base = &word[..word.len() - sfx.len()];
            if vocab.contains(base) { return Some(base.to_string()); }
        }
    }
    None
}
#[pyfunction]
fn tokenize(text: &str) -> Vec<String> {
    let stop = build_stop_set();
    tokenize_inner(text, &stop)
}
#[pyfunction]
fn tokenize_batch(texts: Vec<String>) -> Vec<Vec<String>> {
    let stop = build_stop_set();
    texts.par_iter().map(|t| tokenize_inner(t, &stop)).collect()
}
#[pyfunction]
fn tokenize_and_stem(text: &str, vocab_words: Vec<String>) -> Vec<String> {
    let stop = build_stop_set();
    let vocab: AHashSet<String> = vocab_words.into_iter().collect();
    let tokens = tokenize_inner(text, &stop);
    tokens.iter().filter_map(|w| stem_lookup(w, &vocab)).collect()
}
#[pyfunction]
fn embed_fact_tokens(
    text: &str,
    vocab_words: Vec<String>,
    dim: usize,
    seed: u32,
) -> Option<Vec<f32>> {
    let stop = build_stop_set();
    let vocab: AHashSet<String> = vocab_words.into_iter().collect();
    let tokens = tokenize_inner(text, &stop);
    let resolved: Vec<String> = tokens.iter().filter_map(|w| stem_lookup(w, &vocab)).collect();
    if resolved.is_empty() { return None; }
    let vecs: Vec<Vec<f32>> = resolved.par_iter()
        .map(|w| crate::nonce::hash_vector_inner(w, dim, seed))
        .collect();
    let n = vecs.len() as f32;
    let mut avg = vec![0.0f32; dim];
    for v in &vecs { for (i, val) in v.iter().enumerate() { avg[i] += val; } }
    avg.iter_mut().for_each(|x| *x /= n);
    let norm = avg.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    avg.iter_mut().for_each(|x| *x /= norm);
    Some(avg)
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_batch, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_and_stem, m)?)?;
    m.add_function(wrap_pyfunction!(embed_fact_tokens, m)?)?;
    Ok(())
}
