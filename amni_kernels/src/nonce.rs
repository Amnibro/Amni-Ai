use numpy::PyArray1;
use pyo3::prelude::*;
use sha2::{Sha256, Digest};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
const N_POS: usize = 12;
const N_DOMAINS: usize = 25;
pub fn hash_vector_inner(word: &str, dim: usize, seed: u32) -> Vec<f32> {
    let input = format!("{}:{}", seed, word.to_lowercase().trim());
    let hash = Sha256::digest(input.as_bytes());
    let rng_seed = u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]);
    let mut rng = ChaCha8Rng::seed_from_u64(rng_seed as u64);
    let mut v: Vec<f32> = (0..dim).map(|_| { let s: f64 = StandardNormal.sample(&mut rng); s as f32 }).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}
pub fn encode_nonce_inner(base: &mut [f32], pos_id: usize, domain_id: usize, freq: f32) {
    let dim = base.len();
    let pos_start = dim - N_POS - N_DOMAINS - 1;
    let dom_start = pos_start + N_POS;
    let freq_idx = dom_start + N_DOMAINS;
    for i in pos_start..pos_start + N_POS { base[i] *= 0.5; }
    if pos_id < N_POS { base[pos_start + pos_id] += 0.3; }
    for i in dom_start..dom_start + N_DOMAINS { base[i] *= 0.5; }
    if domain_id < N_DOMAINS { base[dom_start + domain_id] += 0.3; }
    if freq_idx < dim { base[freq_idx] = (freq / 100.0).min(1.0) * 0.2; }
    let norm = base.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    base.iter_mut().for_each(|x| *x /= norm);
}
#[pyfunction]
fn word_to_hash_vector<'py>(py: Python<'py>, word: &str, dim: usize, seed: u32) -> Bound<'py, PyArray1<f32>> {
    let v = hash_vector_inner(word, dim, seed);
    PyArray1::from_vec(py, v)
}
#[pyfunction]
fn encode_structured_nonce<'py>(
    py: Python<'py>,
    base_vec: Vec<f32>,
    pos_id: usize,
    domain_id: usize,
    freq: f32,
) -> Bound<'py, PyArray1<f32>> {
    let mut v = base_vec;
    encode_nonce_inner(&mut v, pos_id, domain_id, freq);
    PyArray1::from_vec(py, v)
}
#[pyfunction]
fn batch_hash_vectors<'py>(
    py: Python<'py>,
    words: Vec<String>,
    dim: usize,
    seed: u32,
) -> Bound<'py, PyArray1<f32>> {
    use rayon::prelude::*;
    let vecs: Vec<Vec<f32>> = words.par_iter()
        .map(|w| hash_vector_inner(w, dim, seed))
        .collect();
    let flat: Vec<f32> = vecs.into_iter().flatten().collect();
    PyArray1::from_vec(py, flat)
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(word_to_hash_vector, m)?)?;
    m.add_function(wrap_pyfunction!(encode_structured_nonce, m)?)?;
    m.add_function(wrap_pyfunction!(batch_hash_vectors, m)?)?;
    Ok(())
}
