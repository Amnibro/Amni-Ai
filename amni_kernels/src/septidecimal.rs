use pyo3::prelude::*;
use rayon::prelude::*;
use sha2::{Sha256, Digest};
const P: u32 = 17;
const P_POW: [u32; 4] = [4913, 289, 17, 1]; // 17^3, 17^2, 17^1, 17^0
pub fn sept_digits_inner(word: &str, seed: u32) -> [u8; 8] {
    let input = format!("{}:{}", seed, word.to_lowercase().trim());
    let hash = Sha256::digest(input.as_bytes());
    let mut digits = [0u8; 8];
    for i in 0..8 {
        let off = i * 4;
        let chunk = u32::from_le_bytes([hash[off], hash[off+1], hash[off+2], hash[off+3]]);
        digits[i] = (chunk % P) as u8;
    }
    digits
}
fn pack_row_col_inner(digits: &[u8; 8]) -> (u32, u32) {
    let row = digits[0] as u32 * P_POW[0] + digits[1] as u32 * P_POW[1]
            + digits[2] as u32 * P_POW[2] + digits[3] as u32 * P_POW[3];
    let col = digits[4] as u32 * P_POW[0] + digits[5] as u32 * P_POW[1]
            + digits[6] as u32 * P_POW[2] + digits[7] as u32 * P_POW[3];
    (row, col)
}
#[pyfunction]
fn sept_word_to_digits(word: &str, seed: u32) -> Vec<u8> {
    sept_digits_inner(word, seed).to_vec()
}
#[pyfunction]
fn sept_pack_row_col(digits: Vec<u8>) -> (u32, u32) {
    let mut d = [0u8; 8];
    d.copy_from_slice(&digits[..8]);
    pack_row_col_inner(&d)
}
#[pyfunction]
fn batch_sept_digits<'py>(
    py: Python<'py>,
    words: Vec<String>,
    seed: u32,
) -> Bound<'py, numpy::PyArray1<u8>> {
    let results: Vec<[u8; 8]> = words.par_iter()
        .map(|w| sept_digits_inner(w, seed))
        .collect();
    let flat: Vec<u8> = results.into_iter().flat_map(|d| d).collect();
    numpy::PyArray1::from_vec(py, flat)
}
#[pyfunction]
fn batch_sept_row_col<'py>(
    py: Python<'py>,
    words: Vec<String>,
    seed: u32,
) -> Bound<'py, numpy::PyArray1<u32>> {
    let results: Vec<(u32, u32)> = words.par_iter()
        .map(|w| {
            let d = sept_digits_inner(w, seed);
            pack_row_col_inner(&d)
        }).collect();
    let flat: Vec<u32> = results.into_iter().flat_map(|rc| [rc.0, rc.1]).collect();
    numpy::PyArray1::from_vec(py, flat)
}
#[pyfunction]
fn sept_field_distance<'py>(
    py: Python<'py>,
    query_digits: Vec<u8>,
    all_digits_flat: Vec<u8>,
    n_words: usize,
) -> Bound<'py, numpy::PyArray1<u32>> {
    let q = &query_digits[..8];
    let dists: Vec<u32> = (0..n_words).into_par_iter().map(|i| {
        let off = i * 8;
        let mut d = 0u32;
        for j in 0..8 {
            let a = q[j] as i32;
            let b = all_digits_flat[off + j] as i32;
            let diff = (a - b).unsigned_abs();
            d += diff.min(P - diff);
        }
        d
    }).collect();
    numpy::PyArray1::from_vec(py, dists)
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sept_word_to_digits, m)?)?;
    m.add_function(wrap_pyfunction!(sept_pack_row_col, m)?)?;
    m.add_function(wrap_pyfunction!(batch_sept_digits, m)?)?;
    m.add_function(wrap_pyfunction!(batch_sept_row_col, m)?)?;
    m.add_function(wrap_pyfunction!(sept_field_distance, m)?)?;
    Ok(())
}
