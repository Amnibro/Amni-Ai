use numpy::PyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;
fn normalize_rows(m: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * d];
    out.par_chunks_mut(d).enumerate().for_each(|(i, row)| {
        let src = &m[i * d..(i + 1) * d];
        let norm = src.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        row.iter_mut().zip(src.iter()).for_each(|(o, s)| *o = s / norm);
    });
    out
}
fn cosine_sims(a: &[f32], b_rows: &[f32], n: usize, d: usize) -> Vec<f32> {
    let a_norm = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    (0..n).into_par_iter().map(|i| {
        let row = &b_rows[i * d..(i + 1) * d];
        let dot: f32 = a.iter().zip(row.iter()).map(|(x, y)| x * y).sum();
        dot / a_norm
    }).collect()
}
fn gaussian_resonance(
    fact_vecs: &[f32], n_facts: usize,
    anchor_vecs: &[f32], n_anchors: usize,
    anchor_weights: Option<&[f32]>,
    dim: usize, sigma: f32,
) -> Vec<f32> {
    let sigma_sq_inv2 = 1.0 / (2.0 * sigma * sigma);
    let fn_norm = normalize_rows(fact_vecs, n_facts, dim);
    let an_norm = normalize_rows(anchor_vecs, n_anchors, dim);
    let w_sum: f32 = anchor_weights.map(|w| w.iter().sum::<f32>()).unwrap_or(n_anchors as f32).max(1e-8);
    (0..n_facts).into_par_iter().map(|fi| {
        let frow = &fn_norm[fi * dim..(fi + 1) * dim];
        let mut score = 0.0f32;
        for ai in 0..n_anchors {
            let arow = &an_norm[ai * dim..(ai + 1) * dim];
            let sim: f32 = frow.iter().zip(arow.iter()).map(|(a, b)| a * b).sum();
            let dist_sq = (1.0 - sim).max(0.0);
            let gauss = (-dist_sq * sigma_sq_inv2).exp();
            let w = anchor_weights.map(|ws| ws[ai]).unwrap_or(1.0) / w_sum;
            score += gauss * w;
        }
        score
    }).collect()
}
#[pyfunction]
fn holographic_resonance_score<'py>(
    py: Python<'py>,
    fact_vecs_flat: Vec<f32>,
    n_facts: usize,
    anchor_vecs_flat: Vec<f32>,
    n_anchors: usize,
    anchor_weights: Option<Vec<f32>>,
    dim: usize,
    sigma: f32,
) -> Bound<'py, PyArray1<f32>> {
    let wref = anchor_weights.as_deref();
    let scores = gaussian_resonance(&fact_vecs_flat, n_facts, &anchor_vecs_flat, n_anchors, wref, dim, sigma);
    PyArray1::from_vec(py, scores)
}
#[pyfunction]
fn cosine_similarity_batch<'py>(
    py: Python<'py>,
    query: Vec<f32>,
    matrix_flat: Vec<f32>,
    n_rows: usize,
    dim: usize,
) -> Bound<'py, PyArray1<f32>> {
    let m_norm = normalize_rows(&matrix_flat, n_rows, dim);
    let sims = cosine_sims(&query, &m_norm, n_rows, dim);
    PyArray1::from_vec(py, sims)
}
#[pyfunction]
fn resonance_rank(
    _py: Python<'_>,
    query_vec: Vec<f32>,
    fact_vecs_flat: Vec<f32>,
    n_facts: usize,
    dim: usize,
    tiers: Vec<u8>,
    anchor_vecs_flat: Option<Vec<f32>>,
    n_anchors: Option<usize>,
    anchor_weights: Option<Vec<f32>>,
    sigma: f32,
    top_k: usize,
    cull_threshold: f32,
) -> (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (a_flat, na) = match anchor_vecs_flat {
        Some(av) => (av, n_anchors.unwrap_or(1)),
        None => (query_vec.clone(), 1),
    };
    let wref = anchor_weights.as_deref();
    let res_scores = gaussian_resonance(&fact_vecs_flat, n_facts, &a_flat, na, wref, dim, sigma);
    let fn_norm = normalize_rows(&fact_vecs_flat, n_facts, dim);
    let cos_scores = cosine_sims(&query_vec, &fn_norm, n_facts, dim);
    let tier_w = |t: u8| -> f32 { match t { 1 => 1.0, 2 => 0.6, 3 => 1.0, _ => 1.0 } };
    let mut combined: Vec<(usize, f32, f32, f32)> = (0..n_facts).map(|i| {
        let tw = tier_w(if i < tiers.len() { tiers[i] } else { 1 });
        let c = res_scores[i] * 0.4 + cos_scores[i] * 0.4 + tw * 0.2;
        (i, c, res_scores[i], cos_scores[i])
    }).collect();
    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    combined.truncate(top_k);
    let mut indices = Vec::new();
    let mut r_out = Vec::new();
    let mut cos_out = Vec::new();
    let mut comb_out = Vec::new();
    for (idx, c, r, cs) in combined {
        if c < cull_threshold { continue; }
        indices.push(idx);
        r_out.push(r);
        cos_out.push(cs);
        comb_out.push(c);
    }
    (indices, r_out, cos_out, comb_out)
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(holographic_resonance_score, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(resonance_rank, m)?)?;
    Ok(())
}
