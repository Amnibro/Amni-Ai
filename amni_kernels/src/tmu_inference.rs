use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
const P: u8 = 17;
const P16: u16 = 17;
const LOG17: [u8; 17] = [255, 0, 14, 1, 12, 5, 15, 11, 10, 2, 3, 7, 13, 4, 9, 6, 8];
const EXP17: [u8; 16] = [1, 3, 9, 10, 13, 5, 15, 11, 16, 14, 8, 7, 4, 12, 2, 6];
const SBOX_17: [u8; 17] = [0, 1, 9, 6, 13, 7, 3, 5, 15, 2, 12, 14, 10, 4, 11, 8, 16];
#[inline(always)]
fn gf17_add(a: u8, b: u8) -> u8 { (a + b) % P }
#[inline(always)]
fn gf17_sub(a: u8, b: u8) -> u8 { ((a as i32 - b as i32).rem_euclid(17)) as u8 }
#[inline(always)]
fn gf17_log_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 { return 0; }
    EXP17[((LOG17[a as usize] as u16 + LOG17[b as usize] as u16) % 16) as usize]
}
#[inline(always)]
fn gf17_log_div(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 { return 0; }
    let la = LOG17[a as usize] as i32;
    let lb = LOG17[b as usize] as i32;
    EXP17[((la - lb).rem_euclid(16)) as usize]
}
#[pyfunction]
fn gf17_quantize_weights<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    out_dim: usize,
    in_dim: usize,
) -> (Bound<'py, PyArray1<u8>>, f32, f32, f32) {
    let slice = data.as_slice().unwrap();
    let total = out_dim * in_dim;
    let n = slice.len().min(total);
    let mut vmin = f32::INFINITY;
    let mut vmax = f32::NEG_INFINITY;
    for &v in &slice[..n] {
        if v < vmin { vmin = v; }
        if v > vmax { vmax = v; }
    }
    let range = (vmax - vmin).max(1e-12);
    let scale = 16.0 / range;
    let bias = vmin;
    let out: Vec<u8> = slice[..n].par_iter()
        .map(|&v| ((v - bias) * scale).round().clamp(0.0, 16.0) as u8)
        .collect();
    let mut padded = out;
    padded.resize(total, 0);
    (PyArray1::from_vec(py, padded), scale, bias, range)
}
#[pyfunction]
fn gf17_quantize_weights_f16<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, u16>,
    out_dim: usize,
    in_dim: usize,
) -> (Bound<'py, PyArray1<u8>>, f32, f32, f32) {
    let slice = data.as_slice().unwrap();
    let total = out_dim * in_dim;
    let n = slice.len().min(total);
    let f32_vals: Vec<f32> = slice[..n].par_iter()
        .map(|&bits| half_to_f32(bits))
        .collect();
    let mut vmin = f32::INFINITY;
    let mut vmax = f32::NEG_INFINITY;
    for &v in &f32_vals {
        if v < vmin { vmin = v; }
        if v > vmax { vmax = v; }
    }
    let range = (vmax - vmin).max(1e-12);
    let scale = 16.0 / range;
    let bias = vmin;
    let out: Vec<u8> = f32_vals.par_iter()
        .map(|&v| ((v - bias) * scale).round().clamp(0.0, 16.0) as u8)
        .collect();
    let mut padded = out;
    padded.resize(total, 0);
    (PyArray1::from_vec(py, padded), scale, bias, range)
}
#[inline(always)]
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    if exp == 0 {
        let f = (sign << 31) | 0;
        return f32::from_bits(f | (frac << 13));
    }
    if exp == 31 {
        return if frac != 0 { f32::NAN } else if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY };
    }
    let new_exp = exp + 112;
    f32::from_bits((sign << 31) | (new_exp << 23) | (frac << 13))
}
#[pyfunction]
fn gf17_log_matmul<'py>(
    py: Python<'py>,
    weight_page: PyReadonlyArray1<'py, u8>,
    input_vec: PyReadonlyArray1<'py, u8>,
    out_dim: usize,
    in_dim: usize,
) -> Bound<'py, PyArray1<u8>> {
    let w = weight_page.as_slice().unwrap();
    let x = input_vec.as_slice().unwrap();
    let od = out_dim.min(w.len() / in_dim.max(1));
    let id = in_dim.min(x.len());
    let out: Vec<u8> = (0..od).into_par_iter().map(|i| {
        let mut acc: u16 = 0;
        let row_base = i * id;
        for j in 0..id {
            acc += gf17_log_mul(w[row_base + j], x[j]) as u16;
        }
        (acc % P16) as u8
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_batch_log_matmul<'py>(
    py: Python<'py>,
    weight_page: PyReadonlyArray1<'py, u8>,
    input_batch: PyReadonlyArray1<'py, u8>,
    batch_size: usize,
    out_dim: usize,
    in_dim: usize,
) -> Bound<'py, PyArray1<u8>> {
    let w = weight_page.as_slice().unwrap();
    let x = input_batch.as_slice().unwrap();
    let od = out_dim.min(w.len() / in_dim.max(1));
    let id = in_dim.min(x.len() / batch_size.max(1));
    let bs = batch_size.min(x.len() / id.max(1));
    let out: Vec<u8> = (0..bs).into_par_iter().flat_map(|b| {
        let x_off = b * id;
        (0..od).map(move |i| {
            let mut acc: u16 = 0;
            let row_base = i * id;
            for j in 0..id {
                acc += gf17_log_mul(w[row_base + j], x[x_off + j]) as u16;
            }
            (acc % P16) as u8
        }).collect::<Vec<u8>>()
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_weight_page_checksum(
    page_data: PyReadonlyArray1<u8>,
) -> u64 {
    let s = page_data.as_slice().unwrap();
    let mut h: u64 = 0xcbf29ce484222325;
    for &byte in s {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
#[pyfunction]
fn gf17_dequantize_page<'py>(
    py: Python<'py>,
    page_data: PyReadonlyArray1<'py, u8>,
    scale: f32,
    bias: f32,
) -> Bound<'py, PyArray1<f32>> {
    let slice = page_data.as_slice().unwrap();
    let inv_scale = 1.0 / scale.max(1e-12);
    let out: Vec<f32> = slice.par_iter()
        .map(|&v| bias + (v as f32) * inv_scale)
        .collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_attention_score<'py>(
    py: Python<'py>,
    q_page: PyReadonlyArray1<'py, u8>,
    k_page: PyReadonlyArray1<'py, u8>,
    seq_len: usize,
    head_dim: usize,
) -> Bound<'py, PyArray1<u8>> {
    let q = q_page.as_slice().unwrap();
    let k = k_page.as_slice().unwrap();
    let sl = seq_len.min(q.len() / head_dim.max(1)).min(k.len() / head_dim.max(1));
    let hd = head_dim;
    let out: Vec<u8> = (0..sl).into_par_iter().flat_map(|qi| {
        let q_off = qi * hd;
        (0..sl).map(move |ki| {
            let k_off = ki * hd;
            let mut acc: u16 = 0;
            for d in 0..hd {
                acc += gf17_log_mul(q[q_off + d], k[k_off + d]) as u16;
            }
            (acc % P16) as u8
        }).collect::<Vec<u8>>()
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_expert_route<'py>(
    py: Python<'py>,
    gate_scores: PyReadonlyArray1<'py, u8>,
    n_experts: usize,
    n_active: usize,
) -> (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u8>>) {
    let gs = gate_scores.as_slice().unwrap();
    let ne = n_experts.min(gs.len());
    let na = n_active.min(ne);
    let mut indexed: Vec<(usize, u8)> = gs[..ne].iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.cmp(&a.1));
    let top_indices: Vec<u32> = indexed[..na].iter().map(|&(i, _)| i as u32).collect();
    let total: u16 = indexed[..na].iter().map(|&(_, v)| v as u16).sum();
    let top_weights: Vec<u8> = indexed[..na].iter()
        .map(|&(_, v)| if total > 0 { ((v as u16 * 16) / total.max(1)).min(16) as u8 } else { (16 / na as u8).min(16) })
        .collect();
    (PyArray1::from_vec(py, top_indices), PyArray1::from_vec(py, top_weights))
}
#[pyfunction]
fn gf17_accumulate_learning<'py>(
    py: Python<'py>,
    learning_page: PyReadonlyArray1<'py, u8>,
    new_data: PyReadonlyArray1<'py, u8>,
    decay_alpha: u8,
    input_beta: u8,
) -> Bound<'py, PyArray1<u8>> {
    let old = learning_page.as_slice().unwrap();
    let new_d = new_data.as_slice().unwrap();
    let n = old.len().min(new_d.len());
    let out: Vec<u8> = (0..n).into_par_iter()
        .map(|i| gf17_add(gf17_log_mul(decay_alpha, old[i]), gf17_log_mul(input_beta, new_d[i])))
        .collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_pack_rgba_page<'py>(
    py: Python<'py>,
    state_data: PyReadonlyArray1<'py, u8>,
    error_data: PyReadonlyArray1<'py, u8>,
    confidence_data: PyReadonlyArray1<'py, u8>,
    context_data: PyReadonlyArray1<'py, u8>,
    dim: usize,
) -> Bound<'py, PyArray1<u8>> {
    let s = state_data.as_slice().unwrap();
    let e = error_data.as_slice().unwrap();
    let c = confidence_data.as_slice().unwrap();
    let x = context_data.as_slice().unwrap();
    let d2 = dim * dim;
    let n = d2.min(s.len()).min(e.len()).min(c.len()).min(x.len());
    let out: Vec<u8> = (0..n).into_par_iter().flat_map(|i| {
        vec![s[i], e[i], c[i], x[i]]
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_unpack_rgba_page<'py>(
    py: Python<'py>,
    rgba_data: PyReadonlyArray1<'py, u8>,
    dim: usize,
) -> (Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>) {
    let d = rgba_data.as_slice().unwrap();
    let d2 = dim * dim;
    let n = d2.min(d.len() / 4);
    let mut s = vec![0u8; n];
    let mut e = vec![0u8; n];
    let mut c = vec![0u8; n];
    let mut x = vec![0u8; n];
    for i in 0..n {
        s[i] = d[i * 4];
        e[i] = d[i * 4 + 1];
        c[i] = d[i * 4 + 2];
        x[i] = d[i * 4 + 3];
    }
    (PyArray1::from_vec(py, s), PyArray1::from_vec(py, e), PyArray1::from_vec(py, c), PyArray1::from_vec(py, x))
}
#[pyfunction]
fn gf17_softmax_approx<'py>(
    py: Python<'py>,
    logits: PyReadonlyArray1<'py, u8>,
    temperature: u8,
) -> Bound<'py, PyArray1<u8>> {
    let l = logits.as_slice().unwrap();
    let n = l.len();
    if n == 0 { return PyArray1::from_vec(py, vec![]); }
    let max_v = *l.iter().max().unwrap_or(&0);
    let shifted: Vec<u8> = l.iter().map(|&v| gf17_sub(v, max_v)).collect();
    let temp = temperature.max(1).min(16);
    let scaled: Vec<u8> = shifted.iter().map(|&v| gf17_log_div(v, temp)).collect();
    let sum: u16 = scaled.iter().map(|&v| v as u16).sum();
    let out: Vec<u8> = if sum > 0 {
        scaled.iter().map(|&v| ((v as u16 * 16) / sum.max(1)).min(16) as u8).collect()
    } else {
        vec![(16 / n as u8).max(1); n]
    };
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_residual_add<'py>(
    py: Python<'py>,
    base: PyReadonlyArray1<'py, u8>,
    residual: PyReadonlyArray1<'py, u8>,
) -> Bound<'py, PyArray1<u8>> {
    let b = base.as_slice().unwrap();
    let r = residual.as_slice().unwrap();
    let n = b.len().min(r.len());
    let out: Vec<u8> = (0..n).into_par_iter()
        .map(|i| gf17_add(b[i], r[i]))
        .collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_rms_norm<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, u8>,
    dim: usize,
) -> Bound<'py, PyArray1<u8>> {
    let d = data.as_slice().unwrap();
    let n = dim.min(d.len());
    let sum_sq: u32 = d[..n].iter().map(|&v| (v as u32) * (v as u32)).sum();
    let rms = ((sum_sq as f64 / n.max(1) as f64).sqrt().max(1.0)) as u16;
    let out: Vec<u8> = d[..n].par_iter()
        .map(|&v| ((v as u16 * 16) / rms.max(1)).min(16) as u8)
        .collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_silu_lut<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, u8>,
) -> Bound<'py, PyArray1<u8>> {
    const SILU_LUT: [u8; 17] = [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16];
    let d = data.as_slice().unwrap();
    let out: Vec<u8> = d.par_iter()
        .map(|&v| SILU_LUT[v.min(16) as usize])
        .collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_gated_ffn<'py>(
    py: Python<'py>,
    gate_weights: PyReadonlyArray1<'py, u8>,
    up_weights: PyReadonlyArray1<'py, u8>,
    down_weights: PyReadonlyArray1<'py, u8>,
    input_vec: PyReadonlyArray1<'py, u8>,
    hidden_dim: usize,
    out_dim: usize,
    in_dim: usize,
) -> Bound<'py, PyArray1<u8>> {
    let gw = gate_weights.as_slice().unwrap();
    let uw = up_weights.as_slice().unwrap();
    let dw = down_weights.as_slice().unwrap();
    let x = input_vec.as_slice().unwrap();
    let id = in_dim.min(x.len());
    let hd = hidden_dim;
    let od = out_dim;
    const SILU_LUT: [u8; 17] = [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16];
    let gate_out: Vec<u8> = (0..hd).into_par_iter().map(|i| {
        let mut acc: u16 = 0;
        let base = i * id;
        for j in 0..id {
            if base + j < gw.len() {
                acc += gf17_log_mul(gw[base + j], x[j]) as u16;
            }
        }
        SILU_LUT[((acc % P16) as u8).min(16) as usize]
    }).collect();
    let up_out: Vec<u8> = (0..hd).into_par_iter().map(|i| {
        let mut acc: u16 = 0;
        let base = i * id;
        for j in 0..id {
            if base + j < uw.len() {
                acc += gf17_log_mul(uw[base + j], x[j]) as u16;
            }
        }
        (acc % P16) as u8
    }).collect();
    let hidden: Vec<u8> = (0..hd).into_par_iter().map(|i| {
        gf17_log_mul(gate_out[i], up_out[i])
    }).collect();
    let out: Vec<u8> = (0..od).into_par_iter().map(|i| {
        let mut acc: u16 = 0;
        let base = i * hd;
        for j in 0..hd {
            if base + j < dw.len() {
                acc += gf17_log_mul(dw[base + j], hidden[j]) as u16;
            }
        }
        (acc % P16) as u8
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_kv_cache_append<'py>(
    py: Python<'py>,
    cache: PyReadonlyArray1<'py, u8>,
    new_kv: PyReadonlyArray1<'py, u8>,
    cache_len: usize,
    head_dim: usize,
    max_len: usize,
) -> (Bound<'py, PyArray1<u8>>, usize) {
    let c = cache.as_slice().unwrap();
    let nk = new_kv.as_slice().unwrap();
    let hd = head_dim;
    let new_tokens = nk.len() / hd.max(1);
    let new_cl = (cache_len + new_tokens).min(max_len);
    let mut out = vec![0u8; max_len * hd];
    if new_cl > new_tokens {
        let keep = new_cl - new_tokens;
        let start = cache_len.saturating_sub(keep);
        let copy_len = keep * hd;
        let src_off = start * hd;
        let src_end = (src_off + copy_len).min(c.len());
        let actual = src_end - src_off;
        out[..actual].copy_from_slice(&c[src_off..src_end]);
        let dst_off = actual;
        let nk_copy = (new_tokens * hd).min(nk.len());
        out[dst_off..dst_off + nk_copy].copy_from_slice(&nk[..nk_copy]);
    } else {
        let nk_copy = (new_cl * hd).min(nk.len());
        let nk_start = nk.len().saturating_sub(nk_copy);
        out[..nk_copy].copy_from_slice(&nk[nk_start..nk_start + nk_copy]);
    }
    (PyArray1::from_vec(py, out), new_cl)
}
#[pyfunction]
fn gf17_inference_step<'py>(
    py: Python<'py>,
    q_weights: PyReadonlyArray1<'py, u8>,
    k_weights: PyReadonlyArray1<'py, u8>,
    v_weights: PyReadonlyArray1<'py, u8>,
    o_weights: PyReadonlyArray1<'py, u8>,
    input_vec: PyReadonlyArray1<'py, u8>,
    kv_cache_k: PyReadonlyArray1<'py, u8>,
    kv_cache_v: PyReadonlyArray1<'py, u8>,
    cache_len: usize,
    hidden_dim: usize,
    head_dim: usize,
    n_heads: usize,
) -> (Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>) {
    let qw = q_weights.as_slice().unwrap();
    let kw = k_weights.as_slice().unwrap();
    let vw = v_weights.as_slice().unwrap();
    let ow = o_weights.as_slice().unwrap();
    let x = input_vec.as_slice().unwrap();
    let ck = kv_cache_k.as_slice().unwrap();
    let cv = kv_cache_v.as_slice().unwrap();
    let hd = head_dim;
    let nh = n_heads;
    let hdim = hidden_dim.min(x.len());
    let total_hd = nh * hd;
    let q_proj: Vec<u8> = (0..total_hd).into_par_iter().map(|i| {
        let mut acc: u16 = 0;
        let base = i * hdim;
        for j in 0..hdim {
            if base + j < qw.len() { acc += gf17_log_mul(qw[base + j], x[j]) as u16; }
        }
        (acc % P16) as u8
    }).collect();
    let k_proj: Vec<u8> = (0..total_hd).into_par_iter().map(|i| {
        let mut acc: u16 = 0;
        let base = i * hdim;
        for j in 0..hdim {
            if base + j < kw.len() { acc += gf17_log_mul(kw[base + j], x[j]) as u16; }
        }
        (acc % P16) as u8
    }).collect();
    let v_proj: Vec<u8> = (0..total_hd).into_par_iter().map(|i| {
        let mut acc: u16 = 0;
        let base = i * hdim;
        for j in 0..hdim {
            if base + j < vw.len() { acc += gf17_log_mul(vw[base + j], x[j]) as u16; }
        }
        (acc % P16) as u8
    }).collect();
    let seq_pos = cache_len;
    let attn_len = seq_pos + 1;
    let attn_out: Vec<u8> = (0..nh).into_par_iter().flat_map(|h| {
        let h_off = h * hd;
        let q_head = &q_proj[h_off..h_off + hd];
        let mut scores: Vec<u16> = Vec::with_capacity(attn_len);
        for t in 0..seq_pos {
            let k_off = t * total_hd + h_off;
            let mut dot: u16 = 0;
            for d in 0..hd {
                if k_off + d < ck.len() {
                    dot += gf17_log_mul(q_head[d], ck[k_off + d]) as u16;
                }
            }
            scores.push(dot % P16);
        }
        let mut self_dot: u16 = 0;
        for d in 0..hd { self_dot += gf17_log_mul(q_head[d], k_proj[h_off + d]) as u16; }
        scores.push(self_dot % P16);
        let sum: u16 = scores.iter().sum::<u16>().max(1);
        let weights: Vec<u8> = scores.iter().map(|&s| ((s * 16) / sum).min(16) as u8).collect();
        let mut head_out = vec![0u16; hd];
        for t in 0..seq_pos {
            let v_off = t * total_hd + h_off;
            for d in 0..hd {
                if v_off + d < cv.len() {
                    head_out[d] += gf17_log_mul(weights[t], cv[v_off + d]) as u16;
                }
            }
        }
        for d in 0..hd {
            head_out[d] += gf17_log_mul(weights[seq_pos], v_proj[h_off + d]) as u16;
        }
        head_out.into_iter().map(|v| (v % P16) as u8).collect::<Vec<u8>>()
    }).collect();
    let out: Vec<u8> = (0..hdim).into_par_iter().map(|i| {
        let mut acc: u16 = 0;
        let base = i * total_hd;
        for j in 0..total_hd {
            if base + j < ow.len() && j < attn_out.len() {
                acc += gf17_log_mul(ow[base + j], attn_out[j]) as u16;
            }
        }
        (acc % P16) as u8
    }).collect();
    let new_k = PyArray1::from_vec(py, k_proj);
    let new_v = PyArray1::from_vec(py, v_proj);
    (PyArray1::from_vec(py, out), new_k, new_v)
}
#[pyfunction]
fn gf17_quantize_f32_to_state<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    dim: usize,
) -> Bound<'py, PyArray1<u8>> {
    let slice = data.as_slice().unwrap();
    let n = dim.min(slice.len());
    let mut vmin = f32::INFINITY;
    let mut vmax = f32::NEG_INFINITY;
    for &v in &slice[..n] {
        if v < vmin { vmin = v; }
        if v > vmax { vmax = v; }
    }
    let range = (vmax - vmin).max(1e-12);
    let scale = 16.0 / range;
    let bias = vmin;
    let out: Vec<u8> = slice[..n].par_iter()
        .map(|&v| ((v - bias) * scale).round().clamp(0.0, 16.0) as u8)
        .collect();
    PyArray1::from_vec(py, out)
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gf17_quantize_weights, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_quantize_weights_f16, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_log_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_batch_log_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_weight_page_checksum, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_dequantize_page, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_attention_score, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_expert_route, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_accumulate_learning, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_pack_rgba_page, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_unpack_rgba_page, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_softmax_approx, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_residual_add, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_rms_norm, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_silu_lut, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_gated_ffn, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_kv_cache_append, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_inference_step, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_quantize_f32_to_state, m)?)?;
    Ok(())
}
