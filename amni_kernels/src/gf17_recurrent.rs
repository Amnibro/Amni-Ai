use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
const P: u8 = 17;
const SBOX_17: [u8; 17] = [0, 1, 9, 6, 13, 7, 3, 5, 15, 2, 12, 14, 10, 4, 11, 8, 16];
const MDS_4: [[u8; 4]; 4] = [
    [13, 7, 3, 5],
    [7, 3, 5, 15],
    [3, 5, 15, 2],
    [5, 15, 2, 12],
];
const LOG17: [u8; 17] = [255, 0, 14, 1, 12, 5, 15, 11, 10, 2, 3, 7, 13, 4, 9, 6, 8];
const EXP17: [u8; 16] = [1, 3, 9, 10, 13, 5, 15, 11, 16, 14, 8, 7, 4, 12, 2, 6];
#[inline(always)]
fn gf17_add(a: u8, b: u8) -> u8 { (a + b) % P }
#[inline(always)]
fn gf17_mul(a: u8, b: u8) -> u8 { ((a as u16 * b as u16) % 17) as u8 }
#[inline(always)]
fn gf17_sub(a: u8, b: u8) -> u8 { ((a as i32 - b as i32).rem_euclid(17)) as u8 }
#[inline(always)]
fn gf17_log_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 { return 0; }
    EXP17[((LOG17[a as usize] as u16 + LOG17[b as usize] as u16) % 16) as usize]
}
#[pyfunction]
fn gf17_state_init<'py>(py: Python<'py>, dim: usize) -> Bound<'py, PyArray1<u8>> {
    PyArray1::from_vec(py, vec![0u8; dim * dim])
}
#[pyfunction]
fn gf17_state_update<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, u8>,
    x_vec: PyReadonlyArray1<'py, u8>,
    v_vec: PyReadonlyArray1<'py, u8>,
    alpha: u8,
    beta: u8,
    dim: usize,
) -> Bound<'py, PyArray1<u8>> {
    let s = state.as_slice().unwrap();
    let x = x_vec.as_slice().unwrap();
    let v = v_vec.as_slice().unwrap();
    let d = dim.min(x.len()).min(v.len());
    let out: Vec<u8> = (0..d).into_par_iter().flat_map(|i| {
        (0..d).map(move |j| {
            let s_ij = s[i * d + j];
            let outer = gf17_mul(x[i], v[j]);
            gf17_add(gf17_mul(alpha, s_ij), gf17_mul(beta, outer))
        }).collect::<Vec<u8>>()
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_state_query<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, u8>,
    query: PyReadonlyArray1<'py, u8>,
    dim: usize,
) -> Bound<'py, PyArray1<u8>> {
    let s = state.as_slice().unwrap();
    let q = query.as_slice().unwrap();
    let d = dim.min(q.len());
    let out: Vec<u8> = (0..d).into_par_iter().map(|i| {
        let mut acc: u16 = 0;
        for j in 0..d {
            acc += gf17_mul(s[i * d + j], q[j]) as u16;
        }
        (acc % 17) as u8
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_state_diffuse<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, u8>,
    dim: usize,
    rounds: u8,
) -> Bound<'py, PyArray1<u8>> {
    let s = state.as_slice().unwrap();
    let d = dim;
    let mut buf: Vec<u8> = s.to_vec();
    for _ in 0..rounds {
        let prev = buf.clone();
        buf = (0..d).into_par_iter().flat_map(|i| {
            let row: Vec<u8> = (0..d).map(|j| {
                let idx = i * d + j;
                let val = SBOX_17[prev[idx] as usize];
                let ci = i % 4;
                let mut acc: u16 = 0;
                for k in 0..4usize {
                    let jj = (j + k).min(d - 1);
                    acc += gf17_mul(MDS_4[ci][k], prev[i * d + jj]) as u16;
                }
                gf17_add(val, (acc % 17) as u8)
            }).collect();
            row
        }).collect();
    }
    PyArray1::from_vec(py, buf)
}
#[pyfunction]
fn gf17_state_compress(
    state: PyReadonlyArray1<u8>,
    dim: usize,
) -> Vec<u8> {
    let s = state.as_slice().unwrap();
    let d = dim;
    let row_sums: Vec<u8> = (0..d).into_par_iter().map(|i| {
        let mut acc: u16 = 0;
        for j in 0..d {
            acc += s[i * d + j] as u16;
        }
        (acc % 17) as u8
    }).collect();
    row_sums
}
#[pyfunction]
fn gf17_state_decompress<'py>(
    py: Python<'py>,
    compressed: PyReadonlyArray1<'py, u8>,
    dim: usize,
) -> Bound<'py, PyArray1<u8>> {
    let c = compressed.as_slice().unwrap();
    let d = dim.min(c.len());
    let out: Vec<u8> = (0..d).into_par_iter().flat_map(|i| {
        (0..d).map(move |j| {
            gf17_mul(c[i], c[j])
        }).collect::<Vec<u8>>()
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_outer_product<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, u8>,
    b: PyReadonlyArray1<'py, u8>,
) -> Bound<'py, PyArray1<u8>> {
    let av = a.as_slice().unwrap();
    let bv = b.as_slice().unwrap();
    let out: Vec<u8> = av.par_iter().flat_map(|&ai| {
        bv.iter().map(move |&bj| gf17_mul(ai, bj)).collect::<Vec<u8>>()
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_vec_matmul<'py>(
    py: Python<'py>,
    mat: PyReadonlyArray1<'py, u8>,
    vec: PyReadonlyArray1<'py, u8>,
    rows: usize,
    cols: usize,
) -> Bound<'py, PyArray1<u8>> {
    let m = mat.as_slice().unwrap();
    let v = vec.as_slice().unwrap();
    let r = rows.min(m.len() / cols.max(1));
    let c = cols.min(v.len());
    let out: Vec<u8> = (0..r).into_par_iter().map(|i| {
        let mut acc: u16 = 0;
        for j in 0..c {
            acc += gf17_mul(m[i * c + j], v[j]) as u16;
        }
        (acc % 17) as u8
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_decay_state<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, u8>,
    decay: u8,
) -> Bound<'py, PyArray1<u8>> {
    let s = state.as_slice().unwrap();
    let out: Vec<u8> = s.par_iter()
        .map(|&v| gf17_mul(v, decay))
        .collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_gate_combine<'py>(
    py: Python<'py>,
    forget_gate: PyReadonlyArray1<'py, u8>,
    input_gate: PyReadonlyArray1<'py, u8>,
    old_state: PyReadonlyArray1<'py, u8>,
    new_input: PyReadonlyArray1<'py, u8>,
) -> Bound<'py, PyArray1<u8>> {
    let fg = forget_gate.as_slice().unwrap();
    let ig = input_gate.as_slice().unwrap();
    let os = old_state.as_slice().unwrap();
    let ni = new_input.as_slice().unwrap();
    let n = fg.len().min(ig.len()).min(os.len()).min(ni.len());
    let out: Vec<u8> = (0..n).into_par_iter()
        .map(|i| gf17_add(gf17_mul(fg[i], os[i]), gf17_mul(ig[i], ni[i])))
        .collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_quantize_f32_to_state<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
) -> (Bound<'py, PyArray1<u8>>, f32, f32) {
    let slice = data.as_slice().unwrap();
    let mut vmin = f32::INFINITY;
    let mut vmax = f32::NEG_INFINITY;
    for &v in slice.iter() {
        if v < vmin { vmin = v; }
        if v > vmax { vmax = v; }
    }
    let range = (vmax - vmin).max(1e-8);
    let scale = 16.0 / range;
    let out: Vec<u8> = slice.par_iter()
        .map(|&v| ((v - vmin) * scale).round().clamp(0.0, 16.0) as u8)
        .collect();
    (PyArray1::from_vec(py, out), vmin, vmax)
}
#[pyfunction]
fn gf17_dequantize_state_to_f32<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, u8>,
    vmin: f32,
    vmax: f32,
) -> Bound<'py, PyArray1<f32>> {
    let slice = data.as_slice().unwrap();
    let range = (vmax - vmin).max(1e-8);
    let scale = range / 16.0;
    let out: Vec<f32> = slice.par_iter()
        .map(|&v| vmin + (v as f32) * scale)
        .collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_log_state_update<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, u8>,
    x_vec: PyReadonlyArray1<'py, u8>,
    v_vec: PyReadonlyArray1<'py, u8>,
    alpha: u8,
    beta: u8,
    dim: usize,
) -> Bound<'py, PyArray1<u8>> {
    let s = state.as_slice().unwrap();
    let x = x_vec.as_slice().unwrap();
    let v = v_vec.as_slice().unwrap();
    let d = dim.min(x.len()).min(v.len());
    let out: Vec<u8> = (0..d).into_par_iter().flat_map(|i| {
        (0..d).map(move |j| {
            let s_ij = s[i * d + j];
            let outer = gf17_log_mul(x[i], v[j]);
            gf17_add(gf17_log_mul(alpha, s_ij), gf17_log_mul(beta, outer))
        }).collect::<Vec<u8>>()
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_dual_texture_step<'py>(
    py: Python<'py>,
    tmu_tex: PyReadonlyArray1<'py, u8>,
    alu_buf: PyReadonlyArray1<'py, u8>,
    x_vec: PyReadonlyArray1<'py, u8>,
    v_vec: PyReadonlyArray1<'py, u8>,
    alpha: u8,
    beta: u8,
    conf_decay: u8,
    tok_hash: u8,
    err_threshold: u8,
    dim: usize,
) -> (Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>, u32, u8, f32, u32) {
    let tmu = tmu_tex.as_slice().unwrap();
    let alu = alu_buf.as_slice().unwrap();
    let x = x_vec.as_slice().unwrap();
    let v = v_vec.as_slice().unwrap();
    let d = dim.min(x.len()).min(v.len());
    let d2 = d * d;
    let mut tmu_out = vec![0u8; d2 * 4];
    let mut alu_out = vec![0u8; d2 * 4];
    let tmu_r: Vec<u8> = (0..d).into_par_iter().flat_map(|i| {
        (0..d).map(move |j| {
            let idx = i * d + j;
            let s_old = tmu[idx * 4];
            let outer = gf17_log_mul(x[i], v[j]);
            gf17_add(gf17_log_mul(alpha, s_old), gf17_log_mul(beta, outer))
        }).collect::<Vec<u8>>()
    }).collect();
    let alu_r: Vec<u8> = (0..d).into_par_iter().flat_map(|i| {
        (0..d).map(move |j| {
            let idx = i * d + j;
            let s_old = alu[idx * 4];
            let outer = gf17_mul(x[i], v[j]);
            gf17_add(s_old, gf17_mul(beta, outer)) % P
        }).collect::<Vec<u8>>()
    }).collect();
    let mut err_energy: u32 = 0;
    let mut max_err: u8 = 0;
    let mut conf_sum: u32 = 0;
    let mut err_flags = vec![false; d2];
    for idx in 0..d2 {
        let i = idx / d;
        let j = idx % d;
        let t_r = tmu_r[idx];
        let a_r = alu_r[idx];
        let g_err = gf17_sub(t_r, a_r);
        let b_old = tmu[idx * 4 + 2];
        let b_new = if g_err > err_threshold {
            err_flags[idx] = true;
            gf17_log_mul(b_old, conf_decay)
        } else {
            gf17_add(b_old, 1) % P
        };
        let a_old = tmu[idx * 4 + 3];
        let a_new = ((a_old as u16 + tok_hash as u16 + (i as u16) * 3 + (j as u16) * 7) % P as u16) as u8;
        tmu_out[idx * 4] = t_r;
        tmu_out[idx * 4 + 1] = g_err;
        tmu_out[idx * 4 + 2] = b_new;
        tmu_out[idx * 4 + 3] = a_new;
        alu_out[idx * 4] = a_r;
        alu_out[idx * 4 + 1] = g_err;
        alu_out[idx * 4 + 2] = b_new;
        alu_out[idx * 4 + 3] = a_new;
        err_energy += g_err as u32;
        if g_err > max_err { max_err = g_err; }
        conf_sum += b_new as u32;
    }
    let err_cell_ct: usize = err_flags.iter().filter(|&&f| f).count();
    let mut cells_corrected: u32 = 0;
    if err_cell_ct > 0 && err_cell_ct < d2 / 2 {
        for idx in 0..d2 {
            if !err_flags[idx] { continue; }
            let i = idx / d;
            let j = idx % d;
            let mut acc: u16 = 0;
            let mut cnt: u8 = 0;
            for di in 0..3u8 {
                for dj in 0..3u8 {
                    let ni = (i + di as usize).wrapping_sub(1);
                    let nj = (j + dj as usize).wrapping_sub(1);
                    if ni < d && nj < d && !(di == 1 && dj == 1) {
                        let nidx = ni * d + nj;
                        if !err_flags[nidx] {
                            acc += tmu_out[nidx * 4] as u16;
                            cnt += 1;
                        }
                    }
                }
            }
            if cnt > 0 {
                let corrected = (acc / cnt as u16) % 17;
                tmu_out[idx * 4] = corrected as u8;
                tmu_out[idx * 4 + 1] = 0;
                cells_corrected += 1;
            }
        }
    }
    let conf_mean = conf_sum as f32 / d2.max(1) as f32;
    (PyArray1::from_vec(py, tmu_out), PyArray1::from_vec(py, alu_out), err_energy, max_err, conf_mean, cells_corrected)
}
#[pyfunction]
fn gf17_error_detect<'py>(
    py: Python<'py>,
    tmu_tex: PyReadonlyArray1<'py, u8>,
    alu_buf: PyReadonlyArray1<'py, u8>,
    dim: usize,
) -> (Bound<'py, PyArray1<u8>>, u32, u8) {
    let tmu = tmu_tex.as_slice().unwrap();
    let alu = alu_buf.as_slice().unwrap();
    let d2 = dim * dim;
    let mut mask = vec![0u8; d2];
    let mut energy: u32 = 0;
    let mut peak: u8 = 0;
    for idx in 0..d2 {
        let diff = gf17_sub(alu[idx * 4], tmu[idx * 4]);
        mask[idx] = diff;
        energy += diff as u32;
        if diff > peak { peak = diff; }
    }
    (PyArray1::from_vec(py, mask), energy, peak)
}
#[pyfunction]
fn gf17_error_correct<'py>(
    py: Python<'py>,
    state_rgba: PyReadonlyArray1<'py, u8>,
    error_mask: PyReadonlyArray1<'py, u8>,
    threshold: u8,
    dim: usize,
) -> Bound<'py, PyArray1<u8>> {
    let s = state_rgba.as_slice().unwrap();
    let m = error_mask.as_slice().unwrap();
    let d = dim;
    let d2 = d * d;
    let mut out = s.to_vec();
    for idx in 0..d2 {
        if m[idx] <= threshold { continue; }
        let i = idx / d;
        let j = idx % d;
        let mut acc: u16 = 0;
        let mut cnt: u8 = 0;
        for di in 0..3u8 {
            for dj in 0..3u8 {
                let ni = (i + di as usize).wrapping_sub(1);
                let nj = (j + dj as usize).wrapping_sub(1);
                if ni < d && nj < d && !(di == 1 && dj == 1) {
                    let nidx = ni * d + nj;
                    if m[nidx] <= threshold {
                        acc += s[nidx * 4] as u16;
                        cnt += 1;
                    }
                }
            }
        }
        if cnt > 0 {
            out[idx * 4] = ((acc / cnt as u16) % 17) as u8;
            out[idx * 4 + 1] = 0;
        }
    }
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_spatial_hash(
    state_rgba: PyReadonlyArray1<u8>,
    row: usize,
    col: usize,
    dim: usize,
) -> u64 {
    let s = state_rgba.as_slice().unwrap();
    let d = dim;
    let mut h: u64 = 0xcbf29ce484222325;
    for di in 0..3u8 {
        for dj in 0..3u8 {
            let ri = (row + di as usize).wrapping_sub(1);
            let ci = (col + dj as usize).wrapping_sub(1);
            if ri < d && ci < d {
                let idx = ri * d + ci;
                for ch in 0..4usize {
                    h ^= s[idx * 4 + ch] as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
        }
    }
    h
}
#[pyfunction]
fn gf17_page_error_density(
    tmu_tex: PyReadonlyArray1<u8>,
    alu_buf: PyReadonlyArray1<u8>,
    dim: usize,
) -> (f32, f32, u32) {
    let tmu = tmu_tex.as_slice().unwrap();
    let alu = alu_buf.as_slice().unwrap();
    let d2 = dim * dim;
    let (mut total_err, mut err_cells): (u32, u32) = (0, 0);
    for idx in 0..d2 {
        let diff = gf17_sub(tmu[idx * 4], alu[idx * 4]);
        total_err += diff as u32;
        if diff > 0 { err_cells += 1; }
    }
    let density = err_cells as f32 / d2.max(1) as f32;
    let avg_err = total_err as f32 / d2.max(1) as f32;
    (density, avg_err, err_cells)
}
#[pyfunction]
fn gf17_page_correction_pass<'py>(
    py: Python<'py>,
    tmu_tex: PyReadonlyArray1<'py, u8>,
    alu_buf: PyReadonlyArray1<'py, u8>,
    conf_threshold: u8,
    dim: usize,
) -> (Bound<'py, PyArray1<u8>>, u32) {
    let tmu = tmu_tex.as_slice().unwrap();
    let alu = alu_buf.as_slice().unwrap();
    let d = dim;
    let d2 = d * d;
    let mut out = tmu.to_vec();
    let mut corrected: u32 = 0;
    let err_map: Vec<u8> = (0..d2).map(|idx| gf17_sub(tmu[idx * 4], alu[idx * 4])).collect();
    let conf_map: Vec<u8> = (0..d2).map(|idx| tmu[idx * 4 + 2]).collect();
    for idx in 0..d2 {
        if conf_map[idx] >= conf_threshold || err_map[idx] == 0 { continue; }
        let i = idx / d;
        let j = idx % d;
        let mut acc: u16 = 0;
        let mut cnt: u8 = 0;
        for di in 0..3u8 {
            for dj in 0..3u8 {
                let ni = (i + di as usize).wrapping_sub(1);
                let nj = (j + dj as usize).wrapping_sub(1);
                if ni < d && nj < d && !(di == 1 && dj == 1) {
                    let nidx = ni * d + nj;
                    if conf_map[nidx] >= conf_threshold {
                        acc += tmu[nidx * 4] as u16;
                        cnt += 1;
                    }
                }
            }
        }
        if cnt > 0 {
            out[idx * 4] = ((acc / cnt as u16) % 17) as u8;
            out[idx * 4 + 1] = 0;
            out[idx * 4 + 2] = gf17_add(conf_map[idx], 1) % P;
            corrected += 1;
        }
    }
    (PyArray1::from_vec(py, out), corrected)
}
#[pyfunction]
fn gf17_batch_spatial_hash(
    state_rgba: PyReadonlyArray1<u8>,
    dim: usize,
) -> Vec<u64> {
    let s = state_rgba.as_slice().unwrap();
    let d = dim;
    (0..d).into_par_iter().flat_map(|i| {
        (0..d).map(move |j| {
            let mut h: u64 = 0xcbf29ce484222325;
            for di in 0..3u8 {
                for dj in 0..3u8 {
                    let ri = (i + di as usize).wrapping_sub(1);
                    let ci = (j + dj as usize).wrapping_sub(1);
                    if ri < d && ci < d {
                        let idx = ri * d + ci;
                        for ch in 0..4usize {
                            h ^= s[idx * 4 + ch] as u64;
                            h = h.wrapping_mul(0x100000001b3);
                        }
                    }
                }
            }
            h
        }).collect::<Vec<u64>>()
    }).collect()
}
#[pyfunction]
fn gf17_mastery_encode<'py>(
    py: Python<'py>,
    code_hash: PyReadonlyArray1<'py, u8>,
    fitness: f32,
    domain_id: u8,
    dim: usize,
) -> Bound<'py, PyArray1<u8>> {
    let h = code_hash.as_slice().unwrap();
    let d = dim;
    let fit_q = ((fitness.clamp(0.0, 1.0) * 16.0).round() as u8).min(16);
    let out: Vec<u8> = (0..d).into_par_iter().flat_map(|i| {
        (0..d).map(move |j| {
            let hi = h[i % h.len()];
            let hj = h[(j + d / 2) % h.len()];
            let base = gf17_log_mul(hi % P, hj % P);
            let mixed = gf17_add(base, gf17_mul(fit_q, domain_id % P));
            gf17_add(mixed, SBOX_17[(i.wrapping_add(j)) % 17])
        }).collect::<Vec<u8>>()
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_mastery_similarity(
    pattern_a: PyReadonlyArray1<u8>,
    pattern_b: PyReadonlyArray1<u8>,
    dim: usize,
) -> f32 {
    let a = pattern_a.as_slice().unwrap();
    let b = pattern_b.as_slice().unwrap();
    let d2 = dim * dim;
    let n = d2.min(a.len()).min(b.len());
    let (mut dot, mut na, mut nb): (u32, u32, u32) = (0, 0, 0);
    for i in 0..n {
        dot += gf17_mul(a[i], b[i]) as u32;
        na += gf17_mul(a[i], a[i]) as u32;
        nb += gf17_mul(b[i], b[i]) as u32;
    }
    let denom = ((na as f64).sqrt() * (nb as f64).sqrt()).max(1e-8);
    (dot as f64 / denom) as f32
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gf17_state_init, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_state_update, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_state_query, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_state_diffuse, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_state_compress, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_state_decompress, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_outer_product, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_vec_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_decay_state, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_gate_combine, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_quantize_f32_to_state, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_dequantize_state_to_f32, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_log_state_update, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_dual_texture_step, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_error_detect, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_error_correct, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_spatial_hash, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_page_error_density, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_page_correction_pass, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_batch_spatial_hash, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_mastery_encode, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_mastery_similarity, m)?)?;
    Ok(())
}
