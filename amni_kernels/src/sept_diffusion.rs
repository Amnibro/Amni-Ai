use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
const P: u8 = 17;
#[inline(always)]
fn gf17_dist(a: u8, b: u8) -> u8 {
    let d = (a as i32 - b as i32).unsigned_abs() as u8;
    d.min(P - d)
}
#[inline(always)]
fn _gf17_add(a: u8, b: u8) -> u8 { (a + b) % P }
#[inline(always)]
fn _gf17_sub(a: u8, b: u8) -> u8 { ((a as i32 - b as i32).rem_euclid(17)) as u8 }
#[pyfunction]
fn gf17_quantize<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f32>) -> (Bound<'py, PyArray1<u8>>, f32, f32) {
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
fn gf17_dequantize<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, u8>, vmin: f32, vmax: f32) -> Bound<'py, PyArray1<f32>> {
    let slice = data.as_slice().unwrap();
    let range = (vmax - vmin).max(1e-8);
    let scale = range / 16.0;
    let out: Vec<f32> = slice.par_iter()
        .map(|&v| vmin + (v as f32) * scale)
        .collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_gradient_field<'py>(
    py: Python<'py>,
    quantized: PyReadonlyArray1<'py, u8>,
    channels: usize, h: usize, w: usize,
) -> Bound<'py, PyArray1<u8>> {
    let q = quantized.as_slice().unwrap();
    let hw = h * w;
    let out: Vec<u8> = (0..hw).into_par_iter().map(|idx| {
        let y = idx / w;
        let x = idx % w;
        let mut grad_sum: u16 = 0;
        for c in 0..channels {
            let base = c * hw;
            let center = q[base + idx];
            let left = if x > 0 { q[base + y * w + x - 1] } else { center };
            let right = if x < w - 1 { q[base + y * w + x + 1] } else { center };
            let top = if y > 0 { q[base + (y - 1) * w + x] } else { center };
            let bot = if y < h - 1 { q[base + (y + 1) * w + x] } else { center };
            let gx = gf17_dist(right, left);
            let gy = gf17_dist(bot, top);
            grad_sum += gx as u16 + gy as u16;
        }
        (grad_sum / channels as u16).min(16) as u8
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_entropy_field<'py>(
    py: Python<'py>,
    quantized: PyReadonlyArray1<'py, u8>,
    channels: usize, h: usize, w: usize, window: usize,
) -> Bound<'py, PyArray1<u8>> {
    let q = quantized.as_slice().unwrap();
    let hw = h * w;
    let pad = window / 2;
    let out: Vec<u8> = (0..hw).into_par_iter().map(|idx| {
        let y = idx / w;
        let x = idx % w;
        let mut total_unique: u32 = 0;
        for c in 0..channels {
            let base = c * hw;
            let mut seen = [false; 17];
            let mut count = 0u32;
            let y_start = y.saturating_sub(pad);
            let y_end = (y + pad + 1).min(h);
            let x_start = x.saturating_sub(pad);
            let x_end = (x + pad + 1).min(w);
            for yy in y_start..y_end {
                for xx in x_start..x_end {
                    let v = q[base + yy * w + xx] as usize;
                    if !seen[v] { seen[v] = true; count += 1; }
                }
            }
            total_unique += count;
        }
        let avg_unique = total_unique / channels as u32;
        (avg_unique.min(16)) as u8
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_problem_mask<'py>(
    py: Python<'py>,
    gradient: PyReadonlyArray1<'py, u8>,
    entropy: PyReadonlyArray1<'py, u8>,
    _h: usize, _w: usize,
    grad_thresh: u8, entropy_thresh: u8,
) -> Bound<'py, PyArray1<u8>> {
    let g = gradient.as_slice().unwrap();
    let e = entropy.as_slice().unwrap();
    let out: Vec<u8> = g.par_iter().zip(e.par_iter())
        .map(|(&gv, &ev)| if gv < grad_thresh && ev >= entropy_thresh { 16 } else { 0 })
        .collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_apply_mask_smooth<'py>(
    py: Python<'py>,
    noise_pred: PyReadonlyArray1<'py, f32>,
    mask: PyReadonlyArray1<'py, u8>,
    channels: usize, h: usize, w: usize,
    strength: f32,
) -> Bound<'py, PyArray1<f32>> {
    let np_data = noise_pred.as_slice().unwrap();
    let m = mask.as_slice().unwrap();
    let hw = h * w;
    let out: Vec<f32> = (0..channels).into_par_iter().flat_map(|c| {
        let base = c * hw;
        (0..hw).map(move |idx| {
            let mask_val = m[idx] as f32 / 16.0;
            let blend = mask_val * strength;
            if blend < 0.001 { return np_data[base + idx]; }
            let y = idx / w;
            let x = idx % w;
            let center = np_data[base + idx];
            let mut sum = 0.0f32;
            let mut count = 0.0f32;
            let y_start = y.saturating_sub(1);
            let y_end = (y + 2).min(h);
            let x_start = x.saturating_sub(1);
            let x_end = (x + 2).min(w);
            for yy in y_start..y_end {
                for xx in x_start..x_end {
                    sum += np_data[base + yy * w + xx];
                    count += 1.0;
                }
            }
            let smoothed = sum / count;
            center * (1.0 - blend) + smoothed * blend
        }).collect::<Vec<f32>>()
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_convergence_delta(
    current: PyReadonlyArray1<u8>,
    previous: PyReadonlyArray1<u8>,
) -> u32 {
    let c = current.as_slice().unwrap();
    let p = previous.as_slice().unwrap();
    c.par_iter().zip(p.par_iter())
        .map(|(&a, &b)| gf17_dist(a, b) as u32)
        .sum()
}
#[pyfunction]
fn gf17_adaptive_cfg(
    quantized: PyReadonlyArray1<u8>,
    channels: usize, h: usize, w: usize,
    base_guidance: f32, progress: f32,
) -> f32 {
    let q = quantized.as_slice().unwrap();
    let hw = h * w;
    let total_var: u64 = (0..hw).into_par_iter().map(|idx| {
        let y = idx / w;
        let x = idx % w;
        let mut var_sum: u64 = 0;
        for c in 0..channels {
            let base = c * hw;
            let center = q[base + idx];
            if x > 0 { var_sum += gf17_dist(center, q[base + y * w + x - 1]) as u64; }
            if x < w - 1 { var_sum += gf17_dist(center, q[base + y * w + x + 1]) as u64; }
            if y > 0 { var_sum += gf17_dist(center, q[base + (y - 1) * w + x]) as u64; }
            if y < h - 1 { var_sum += gf17_dist(center, q[base + (y + 1) * w + x]) as u64; }
        }
        var_sum
    }).sum();
    let avg_var = total_var as f32 / (hw * channels * 4) as f32;
    let stability = 1.0 - (avg_var / 8.0).min(1.0);
    let phase_factor = if progress < 0.2 { 1.0 }
        else if progress < 0.5 { 1.0 - (progress - 0.2) * stability * 0.6 }
        else { (0.4 + 0.2 * (1.0 - stability)).max(0.3) };
    (base_guidance * phase_factor).max(1.5)
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gf17_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_dequantize, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_gradient_field, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_entropy_field, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_problem_mask, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_apply_mask_smooth, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_convergence_delta, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_adaptive_cfg, m)?)?;
    Ok(())
}
