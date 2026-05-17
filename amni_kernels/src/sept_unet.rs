use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use memmap2::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::path::Path;
const P: u8 = 17;
const SBOX_17: [u8; 17] = [0, 1, 9, 6, 13, 7, 3, 5, 15, 2, 12, 14, 10, 4, 11, 8, 16];
const MDS_17: [[u8; 4]; 4] = [
    [13, 7, 3, 5],
    [7, 3, 5, 15],
    [3, 5, 15, 2],
    [5, 15, 2, 12],
];
#[inline(always)]
fn gf17_mul(a: u8, b: u8) -> u8 { ((a as u16 * b as u16) % 17) as u8 }
#[inline(always)]
fn gf17_add(a: u8, b: u8) -> u8 { (a + b) % P }
#[inline(always)]
fn fnv1a_hash(data: &[u8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &b in data { h ^= b as u32; h = h.wrapping_mul(16777619); }
    h
}
#[pyfunction]
fn gf17_spatial_fingerprint<'py>(
    py: Python<'py>,
    quantized: PyReadonlyArray1<'py, u8>,
    channels: usize, h: usize, w: usize,
) -> Bound<'py, PyArray1<u32>> {
    let q = quantized.as_slice().unwrap();
    let hw = h * w;
    let ch = channels.min(4);
    let out: Vec<u32> = (0..hw).into_par_iter().map(|idx| {
        let y = idx / w;
        let x = idx % w;
        let mut buf = [0u8; 36];
        let mut bi = 0;
        for c in 0..ch {
            let base = c * hw;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let ny = (y as i32 + dy).clamp(0, (h - 1) as i32) as usize;
                    let nx = (x as i32 + dx).clamp(0, (w - 1) as i32) as usize;
                    buf[bi] = q[base + ny * w + nx];
                    bi += 1;
                }
            }
        }
        fnv1a_hash(&buf[..bi])
    }).collect();
    PyArray1::from_vec(py, out)
}
fn atlas_key_index(fprint: u32, tbin: u8, table_size: u32) -> usize {
    let kb = [
        (fprint & 0xFF) as u8, ((fprint >> 8) & 0xFF) as u8,
        ((fprint >> 16) & 0xFF) as u8, ((fprint >> 24) & 0xFF) as u8, tbin,
    ];
    (fnv1a_hash(&kb) % table_size) as usize
}
#[pyfunction]
fn gf17_atlas_lookup<'py>(
    py: Python<'py>,
    fingerprints: PyReadonlyArray1<'py, u32>,
    timestep_bin: u8,
    atlas_path: &str,
    table_size: u32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<u8>>)> {
    let fp = fingerprints.as_slice().unwrap();
    let n = fp.len();
    let ts = table_size as usize;
    let file_size = (ts + ts * 16) as u64;
    let path = Path::new(atlas_path);
    if !path.exists() || std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) < file_size {
        let residuals = vec![0.0f32; n * 4];
        let hits = vec![0u8; n];
        return Ok((PyArray1::from_vec(py, residuals), PyArray1::from_vec(py, hits)));
    }
    let file = OpenOptions::new().read(true).open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let valid_sec = &mmap[..ts];
    let noise_sec = &mmap[ts..];
    let results: Vec<([f32; 4], u8)> = fp.par_iter().map(|&fprint| {
        let idx = atlas_key_index(fprint, timestep_bin, table_size);
        if valid_sec[idx] != 0 {
            let off = idx * 16;
            let mut vals = [0.0f32; 4];
            for c in 0..4 {
                let o = off + c * 4;
                vals[c] = f32::from_le_bytes([noise_sec[o], noise_sec[o+1], noise_sec[o+2], noise_sec[o+3]]);
            }
            (vals, 1u8)
        } else {
            ([0.0f32; 4], 0u8)
        }
    }).collect();
    let mut residuals = Vec::with_capacity(n * 4);
    let mut hits = Vec::with_capacity(n);
    for (vals, hit) in results {
        residuals.extend_from_slice(&vals);
        hits.push(hit);
    }
    Ok((PyArray1::from_vec(py, residuals), PyArray1::from_vec(py, hits)))
}
#[pyfunction]
fn gf17_atlas_store(
    fingerprints: PyReadonlyArray1<u32>,
    noise_pred: PyReadonlyArray1<f32>,
    timestep_bin: u8,
    channels: usize,
    h: usize, w: usize,
    atlas_path: &str,
    table_size: u32,
) -> PyResult<u32> {
    let fp = fingerprints.as_slice().unwrap();
    let np_data = noise_pred.as_slice().unwrap();
    let hw = h * w;
    let ts = table_size as usize;
    let file_size = (ts + ts * 16) as u64;
    let path = Path::new(atlas_path);
    if !path.exists() || std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) < file_size {
        let file = OpenOptions::new().write(true).create(true).truncate(true).open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        file.set_len(file_size)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    }
    let file = OpenOptions::new().read(true).write(true).open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut mmap = unsafe { MmapMut::map_mut(&file) }
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let ch = channels.min(4);
    let mut stored = 0u32;
    for pix in 0..hw {
        let idx = atlas_key_index(fp[pix], timestep_bin, table_size);
        mmap[idx] = 1;
        let off = ts + idx * 16;
        for c in 0..ch {
            let val = np_data[c * hw + pix];
            let bytes = val.to_le_bytes();
            mmap[off + c * 4..off + c * 4 + 4].copy_from_slice(&bytes);
        }
        stored += 1;
    }
    mmap.flush().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(stored)
}
#[pyfunction]
fn gf17_shadow_mix<'py>(
    py: Python<'py>,
    quantized: PyReadonlyArray1<'py, u8>,
    shadow: PyReadonlyArray1<'py, u8>,
    channels: usize, h: usize, w: usize,
    rounds: u8,
) -> Bound<'py, PyArray1<u8>> {
    let q = quantized.as_slice().unwrap();
    let s = shadow.as_slice().unwrap();
    let hw = h * w;
    let ch = channels.min(4);
    let out: Vec<u8> = (0..hw).into_par_iter().flat_map(|pix| {
        let mut d = [0u8; 4];
        let mut sh = [0u8; 4];
        for c in 0..ch { d[c] = q[c * hw + pix]; sh[c] = s[c * hw + pix]; }
        for _ in 0..rounds {
            for c in 0..ch { d[c] = gf17_add(d[c], sh[c]); }
            for c in 0..ch { d[c] = SBOX_17[d[c] as usize]; }
            let prev = d;
            for i in 0..ch {
                let mut acc: u16 = 0;
                for j in 0..ch { acc += gf17_mul(MDS_17[i][j], prev[j]) as u16; }
                d[i] = (acc % 17) as u8;
            }
            for c in 0..ch { sh[c] = SBOX_17[gf17_add(sh[c], d[c]) as usize]; }
        }
        (0..ch).map(move |c| d[c]).collect::<Vec<u8>>()
    }).collect();
    PyArray1::from_vec(py, out)
}
#[pyfunction]
fn gf17_tier_classify<'py>(
    py: Python<'py>,
    gradient: PyReadonlyArray1<'py, u8>,
    entropy: PyReadonlyArray1<'py, u8>,
    atlas_hits: PyReadonlyArray1<'py, u8>,
    grad_thresh: u8,
    entropy_thresh: u8,
) -> Bound<'py, PyArray1<u8>> {
    let g = gradient.as_slice().unwrap();
    let e = entropy.as_slice().unwrap();
    let hits = atlas_hits.as_slice().unwrap();
    let out: Vec<u8> = g.par_iter().zip(e.par_iter()).zip(hits.par_iter())
        .map(|((&gv, &ev), &hit)| {
            if hit != 0 && gv < grad_thresh { 0 }
            else if gv < grad_thresh && ev < entropy_thresh { 1 }
            else { 2 }
        }).collect();
    PyArray1::from_vec(py, out)
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gf17_spatial_fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_atlas_lookup, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_atlas_store, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_shadow_mix, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_tier_classify, m)?)?;
    Ok(())
}
