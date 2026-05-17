use pyo3::prelude::*;
use rayon::prelude::*;
fn xy2d_inner(n: i64, mut x: i64, mut y: i64) -> i64 {
    let mut d: i64 = 0;
    let mut s = n / 2;
    while s > 0 {
        let rx = if (x & s) > 0 { 1i64 } else { 0 };
        let ry = if (y & s) > 0 { 1i64 } else { 0 };
        d += s * s * ((3 * rx) ^ ry);
        if ry == 0 {
            if rx == 1 { x = s - 1 - x; y = s - 1 - y; }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}
#[pyfunction]
fn hilbert_xy2d_batch<'py>(
    py: Python<'py>,
    grid_size: i64,
    xs: Vec<i32>,
    ys: Vec<i32>,
) -> Bound<'py, numpy::PyArray1<i64>> {
    let n = xs.len();
    let results: Vec<i64> = (0..n).into_par_iter()
        .map(|i| xy2d_inner(grid_size, xs[i] as i64, ys[i] as i64))
        .collect();
    numpy::PyArray1::from_vec(py, results)
}
#[pyfunction]
fn hilbert_sort_indices<'py>(
    py: Python<'py>,
    points_flat: Vec<f32>,
    n_points: usize,
    grid_order: u32,
) -> Bound<'py, numpy::PyArray1<usize>> {
    let n_grid = 1i64 << grid_order;
    let max_coord = (n_grid - 1) as f32;
    let mut indexed: Vec<(i64, usize)> = (0..n_points).into_par_iter().map(|i| {
        let x = (points_flat[i * 2] * max_coord).clamp(0.0, max_coord) as i64;
        let y = (points_flat[i * 2 + 1] * max_coord).clamp(0.0, max_coord) as i64;
        (xy2d_inner(n_grid, x, y), i)
    }).collect();
    indexed.sort_unstable_by_key(|&(h, _)| h);
    let order: Vec<usize> = indexed.into_iter().map(|(_, i)| i).collect();
    numpy::PyArray1::from_vec(py, order)
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hilbert_xy2d_batch, m)?)?;
    m.add_function(wrap_pyfunction!(hilbert_sort_indices, m)?)?;
    Ok(())
}
