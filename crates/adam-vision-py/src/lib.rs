use numpy::PyReadonlyArray3;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use vision::track::Eye;
use vision::{MipPyramid, Object};
#[pyclass(name = "Eye", module = "adam_vision")]
struct PyEye { e: Eye }
#[pymethods]
impl PyEye {
    #[new]
    #[pyo3(signature = (world = "A"))]
    fn new(world: &str) -> Self { PyEye { e: if world.eq_ignore_ascii_case("B") { Eye::world_b() } else { Eye::world_a() } } }
    fn reset(&mut self) { self.e.reset(); }
    #[getter]
    fn match_dist(&self) -> f32 { self.e.match_dist }
    #[getter]
    fn scale(&self) -> f32 { self.e.scale }
    fn step<'py>(&mut self, py: Python<'py>, frame: PyReadonlyArray3<u8>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let a = frame.as_array();
        let (h, w) = (a.shape()[0], a.shape()[1]);
        assert_eq!(a.shape()[2], 3);
        if self.e.w != w || self.e.h != h { self.e.w = w; self.e.h = h; }
        let rgb: Vec<u8> = a.iter().copied().collect();
        let tracked = py.allow_threads(|| self.e.step(&rgb));
        let mut out = Vec::with_capacity(tracked.objects.len());
        for o in &tracked.objects {
            let d = PyDict::new(py);
            d.set_item("id", o.id)?;
            d.set_item("cx", o.cx)?;
            d.set_item("cy", o.cy)?;
            d.set_item("area", o.area)?;
            d.set_item("mean", o.mean.to_vec())?;
            d.set_item("gf17", o.gf17)?;
            d.set_item("digits", o.digits.to_vec())?;
            d.set_item("colour_code", o.colour_code)?;
            out.push(d);
        }
        Ok(out)
    }
    fn observe_vocab(&mut self, colour_code: u32, kind: &str) {
        let o = Object { id: 0, cx: 0.0, cy: 0.0, area: 0, mean: [0; 3], gf17: colour_code, digits: [0; 4], colour_code: colour_code as u16 };
        self.e.observe_vocab(&o, kind);
    }
    fn vocab_report(&self) -> Vec<(u32, String, u32)> { self.e.vocab.report() }
    #[getter]
    fn n_slots_hit(&self) -> usize { self.e.table.combo.iter().filter(|&&c| c > 0).count() }
}
#[pyfunction]
fn mip_levels(frame: PyReadonlyArray3<u8>) -> PyResult<Vec<(usize, usize)>> {
    let a = frame.as_array();
    let (h, w) = (a.shape()[0], a.shape()[1]);
    let rgb: Vec<u8> = a.iter().copied().collect();
    let m = MipPyramid::from_rgb(w, h, &rgb);
    Ok(m.levels.iter().map(|(ww, hh, _)| (*ww, *hh)).collect())
}
#[pyfunction]
fn gf17_digits(n: u32) -> [u8; 4] { vision::gf17_digits(n) }
#[pyfunction]
fn gf17_pack(d: Vec<u8>) -> u32 { vision::gf17_pack([d[0], d[1], d[2], d[3]]) }
#[pymodule]
fn adam_vision(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEye>()?;
    m.add_function(wrap_pyfunction!(mip_levels, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_digits, m)?)?;
    m.add_function(wrap_pyfunction!(gf17_pack, m)?)?;
    Ok(())
}
