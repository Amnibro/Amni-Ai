use crate::Store as Core;
use numpy::{IntoPyArray, PyArray1};
use pyo3::{exceptions::{PyIOError, PyKeyError}, prelude::*, types::PyDict};
use std::sync::Mutex;
#[pyclass(name = "Store", module = "adam_gf17")]
struct Store { s: Mutex<Core> }
fn ioe(e: std::io::Error) -> PyErr { PyIOError::new_err(e.to_string()) }
fn arr<'py>(py: Python<'py>, b: Vec<u8>, descr: &str, shape: Vec<usize>) -> PyResult<Bound<'py, PyAny>> { b.into_pyarray(py).into_any().call_method1("view", (descr,))?.call_method1("reshape", (shape,)) }
#[pymethods]
impl Store {
    #[new]
    #[pyo3(signature = (path, budget_mb=256))]
    fn new(path: &str, budget_mb: usize) -> PyResult<Self> { Core::open(path, budget_mb << 20).map(|s| Store { s: Mutex::new(s) }).map_err(ioe) }
    #[getter]
    fn path(&self) -> String { self.s.lock().unwrap().path.clone() }
    #[getter]
    fn tables(&self) -> Vec<(String, String, Vec<usize>, bool, bool)> { self.s.lock().unwrap().tabs.iter().map(|t| (t.name.clone(), t.descr().into(), t.shape.clone(), t.exact, t.ord)).collect() }
    #[getter]
    fn tile(&self) -> usize { self.s.lock().unwrap().tile }
    #[getter]
    fn size(&self) -> usize { self.s.lock().unwrap().len() }
    #[getter]
    fn streams(&self) -> [u64; 4] { let s = self.s.lock().unwrap(); s.ents.iter().fold([0u64; 4], |mut a, e| { a[crate::stream(&s.tabs, e)] += e.len; a }) }
    #[getter]
    fn stats(&self) -> (u64, u64, u64, usize, usize, usize) { let s = self.s.lock().unwrap(); (s.hits, s.miss, s.ev, s.used, s.budget, crate::rss().0) }
    #[pyo3(signature = (level, cell, slot, tiers=4))]
    fn fetch(&self, level: usize, cell: usize, slot: usize, tiers: usize) -> PyResult<(u8, f32)> { let (i, w) = self.s.lock().unwrap().fetch_many(&[(level, cell, slot)], tiers).map_err(ioe)?[0]; Ok((i, half::f16::from_bits(w).to_f32())) }
    #[pyo3(signature = (levels, cells, slots, tiers=4))]
    fn fetch_many<'py>(&self, py: Python<'py>, levels: Vec<usize>, cells: Vec<usize>, slots: Vec<usize>, tiers: usize) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyAny>)> {
        let q: Vec<(usize, usize, usize)> = levels.into_iter().zip(cells).zip(slots).map(|((l, c), s)| (l, c, s)).collect();
        let r = py.allow_threads(|| self.s.lock().unwrap().fetch_many(&q, tiers)).map_err(ioe)?;
        Ok((r.iter().map(|x| x.0).collect::<Vec<_>>().into_pyarray(py), arr(py, r.iter().flat_map(|x| x.1.to_le_bytes()).collect(), "<f2", vec![r.len()])?))
    }
    #[pyo3(signature = (tiers=4))]
    fn unpack<'py>(&self, py: Python<'py>, tiers: usize) -> PyResult<Bound<'py, PyDict>> {
        let s = self.s.lock().unwrap();
        let d = PyDict::new(py);
        for (t, tb) in s.tabs.iter().enumerate() { let (b, _) = py.allow_threads(|| s.unpack(t, tiers)).map_err(ioe)?; d.set_item(&tb.name, arr(py, b, tb.descr(), tb.shape.clone())?)?; }
        Ok(d)
    }
    fn page<'py>(&self, py: Python<'py>, name: &str, level: usize, page: usize) -> PyResult<Bound<'py, PyAny>> {
        let s = self.s.lock().unwrap();
        let t = s.table(name).ok_or_else(|| PyKeyError::new_err(name.to_string()))?;
        let rows = s.tabs[t].rows(page);
        arr(py, py.allow_threads(|| s.rgba(t, level, page)).map_err(ioe)?, "|u1", vec![rows, crate::PW, 4])
    }
    #[pyo3(signature = (level=2))]
    fn cold(&self, level: usize) { self.s.lock().unwrap().reset(level) }
}
#[pyfunction]
#[pyo3(signature = (prefix, out=None, ord=false, budget_mb=256, tile=0))]
fn pack(py: Python<'_>, prefix: &str, out: Option<String>, ord: bool, budget_mb: usize, tile: usize) -> PyResult<Store> { let o = out.unwrap_or_else(|| format!("{prefix}.gf17")); py.allow_threads(|| crate::pack(prefix, &o, ord, tile)).map_err(ioe)?; Store::new(&o, budget_mb) }
#[pyfunction]
#[pyo3(signature = (store, tiers=4))]
fn unpack<'py>(py: Python<'py>, store: &Store, tiers: usize) -> PyResult<Bound<'py, PyDict>> { store.unpack(py, tiers) }
#[pymodule]
fn adam_gf17(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Store>()?;
    m.add_function(wrap_pyfunction!(pack, m)?)?;
    m.add_function(wrap_pyfunction!(unpack, m)?)
}
