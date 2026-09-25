use adam_qa::{Hit, Qa as QaCore};
use ray::{Dec, RayField as Core, Stream};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use parking_lot::{lock_api::ArcMutexGuard, Mutex, RawMutex};
use pyo3::{exceptions::PyIOError, prelude::*, types::PyBytes};
use std::sync::Arc;
type Guard = ArcMutexGuard<RawMutex, Core>;
#[pyclass(name = "RayField", module = "adam_ray")]
struct RayField { e: Arc<Mutex<Core>> }
#[pyclass(module = "adam_ray")]
struct StreamIter { e: Arc<Mutex<Core>>, g: Option<Guard>, s: Option<Stream>, a: Option<(String, usize, f64, usize, Vec<String>, bool, Option<u64>, Option<Vec<u8>>, f64, Dec)> }
#[pyclass(name = "QA", module = "adam_ray")]
struct Qa { q: Arc<QaCore> }
fn lk(py: Python<'_>, e: &Arc<Mutex<Core>>) -> PyResult<Guard> { let mut g = py.allow_threads(|| e.lock_arc()); py.allow_threads(|| g.load().map(|_| ()).map_err(|x| PyIOError::new_err(x.to_string())))?; Ok(g) }
fn arr2(py: Python<'_>, v: Vec<f32>) -> PyResult<Bound<'_, PyArray2<f64>>> { let n = v.len() / 256; v.into_iter().map(|x| x as f64).collect::<Vec<_>>().into_pyarray(py).reshape([n, 256]) }
fn hit<'py>(py: Python<'py>, h: Hit) -> (f64, f64, String, Bound<'py, PyBytes>, i64) { (h.score, h.cov, h.q, PyBytes::new(py, &h.a), h.src) }
#[pymethods]
impl RayField {
    #[new]
    #[pyo3(signature = (path=None, syn=None, words=None, wcnt=None, arch=None, home=None, pages=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(path: Option<String>, syn: Option<String>, words: Option<String>, wcnt: Option<String>, arch: Option<String>, home: Option<String>, pages: Option<String>) -> Self { RayField { e: Arc::new(Mutex::new(Core::new(pages.or(path), syn, words, wcnt, arch, home))) } }
    #[getter]
    fn pages(&self) -> Option<String> { let p = self.e.lock().path.clone(); p.ends_with(".gf17").then_some(p) }
    #[staticmethod]
    fn page_stats() -> Vec<u64> { ray::src_gf17::stats().to_vec() }
    fn load(slf: Py<Self>, py: Python<'_>) -> PyResult<Py<Self>> { lk(py, &slf.borrow(py).e).map(drop)?; Ok(slf) }
    #[getter]
    fn path(&self) -> String { self.e.lock().path.clone() }
    #[getter]
    fn syn(&self) -> String { self.e.lock().syn.clone() }
    #[getter]
    fn words_path(&self) -> String { self.e.lock().words_path.clone() }
    #[getter]
    fn wcnt_path(&self) -> String { self.e.lock().wcnt_path.clone() }
    #[getter]
    fn arch(&self) -> String { self.e.lock().arch.clone() }
    #[getter]
    fn ready(&self) -> bool { self.e.try_lock().is_none_or(|g| g.ready) }
    #[getter]
    fn backend(&self) -> String { self.e.lock().backend.clone() }
    #[getter(L)]
    fn l(&self) -> usize { self.e.lock().l }
    #[getter(T)]
    fn t(&self) -> usize { self.e.lock().t }
    #[getter]
    fn qt(&self) -> bool { self.e.lock().qt }
    #[getter]
    fn copy(&self) -> usize { self.e.lock().copy }
    #[getter]
    fn v19(&self) -> bool { self.e.lock().copy > 0 }
    #[getter(r#impl)]
    fn imp(&self) -> &'static str { "rust" }
    #[pyo3(signature = (data, nb=1))]
    fn prime<'py>(&self, py: Python<'py>, data: Vec<u8>, nb: usize) -> PyResult<Bound<'py, PyArray2<f64>>> { let mut g = lk(py, &self.e)?; arr2(py, py.allow_threads(|| g.prime(&data, nb))) }
    fn logits_for<'py>(&self, py: Python<'py>, data: Vec<u8>) -> PyResult<Bound<'py, PyArray2<f64>>> { let mut g = lk(py, &self.e)?; arr2(py, py.allow_threads(|| g.logits_for(&data))) }
    #[pyo3(signature = (par, byt, want=true))]
    fn _advance<'py>(&self, py: Python<'py>, par: Vec<u32>, byt: Vec<u8>, want: bool) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> { let mut g = lk(py, &self.e)?; py.allow_threads(|| g.advance(&par, &byt, want)).map(|v| arr2(py, v)).transpose() }
    fn score<'py>(&self, py: Python<'py>, prompts: Vec<Vec<u8>>, conts: Vec<Vec<u8>>) -> PyResult<Bound<'py, PyArray1<f64>>> { let mut g = lk(py, &self.e)?; Ok(py.allow_threads(|| g.score(&prompts, &conts)).into_pyarray(py)) }
    #[pyo3(signature = (hold=None, batches=60, ctx=128, bs=256, seed=1234))]
    fn bpb(&self, py: Python<'_>, hold: Option<String>, batches: usize, ctx: usize, bs: usize, seed: u64) -> PyResult<f64> { let mut g = lk(py, &self.e)?; py.allow_threads(|| g.bpb(hold.as_deref(), batches, ctx, bs, seed)).map_err(|x| PyIOError::new_err(x.to_string())) }
    #[pyo3(signature = (prompt, max_bytes=400, temp=0.7, topk=12, stop=vec!["\n\n".to_string()], veto=true, seed=None, guide=None, beta=6.0, lock=false, stop_sentence=false, rep=0.0, dlock=false))]
    #[allow(clippy::too_many_arguments)]
    fn stream(&self, prompt: String, max_bytes: usize, temp: f64, topk: usize, stop: Vec<String>, veto: bool, seed: Option<u64>, guide: Option<Vec<u8>>, beta: f64, lock: bool, stop_sentence: bool, rep: f64, dlock: bool) -> StreamIter { StreamIter { e: self.e.clone(), g: None, s: None, a: Some((prompt, max_bytes, temp, topk, stop, veto, seed, guide, beta, Dec { lock, stop_sentence, rep, dlock })) } }
    #[pyo3(signature = (prompt, width=8, expand=4, maxlen=300, rep=16, veto=true, lock=false, stop_sentence=false, dlock=false))]
    #[allow(clippy::too_many_arguments)]
    fn beam(&self, py: Python<'_>, prompt: String, width: usize, expand: usize, maxlen: usize, rep: usize, veto: bool, lock: bool, stop_sentence: bool, dlock: bool) -> PyResult<String> { let mut g = lk(py, &self.e)?; Ok(py.allow_threads(|| g.beam(&prompt, width, expand, maxlen, rep, veto, Dec { lock, stop_sentence, rep: 0.0, dlock }))) }
    fn word_ok(&self, seq: Vec<u8>, c: u8) -> bool { self.e.lock().word_ok(&seq, c) }
}
#[pymethods]
impl StreamIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<String>> {
        if let Some((p, mb, t, k, st, v, sd, gd, b, dc)) = self.a.take() { let mut g = lk(py, &self.e)?; self.s = Some(py.allow_threads(|| g.stream(&p, mb, t, k, &st, v, sd, gd.as_deref(), b, dc))); self.g = Some(g); }
        let r = match (self.g.as_mut(), self.s.as_mut()) { (Some(g), Some(s)) => py.allow_threads(|| s.next(g)), _ => None };
        r.is_none().then(|| self.close());
        Ok(r)
    }
    fn close(&mut self) { (self.g, self.s) = (None, None); }
}
#[pymethods]
impl Qa {
    #[new]
    #[pyo3(signature = (path=None))]
    fn new(py: Python<'_>, path: Option<String>) -> PyResult<Self> { py.allow_threads(|| QaCore::load(path.as_deref())).map(|q| Qa { q: Arc::new(q) }).map_err(|x| PyIOError::new_err(x.to_string())) }
    fn __len__(&self) -> usize { self.q.len() }
    #[pyo3(signature = (q, k=3))]
    fn search<'py>(&self, py: Python<'py>, q: &str, k: usize) -> Vec<(f64, f64, String, Bound<'py, PyBytes>, i64)> { self.q.search(q, k).into_iter().map(|h| hit(py, h)).collect() }
    fn search_ids(&self, q: &str, k: usize) -> Vec<(usize, f64)> { self.q.search(q, k).into_iter().map(|h| (h.id, h.score)).collect() }
    fn overlap(&self, q: &str, c: &str) -> (f64, f64, f64) { self.q.overlap(q, c, false) }
    fn answer<'py>(&self, py: Python<'py>, i: usize) -> Bound<'py, PyBytes> { PyBytes::new(py, &self.q.answer(i)) }
}
#[pyfunction]
fn words<'py>(py: Python<'py>, b: Vec<u8>) -> Vec<Bound<'py, PyBytes>> { adam_qa::words(&b).iter().map(|w| PyBytes::new(py, w)).collect() }
#[pyfunction]
fn stem<'py>(py: Python<'py>, w: Vec<u8>) -> Bound<'py, PyBytes> { PyBytes::new(py, &adam_qa::stem(&w)) }
#[pyfunction]
#[pyo3(signature = (q, c, a=vec![]))]
fn hard(q: &str, c: &str, a: Vec<u8>) -> bool { adam_qa::hard(q, c, &a) }
#[pyfunction]
fn pick<'py>(py: Python<'py>, e: PyRef<'py, RayField>, qa: PyRef<'py, Qa>, q: &str) -> PyResult<Vec<(f64, f64, String, Bound<'py, PyBytes>, i64)>> { let (mut g, x) = (lk(py, &e.e)?, qa.q.clone()); Ok(py.allow_threads(|| adam_qa::pick(&mut g, &x, q)).into_iter().map(|h| hit(py, h)).collect()) }
#[pyfunction]
#[pyo3(signature = (e, qa, q, cov=None))]
fn guide<'py>(py: Python<'py>, e: PyRef<'py, RayField>, qa: PyRef<'py, Qa>, q: &str, cov: Option<f64>) -> PyResult<Option<Bound<'py, PyBytes>>> { let (mut g, x) = (lk(py, &e.e)?, qa.q.clone()); Ok(py.allow_threads(|| adam_qa::guide(&mut g, &x, q, cov)).map(|a| PyBytes::new(py, &a))) }
#[pymodule]
fn adam_ray(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RayField>()?;
    m.add_class::<StreamIter>()?;
    m.add_class::<Qa>()?;
    m.add_function(wrap_pyfunction!(words, m)?)?;
    m.add_function(wrap_pyfunction!(stem, m)?)?;
    m.add_function(wrap_pyfunction!(hard, m)?)?;
    m.add_function(wrap_pyfunction!(pick, m)?)?;
    m.add_function(wrap_pyfunction!(guide, m)?)?;
    Ok(())
}
