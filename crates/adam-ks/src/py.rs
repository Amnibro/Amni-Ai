use crate::{pyjson, store::{read_docs, write_store, Store}, Ks};
use pyo3::{exceptions::PyIOError, prelude::*, types::PyBytes};
use std::{path::Path, sync::Arc};
fn er(e: std::io::Error) -> PyErr { PyIOError::new_err(e.to_string()) }
fn bs(v: &[Bound<'_, PyBytes>]) -> Vec<Vec<u8>> { v.iter().map(|p| p.as_bytes().to_vec()).collect() }
#[pyclass(name = "Codec", module = "adam_ks", frozen)]
struct Codec { k: Arc<Ks> }
#[pymethods]
impl Codec {
    #[new]
    #[pyo3(signature = (pack, lex, ver=None, threads=0))]
    fn new(py: Python<'_>, pack: String, lex: String, ver: Option<String>, threads: usize) -> PyResult<Self> { py.allow_threads(|| Ks::open(&pack, &lex, ver.as_deref(), threads)).map(|k| Codec { k: Arc::new(k) }).map_err(er) }
    fn ident(&self) -> String { pyjson(&self.k.id.json(), None, 0) }
    #[getter]
    fn threads(&self) -> usize { self.k.threads() }
    #[pyo3(signature = (pages, stats=false))]
    fn encode<'py>(&self, py: Python<'py>, pages: Vec<Bound<'py, PyBytes>>, stats: bool) -> (Vec<Bound<'py, PyBytes>>, f64) { let v = bs(&pages); let (c, n) = py.allow_threads(|| self.k.encode_many(&v.iter().map(Vec::as_slice).collect::<Vec<_>>(), stats)); (c.iter().map(|x| PyBytes::new(py, x)).collect(), n) }
    fn decode<'py>(&self, py: Python<'py>, lens: Vec<usize>, codes: Vec<Bound<'py, PyBytes>>) -> Vec<Bound<'py, PyBytes>> { let v = bs(&codes); py.allow_threads(|| self.k.decode_many(&lens, &v.iter().map(Vec::as_slice).collect::<Vec<_>>())).iter().map(|x| PyBytes::new(py, x)).collect() }
    #[pyo3(signature = (parquet, out, src="wikipedia".to_string(), chunk=4096))]
    fn encode_store(&self, py: Python<'_>, parquet: String, out: String, src: String, chunk: usize) -> PyResult<String> { py.allow_threads(|| read_docs(Path::new(&parquet), 256).and_then(|d| write_store(Path::new(&out), &src, d, &self.k, chunk, &mut |_, _, _| {}))).map(|m| pyjson(&m, None, 0)).map_err(er) }
    #[pyo3(signature = (path, ids, check=true))]
    fn store_get<'py>(&self, py: Python<'py>, path: String, ids: Vec<usize>, check: bool) -> PyResult<Vec<Bound<'py, PyBytes>>> { py.allow_threads(|| Store::open(Path::new(&path)).and_then(|s| s.get(&ids, &[&self.k], check))).map(|v| v.iter().map(|x| PyBytes::new(py, x)).collect()).map_err(er) }
}
#[pymodule]
fn adam_ks(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_class::<Codec>() }
