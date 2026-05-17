use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::thread;
use std::time::{Duration, Instant};

#[pyclass(unsendable)]
pub struct AngelThread {
    target: Option<PyObject>,
    args: Option<Py<PyTuple>>,
    kwargs: Option<Py<PyDict>>,
    handle: Option<thread::JoinHandle<()>>,
}

#[pymethods]
impl AngelThread {
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(py: Python, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut target = None;
        let mut t_args = None;
        let mut t_kwargs = None;
        
        if let Some(kw) = kwargs {
            if let Ok(Some(t)) = kw.get_item("target") {
                target = Some(t.into());
                let _ = kw.del_item("target");
            }
            if let Ok(Some(a)) = kw.get_item("args") {
                if let Ok(tup) = a.downcast::<PyTuple>() {
                    t_args = Some(tup.clone().unbind());
                }
                let _ = kw.del_item("args");
            }
            if let Ok(Some(k)) = kw.get_item("kwargs") {
                if let Ok(dct) = k.downcast::<PyDict>() {
                    t_kwargs = Some(dct.clone().unbind());
                }
                let _ = kw.del_item("kwargs");
            }
        }
        
        Ok(AngelThread {
            target,
            args: t_args,
            kwargs: t_kwargs,
            handle: None,
        })
    }
    
    fn start(&mut self, py: Python) -> PyResult<()> {
        let target = self.target.as_ref().map(|t| t.clone_ref(py));
        let args = self.args.as_ref().map(|a| a.clone_ref(py));
        let kwargs = self.kwargs.as_ref().map(|k| k.clone_ref(py));
        
        let handle = thread::spawn(move || {
            Python::with_gil(|py| {
                if let Some(t) = target {
                    let empty_args = PyTuple::empty_bound(py);
                    let a = if let Some(aa) = &args { aa.bind(py) } else { &empty_args };
                    let k = kwargs.as_ref().map(|kk| kk.bind(py));
                    let _ = t.call_bound(py, a, k);
                }
            });
        });
        self.handle = Some(handle);
        Ok(())
    }
    
    #[pyo3(signature = (timeout=None))]
    fn join(&mut self, py: Python, timeout: Option<f64>) -> PyResult<()> {
        let start = Instant::now();
        if let Some(t) = timeout {
            py.allow_threads(|| {
                if let Some(handle) = &self.handle {
                    while !handle.is_finished() {
                        if start.elapsed().as_secs_f64() > t {
                            break;
                        }
                        thread::sleep(Duration::from_millis(10));
                    }
                }
            });
            if let Some(handle) = &self.handle {
                if handle.is_finished() {
                    let h = self.handle.take().unwrap();
                    let _ = py.allow_threads(|| h.join());
                }
            }
        } else {
             if let Some(handle) = self.handle.take() {
                 let _ = py.allow_threads(|| handle.join());
             }
        }
        Ok(())
    }
    
    fn is_alive(&self) -> bool {
        if let Some(handle) = &self.handle {
            !handle.is_finished()
        } else {
            false
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AngelThread>()?;
    Ok(())
}
