use pyo3::prelude::*;
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::path::PathBuf;
const TEX_MAGIC: u32 = 0x544D5558;
const HEADER_SIZE: usize = 12;
fn read_tex_page_inner(path: &str) -> PyResult<(Vec<u8>, i32, i32)> {
    let file = File::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}: {}", path, e)))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("mmap {}: {}", path, e)))?;
    if mmap.len() < HEADER_SIZE {
        return Err(pyo3::exceptions::PyValueError::new_err("tex file too small"));
    }
    let magic = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]);
    if magic != TEX_MAGIC {
        return Err(pyo3::exceptions::PyValueError::new_err(format!("bad tex magic {:#x}", magic)));
    }
    let w = i32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]);
    let h = i32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]);
    let expected = (w as usize) * (h as usize) * 4 + HEADER_SIZE;
    if mmap.len() < expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!("tex truncated: expected {} got {}", expected, mmap.len())));
    }
    Ok((mmap[HEADER_SIZE..expected].to_vec(), w, h))
}
#[pyfunction]
fn read_tex_page(path: &str) -> PyResult<(Vec<u8>, i32, i32)> {
    let (data, w, h) = read_tex_page_inner(path)?;
    Ok((data, w, h))
}
#[pyfunction]
fn load_vectors_from_pages<'py>(
    py: Python<'py>,
    page_paths: Vec<Vec<String>>,
    n_words: usize,
    dim: usize,
    chunk_size: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    let mut full = vec![0u8; n_words * dim * 4];
    for (ci, chunk_pages) in page_paths.iter().enumerate() {
        let ds = ci * chunk_size;
        let de = (ds + chunk_size).min(dim);
        let cc = de - ds;
        for (pi, path) in chunk_pages.iter().enumerate() {
            if path.is_empty() { continue; }
            let (data, _w, _h) = read_tex_page_inner(path)?;
            let rs = pi * 4096;
            let re = rs.min(n_words) + (4096.min(n_words.saturating_sub(rs)));
            let re = re.min(n_words);
            let nr = re - rs;
            for row in 0..nr {
                let src_off = row * (cc as usize) * 4;
                let dst_off = (rs + row) * dim * 4 + ds * 4;
                let copy_len = (cc as usize) * 4;
                if src_off + copy_len <= data.len() && dst_off + copy_len <= full.len() {
                    full[dst_off..dst_off + copy_len].copy_from_slice(&data[src_off..src_off + copy_len]);
                }
            }
        }
    }
    let floats: Vec<f32> = full.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(numpy::PyArray1::from_vec(py, floats))
}
#[pyfunction]
fn write_tex_page(
    path: &str,
    data: Vec<u8>,
    w: i32,
    h: i32,
) -> PyResult<()> {
    use std::io::Write;
    let dir = PathBuf::from(path).parent().map(|p| p.to_path_buf());
    if let Some(d) = dir { std::fs::create_dir_all(d).ok(); }
    let mut f = File::create(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
    f.write_all(&TEX_MAGIC.to_le_bytes())?;
    f.write_all(&w.to_le_bytes())?;
    f.write_all(&h.to_le_bytes())?;
    f.write_all(&data)?;
    Ok(())
}
#[pyfunction]
fn scan_json_facts(dir_path: &str, _tier: u8, _source: &str) -> PyResult<Vec<(String, String)>> {
    let dir = PathBuf::from(dir_path);
    if !dir.exists() { return Ok(Vec::new()); }
    let entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .filter(|p| p.file_name().is_some_and(|n| n != "delta_index.json"))
        .collect();
    let results: Vec<(String, String)> = entries.par_iter().filter_map(|fp| {
        let content = std::fs::read_to_string(fp).ok()?;
        let val: serde_json::Value = serde_json::from_str(&content).ok()?;
        let mut out = Vec::new();
        if let Some(obj) = val.as_object() {
            let subj = obj.get("subject").and_then(|s| s.as_str()).unwrap_or("").to_string();
            if let Some(facts_arr) = obj.get("facts").and_then(|f| f.as_array()) {
                for fact in facts_arr {
                    if let Some(txt) = fact.get("text").and_then(|t| t.as_str()) {
                        let t = txt.trim();
                        if t.len() > 10 { out.push((t.to_string(), subj.clone())); }
                    }
                }
            } else {
                let txt_keys = ["text", "description", "fact", "knowledge", "insight", "reflection", "content"];
                let subj_keys = ["subject", "topic", "name"];
                let txt = txt_keys.iter().find_map(|k| obj.get(*k).and_then(|v| v.as_str()).filter(|s| s.trim().len() > 10)).unwrap_or("").trim().to_string();
                let sub = subj_keys.iter().find_map(|k| obj.get(*k).and_then(|v| v.as_str())).unwrap_or("").to_string();
                if !txt.is_empty() { out.push((txt, if sub.is_empty() { subj } else { sub })); }
            }
        } else if let Some(arr) = val.as_array() {
            for entry in arr {
                if let Some(obj) = entry.as_object() {
                    let txt_keys = ["text", "description", "fact", "knowledge", "insight", "reflection", "content"];
                    let subj_keys = ["subject", "topic", "name"];
                    let txt = txt_keys.iter().find_map(|k| obj.get(*k).and_then(|v| v.as_str()).filter(|s| s.trim().len() > 10)).unwrap_or("").trim().to_string();
                    let sub = subj_keys.iter().find_map(|k| obj.get(*k).and_then(|v| v.as_str())).unwrap_or("").to_string();
                    let fallback = fp.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
                    if !txt.is_empty() { out.push((txt, if sub.is_empty() { fallback } else { sub })); }
                }
            }
        }
        Some(out)
    }).flatten().collect();
    Ok(results)
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_tex_page, m)?)?;
    m.add_function(wrap_pyfunction!(load_vectors_from_pages, m)?)?;
    m.add_function(wrap_pyfunction!(write_tex_page, m)?)?;
    m.add_function(wrap_pyfunction!(scan_json_facts, m)?)?;
    Ok(())
}
