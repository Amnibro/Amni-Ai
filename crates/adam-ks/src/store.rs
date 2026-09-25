use crate::{bad, crc16, jstr, paginate, pyjson, Ks, CMAX, PAGE};
use arrow_array::{cast::AsArray, types::{Float32Type, Float64Type, Int32Type, Int64Type, UInt8Type}, Array, ArrayRef, BooleanArray, Float32Array, RecordBatch, StringArray, UInt32Array, UInt8Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use memmap2::Mmap;
use parquet::{arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter, ProjectionMask}, basic::{Compression, ZstdLevel}, file::properties::WriterProperties};
use serde_json::{json, Value};
use std::{fs::{self, File}, io::{self, BufWriter, Write}, mem::take, path::{Path, PathBuf}, sync::Arc};
pub const MAGIC: &[u8; 4] = b"AKS1";
pub const ATW: usize = 4096;
pub const TIERS: [&str; 4] = ["verified", "endorsed", "unverified", "rejected"];
const K8: [u64; 8] = [1, 17, 289, 4913, 83521, 1419857, 24137569, 410338673];
fn pe(e: impl ToString) -> io::Error { bad(e.to_string()) }
fn dig(v: u64) -> [u8; 8] { std::array::from_fn(|k| (v / K8[k] % 17) as u8) }
fn chk(d: &[u8]) -> i64 { (0..7).map(|k| d[k] as i64 * (k as i64 + 1)).sum::<i64>() + (0..4).map(|k| d[8 + k] as i64 * (k as i64 + 8)).sum::<i64>() }
pub fn dir_pack(clen: &[u32], nby: &[u32], crc: &[u16]) -> Vec<[u8; 12]> {
    (0..clen.len()).map(|i| { let (a, b) = (dig(clen[i] as u64 + 83521 * nby[i] as u64), dig(crc[i] as u64)); let mut d: [u8; 12] = std::array::from_fn(|k| if k < 8 { a[k] } else { b[k - 8] }); d[7] = (-chk(&d)).rem_euclid(17) as u8; d }).collect()
}
pub fn dir_unpack(d: &[[u8; 12]]) -> io::Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    if d.iter().any(|r| r.iter().any(|&x| x > 16) || (chk(r) + r[7] as i64) % 17 != 0) { return Err(bad("KS index corrupt: GF(17) check digit mismatch")); }
    let v: Vec<u64> = d.iter().map(|r| (0..7).map(|k| r[k] as u64 * K8[k]).sum()).collect();
    Ok((v.iter().map(|&x| (x % 83521) as u32).collect(), v.iter().map(|&x| (x / 83521) as u32).collect(), d.iter().map(|r| (0..4).map(|k| r[8 + k] as u64 * K8[k]).sum::<u64>() as u32).collect()))
}
pub fn write_dir(out: &Path, clen: &[u32], nby: &[u32], crc: &[u16]) -> io::Result<()> {
    let d = dir_pack(clen, nby, crc);
    let raw: Vec<u8> = (0..12).flat_map(|j| d.iter().map(move |r| r[j])).collect();
    let pay = zstd::bulk::compress(&raw, 19)?;
    let hd = format!("{{\"n\": {}, \"crc32\": {}, \"raw\": {}, \"atlas\": [{}, {ATW}, 4], \"code_bytes\": {}}}", d.len(), crc32fast::hash(&pay), raw.len(), (3 * d.len()).div_ceil(ATW).max(1), clen.iter().map(|&x| x as u64).sum::<u64>());
    fs::write(out.join("dir.ptex"), [&MAGIC[..], &(hd.len() as u32).to_le_bytes(), hd.as_bytes(), &pay].concat())
}
fn reader(p: &Path, cols: &[&str], batch: usize) -> io::Result<(parquet::arrow::arrow_reader::ParquetRecordBatchReader, Vec<String>)> {
    let b = ParquetRecordBatchReaderBuilder::try_new(File::open(p)?).map_err(pe)?;
    let have: Vec<String> = cols.iter().filter(|c| b.schema().fields().iter().any(|f| f.name() == *c)).map(|c| c.to_string()).collect();
    let m = ProjectionMask::columns(b.parquet_schema(), have.iter().map(String::as_str));
    Ok((b.with_projection(m).with_batch_size(batch).build().map_err(pe)?, have))
}
fn strs(b: &RecordBatch, k: &str) -> Vec<Option<String>> { b.column_by_name(k).map_or_else(|| vec![None; b.num_rows()], |c| match c.data_type() { DataType::LargeUtf8 => c.as_string::<i64>().iter().map(|x| x.map(String::from)).collect(), DataType::Utf8View => c.as_string_view().iter().map(|x| x.map(String::from)).collect(), _ => c.as_string::<i32>().iter().map(|x| x.map(String::from)).collect() }) }
fn nums(b: &RecordBatch, k: &str) -> Vec<Option<f64>> { b.column_by_name(k).map_or_else(|| vec![None; b.num_rows()], |c| match c.data_type() { DataType::Float32 => c.as_primitive::<Float32Type>().iter().map(|x| x.map(|v| v as f64)).collect(), DataType::Int64 => c.as_primitive::<Int64Type>().iter().map(|x| x.map(|v| v as f64)).collect(), DataType::Int32 => c.as_primitive::<Int32Type>().iter().map(|x| x.map(|v| v as f64)).collect(), DataType::Boolean => c.as_boolean().iter().map(|x| x.map(|v| v as u8 as f64)).collect(), _ => c.as_primitive::<Float64Type>().iter().collect() }) }
pub struct Doc { pub title: Option<String>, pub text: String, pub tier: u8, pub score: f64, pub stale: bool, pub date: Option<String>, pub ev: String }
pub fn read_docs(p: &Path, batch: usize) -> io::Result<impl Iterator<Item = io::Result<Doc>>> {
    let (r, have) = reader(p, &["title", "text", "url", "date", "score", "stale", "refs", "license"], batch)?;
    let lic = have.iter().any(|c| c == "license");
    Ok(r.flat_map(move |rb| match rb {
        Err(e) => vec![Err(pe(e))],
        Ok(b) => {
            let (ti, tx, url, dt, sc, st, rf, li) = (strs(&b, "title"), strs(&b, "text"), strs(&b, "url"), strs(&b, "date"), nums(&b, "score"), nums(&b, "stale"), nums(&b, "refs"), strs(&b, "license"));
            let js = |s: &Option<String>| s.as_deref().map_or("null".to_string(), jstr);
            (0..b.num_rows()).map(|i| Ok(Doc { title: ti[i].clone(), text: tx[i].clone().unwrap_or_default(), tier: 1, score: sc[i].unwrap_or(f64::NAN), stale: st[i].is_some_and(|x| x != 0.0), date: dt[i].clone(), ev: format!("{{\"urls\":[{}],\"agree\":1,\"refs\":{}{},\"signals\":{{}},\"code\":null}}", js(&url[i]), rf[i].map_or("null".to_string(), |x| (x as i64).to_string()), if lic { format!(",\"license\":{}", js(&li[i])) } else { String::new() }) })).collect()
        }
    }))
}
fn props() -> WriterProperties { WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::try_new(19).unwrap())).build() }
fn schema(f: &[(&str, DataType)]) -> SchemaRef { Arc::new(Schema::new(f.iter().map(|(n, t)| Field::new(*n, t.clone(), true)).collect::<Vec<_>>())) }
struct W { pf: BufWriter<File>, dw: ArrowWriter<File>, pw: ArrowWriter<File>, ds: SchemaRef, ps: SchemaRef, clen: Vec<u32>, nby: Vec<u32>, crc: Vec<u16>, buf: Vec<Vec<u8>>, ti: Vec<Option<String>>, fi: Vec<u32>, tr: Vec<u8>, sc: Vec<f32>, stl: Vec<bool>, dt: Vec<Option<String>>, ev: Vec<String> }
impl W {
    fn flush(&mut self, ks: &Ks) -> io::Result<()> {
        let (cs, _) = ks.encode_many(&self.buf.iter().map(Vec::as_slice).collect::<Vec<_>>(), false);
        for (c, p) in cs.iter().zip(&self.buf) { self.pf.write_all(c)?; self.clen.push(c.len() as u32); self.nby.push(p.len() as u32); self.crc.push(crc16(p)); }
        let n = self.buf.len();
        let pc: Vec<ArrayRef> = vec![Arc::new(UInt8Array::from(vec![0u8; n])), Arc::new(UInt8Array::from(take(&mut self.tr))), Arc::new(Float32Array::from(take(&mut self.sc))), Arc::new(BooleanArray::from(take(&mut self.stl))), Arc::new(StringArray::from(take(&mut self.dt))), Arc::new(StringArray::from(take(&mut self.ev)))];
        let dc: Vec<ArrayRef> = vec![Arc::new(UInt8Array::from(vec![0u8; self.ti.len()])), Arc::new(StringArray::from(take(&mut self.ti))), Arc::new(UInt32Array::from(take(&mut self.fi)))];
        self.pw.write(&RecordBatch::try_new(self.ps.clone(), pc).map_err(pe)?).map_err(pe)?;
        self.dw.write(&RecordBatch::try_new(self.ds.clone(), dc).map_err(pe)?).map_err(pe)?;
        self.buf.clear();
        Ok(())
    }
}
pub fn write_store(out: &Path, src: &str, docs: impl Iterator<Item = io::Result<Doc>>, ks: &Ks, chunk: usize, log: &mut dyn FnMut(usize, u64, u64)) -> io::Result<Value> {
    fs::create_dir_all(out)?;
    let (ds, ps) = (schema(&[("src", DataType::UInt8), ("title", DataType::Utf8), ("first", DataType::UInt32)]), schema(&[("coder", DataType::UInt8), ("tier", DataType::UInt8), ("score", DataType::Float32), ("stale", DataType::Boolean), ("date", DataType::Utf8), ("evidence", DataType::Utf8)]));
    let mut w = W { pf: BufWriter::with_capacity(1 << 20, File::create(out.join("pages.bin"))?), dw: ArrowWriter::try_new(File::create(out.join("docs.parquet"))?, ds.clone(), Some(props())).map_err(pe)?, pw: ArrowWriter::try_new(File::create(out.join("pagemeta.parquet"))?, ps.clone(), Some(props())).map_err(pe)?, ds, ps, clen: vec![], nby: vec![], crc: vec![], buf: vec![], ti: vec![], fi: vec![], tr: vec![], sc: vec![], stl: vec![], dt: vec![], ev: vec![] };
    let (mut tb, mut nd) = (0u64, 0usize);
    for d in docs {
        let d = d?;
        let ps = paginate(d.text.as_bytes());
        w.fi.push((w.clen.len() + w.buf.len()) as u32);
        w.ti.push(d.title);
        w.buf.extend(ps.iter().map(|p| p.to_vec()));
        w.tr.extend(std::iter::repeat_n(d.tier, ps.len()));
        w.sc.extend(std::iter::repeat_n(d.score as f32, ps.len()));
        w.stl.extend(std::iter::repeat_n(d.stale, ps.len()));
        w.dt.extend(std::iter::repeat_n(d.date, ps.len()));
        w.ev.extend(std::iter::repeat_n(d.ev, ps.len()));
        (tb, nd) = (tb + d.text.len() as u64, nd + 1);
        if w.buf.len() >= chunk { w.flush(ks)?; log(w.clen.len(), tb, w.clen.iter().map(|&x| x as u64).sum()); }
    }
    if !w.buf.is_empty() { w.flush(ks)?; }
    w.pf.flush()?;
    w.dw.close().map_err(pe)?;
    w.pw.close().map_err(pe)?;
    write_dir(out, &w.clen, &w.nby, &w.crc)?;
    let meta = json!({"coders": [ks.id.json()], "tiers": TIERS, "sources": [src], "docs": nd, "pages": w.clen.len(), "src_bytes": tb, "code_bytes": w.clen.iter().map(|&x| x as u64).sum::<u64>()});
    fs::write(out.join("meta.json"), pyjson(&meta, Some(1), 0))?;
    Ok(meta)
}
pub struct Store { pub path: PathBuf, pub meta: Value, pub clen: Vec<u32>, pub nby: Vec<u32>, pub crc: Vec<u32>, pub off: Vec<u64>, pages: Option<Mmap>, pub coder: Vec<u8> }
impl Store {
    pub fn open(p: &Path) -> io::Result<Store> {
        let meta: Value = serde_json::from_slice(&fs::read(p.join("meta.json"))?).map_err(pe)?;
        let raw = fs::read(p.join("dir.ptex"))?;
        if raw.len() < 8 || &raw[..4] != MAGIC { return Err(bad("KS index corrupt: bad magic")); }
        let n = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
        let hd: Value = serde_json::from_slice(&raw[8..8 + n]).map_err(pe)?;
        let pay = &raw[8 + n..];
        if Some(crc32fast::hash(pay) as u64) != hd["crc32"].as_u64() { return Err(bad("KS index corrupt: directory crc32 mismatch")); }
        let dr = zstd::bulk::decompress(pay, hd["raw"].as_u64().unwrap_or(0) as usize)?;
        let np = dr.len() / 12;
        let (clen, nby, crc) = dir_unpack(&(0..np).map(|i| std::array::from_fn(|j| dr[j * np + i])).collect::<Vec<[u8; 12]>>())?;
        let off: Vec<u64> = std::iter::once(0).chain(clen.iter().scan(0u64, |a, &x| { *a += x as u64; Some(*a) })).collect();
        let f = File::open(p.join("pages.bin"))?;
        let pl = f.metadata()?.len();
        let pages = if pl > 0 { Some(unsafe { Mmap::map(&f)? }) } else { None };
        if Some(np as u64) != hd["n"].as_u64() || off[np] != pl || nby.iter().any(|&x| x as usize > PAGE) || clen.iter().any(|&x| x as usize > CMAX) { return Err(bad("KS index corrupt: directory does not match pages.bin")); }
        let (r, _) = reader(&p.join("pagemeta.parquet"), &["coder"], 1 << 16)?;
        let mut coder = vec![];
        for b in r { let b = b.map_err(pe)?; coder.extend(b.column(0).as_primitive::<UInt8Type>().values().iter().copied()); }
        let nc = meta["coders"].as_array().map_or(0, Vec::len);
        if coder.len() != np || coder.iter().any(|&c| c as usize >= nc) { return Err(bad("KS index corrupt: page metadata does not match directory")); }
        Ok(Store { path: p.to_path_buf(), meta, clen, nby, crc, off, pages, coder })
    }
    pub fn len(&self) -> usize { self.clen.len() }
    pub fn is_empty(&self) -> bool { self.clen.is_empty() }
    pub fn code(&self, i: usize) -> &[u8] { self.pages.as_ref().map_or(&[][..], |m| &m[self.off[i] as usize..self.off[i + 1] as usize]) }
    pub fn matches(&self, k: usize, ks: &Ks, check: bool) -> bool {
        let (m, id) = (&self.meta["coders"][k], ks.id.json());
        ["coder", "pack_id", "lex_id", "impl", "arch"][..if check { 5 } else { 3 }].iter().all(|x| m.get(*x).unwrap_or(if *x == "lex_id" { &id[*x] } else { &Value::Null }) == &id[*x])
    }
    pub fn get(&self, ids: &[usize], cs: &[&Ks], check: bool) -> io::Result<Vec<Vec<u8>>> {
        let ks: Vec<Option<&Ks>> = (0..self.meta["coders"].as_array().map_or(0, Vec::len)).map(|k| cs.iter().copied().find(|c| self.matches(k, c, check))).collect();
        let miss: Vec<usize> = ids.iter().copied().filter(|&i| ks[self.coder[i] as usize].is_none()).collect();
        if let Some(&i) = miss.first() { let m = &self.meta["coders"][self.coder[i] as usize]; return Err(bad(format!("KS frozen-coder mismatch on pages {:?}: stored {} (lex {}) vs codecs {:?}", &miss[..miss.len().min(8)], m["coder"].as_str().unwrap_or(""), m["lex"].as_str().unwrap_or("big2"), cs.iter().map(|c| format!("{} lex {}", c.id.coder, c.id.lex)).collect::<Vec<_>>()))); }
        let mut out = vec![vec![]; ids.len()];
        for (k, c) in ks.iter().enumerate().filter_map(|(k, c)| c.map(|c| (k, c))) { let j: Vec<usize> = (0..ids.len()).filter(|&j| self.coder[ids[j]] as usize == k).collect(); j.iter().zip(c.decode_many(&j.iter().map(|&j| self.nby[ids[j]] as usize).collect::<Vec<_>>(), &j.iter().map(|&j| self.code(ids[j])).collect::<Vec<_>>())).for_each(|(&j, x)| out[j] = x); }
        let bad_: Vec<usize> = ids.iter().zip(&out).filter(|(&i, x)| crc16(x) as u32 != self.crc[i]).map(|x| *x.0).collect();
        if !bad_.is_empty() { return Err(bad(format!("KS decode crc check failed on pages {:?} (model/arch mismatch?)", &bad_[..bad_.len().min(8)]))); }
        Ok(out)
    }
}
