use adam_vision::track::{gt_in_frame, gt_pixel_a, gt_pixel_b, match_frame, Eye, TrackMetrics};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;
fn load_frames(npz: &Path) -> (Vec<u8>, usize, usize, usize, usize) {
    let mut zip = npyz::npz::NpzArchive::new(BufReader::new(File::open(npz).expect("npz"))).expect("zip");
    let name = {
        let has = matches!(zip.by_name("frames.npy"), Ok(Some(_)));
        if has { "frames.npy" } else { "frames" }
    };
    let arr = zip.by_name(name).ok().flatten().expect("frames");
    let shape = arr.shape().to_vec();
    assert_eq!(shape.len(), 5);
    let (ne, nt, h, w, c) = (shape[0] as usize, shape[1] as usize, shape[2] as usize, shape[3] as usize, shape[4] as usize);
    assert_eq!(c, 3);
    let data: Vec<u8> = arr.into_vec().expect("vec");
    assert_eq!(data.len(), ne * nt * h * w * 3);
    (data, ne, nt, h, w)
}
fn frame_at(data: &[u8], ne_i: usize, t: usize, nt: usize, h: usize, w: usize) -> &[u8] {
    let stride = nt * h * w * 3;
    let off = ne_i * stride + t * h * w * 3;
    &data[off..off + h * w * 3]
}
fn eval_world(root: &Path, world: &str) -> (TrackMetrics, f64, usize, Vec<(u32, String, u32)>) {
    let mut metrics = TrackMetrics::new();
    let mut nfr = 0usize;
    let t0 = Instant::now();
    let mut vocab_eye = if world == "a" { Eye::world_a() } else { Eye::world_b() };
    let mut shards: Vec<PathBuf> = fs::read_dir(root).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().and_then(|s| s.to_str()) == Some("npz") && p.file_name().and_then(|s| s.to_str()).unwrap_or("").starts_with("hold_")).collect();
    shards.sort();
    for sp in shards {
        let (data, ne, nt, h, w) = load_frames(&sp);
        let mut jp = sp.clone(); jp.set_extension("json");
        let meta: Value = serde_json::from_str(&fs::read_to_string(&jp).unwrap()).unwrap();
        let states = meta["states"].as_array().unwrap();
        for ei in 0..ne {
            let mut eye = if world == "a" { Eye::world_a() } else { Eye::world_b() };
            let mut last: HashMap<String, u32> = HashMap::new();
            let ep_states = states[ei].as_array().unwrap();
            for t in 0..nt {
                let rgb = frame_at(&data, ei, t, nt, h, w);
                let tracked = eye.step(rgb);
                nfr += 1;
                let mut gt = vec![];
                for o in ep_states[t].as_array().unwrap() {
                    let id = o["id"].as_str().unwrap().to_string();
                    let kind = o["kind"].as_str().unwrap().to_string();
                    let pos = o["pos"].as_array().unwrap();
                    let (gx, gy) = if world == "a" {
                        let flags = &o["flags"];
                        if kind == "door" && flags.get("open").and_then(|v| v.as_bool()).unwrap_or(false) { continue; }
                        gt_pixel_a(&[pos[0].as_f64().unwrap() as f32, pos[1].as_f64().unwrap() as f32], eye.scale)
                    } else {
                        let p = gt_pixel_b(&[pos[0].as_f64().unwrap() as f32, pos[1].as_f64().unwrap() as f32]);
                        if !gt_in_frame(p.0, p.1, 8.0) { continue; }
                        p
                    };
                    gt.push((id, kind, gx, gy));
                }
                let tol = if world == "a" { 8.0 } else { 22.0 };
                match_frame(&tracked.objects, &gt, tol, &mut last, &mut metrics);
                for (gid, kind, gx, gy) in &gt {
                    if let Some(&pid) = last.get(gid) {
                        if let Some(o) = tracked.objects.iter().find(|o| o.id == pid) {
                            if ((o.cx - gx).powi(2) + (o.cy - gy).powi(2)).sqrt() <= tol { vocab_eye.observe_vocab(o, kind); }
                        }
                    }
                }
            }
        }
    }
    let fps = nfr as f64 / t0.elapsed().as_secs_f64().max(1e-9);
    (metrics, fps, nfr, vocab_eye.vocab.report())
}
fn main() {
    let data_root = std::env::args().nth(1).unwrap_or_else(|| format!("{}/data/vision/v0", std::env::var("AMNI_HOME").unwrap_or_else(|_| ".".into())));
    let out_dir = std::env::args().nth(2).unwrap_or_else(|| format!("{}/docs/tasks/v20", std::env::var("AMNI_HOME").unwrap_or_else(|_| ".".into())));
    let root = Path::new(&data_root);
    let (ma, fa, na, va) = eval_world(&root.join("world_a"), "a");
    let (mb, fb, nb, vb) = eval_world(&root.join("world_b"), "b");
    let tp = ma.tp + mb.tp; let fp = ma.fp + mb.fp; let fn_ = ma.fn_ + mb.fn_; let ids = ma.id_switches + mb.id_switches;
    let p = if tp + fp == 0 { 0.0 } else { tp as f64 / (tp + fp) as f64 };
    let r = if tp + fn_ == 0 { 0.0 } else { tp as f64 / (tp + fn_) as f64 };
    let f1 = if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) };
    let fps = ((na + nb) as f64) / ((na as f64 / fa) + (nb as f64 / fb)).max(1e-9);
    println!("world_a P={:.4} R={:.4} F1={:.4} ids={} fps={:.1} tp={} fp={} fn={}", ma.precision(), ma.recall(), ma.f1(), ma.id_switches, fa, ma.tp, ma.fp, ma.fn_);
    println!("world_b P={:.4} R={:.4} F1={:.4} ids={} fps={:.1} tp={} fp={} fn={}", mb.precision(), mb.recall(), mb.f1(), mb.id_switches, fb, mb.tp, mb.fp, mb.fn_);
    println!("hold_combined P={:.4} R={:.4} F1={:.4} ids={} fps={:.1} tp={} fp={} fn={}", p, r, f1, ids, fps, tp, fp, fn_);
    let out = Path::new(&out_dir);
    fs::create_dir_all(out.join("reports")).ok();
    fs::create_dir_all(out.join("status")).ok();
    let report = format!("# M01 hold tracking eval\n\n| split | P | R | F1 | ID switches | fps | tp | fp | fn |\n|---|---|---|---|---|---|---|---|---|\n| world_a | {:.4} | {:.4} | {:.4} | {} | {:.1} | {} | {} | {} |\n| world_b | {:.4} | {:.4} | {:.4} | {} | {:.1} | {} | {} | {} |\n| hold combined | {:.4} | {:.4} | {:.4} | {} | {:.1} | {} | {} | {} |\n\nGates: object-level F1>=0.95 => **{}**; fps>=30 @128x128 CPU => **{}**.\n\nGT protocol: World A omits open doors from GT; World B keeps objects whose projected centers lie inside the 128x128 frame (margin +/-8px). Off-screen World B bodies are excluded.\n", ma.precision(), ma.recall(), ma.f1(), ma.id_switches, fa, ma.tp, ma.fp, ma.fn_, mb.precision(), mb.recall(), mb.f1(), mb.id_switches, fb, mb.tp, mb.fp, mb.fn_, p, r, f1, ids, fps, tp, fp, fn_, if f1 >= 0.95 { "PASS" } else { "FAIL" }, if fps >= 30.0 { "PASS" } else { "FAIL" });
    fs::write(out.join("reports/M01_hold_eval.md"), &report).unwrap();
    let mut vocab_md = String::from("# M01 visual vocabulary report\n\nPatch/object colour codes observed on hold, mapped to majority object kind.\n\n## World A\n\n| colour_code | kind | count |\n|---|---|---|\n");
    for (code, kind, n) in &va { vocab_md.push_str(&format!("| {} | {} | {} |\n", code, kind, n)); }
    vocab_md.push_str("\n## World B\n\n| colour_code | kind | count |\n|---|---|---|\n");
    for (code, kind, n) in &vb { vocab_md.push_str(&format!("| {} | {} | {} |\n", code, kind, n)); }
    vocab_md.push_str("\nCodes are `colour_code` from mean RGB (GF(17)-mixed nibble hash). Object GF(17) addresses use mean colour digits plus a position/area digit.\n");
    fs::write(out.join("reports/M01_visual_vocab.md"), &vocab_md).unwrap();
    println!("wrote reports under {}", out.join("reports").display());
}
