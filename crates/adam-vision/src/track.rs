use crate::hash::{colour_code, encode_frame, fg_mask_ex, kind_id, object_code, SlotTable, Vocab};
use crate::mip::MipPyramid;
use crate::{colour_dist, gf17_digits};
#[derive(Clone, Debug)]
pub struct Object { pub id: u32, pub cx: f32, pub cy: f32, pub area: u32, pub mean: [u8; 3], pub gf17: u32, pub digits: [u8; 4], pub colour_code: u16 }
#[derive(Clone, Debug)]
pub struct Tracked { pub frame: usize, pub objects: Vec<Object> }
pub struct TrackMetrics { pub tp: u64, pub fp: u64, pub fn_: u64, pub id_switches: u64, pub frames: u64 }
impl TrackMetrics {
    pub fn new() -> TrackMetrics { TrackMetrics { tp: 0, fp: 0, fn_: 0, id_switches: 0, frames: 0 } }
    pub fn precision(&self) -> f64 { let d = self.tp + self.fp; if d == 0 { 0.0 } else { self.tp as f64 / d as f64 } }
    pub fn recall(&self) -> f64 { let d = self.tp + self.fn_; if d == 0 { 0.0 } else { self.tp as f64 / d as f64 } }
    pub fn f1(&self) -> f64 { let (p, r) = (self.precision(), self.recall()); if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) } }
}
pub struct Eye {
    pub w: usize, pub h: usize, pub table: SlotTable, pub vocab: Vocab, pub tracks: Vec<(u32, f32, f32, [u8; 3], u32, u32)>, pub next_id: u32, pub prev: Option<Vec<u8>>, pub min_area: u32, pub max_area: u32, pub fg_thr: u32, pub match_dist: f32, pub scale: f32, pub colour_link: u32, pub reject_ramp: bool, pub coast: u32,
}
impl Eye {
    pub fn new(w: usize, h: usize) -> Eye {
        Eye { w, h, table: SlotTable::new(), vocab: Vocab::new(), tracks: vec![], next_id: 1, prev: None, min_area: 6, max_area: 800, fg_thr: 900, match_dist: 18.0, scale: 1.0, colour_link: 2500, reject_ramp: false, coast: 0 }
    }
    pub fn world_a() -> Eye {
        let mut e = Eye::new(128, 128); e.min_area = 8; e.max_area = 48; e.fg_thr = 600; e.match_dist = 8.0; e.scale = 4.0; e.colour_link = 1800; e.reject_ramp = false; e.coast = 0; e
    }
    pub fn world_b() -> Eye {
        let mut e = Eye::new(128, 128); e.min_area = 3; e.max_area = 2500; e.fg_thr = 80; e.match_dist = 30.0; e.scale = 1.0; e.colour_link = 6000; e.reject_ramp = true; e.coast = 8; e
    }
    pub fn reset(&mut self) { self.tracks.clear(); self.next_id = 1; self.prev = None; }
    fn pix(&self, rgb: &[u8], x: usize, y: usize) -> [u8; 3] { let i = (y * self.w + x) * 3; [rgb[i], rgb[i + 1], rgb[i + 2]] }
    fn components(&self, rgb: &[u8], mask: &[u8]) -> Vec<(f32, f32, u32, [u8; 3])> {
        let (w, h) = (self.w, self.h);
        let mut seen = vec![0u8; w * h];
        let mut out = vec![];
        let mut stack = vec![];
        for y0 in 0..h {
            for x0 in 0..w {
                let i0 = y0 * w + x0;
                if mask[i0] == 0 || seen[i0] != 0 { continue; }
                let seed = self.pix(rgb, x0, y0);
                stack.clear(); stack.push((x0, y0)); seen[i0] = 1;
                let (mut sx, mut sy, mut n, mut sum) = (0u64, 0u64, 0u32, [0u64; 3]);
                while let Some((x, y)) = stack.pop() {
                    let c = self.pix(rgb, x, y);
                    sx += x as u64; sy += y as u64; n += 1;
                    sum[0] += c[0] as u64; sum[1] += c[1] as u64; sum[2] += c[2] as u64;
                    for (dx, dy) in [(-1isize, 0), (1, 0), (0, -1), (0, 1)] {
                        let (nx, ny) = (x as isize + dx, y as isize + dy);
                        if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize { continue; }
                        let (nx, ny) = (nx as usize, ny as usize); let j = ny * w + nx;
                        if mask[j] == 0 || seen[j] != 0 { continue; }
                        let nc = self.pix(rgb, nx, ny);
                        if colour_dist(nc, seed) <= self.colour_link && colour_dist(nc, c) <= self.colour_link { seen[j] = 1; stack.push((nx, ny)); }
                    }
                }
                if n >= self.min_area && n <= self.max_area {
                    out.push((sx as f32 / n as f32, sy as f32 / n as f32, n, [(sum[0] / n as u64) as u8, (sum[1] / n as u64) as u8, (sum[2] / n as u64) as u8]));
                }
            }
        }
        out
    }
    pub fn step(&mut self, rgb: &[u8]) -> Tracked {
        assert_eq!(rgb.len(), self.w * self.h * 3);
        let _mip = MipPyramid::from_rgb(self.w, self.h, rgb);
        let _codes = encode_frame(rgb, self.prev.as_deref(), self.w, self.h, &mut self.table);
        let mask = fg_mask_ex(rgb, self.w, self.h, self.fg_thr, self.reject_ramp);
        let comps = self.components(rgb, &mask);
        let mut dets = Vec::with_capacity(comps.len());
        for (cx, cy, area, mean) in comps {
            dets.push((cx, cy, area, mean, colour_code(mean), object_code(mean, cx as u16, cy as u16, area)));
        }
        let mut used = vec![false; self.tracks.len()];
        let mut objects = vec![];
        let mut order: Vec<usize> = (0..dets.len()).collect();
        order.sort_by(|&a, &b| dets[b].2.cmp(&dets[a].2));
        let mut new_tracks = vec![];
        for di in order {
            let (cx, cy, area, mean, cc, gf) = dets[di];
            let mut best = None;
            for (ti, t) in self.tracks.iter().enumerate() {
                if used[ti] { continue; }
                let dist = ((t.1 - cx).powi(2) + (t.2 - cy).powi(2)).sqrt();
                let score = dist + colour_dist(t.3, mean) as f32 * 0.015;
                if dist <= self.match_dist && best.as_ref().map_or(true, |&(_, s): &(usize, f32)| score < s) { best = Some((ti, score)); }
            }
            let id = match best {
                Some((ti, _)) => { used[ti] = true; self.tracks[ti].0 }
                None => { let id = self.next_id; self.next_id += 1; id }
            };
            new_tracks.push((id, cx, cy, mean, area, 0));
            objects.push(Object { id, cx, cy, area, mean, gf17: gf, digits: gf17_digits(gf), colour_code: cc });
        }
        for (ti, t) in self.tracks.iter().enumerate() {
            if used[ti] { continue; }
            let age = t.5 + 1;
            if age <= self.coast { new_tracks.push((t.0, t.1, t.2, t.3, t.4, age)); }
        }
        self.tracks = new_tracks;
        self.prev = Some(rgb.to_vec());
        Tracked { frame: 0, objects }
    }
    pub fn observe_vocab(&mut self, obj: &Object, kind: &str) {
        let kid = kind_id(kind);
        self.vocab.observe(obj.colour_code as u32, kid, kind);
        self.table.bind_label(obj.gf17, kid);
    }
}
pub fn match_frame(pred: &[Object], gt: &[(String, String, f32, f32)], match_dist: f32, last_match: &mut std::collections::HashMap<String, u32>, metrics: &mut TrackMetrics) {
    let mut used_p = vec![false; pred.len()];
    for (gid, _kind, gx, gy) in gt.iter() {
        let mut best = None;
        for (pi, p) in pred.iter().enumerate() {
            if used_p[pi] { continue; }
            let d = ((p.cx - gx).powi(2) + (p.cy - gy).powi(2)).sqrt();
            if d <= match_dist && best.as_ref().map_or(true, |&(_, dd): &(usize, f32)| d < dd) { best = Some((pi, d)); }
        }
        if let Some((pi, _)) = best {
            used_p[pi] = true;
            let pid = pred[pi].id;
            if let Some(&prev) = last_match.get(gid) { if prev != pid { metrics.id_switches += 1; } }
            last_match.insert(gid.clone(), pid);
            metrics.tp += 1;
        } else {
            metrics.fn_ += 1;
        }
    }
    metrics.fp += used_p.iter().filter(|&&u| !u).count() as u64;
    metrics.frames += 1;
}
pub fn gt_pixel_a(pos: &[f32], scale: f32) -> (f32, f32) { (pos[0] * scale + scale * 0.5, pos[1] * scale + scale * 0.5) }
pub fn gt_pixel_b(pos: &[f32]) -> (f32, f32) { (pos[0] * 128.0 / 640.0, pos[1] * 128.0 / 400.0) }
pub fn gt_in_frame(gx: f32, gy: f32, margin: f32) -> bool { gx >= -margin && gy >= -margin && gx < 128.0 + margin && gy < 128.0 + margin }
