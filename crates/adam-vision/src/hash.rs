use crate::{colour_dist, gf17_digits, gf17_pack, is_bg, is_ramp, is_wall, PATCH, BG};
pub const SLOT_N: usize = 4096;
pub struct PatchCodes { pub colour: u16, pub edge: u16, pub motion: u16, pub combined: u32 }
pub struct SlotTable { pub colour: Vec<u32>, pub edge: Vec<u32>, pub motion: Vec<u32>, pub combo: Vec<u32>, pub label: Vec<u16> }
impl SlotTable {
    pub fn new() -> SlotTable { SlotTable { colour: vec![0; SLOT_N], edge: vec![0; SLOT_N], motion: vec![0; SLOT_N], combo: vec![0; SLOT_N], label: vec![0; SLOT_N] } }
    pub fn hit(&mut self, c: &PatchCodes) {
        self.colour[c.colour as usize % SLOT_N] = self.colour[c.colour as usize % SLOT_N].saturating_add(1);
        self.edge[c.edge as usize % SLOT_N] = self.edge[c.edge as usize % SLOT_N].saturating_add(1);
        self.motion[c.motion as usize % SLOT_N] = self.motion[c.motion as usize % SLOT_N].saturating_add(1);
        self.combo[c.combined as usize % SLOT_N] = self.combo[c.combined as usize % SLOT_N].saturating_add(1);
    }
    pub fn bind_label(&mut self, combined: u32, lab: u16) { let i = combined as usize % SLOT_N; if self.label[i] == 0 || self.combo[i] < 4 { self.label[i] = lab; } }
    pub fn lookup(&self, combined: u32) -> u16 { self.label[combined as usize % SLOT_N] }
}
pub struct Vocab { pub entries: Vec<(u32, u16, u32, String)> }
impl Vocab {
    pub fn new() -> Vocab { Vocab { entries: vec![] } }
    pub fn observe(&mut self, code: u32, kind_id: u16, kind: &str) {
        match self.entries.iter_mut().find(|e| e.0 == code && e.1 == kind_id) { Some(e) => e.2 = e.2.saturating_add(1), None => self.entries.push((code, kind_id, 1, kind.to_string())) }
    }
    pub fn report(&self) -> Vec<(u32, String, u32)> {
        let mut best: std::collections::BTreeMap<u32, (String, u32)> = std::collections::BTreeMap::new();
        for (code, _, cnt, kind) in &self.entries {
            match best.get_mut(code) { Some((_, c)) if *c >= *cnt => {}, Some(v) => *v = (kind.clone(), *cnt), None => { best.insert(*code, (kind.clone(), *cnt)); } }
        }
        best.into_iter().map(|(c, (k, n))| (c, k, n)).collect()
    }
}
fn mix(a: u32, b: u32) -> u32 { a.wrapping_mul(0x9E3779B1).wrapping_add(b).wrapping_mul(0x85EBCA77) ^ (a >> 13) }
pub fn patch_mean_hw(rgb: &[u8], w: usize, h: usize, x0: usize, y0: usize, pw: usize, ph: usize) -> [u8; 3] {
    let mut s = [0u32; 3]; let mut n = 0u32;
    for y in y0..(y0 + ph).min(h) { for x in x0..(x0 + pw).min(w) { let i = (y * w + x) * 3; s[0] += rgb[i] as u32; s[1] += rgb[i + 1] as u32; s[2] += rgb[i + 2] as u32; n += 1; } }
    if n == 0 { return BG; }
    [(s[0] / n) as u8, (s[1] / n) as u8, (s[2] / n) as u8]
}
pub fn colour_code(mean: [u8; 3]) -> u16 {
    let d = gf17_digits(mean[0] as u32 + mean[1] as u32 * 256 + mean[2] as u32 * 65536);
    let q = [mean[0] / 16, mean[1] / 16, mean[2] / 16];
    (((q[0] as u16) << 8) | ((q[1] as u16) << 4) | (q[2] as u16)) ^ (((d[0] as u16) << 12) | ((d[1] as u16) << 8) | ((d[2] as u16) << 4) | (d[3] as u16))
}
pub fn edge_code(rgb: &[u8], w: usize, h: usize, x0: usize, y0: usize, pw: usize, ph: usize) -> u16 {
    let mut gx = 0u32; let mut gy = 0u32;
    let y1 = (y0 + ph).min(h); let x1 = (x0 + pw).min(w);
    for y in y0..y1.saturating_sub(1) {
        for x in x0..x1.saturating_sub(1) {
            let i = (y * w + x) * 3; let ir = (y * w + x + 1) * 3; let id = ((y + 1) * w + x) * 3;
            let l = rgb[i] as i32 + rgb[i + 1] as i32 + rgb[i + 2] as i32;
            let r = rgb[ir] as i32 + rgb[ir + 1] as i32 + rgb[ir + 2] as i32;
            let d = rgb[id] as i32 + rgb[id + 1] as i32 + rgb[id + 2] as i32;
            gx += (r - l).unsigned_abs(); gy += (d - l).unsigned_abs();
        }
    }
    ((gx / 64) as u16 & 0xFF) << 8 | ((gy / 64) as u16 & 0xFF)
}
pub fn motion_code(cur: &[u8], prev: Option<&[u8]>, w: usize, h: usize, x0: usize, y0: usize, pw: usize, ph: usize) -> u16 {
    let Some(prev) = prev else { return 0 };
    let mut s = 0u32; let mut n = 0u32;
    for y in y0..(y0 + ph).min(h) { for x in x0..(x0 + pw).min(w) { let i = (y * w + x) * 3; s += colour_dist([cur[i], cur[i + 1], cur[i + 2]], [prev[i], prev[i + 1], prev[i + 2]]); n += 1; } }
    if n == 0 { return 0 }
    ((s / n.max(1)) as u16).min(65535)
}
pub fn encode_patch(rgb: &[u8], prev: Option<&[u8]>, w: usize, h: usize, x0: usize, y0: usize) -> PatchCodes {
    let mean = patch_mean_hw(rgb, w, h, x0, y0, PATCH, PATCH);
    let colour = colour_code(mean);
    let edge = edge_code(rgb, w, h, x0, y0, PATCH, PATCH);
    let motion = motion_code(rgb, prev, w, h, x0, y0, PATCH, PATCH);
    PatchCodes { colour, edge, motion, combined: mix(colour as u32, mix(edge as u32, motion as u32)) }
}
pub fn encode_frame(rgb: &[u8], prev: Option<&[u8]>, w: usize, h: usize, table: &mut SlotTable) -> Vec<PatchCodes> {
    let (nx, ny) = (w / PATCH, h / PATCH);
    let mut out = Vec::with_capacity(nx * ny);
    for py in 0..ny { for px in 0..nx { let c = encode_patch(rgb, prev, w, h, px * PATCH, py * PATCH); table.hit(&c); out.push(c); } }
    out
}
pub fn fg_mask(rgb: &[u8], w: usize, h: usize, thr: u32) -> Vec<u8> { fg_mask_ex(rgb, w, h, thr, false) }
pub fn fg_mask_ex(rgb: &[u8], w: usize, h: usize, thr: u32, reject_ramp: bool) -> Vec<u8> {
    let mut m = vec![0u8; w * h];
    for y in 0..h { for x in 0..w { let i = (y * w + x) * 3; let c = [rgb[i], rgb[i + 1], rgb[i + 2]]; let ok = !is_bg(c, thr) && !is_wall(c, thr * 2) && (!reject_ramp || !is_ramp(c, thr * 2)); m[y * w + x] = if ok { 1 } else { 0 }; } }
    m
}
pub fn object_code(mean: [u8; 3], cx: u16, cy: u16, area: u32) -> u32 {
    gf17_pack([(mean[0] / 15) % 17, (mean[1] / 15) % 17, (mean[2] / 15) % 17, ((cx as u32 ^ cy as u32 ^ area) % 17) as u8])
}
pub fn kind_id(name: &str) -> u16 {
    match name { "agent" => 1, "block" => 2, "ball" => 3, "lamp" => 4, "switch" => 5, "door" => 6, "ice" => 7, "fire" => 8, "box" => 9, "seesaw" => 10, "wall" => 11, "ramp" => 12, "water" => 13, _ => 0 }
}
