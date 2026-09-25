pub mod hash;
pub mod mip;
pub mod track;
pub use hash::{PatchCodes, SlotTable, Vocab};
pub use mip::MipPyramid;
pub use track::{Eye, Object, TrackMetrics, Tracked};
pub const REFFELT_K4: [u32; 4] = [1, 17, 289, 4913];
pub const PATCH: usize = 8;
pub const BG: [u8; 3] = [240, 240, 245];
pub const WALL: [u8; 3] = [40, 40, 50];
pub const RAMP: [u8; 3] = [100, 100, 110];
pub fn gf17_digits(n: u32) -> [u8; 4] { [(n % 17) as u8, ((n / 17) % 17) as u8, ((n / 289) % 17) as u8, ((n / 4913) % 17) as u8] }
pub fn gf17_pack(d: [u8; 4]) -> u32 { d[0] as u32 + d[1] as u32 * 17 + d[2] as u32 * 289 + d[3] as u32 * 4913 }
pub fn colour_dist(a: [u8; 3], b: [u8; 3]) -> u32 { let dr = a[0] as i32 - b[0] as i32; let dg = a[1] as i32 - b[1] as i32; let db = a[2] as i32 - b[2] as i32; (dr * dr + dg * dg + db * db) as u32 }
pub fn is_bg(c: [u8; 3], thr: u32) -> bool { colour_dist(c, BG) <= thr }
pub fn is_wall(c: [u8; 3], thr: u32) -> bool { colour_dist(c, WALL) <= thr }
pub fn is_ramp(c: [u8; 3], thr: u32) -> bool { colour_dist(c, RAMP) <= thr }
#[cfg(test)]
mod tests {
use super::*;
use crate::hash::{encode_frame, fg_mask, SlotTable};
use crate::mip::MipPyramid;
use crate::track::Eye;
#[test]
fn gf17_roundtrip() {
let d = gf17_digits(12345);
assert_eq!(gf17_pack(d), 12345 % (17 * 17 * 17 * 17));
assert!(d.iter().all(|&x| x < 17));
}
#[test]
fn mip_halves() {
let w = 16usize; let h = 16usize; let rgb = vec![10u8; w * h * 3];
let m = MipPyramid::from_rgb(w, h, &rgb);
assert!(m.n_levels() >= 4);
assert_eq!(m.level(0).0, 16);
assert_eq!(m.level(1).0, 8);
}
#[test]
fn encode_hits_slots() {
let w = 16usize; let h = 16usize; let mut rgb = vec![240u8; w * h * 3];
for y in 4..12 { for x in 4..12 { let i = (y * w + x) * 3; rgb[i] = 30; rgb[i + 1] = 144; rgb[i + 2] = 255; } }
let mut t = SlotTable::new();
let codes = encode_frame(&rgb, None, w, h, &mut t);
assert_eq!(codes.len(), (w / PATCH) * (h / PATCH));
assert!(t.combo.iter().any(|&c| c > 0));
}
#[test]
fn world_a_finds_cell() {
let mut rgb = vec![240u8, 240, 245].repeat(128 * 128);
for y in 40..44 { for x in 40..44 { let i = (y * 128 + x) * 3; rgb[i] = 30; rgb[i + 1] = 144; rgb[i + 2] = 255; } }
let mut e = Eye::world_a();
let tr = e.step(&rgb);
assert_eq!(tr.objects.len(), 1);
assert!((tr.objects[0].cx - 41.5).abs() < 1.0);
assert!((tr.objects[0].cy - 41.5).abs() < 1.0);
let m = fg_mask(&rgb, 128, 128, 600);
assert_eq!(m.iter().filter(|&&v| v > 0).count(), 16);
}
#[test]
fn track_persists_id() {
let mut rgb = vec![240u8, 240, 245].repeat(128 * 128);
for y in 40..44 { for x in 40..44 { let i = (y * 128 + x) * 3; rgb[i] = 220; rgb[i + 1] = 20; rgb[i + 2] = 60; } }
let mut e = Eye::world_a();
let a = e.step(&rgb);
for y in 40..44 { for x in 40..44 { let i = (y * 128 + x) * 3; rgb[i] = 240; rgb[i + 1] = 240; rgb[i + 2] = 245; } }
for y in 42..46 { for x in 44..48 { let i = (y * 128 + x) * 3; rgb[i] = 220; rgb[i + 1] = 20; rgb[i + 2] = 60; } }
let b = e.step(&rgb);
assert_eq!(a.objects[0].id, b.objects[0].id);
}
}
