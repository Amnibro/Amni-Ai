pub struct MipPyramid { pub levels: Vec<(usize, usize, Vec<u8>)> }
impl MipPyramid {
    pub fn from_rgb(w: usize, h: usize, rgb: &[u8]) -> MipPyramid {
        assert_eq!(rgb.len(), w * h * 3);
        let mut levels = vec![(w, h, rgb.to_vec())];
        let (mut cw, mut ch, mut cur) = (w, h, rgb.to_vec());
        while cw >= 2 && ch >= 2 {
            let (nw, nh) = (cw / 2, ch / 2);
            let mut nxt = vec![0u8; nw * nh * 3];
            for y in 0..nh {
                for x in 0..nw {
                    let mut s = [0u32; 3];
                    for dy in 0..2 { for dx in 0..2 { let i = ((y * 2 + dy) * cw + (x * 2 + dx)) * 3; s[0] += cur[i] as u32; s[1] += cur[i + 1] as u32; s[2] += cur[i + 2] as u32; } }
                    let o = (y * nw + x) * 3; nxt[o] = (s[0] / 4) as u8; nxt[o + 1] = (s[1] / 4) as u8; nxt[o + 2] = (s[2] / 4) as u8;
                }
            }
            levels.push((nw, nh, nxt.clone()));
            cw = nw; ch = nh; cur = nxt;
        }
        MipPyramid { levels }
    }
    pub fn level(&self, i: usize) -> &(usize, usize, Vec<u8>) { &self.levels[i.min(self.levels.len() - 1)] }
    pub fn n_levels(&self) -> usize { self.levels.len() }
}
