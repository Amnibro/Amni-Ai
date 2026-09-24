use crate::{cpu::{hm, init, update, wordch, St}, pack::{Geo, Lex, MAXB}};
use half::f16;
use rayon::prelude::*;
pub const NSET: u32 = 128;
pub const MS: usize = 512;
pub const NR: usize = 64;
pub const CG: f32 = 4.0;
pub const MG: f32 = 4.0;
pub const STOPW: &str = "the a an and or of to in on for with is are was were be been being do does did what why how whom which when where can could would should will shall may might must i you me my your it its this that these those there their they them he she his her we our us as at by from about into than then so if not no yes please tell give explain describe there here just also some any";
const SELFK: u32 = 1;
const ISINK: u32 = 3;
const WM: u32 = (1 << 20) - 1;
const BSOL: i32 = 1 << 20;
const BFRO: i32 = 1 << 19;
const BVOP: i32 = 1 << 18;
const BART: i32 = 1 << 17;
const BNEG: i32 = 1 << 16;
const BDO: i32 = 1 << 15;
const BWHE: i32 = 1 << 14;
const BWHO: i32 = 1 << 13;
const BWHEN: i32 = 1 << 12;
const BMETI: i32 = 1 << 11;
const BSKMU: i32 = 1 << 10;
const BSKNO: i32 = 1 << 9;
const BACO: i32 = 1 << 8;
const BKINL: i32 = 1 << 7;
const BHAV: i32 = 1 << 6;
const BPRE: i32 = 1 << 5;
const BENTK: i32 = 1 << 4;
const BCALL: i32 = 1 << 3;
const BSTUD: i32 = 1 << 2;
const BMOST: i32 = 1 << 1;
const BFAV: i32 = 1;
const BJOB: i32 = 1 << 21;
const BQLOCK: i32 = 1 << 22;
const BKEEP: i32 = BDO | BWHE | BWHO | BWHEN | BMETI | BQLOCK;
pub const fn bh(w: &[u8]) -> u32 { let (mut h, mut i) = (0u32, 0); while i < w.len() { h = h.wrapping_mul(83492791).wrapping_add(w[i] as u32 + 1); i += 1; } h }
const H_TO: u32 = bh(b"to");
const H_IN: u32 = bh(b"in");
const H_A: u32 = bh(b"a");
const H_HOW: u32 = bh(b"how");
const H_YEARS: u32 = bh(b"years");
const H_CRAZY: u32 = bh(b"crazy");
const H_ALL: u32 = bh(b"all");
const H_FAN: u32 = bh(b"fan");
const H_GO: u32 = bh(b"go");
const H_GOES: u32 = bh(b"goes");
const H_WENT: u32 = bh(b"went");
const H_GOING: u32 = bh(b"going");
const H_DEGREE: u32 = bh(b"degree");
const H_SCHOOL: u32 = bh(b"school");
const H_AROUND: u32 = bh(b"around");
const H_AT: u32 = bh(b"at");
const H_FROM: u32 = bh(b"from");
const H_MAJOR: u32 = bh(b"major");
const H_NOT: u32 = bh(b"not");
const H_LET: u32 = bh(b"let");
const H_TELL: u32 = bh(b"tell");
const H_ASK: u32 = bh(b"ask");
pub struct Lx19 { pub cls: Vec<i32>, pub rid: Vec<i32>, pub fun: Vec<u8>, pub stop: Vec<u8> }
impl Lx19 {
    pub fn new() -> Lx19 {
        let n = WM as usize + 1;
        let (mut cls, mut rid, mut fun, mut stop) = (vec![0i32; n], vec![0i32; n], vec![0u8; n], vec![0u8; n]);
        let both = |w: &str| { let b = w.as_bytes(); [Some(bh(b)), b[0].is_ascii_lowercase().then(|| bh(&[&[b[0] - 32], &b[1..]].concat()))] };
        let stops: std::collections::HashSet<&str> = STOPW.split(' ').chain("my i'm i is are was in at to of the a an and called named works as from actually oh well now really anyway okay ok yeah hey btw recently fun fact guess what though".split(' ')).collect();
        stops.iter().flat_map(|w| both(w)).flatten().for_each(|h| stop[(h & WM) as usize] = 1);
        "a an the and or of to in on for with as at by from into about over under is are was were be been being am do does did my your his her their you it its this that these those he she they them him we us actually oh well now really anyway okay ok yeah hey btw what why how when where who whom which can could would should will shall may might must just also some any there here so if not no yes please tell give than then fun fact guess though recently hi hello".split(' ').flat_map(both).flatten().for_each(|h| fun[(h & WM) as usize] = 1);
        let mut put = |w: &str, bit: i32, r: i32| both(w).into_iter().flatten().for_each(|h| { let i = (h & WM) as usize; cls[i] |= bit; rid[i] = if r != 0 { r } else { rid[i] }; });
        let lists = [("again just still year years trade while last every most way really lately recently home grad alum baby days day out back today ever night anything more down easily around away other always than seriously mostly new big whole own same very all even quite pretty quite super totally already never often usually sometimes ago together hands believe context record case curious wrong earlier few couple bit lots long several much many these those now please though then also too actually here there so if while during with without per via next each another any some fact know people family thing things stuff pretty stay stays crazy we wed older comes falls badly because", 1), ("wait sorry oops hang scratch let update correction honestly heads hi hey so yes no thanks please hmm ah wow ooh right alright cool nice great good ok okay and but or um uh yep yeah nope gotcha noted awesome perfect sure exactly absolutely definitely maybe perhaps anyway meanwhile also plus then oh actually well fun fact just", 2), ("a an that old my really just still very seriously mostly new big whole own same more", 4), ("did do does can could have has had would should", 8), ("say said tell told mention mentioned remember remembered recall recalled remind know", 16), ("sister mom aunt grandma niece daughter wife girlfriend stepmom stepsister godmother goddaughter brother dad uncle grandpa nephew son husband boyfriend stepdad stepbrother godfather godson cousin coworker boss landlord neighbor roommate bandmate classmate teammate mentor tutor partner friend manager dog cat parrot hamster rabbit ferret turtle gecko horse goldfish hedgehog iguana cockatiel chinchilla tortoise rat snake budgie buddy pal pig dragon law guinea father mother", 32), ("january february march april may june july august september october november december", 64), ("de del la el da di van von st", 128)];
        lists.iter().for_each(|(ws, bit)| ws.split(' ').for_each(|w| put(w, *bit, 0)));
        let noun = [("color", 10), ("colors", 10), ("colour", 10), ("food", 11), ("meal", 11), ("hobby", 7), ("hobbies", 7), ("job", 5), ("jobs", 5), ("car", 9), ("cars", 9), ("ride", 9), ("allergy", 8), ("allergies", 8), ("birthday", 15), ("born", 15), ("month", 15), ("age", 14), ("name", 6), ("school", 13), ("college", 13), ("university", 13), ("degree", 12), ("city", 4), ("town", 4), ("apartment", 4), ("anything", 11), ("everything", 11), ("trade", 5)];
        let wait = [("live", 4), ("lives", 4), ("lived", 4), ("living", 4), ("based", 4), ("base", 4), ("moved", 4), ("move", 4), ("moving", 4), ("work", 5), ("works", 5), ("worked", 5), ("working", 5), ("allergic", 8), ("react", 8), ("reacts", 8), ("reacted", 8), ("major", 12), ("majored", 12), ("graduate", 13), ("graduated", 13), ("turned", 14), ("turn", 14), ("went", 13), ("go", 13), ("goes", 13), ("going", 13)];
        let obj = [("like", 7), ("likes", 7), ("liked", 7), ("love", 7), ("loves", 7), ("loved", 7), ("drive", 9), ("drives", 9), ("drove", 9), ("driving", 9), ("eat", 11), ("eats", 11), ("eating", 11), ("order", 11), ("orders", 11), ("ordered", 11), ("studied", 12), ("study", 12), ("studying", 12), ("bought", 9), ("buy", 9), ("pick", 7), ("picks", 7), ("into", 7)];
        [(&noun[..], 256), (&wait[..], 512), (&obj[..], 1024)].iter().for_each(|(d, bit)| d.iter().for_each(|(w, r)| put(w, *bit, *r)));
        Lx19 { cls, rid, fun, stop }
    }
}
impl Default for Lx19 { fn default() -> Self { Self::new() } }
#[derive(Clone)]
pub struct Mem { wmk: [u32; MS], wmb: [[[u8; 16]; 2]; MS], wml: [[i32; 2]; MS], wmt: [i32; MS], wmv: [[i32; 2]; MS], wclk: i32, cwb: [u8; 16], flg: i32, pre: u32, subj: u32, meta: i32, ante: u32, own: u32, pbd: i32, inv: i32, pub ent: [u32; NR], pub rel: [u32; NR], pub val: [[u8; 24]; NR], pub vlen: [i32; NR], vtype: [i32; NR], pub stamp: [i32; NR], ep: i32, poss: i32, entc: u32, relc: u32, gapc: i32, vlc: i32, vb: [u8; 24], pub qent: u32, pub qrel: u32, pub qtyp: i32, ptag: i32, pub cand: [i32; 12], woff: [i32; 13], cnr: [i32; 13] }
fn lc(c: u8) -> i32 { let x = c as i32; if (65..=90).contains(&x) { x + 32 } else { x } }
impl Default for Mem { fn default() -> Self { Self::new() } }
impl Mem {
    pub fn new() -> Mem { Mem { wmk: [0; MS], wmb: [[[0; 16]; 2]; MS], wml: [[0; 2]; MS], wmt: [0; MS], wmv: [[0; 2]; MS], wclk: 0, cwb: [0; 16], flg: 0, pre: 0, subj: 0, meta: 0, ante: 0, own: 0, pbd: 0, inv: 0, ent: [0; NR], rel: [0; NR], val: [[0; 24]; NR], vlen: [0; NR], vtype: [0; NR], stamp: [0; NR], ep: 0, poss: 0, entc: 0, relc: 0, gapc: 0, vlc: 0, vb: [0; 24], qent: 0, qrel: 0, qtyp: 0, ptag: 0, cand: [-1; 12], woff: [0; 13], cnr: [0; 13] } }
    fn bit(&self, m: i32) -> bool { self.flg & m != 0 }
    fn bon(&mut self, m: i32) { self.flg |= m; }
    fn boff(&mut self, m: i32) { self.flg &= 2147483647 ^ m; }
    fn wm_clear(&mut self) { self.wmk.fill(0); self.wml.fill([0; 2]); self.wmt.fill(0); self.wmv.fill([0; 2]); }
    fn is_pron(&self, n: i32) -> bool {
        let [a, c, d, e, f] = std::array::from_fn(|i| lc(self.cwb[i]));
        (n == 2 && ((a == 104 && c == 101) || (a == 105 && c == 116))) || (n == 3 && ((a == 115 && c == 104 && d == 101) || (a == 104 && c == 105 && d == 115) || (a == 104 && c == 101 && d == 114))) || (n == 4 && a == 116 && c == 104 && d == 101 && e == 121) || (n == 5 && a == 116 && c == 104 && d == 101 && e == 105 && f == 114)
    }
    fn wm_find(&self, key: u32) -> i32 { if key == 0 { return -1; } let b = ((key % NSET) * 4) as usize; (0..4).fold(-1, |s, w| if self.wmk[b + w] == key { (b + w) as i32 } else { s }) }
    fn val_p4(&self, s: i32, vi: usize) -> u32 { if s < 0 { return 0; } let m = self.wml[s as usize][vi].min(4); (0..4).filter(|&i| i < m).fold(0u32, |h, i| h.wrapping_mul(83492791).wrapping_add(self.wmb[s as usize][vi][i as usize] as u32 + 1)) }
    fn next_byte(&self, s: i32, vi: usize, n: i32) -> i32 {
        if s < 0 { return -1; }
        let (su, ln) = (s as usize, self.wml[s as usize][vi]);
        if ln > 0 && n < ln { if (0..16).all(|i| i >= n || self.cwb[i as usize] == self.wmb[su][vi][i as usize]) { self.wmb[su][vi][n as usize] as i32 } else { -1 } } else { -1 }
    }
    fn bytes_eq(&self, s: i32, vi: usize, n: i32) -> bool { s >= 0 && n > 0 && n <= 16 && self.wml[s as usize][vi] == n && (0..n as usize).all(|i| self.cwb[i] == self.wmb[s as usize][vi][i]) }
    fn vs(&self, s: i32, vi: usize, qc: i32) -> i32 { if s >= 0 && self.wmv[s as usize][vi] <= qc { s } else { -1 } }
    fn weq(&self, n: i32, m: i32, cs: &[u8]) -> bool { n == m && (0..8).all(|i| i >= m || lc(self.cwb[i as usize]) == *cs.get(i as usize).unwrap_or(&0) as i32) }
    fn is_numw(&self, n: i32) -> bool { n > 0 && (0..16).all(|i| i >= n || self.cwb[i as usize].is_ascii_digit()) }
    fn is_capw(&self, n: i32) -> bool { n >= 2 && self.cwb[0].is_ascii_uppercase() }
    fn fact_reset(&mut self) { (self.poss, self.entc, self.relc, self.gapc, self.vlc, self.inv, self.meta, self.subj) = (0, 0, 0, 0, 0, 0, 0, 0); self.flg &= BKEEP; }
    fn ent_clr(&mut self) {
        (self.ent, self.rel, self.vlen, self.vtype, self.stamp) = ([0; NR], [0; NR], [0; NR], [0; NR], [0; NR]);
        (self.ep, self.qent, self.qrel, self.qtyp, self.ptag, self.ante, self.own, self.pbd, self.flg, self.pre, self.subj) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        self.fact_reset();
        (self.cand, self.woff, self.cnr) = ([-1; 12], [0; 13], [0; 13]);
    }
    fn tail(&self, cs: &[u8]) -> bool { let (l, n) = (self.vlc, cs.len() as i32); l > n && (0..n).all(|i| { let k = l - n + i; (0..24).contains(&k) && self.vb[k as usize] == cs[i as usize] }) }
    fn trim_val(&mut self) {
        for _ in 0..4 { if let Some(t) = [&b" last year"[..], b" this year", b" the most", b" these days", b" by the way", b" the", b" of", b" now", b" again"].iter().find(|t| self.tail(t)) { self.vlc -= t.len() as i32; } }
    }
    fn commit_fact(&mut self) {
        self.trim_val();
        let (mut rel, ent) = (self.relc, if self.relc == ISINK { self.subj } else { self.entc });
        if self.bit(BSOL) && self.vlc > 0 && rel == 0 && (self.bit(BFAV) || self.bit(BMOST)) { let mut bs = -1; for j in 0..NR { if self.ent[j] == ent && self.stamp[j] > bs && self.rel[j] >= 4 { rel = self.rel[j]; bs = self.stamp[j]; } } self.relc = rel; }
        if self.bit(BSOL) && rel == 7 && self.bit(BMOST) { let mut bs = -1; for j in 0..NR { if self.ent[j] == ent && self.stamp[j] > bs && (self.rel[j] == 10 || self.rel[j] == 11) { rel = self.rel[j]; bs = self.stamp[j]; } } self.relc = rel; }
        if self.bit(BSOL) && self.vlc > 0 && (3..=15).contains(&rel) && ent != 0 && (rel == ISINK || ent == SELFK || self.bit(BENTK)) {
            let (dig, c0, i) = ((0..24).all(|t| t >= self.vlc || self.vb[t as usize].is_ascii_digit()), self.vb[0], (self.ep % NR as i32) as usize);
            (self.ent[i], self.rel[i], self.vlen[i], self.vtype[i]) = (ent, rel, self.vlc, if dig { 2 } else if c0.is_ascii_uppercase() { 1 } else { 0 });
            self.val[i] = std::array::from_fn(|t| if (t as i32) < self.vlc { self.vb[t] } else { 0 });
            self.wclk += 1;
            self.stamp[i] = self.wclk;
            self.ep += 1;
        }
        self.fact_reset();
    }
    fn bump_q(&mut self, n: i32) {
        let (tag, w) = (self.ptag, |s: &[u8]| self.weq(n, s.len() as i32, s));
        let mut qt = self.qtyp;
        qt = if w(b"when") { qt.max(3) } else { qt };
        qt = if w(b"where") { qt.max(2) } else { qt };
        qt = if w(b"who") || w(b"name") || w(b"called") { qt.max(1) } else { qt };
        qt = if tag == 1 && (w(b"many") || w(b"old") || w(b"much")) { 3 } else { qt };
        qt = if tag == 2 && w(b"year") { 3 } else { qt };
        let nt = if w(b"how") { 1 } else if w(b"what") { 2 } else { 0 };
        (self.qtyp, self.ptag) = (qt, nt);
    }
    fn lhash(&self, n: i32) -> u32 { (0..16).filter(|&i| i < n).fold(0u32, |h, i| h.wrapping_mul(83492791).wrapping_add(lc(self.cwb[i as usize]) as u32 + 1)) }
    fn setrel(&mut self, r: u32, noun: bool, force: bool) {
        if force || noun || self.relc == 0 { self.relc = r; }
        if noun { self.qrel = r; self.bon(BQLOCK); } else if !self.bit(BQLOCK) { self.qrel = r; }
        if !self.bit(BSOL) && self.bit(BPRE) { self.vlc = 0; }
        self.boff(BJOB); self.boff(BHAV); self.boff(BPRE);
    }
    fn person(&mut self, e: u32, kin: bool) {
        if e == 0 { return; }
        if !self.bit(BFRO) && !self.bit(BSOL) { self.vlc = 0; }
        (self.entc, self.poss, self.inv) = (e, 0, 0);
        if e != SELFK { self.ante = e; self.own = e; } else { self.own = SELFK; }
        self.qent = e;
        if kin || e == SELFK { self.bon(BENTK) } else { self.boff(BENTK) }
    }
    fn push(&mut self, sep: Option<u8>, n: i32) {
        if let Some(c) = sep { if self.vlc > 0 && self.vlc < 24 { self.vb[self.vlc as usize] = c; self.vlc += 1; } }
        for i in 0..16 { if i < n && self.vlc < 24 { self.vb[self.vlc as usize] = self.cwb[i as usize]; self.vlc += 1; } }
    }
    fn addv(&mut self, n: i32, cap: bool, cls: i32) {
        let skip = self.bit(BFRO) || (!self.bit(BSOL) && (cls & 4 != 0 || cls & 1 != 0) && !cap) || (self.bit(BSOL) && cls & 1 != 0 && !cap);
        if skip { return; }
        self.push(Some(if self.pbd == 1 { 45 } else if self.pbd == 2 { 39 } else { 32 }), n);
        self.bon(BSOL); self.boff(BART); self.boff(BPRE);
        if self.bit(BJOB) && self.relc == 0 { self.setrel(if cap { 13 } else { 5 }, false, false); }
    }
    fn freeze(&mut self) { self.bon(BSOL); self.bon(BFRO); self.bon(BVOP); self.inv = 3; self.boff(BPRE); if self.entc == 0 || self.entc == self.subj { self.entc = 0; } }
    fn take_there(&mut self) {
        let (mut best, mut bs) = (-1i32, -1);
        for i in 0..NR { let st = self.stamp[i]; if st > bs && self.vlen[i] > 0 && self.val[i][0].is_ascii_uppercase() { best = i as i32; bs = st; } }
        if best < 0 { return; }
        let b = best as usize;
        self.vlc = 0;
        for t in 0..24 { if t < self.vlen[b] && self.vlc < 24 { self.vb[self.vlc as usize] = self.val[b][t as usize]; self.vlc += 1; } }
        self.bon(BSOL); self.bon(BFRO); self.bon(BVOP);
        if self.relc == 0 { self.relc = 4; }
    }
    fn step(&mut self, lx: &Lx19, n: i32, whw: u32, p4w: u32) -> i32 {
        let hw = self.lhash(n);
        let ix = (whw & WM) as usize;
        let (cls, fun, rid, cap, num, pre) = (lx.cls[ix], lx.fun[ix] as i32, lx.rid[ix], self.is_capw(n), self.is_numw(n), self.pre);
        let hd = cls & 32 != 0 || (n >= 3 && fun == 0 && cls & 2015 == 0);
        let [c0, c1, c2, c3] = std::array::from_fn(|i| lc(self.cwb[i]));
        let mut r = 1;
        let skw = (n == 3 && c0 == 121 && c1 == 111 && c2 == 117) || (n == 2 && c0 == 109 && c1 == 101) || (n == 5 && c0 == 97 && c1 == 98) || (n == 2 && c0 == 116 && c1 == 111) || (n == 4 && c0 == 116 && c1 == 104 && c2 == 97 && c3 == 116);
        if self.bit(BSKMU) && !skw { self.boff(BSKMU); }
        if self.meta == 1 && !(n == 1 && c0 == 105) && !(n == 3 && c0 == 121 && c1 == 111 && c2 == 117) { self.meta = 0; }
        if (self.meta == 2 || self.meta == 3) && cls & 16 == 0 { let was = self.meta; self.meta = 0; if was == 2 { self.person(SELFK, true); } }
        if self.meta == 8 && !((n == 2 && c0 == 109 && c1 == 101) || (n == 3 && c0 == 104 && c1 == 105 && c2 == 109) || (n == 3 && c0 == 104 && c1 == 101 && c2 == 114) || (n == 4 && c0 == 116 && c1 == 104 && c2 == 101 && c3 == 109)) { self.meta = 0; }
        if self.meta >= 9 { self.meta = 0; if hd && self.pbd != 2 { self.poss = 1; self.own = if self.ante != 0 { self.ante } else { self.own }; } else { self.person(if self.ante != 0 { self.ante } else { self.entc }, true); } }
        let (bsol, bfro, bvop, bstud) = (|m: &Mem| m.bit(BSOL), |m: &Mem| m.bit(BFRO), |m: &Mem| m.bit(BVOP), |m: &Mem| m.bit(BSTUD));
        if self.bit(BACO) && (hw == H_NOT || (n == 3 && c0 == 110 && c1 == 111 && c2 == 116)) { self.bon(BSKNO); self.boff(BACO); }
        else if self.bit(BACO) { self.boff(BACO); }
        else if self.bit(BSKNO) { r = 1; }
        else if self.meta == 4 { self.meta = 0; }
        else if n == 6 && c0 == 114 && c1 == 101 && c2 == 109 { self.meta = 4; }
        else if self.bit(BSKMU) { self.boff(BSKMU); }
        else if self.meta == 1 && n == 1 && c0 == 105 { self.meta = 2; }
        else if self.meta == 1 && n == 3 && c0 == 121 && c1 == 111 && c2 == 117 { self.meta = 3; }
        else if (self.meta == 2 || self.meta == 3) && cls & 16 != 0 { self.bon(BSKMU); if self.meta == 2 { self.bon(BMETI); } self.meta = 0; }
        else if self.meta == 5 { r = 1; }
        else if n == 2 && c0 == 105 && c1 == 102 && !bsol(self) { self.meta = 6; }
        else if self.meta == 6 && n == 3 && c0 == 121 { self.meta = 7; }
        else if self.meta == 6 { self.meta = 0; }
        else if self.meta == 7 && n == 3 && c0 == 97 && c1 == 115 && c2 == 107 { self.meta = 8; }
        else if self.meta == 7 && (cls & 16 != 0 || self.pbd == 2) { self.meta = 5; }
        else if self.meta == 7 { self.meta = 0; }
        else if self.meta == 8 && n == 2 && c0 == 109 && c1 == 101 { self.person(SELFK, true); self.meta = 0; }
        else if self.meta == 8 && (n == 3 || n == 4) { self.person(self.ante, true); self.meta = 0; }
        else if self.pbd == 2 && bsol(self) && !bfro(self) { self.push(Some(39), n); }
        else if self.pbd == 2 && n == 1 && c0 == 116 { self.bon(BNEG); }
        else if self.pbd == 2 && n == 1 && (c0 == 109 || c0 == 114 || (c0 == 115 && self.poss == 0)) { self.gapc = 1; self.inv = if self.inv == 1 { 2 } else { self.inv }; if self.bit(BPRE) && self.vlc > 0 { self.bon(BSOL); self.bon(BFRO); self.bon(BVOP); self.boff(BPRE); } }
        else if self.pbd == 2 && n == 1 && c0 == 115 && self.poss != 0 { self.gapc = 1; self.own = if self.entc != 0 { self.entc } else { self.own }; }
        else if self.pbd == 2 && n == 1 { r = 1; }
        else if cls & 2 != 0 && !bsol(self) && !bvop(self) && !(n == 3 && c0 == 102) && !(n == 2 && c0 == 105 && c1 == 110) { r = 1; }
        else if n == 2 && c0 == 109 && c1 == 121 && self.relc == 9 && pre == H_TO && !bsol(self) { r = 1; }
        else if (n == 2 && c0 == 109 && c1 == 121) || (n == 4 && c0 == 121 && c1 == 111 && c2 == 117 && c3 == 114) {
            if self.inv == 2 && self.vlc > 0 { self.freeze(); }
            if bsol(self) && !bfro(self) && self.relc == 0 { self.bon(BFRO); } else if bsol(self) && !bfro(self) { self.commit_fact(); }
            self.poss = 1;
            self.own = if n == 2 { SELFK } else { self.ante };
            if n == 2 && self.qent == 0 { self.qent = SELFK; }
        }
        else if n == 1 && c0 == 105 { if bsol(self) && !bfro(self) && self.relc == 0 { self.bon(BFRO); } else if bsol(self) && !bfro(self) { self.commit_fact(); } self.person(SELFK, true); }
        else if n == 2 && c0 == 109 && c1 == 101 && (pre == H_LET || pre == H_TELL || pre == H_ASK) { r = 1; }
        else if n == 2 && c0 == 109 && c1 == 101 { if bsol(self) && self.relc == 0 { self.bon(BFRO); } self.person(SELFK, true); }
        else if (n == 2 && c0 == 104 && c1 == 101) || (n == 3 && c0 == 115 && c1 == 104 && c2 == 101) || (n == 4 && c0 == 116 && c1 == 104 && c2 == 101 && c3 == 121) || (n == 2 && c0 == 105 && c1 == 116) || (n == 3 && c0 == 104 && c1 == 105 && c2 == 109) { let e = if n == 2 && c0 == 105 && self.own != 0 { self.own } else { self.ante }; if e != 0 { self.person(e, true); } }
        else if (n == 3 && c0 == 104 && c2 == 114) || (n == 3 && c0 == 104 && c1 == 105 && c2 == 115) || (n == 5 && c0 == 116 && c1 == 104 && c2 == 101 && c3 == 105) { if self.inv == 2 && self.vlc > 0 { self.freeze(); } self.meta = if n == 3 && c2 == 114 { 9 } else if n == 3 { 10 } else { 11 }; }
        else if num { if self.relc == 0 || self.relc == 14 { self.setrel(14, true, false); } self.addv(n, false, 0); }
        else if n == 3 && c0 == 111 && c1 == 108 && c2 == 100 { if pre == H_HOW || pre == H_YEARS || (!bvop(self) && !bsol(self)) { self.setrel(14, true, false); } }
        else if cls & 512 != 0 && rid == 4 && pre == H_A && n >= 6 { self.setrel(5, true, false); self.poss = 0; }
        else if (n == 4 && c0 == 99 && c1 == 97 && c2 == 108 && c3 == 108) || (n == 5 && c0 == 99 && c3 == 108 && c1 == 97) { self.vlc = 0; self.inv = 0; self.poss = 0; self.setrel(6, false, false); self.bon(BCALL); }
        else if (bsol(self) || bvop(self)) && !bfro(self) && n == 4 && c0 == 119 && c1 == 111 && c2 == 114 && c3 == 107 && pre != H_TO { self.addv(n, false, cls); }
        else if (n == 6 && c0 == 99 && c1 == 97 && c2 == 108 && c3 == 108) || (n == 5 && c0 == 110 && c1 == 97 && c2 == 109 && c3 == 101) { self.setrel(6, n == 5, false); self.poss = 0; if self.gapc == 1 { self.bon(BVOP) } else { self.bon(BCALL) } }
        else if n == 4 && c0 == 104 && c1 == 111 && c2 == 109 && c3 == 101 {
            if self.inv == 2 && self.vlc > 0 { self.freeze(); self.setrel(4, false, false); r = 0; }
            else if self.bit(BPRE) || self.bit(BCALL) || self.inv == 1 { self.bon(BSOL); self.bon(BFRO); self.bon(BVOP); self.relc = 4; }
            else if self.relc == 0 { self.setrel(4, false, false); }
            if !self.bit(BQLOCK) { self.qrel = 4; }
        }
        else if n == 5 && c0 == 119 && c1 == 104 && c2 == 101 && c3 == 110 && bsol(self) { r = 1; }
        else if n == 5 && c0 == 119 && c1 == 104 && c2 == 101 && c3 == 114 && self.inv == 2 && self.vlc > 0 { self.freeze(); self.setrel(4, false, false); self.bon(BWHE); r = 0; }
        else if n == 5 && c0 == 119 && c1 == 104 && c2 == 101 && c3 == 114 { self.bon(BWHE); self.setrel(4, false, false); }
        else if n == 4 && c0 == 119 && c1 == 104 && c2 == 101 && c3 == 110 { self.bon(BWHEN); self.setrel(15, false, false); }
        else if n == 3 && c0 == 119 && c1 == 104 && c2 == 111 { self.bon(BWHO); self.setrel(6, false, false); }
        else if cls & 64 != 0 && cap && self.entc == 0 && self.gapc == 0 && !bvop(self) && !self.bit(BART) && self.inv == 0 { self.vlc = 0; self.push(None, n); self.subj = p4w; self.inv = 1; }
        else if cls & 64 != 0 && (self.bit(BART) || self.gapc == 1 || bvop(self) || (self.entc != 0 && (self.relc == 0 || self.relc == 15 || self.relc == 4))) { if self.relc == 0 || self.relc == 15 || (self.relc == 4 && !bsol(self)) { self.setrel(15, true, false); } self.addv(n, cap, 0); }
        else if bstud(self) && !bfro(self) && n > 2 && fun == 0 && cls & 2 == 0 && cls & 8 == 0 && !(n == 6 && c0 == 115 && c1 == 99) && !(n == 7 && c0 == 99 && c1 == 111 && c2 == 108) && !(n == 10 && c0 == 117) { self.addv(n, cap, 0); }
        else if cls & 256 != 0 && (n == 6 || n == 7 || n == 10) && (bstud(self) || self.relc == 12 || self.relc == 13) && (pre == H_AT || pre == H_IN || pre == H_FROM || pre == H_TO) { r = 1; }
        else if cls & 256 != 0 && (n == 6 || n == 7) && self.entc == 0 && !bsol(self) && pre == H_IN { r = 1; }
        else if cls & 256 != 0 && self.qrel == 12 && pre == H_IN && !bstud(self) { r = 1; }
        else if cls & 256 != 0 && cap && bsol(self) && (n == 4 || n == 6 || n == 7 || n == 10) { self.addv(n, true, 0); }
        else if cls & 256 != 0 {
            let own = if self.own != 0 { self.own } else { SELFK };
            if self.poss != 0 { self.person(own, true); }
            if (rid == 10 || rid == 11) && self.bit(BFAV) && !bsol(self) {
                let pe = if self.entc != 0 { self.entc } else if self.own != 0 { self.own } else { self.ante };
                let (mut bs, mut bi) = (-1, -1i32);
                for j in 0..NR { if pe != 0 && self.ent[j] == pe && self.rel[j] == 7 && self.stamp[j] > bs { bi = j as i32; bs = self.stamp[j]; } }
                if bi >= 0 { self.rel[bi as usize] = rid as u32; }
            }
            self.setrel(rid as u32, true, false);
            self.poss = 0;
        }
        else if cls & 512 != 0 { let rr = if n >= 6 && pre == H_A && rid == 4 { 5 } else { rid }; if n == 5 && c0 == 109 { self.bon(BSTUD); } self.setrel(rr as u32, false, false); self.poss = 0; }
        else if cls & 1024 != 0 { if n >= 5 && c0 == 115 && c1 == 116 { self.bon(BSTUD); } self.setrel(rid as u32, false, rid == 11 && self.relc == 7); self.poss = 0; self.bon(BVOP); self.boff(BCALL); }
        else if n == 3 && c0 == 102 && c1 == 97 && c2 == 110 { self.setrel(7, false, false); self.poss = 0; }
        else if n == 4 && c0 == 98 && c1 == 97 && c2 == 98 && c3 == 121 { self.setrel(15, true, false); }
        else if (n == 4 && c0 == 103 && c1 == 114) || (n == 4 && c0 == 97 && c1 == 108 && c2 == 117) { self.setrel(13, true, false); }
        else if n == 6 && c0 == 97 && c1 == 114 && (self.bit(BNEG) || self.relc == 8) { self.setrel(8, false, false); self.bon(BVOP); self.poss = 0; }
        else if n == 4 && c0 == 97 && c1 == 119 && c2 == 97 && c3 == 121 { self.setrel(8, false, false); }
        else if cls & 8 != 0 && !bsol(self) && !bvop(self) && self.gapc == 0 && !(n == 3 && c0 == 104 && c1 == 97 && c2 == 100) {
            let hv = (n == 4 && c0 == 104 && c1 == 97 && c2 == 118) || (n == 3 && c0 == 104 && c1 == 97 && c2 == 115);
            let dw = (n == 2 && c0 == 100) || (n == 3 && c0 == 100 && c1 == 105) || (n == 4 && c0 == 100 && c1 == 111);
            if hv && (self.entc != 0 || self.poss != 0) { self.bon(BHAV); self.poss = 0; } else { if dw { self.bon(BDO); } self.meta = 1; }
        }
        else if ((n == 3 && c0 == 103 && c1 == 111 && c2 == 116) || (n == 4 && c0 == 104 && c1 == 97 && c2 == 118 && c3 == 101) || (n == 6 && c0 == 103 && c1 == 111 && c2 == 116 && c3 == 116)) && self.entc == 0 && !bsol(self) && self.poss == 0 { self.meta = 1; }
        else if (n == 3 && c0 == 103 && c1 == 111 && c2 == 116) || (n == 4 && c0 == 104 && c1 == 97 && c2 == 118 && c3 == 101) || (n == 3 && c0 == 104 && c1 == 97 && c2 == 115) || (n == 3 && c0 == 104 && c1 == 97 && c2 == 100) || (n == 6 && c0 == 103 && c1 == 111 && c2 == 116 && c3 == 116) { self.bon(BHAV); self.poss = 0; }
        else if n == 4 && c0 == 98 && c1 == 101 && c2 == 101 && c3 == 110 { self.boff(BHAV); self.gapc = 1; }
        else if self.bit(BART) && self.bit(BHAV) && cls & 32 != 0 && !cap { self.person(p4w, true); self.boff(BHAV); self.boff(BART); }
        else if n == 4 && c0 == 109 && c1 == 111 && c2 == 115 && c3 == 116 { self.bon(BMOST); }
        else if n == 8 && c0 == 102 && c1 == 97 && c2 == 118 { self.bon(BFAV); }
        else if self.poss != 0 && ((n == 3 && c0 == 98 && c1 == 105) || (n == 4 && c0 == 98 && c1 == 101) || (n == 6 && c0 == 108) || (n == 3 && c0 == 110 && c1 == 101 && c2 == 119)) { r = 1; }
        else if self.poss != 0 && self.pbd == 1 && n == 2 && c0 == 105 && c1 == 110 { r = 1; }
        else if self.poss != 0 && self.pbd == 1 && hd { self.person(p4w, cls & 32 != 0); self.poss = 1; if cls & 32 != 0 { self.bon(BKINL); } }
        else if self.poss != 0 && hd && self.bit(BKINL) && cls & 32 == 0 && self.pbd != 1 { self.poss = 0; r = 0; }
        else if self.poss != 0 && hd { self.person(p4w, cls & 32 != 0); self.poss = 1; if cls & 32 != 0 { self.bon(BKINL); } }
        else if (n == 2 && c0 == 105 && c1 == 115) || (n == 3 && c0 == 97 && c1 == 114 && c2 == 101) || (n == 3 && c0 == 119 && c1 == 97 && c2 == 115) || (n == 4 && c0 == 119 && c1 == 101 && c2 == 114 && c3 == 101) || (n == 2 && c0 == 97 && c1 == 109) { self.gapc = 1; if self.inv == 1 { self.inv = 2; } if self.bit(BPRE) && self.vlc > 0 { self.bon(BSOL); self.bon(BFRO); self.bon(BVOP); self.boff(BPRE); } }
        else if (n == 1 && c0 == 97) || (n == 2 && c0 == 97 && c1 == 110) { self.bon(BART); if (self.gapc == 1 || self.bit(BHAV)) && self.relc == 0 { self.bon(BJOB); } if self.gapc == 1 || self.relc != 0 || self.entc != 0 { self.bon(BVOP); } }
        else if self.inv == 2 && self.vlc > 0 && ((n == 4 && c0 == 111 && c1 == 118 && c2 == 101 && c3 == 114) || (n == 3 && c0 == 97 && c1 == 108 && c2 == 108) || (n == 3 && c0 == 116 && c1 == 104 && c2 == 101) || (n == 3 && c0 == 119 && c1 == 97 && c2 == 121)) { r = 1; }
        else if self.inv == 2 && self.vlc > 0 && n == 2 && c0 == 105 && c1 == 110 {
            let (mut fw, mut m) = (0u32, 0);
            for i in 0..4 { if i < self.vlc && self.vb[i as usize] != 32 && m < 4 { fw = fw.wrapping_mul(83492791).wrapping_add(self.vb[i as usize] as u32 + 1); m += 1; } }
            (self.subj, self.entc, self.relc, self.vlc) = (fw, fw, 3, 0);
            self.boff(BSOL); self.bon(BVOP); self.boff(BFRO); self.inv = 0;
        }
        else if self.inv == 2 && self.vlc > 0 { self.freeze(); r = 0; }
        else if self.inv == 1 && (cap || (cls & 128 != 0 && self.vlc > 0)) { self.push(Some(32), n); if cap && self.subj == 0 { self.subj = p4w; } }
        else if n == 2 && c0 == 105 && c1 == 110 && bsol(self) { self.addv(n, false, 0); }
        else if n == 4 && c0 == 105 && c1 == 110 && c2 == 116 && c3 == 111 { self.setrel(7, false, true); self.bon(BVOP); }
        else if n == 4 && c0 == 116 && c1 == 104 && c2 == 97 && c3 == 110 && self.relc == 0 { self.setrel(11, false, true); self.bon(BVOP); }
        else if n == 5 && c0 == 97 && c1 == 98 && pre == H_CRAZY { self.setrel(11, true, false); self.bon(BVOP); }
        else if n == 5 && c0 == 97 && c1 == 98 && pre == H_ALL { self.setrel(7, false, true); self.bon(BVOP); }
        else if n == 2 && c0 == 111 && c1 == 102 && pre == H_FAN { self.setrel(7, false, true); self.bon(BVOP); }
        else if n == 2 && c0 == 98 && c1 == 121 && (pre == H_GO || pre == H_GOES) { self.setrel(6, true, false); self.bon(BVOP); self.boff(BSOL); self.vlc = 0; }
        else if n == 2 && c0 == 105 && c1 == 110 && (pre == H_DEGREE || pre == H_MAJOR) { self.setrel(12, true, false); self.bon(BVOP); }
        else if ((n == 2 && c0 == 97 && c1 == 116) || (n == 4 && c0 == 102 && c1 == 114 && c2 == 111)) && self.relc == 12 && !bsol(self) { self.setrel(13, true, false); self.bon(BVOP); }
        else if n == 3 && c0 == 102 && c1 == 111 && c2 == 114 && pre == H_SCHOOL { self.setrel(12, true, false); self.boff(BSOL); self.vlc = 0; self.bon(BVOP); }
        else if n == 2 && c0 == 116 && c1 == 111 && (pre == H_WENT || pre == H_GO || pre == H_GOING) { self.setrel(13, false, false); self.bon(BVOP); }
        else if n == 2 && c0 == 105 && c1 == 110 && pre == H_AROUND { self.setrel(9, false, true); self.bon(BVOP); }
        else if n == 3 && c0 == 102 && c1 == 111 && c2 == 114 && pre == H_IN { self.setrel(9, false, true); self.bon(BVOP); self.boff(BSOL); self.vlc = 0; }
        else if n == 2 && c0 == 105 && c1 == 110 && self.relc == 0 && self.entc != 0 && self.inv == 0 && !bsol(self) { self.setrel(4, false, false); self.bon(BVOP); }
        else if (n == 3 && c0 == 102 && c1 == 111 && c2 == 114) || (n == 2 && c0 == 105 && c1 == 110) || (n == 2 && c0 == 116 && c1 == 111) || (n == 4 && c0 == 102 && c1 == 114) || (n == 5 && c0 == 97 && c1 == 98) || (n == 2 && c0 == 111 && c1 == 102) || (n == 4 && c0 == 105 && c1 == 110) || (n == 2 && c0 == 97 && c1 == 115) || (n == 2 && c0 == 98 && c1 == 121) {
            if bsol(self) && bvop(self) && !bfro(self) && ((n == 2 && c0 == 105 && c1 == 110) || (n == 2 && c0 == 111 && c1 == 102) || (n == 4 && c0 == 105 && c1 == 110)) { self.addv(n, false, 0); }
            else if bsol(self) { self.boff(BVOP); }
            else if self.entc == 0 && self.inv == 0 && !bsol(self) && !bvop(self) && self.relc == 0 { r = 1; }
            else if !bfro(self) && !bsol(self) { if self.bit(BPRE) { self.vlc = 0; self.boff(BPRE); } self.bon(BVOP); }
            if self.inv == 1 { self.inv = 0; }
        }
        else if (n == 5 && c0 == 116 && c1 == 104 && c2 == 101 && c3 == 114) || (n == 4 && c0 == 104 && c1 == 101 && c2 == 114 && c3 == 101) { if self.entc != 0 && !bsol(self) { self.take_there(); } }
        else if self.bit(BHAV) && cap && !bfro(self) && (self.relc == 0 || self.relc == 9) { self.setrel(9, false, false); self.addv(n, true, 0); }
        else if self.bit(BCALL) && cap && !bfro(self) { if self.relc == 0 { self.setrel(6, false, false); } self.addv(n, true, 0); }
        else if n == 3 && c0 == 116 && c1 == 104 && c2 == 101 && bvop(self) && !bsol(self) && self.relc != 3 { self.addv(n, false, 0); }
        else if self.entc != 0 && self.poss == 0 && !bsol(self) && self.gapc == 0 && !bvop(self) && !bfro(self) && !self.bit(BCALL) && fun == 0 && cls & 1 == 0 && cls & 2 == 0 && cls & 8 == 0 && hd { if self.vlc == 0 { self.push(None, n); } self.bon(BPRE); }
        else if n == 5 && c0 == 102 && c1 == 97 && !bsol(self) && !bvop(self) { r = 1; }
        else if bsol(self) && !bfro(self) && n == 3 && c0 == 97 && c1 == 110 && c2 == 100 { self.push(Some(32), n); }
        else if bsol(self) && bvop(self) && !bfro(self) && ((n == 3 && c0 == 116 && c1 == 104 && c2 == 101) || (n == 3 && c0 == 97 && c1 == 110 && c2 == 100) || (n == 2 && c0 == 111 && c1 == 102) || (n == 2 && c0 == 105 && c1 == 110)) { self.addv(n, false, 0); }
        else if n == 5 && c0 == 101 && c1 == 118 && bsol(self) && self.relc == 0 { self.setrel(11, true, false); }
        else if (bvop(self) || self.gapc == 1 || self.bit(BCALL)) && !bfro(self) && (fun == 0 || cap) && (cls & (1 | 2 | 8) == 0 || cap) { if cap && self.gapc == 1 && !self.bit(BART) && (self.relc == 0 || self.relc == 6) { self.setrel(6, false, false); } self.addv(n, cap, cls); }
        else if bsol(self) && bvop(self) && !bfro(self) && cls & 1 == 0 && cls & 2 == 0 && cls & 8 == 0 && fun == 0 { self.addv(n, cap, cls); }
        else if cap && self.entc == 0 && self.poss == 0 && fun == 0 { self.vlc = 0; self.push(None, n); self.subj = p4w; self.inv = 1; }
        else if self.bit(BPRE) && cap { self.push(Some(32), n); }
        else { r = 1; }
        r
    }
    fn on_user_word(&mut self, lx: &Lx19, p4w: u32, n: i32, whw: u32) { self.bump_q(n); for _ in 0..2 { if self.step(lx, n, whw, p4w) == 1 { break; } } self.pre = self.lhash(n); }
    fn finish_q(&mut self) {
        if self.qent == 0 && self.bit(BMETI) { self.qent = SELFK; }
        self.qrel = if self.qrel != 0 { self.qrel } else if self.bit(BWHO) { 6 } else if self.bit(BWHEN) { 15 } else if self.bit(BWHE) { 4 } else if self.bit(BDO) { 5 } else { 0 };
    }
    fn on_punct(&mut self, bb: u32) {
        let end = bb == 46 || bb == 33 || bb == 63 || bb == 10;
        if self.meta == 5 && bb == 44 { self.meta = 0; }
        else if self.bit(BSKNO) && end { self.boff(BSKNO); self.fact_reset(); }
        else if bb == 44 && !self.bit(BSOL) { self.poss = 0; self.boff(BKINL); self.boff(BVOP); self.gapc = 0; self.inv = 0; self.vlc = 0; self.boff(BPRE); }
        else if bb == 63 && !self.bit(BSOL) { self.finish_q(); self.boff(BVOP); self.gapc = 0; self.inv = 0; self.vlc = 0; }
        else if bb == 44 { self.commit_fact(); self.bon(BACO); }
        else if end { if bb == 63 { self.finish_q(); } self.commit_fact(); }
    }
    fn ent_p4(&self, i: usize) -> u32 { let (mut h, mut n) = (0u32, 0); for t in 0..4 { if t < self.vlen[i] && self.val[i][t as usize] != 32 && n < 4 { h = h.wrapping_mul(83492791).wrapping_add(self.val[i][t as usize] as u32 + 1); n += 1; } } h }
    fn next_ent(&self, i: i32, off: i32, n: i32) -> i32 {
        if !(i >= 0 && off >= 0 && off + n < self.vlen[i as usize] && off + n < 24) { return -1; }
        let v = &self.val[i as usize];
        if (0..16).all(|t| t >= n || (off + t < 24 && self.cwb[t as usize] == v[(off + t) as usize])) { v[(off + n) as usize] as i32 } else { -1 }
    }
    fn adv_woff(&mut self, n: i32) {
        for k in 0..12 {
            let (i, off) = (self.cand[k], self.woff[k]);
            if !(i >= 0 && n > 0 && off >= 0) { continue; }
            let (v, ln) = (&self.val[i as usize], self.vlen[i as usize]);
            let ok = off + n <= ln && off < 24 && (0..16).all(|t| t >= n || (off + t < 24 && self.cwb[t as usize] == v[(off + t) as usize]));
            let sp = ok && off + n < ln && off + n < 24 && v[(off + n) as usize] == 32;
            self.woff[k] = if sp { off + n + 1 } else { 0 };
        }
    }
    fn newest_rel(&self, i: i32, qc: i32) -> i32 { if i < 0 { return 0; } let (st, e, rv) = (self.stamp[i as usize], self.ent[i as usize], self.rel[i as usize]); (!(0..NR).any(|j| self.ent[j] == e && self.rel[j] == rv && self.stamp[j] > st && self.stamp[j] <= qc && self.stamp[j] > 0)) as i32 }
    fn pick_cands(&mut self, qc: i32) {
        let (mut prev, qe) = (2147483647, self.qent);
        for k in 0..8 {
            let (mut best, mut bs) = (-1i32, -1);
            for i in 0..NR { let st = self.stamp[i]; if qe != 0 && self.ent[i] == qe && st > 0 && st <= qc && st < prev && st > bs && (self.qrel == 0 || self.rel[i] == self.qrel) { best = i as i32; bs = st; } }
            self.cand[k] = best;
            prev = if best >= 0 { bs } else { -1 };
            self.cnr[k] = self.newest_rel(best, qc);
        }
        for k in 0..4 {
            let (src, mut best, mut bs) = (self.cand[k], -1i32, -1);
            if src >= 0 { let vp = self.ent_p4(src as usize); for i in 0..NR { let st = self.stamp[i]; if i as i32 != src && vp != 0 && self.ent[i] == vp && st > 0 && st <= qc && st > bs { best = i as i32; bs = st; } } }
            self.cand[8 + k] = best;
            self.cnr[8 + k] = self.newest_rel(best, qc);
        }
        self.woff = [0; 13];
    }
    fn wm_put(&mut self, key: u32, vi: usize, n: i32) {
        if key == 0 || n <= 0 { return; }
        let b = ((key % NSET) * 4) as usize;
        let (mut hit, mut empty, mut lru, mut lt) = (-1i32, -1i32, b, self.wmt[b]);
        for i in b..b + 4 { if self.wmk[i] == key { hit = i as i32; } if self.wmk[i] == 0 && empty < 0 { empty = i as i32; } if self.wmt[i] < lt { lt = self.wmt[i]; lru = i; } }
        let s = if hit >= 0 { hit as usize } else if empty >= 0 { empty as usize } else { lru };
        if hit < 0 { self.wml[s] = [0; 2]; self.wmv[s] = [0; 2]; }
        let l = n.min(16);
        self.wmk[s] = key;
        self.wml[s][vi] = l;
        (0..l as usize).for_each(|i| self.wmb[s][vi][i] = self.cwb[i]);
        self.wclk += 1;
        self.wmt[s] = self.wclk;
        self.wmv[s][vi] = self.wclk;
    }
}
#[derive(Clone, Copy, Default)]
pub struct Q { qclk: i32, inq: i32, q1: u32, q2: u32, c1: i32, c2: i32, qon: i32, wlen: i32, p4: u32, qp1: u32, qp2: u32, m1: u32, m2: u32, cwl: i32, ap4: u32, acf: i32, subj: u32, fst: i32, pub ap: i32, pub anc: bool }
#[derive(Clone)]
pub struct W19 { pub s: St, pub q: Q, pub m: Mem }
impl W19 {
    pub fn new(cp: usize) -> W19 { let mut w = W19 { s: init(), q: Q::default(), m: Mem::new() }; w.begin(cp); w }
    pub fn begin(&mut self, cp: usize) { (self.s, self.q, self.m.wclk) = (init(), Q { anc: true, ..Q::default() }, 0); self.m.wm_clear(); (cp >= 13).then(|| self.m.ent_clr()); }
    pub fn step(&mut self, g: &Geo, lex: &Lex, lx: &Lx19, bx: u8) {
        let (cp, qt, m, g0, mut s) = (g.copy, g.qt, &mut self.m, Geo { qt: false, ..g.clone() }, self.s);
        let Q { mut qclk, mut inq, mut q1, mut q2, mut c1, mut c2, mut qon, mut wlen, mut p4, mut qp1, mut qp2, mut m1, mut m2, mut cwl, mut ap4, mut acf, mut subj, mut fst, mut ap, mut anc } = self.q;
        let bb = bx as u32;
        let (isw, wh) = (wordch(bb), s[16]);
        let ends = !isw && wh != 0;
        if bb == 30 {
            (inq, qon, q1, q2, c1, c2, qp1, qp2, m1, m2, subj, fst, ap4, acf, qclk, anc) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, m.wclk, true);
            m.wm_clear();
            (cp >= 13).then(|| m.ent_clr());
        }
        if ends {
            let pw1 = wh;
            let madam = bb == 58 && s[0] == 109 && s[1] == 97 && s[2] == 100 && s[3] == 65;
            if qt {
                inq = if madam { 0 } else { inq };
                if inq == 1 && !madam {
                    let st = lx.stop[(pw1 & WM) as usize];
                    if wlen >= 3 && st == 0 {
                        let cw = match lex.lex[(pw1 & WM) as usize][2] as i32 { 0 => 1, v => v };
                        if c1 == 0 || cw < c1 { (q2, c2, qp2, q1, c1, qp1) = (q1, c1, qp1, pw1, cw, p4); } else if pw1 != q1 && (c2 == 0 || cw < c2) { (q2, c2, qp2) = (pw1, cw, p4); }
                    }
                    if fst == 1 && m.is_pron(cwl) { m1 = subj; fst = 0; }
                    else if wlen >= 3 && st == 0 { if fst == 1 { subj = p4; fst = 0; } m.wm_put(m1, 0, cwl); m.wm_put(m2, 1, cwl); m2 = m1; m1 = p4; }
                }
                if qon == 1 {
                    let mut hit = false;
                    for k in 0..4 { let ss = m.wm_find(if k < 2 { qp1 } else { qp2 }); hit |= m.bytes_eq(ss, k % 2, cwl) || m.bytes_eq(m.wm_find(m.val_p4(ss, k % 2)), 0, cwl); }
                    hit |= acf == 1 && m.bytes_eq(m.wm_find(ap4), 0, cwl);
                    (ap4, acf) = (p4, hit as i32);
                }
            }
            if cp >= 13 {
                (inq == 1).then(|| m.on_user_word(lx, p4, cwl, wh));
                m.pbd = if bb == 45 { 1 } else if bb == 39 { 2 } else { 0 };
                (qon == 1).then(|| m.adv_woff(cwl));
            }
        }
        wlen = if qt { if isw { wlen + 1 } else { 0 } } else { wlen };
        if isw { if cwl < 4 { p4 = p4.wrapping_mul(83492791).wrapping_add(bb + 1); } if cwl < 16 { m.cwb[cwl as usize] = bx; } cwl += 1; } else { p4 = 0; cwl = 0; }
        s = update(lex, &g0, &s, bb);
        if qt {
            let (b1, b2, b3, b4, b5, b6) = (s[0], s[1], s[2], s[3], s[4], s[5]);
            let user = b6 == 85 && b5 == 115 && b4 == 101 && b3 == 114 && b2 == 58 && b1 == 32;
            if user { (q1, q2, c1, c2, qp1, qp2, m1, m2, inq, qon, qclk, fst, ap4, acf) = (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, m.wclk, 1, 0, 0); }
            let adam = b6 == 65 && b5 == 100 && b4 == 97 && b3 == 109 && b2 == 58 && b1 == 32;
            if adam { inq = 0; qon = (q1 != 0) as i32; ap4 = 0; acf = 0; }
            ap = if adam || qon == 0 { 0 } else { ap + 1 };
            if b1 == 10 && b2 == 10 { qon = 0; inq = 0; }
            if cp >= 13 {
                if user { (m.qent, m.qrel, m.qtyp, m.ptag, m.flg) = (0, 0, 0, 0, 0); m.fact_reset(); }
                if adam && qon == 1 { m.pick_cands(qclk); }
            }
        }
        if cp >= 13 && inq == 1 && (bb == 46 || bb == 44 || bb == 33 || bb == 63 || bb == 10) { m.on_punct(bb); }
        (s[31], s[32], s[35]) = (q1, q2, qon as u32);
        (self.s, self.q) = (s, Q { qclk, inq, q1, q2, c1, c2, qon, wlen, p4, qp1, qp2, m1, m2, cwl, ap4, acf, subj, fst, ap, anc });
    }
    pub fn gates(&self, g: &Geo, mut f: impl FnMut(usize, i32, [u32; 4], usize)) {
        let (cp, c0, m, s, Q { qon, cwl, acf, ap4, qclk, qp1, qp2, .. }) = (g.copy, g.c0(), &self.m, &self.s, self.q);
        let e13 = |k: usize, t: [u32; 4]| t.map(|x| g.sl(x, 0, c0 + k));
        if cp >= 13 {
            let (qrel, qtyp) = (m.qrel, m.qtyp as u32);
            for k in 0..12 {
                let (i, kind, rank) = (m.cand[k], (k >= 8) as u32, if k >= 8 { k - 8 } else { k } as u32);
                let (mut byte, mut rel, mut vt, mut em, mut nr) = (-1, 0u32, 0u32, 0u32, 0u32);
                if qon == 1 && i >= 0 { byte = m.next_ent(i, m.woff[k], cwl); rel = m.rel[i as usize]; vt = m.vtype[i as usize] as u32; em = if k < 8 || m.ent[i as usize] == m.qent { 1 } else { 0 }; nr = m.cnr[k] as u32; }
                f(k, byte, e13(k, [hm(kind, rank, (cwl > 0) as u32), hm(qrel, rel, 1), hm(qtyp, vt, 2), hm(em, nr, 3)]), 4);
            }
            let byte = if qon == 1 && acf == 1 { m.next_byte(m.vs(m.wm_find(ap4), 0, qclk), 0, cwl) } else { -1 };
            f(12, byte, e13(12, [hm(2, 0, (cwl > 0) as u32), hm(qrel, 0, 1), hm(qtyp, 0, 2), hm((byte >= 0) as u32, 0, 3)]), 4);
        } else if cp > 0 {
            let nk = cp.min(4);
            let slot = |k: usize| m.vs(m.wm_find(if k < 2 { qp1 } else { qp2 }), k % 2, qclk);
            let mx = if qon == 1 { (0..nk).map(|k| { let ss = slot(k); if ss >= 0 && m.next_byte(ss, k % 2, cwl) >= 0 { m.wmv[ss as usize][k % 2] } else { 0 } }).fold(0, i32::max) } else { 0 };
            let mut cpk = |k: usize, kind: u32, byte: i32, nw: u32| f(k, byte, [g.sl(hm(k as u32, (cwl > 0) as u32 + 2 * kind + 8 * nw, 131 + k as u32), s[17] & 4095, c0 + k), 0, 0, 0], 1);
            for k in 0..nk { let (ss, mut byte, mut nw) = (slot(k), -1, 0); if qon == 1 { byte = m.next_byte(ss, k % 2, cwl); nw = (ss >= 0 && byte >= 0 && mx > 0 && m.wmv[ss as usize][k % 2] == mx) as u32; } cpk(k, 0, byte, nw); }
            if cp >= 9 {
                for k in 4..8 { let dk = k - 4; let byte = if qon == 1 { let ss = slot(dk); m.next_byte(m.vs(m.wm_find(m.val_p4(ss, dk % 2)), 0, qclk), 0, cwl) } else { -1 }; cpk(k, 1, byte, 0); }
                let byte = if qon == 1 && acf == 1 { m.next_byte(m.vs(m.wm_find(ap4), 0, qclk), 0, cwl) } else { -1 };
                cpk(8, 2, byte, 0);
            }
        }
    }
    pub fn mstate(&self) -> u32 { let m = &self.m; (m.cand[..8].iter().filter(|&&i| i >= 0).count().min(2) + 3 * (m.cand[8] >= 0) as usize + 6 * (m.qrel != 0) as usize + 12 * (m.qent != 0) as usize + 12 * (m.qent > SELFK) as usize) as u32 * 4 + m.qtyp.clamp(0, 3) as u32 }
    pub fn mtaps(&self, g: &Geo) -> [Option<u32>; 2] {
        let (m, ap, s) = (&self.m, self.q.ap as u32, &self.s);
        let k = self.mstate() * 16 + if m.cand[0] >= 0 { m.rel[m.cand[0] as usize] } else { m.qrel };
        let ab = if ap < 4 { ap } else if ap < 8 { 4 } else if ap < 16 { 5 } else if ap < 32 { 6 } else { 7 };
        let key = [(hm(k / 64, ap.min(23), 139), s[0]), (hm(k, ab, 141), hm(s[0] | s[1] << 8 | s[2] << 16, s[3], 143))];
        std::array::from_fn(|i| (g.copy >= 13 && self.q.anc && self.q.qon == 1 && i < g.mr).then(|| g.sl(key[i].0, key[i].1, g.mb() + i)))
    }
    pub fn gv(&self, g: &Geo, cw: &[u16]) -> [(i32, f32); 13] { let mut o = [(-1, 0f32); 13]; self.gates(g, |k, b, e, n| o[k] = (b, if b >= 0 { CG * e[..n].iter().fold(0f32, |a, &x| a + f16::from_bits(cw[k * g.t as usize + x as usize]).to_f32()) } else { 0.0 })); o }
    pub fn cv(&self, k: usize) -> (Vec<u8>, usize) { let (m, q) = (&self.m, &self.q); if k < 12 { let i = m.cand[k].max(0) as usize; (m.val[i][..m.vlen[i].clamp(0, 24) as usize].to_vec(), (m.woff[k] + q.cwl).max(0) as usize) } else { let s = m.vs(m.wm_find(q.ap4), 0, q.qclk).max(0) as usize; (m.wmb[s][0][..m.wml[s][0].clamp(0, 16) as usize].to_vec(), q.cwl.max(0) as usize) } }
}
pub struct Host { pub g: Geo, pub lex: Lex, pub lx: Lx19, pub cw: Vec<u16>, pub w: Vec<W19> }
impl Host {
    pub fn new(g: Geo, lex: Lex, cw: Vec<u16>) -> Host { Host { w: vec![W19::new(g.copy); 2 * MAXB], lx: Lx19::new(), g, lex, cw } }
    pub fn reset(&mut self, cur: usize, nb: usize) { let cp = self.g.copy; self.w[cur * MAXB..cur * MAXB + nb].par_iter_mut().for_each(|w| *w = W19::new(cp)); }
    pub fn step<T: Send>(&mut self, cur: usize, par: &[u32], byt: &[u8], sink: Vec<T>, f: impl Fn(&W19, &Geo, &[u16], T) + Sync + Send) {
        let Host { g, lex, lx, cw, w } = self;
        let (lo, hi) = w.split_at_mut(MAXB);
        let (src, dst) = if cur == 0 { (&*lo, hi) } else { (&*hi, lo) };
        dst[..par.len()].par_iter_mut().zip(sink).enumerate().with_min_len(4).for_each(|(b, (d, t))| { d.clone_from(&src[par[b] as usize]); d.step(g, lex, lx, byt[b]); f(d, g, cw, t) });
    }
}
