use adam_ray::{cpu::hm, pack::{lexicon, Lex}};
use adam_train::{cfg::Cfg, v19::{bh, Drv19, MG}};
struct R { c: Cfg, d: Drv19, lex: Lex }
fn find(h: &[u8], n: &[u8]) -> usize { h.windows(n.len()).position(|w| w == n).unwrap() }
fn rfind(h: &[u8], n: &[u8]) -> usize { h.windows(n.len()).rposition(|w| w == n).unwrap() }
fn p4(w: &[u8]) -> u32 { bh(&w[..w.len().min(4)]) }
impl R {
    fn new(kv: &[(&str, &str)]) -> R { let c = Cfg::with(kv); R { lex: lexicon(&c.syn, Some(&c.wcnt)), d: Drv19::new(c.bs), c } }
    fn run(&mut self, t: &[u8]) -> (Vec<Vec<i32>>, Vec<Vec<u32>>) {
        let (n, lv, cp) = (t.len(), self.c.lv, self.c.copy);
        let (mut pi, mut pw, mut cpb) = (vec![0u32; n * lv * 4], vec![0f32; n * lv * 4], vec![0i32; n * cp]);
        self.d.run(&self.c, &self.lex, t, n, 0, true, &mut pi, &mut pw, &mut cpb);
        (cpb.chunks(cp).map(<[i32]>::to_vec).collect(), pi.chunks(lv * 4).map(<[u32]>::to_vec).collect())
    }
    fn sl(&self, u: u32, v: u32, l: usize) -> u32 { hm(u, v, l as u32 + 1) % self.c.te as u32 }
}
fn has(r: &[i32], c: u8) -> bool { r.contains(&(c as i32)) }
#[test]
fn probe_copy9() {
    let mut r = R::new(&[("VER", "19"), ("COPY", "9"), ("QT", "1"), ("LOG2_ENTRIES", "16"), ("BS", "2"), ("CTX", "768")]);
    let (c0, tl) = (r.c.geo.c0(), r.c.tl());
    let t: &[u8] = b"User: My sister lives in Denver.\nAdam: Noted.\n\nUser: Where does my sister live?\nAdam: Your sister lives in D";
    let (cpb, idx) = r.run(t);
    assert!(cpb[..find(t, b"Adam:")].iter().flatten().all(|&v| v < 0));
    let (pos, pos2) = (find(t, b"Adam: Your sister lives in ") + 26, rfind(t, b"in D") + 3);
    assert!(has(&cpb[pos], b'D') && has(&cpb[pos2], b'e'), "{:?} {:?}", cpb[pos], cpb[pos2]);
    let (pw, lv) = (bh(b"in") & 4095, c0 + 3);
    let set = |b: u32| [0, 1].map(|n| hm(hm(3, b + 8 * n, 134), pw, lv as u32 + 1) % r.c.te as u32);
    assert!(set(0).contains(&idx[pos][lv * 4]) && set(1).contains(&idx[pos2][lv * 4]));
    let cor: &[u8] = b"User: My sister lives in Denver.\nAdam: Nice.\n\nUser: Actually she moved to Austin.\nAdam: Got it.\n\nUser: Where does my sister live?\nAdam: Your sister lives in ";
    let (cpb, idx) = r.run(cor);
    let (cb, ix) = (&cpb[cor.len() - 1][..4], &idx[cor.len() - 1]);
    let nw: Vec<bool> = (0..4).map(|k| ix[(c0 + k) * 4] == hm(hm(k as u32, 8, 131 + k as u32), bh(b"in") & 4095, (c0 + k) as u32 + 1) % r.c.te as u32).collect();
    assert!(has(cb, b'A') && (0..4).all(|k| cb[k] < 0 || nw[k] == (cb[k] == b'A' as i32)), "{cb:?} {nw:?}");
    let san: &[u8] = b"User: My sister lives in San Diego.\n\nUser: Where does my sister live?\nAdam: Your sister lives in San ";
    let (cpb, _) = r.run(san);
    assert!(cpb[rfind(san, b"San ") + 3][8] == b'D' as i32 && cpb[rfind(san, b"in San") + 2][8] < 0);
    let hop: &[u8] = b"User: My sister lives in Denver.\n\nUser: Denver is in Colorado.\n\nUser: In which state does my sister live?\nAdam: Your sister lives in ";
    let (cpb, _) = r.run(hop);
    let hr = &cpb[hop.len() - 1];
    assert!(has(&hr[4..8], b'C') && !has(&hr[..4], b'C'), "{hr:?}");
    let cor2: &[u8] = b"User: My sister lives in Denver.\n\nUser: Actually my sister moved to Austin.\n\nUser: Where is my sister?\nAdam: Your sister lives in ";
    let (cpb, _) = r.run(cor2);
    let dr = &cpb[cor2.len() - 1][..4];
    assert!(has(dr, b'A') && !has(dr, b'D'), "{dr:?}");
    let stop: std::collections::HashSet<String> = adam_qa::STOP.iter().map(|w| w.to_lowercase()).chain(["my", "i'm", "i", "is", "are", "was", "in", "at", "to", "of", "the", "a", "an", "and", "called", "named", "works", "as", "from"].map(String::from)).collect();
    let (al, ss) = (b"bcdfghjklmnpqrstvwxyz", p4(b"sister"));
    let (mut same, mut other, mut n) = (vec![], vec![], 0usize);
    while same.len() < 3 || other.len() < 37 {
        let mut x = n;
        n += 1;
        let cs: Vec<u8> = (0..4).map(|_| { let c = al[x % 21]; x /= 21; c }).collect();
        let w: Vec<u8> = [&[cs[0].to_ascii_uppercase()], &cs[1..], b"a"].concat();
        let h = p4(&w);
        if stop.contains(&String::from_utf8_lossy(&w).to_lowercase()) || h == ss { continue; }
        if h % 128 == ss % 128 && same.len() < 3 { same.push(w) } else if h % 128 != ss % 128 && other.len() < 37 { other.push(w) }
    }
    let gap = [&b"User: My sister lives in Denver.\n\nUser: "[..], &[same, other].concat().join(&b' '), b".\n\nUser: Where is my sister?\nAdam: Your sister lives in "].concat();
    let (cpb, _) = r.run(&gap);
    assert!(has(&cpb[gap.len() - 1][..4], b'D'), "{:?}", cpb[gap.len() - 1]);
    let qs = |r: &mut R, t: &[u8], cands: &[&[u8]]| {
        let (_, idx) = r.run(t);
        let (ix, pw) = (&idx[t.len() - 1], bh(b"Adam") & 4095);
        let q1 = cands.iter().find(|w| r.sl(bh(w), 0, tl + 1) == ix[(tl + 1) * 4]).copied();
        let q2 = q1.and_then(|a| cands.iter().find(|w| r.sl(hm(bh(a), bh(w), 107), pw, tl + 2) == ix[(tl + 2) * 4]).copied());
        (q1.map(<[u8]>::to_vec), q2.map(<[u8]>::to_vec))
    };
    assert_eq!(qs(&mut r, b"User: Thumper Zanzibar?\nAdam: ", &[b"Where", b"Thumper", b"Zanzibar", b"live", b"does"]), (Some(b"Thumper".to_vec()), Some(b"Zanzibar".to_vec())));
    let u = (bh(b"Xylophrix") & ((1 << 20) - 1)) as usize;
    assert!(r.lex.lex[u][2] == 0 && r.d.lx.stop[u] == 0);
    assert_eq!(qs(&mut r, b"User: Xylophrix Thumper?\nAdam: ", &[b"Where", b"Xylophrix", b"Thumper", b"live", b"does"]), (Some(b"Xylophrix".to_vec()), Some(b"Thumper".to_vec())));
    println!("PROBE copy=9 levels={} old_denver D/e, correction newest, San Diego chain, two-hop C, direct Austin, gap40 D, question tori Thumper/Zanzibar, unknown-word question", r.c.lv);
}
#[test]
fn probe_copy13() {
    let mut r = R::new(&[("VER", "19"), ("COPY", "13"), ("QT", "1"), ("LEVELS", "21"), ("HOPS", "4"), ("LOG2_ENTRIES", "16"), ("BS", "2"), ("CTX", "768")]);
    let c0 = r.c.geo.c0();
    assert!(c0 == 24 && r.c.geo.hb() == 37 && r.c.lv == 41);
    let bro: &[u8] = b"User: My brother works as a product manager. My brother is 63. My brother drives a Rivian.\n\nUser: How old is my brother?\nAdam: ";
    let (cpb, idx) = r.run(bro);
    let pos = bro.len() - 1;
    let k = (0..8).find(|&k| cpb[pos][k] == b'6' as i32).unwrap_or_else(|| panic!("{:?}", cpb[pos]));
    assert_eq!(idx[pos][(c0 + k) * 4 + 2], r.sl(hm(3, 2, 2), 0, c0 + k));
    let bro6 = [bro, b"6"].concat();
    assert_eq!(r.run(&bro6).0[bro6.len() - 1][k], b'3' as i32);
    let drv: &[u8] = b"User: My brother works as a product manager. My brother is 63. My brother drives a Rivian.\n\nUser: What does my brother drive?\nAdam: ";
    assert!(has(&r.run(drv).0[drv.len() - 1][..8], b'R'));
    let unk: &[u8] = b"User: My brother drives a Rivian.\n\nUser: What does my uncle drive?\nAdam: ";
    assert!(r.run(unk).0[unk.len() - 1][..8].iter().all(|&v| v < 0));
    let cor: &[u8] = b"User: My sister lives in Denver.\n\nUser: Actually she moved to Austin.\n\nUser: Where does my sister live?\nAdam: ";
    let row = r.run(cor).0[cor.len() - 1].clone();
    assert!(row[0] == b'A' as i32 && has(&row[1..8], b'D'), "{row:?}");
    let hop: &[u8] = b"User: My sister lives in Denver.\n\nUser: Denver is in Colorado.\n\nUser: Where does my sister live?\nAdam: ";
    let hr = r.run(hop).0[hop.len() - 1].clone();
    assert!(has(&hr[8..12], b'C') && has(&hr[..8], b'D'), "{hr:?}");
    let san: &[u8] = b"User: My sister lives in San Diego.\n\nUser: Where does my sister live?\nAdam: Your sister lives in San ";
    let (cpb, _) = r.run(san);
    let m = &r.d.rows[0].m;
    assert!((0..64).any(|i| m.stamp[i] > 0 && &m.val[i][..m.vlen[i] as usize] == b"San Diego"));
    let sp = rfind(san, b"San ");
    assert!(has(&cpb[sp + 2][..8], b' ') && has(&cpb[sp + 3][..8], b'D'));
    println!("PROBE13 old63 k={k} gate ok, drive R, uncle clear, austin first {}, hop {:?}, san diego", row[0], &hr[8..12]);
}
#[test]
fn probe_mr() {
    let c = Cfg::with(&[("VER", "19"), ("COPY", "13"), ("MR", "2"), ("QT", "1"), ("LOG2_ENTRIES", "16"), ("BS", "1"), ("CTX", "512")]);
    let (lex, mut d, lv, mb) = (lexicon(&c.syn, Some(&c.wcnt)), Drv19::new(1), c.lv, c.geo.mb());
    assert!(lv == mb + 2 && mb == c.geo.hb() + c.geo.hops);
    let mut on = |t: &[u8], anc: bool| { let n = t.len(); let (mut pi, mut pw, mut cpb) = (vec![0u32; n * lv * 4], vec![0f32; n * lv * 4], vec![0i32; n * 13]); d.run(&c, &lex, t, n, 0, anc, &mut pi, &mut pw, &mut cpb); ((0..n).map(|q| [pw[(q * lv + mb) * 4], pw[(q * lv + mb + 1) * 4]]).collect::<Vec<_>>(), d.rows[0].mstate()) };
    let kn: &[u8] = b"User: My sister lives in Denver.\nAdam: Noted.\n\nUser: Where does my sister live?\nAdam: Your sister";
    let (w, k) = on(kn, true);
    let a = rfind(kn, b"Adam: ") + 5;
    assert!(w[rfind(kn, b"User: ")..a].iter().all(|x| x == &[0.0; 2]) && w[a..].iter().all(|x| x == &[MG; 2]) && k / 4 % 3 == 1 && k / 24 % 2 == 1 && k / 48 == 2, "{k}");
    let (w, k) = on(b"User: My sister lives in Denver.\nAdam: Noted.\n\nUser: Where does my brother live?\nAdam: I", true);
    assert!(w[w.len() - 1] == [MG; 2] && k / 4 % 3 == 0, "{k}");
    let (w, _) = on(kn, false);
    assert!(w.iter().all(|x| x == &[0.0; 2]));
    let sep = [b"sister.\n\n\x1e".as_slice(), kn].concat();
    let (w, k) = on(&sep, false);
    assert!(w[w.len() - 1] == [MG; 2] && k / 4 % 3 == 1);
    println!("PROBE MR taps gated by answer and anchor, cand states known {k}");
}
#[test]
fn probe_echo() {
    let mut r = R::new(&[("VER", "19"), ("COPY", "16"), ("MR", "2"), ("QT", "1"), ("LEVELS", "21"), ("HOPS", "4"), ("LOG2_ENTRIES", "16"), ("BS", "1"), ("CTX", "768")]);
    let (c0, hb) = (r.c.geo.c0(), r.c.geo.hb());
    assert!(r.c.copy == 16 && hb == 40 && r.c.geo.mb() == 44 && r.c.lv == 46);
    let ack: &[u8] = b"User: My brother works as a pilot.\nAdam: ";
    let (cpb, idx) = r.run(ack);
    let row = &cpb[ack.len() - 1];
    assert!(row[13] == b'p' as i32 && row[14] < 0 && row[15] == b'b' as i32 && row[..13].iter().all(|&v| v < 0), "{row:?}");
    assert_eq!(idx[ack.len() - 1][(c0 + 13) * 4], r.sl(hm(3, 0, 2 + (bh(b"Adam") & 4095) * 2 + 1), 0, c0 + 13));
    assert_eq!(idx[ack.len() - 1][(c0 + 13) * 4 + 2], r.sl(hm(4, 0, 2), 0, c0 + 13));
    let pil = [ack, b"Oh, a pil"].concat();
    assert_eq!(r.run(&pil).0[pil.len() - 1][13], b'o' as i32);
    let cor: &[u8] = b"User: My sister lives in Denver.\nAdam: Nice.\n\nUser: Actually, my sister moved to Austin.\nAdam: ";
    let row = r.run(cor).0[cor.len() - 1].clone();
    assert!(row[0] == b'D' as i32 && row[13] == b'A' as i32 && row[15] == b's' as i32, "{row:?}");
    let q: &[u8] = b"User: My sister lives in Denver.\nAdam: Nice.\n\nUser: Where does my sister live?\nAdam: ";
    let (cpb, idx) = r.run(q);
    let row = &cpb[q.len() - 1];
    assert!(row[0] == b'D' as i32 && row[13] < 0 && row[15] == b's' as i32 && idx[q.len() - 1][c0 * 4 + 2] == r.sl(hm(2, 1, 2), 0, c0), "{row:?}");
    let inl: &[u8] = b"User: My sister-in-law lives in Denver.\n\nUser: Where does my sister-in-law live?\nAdam: Your sister-";
    let row = r.run(inl).0[inl.len() - 1].clone();
    assert!(row[15] == b'i' as i32 && row[0] == b'D' as i32, "{row:?}");
    let lab: &[u8] = b"User: My lab partner drives a Rivian.\n\nUser: What does my lab partner drive?\nAdam: Your lab ";
    let row = r.run(lab).0[lab.len() - 1].clone();
    assert!(row[15] == b'p' as i32 && r.run(&lab[..lab.len() - 4]).0[lab.len() - 5][15] == b'l' as i32, "{row:?}");
    let me: &[u8] = b"User: My name is Anthony.\nAdam: ";
    let row = r.run(me).0[me.len() - 1].clone();
    assert!(row[13] == b'A' as i32 && row[15] < 0, "{row:?}");
    let pr: &[u8] = b"User: My brother works as a pilot.\nAdam: Nice.\n\nUser: He drives a Rivian.\nAdam: ";
    let row = r.run(pr).0[pr.len() - 1].clone();
    assert!(row[13] == b'R' as i32 && row[15] == b'b' as i32, "{row:?}");
    let pq: &[u8] = b"User: My brother works as a pilot.\nAdam: Nice.\n\nUser: What does he drive?\nAdam: ";
    assert_eq!(r.run(pq).0[pq.len() - 1][15], b'b' as i32);
    let (st, qs) = ([ack, b"O"].concat(), [q, b"Y"].concat());
    let mut w = adam_train::v19::W19::new(16);
    let (g, lx) = (r.c.geo.clone(), adam_train::v19::Lx19::new());
    st.iter().for_each(|&b| w.step(&g, &r.lex, &lx, b));
    let (ks, pil) = (w.mstate(), [ack, b"Oh, a pilot"].concat());
    let mut w2 = adam_train::v19::W19::new(16);
    qs.iter().for_each(|&b| w2.step(&g, &r.lex, &lx, b));
    let mut w3 = adam_train::v19::W19::new(16);
    pil.iter().for_each(|&b| w3.step(&g, &r.lex, &lx, b));
    let a3 = |w: &adam_train::v19::W19, ap: u32, v: u32| { let m = w.mstate() * 16 + if w.m.cand[0] >= 0 { w.m.rel[w.m.cand[0] as usize] } else { w.m.qrel }; r.sl(hm(m / 64, if v == 1 { 24 } else { ap }, 139), w.s[0], r.c.geo.mb()) };
    assert!(ks >= 144 && w2.mstate() < 144 && w3.mtaps(&g)[0] == Some(a3(&w3, w3.q.ap.min(23) as u32, 1)), "{ks} {}", w2.mstate());
    let vq = |t: &[u8]| { let mut w = adam_train::v19::W19::new(16); t.iter().for_each(|&b| w.step(&g, &r.lex, &lx, b)); w.q.vp };
    assert!(w3.q.vp == 1 && vq(&[pil.as_slice(), b"! Nic"].concat()) == 6 && vq(&[ack, b"Oh, a pil"].concat()) == 0 && vq(ack) == 0);
    println!("PROBE echo pilot/entity brother, correction ack Austin echo + old Denver direct, question entity sister, sister-in-law, lab partner, name echo");
}
#[test]
fn probe_mask() {
    let kv = [("VER", "19"), ("COPY", "16"), ("MR", "2"), ("QT", "1"), ("LEVELS", "21"), ("HOPS", "4"), ("LOG2_ENTRIES", "16"), ("BS", "1"), ("CTX", "768")];
    let (mut r, mut r0) = (R::new(&[&kv[..], &[("MASKV", "1")]].concat()), R::new(&kv));
    assert!(r.c.geo.mv && !r0.c.geo.mv && !R::new(&[("VER", "19"), ("COPY", "13"), ("MR", "2"), ("MASKV", "1")]).c.geo.mv);
    let (g, lx, c0) = (r.c.geo.clone(), adam_train::v19::Lx19::new(), r.c.geo.c0());
    let pre: &[u8] = b"User: My sister lives in Las Cruces.\nAdam: Nice.\n\nUser: Where does my sister live?\nAdam: She lives in ";
    let (a, b) = ([pre, b"Las Cruces, from what you told me."].concat(), [pre, b"_, from what you told me."].concat());
    let ((_, ia), (_, ib), (_, i0)) = (r.run(&a), r.run(&b), r0.run(&a));
    let run = |t: &[u8]| { let mut w = adam_train::v19::W19::new(16); t.iter().for_each(|&b| w.step(&g, &r.lex, &lx, b)); w };
    let tail = |w: &adam_train::v19::W19, n: usize| w.s[..n].iter().rev().map(|&c| c as u8).collect::<Vec<u8>>();
    let (wv, wu) = (run(&a[..pre.len() + 10]), run(b"User: My sister lives in Las Cruces"));
    assert!(tail(&wv, 10) == b"lives in _" && wv.s[16] == bh(b"_") && wv.q.vp == 1 && tail(&wu, 6) == b"Cruces", "{:?}", String::from_utf8_lossy(&tail(&wv, 16)));
    let wf = run(&a[..pre.len() + 17]);
    assert!(tail(&wf, 11) == b"in _, from " && wf.s[17] == bh(b"from") && wf.s[18] == bh(b"_") && wf.s[19] == bh(b"in"), "{:?}", String::from_utf8_lossy(&tail(&wf, 16)));
    let (ea, eb) = (pre.len() + 9, pre.len());
    assert!((0..a.len() - ea).all(|d| ia[ea + d][..c0 * 4] == ib[eb + d][..c0 * 4]) && i0[ea][..c0 * 4] != ib[eb][..c0 * 4] && ia[..ea] == i0[..ea]);
    let inl: &[u8] = b"User: My sister-in-law lives in Denver.\n\nUser: Where does my sister-in-law live?\nAdam: Your sister-in-law";
    let wi = run(inl);
    assert!(tail(&wi, 6) == b"Your _" && run(&[inl, b"'s"].concat()).s[17] == bh(b"_"), "{:?}", String::from_utf8_lossy(&tail(&wi, 16)));
    let ack: &[u8] = b"User: My name is Anthony.\nAdam: Anthony";
    assert!(tail(&run(ack), 7) == b"Adam: _" && tail(&run(&[ack, b"x"].concat()), 8) == b"Adam: _x");
    let run0 = |t: &[u8]| { let mut w = adam_train::v19::W19::new(16); t.iter().for_each(|&b| w.step(&r0.c.geo, &r.lex, &lx, b)); w };
    let (qa, qa0, qq, qq0) = (run(&ack[..ack.len() - 7]), run0(&ack[..ack.len() - 7]), run(&a[..pre.len()]), run0(&a[..pre.len()]));
    assert!((31..33).any(|i| qa0.s[i] == bh(b"Anthony") && qa.s[i] == bh(b"_") && qa.s[63 - i] == qa0.s[63 - i]) && qq.s[31] == qq0.s[31] && qq.s[32] == qq0.s[32] && qq.s[31] != bh(b"_"), "{} {}", qa.s[31], qa0.s[31]);
    println!("PROBE mask: reply values collapse to one '_' word (multiword, hyphen entity, echo), question words that are stored values become '_', user-turn bytes and MASKV=0 untouched, masked taps equal a literal '_' reply");
}
#[test]
fn probe_latest() {
    let kv = [("VER", "19"), ("COPY", "16"), ("MR", "2"), ("MASKV", "1"), ("QT", "1"), ("LEVELS", "21"), ("HOPS", "4"), ("LOG2_ENTRIES", "16"), ("BS", "1"), ("CTX", "768")];
    let (mut r, mut r0) = (R::new(&[&kv[..], &[("LATEST", "1")]].concat()), R::new(&kv));
    assert!(r.c.geo.lt && !r0.c.geo.lt && !R::new(&[("VER", "19"), ("COPY", "13"), ("MR", "2"), ("LATEST", "1")]).c.geo.lt);
    let last = |r: &mut R, t: &[u8]| r.run(t).0[t.len() - 1].clone();
    let q = |x: &[u8]| [x, b"\nAdam: "].concat();
    let cor = q(b"User: My sister lives in Denver.\nAdam: Nice.\n\nUser: Actually, my sister moved to Austin.\nAdam: Got it.\n\nUser: Where does my sister live?");
    let (a, a0) = (last(&mut r, &cor), last(&mut r0, &cor));
    assert!(a[0] == b'A' as i32 && a[1..8].iter().all(|&v| v < 0) && a0[0] == b'A' as i32 && a0[1] == b'D' as i32, "{a:?} {a0:?}");
    let same = q(b"User: My sister lives in Denver.\nAdam: Nice.\n\nUser: Actually, my sister moved to Austin. Where does she live now?");
    let (a, a0) = (last(&mut r, &same), last(&mut r0, &same));
    assert!(a[0] == b'A' as i32 && !a[..13].contains(&(b'D' as i32)) && a0[0] == b'D' as i32 && a0[13] == b'A' as i32, "{a:?} {a0:?}");
    let ack = q(b"User: My sister lives in Denver.\nAdam: Nice.\n\nUser: Actually, my sister moved to Austin.");
    let a = last(&mut r, &ack);
    assert!(a[0] == b'D' as i32 && a[13] == b'A' as i32, "{a:?}");
    let jk = q(b"User: I live in Kosice.\nAdam: Nice.\n\nUser: I live in a cozy flat.\nAdam: Ok.\n\nUser: Where do I live?");
    let jc = q(b"User: I live in Kosice.\nAdam: Nice.\n\nUser: Update: I live in a cozy flat.\nAdam: Ok.\n\nUser: Where do I live?");
    let (a, b) = (last(&mut r, &jk), last(&mut r, &jc));
    assert!(a[0] == b'c' as i32 && a[1] == b'K' as i32 && b[0] == b'c' as i32 && b[1] < 0, "{a:?} {b:?}");
    let ing = q(b"User: I live in Kosice.\nAdam: Nice.\n\nUser: I'm looking for ideas to decorate my living room.\nAdam: Ok.\n\nUser: Where do I live?");
    let (a, a0) = (last(&mut r, &ing), last(&mut r0, &ing));
    assert!(a[0] == b'K' as i32 && a0[0] != b'K' as i32, "{a:?} {a0:?}");
    let loc = q(b"User: My friend lives in Zurich.\nAdam: Nice.\n\nUser: Where is my friend located?");
    let (a, a0) = (last(&mut r, &loc), last(&mut r0, &loc));
    assert!(a[0] == b'Z' as i32 && a[13] < 0 && a0[13] == b'l' as i32, "{a:?} {a0:?}");
    let inl = q(b"User: My mother-in-law lives in Stamford.\nAdam: Nice.\n\nUser: My father-in-law lives in Bangkok.\nAdam: Ok.\n\nUser: Where does my mother-in-law live?");
    let (a, a0) = (last(&mut r, &inl), last(&mut r0, &inl));
    assert!(a[0] == b'S' as i32 && a[1] < 0 && a0[0] == b'B' as i32 && a[15] == b'm' as i32, "{a:?} {a0:?}");
    let lab = q(b"User: My lab partner drives a Rivian.\nAdam: Nice.\n\nUser: My partner drives a Volvo.\nAdam: Ok.\n\nUser: What does my lab partner drive?");
    let (a, a0) = (last(&mut r, &lab), last(&mut r0, &lab));
    assert!(a[0] == b'R' as i32 && a[1] < 0 && a0[0] == b'V' as i32 && a[15] == b'l' as i32, "{a:?} {a0:?}");
    let short = q(b"User: My favorite hobby is tennis.\nAdam: Fun.\n\nUser: What am I into?");
    let (a, a0) = (last(&mut r, &short), last(&mut r0, &short));
    assert!(a[0] == b't' as i32 && a0.iter().all(|&v| v < 0), "{a:?} {a0:?}");
    let obj = q(b"User: My bike is silver.\nAdam: Nice.\n\nUser: My plant is called Minty.\nAdam: Cute.\n\nUser: What color is my bike painted?");
    let pl = q(b"User: My bike is silver.\nAdam: Nice.\n\nUser: My plant is called Minty.\nAdam: Cute.\n\nUser: What is my plant called?");
    let (a, b, a0) = (last(&mut r, &obj), last(&mut r, &pl), last(&mut r0, &obj));
    assert!(a[0] == b's' as i32 && b[0] == b'M' as i32 && a0[0] < 0, "{a:?} {b:?} {a0:?}");
    let syn = q(b"User: I work as a lawyer.\nAdam: Nice.\n\nUser: I hate anchovies.\nAdam: Noted.\n\nUser: What is my profession?");
    let hate = q(b"User: I work as a lawyer.\nAdam: Nice.\n\nUser: I hate anchovies.\nAdam: Noted.\n\nUser: Which food do I dislike?");
    let (a, b, a0, b0) = (last(&mut r, &syn), last(&mut r, &hate), last(&mut r0, &syn), last(&mut r0, &hate));
    assert!(a[0] == b'l' as i32 && b[0] == b'a' as i32 && a0[0] < 0 && b0[0] < 0, "{a:?} {b:?} {a0:?} {b0:?}");
    let (g, lx) = (r.c.geo.clone(), adam_train::v19::Lx19::new());
    let mut w = adam_train::v19::W19::new(16);
    cor.iter().for_each(|&b| w.step(&g, &r.lex, &lx, b));
    assert!((0..64).filter(|&i| w.m.stamp[i] > 0 && w.m.rel[i] == 4).count() == 2 && w.m.cor.iter().filter(|&&c| c).count() == 1 && w.q.mq == 1);
    println!("PROBE latest: correction supersedes (question turn, same turn), ack keeps old, junk w/o marker keeps old, -ing freeze, no facts from questions, in-law/lab-partner compounds, stopword-only question, object entities + color, synonyms + dislike, history kept");
}
#[test]
fn probe_gen() {
    let kv = [("VER", "19"), ("COPY", "16"), ("MR", "2"), ("MASKV", "1"), ("QT", "1"), ("LEVELS", "21"), ("HOPS", "4"), ("LOG2_ENTRIES", "16"), ("BS", "1"), ("CTX", "768"), ("LATEST", "1")];
    let (mut r, mut r0) = (R::new(&[&kv[..], &[("GEN", "1")]].concat()), R::new(&kv));
    assert!(r.c.geo.gn && !r0.c.geo.gn && !R::new(&[("VER", "19"), ("COPY", "16"), ("MR", "2"), ("GEN", "1")]).c.geo.gn);
    let last = |r: &mut R, t: &[u8]| r.run(t).0[t.len() - 1].clone();
    let q = |x: &[u8]| [x, b"\nAdam: "].concat();
    let cases: [(&[u8], u8); 5] = [(b"User: My aunt lives in Tromso.\nAdam: Nice.\n\nUser: My cousin lives in the same town as my aunt.\nAdam: Ok.\n\nUser: Where does my cousin live?", b'T'), (b"User: My horse can't stand fireworks.\nAdam: Ok.\n\nUser: What does my horse hate?", b'f'), (b"User: My study buddy is a court reporter.\nAdam: Ok.\n\nUser: What does my study buddy do for a living?", b'c'), (b"User: My sister lives in Oslo.\nAdam: Ok.\n\nUser: Where does my sister live? I forget.", b'O'), (b"User: My uncle adores bonsai.\nAdam: Ok.\n\nUser: What does my uncle love?", b'b')];
    for (t, c) in cases { let t = q(t); let (a, a0) = (last(&mut r, &t), last(&mut r0, &t)); assert!(a[0] == c as i32 && a0[0] != c as i32, "{:?} {a:?} {a0:?}", String::from_utf8_lossy(&t)); }
    let reg = q(b"User: My neighbour lives in Ipoh, a city in Malaysia.\nAdam: Ok.\n\nUser: Which country does my neighbour live in?");
    let (a, a0) = (last(&mut r, &reg), last(&mut r0, &reg));
    assert!(a[8] == b'M' as i32 && a0[8] < 0, "{a:?} {a0:?}");
    println!("PROBE gen: same-as chain, negated tolerate -> dislike, relation-word compound entity, question freeze past '?', WordNet verb, appositive region hop");
}
#[test]
fn probe_gen2() {
    let kv = [("VER", "19"), ("COPY", "16"), ("MR", "2"), ("MASKV", "1"), ("QT", "1"), ("LEVELS", "21"), ("HOPS", "4"), ("LOG2_ENTRIES", "16"), ("BS", "1"), ("CTX", "768"), ("LATEST", "1"), ("GEN", "1")];
    let (mut r, mut r0) = (R::new(&[&kv[..], &[("GEN2", "1")]].concat()), R::new(&kv));
    assert!(r.c.geo.g2 && !r0.c.geo.g2 && !R::new(&[("VER", "19"), ("COPY", "16"), ("MR", "2"), ("LATEST", "1"), ("GEN2", "1")]).c.geo.g2);
    let last = |r: &mut R, t: &[u8]| r.run(t).0[t.len() - 1].clone();
    let q = |x: &[u8]| [x, b"\nAdam: "].concat();
    let cases: [(&[u8], i32); 4] = [(b"User: My study buddy rents a little flat in Napier.\nAdam: Ok.\n\nUser: Can you tell me where my study buddy lives?", b'N' as i32), (b"User: My goddaughter breaks out in hives around strawberries.\nAdam: Ok.\n\nUser: What triggers my goddaughter's allergy?", b's' as i32), (b"User: If you ever go to Tainan, that's where my partner lives.\nAdam: Ok.\n\nUser: Where does my partner live?", b'T' as i32), (b"User: My tortoise shares a name with my gecko.\nAdam: Ok.\n\nUser: My gecko is called Biscuit.\nAdam: Ok.\n\nUser: What do we call my tortoise?", b'B' as i32)];
    for (t, c) in cases { let t = q(t); let (a, a0) = (last(&mut r, &t), last(&mut r0, &t)); assert!(a[0] == c && a0[0] != c, "{:?} {a:?} {a0:?}", String::from_utf8_lossy(&t)); }
    let f13: [(&[u8], usize, i32); 9] = [(b"User: My sister was born in 1993.\nAdam: Ok.\n\nUser: What is my sister's birth year?", 0, b'1' as i32), (b"User: My car is called Bolt.\nAdam: Ok.\n\nUser: What is my car called?", 0, b'B' as i32), (b"User: My sister's car is called Daisy.\nAdam: Ok.\n\nUser: My car is called Bolt.\nAdam: Ok.\n\nUser: What is my sister's car called?", 0, b'D' as i32), (b"User: My friend likes tea.\nAdam: Ok.\n\nUser: What does my friend like?", 0, b't' as i32), (b"User: My parrot is called Kiwi.\nAdam: Ok.\n\nUser: Kiwi only eats seeds.\nAdam: Ok.\n\nUser: What does my parrot eat?", 0, b's' as i32), (b"User: I bought the coral bag in Miami.\nAdam: Ok.\n\nUser: Where did I buy my bag?", 0, b'M' as i32), (b"User: My favourite drink is lemonade.\nAdam: Ok.\n\nUser: What's my favorite drink?", 0, b'l' as i32), (b"User: My brother has a birthday in May.\nAdam: Ok.\n\nUser: When is my brother's birthday?", 0, b'M' as i32), (b"User: I work as a plumber in Chicago.\nAdam: Ok.\n\nUser: Chicago is in Illinois.\nAdam: Ok.\n\nUser: In which state do I work as a plumber?", 8, b'I' as i32)];
    for (t, k, c) in f13 { let t = q(t); let a = last(&mut r, &t); assert!(a[k] == c, "{:?} {a:?}", String::from_utf8_lossy(&t)); }
    let hy = q(b"User: If my brother were a pilot, he would be thrilled.\nAdam: Ok.\n\nUser: Pop quiz: what does my brother do for work? I forget.");
    assert!(last(&mut r, &hy)[0] < 0);
    let (g, lx) = (r.c.geo.clone(), adam_train::v19::Lx19::new());
    let mut w = adam_train::v19::W19::new(16);
    q(b"User: My aunt lives in Oslo.\nAdam: Ok.\n\nUser: Random question, but what does my uncle drive? I forget.").iter().for_each(|&b| w.step(&g, &r.lex, &lx, b));
    assert!(w.m.cand[0] < 0 && w.m.qrel == 9 && w.m.qent > 1 && w.mtaps(&g)[0] == Some(g.sl(hm(9001, 0, 147), 0, g.mb())), "{:?} {} {}", w.m.cand, w.m.qent, w.m.qrel);
    println!("PROBE gen2: flat/hives idioms, value before entity, pending same-as link, hypothetical not stored + framing-robust question, pooled decline keys; F13 birth years, object names, like fallback, name subjects, purchase place, favourite nouns, has-a-birthday, workplace hop");
}
