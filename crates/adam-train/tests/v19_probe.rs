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
