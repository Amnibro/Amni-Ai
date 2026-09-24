use adam_ray::RayField;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let home = std::env::var("AMNI_HOME").unwrap_or_else(|_| ".".into());
    let pk = format!("{home}/exports/gf17_continuum/{}", if a.get(2).map_or("v16", |s| s) == "v18" { "rayhash_v18_chat2_qt_s20k_sid.npy" } else { "rayhash_v16_big2_L21_e20_hop4_q5_s150k_pk_q5_i8.npz" });
    let mut e = RayField::new(Some(pk), None, None, None, a.get(1).cloned(), Some(home));
    let t = Instant::now();
    e.load().unwrap();
    let lt = t.elapsed().as_secs_f64();
    let n = 400;
    e.prime(b"User: Tell me about volcanoes.\nAdam:", 1);
    let t = Instant::now();
    (0..n).for_each(|i| { e.advance(&[0], &[b"abcdefgh "[i % 9]], true); });
    let adv = n as f64 / t.elapsed().as_secs_f64();
    let t = Instant::now();
    (0..n).for_each(|i| { e.advance(&[0], &[b"abcdefgh "[i % 9]], false); });
    let nowant = n as f64 / t.elapsed().as_secs_f64();
    let t = Instant::now();
    e.logits_for(&vec![b'a'; n]);
    let batch = n as f64 / t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut s = e.stream("User: Tell me about volcanoes.\nAdam:", n, 0.7, 12, &[], true, Some(7), None, 6.0, Default::default());
    let mut out = 0;
    while let Some(c) = s.next(&mut e) { out += c.len(); }
    let st = out as f64 / t.elapsed().as_secs_f64();
    let (p, c): (Vec<Vec<u8>>, Vec<Vec<u8>>) = (0..20).map(|i| (b"User: What is the capital of France?\nAdam: ".to_vec(), b"Paris is the capital and largest city of France, known for art, fashion, food and culture. Paris is the capital and largest city of France, known for art, fashion, food and culture. "[..150 + i].to_vec())).unzip();
    let t = Instant::now();
    (0..3).for_each(|_| { e.score(&p, &c); });
    let sc = t.elapsed().as_secs_f64() / 3.0;
    let t = Instant::now();
    let bp = e.bpb(None, 2, 128, 256, 1234).unwrap();
    let bt = 2.0 * 128.0 * 256.0 / t.elapsed().as_secs_f64();
    println!("{} | bpb(256 beams) {bt:.0} B/s = {bp:.4} | load {lt:.2}s | advance+readback {adv:.0}/s | advance no-readback {nowant:.0}/s | teacher-forced 1 beam {batch:.0} B/s | stream {st:.0} B/s | score batch20 {:.0} B/s ({:.1} ms)", e.backend, 20.0 * (43 + 169) as f64 / sc, sc * 1e3);
}
