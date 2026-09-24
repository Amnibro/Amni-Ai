use adam_train::{cfg::Cfg, engine, train};
fn main() {
    let c = Cfg::env();
    let mut e = engine(&c);
    train(&c, e.as_mut()).unwrap_or_else(|x| panic!("adam-train: {x}"));
}
