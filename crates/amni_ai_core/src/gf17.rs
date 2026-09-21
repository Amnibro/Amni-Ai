use std::fmt;
pub const MODULUS: u8 = 17;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GF17(pub u8);
impl GF17 {
pub fn new(val: u8) -> Self { Self(val % MODULUS) }
pub fn add(self, rhs: Self) -> Self { Self((self.0 + rhs.0) % MODULUS) }
pub fn sub(self, rhs: Self) -> Self { Self((self.0 + MODULUS - rhs.0 % MODULUS) % MODULUS) }
pub fn mul(self, rhs: Self) -> Self { Self(((self.0 as u16 * rhs.0 as u16) % MODULUS as u16) as u8) }
pub fn pow(self, mut exp: u32) -> Self {
let (mut base, mut res) = (self, Self(1));
while exp > 0 {
if exp % 2 == 1 { res = res.mul(base); }
base = base.mul(base);
exp /= 2;
}
res
}
pub fn inv(self) -> Option<Self> {
if self.0 == 0 { None } else { Some(self.pow(MODULUS as u32 - 2)) }
}
}
impl fmt::Display for GF17 {
fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.0) }
}
pub struct FibonacciRay {
pub state: [GF17; 3],
pub step_index: usize,
}
impl FibonacciRay {
pub fn new(seed: u8) -> Self {
Self { state: [GF17::new(seed), GF17::new(seed.wrapping_add(1)), GF17::new(1)], step_index: 0 }
}
pub fn step(&mut self) -> (GF17, GF17, GF17) {
let next_val = self.state[0].add(self.state[1]);
self.state[0] = self.state[1];
self.state[1] = next_val;
self.state[2] = self.state[2].mul(GF17::new(3));
self.step_index += 1;
(self.state[0], self.state[1], self.state[2])
}
pub fn project_sequence(&mut self, steps: usize) -> Vec<u8> {
let mut out = Vec::with_capacity(steps);
for _ in 0..steps { out.push(self.step().0.0); }
out
}
pub fn route_tokens(tokens: &[u8]) -> (u8, u8, u8) {
let mut acc = GF17::new(7);
for (i, &t) in tokens.iter().enumerate() {
let weight = GF17::new(((i % 16) + 1) as u8);
acc = acc.add(GF17::new(t).mul(weight));
}
(acc.0, acc.pow(2).0, acc.pow(3).0)
}
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ManifoldRule {
pub pattern: String,
pub invariant: String,
pub reinforcement_weight: u8,
pub observations_count: usize,
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RoutingManifold {
pub rules: Vec<ManifoldRule>,
}
impl RoutingManifold {
pub fn load_or_default() -> Self {
let mut p = std::path::PathBuf::from("data").join("routing_manifold.json");
if !p.exists() {
if let Ok(cur) = std::env::current_dir() {
if cur.ends_with("crates\\amni_ai_core") || cur.ends_with("crates/amni_ai_core") {
p = std::path::PathBuf::from("..").join("..").join("data").join("routing_manifold.json");
}
}
}
if let Ok(bytes) = std::fs::read(&p) {
if let Ok(m) = serde_json::from_slice::<RoutingManifold>(&bytes) {
return m;
}
}
Self {
rules: vec![
ManifoldRule { pattern: "quantum".into(), invariant: "exact_partial_trace_rho".into(), reinforcement_weight: 16, observations_count: 1 },
ManifoldRule { pattern: "raymarch".into(), invariant: "offscreen_buffer_scaling_60fps".into(), reinforcement_weight: 16, observations_count: 1 },
ManifoldRule { pattern: "audio".into(), invariant: "user_gesture_resume_guard".into(), reinforcement_weight: 16, observations_count: 1 },
ManifoldRule { pattern: "canvas".into(), invariant: "hex_color_parsing_no_css_vars".into(), reinforcement_weight: 16, observations_count: 1 },
ManifoldRule { pattern: "gravity".into(), invariant: "softening_epsilon_and_dt_scaling".into(), reinforcement_weight: 16, observations_count: 1 },
]
}
}
pub fn record_learning(&mut self, pattern: &str, invariant: &str) {
if let Some(existing) = self.rules.iter_mut().find(|r| r.pattern == pattern) {
existing.observations_count += 1;
existing.reinforcement_weight = (existing.reinforcement_weight.saturating_add(1)).min(16);
existing.invariant = invariant.to_string();
} else {
self.rules.push(ManifoldRule {
pattern: pattern.to_string(),
invariant: invariant.to_string(),
reinforcement_weight: 1,
observations_count: 1,
});
}
self.persist();
}
pub fn persist(&self) {
let mut p = std::path::PathBuf::from("data").join("routing_manifold.json");
if !p.exists() {
if let Ok(cur) = std::env::current_dir() {
if cur.ends_with("crates\\amni_ai_core") || cur.ends_with("crates/amni_ai_core") {
p = std::path::PathBuf::from("..").join("..").join("data").join("routing_manifold.json");
}
}
}
if let Some(parent) = p.parent() {
let _ = std::fs::create_dir_all(parent);
}
if let Ok(json) = serde_json::to_string_pretty(self) {
let _ = std::fs::write(&p, json);
}
}
pub fn get_active_invariants(&self, prompt: &str) -> Vec<String> {
let lower = prompt.to_lowercase();
self.rules.iter()
.filter(|r| lower.contains(&r.pattern.to_lowercase()))
.map(|r| r.invariant.clone())
.collect()
}
}

