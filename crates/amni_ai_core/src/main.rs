mod evaluator_swarm;
mod gf17;
mod harness;
mod lattice;
mod server;
mod steering;
use std::env;
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use tokio::net::TcpListener;
use evaluator_swarm::{EvaluatorSwarm, save_output_artifact};
use harness::{IngressIntentProbe, ReflectiveHarness};
fn open_browser(url: &str) {
#[cfg(target_os = "windows")]
let _ = std::process::Command::new("cmd").args(["/C", "start", url]).spawn();
#[cfg(target_os = "macos")]
let _ = std::process::Command::new("open").arg(url).spawn();
#[cfg(target_os = "linux")]
let _ = std::process::Command::new("xdg-open").arg(url).spawn();
}
async fn bind_with_fallback(start_port: u16, max_tries: u16) -> Option<(u16, TcpListener)> {
for p in start_port..(start_port + max_tries) {
let addr = SocketAddr::from(([0, 0, 0, 0], p));
if let Ok(listener) = TcpListener::bind(addr).await {
return Some((p, listener));
}
}
None
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
let args: Vec<String> = env::args().collect();
if args.len() > 1 && (args[1] == "--help" || args[1] == "-h" || args[1] == "help") {
println!("==============================================================");
println!("  Amni-AI Native GF(17) Engine & Evaluator Swarm (v6.20.293)");
println!("==============================================================");
println!("USAGE:");
println!("  AmniAI.exe                       Launch Studio with browser & auto-port");
println!("  AmniAI.exe serve [--port <p>]    Run web server on specific/auto port");
println!("  AmniAI.exe run \"<prompt>\" [--open] Synthesize code & run parallel checkers");
println!("  AmniAI.exe check <file.html>     Evaluate file with 7 discretized models");
println!("  AmniAI.exe --help                Show this help message\n");
println!("OUTPUTS:");
println!("  Generated artifacts are saved to: workspace/outputs/");
println!("==============================================================");
return Ok(());
}
if args.len() > 1 && args[1] == "check" {
if args.len() < 3 {
eprintln!("[Amni-AI] Error: Specify a file to check. Example: AmniAI.exe check workspace/outputs/quantum_circuit.html");
return Ok(());
}
let file_path = PathBuf::from(&args[2]);
if !file_path.exists() {
eprintln!("[Amni-AI] Error: File not found at {:?}", file_path);
return Ok(());
}
let content = fs::read_to_string(&file_path)?;
let swarm = EvaluatorSwarm::new();
let audit = swarm.evaluate_parallel(&content).await;
println!("==============================================================");
println!(" [Amni-AI] 7-Model Discretized Evaluator Swarm (~64KB Stack)");
println!(" Target File: {:?}", file_path);
println!(" Composite Score: {:.0}/100 | Status: {}", audit.composite_score, if audit.all_passed { "PASS" } else { "ATTENTION" });
println!("--------------------------------------------------------------");
for d in &audit.diagnostics {
let stat = if d.passed { "PASS" } else { "FAIL" };
println!(" - {:<20} [{}] (Score: {:>3.0}%)", d.checker, stat, d.score * 100.0);
for f in &d.findings { println!("     * {}", f); }
}
println!("==============================================================");
return Ok(());
}
if args.len() > 1 && args[1] == "run" {
if args.len() < 3 {
eprintln!("[Amni-AI] Error: Specify a prompt. Example: AmniAI.exe run \"Quantum circuit with bloch sphere\" --open");
return Ok(());
}
let prompt = &args[2];
let open_in_browser = args.iter().any(|a| a == "--open");
println!("[Amni-AI] Synthesizing artifact for prompt: \"{}\"...", prompt);
let mut harness_res = ReflectiveHarness::execute(prompt);
let probe = IngressIntentProbe::probe(prompt);
let swarm = EvaluatorSwarm::new();
let mut audit = swarm.evaluate_parallel(&harness_res.verified_artifact).await;
if !audit.all_passed || audit.composite_score < 100.0 {
let (patched, count) = swarm.apply_patches(&harness_res.verified_artifact, &audit);
if count > 0 {
harness_res.verified_artifact = patched;
audit = swarm.evaluate_parallel(&harness_res.verified_artifact).await;
}
}
let slug = if probe.is_quantum { "quantum_circuit" }
else if probe.is_raymarch { "raymarching_sdf" }
else if probe.is_particle { "particle_gravity_wells" }
else if probe.is_audio { "fm_synth" }
else if probe.is_webgpu { "webgpu_app" }
else if probe.is_game { "asteroids_game" }
else { "synthesized_artifact" };
let (out_file, audit_file) = save_output_artifact(slug, &harness_res.verified_artifact, &audit)?;
let abs_out = fs::canonicalize(&out_file).unwrap_or(out_file);
let abs_audit = fs::canonicalize(&audit_file).unwrap_or(audit_file);
let clean_out = evaluator_swarm::clean_path_str(&abs_out);
let clean_audit = evaluator_swarm::clean_path_str(&abs_audit);
let file_url = format!("file:///{}", clean_out.replace('\\', "/"));
println!("==============================================================");
println!(" [Amni-AI] Synthesis & Parallel Multi-Modal Verification Complete");
println!(" Composite Score: {:.0}/100 | Invariants: {}", audit.composite_score, if audit.all_passed { "ALL VERIFIED" } else { "PATCHED" });
println!(" Memory Footprint: {} bytes (~64 KB)", audit.memory_footprint_bytes);
println!("--------------------------------------------------------------");
for d in &audit.diagnostics {
let stat = if d.passed { "PASS" } else { "FAIL" };
println!(" - {:<20} [{}] (Score: {:>3.0}%)", d.checker, stat, d.score * 100.0);
}
println!("--------------------------------------------------------------");
println!(" Output Saved: {}", clean_out);
println!(" Audit Saved:  {}", clean_audit);
println!(" Direct Link:  {}", file_url);
println!("==============================================================");
if open_in_browser { open_browser(&file_url); }
return Ok(());
}
let mut target_port: u16 = env::var("ADAM_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(12000);
let mut dual_port: u16 = 7700;
let mut no_browser = false;
let mut i = 1;
while i < args.len() {
if args[i] == "--port" && i + 1 < args.len() {
if let Ok(p) = args[i + 1].parse() { target_port = p; }
i += 1;
} else if args[i] == "--dual-port" && i + 1 < args.len() {
if let Ok(p) = args[i + 1].parse() { dual_port = p; }
i += 1;
} else if args[i] == "--no-browser" {
no_browser = true;
}
i += 1;
}
println!("[Amni-AI] Starting native GF(17) engine v6.20.293...");
let (actual_port, listener_primary) = match bind_with_fallback(target_port, 50).await {
Some(pair) => pair,
None => {
eprintln!("[Amni-AI] Fatal: Could not bind primary port {} or any fallback up to {}", target_port, target_port + 50);
return Err("Address allocation failed".into());
}
};
if actual_port != target_port {
println!("[Amni-AI] Port {} was occupied; automatically bound fallback port {}", target_port, actual_port);
} else {
println!("[Amni-AI] Successfully bound primary port {}", actual_port);
}
let router_primary = server::build_router();
println!("[Amni-AI] Primary listening on http://0.0.0.0:{}", actual_port);
let primary_handle = tokio::spawn(async move {
let _ = axum::serve(listener_primary, router_primary).await;
});
let dual_handle = if dual_port != 0 && dual_port != actual_port {
match bind_with_fallback(dual_port, 20).await {
Some((p, listener_dual)) => {
let router_dual = server::build_router();
let addr_dual = SocketAddr::from(([0, 0, 0, 0], p));
println!("[Amni-AI] Braid fallback listening on http://{}", addr_dual);
Some(tokio::spawn(async move {
let _ = axum::serve(listener_dual, router_dual).await;
}))
}
None => {
println!("[Amni-AI] Port {} and fallbacks unavailable; continuing with primary", dual_port);
None
}
}
} else { None };
let studio_url = format!("http://localhost:{}", actual_port);
println!("==============================================================");
println!("  ⚡ AMNI-AI NATIVE STUDIO ACTIVE");
println!("  Studio URL:       {}", studio_url);
println!("  Output Directory: workspace/outputs/");
println!("  Evaluator Swarm:  7 parallel models (~64KB memory)");
println!("==============================================================");
if !no_browser { open_browser(&studio_url); }
if let Some(h) = dual_handle {
let _ = tokio::try_join!(primary_handle, h);
} else {
let _ = primary_handle.await;
}
Ok(())
}
