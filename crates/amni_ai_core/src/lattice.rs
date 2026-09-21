use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Theme {
pub id: usize,
pub name: String,
pub archetype: String,
pub bg: String,
pub primary: String,
pub secondary: String,
pub accent: String,
pub border: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayoutMetrics {
pub mode: String,
pub padding_rem: f32,
pub gap_rem: f32,
pub border_radius_px: u32,
pub canvas_scale: f32,
}
pub struct ThemeSynthesizer;
impl ThemeSynthesizer {
pub fn generate_100() -> Vec<Theme> {
let archetypes: [(&str, f32, f32, f32); 8] = [
("cyberpunk", 180.0, 300.0, 0.95),
("solar", 35.0, 55.0, 0.85),
("midnight", 220.0, 260.0, 0.70),
("emerald", 130.0, 160.0, 0.80),
("vaporwave", 280.0, 330.0, 0.90),
("monolith", 0.0, 0.0, 0.10),
("sunset", 10.0, 45.0, 0.85),
("deepsea", 195.0, 225.0, 0.75),
];
let mut out = Vec::with_capacity(100);
for i in 0..100 {
let arch = archetypes[i % archetypes.len()];
let step = (i / archetypes.len()) as f32;
let hue1 = (arch.1 + step * 7.3) % 360.0;
let hue2 = (arch.2 + step * 9.1) % 360.0;
let sat = (arch.3 * 100.0).clamp(10.0, 100.0) as u32;
let (bg, border) = if arch.0 == "monolith" {
(format!("#{:02x}{:02x}{:02x}", 12 + i % 10, 12 + i % 10, 14 + i % 10), "#2a2d36".to_string())
} else {
(format!("hsl({:.0}, 25%, 7%)", hue1), format!("hsl({:.0}, 50%, 25%)", hue1))
};
out.push(Theme {
id: i,
name: format!("{}-variant-{:03}", arch.0, i),
archetype: arch.0.to_string(),
bg,
primary: format!("hsl({:.0}, {}%, 62%)", hue1, sat),
secondary: format!("hsl({:.0}, {}%, 75%)", hue2, sat.saturating_sub(15)),
accent: format!("hsl({:.0}, 95%, 68%)", (hue1 + 140.0) % 360.0),
border,
});
}
out
}
}
pub struct LayoutScaler;
impl LayoutScaler {
pub fn resolve(mode: &str) -> LayoutMetrics {
match mode.to_lowercase().as_str() {
"compact" => LayoutMetrics { mode: "compact".into(), padding_rem: 0.5, gap_rem: 0.4, border_radius_px: 4, canvas_scale: 0.85 },
"spacious" => LayoutMetrics { mode: "spacious".into(), padding_rem: 1.5, gap_rem: 1.2, border_radius_px: 12, canvas_scale: 1.15 },
_ => LayoutMetrics { mode: "normal".into(), padding_rem: 1.0, gap_rem: 0.8, border_radius_px: 8, canvas_scale: 1.0 },
}
}
}
pub struct MicroLatticeComposer;
impl MicroLatticeComposer {
pub fn synthesize_webgpu_app(theme_count: usize, layout_mode: &str) -> String {
let themes = ThemeSynthesizer::generate_100();
let themes_slice = &themes[..theme_count.min(themes.len())];
let layout = LayoutScaler::resolve(layout_mode);
let themes_json = serde_json::to_string(themes_slice).unwrap_or_else(|_| "[]".to_string());
format!(
r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Amni-AI WebGPU Micro-Lattice ({0} Themes, {1})</title>
<style>
:root{{--bg:#070a13;--primary:#38bdf8;--secondary:#818cf8;--accent:#f43f5e;--border:#1e293b;--pad:{2}rem;--gap:{3}rem;--rad:{4}px;}}
*{{margin:0;padding:0;box-sizing:border-box;font-family:system-ui,-apple-system,sans-serif;}}
body{{background:var(--bg);color:#f8fafc;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:var(--pad);transition:background 0.3s,color 0.3s;}}
header{{width:100%;max-width:1100px;display:flex;justify-content:space-between;align-items:center;padding:var(--gap) 0;border-bottom:1px solid var(--border);margin-bottom:var(--gap);}}
h1{{font-size:1.2rem;font-weight:700;color:var(--primary);letter-spacing:0.05em;}}
.controls{{display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;}}
select,button{{background:rgba(255,255,255,0.06);color:inherit;border:1px solid var(--border);border-radius:var(--rad);padding:0.4rem 0.8rem;cursor:pointer;font-size:0.85rem;}}
select:focus,button:hover{{border-color:var(--accent);outline:none;}}
.viewport-card{{width:100%;max-width:1100px;background:rgba(15,23,42,0.6);border:1px solid var(--border);border-radius:var(--rad);padding:var(--gap);display:flex;flex-direction:column;align-items:center;position:relative;}}
canvas{{width:100%;height:520px;max-width:1000px;border-radius:var(--rad);background:#020617;touch-action:none;}}
.hud{{width:100%;max-width:1000px;display:flex;justify-content:space-between;margin-top:0.6rem;font-size:0.75rem;color:#94a3b8;}}
.badge{{color:var(--accent);font-weight:600;}}
</style>
</head>
<body>
<header>
<h1>AMNI-AI WEBGPU LATTICE</h1>
<div class="controls">
<label for="theme-sel">Theme:</label>
<select id="theme-sel"></select>
<button id="anim-btn">Pause Motion</button>
<button id="reset-cam">Reset View</button>
</div>
</header>
<main class="viewport-card">
<canvas id="gpu-canvas"></canvas>
<div class="hud">
<span>PIPELINE: <span id="pipe-status" class="badge">Initializing...</span></span>
<span>INSTANCES: <span class="badge">12 Meshes (Single Draw)</span></span>
<span>FPS: <span id="fps-val" class="badge">60</span></span>
</div>
</main>
<script>
const THEMES = {5};
const sel = document.getElementById('theme-sel');
THEMES.forEach((t, i) => {{
const opt = document.createElement('option');
opt.value = i;
opt.textContent = `[${{t.archetype.toUpperCase()}}] ${{t.name}}`;
sel.appendChild(opt);
}});
function applyTheme(t) {{
document.documentElement.style.setProperty('--bg', t.bg);
document.documentElement.style.setProperty('--primary', t.primary);
document.documentElement.style.setProperty('--secondary', t.secondary);
document.documentElement.style.setProperty('--accent', t.accent);
document.documentElement.style.setProperty('--border', t.border);
}}
applyTheme(THEMES[0]);
sel.addEventListener('change', (e) => applyTheme(THEMES[e.target.value]));
let rotating = true;
document.getElementById('anim-btn').onclick = () => {{
rotating = !rotating;
document.getElementById('anim-btn').textContent = rotating ? 'Pause Motion' : 'Resume Motion';
}};
let camAngle = 0, camElevation = 0.3;
document.getElementById('reset-cam').onclick = () => {{ camAngle = 0; camElevation = 0.3; }};
const cvs = document.getElementById('gpu-canvas');
cvs.width = Math.floor(cvs.clientWidth * window.devicePixelRatio);
cvs.height = Math.floor(cvs.clientHeight * window.devicePixelRatio);
async function runEngine() {{
const pipeEl = document.getElementById('pipe-status');
if (!navigator.gpu) {{
pipeEl.textContent = 'Software Canvas Fallback (No WebGPU)';
runCanvasFallback();
return;
}}
try {{
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error('No GPUAdapter');
const device = await adapter.requestDevice();
const ctx = cvs.getContext('webgpu');
const format = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({{ device, format, alphaMode: 'premultiplied' }});
const shaderCode = `@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> @builtin(position) vec4f {{
var pos = array<vec2f, 3>(vec2f(0.0, 0.4), vec2f(-0.35, -0.3), vec2f(0.35, -0.3));
let col = f32(iid % 4u) - 1.5;
let row = f32(iid / 4u) - 1.0;
let p = pos[vid] * 0.45 + vec2f(col * 0.45, row * 0.55);
return vec4f(p, 0.0, 1.0);
}}
@fragment
fn fs_main() -> @location(0) vec4f {{
return vec4f(0.22, 0.74, 0.97, 1.0);
}}`;
const shader = device.createShaderModule({{ code: shaderCode }});
const pipeline = device.createRenderPipeline({{
layout: 'auto',
vertex: {{ module: shader, entryPoint: 'vs_main' }},
fragment: {{ module: shader, entryPoint: 'fs_main', targets: [{{ format }}] }},
primitive: {{ topology: 'triangle-list' }}
}});
pipeEl.textContent = 'Active (WebGPU Native)';
let lastTime = performance.now(), frames = 0;
function frame(now) {{
frames++;
if (now - lastTime >= 1000) {{
document.getElementById('fps-val').textContent = frames;
frames = 0;
lastTime = now;
}}
const encoder = device.createCommandEncoder();
const pass = encoder.beginRenderPass({{
colorAttachments: [{{
view: ctx.getCurrentTexture().createView(),
clearValue: {{ r: 0.01, g: 0.03, b: 0.08, a: 1.0 }},
loadOp: 'clear',
storeOp: 'store'
}}]
}});
pass.setPipeline(pipeline);
pass.draw(3, 12, 0, 0);
pass.end();
device.queue.submit([encoder.finish()]);
requestAnimationFrame(frame);
}}
requestAnimationFrame(frame);
}} catch (err) {{
pipeEl.textContent = 'Software Canvas Fallback (' + err.message + ')';
runCanvasFallback();
}}
}}
function runCanvasFallback() {{
const ctx2d = cvs.getContext('2d');
let t = 0;
function drawFallback() {{
t += 0.02;
ctx2d.fillStyle = '#020617';
ctx2d.fillRect(0, 0, cvs.width, cvs.height);
for (let i = 0; i < 12; i++) {{
const col = (i % 4) - 1.5;
const row = Math.floor(i / 4) - 1.0;
const cx = cvs.width / 2 + col * 120;
const cy = cvs.height / 2 + row * 100;
ctx2d.save();
ctx2d.translate(cx, cy);
if (rotating) ctx2d.rotate(t + i * 0.4);
ctx2d.strokeStyle = '#38bdf8';
ctx2d.lineWidth = 2;
ctx2d.strokeRect(-24, -24, 48, 48);
ctx2d.restore();
}}
requestAnimationFrame(drawFallback);
}}
requestAnimationFrame(drawFallback);
}}
runEngine();
</script>
</body>
</html>"#,
theme_count, layout.mode, layout.padding_rem, layout.gap_rem, layout.border_radius_px, themes_json
)
}
}
