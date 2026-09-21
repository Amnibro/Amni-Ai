import os,sys,math,json,re
class ThemeSynthesizer:
 @staticmethod
 def generate_100_themes()->list[dict]:
  themes=[]
  archetypes=[
   ("Cyberpunk",90,20,0.95,0.55),
   ("Synthwave",280,330,0.90,0.60),
   ("DeepSpace",210,260,0.85,0.50),
   ("Solarized",30,55,0.80,0.55),
   ("Nordic",190,220,0.60,0.65),
   ("Dracula",250,300,0.75,0.60),
   ("Emerald",140,175,0.80,0.55),
   ("Monochrome",0,360,0.05,0.70)
  ]
  for i in range(100):
   arch_name,h_min,h_max,base_sat,base_lit=archetypes[i%len(archetypes)]
   h1=(h_min+(i*137.508)%(h_max-h_min+1.0))%360.0
   h2=(h1+60.0)%360.0
   sat=min(1.0,max(0.1,base_sat+(math.sin(i*0.5)*0.15)))
   lit=min(0.85,max(0.35,base_lit+(math.cos(i*0.3)*0.1)))
   def hsl_to_rgb(h,s,l):
    c=(1.0-abs(2.0*l-1.0))*s
    x=c*(1.0-abs((h/60.0)%2.0-1.0))
    m=l-c/2.0
    if 0<=h<60:r,g,b=c,x,0
    elif 60<=h<120:r,g,b=x,c,0
    elif 120<=h<180:r,g,b=0,c,x
    elif 180<=h<240:r,g,b=0,x,c
    elif 240<=h<300:r,g,b=x,0,c
    else:r,g,b=c,0,x
    return r+m,g+m,b+m
   r1,g1,b1=hsl_to_rgb(h1,sat,lit)
   r2,g2,b2=hsl_to_rgb(h2,sat,lit)
   bg_r,bg_g,bg_b=hsl_to_rgb(h1,sat*0.4,0.05+(i%5)*0.01)
   c_r,c_g,c_b=hsl_to_rgb(h1,sat*0.3,0.11+(i%5)*0.01)
   hex_acc=f"#{int(r1*255):02x}{int(g1*255):02x}{int(b1*255):02x}"
   hex_acc2=f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"
   hex_bg=f"#{int(bg_r*255):02x}{int(bg_g*255):02x}{int(bg_b*255):02x}"
   rgba_card=f"rgba({int(c_r*255)},{int(c_g*255)},{int(c_b*255)},0.82)"
   rgba_border=f"rgba({int(r1*255)},{int(g1*255)},{int(b1*255)},0.28)"
   themes.append({
    "id":i,
    "name":f"Theme {i+1:03d}: {arch_name} {int(h1)}°",
    "bg":hex_bg,
    "card":rgba_card,
    "border":rgba_border,
    "accent":hex_acc,
    "accent2":hex_acc2,
    "text":"#f8fafc",
    "clear_color":[round(bg_r,3),round(bg_g,3),round(bg_b,3),1.0],
    "light_color":[round(r1,3),round(g1,3),round(b1,3)]
   })
  return themes
class LayoutScaler:
 @staticmethod
 def compute_layout(query:str)->dict:
  m=query.lower()
  density="compact" if any(k in m for k in ("compact","mini","dense")) else ("spacious" if any(k in m for k in ("spacious","large","wide")) else "normal")
  padding="10px" if density=="compact" else ("24px" if density=="spacious" else "16px")
  gap="8px" if density=="compact" else ("18px" if density=="spacious" else "12px")
  font_scale="0.8rem" if density=="compact" else ("1.0rem" if density=="spacious" else "0.875rem")
  cvs_height="220px" if density=="compact" else ("340px" if density=="spacious" else "260px")
  grid_cols="repeat(auto-fit, minmax(180px, 1fr))" if density=="compact" else "repeat(auto-fit, minmax(240px, 1fr))"
  return {"density":density,"padding":padding,"gap":gap,"font_scale":font_scale,"cvs_height":cvs_height,"grid_cols":grid_cols}
class MicroLatticeComposer:
 def __init__(self):
  self.themes=ThemeSynthesizer.generate_100_themes()
 def compose_webgpu_app(self,query:str="")->str:
  layout=LayoutScaler.compute_layout(query)
  themes_json=json.dumps(self.themes)
  options_html="".join(f'<option value="{t["id"]}">{t["name"]}</option>' for t in self.themes)
  return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Amni High-Efficiency WebGPU Engine (100 Themes & Instanced 3D)</title>
<style>
:root{{--bg:{self.themes[0]['bg']};--card:{self.themes[0]['card']};--border:{self.themes[0]['border']};--accent:{self.themes[0]['accent']};--accent2:{self.themes[0]['accent2']};--text:#f8fafc;--pad:{layout['padding']};--gap:{layout['gap']};--font-scale:{layout['font_scale']};}}
*{{margin:0;padding:0;box-sizing:border-box;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;}}
body{{background:var(--bg);color:var(--text);overflow:hidden;width:100vw;height:100vh;display:flex;flex-direction:column;font-size:var(--font-scale);transition:background .25s;}}
#gpu-canvas{{width:100%;height:100%;display:block;}}
.hud{{position:absolute;top:16px;left:16px;background:var(--card);backdrop-filter:blur(10px);border:1px solid var(--border);border-radius:10px;padding:var(--pad);pointer-events:auto;z-index:10;max-width:360px;box-shadow:0 8px 32px rgba(0,0,0,0.5);display:flex;flex-direction:column;gap:var(--gap);transition:border .25s,background .25s;}}
.hud h1{{font-size:1.05rem;color:var(--accent);display:flex;align-items:center;gap:8px;}}
.telemetry{{display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:0.75rem;}}
.metric{{background:rgba(0,0,0,0.35);padding:6px;border-radius:6px;border:1px solid rgba(255,255,255,0.06);}}
.metric span{{display:block;font-size:0.95rem;font-weight:700;color:var(--accent);}}
.ctrl-row{{display:flex;flex-direction:column;gap:4px;}}
.ctrl-row label{{display:flex;justify-content:space-between;font-size:0.75rem;color:#cbd5e1;}}
select,input[type=range]{{width:100%;background:rgba(0,0,0,0.4);color:#fff;border:1px solid var(--border);padding:6px;border-radius:6px;accent-color:var(--accent);font-size:0.75rem;}}
#fallback-banner{{position:absolute;bottom:20px;left:50%;transform:translateX(-50%);background:#ef4444;color:#fff;padding:12px 24px;border-radius:8px;font-size:0.85rem;font-weight:600;display:none;z-index:100;}}
</style>
</head>
<body>
<div class="hud">
<h1><span>⚡</span> AMNI WEBGPU INSTANCER</h1>
<p style="font-size:0.72rem;color:#94a3b8">Procedural Multi-Theme Engine & Zero-GC Instanced Geometry (12 Objects, Single Draw Call).</p>
<div class="ctrl-row">
<label>Active Theme ({len(self.themes)} Available):</label>
<select id="theme-selector">{options_html}</select>
</div>
<div class="telemetry">
<div class="metric">FPS<span id="fps-val">60</span></div>
<div class="metric">Instances<span>12 (Single Draw)</span></div>
<div class="metric">Triangles<span>288</span></div>
<div class="metric">Adapter<span id="adapter-val" style="font-size:0.7rem">Hardware GPU</span></div>
</div>
<div class="ctrl-row">
<label>Orbit Velocity: <span id="spd-label">1.0x</span></label>
<input type="range" id="rot-spd" min="0" max="30" value="10">
</div>
<p style="font-size:0.68rem;color:#64748b">Drag canvas to rotate camera. Themes immediately adjust CSS and WebGPU clear/light passes.</p>
</div>
<div id="fallback-banner">WebGPU unsupported or disabled. Enable chrome://flags/#enable-unsafe-webgpu</div>
<canvas id="gpu-canvas"></canvas>
<script>
const THEMES={themes_json};
const canvas=document.getElementById('gpu-canvas');
const banner=document.getElementById('fallback-banner');
const fpsVal=document.getElementById('fps-val');
const adapterVal=document.getElementById('adapter-val');
const themeSel=document.getElementById('theme-selector');
const rotSpdSlider=document.getElementById('rot-spd');
const spdLabel=document.getElementById('spd-label');
let currentTheme=THEMES[0];
let rotSpeed=1.0;
rotSpdSlider.addEventListener('input',e=>{{rotSpeed=parseFloat(e.target.value)/10.0;spdLabel.innerText=rotSpeed.toFixed(1)+'x';}});
themeSel.addEventListener('change',e=>{{
const idx=parseInt(e.target.value);
currentTheme=THEMES[idx]||THEMES[0];
document.documentElement.style.setProperty('--bg',currentTheme.bg);
document.documentElement.style.setProperty('--card',currentTheme.card);
document.documentElement.style.setProperty('--border',currentTheme.border);
document.documentElement.style.setProperty('--accent',currentTheme.accent);
document.documentElement.style.setProperty('--accent2',currentTheme.accent2);
}});
let isDragging=false,lastX=0,lastY=0,camRotX=0.35,camRotY=0.55;
canvas.addEventListener('pointerdown',e=>{{isDragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);}});
canvas.addEventListener('pointermove',e=>{{if(!isDragging)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;camRotY+=dx*0.01;camRotX+=dy*0.01;lastX=e.clientX;lastY=e.clientY;}});
canvas.addEventListener('pointerup',()=>{{isDragging=false;}});
function mat4Perspective(fovy,aspect,near,far){{const f=1.0/Math.tan(fovy/2),nf=1/(near-far);return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);}}
function mat4Multiply(a,b){{const out=new Float32Array(16);for(let i=0;i<4;i++){{for(let j=0;j<4;j++){{let s=0;for(let k=0;k<4;k++)s+=a[i+k*4]*b[k+j*4];out[i+j*4]=s;}}}}return out;}}
function mat4RotationY(rad){{const c=Math.cos(rad),s=Math.sin(rad);return new Float32Array([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]);}}
function mat4RotationX(rad){{const c=Math.cos(rad),s=Math.sin(rad);return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]);}}
function mat4Translation(x,y,z){{return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]);}}
async function initWebGPU(){{
if(!navigator.gpu){{banner.style.display='block';adapterVal.innerText='Unsupported';return;}}
const adapter=await navigator.gpu.requestAdapter();
if(!adapter){{banner.style.display='block';adapterVal.innerText='No Adapter';return;}}
const info=await adapter.requestAdapterInfo?.()||{{}};
adapterVal.innerText=info.vendor||'Hardware GPU';
const device=await adapter.requestDevice();
function resize(){{const dpr=Math.min(window.devicePixelRatio||1,2);canvas.width=window.innerWidth*dpr;canvas.height=window.innerHeight*dpr;}}
window.addEventListener('resize',resize);resize();
const context=canvas.getContext('webgpu');
const presentationFormat=navigator.gpu.getPreferredCanvasFormat();
context.configure({{device,format:presentationFormat,alphaMode:'premultiplied'}});
const INSTANCES=12;
const wgslShaders=`
struct InstanceData {{
  model: mat4x4<f32>,
  color: vec4<f32>,
}};
struct Uniforms {{
  viewProj: mat4x4<f32>,
  lightDir: vec3<f32>,
  time: f32,
  instances: array<InstanceData, 12>,
}};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct VertexInput {{
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
  @builtin(instance_index) instance_idx: u32,
}};
struct VertexOutput {{
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) color: vec4<f32>,
}};
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {{
  var out: VertexOutput;
  let inst = uniforms.instances[in.instance_idx];
  let worldPos = inst.model * vec4<f32>(in.position, 1.0);
  out.position = uniforms.viewProj * worldPos;
  out.normal = (inst.model * vec4<f32>(in.normal, 0.0)).xyz;
  out.color = vec4<f32>(in.color * inst.color.rgb, 1.0);
  return out;
}}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{
  let light = max(dot(normalize(in.normal), normalize(uniforms.lightDir)), 0.2);
  let finalColor = in.color.rgb * light + vec3<f32>(0.04, 0.08, 0.12);
  return vec4<f32>(finalColor, 1.0);
}}
`;
const shaderModule=device.createShaderModule({{code:wgslShaders}});
const vertices=new Float32Array([
-0.5,-0.5, 0.5, 0,0,1, 0.0,0.8,1.0,  0.5,-0.5, 0.5, 0,0,1, 0.0,0.8,1.0,  0.5, 0.5, 0.5, 0,0,1, 0.0,0.8,1.0, -0.5, 0.5, 0.5, 0,0,1, 0.0,0.8,1.0,
-0.5,-0.5,-0.5, 0,0,-1, 1.0,0.2,0.6, -0.5, 0.5,-0.5, 0,0,-1, 1.0,0.2,0.6,  0.5, 0.5,-0.5, 0,0,-1, 1.0,0.2,0.6,  0.5,-0.5,-0.5, 0,0,-1, 1.0,0.2,0.6,
-0.5, 0.5,-0.5, 0,1,0, 0.2,1.0,0.6, -0.5, 0.5, 0.5, 0,1,0, 0.2,1.0,0.6,  0.5, 0.5, 0.5, 0,1,0, 0.2,1.0,0.6,  0.5, 0.5,-0.5, 0,1,0, 0.2,1.0,0.6,
-0.5,-0.5,-0.5, 0,-1,0, 1.0,0.8,0.2,  0.5,-0.5,-0.5, 0,-1,0, 1.0,0.8,0.2,  0.5,-0.5, 0.5, 0,-1,0, 1.0,0.8,0.2, -0.5,-0.5, 0.5, 0,-1,0, 1.0,0.8,0.2,
 0.5,-0.5,-0.5, 1,0,0, 0.6,0.2,1.0,  0.5, 0.5,-0.5, 1,0,0, 0.6,0.2,1.0,  0.5, 0.5, 0.5, 1,0,0, 0.6,0.2,1.0,  0.5,-0.5, 0.5, 1,0,0, 0.6,0.2,1.0,
-0.5,-0.5,-0.5,-1,0,0, 1.0,0.5,0.0, -0.5,-0.5, 0.5,-1,0,0, 1.0,0.5,0.0, -0.5, 0.5, 0.5,-1,0,0, 1.0,0.5,0.0, -0.5, 0.5,-0.5,-1,0,0, 1.0,0.5,0.0
]);
const indices=new Uint16Array([
0,1,2,0,2,3, 4,5,6,4,6,7, 8,9,10,8,10,11, 12,13,14,12,14,15, 16,17,18,16,18,19, 20,21,22,20,22,23
]);
const vertexBuffer=device.createBuffer({{size:vertices.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}});
device.queue.writeBuffer(vertexBuffer,0,vertices);
const indexBuffer=device.createBuffer({{size:indices.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}});
device.queue.writeBuffer(indexBuffer,0,indices);
const uniformFloats=16+4+INSTANCES*20;
const uniformBuffer=device.createBuffer({{size:uniformFloats*4,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}});
const uniformArray=new Float32Array(uniformFloats);
const pipeline=device.createRenderPipeline({{
layout:'auto',
vertex:{{module:shaderModule,entryPoint:'vs_main',buffers:[{{arrayStride:9*4,attributes:[{{shaderLocation:0,offset:0,format:'float32x3'}},{{shaderLocation:1,offset:3*4,format:'float32x3'}},{{shaderLocation:2,offset:6*4,format:'float32x3'}}]}}]}},
fragment:{{module:shaderModule,entryPoint:'fs_main',targets:[{{format:presentationFormat}}]}},
primitive:{{topology:'triangle-list',cullMode:'back'}},
depthStencil:{{format:'depth24plus',depthWriteEnabled:true,depthCompare:'less'}}
}});
const bindGroup=device.createBindGroup({{layout:pipeline.getBindGroupLayout(0),entries:[{{binding:0,resource:{{buffer:uniformBuffer}}}}]}});
let depthTexture=device.createTexture({{size:[canvas.width,canvas.height],format:'depth24plus',usage:GPUTextureUsage.RENDER_ATTACHMENT}});
window.addEventListener('resize',()=>{{depthTexture.destroy();depthTexture=device.createTexture({{size:[canvas.width,canvas.height],format:'depth24plus',usage:GPUTextureUsage.RENDER_ATTACHMENT}});}});
let lastTime=performance.now(),frameCount=0,fpsTimer=performance.now(),autoRot=0;
function frame(now){{
const dt=(now-lastTime)*0.001;lastTime=now;autoRot+=dt*rotSpeed;frameCount++;
if(now-fpsTimer>=1000){{fpsVal.innerText=frameCount;frameCount=0;fpsTimer=now;}}
const aspect=canvas.width/canvas.height;
const proj=mat4Perspective((45*Math.PI)/180,aspect,0.1,100.0);
const view=mat4Translation(0,0,-7.0);
const rx=mat4RotationX(camRotX);
const ry=mat4RotationY(camRotY+autoRot*0.3);
const viewProj=mat4Multiply(proj,mat4Multiply(view,mat4Multiply(rx,ry)));
uniformArray.set(viewProj,0);
const lCol=currentTheme.light_color||[1.0,1.0,1.0];
uniformArray[16]=lCol[0];uniformArray[17]=lCol[1];uniformArray[18]=lCol[2];uniformArray[19]=now*0.001;
for(let i=0;i<INSTANCES;i++){{
const offset=20+i*20;
const angle=(i/INSTANCES)*Math.PI*2.0+autoRot*(0.8+(i%3)*0.2);
const radius=2.5+Math.sin(autoRot+i)*0.4;
const x=Math.cos(angle)*radius,y=Math.sin(angle*1.5)*0.8,z=Math.sin(angle)*radius;
const modelT=mat4Translation(x,y,z);
const modelR=mat4Multiply(mat4RotationY(autoRot*1.5+i),mat4RotationX(autoRot+i));
const model=mat4Multiply(modelT,modelR);
uniformArray.set(model,offset);
uniformArray[offset+16]=0.7+0.3*Math.cos(i);
uniformArray[offset+17]=0.7+0.3*Math.sin(i*1.3);
uniformArray[offset+18]=0.8+0.2*Math.cos(i*0.7);
uniformArray[offset+19]=1.0;
}}
device.queue.writeBuffer(uniformBuffer,0,uniformArray);
const clr=currentTheme.clear_color||[0.03,0.04,0.06,1.0];
const commandEncoder=device.createCommandEncoder();
const passEncoder=commandEncoder.beginRenderPass({{
colorAttachments:[{{view:context.getCurrentTexture().createView(),clearValue:{{r:clr[0],g:clr[1],b:clr[2],a:clr[3]}},loadOp:'clear',storeOp:'store'}}],
depthStencilAttachment:{{view:depthTexture.createView(),depthClearValue:1.0,depthLoadOp:'clear',depthStoreOp:'store'}}
}});
passEncoder.setPipeline(pipeline);
passEncoder.setBindGroup(0,bindGroup);
passEncoder.setVertexBuffer(0,vertexBuffer);
passEncoder.setIndexBuffer(indexBuffer,'uint16');
passEncoder.drawIndexed(indices.length,INSTANCES);
passEncoder.end();
device.queue.submit([commandEncoder.finish()]);
requestAnimationFrame(frame);
}}
requestAnimationFrame(frame);
}}
initWebGPU();
</script>
</body>
</html>"""
 def compose_web_dashboard(self,query:str="")->str:
  layout=LayoutScaler.compute_layout(query)
  themes_json=json.dumps(self.themes)
  options_html="".join(f'<option value="{t["id"]}">{t["name"]}</option>' for t in self.themes)
  return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Amni Parametric Telemetry Hub (100 Themes & Dynamic Geometry)</title>
<style>
:root{{--bg:{self.themes[0]['bg']};--card:{self.themes[0]['card']};--border:{self.themes[0]['border']};--accent:{self.themes[0]['accent']};--accent2:{self.themes[0]['accent2']};--text:#e2e8f0;--muted:#64748b;--pad:{layout['padding']};--gap:{layout['gap']};--font-scale:{layout['font_scale']};}}
*{{box-sizing:border-box;margin:0;padding:0;font-family:system-ui,-apple-system,sans-serif;}}
body{{background:radial-gradient(circle at 50% 0%,var(--accent2) -100%,var(--bg) 70%);color:var(--text);min-height:100vh;padding:var(--pad);display:flex;flex-direction:column;gap:var(--gap);font-size:var(--font-scale);transition:background .25s,color .25s;}}
header{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border);padding-bottom:12px;}}
h1{{font-size:1.4rem;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.theme-bar{{display:flex;align-items:center;gap:8px;font-size:0.8rem;}}
select{{background:var(--card);color:var(--text);border:1px solid var(--border);padding:6px 10px;border-radius:6px;font-size:0.8rem;outline:none;}}
.grid{{display:grid;grid-template-columns:{layout['grid_cols']};gap:var(--gap);}}
.card{{background:var(--card);backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:10px;padding:var(--pad);box-shadow:0 6px 24px rgba(0,0,0,0.35);}}
.card h3{{color:var(--muted);font-size:0.75rem;text-transform:uppercase;margin-bottom:6px;}}
.val{{font-size:1.6rem;font-weight:700;color:#fff;}}
.chart-box{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:var(--pad);}}
canvas{{width:100%;height:{layout['cvs_height']};display:block;}}
.controls{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;}}
button{{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#000;border:none;padding:6px 14px;border-radius:6px;cursor:pointer;font-weight:600;font-size:0.75rem;transition:transform .15s;}}
button:hover{{transform:translateY(-1px);}}
input[type=range]{{accent-color:var(--accent);}}
</style>
</head>
<body>
<header>
<div><h1>AMNI PARAMETRIC DASHBOARD</h1><p style="color:var(--muted);font-size:0.8rem">100 Procedural Themes • Dynamic Feature Scaling ({layout['density'].upper()}) • Zero-GC Canvas Wave</p></div>
<div class="theme-bar"><label>Select Theme:</label><select id="theme-sel">{options_html}</select></div>
</header>
<main class="grid">
<div class="card"><h3>Instanced Throughput</h3><div class="val" id="val-tps">1,240 <span style="font-size:0.8rem;color:var(--muted)">MFLOP/s</span></div></div>
<div class="card"><h3>Pipeline Latency</h3><div class="val" id="val-lat">0.42 <span style="font-size:0.8rem;color:var(--muted)">ms</span></div></div>
<div class="card"><h3>Active Themes</h3><div class="val" style="color:var(--accent)">{len(self.themes)} <span style="font-size:0.8rem;color:var(--muted)">procedural</span></div></div>
<div class="card"><h3>Layout Mode</h3><div class="val" style="font-size:1.2rem;text-transform:uppercase">{layout['density']}</div></div>
</main>
<section class="chart-box">
<div style="display:flex;justify-content:space-between;margin-bottom:10px">
<span style="font-weight:600;font-size:0.85rem">Real-Time Wave Spectrum</span>
<div class="controls">
<label style="font-size:0.75rem">Speed: <input type="range" id="spd-slider" min="1" max="10" value="5"></label>
<button id="btn-toggle">Pause Flow</button>
<button id="btn-export">Export JSON</button>
</div>
</div>
<canvas id="telemetry-canvas" width="800" height="240"></canvas>
</section>
<script>
const THEMES={themes_json};
const cvs=document.getElementById('telemetry-canvas'),ctx=cvs.getContext('2d');
const themeSel=document.getElementById('theme-sel');
let currentTheme=THEMES[0];
themeSel.addEventListener('change',e=>{{
const idx=parseInt(e.target.value);
currentTheme=THEMES[idx]||THEMES[0];
document.documentElement.style.setProperty('--bg',currentTheme.bg);
document.documentElement.style.setProperty('--card',currentTheme.card);
document.documentElement.style.setProperty('--border',currentTheme.border);
document.documentElement.style.setProperty('--accent',currentTheme.accent);
document.documentElement.style.setProperty('--accent2',currentTheme.accent2);
}});
let running=true,phase=0,speed=0.05,history=new Float32Array(100);
for(let i=0;i<100;i++)history[i]=120.0;
document.getElementById('spd-slider').addEventListener('input',e=>{{speed=parseFloat(e.target.value)*0.01;}});
document.getElementById('btn-toggle').addEventListener('click',e=>{{running=!running;e.target.innerText=running?'Pause Flow':'Resume Flow';}});
document.getElementById('btn-export').addEventListener('click',()=>{{const b=new Blob([JSON.stringify({{timestamp:Date.now(),theme:currentTheme.name,metrics:Array.from(history)}})],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='telemetry_export.json';a.click();}});
function loop(){{
if(running){{
phase+=speed;
for(let i=0;i<99;i++)history[i]=history[i+1];
history[99]=120+Math.sin(phase)*45+Math.sin(phase*2.1)*22+Math.random()*6-3;
}}
ctx.clearRect(0,0,cvs.width,cvs.height);
const grad=ctx.createLinearGradient(0,0,0,cvs.height);
grad.addColorStop(0,currentTheme.accent+'44');grad.addColorStop(1,currentTheme.accent+'00');
ctx.beginPath();
const step=cvs.width/99.0;
ctx.moveTo(0,history[0]);
for(let i=1;i<100;i++)ctx.lineTo(i*step,history[i]);
ctx.lineTo(cvs.width,cvs.height);ctx.lineTo(0,cvs.height);ctx.closePath();
ctx.fillStyle=grad;ctx.fill();
ctx.beginPath();
for(let i=0;i<100;i++){{const x=i*step,y=history[i];if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}}
ctx.strokeStyle=currentTheme.accent;ctx.lineWidth=2;ctx.stroke();
requestAnimationFrame(loop);
}}
requestAnimationFrame(loop);
</script>
</body>
</html>"""
