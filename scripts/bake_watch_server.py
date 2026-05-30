import os,sys,io,json,time,argparse
from pathlib import Path
import numpy as np
from flask import Flask,request,jsonify,Response
from PIL import Image
_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(_ROOT))
from amni.compute.reffelt4 import decode_rgba4_to_fp16
app=Flask(__name__)
def _load(p):
    try:return json.loads(Path(p).read_text())
    except Exception:return {}
def scan(bake):
    bd=Path(bake);td=bd/'tensors';man=_load(bd/'manifest.json');exp=_load(bd/'_expected.json')
    if not(bake and (td.exists() or man or exp)):return {'ok':True,'phase':'idle','model':None,'total':0,'done':0,'bytes':0,'gb':0.0,'pct':0.0,'finished':False,'rate':0.0,'eta':0,'current':None,'recent':[],'ratio':None,'params':0,'dl_gb':0.0}
    files=sorted(td.glob('*.ptex'),key=lambda p:p.stat().st_mtime) if td.exists() else []
    done=len(files);bts=sum(p.stat().st_size for p in files);total=int(man.get('tensor_count') or exp.get('total') or 0)
    finished=bool(man);mt=[p.stat().st_mtime for p in files];elapsed=(mt[-1]-mt[0]) if len(mt)>1 else 0.0
    rate=(done/elapsed) if elapsed>0 else 0.0;eta=((total-done)/rate) if (rate>0 and total>done) else 0.0
    dest=exp.get('dest');dl=sum(p.stat().st_size for p in Path(dest).glob('*.safetensors')) if dest and Path(dest).exists() else 0
    phase='done' if finished else('baking' if done>0 else(exp.get('phase') or 'preparing'))
    recent=[{'name':p.name,'kb':round(p.stat().st_size/1024,1)} for p in files[-10:][::-1]]
    return {'ok':True,'phase':phase,'model':(man.get('model_name') or exp.get('model') or bd.name),'total':total,'done':done,'bytes':bts,'gb':round(bts/1e9,3),'pct':round(100*done/total,1) if total else 0.0,'finished':finished,'rate':round(rate,2),'eta':int(eta),'current':files[-1].name if files else None,'recent':recent,'ratio':man.get('compression_ratio'),'params':int(man.get('total_params') or exp.get('total_params') or 0),'dl_gb':round(dl/1e9,2)}
def list_tensors(bake,limit=6000):
    td=Path(bake)/'tensors'
    return [] if not td.exists() else [{'file':p.name,'kb':round(p.stat().st_size/1024,1)} for p in sorted(td.glob('*.ptex'))[:limit]]
def _cmap(x):
    s=np.array([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],dtype=np.float32)
    x=np.clip(x,0,1)*(len(s)-1);lo=np.floor(x).astype(int);hi=np.clip(lo+1,0,len(s)-1);f=(x-lo)[...,None]
    return (s[lo]*(1-f)+s[hi]*f).astype(np.uint8)
def render_atlas(bake,name,maxpx):
    bd=Path(bake);p=bd/'tensors'/name;man=_load(bd/'manifest.json')
    if not p.exists():return None
    raw=np.fromfile(p,dtype=np.uint8);tp=raw.size//4
    if tp==0:return None
    pw=0
    for v in man.get('tensors',{}).values():
        if v.get('ptex_path','').endswith('/'+name):pw=int(v.get('page_w') or 0);break
    pw=pw if pw>0 else (int(tp**0.5) or 1);ph=tp//pw
    if ph*pw==0:return None
    w=decode_rgba4_to_fp16(raw.reshape(-1,4)[:ph*pw]).astype(np.float32).reshape(ph,pw)
    a=np.log1p(np.abs(w));mx=float(a.max()) or 1.0;img=Image.fromarray(_cmap(a/mx),'RGB');img.thumbnail((maxpx,maxpx));buf=io.BytesIO();img.save(buf,'PNG');return buf.getvalue()
@app.route('/api/status')
def _status():return jsonify(scan(request.args.get('bake','')))
@app.route('/api/list')
def _list():return jsonify(list_tensors(request.args.get('bake','')))
@app.route('/api/atlas')
def _atlas():
    png=render_atlas(request.args.get('bake',''),request.args.get('name',''),int(request.args.get('max',256)))
    return (Response(png,mimetype='image/png') if png else ('',404))
@app.route('/')
def _index():return Response(HTML,mimetype='text/html')
HTML=r"""<!doctype html><html><head><meta charset=utf-8><title>Adam Bake — live</title>
<style>
:root{--bg:#0b0e14;--fg:#e6edf3;--mut:#7d8aa0;--ac:#4ade80;--ac2:#38bdf8;--pan:#121826}
*{box-sizing:border-box}body{margin:0;font:13px ui-monospace,Menlo,Consolas,monospace;background:var(--bg);color:var(--fg)}
header{padding:12px 18px;border-bottom:1px solid #1e2636;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;color:var(--ac)}.sub{color:var(--mut)}
.bar{flex:1;min-width:240px;height:16px;background:#1a2233;border-radius:8px;overflow:hidden}
.fill{height:100%;width:0;background:linear-gradient(90deg,var(--ac2),var(--ac));transition:width .4s}
.tiles{display:flex;gap:14px;flex-wrap:wrap}.tile{background:var(--pan);border:1px solid #1e2636;border-radius:8px;padding:6px 12px}
.tile b{font-size:16px}.tile span{color:var(--mut);font-size:11px;display:block}
.tabs{display:flex;gap:8px;padding:8px 18px}.tabs button{background:var(--pan);color:var(--fg);border:1px solid #283349;padding:6px 14px;border-radius:6px;cursor:pointer}
.tabs button.on{border-color:var(--ac);color:var(--ac)}
#wrap{position:absolute;inset:160px 0 0 0}.view{display:none;height:100%}.view.on{display:block}
#g2d{display:grid;grid-template-columns:320px 1fr;height:100%}
#chips{overflow:auto;border-right:1px solid #1e2636;padding:8px}
.chip{padding:4px 8px;border-radius:5px;cursor:pointer;color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chip:hover{background:#16203200}.chip.on{background:#16243a;color:var(--ac2)}
#canwrap{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;padding:16px}
#atlas{image-rendering:pixelated;max-width:90%;max-height:80%;border:1px solid #283349;background:#000}
#3d{width:100%;height:100%;display:block}
#empty{position:absolute;inset:0;display:none;flex-direction:column;align-items:center;justify-content:center;text-align:center;color:#9fb0c8;font-size:17px;line-height:1.7;pointer-events:none;z-index:5}
#legend{position:absolute;top:10px;left:14px;font-size:11px;color:#9fb0c8;z-index:6;line-height:1.9;pointer-events:none}
label.fl{color:var(--mut);font-size:12px;cursor:pointer}
</style></head><body>
<header>
<h1>◣ ADAM BAKE</h1><div class=sub id=model>—</div>
<div class=bar><div class=fill id=fill></div></div>
<div class=tiles>
<div class=tile><b id=t_done>0</b><span>tensors / <i id=t_total>?</i></span></div>
<div class=tile><b id=t_gb>0</b><span>GB on disk</span></div>
<div class=tile><b id=t_rate>0</b><span>tensors/s</span></div>
<div class=tile><b id=t_eta>—</b><span>eta</span></div>
<div class=tile><b id=t_state>booting</b><span>state</span></div>
</div></header>
<div class=tabs><button id=b2 class=on onclick=tab(0)>2D ATLAS</button><button id=b3 onclick=tab(1)>3D MODEL MAP</button></div>
<div id=wrap>
<div id=empty></div>
<div class="view on" id=v2><div id=g2d><div id=chips></div>
<div id=canwrap><label class=fl><input type=checkbox id=follow checked> follow newest</label><canvas id=atlas width=256 height=256></canvas><div class=sub id=aname>click a tensor</div></div></div></div>
<div class=view id=v3><canvas id=3d></canvas><div id=legend></div></div>
</div>
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const BAKE=new URLSearchParams(location.search).get('bake')||'';
const $=i=>document.getElementById(i);let cur=null,sel=null,meshes={},scene,cam,ren,ctrl,raf;
function fmtEta(s){if(!s)return'—';let m=Math.floor(s/60),h=Math.floor(m/60);return h?h+'h'+(m%60)+'m':(m?m+'m'+(s%60)+'s':s+'s')}
async function poll(){try{const s=await(await fetch('/api/status?bake='+encodeURIComponent(BAKE))).json();
$('model').textContent=s.model+(s.params?'  ·  '+(s.params/1e9).toFixed(2)+'B params':'');
$('fill').style.width=s.pct+'%';$('t_done').textContent=s.done;$('t_total').textContent=s.total||'?';
$('t_gb').textContent=s.gb;$('t_rate').textContent=s.rate;$('t_eta').textContent=fmtEta(s.eta);
const PH={done:['✓ lossless'+(s.ratio?' '+s.ratio.toFixed(2)+'x':''),'#4ade80'],baking:['baking…','#fbbf24'],downloading:['⬇ '+s.dl_gb+'GB','#38bdf8'],preparing:['preparing…','#fbbf24'],idle:['no bake','#7d8aa0']};
const ph=PH[s.phase]||['…','#7d8aa0'];$('t_state').textContent=ph[0];$('t_state').style.color=ph[1];
const ov=$('empty');ov.style.display=s.done>0?'none':'flex';
if(s.done===0)ov.innerHTML=s.phase==='idle'?'No bake selected<br><span class=sub>open <b>?bake=bakes/granite41_8b_gf17</b></span>':(s.phase==='downloading'?'⬇ Downloading weights… <b>'+s.dl_gb+' GB</b><br><span class=sub>atlases appear here once baking starts (~min for 16GB)</span>':'Waiting for bake to start…<br><span class=sub>each tensor lights up here as it is born</span>');
if(s.current&&s.current!==cur){cur=s.current;if($('follow').checked)showAtlas(cur)}}catch(e){}}
async function refreshList(){try{const L=await(await fetch('/api/list?bake='+encodeURIComponent(BAKE))).json();
const c=$('chips');if(L.length!==c.childElementCount){c.innerHTML='';L.forEach(t=>{const d=document.createElement('div');d.className='chip'+(t.file===sel?' on':'');d.textContent=t.file.replace('.ptex','').replace(/^model_/,'');d.title=t.file+'  '+t.kb+'KB';d.onclick=()=>showAtlas(t.file);c.appendChild(d)})}
if(scene)syncMeshes(L)}catch(e){}}
function showAtlas(name){sel=name;$('aname').textContent=name;[...$('chips').children].forEach(d=>d.classList.toggle('on',d.title&&d.title.startsWith(name)));
const img=new Image();img.onload=()=>{const cv=$('atlas');cv.width=img.width;cv.height=img.height;cv.getContext('2d').drawImage(img,0,0)};img.src='/api/atlas?max=256&bake='+encodeURIComponent(BAKE)+'&name='+encodeURIComponent(name)+'&_='+Date.now()}
const KIND=[['embed',0,0x8b5cf6],['q_proj',1,0x38bdf8],['k_proj',2,0x22d3ee],['v_proj',3,0x2dd4bf],['o_proj',4,0x4ade80],['gate_proj',5,0xfbbf24],['up_proj',6,0xf59e0b],['down_proj',7,0xf97316],['lm_head',8,0xf472b6],['norm',9,0x64748b]];
function kindOf(f){for(const k of KIND)if(f.includes(k[0]))return k;return['other',10,0x475569]}
function layerOf(f){const m=f.match(/layers_(\d+)/);return m?+m[1]:-1}
let grp,camSet=false;
function init3d(){scene=new THREE.Scene();scene.background=new THREE.Color(0x0b0e14);
cam=new THREE.PerspectiveCamera(55,1,.1,4000);cam.position.set(0,30,120);
ren=new THREE.WebGLRenderer({canvas:$('3d'),antialias:true});scene.add(new THREE.AmbientLight(0xffffff,.85));
const dl=new THREE.DirectionalLight(0xffffff,.7);dl.position.set(40,90,60);scene.add(dl);grp=new THREE.Group();scene.add(grp);
ctrl=THREE.OrbitControls?new THREE.OrbitControls(cam,ren.domElement):null;if(ctrl){ctrl.autoRotate=true;ctrl.autoRotateSpeed=.9;ctrl.enableDamping=true}
$('legend').innerHTML='drag to orbit · columns = tensor type · depth = layer<br>'+KIND.map(k=>'<span style="color:#'+k[2].toString(16).padStart(6,'0')+'">■</span> '+k[0]).join('  ');
resize3d();(function loop(){raf=requestAnimationFrame(loop);if(ctrl)ctrl.update();else grp.rotation.y+=.003;ren.render(scene,cam)})()}
function recenter(){const b=new THREE.Box3().setFromObject(grp);if(!isFinite(b.min.x))return;const c=b.getCenter(new THREE.Vector3()),sz=b.getSize(new THREE.Vector3()),r=Math.max(sz.x,sz.y,sz.z,10);if(ctrl)ctrl.target.copy(c);if(!camSet){cam.position.set(c.x,c.y+r*.4,c.z+r*1.4);camSet=true}}
function syncMeshes(L){let added=0;L.forEach(t=>{const f=t.file;if(meshes[f])return;const k=kindOf(f),ly=layerOf(f);
const h=Math.max(.6,Math.log10(Math.max(2,t.kb))*1.7),g=new THREE.BoxGeometry(1.6,h,1.6);
const m=new THREE.MeshLambertMaterial({color:k[2],emissive:k[2],emissiveIntensity:.4}),box=new THREE.Mesh(g,m);
box.position.set((k[1]-5)*2.6,h/2,(ly<0?-2:ly)*1.7);box.scale.set(.01,.01,.01);meshes[f]=box;grp.add(box);added++;
let s=.01;const grow=()=>{s+=(1-s)*.18;box.scale.set(s,s,s);if(s<.98)requestAnimationFrame(grow)};grow()});if(added)recenter()}
function tab(i){$('b2').classList.toggle('on',!i);$('b3').classList.toggle('on',!!i);$('v2').classList.toggle('on',!i);$('v3').classList.toggle('on',!!i);
if(i&&!scene&&window.THREE){init3d();refreshList()}if(i)resize3d()}
function resize3d(){if(!ren)return;const w=$('v3').clientWidth,h=$('v3').clientHeight;ren.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix()}
addEventListener('resize',resize3d);poll();refreshList();setInterval(poll,1500);setInterval(refreshList,3000);
</script></body></html>"""
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=7788);ap.add_argument('--host',default='127.0.0.1');a=ap.parse_args()
    print(f'[bake-watch] http://{a.host}:{a.port}/?bake=<bake_dir>  (e.g. ?bake=bakes/granite41_8b_gf17)',flush=True)
    app.run(host=a.host,port=a.port,threaded=True)
if __name__=='__main__':main()
