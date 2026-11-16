"""GF(17)-ATEX 4-bit bake of a dense HF causal LM (default: ibm-granite/granite-4.2-3b; granite-4.1-3b/8b still work) -> GPU-resident, int4 group-128, decode-on-kernel (same recipe as bake_gf17_atex_4bit but AutoModelForCausalLM + granite paths). Quantizes each Linear to int4 group-128, packs codes to RGBA, keeps embed/lm_head + non-Linears bf16. Out: bakes/granite42_3b_gf17_atex/.
Run (GPU not required: packing is pure torch on CPU, ~6.5 GB RAM for 3B):
  python scripts/bake_granite_atex.py                                  # downloads ibm-granite/granite-4.2-3b if --src is missing
  python scripts/bake_granite_atex.py --src downloaded_models/granite-4.1-3b --out bakes/granite41_3b_gf17_atex   # legacy
"""
import torch,torch.nn as nn,torch.nn.functional as F,os,glob,gc,json,time,argparse,safetensors.torch as st
from transformers import AutoConfig,AutoModelForCausalLM
from accelerate import init_empty_weights
from safetensors import safe_open
_ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument('--hf-id',default='ibm-granite/granite-4.2-3b',help='HF repo to fetch when --src does not exist')
_ap.add_argument('--src',default=None,help='local HF snapshot dir (default: downloaded_models/<repo name>)')
_ap.add_argument('--out',default=None,help='bake output dir (default: bakes/<base_models bake_dir>)')
_ap.add_argument('--gs',type=int,default=128)
_ap.add_argument('--model-name',default=None,help='accepted for model_installer compatibility (unused)')
_a=_ap.parse_args()
import sys;sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amni.utils.base_models import resolve as _resolve
_bm=_resolve(_a.hf_id) or {}
P=_a.src or os.path.join('downloaded_models',_a.hf_id.split('/')[-1])
OUT=_a.out or os.path.join('bakes',_bm.get('bake_dir') or (_a.hf_id.split('/')[-1].replace('.','').replace('-','_')+'_gf17_atex'));GS=_a.gs
if not glob.glob(os.path.join(P,'*.safetensors')):
    from huggingface_hub import snapshot_download
    print(f'fetching {_a.hf_id} -> {P}',flush=True);snapshot_download(repo_id=_a.hf_id,local_dir=P,allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','tokenizer*'])
os.makedirs(OUT,exist_ok=True)
cfg=AutoConfig.from_pretrained(P)
with init_empty_weights():m=AutoModelForCausalLM.from_config(cfg)
LINW={n+'.weight' for n,mod in m.named_modules() if isinstance(mod,nn.Linear) and not any(k in n for k in('lm_head','embed'))}
MODEL_KEYS={n for n,_ in m.named_parameters()}|{n for n,_ in m.named_buffers()}
del m;gc.collect()
_MM='model.language_model.'
def norm_key(k):
    """Qwen3.5/3.8 checkpoints are *ForConditionalGeneration (keys model.language_model.*, model.visual.*, mtp.*) while the text-only
    causal LM built from the same config names them model.*. Map to the causal-LM names; returns None for vision-tower / MTP-head tensors (text-only bake)."""
    if k in MODEL_KEYS:return k
    if k.startswith(_MM) and ('model.'+k[len(_MM):]) in MODEL_KEYS:return 'model.'+k[len(_MM):]
    if k.startswith(('model.visual.','mtp.')):return None
    return k
def pack(W):
    out,inf=W.shape;Wg=W.float().reshape(out,-1,GS)
    sc=Wg.abs().amax(-1,keepdim=True).clamp_min(1e-8)/7.0
    codes=(torch.clamp(torch.round(Wg/sc),-8,7).to(torch.int32)+8).reshape(out,inf)
    pad=(4-inf%4)%4;cf=F.pad(codes,(0,pad)) if pad else codes
    rgba=cf.reshape(out,cf.shape[1]//4,4).to(torch.uint8)
    return rgba.contiguous(),sc.squeeze(-1).float().contiguous(),inf
t=time.time();manifest={'gs':GS,'tensors':{}};nq=0;nb=0;nskip=0;tot_bytes=0;chk=[]
for sh in sorted(glob.glob(P+'/*.safetensors')):
    sd={}
    with safe_open(sh,framework='pt') as f:
        for k0 in f.keys():
            k=norm_key(k0)
            if k is None:nskip+=1;continue
            w=f.get_tensor(k0)
            if k in LINW and w.dim()==2:
                rgba,sc,inf=pack(w)
                sd[k+'.codes']=rgba;sd[k+'.scale']=sc
                manifest['tensors'][k]={'q':1,'shape':list(w.shape),'inf':inf}
                tot_bytes+=rgba.numel()*0.5+sc.numel()*4;nq+=1
                if len(chk)<3:
                    dec=(rgba.reshape(w.shape[0],-1)[:,:inf].to(torch.int32)-8).reshape(w.shape[0],-1,GS).float()*sc.unsqueeze(-1).float()
                    ref=(torch.clamp(torch.round(w.float().reshape(w.shape[0],-1,GS)/(sc.unsqueeze(-1).float())),-8,7)*sc.unsqueeze(-1).float())
                    chk.append((k,(dec-ref).abs().max().item()))
            else:
                sd[k]=w.to(torch.bfloat16) if w.is_floating_point() else w
                manifest['tensors'][k]={'q':0,'shape':list(w.shape)};tot_bytes+=w.numel()*2;nb+=1
    st.save_file(sd,os.path.join(OUT,os.path.basename(sh)));del sd;gc.collect()
    print(f'  baked {os.path.basename(sh)} | q={nq} bf16={nb}',flush=True)
json.dump(manifest,open(os.path.join(OUT,'bake_manifest.json'),'w'))
manifest['base_model']=_bm.get('key') or _a.hf_id;json.dump(manifest,open(os.path.join(OUT,'bake_manifest.json'),'w'))
for f in [x for x in glob.glob(P+'/*.json') if not x.endswith('.safetensors.index.json')]+glob.glob(P+'/tokenizer*')+glob.glob(P+'/*.model')+glob.glob(P+'/*.jinja'):
    import shutil;shutil.copy(f,OUT)
print(f'BAKE DONE in {time.time()-t:.0f}s | {nq} Linears quantized + {nb} bf16 | {nskip} vision/MTP tensors skipped | size {tot_bytes/1e9:.2f}GB',flush=True)
if nq<len(LINW):print(f'WARNING: only {nq}/{len(LINW)} Linear weights were found in the checkpoint - check key names',flush=True)
print('roundtrip maxerr (sample):',[(k.split('.')[-2],f'{e:.1e}') for k,e in chk],flush=True)
print('BAKE_GRANITE_ATEX_DONE',flush=True)
