"""GF(17)-ATEX 4-bit bake of granite-4.1-3B -> GPU-resident, int4 group-128, decode-on-kernel (same recipe as bake_gf17_atex_4bit but AutoModelForCausalLM + granite paths). Quantizes each Linear to int4 group-128, packs codes to RGBA, keeps embed/lm_head + non-Linears bf16. Out: bakes/granite41_3b_gf17_atex/. Run: HIP_VISIBLE_DEVICES=1 python scripts/bake_granite_atex.py"""
import torch,torch.nn as nn,torch.nn.functional as F,os,glob,gc,json,time,safetensors.torch as st
from transformers import AutoConfig,AutoModelForCausalLM
from accelerate import init_empty_weights
from safetensors import safe_open
P='downloaded_models/granite-4.1-3b';OUT='bakes/granite41_3b_gf17_atex';GS=128
os.makedirs(OUT,exist_ok=True)
cfg=AutoConfig.from_pretrained(P)
with init_empty_weights():m=AutoModelForCausalLM.from_config(cfg)
LINW={n+'.weight' for n,mod in m.named_modules() if isinstance(mod,nn.Linear) and not any(k in n for k in('lm_head','embed'))}
del m;gc.collect()
def pack(W):
    out,inf=W.shape;Wg=W.float().reshape(out,-1,GS)
    sc=Wg.abs().amax(-1,keepdim=True).clamp_min(1e-8)/7.0
    codes=(torch.clamp(torch.round(Wg/sc),-8,7).to(torch.int32)+8).reshape(out,inf)
    pad=(4-inf%4)%4;cf=F.pad(codes,(0,pad)) if pad else codes
    rgba=cf.reshape(out,cf.shape[1]//4,4).to(torch.uint8)
    return rgba.contiguous(),sc.squeeze(-1).float().contiguous(),inf
t=time.time();manifest={'gs':GS,'tensors':{}};nq=0;nb=0;tot_bytes=0;chk=[]
for sh in sorted(glob.glob(P+'/*.safetensors')):
    sd={}
    with safe_open(sh,framework='pt') as f:
        for k in f.keys():
            w=f.get_tensor(k)
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
for f in glob.glob(P+'/*.json')+glob.glob(P+'/tokenizer*')+glob.glob(P+'/*.model')+glob.glob(P+'/*.jinja'):
    import shutil;shutil.copy(f,OUT)
print(f'BAKE DONE in {time.time()-t:.0f}s | {nq} Linears quantized + {nb} bf16 | size {tot_bytes/1e9:.2f}GB',flush=True)
print('roundtrip maxerr (sample):',[(k.split('.')[-2],f'{e:.1e}') for k,e in chk],flush=True)
print('BAKE_GRANITE_ATEX_DONE',flush=True)
