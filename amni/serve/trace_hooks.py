"""trace_hooks — read-only per-layer residual-stream state counter for Adam's underlying transformer model.
Captures distinct residual-vector hashes per layer to measure |S_l| for the resonance-substrate decision
(see docs/checklists/trace_pass_design_v1.md). Optionally also saves raw fp16 residuals (downsampled) for
offline PCA + grid quantization analysis. Lossless: never mutates activations or weights; only reads
output tensors via PyTorch forward hooks. Single-process; tiny memory footprint."""
import hashlib,json
from pathlib import Path
from collections import defaultdict,Counter
from typing import List,Any,Optional
try:import torch
except Exception:torch=None
try:import numpy as np
except Exception:np=None
_STATE={'enabled':False,'counter':defaultdict(Counter),'handles':[],'total_obs':0,'last_hash':None,'attached_to':None,'raws':defaultdict(list),'save_raws':False,'raw_every':1,'raw_max_per_layer':1000}
def _hash_last_pos(t)->int:
    last=t[:,-1,:].detach().to(torch.float16).contiguous().cpu().numpy()
    return int.from_bytes(hashlib.blake2b(last.tobytes(),digest_size=8).digest(),'big')
def _make_hook(layer_idx:int):
    def hook(mod,inp,out):
        if not _STATE['enabled']:return
        try:
            t=out[0] if isinstance(out,(tuple,list)) else out
            if torch is None or not hasattr(t,'detach') or t.dim()<3:return
            last_np=t[:,-1,:].detach().to(torch.float16).contiguous().cpu().numpy()
            h=int.from_bytes(hashlib.blake2b(last_np.tobytes(),digest_size=8).digest(),'big')
            _STATE['counter'][layer_idx][h]+=1
            cnt=_STATE['total_obs']
            _STATE['total_obs']=cnt+1
            _STATE['last_hash']=h
            if _STATE['save_raws'] and (cnt%_STATE['raw_every']==0) and len(_STATE['raws'][layer_idx])<_STATE['raw_max_per_layer']:
                _STATE['raws'][layer_idx].append(last_np.reshape(-1).copy())
        except Exception:pass
    return hook
def _find_layers(model)->Optional[List]:
    for path in ('model.layers','model.model.layers','transformer.h','transformer.layers','language_model.model.layers','text_model.layers'):
        cur=model
        ok=True
        for p in path.split('.'):
            if hasattr(cur,p):cur=getattr(cur,p)
            else:ok=False;break
        if ok and hasattr(cur,'__len__') and len(cur)>0:return list(cur)
    return None
def attach(model)->dict:
    if _STATE['attached_to'] is not None:return {'status':'already_attached','n_layers':len(_STATE['handles'])}
    layers=_find_layers(model)
    if not layers:return {'status':'error','reason':'could not locate decoder layers (tried model.layers, model.model.layers, transformer.h, transformer.layers, language_model.model.layers, text_model.layers)'}
    for i,layer in enumerate(layers):
        _STATE['handles'].append(layer.register_forward_hook(_make_hook(i)))
    _STATE['attached_to']=id(model)
    return {'status':'attached','n_layers':len(layers)}
def detach()->dict:
    n=len(_STATE['handles'])
    for h in _STATE['handles']:
        try:h.remove()
        except Exception:pass
    _STATE['handles']=[]
    _STATE['attached_to']=None
    return {'status':'detached','n_removed':n}
def start()->dict:
    _STATE['enabled']=True
    return {'status':'started','attached':_STATE['attached_to'] is not None,'n_handles':len(_STATE['handles'])}
def stop()->dict:
    _STATE['enabled']=False
    return snapshot()
def reset()->dict:
    _STATE['counter'].clear();_STATE['total_obs']=0;_STATE['last_hash']=None;_STATE['raws'].clear()
    return {'status':'reset'}
def configure_raws(save:bool=True,every:int=1,max_per_layer:int=1000)->dict:
    _STATE['save_raws']=bool(save);_STATE['raw_every']=max(1,int(every));_STATE['raw_max_per_layer']=max(1,int(max_per_layer))
    return {'save_raws':_STATE['save_raws'],'raw_every':_STATE['raw_every'],'raw_max_per_layer':_STATE['raw_max_per_layer']}
def dump_raws(path:str)->dict:
    if np is None:return {'status':'error','reason':'numpy unavailable'}
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    data={}
    for li,vecs in _STATE['raws'].items():
        if not vecs:continue
        arr=np.stack(vecs,axis=0)
        data[str(li)]=arr
    np.savez_compressed(str(p),**data)
    sizes={li:int(_STATE['raws'][li].__len__()) for li in _STATE['raws']}
    return {'status':'dumped','path':str(p),'per_layer_count':sizes,'total_vectors':sum(sizes.values())}
def snapshot()->dict:
    out={}
    for i,c in _STATE['counter'].items():
        total=sum(c.values())
        out[i]={'distinct':len(c),'total':total,'top_5':[(f'{h:016x}',n) for h,n in c.most_common(5)]}
    return {'enabled':_STATE['enabled'],'total_obs':_STATE['total_obs'],'n_layers_observed':len(out),'layers':out}
def status()->dict:
    return {'enabled':_STATE['enabled'],'attached':_STATE['attached_to'] is not None,'n_handles':len(_STATE['handles']),'n_layers_observed':len(_STATE['counter']),'total_obs':_STATE['total_obs']}
