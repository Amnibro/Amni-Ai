import numpy as np, sys, os, time, hashlib, hmac as _hmac
from typing import Optional, Tuple, Dict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from amni.compute.gf17_ops import (P,GF17_INV,gf17_matmul_t,gf17_add,gf17_mul,gf17_activate,gf17_rms_norm,gf17_init_weights,gf17_distance,ACTIVATION_LUTS,_CUBE_LUT,GF17_ALT_ENCODE,gf17_alt_mul,gf17_fused_mlp,gf17_norm_matmul_t,gf17_to_float,dq_matmul_t,f32_rms_norm,b17_to_f32)
try:from amni.compute.fused_dsp import fused_dsp_score as _fused_dsp;_USE_FUSED=True
except: _USE_FUSED=False
try:from amni.compute.gf17_engine import is_available as _hip_ok,get_engine as _hip_eng,f32_available as _f32_hip_ok,get_f32_engine as _f32_eng_get
except:_hip_ok=lambda:False;_hip_eng=lambda:None;_f32_hip_ok=lambda:False;_f32_eng_get=lambda:None
try:from amni.compute.ari_engine import is_available as _ari_ok,get_engine as _ari_eng,load_atex
except:_ari_ok=lambda:False;_ari_eng=lambda:None;load_atex=None
try:from amni.compute.prismtex import load_ptex as _ptex_load,MODE_NAMES as _PTEX_MODES;from amni.compute.prismtex_converter import convert_npz_to_ptex as _ptex_convert;_ptex_ok=True
except:_ptex_ok=False;_ptex_load=None;_ptex_convert=None;_PTEX_MODES={}
try:from amni.compute.prismtex_engine import is_available as _ptex_gpu_ok,get_engine as _ptex_gpu_eng,load_manifest_to_engine as _ptex_gpu_load
except:_ptex_gpu_ok=lambda:False;_ptex_gpu_eng=lambda:None;_ptex_gpu_load=None
try:from safetensors.numpy import save_file as stnsave,load_file as stnload;_STN=True
except:stnsave=stnload=None;_STN=False
from amni.a1.asimov import _AXIOMS,_HARM_KEYWORDS,_JAILBREAK_PATTERNS,gf17_hash_pattern,gf17_safety_score,_HARM_LUT,_JAIL_LUT
def _rope_f32(x, head_dim, offset=0):
    B,H,S,D=x.shape
    d2=D//2
    pos=np.arange(offset,offset+S,dtype=np.float32)
    inv_freq=1.0/(10000.0**(np.arange(0,d2,dtype=np.float32)*2.0/D))
    theta=np.outer(pos,inv_freq)
    cos_t=np.cos(theta)[np.newaxis,np.newaxis,:,:].astype(np.float32)
    sin_t=np.sin(theta)[np.newaxis,np.newaxis,:,:].astype(np.float32)
    out=x.copy()
    x1,x2=x[...,:d2],x[...,d2:2*d2]
    out[...,:d2]=x1*cos_t-x2*sin_t
    out[...,d2:2*d2]=x1*sin_t+x2*cos_t
    return out
ADAM_CONFIGS = {
    "adam-nano": {"hidden": 17, "n_heads": 1, "n_kv_heads": 1, "inter": 34, "n_blocks": 1},
    "adam-micro": {"hidden": 34, "n_heads": 2, "n_kv_heads": 1, "inter": 68, "n_blocks": 2},
    "adam-micro17": {"hidden": 51, "n_heads": 3, "n_kv_heads": 1, "inter": 102, "n_blocks": 3, "head_dim": 17, "vocab": 17},
    "adam-mini": {"hidden": 68, "n_heads": 4, "n_kv_heads": 2, "inter": 136, "n_blocks": 2},
    "adam-tiny": {"hidden": 128, "n_heads": 4, "n_kv_heads": 2, "inter": 256, "n_blocks": 4},
    "adam-1": {"hidden": 1536, "n_heads": 8, "n_kv_heads": 1, "inter": 6144, "n_blocks": 35, "head_dim": 256, "vocab": 262144, "layer_overrides": {**{i:{"head_dim":512} for i in (4,9,14)}, **{i:{"inter":12288} for i in range(15,35) if i not in (19,24,29,34)}, **{i:{"head_dim":512,"inter":12288} for i in (19,24,29,34)}}},
    "adam-small": {"hidden": 256, "n_heads": 8, "n_kv_heads": 4, "inter": 512, "n_blocks": 6},
    "adam-medium": {"hidden": 512, "n_heads": 16, "n_kv_heads": 8, "inter": 1024, "n_blocks": 8},
    "adam-large": {"hidden": 1024, "n_heads": 16, "n_kv_heads": 8, "inter": 2048, "n_blocks": 12},
    "adam-1b": {"hidden": 2048, "n_heads": 16, "n_kv_heads": 8, "inter": 5504, "n_blocks": 24},
    "adam-7b": {"hidden": 4096, "n_heads": 32, "n_kv_heads": 8, "inter": 11008, "n_blocks": 32},
    "adam-14b": {"hidden": 5120, "n_heads": 40, "n_kv_heads": 8, "inter": 13824, "n_blocks": 48},
    "adam-24b": {"hidden": 6656, "n_heads": 52, "n_kv_heads": 8, "inter": 17920, "n_blocks": 48},
    "adam-70b": {"hidden": 8192, "n_heads": 64, "n_kv_heads": 8, "inter": 22016, "n_blocks": 80},
    "adam-140b": {"hidden": 10240, "n_heads": 80, "n_kv_heads": 8, "inter": 27648, "n_blocks": 80},
    "adam-gemma-31b": {"hidden": 5376, "n_heads": 32, "n_kv_heads": 16, "inter": 21504, "n_blocks": 60, "head_dim": 256, "vocab": 262144, "layer_overrides": {i:{"n_heads":32,"n_kv_heads":4,"head_dim":512} for i in [5,11,17,23,29,35,41,47,53,59]}},
    "qwen35-9b": {"hidden": 4096, "n_heads": 16, "n_kv_heads": 4, "inter": 12288, "n_blocks": 32, "head_dim": 256, "vocab": 248320},
}
class GF17Linear:
    __slots__ = ('in_f', 'out_f', 'w', 'bias', '_f32')
    def __init__(self, in_f: int, out_f: int, bias: bool = False, _lazy: bool = False):
        self.in_f, self.out_f = in_f, out_f
        self.w = np.zeros((out_f, in_f), dtype=np.uint8) if _lazy else gf17_init_weights((out_f, in_f), "centered")
        self.bias = (np.zeros((out_f,), dtype=np.uint8) if _lazy else gf17_init_weights((out_f,), "centered")) if bias else None
        self._f32 = None
    def forward(self, x: np.ndarray) -> np.ndarray:
        orig = x.shape
        y = gf17_matmul_t(x.reshape(-1, self.in_f), self.w)
        if self.bias is not None:
            y = gf17_add(y, self.bias.reshape(1, -1))
        return y.reshape(*orig[:-1], self.out_f)
    def forward_f32(self,x,s=2.0):orig=x.shape;return (x.reshape(-1,self.in_f).astype(np.float32)@self._f32.T).reshape(*orig[:-1],self.out_f) if self._f32 is not None else dq_matmul_t(x.reshape(-1,self.in_f),self.w,s).reshape(*orig[:-1],self.out_f)
    def param_count(self) -> int:
        return self.in_f * self.out_f + (self.out_f if self.bias is not None else 0)
    def get_weights(self) -> list:
        return [self.w] if self.bias is None else [self.w, self.bias]
class GF17Attention:
    __slots__ = ('hidden', 'n_heads', 'n_kv_heads', 'head_dim', 'q_dim', 'kv_dim', '_reps',
                 'q_proj', 'k_proj', 'v_proj', 'o_proj', '_k_cache', '_v_cache', '_pos_offset')
    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int, head_dim: int = 0, _lazy: bool = False):
        self.hidden, self.n_heads, self.n_kv_heads = hidden, n_heads, n_kv_heads
        self.head_dim = head_dim if head_dim > 0 else hidden // n_heads
        self.q_dim = n_heads * self.head_dim
        self.kv_dim = n_kv_heads * self.head_dim
        self._reps = n_heads // n_kv_heads
        self.q_proj = GF17Linear(hidden, self.q_dim, _lazy=_lazy)
        self.k_proj = GF17Linear(hidden, self.kv_dim, _lazy=_lazy)
        self.v_proj = GF17Linear(hidden, self.kv_dim, _lazy=_lazy)
        self.o_proj = GF17Linear(self.q_dim, hidden, _lazy=_lazy)
        self._k_cache = None
        self._v_cache = None
        self._pos_offset = 0
    def _dsp_score_negacyclic(self, q, k, v):
        if _USE_FUSED:
            try:return _fused_dsp(q, k, v)
            except Exception:pass
        qb = GF17_ALT_ENCODE[q.ravel()].reshape(*q.shape, 4)
        kb = GF17_ALT_ENCODE[k.ravel()].reshape(*k.shape, 4)
        qt, kt = qb.transpose(0,2,1,3,4), kb.transpose(0,2,1,3,4)
        pb = gf17_alt_mul(qt[:,:,:,None,:,:], kt[:,:,None,:,:,:])
        es = ((8*pb[...,0]+4*pb[...,1]+2*pb[...,2]+pb[...,3]) % P).astype(np.uint32)
        rw = es.sum(axis=-1) % P
        ws = rw.sum(axis=-1, keepdims=True) % P
        wf = np.where(ws==0, np.uint32(1), ws).astype(np.uint8)
        aw = (rw * GF17_INV[wf.ravel()].reshape(ws.shape).astype(np.uint32) % P).astype(np.uint8)
        vt = v.transpose(0,2,1,3).astype(np.uint32)
        return (np.einsum('bhst,bhtd->bhsd', aw.astype(np.uint32), vt) % P).astype(np.uint8).transpose(0,2,1,3)
    def forward(self, x: np.ndarray) -> np.ndarray:
        expand = x.ndim == 2
        xin = x[np.newaxis] if expand else x
        B, S, D = xin.shape
        H, Hkv, Hd = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.q_proj.forward(xin).reshape(B, S, H, Hd)
        k = self.k_proj.forward(xin).reshape(B, S, Hkv, Hd)
        v = self.v_proj.forward(xin).reshape(B, S, Hkv, Hd)
        if self._reps > 1:
            k = np.repeat(k, self._reps, axis=2)
            v = np.repeat(v, self._reps, axis=2)
        out = self._dsp_score_negacyclic(q, k, v)
        result = self.o_proj.forward(out.reshape(B, S, self.q_dim))
        return result[0] if expand else result
    def forward_cached(self, x: np.ndarray) -> np.ndarray:
        expand = x.ndim == 2
        xin = x[np.newaxis] if expand else x
        B, S, D = xin.shape
        H, Hkv, Hd = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.q_proj.forward(xin).reshape(B, S, H, Hd)
        k_new = self.k_proj.forward(xin).reshape(B, S, Hkv, Hd)
        v_new = self.v_proj.forward(xin).reshape(B, S, Hkv, Hd)
        k = np.concatenate([self._k_cache, k_new], axis=1) if self._k_cache is not None else k_new
        v = np.concatenate([self._v_cache, v_new], axis=1) if self._v_cache is not None else v_new
        self._k_cache, self._v_cache = k, v
        if self._reps > 1:
            k = np.repeat(k, self._reps, axis=2)
            v = np.repeat(v, self._reps, axis=2)
        out = self._dsp_score_negacyclic(q, k, v)
        result = self.o_proj.forward(out.reshape(B, S, self.q_dim))
        return result[0] if expand else result
    def clear_cache(self):
        self._k_cache, self._v_cache = None, None
        self._pos_offset = 0
    def forward_f32(self,x,s=2.0):
        expand=x.ndim==2;xin=x[np.newaxis]if expand else x;B,S,D=xin.shape;H,Hkv,Hd=self.n_heads,self.n_kv_heads,self.head_dim
        q=_rope_f32(self.q_proj.forward_f32(xin,s).reshape(B,S,H,Hd).transpose(0,2,1,3),Hd);k=_rope_f32(self.k_proj.forward_f32(xin,s).reshape(B,S,Hkv,Hd).transpose(0,2,1,3),Hd);v=self.v_proj.forward_f32(xin,s).reshape(B,S,Hkv,Hd).transpose(0,2,1,3)
        k,v=(np.repeat(k,self._reps,axis=1),np.repeat(v,self._reps,axis=1))if self._reps>1 else(k,v)
        sc=(q@k.transpose(0,1,3,2)/np.float32(np.sqrt(Hd))).astype(np.float32);sc=(sc+np.triu(np.full((S,S),np.float32(-1e9)),k=1))if S>1 else sc
        aw=np.exp(sc-sc.max(axis=-1,keepdims=True));aw=(aw/aw.sum(axis=-1,keepdims=True)).astype(np.float32)
        result=self.o_proj.forward_f32((aw@v).transpose(0,2,1,3).reshape(B,S,self.q_dim),s)
        return result[0]if expand else result
    def forward_f32_cached(self,x,s=2.0):
        expand=x.ndim==2;xin=x[np.newaxis]if expand else x;B,S,D=xin.shape;H,Hkv,Hd=self.n_heads,self.n_kv_heads,self.head_dim
        q=_rope_f32(self.q_proj.forward_f32(xin,s).reshape(B,S,H,Hd).transpose(0,2,1,3),Hd,self._pos_offset)
        kn=_rope_f32(self.k_proj.forward_f32(xin,s).reshape(B,S,Hkv,Hd).transpose(0,2,1,3),Hd,self._pos_offset);vn=self.v_proj.forward_f32(xin,s).reshape(B,S,Hkv,Hd).transpose(0,2,1,3)
        self._pos_offset+=S
        k=np.concatenate([self._k_cache,kn],axis=2)if self._k_cache is not None else kn;v=np.concatenate([self._v_cache,vn],axis=2)if self._v_cache is not None else vn
        self._k_cache,self._v_cache=k,v
        k,v=(np.repeat(k,self._reps,axis=1),np.repeat(v,self._reps,axis=1))if self._reps>1 else(k,v)
        Sk=k.shape[2];sc=(q@k.transpose(0,1,3,2)/np.float32(np.sqrt(Hd))).astype(np.float32);sc=(sc+np.triu(np.full((S,Sk),np.float32(-1e9)),k=Sk-S+1))if S>1 else sc
        aw=np.exp(sc-sc.max(axis=-1,keepdims=True));aw=(aw/aw.sum(axis=-1,keepdims=True)).astype(np.float32)
        result=self.o_proj.forward_f32((aw@v).transpose(0,2,1,3).reshape(B,S,self.q_dim),s)
        return result[0]if expand else result
    def param_count(self) -> int:
        return sum(p.param_count() for p in [self.q_proj, self.k_proj, self.v_proj, self.o_proj])
    def get_weights(self) -> list:
        r = []
        for p in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            r.extend(p.get_weights())
        return r
class GF17MLP:
    __slots__ = ('gate', 'up', 'down', '_act')
    def __init__(self, hidden: int, inter: int, act: str = "cube", _lazy: bool = False):
        self.gate = GF17Linear(hidden, inter, _lazy=_lazy)
        self.up = GF17Linear(hidden, inter, _lazy=_lazy)
        self.down = GF17Linear(inter, hidden, _lazy=_lazy)
        self._act = act
    def forward(self, x: np.ndarray) -> np.ndarray:
        return gf17_fused_mlp(x, self.gate.w, self.up.w, self.down.w)
    def forward_f32(self,x,s=2.0):g=self.gate.forward_f32(x,s);u=self.up.forward_f32(x,s);return self.down.forward_f32((g/(1.0+np.exp(-np.clip(g.astype(np.float32),-20,20))).astype(np.float32))*u,s)
    def param_count(self) -> int:
        return sum(p.param_count() for p in [self.gate, self.up, self.down])
    def get_weights(self) -> list:
        r = []
        for p in [self.gate, self.up, self.down]:
            r.extend(p.get_weights())
        return r
class GF17Block:
    __slots__ = ('attn', 'mlp')
    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int, inter: int, act: str = "cube", head_dim: int = 0, _lazy: bool = False):
        self.attn = GF17Attention(hidden, n_heads, n_kv_heads, head_dim, _lazy=_lazy)
        self.mlp = GF17MLP(hidden, inter, act, _lazy=_lazy)
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = gf17_add(x, self.attn.forward(gf17_rms_norm(x)))
        normed = gf17_rms_norm(x)
        return gf17_add(x, self.mlp.forward(normed))
    def forward_f32(self,x,s=2.0):x=x+self.attn.forward_f32(f32_rms_norm(x),s);return x+self.mlp.forward_f32(f32_rms_norm(x),s)
    def forward_cached(self, x: np.ndarray) -> np.ndarray:
        x = gf17_add(x, self.attn.forward_cached(gf17_rms_norm(x)))
        normed = gf17_rms_norm(x)
        return gf17_add(x, self.mlp.forward(normed))
    def forward_f32_cached(self,x,s=2.0):x=x+self.attn.forward_f32_cached(f32_rms_norm(x),s);return x+self.mlp.forward_f32(f32_rms_norm(x),s)
    def clear_cache(self):
        self.attn.clear_cache()
    def param_count(self) -> int:
        return self.attn.param_count() + self.mlp.param_count()
    def get_weights(self) -> list:
        return self.attn.get_weights() + self.mlp.get_weights()
_TEX_ROOT=Path(__file__).resolve().parent.parent.parent/"textures"
class AdamModel:
    __slots__ = ('config_name','vocab','hidden','n_heads','n_kv_heads','inter','n_blocks','max_ctx','_act','embed','blocks','head','axioms','harm','jail','_kv_cached','_tex_root','_loaded_from','_gpu_ws','_hip','_layer_cfgs','_streaming','_layer_dir','_head_dim','_rg_ws','_emb_buf','_hd_buf','_ari','_ari_ws','_ari_manifest','_ptex_ws','_ptex_manifest','_ptex_eng','_ptex_gpu_ws','_mode','_dq_scales','_f32_eng','_f32_gpu_ws','_kv_pos','_b17','_embed_f32','_head_f32','_b17_stream_ws','_b17_src')
    def __init__(self,config_name:str="adam-nano",vocab:int=17,max_ctx:int=512,act:str="cube",auto_load:bool=True,tex_root=None):
        cfg=ADAM_CONFIGS.get(config_name)or(_ for _ in[1]).throw(ValueError(f"Unknown config:{config_name}. Available:{list(ADAM_CONFIGS.keys())}"))
        self.config_name=config_name;self.vocab=cfg.get("vocab",vocab);self.hidden=cfg["hidden"];self.n_heads=cfg["n_heads"];self.n_kv_heads=cfg["n_kv_heads"];self.inter=cfg["inter"];self.n_blocks=cfg["n_blocks"];self.max_ctx=max_ctx;self._act=act;self.axioms=_AXIOMS;self.harm=_HARM_KEYWORDS;self.jail=_JAILBREAK_PATTERNS;self._tex_root=Path(tex_root) if tex_root else _TEX_ROOT;self._loaded_from=None;self._gpu_ws=None;self._hip=None;self._layer_dir=None;self._rg_ws=None;self._f32_eng=None;self._f32_gpu_ws=None
        self._head_dim=cfg.get('head_dim',cfg['hidden']//cfg['n_heads'])
        ovr=cfg.get("layer_overrides",{})
        self._layer_cfgs=[{"n_heads":ovr[i].get("n_heads",self.n_heads),"n_kv_heads":ovr[i].get("n_kv_heads",self.n_kv_heads),"head_dim":ovr[i].get("head_dim",self._head_dim),"inter":ovr[i].get("inter",self.inter)}if i in ovr else{"n_heads":self.n_heads,"n_kv_heads":self.n_kv_heads,"head_dim":self._head_dim,"inter":self.inter}for i in range(self.n_blocks)]
        total_params=self.vocab*self.hidden*2+sum(self._block_param_count(i) for i in range(self.n_blocks))
        self._streaming=(total_params>5_000_000_000)
        import os as _os
        _b17_exists=(self._tex_root/"models"/(config_name+"-b17")/"meta.json").exists() or (Path(_os.path.expanduser("~"))/(config_name+"-b17")/"meta.json").exists()
        if self._streaming:
            self.embed=None;self.head=None;self.blocks=None
            self._layer_dir=str(self._tex_root/"models"/self.config_name)
        elif _b17_exists:
            self.embed=np.zeros((self.vocab,self.hidden),dtype=np.uint8)
            self.blocks=[GF17Block(self.hidden,lc["n_heads"],lc["n_kv_heads"],lc.get("inter",self.inter),act,lc.get("head_dim",self._head_dim),_lazy=True)for lc in self._layer_cfgs]
            self.head=self.embed
        else:
            self.embed=gf17_init_weights((self.vocab,self.hidden),'centered')
            self.blocks=[GF17Block(self.hidden,lc["n_heads"],lc["n_kv_heads"],lc.get("inter",self.inter),act,lc.get("head_dim",self._head_dim))for lc in self._layer_cfgs]
            self.head=gf17_init_weights((self.vocab,self.hidden),'centered')
        self._kv_cached=False;self._emb_buf=None;self._hd_buf=None;self._ari=None;self._ari_ws=None;self._ari_manifest=None;self._ptex_ws=None;self._ptex_manifest=None;self._ptex_eng=None;self._ptex_gpu_ws=None;self._kv_pos=0
        self._dq_scales=[3.0 if(lc.get('head_dim',self._head_dim)>256 and lc.get('inter',self.inter)>6144)else 2.0 for lc in self._layer_cfgs]
        self._b17=False;self._embed_f32=None;self._head_f32=None;self._b17_stream_ws=None;self._b17_src=None
        self._mode='hip' if _f32_hip_ok() else 'f32'
        if auto_load and not self._streaming:self._auto_load()
    def _block_param_count(self,i):
        lc=self._layer_cfgs[i];hd=lc.get('head_dim',self._head_dim)
        q_dim=lc['n_heads']*hd;q_sz=self.hidden*q_dim;kv_sz=lc['n_kv_heads']*hd*self.hidden
        li_inter=lc.get('inter',self.inter);o_sz=q_dim*self.hidden;mlp_sz=li_inter*self.hidden*3
        return q_sz+kv_sz*2+o_sz+mlp_sz
    def load_layer_weights(self,li:int)->Dict:
        fn=f'layer_{li:03d}.npz';p=Path(self._layer_dir)/fn
        if not p.exists():p=Path(self._layer_dir)/'ssd'/fn
        if not p.exists():return {}
        d=np.load(str(p));return{k:d[k] for k in d.files if not k.endswith('_scale')}
    def load_embed(self)->np.ndarray:
        p=Path(self._layer_dir)
        npz=p/'embed.npz'
        if npz.exists():return np.load(str(npz))['embed']
        done=p/'embed.done'
        if done.exists():
            chunks_dir=p/'embed_chunks'
            parts=sorted(chunks_dir.glob('emb_*.npz'))
            if parts:return np.concatenate([np.load(str(c))['gf17'] for c in parts],axis=0)
        return None
    def load_embed_rows(self,token_ids)->np.ndarray:
        p=Path(self._layer_dir)
        npz=p/'embed.npz'
        if npz.exists():
            d=np.load(str(npz),mmap_mode='r')
            return d['embed'][token_ids].copy()
        done=p/'embed.done'
        if done.exists():
            chunks_dir=p/'embed_chunks'
            meta_p=p/'meta.json'
            chunk_sz=4096
            if meta_p.exists():
                import json
                m=json.loads(meta_p.read_text())
                chunk_sz=m.get('embed_chunk_size',4096)
            uids=np.unique(token_ids)
            needed_chunks={}
            for uid in uids:
                ci=int(uid)//chunk_sz
                needed_chunks.setdefault(ci,[]).append(int(uid))
            row_map={}
            for ci,rows in needed_chunks.items():
                cp=chunks_dir/f'emb_{ci:04d}.npz'
                if not cp.exists():continue
                d=np.load(str(cp))
                gf=d['gf17']
                base=ci*chunk_sz
                for r in rows:
                    row_map[r]=gf[r-base]
            out=np.zeros((len(token_ids),self.hidden),dtype=np.uint8)
            for j,tid in enumerate(token_ids):
                if tid in row_map:out[j]=row_map[tid]
            return out
        return None
    def setup_streaming(self,layer_dir:str=None):
        if layer_dir:self._layer_dir=layer_dir
        self._streaming=True;self.blocks=None;self.embed=None;self.head=None
    def load_railgun_vram(self,rg_dir=None):
        import gc as _gc
        from amni.compute.gf17_engine import get_engine as _hip_eng,is_available
        if not is_available():return False
        try:eng=self._hip or _hip_eng();self._hip=eng
        except:return False
        ld=Path(self._layer_dir)if self._layer_dir else self._tex_root/"models"/self.config_name;rd=Path(rg_dir)if rg_dir else ld/'railgun'
        if not rd.exists():return False
        ws={};ep=ld/'embed.npz'
        emb=np.load(str(ep))['embed']if ep.exists()else(self.load_embed()or gf17_init_weights((self.vocab,self.hidden),'centered'));ws['emb']=eng.upload(emb);del emb;_gc.collect()
        for i in range(self.n_blocks):
            lp=rd/f'layer_{i:03d}.npz'
            if not lp.exists():self._rg_ws=None;return False
            ld_npz=np.load(str(lp))
            for key in['q','k','v','o','gate','up','down']:
                rk=f'{key}_r2b';lk=f'{key}_lut';nk=f'{key}_n';sk=f'{key}_shape'
                if rk in ld_npz:ws[f'{i}{key}_r2b']=eng.upload(ld_npz[rk]);ws[f'{i}{key}_lut']=eng.upload(ld_npz[lk]);ws[f'{i}{key}_n']=int(ld_npz[nk][0]);ws[f'{i}{key}_shape']=tuple(ld_npz[sk])
            del ld_npz;_gc.collect()
        self._rg_ws=ws;return True
    def _forward_gpu_railgun(self,token_ids):
        eng=self._hip;rg=self._rg_ws;B=token_ids.shape[0] if token_ids.ndim==2 else 1;S=token_ids.shape[-1] if token_ids.ndim==2 else token_ids.shape[0];D=self.hidden;tids=token_ids if token_ids.ndim==2 else token_ids[np.newaxis];x=eng.embed(rg['emb'],tids,B*S,D)
        for i in range(self.n_blocks):
            lc=self._layer_cfgs[i];H,Hkv,Hd=lc['n_heads'],lc['n_kv_heads'],lc.get('head_dim',self._head_dim);wq=eng.unpack2b(rg[f'{i}q_r2b'],rg[f'{i}q_lut'],rg[f'{i}q_n']).reshape(*rg[f'{i}q_shape']);wk=eng.unpack2b(rg[f'{i}k_r2b'],rg[f'{i}k_lut'],rg[f'{i}k_n']).reshape(*rg[f'{i}k_shape']);wv=eng.unpack2b(rg[f'{i}v_r2b'],rg[f'{i}v_lut'],rg[f'{i}v_n']).reshape(*rg[f'{i}v_shape']);wo=eng.unpack2b(rg[f'{i}o_r2b'],rg[f'{i}o_lut'],rg[f'{i}o_n']).reshape(*rg[f'{i}o_shape']);gw=eng.unpack2b(rg[f'{i}gate_r2b'],rg[f'{i}gate_lut'],rg[f'{i}gate_n']).reshape(*rg[f'{i}gate_shape']);uw=eng.unpack2b(rg[f'{i}up_r2b'],rg[f'{i}up_lut'],rg[f'{i}up_n']).reshape(*rg[f'{i}up_shape']);dw=eng.unpack2b(rg[f'{i}down_r2b'],rg[f'{i}down_lut'],rg[f'{i}down_n']).reshape(*rg[f'{i}down_shape']);x=eng.block_forward(x,[wq,wk,wv,wo],[gw,uw,dw],B,S,H,Hkv,Hd);[eng.free(w) for w in (wq,wk,wv,wo,gw,uw,dw)]
        xn=eng.rms_norm(x.reshape(B*S,D));eng.free(x);logits=eng.matmul_t(xn,rg['emb']);eng.free(xn);r=eng.download(logits);eng.free(logits)
        return r.reshape(B,S,self.vocab)[:,-1,:] if S>1 else r.reshape(B,self.vocab)
    def save_attn_lexicon(self,dir:str=None)->str:
        from amni.training.memory_texture import MemoryTexture
        td=Path(dir) if dir else self._tex_root/"models"/self.config_name/"attn_lexicon"
        td.mkdir(parents=True,exist_ok=True)
        [MemoryTexture.from_attn_qkvo(b.attn.q_proj.w,b.attn.k_proj.w,b.attn.v_proj.w,b.attn.o_proj.w,name=f"b{i}_qkvo").save(str(td/f"b{i}")) for i,b in enumerate(self.blocks)]
        return str(td)
    def load_attn_lexicon(self,dir:str=None)->int:
        from amni.training.memory_texture import MemoryTexture
        td=Path(dir) if dir else self._tex_root/"models"/self.config_name/"attn_lexicon"
        loaded=0
        for i,blk in enumerate(self.blocks):
            bp=td/f"b{i}"
            if not (bp/"meta.json").exists():continue
            mt=MemoryTexture.load(str(bp));q,k,v,o=mt.read_qkvo(self.hidden,blk.attn.kv_dim)
            blk.attn.q_proj.w[:q.shape[0],:]=q;blk.attn.k_proj.w[:]=k;blk.attn.v_proj.w[:]=v;blk.attn.o_proj.w[:o.shape[0],:]=o;loaded+=1
        return loaded
    def _load_b17(self,override_path=None):
        import json as _j,os as _os
        b17d=Path(override_path) if override_path else (self._tex_root/"models"/(self.config_name+"-b17"))
        if not b17d.exists():
            alt=Path(_os.path.expanduser("~/adam-1-b17"))
            if alt.exists():b17d=alt
        npy_cache=Path(_os.path.expanduser("~/.amni-cache"))/(self.config_name+"-b17-npy")
        npz_cache=Path(_os.path.expanduser("~/.amni-cache"))/(self.config_name+"-b17")
        b17d=npy_cache if (npy_cache/"meta.json").exists() else (npz_cache if (npz_cache/"meta.json").exists() else b17d)
        mf=b17d/"meta.json"
        if not mf.exists():return False
        meta=_j.loads(mf.read_text())
        if meta.get("format")!="b17":return False
        gpu=self._mode=='hip'
        self._b17_src=b17d
        en=b17d/"embed.npy"
        ed=b17d/"embed.npz"
        if en.exists():
            raw=np.load(str(en),mmap_mode='r' if gpu else None)
            self.embed=raw;self.head=raw
        elif ed.exists():
            raw=np.load(str(ed))['embed']
            self.embed=raw;self.head=raw
        if not gpu and self.embed is not None:
            self._embed_f32=b17_to_f32(self.embed).reshape(self.embed.shape[0],self.embed.shape[1]//4)
            self._head_f32=self._embed_f32
        if gpu:
            self.blocks=None
            self._b17=True;self._loaded_from=str(b17d);return True
        def _dec_proj(lin,raw):
            lin.w=raw
            lin._f32=b17_to_f32(raw).reshape(raw.shape[0],raw.shape[1]//4)
        for i,blk in enumerate(self.blocks):
            lf=b17d/f"layer_{i:03d}.npz"
            if not lf.exists():continue
            d=np.load(str(lf))
            _dec_proj(blk.attn.q_proj,d['q']);_dec_proj(blk.attn.k_proj,d['k']);_dec_proj(blk.attn.v_proj,d['v']);_dec_proj(blk.attn.o_proj,d['o'])
            _dec_proj(blk.mlp.gate,d['gate']);_dec_proj(blk.mlp.up,d['up']);_dec_proj(blk.mlp.down,d['down'])
        self._b17=True;self._loaded_from=str(b17d);return True
    def _auto_load(self):
        if self._load_b17():return True
        c=self._tex_root/"models"/self.config_name;npz=c/f"{self.config_name}.npz";stn=c/f"{self.config_name}.safetensors";(stn.exists()and _STN and(t:=stnload(str(stn)))and((f:=t.get('weights',next(iter(t.values()))),True)[1])and((ws:=self.get_all_weights(),True)[1])and((sizes:=[w.size for w in ws],True)[1])and((cum:=np.cumsum([0]+sizes).tolist(),True)[1])and[w.flat.__setitem__(slice(None),f[cum[i]:cum[i+1]].astype(np.uint8))for i,w in enumerate(ws)]and setattr(self,'_loaded_from',str(stn))or True)or(npz.exists()and(self.load_gf17(str(npz))or self.load_attn_lexicon()or setattr(self,'_loaded_from',str(npz))or True))or((m:=c/"meta.json").exists()and(self._load_from_tex(str(c))or setattr(self,'_loaded_from',str(c))or True))or((tm:=self._tex_root/"params"/"meta.json").exists()and(self._distill_from_teacher(str(self._tex_root/"params"))or setattr(self,'_loaded_from',f"distilled:{self._tex_root/'params'}")or True))or False;return bool(self._loaded_from)
    def _load_from_tex(self,tex_dir:str):
        from amni.training.memory_texture import MemoryTexture
        mt=MemoryTexture.load(tex_dir);ws=self.get_all_weights();off=0
        for w in ws:
            n=w.size;raw=mt.read_region(off,n);w.flat[:n]=raw[:n];off+=n
    def _distill_from_teacher(self,teacher_dir:str):
        from amni.training.memory_texture import MemoryTexture
        import json
        mt=MemoryTexture.load(teacher_dir);tmeta=json.loads((Path(teacher_dir)/"meta.json").read_text())
        t_total=tmeta.get("total",0)
        if t_total<=0:return
        ws=self.get_all_weights();sizes=[w.size for w in ws];ts=sum(sizes);cum=np.cumsum([0]+sizes)
        for li,w in enumerate(ws):
            t0=int(cum[li]/ts*t_total);t1=int(cum[li+1]/ts*t_total);rsz=max(min(t1-t0,t_total-t0),1)
            chunk=mt.read_region(t0,rsz);n=w.size;clen=len(chunk);stride=max(clen//n,1)
            reduced=(chunk[:stride*n].reshape(n,stride).astype(np.uint32).sum(axis=1)%P).astype(np.uint8) if stride>1 and clen>=stride*n else (np.tile(chunk,(n//max(clen,1))+1)[:n]%P if clen<n else chunk[:n]%P)
            w.flat[:n]=reduced[:n]
    def save_tex(self,f='tex',td=None):
        from amni.training.memory_texture import MemoryTexture;from amni.compute.noncelex import _get_signing_key;td=Path(td)if td else self._tex_root/"models"/self.config_name;td.mkdir(parents=True,exist_ok=True);ws=self.get_all_weights();flat=np.concatenate([w.ravel()for w in ws]).astype(np.uint8);mt=MemoryTexture.from_flat_gf17(flat,name=self.config_name);mt.save(str(td));ckpt=td/f"{self.config_name}.npz";self.save_gf17(str(ckpt));s=gf17_safety_score(np.array([[0]]),_HARM_LUT,12)[0];sk=_get_signing_key(td);whash=_hmac.new(sk,flat.tobytes(),hashlib.sha256).hexdigest();d={'weights':flat,'safety':np.array([int(s)])};stnsave(d,metadata={'safety_net':'asimov_baked_v4.13.4','score':str(s),'weight_hmac':whash},filename=str(td/f"{self.config_name}.safetensors"))if _STN and f=='stn'else 0;return str(td)
    def upload_gpu(self):
        if not _hip_ok():return False
        eng=_hip_eng();ws={}
        ws['emb']=eng.upload(self.embed);ws['hd']=eng.upload(self.head)
        for i,blk in enumerate(self.blocks):
            ws[f'{i}wq']=eng.upload(blk.attn.q_proj.w);ws[f'{i}wk']=eng.upload(blk.attn.k_proj.w)
            ws[f'{i}wv']=eng.upload(blk.attn.v_proj.w);ws[f'{i}wo']=eng.upload(blk.attn.o_proj.w)
            ws[f'{i}gw']=eng.upload(blk.mlp.gate.w);ws[f'{i}uw']=eng.upload(blk.mlp.up.w)
            ws[f'{i}dw']=eng.upload(blk.mlp.down.w)
        self._gpu_ws=ws;self._hip=eng;return True
    def free_gpu(self):
        if not self._gpu_ws or not self._hip:return
        for v in self._gpu_ws.values():self._hip.free(v)
        self._gpu_ws=None
    def upload_gpu_f32(self):
        try:eng=_f32_eng_get()
        except Exception:return False
        ws={}
        ws['emb']=eng.upload_u8(self.embed);ws['hd']=eng.upload_u8(self.head)
        mc=self.max_ctx
        for i,blk in enumerate(self.blocks):
            ws[f'{i}wq']=eng.upload_u8(blk.attn.q_proj.w);ws[f'{i}wk']=eng.upload_u8(blk.attn.k_proj.w)
            ws[f'{i}wv']=eng.upload_u8(blk.attn.v_proj.w);ws[f'{i}wo']=eng.upload_u8(blk.attn.o_proj.w)
            ws[f'{i}gw']=eng.upload_u8(blk.mlp.gate.w);ws[f'{i}uw']=eng.upload_u8(blk.mlp.up.w)
            ws[f'{i}dw']=eng.upload_u8(blk.mlp.down.w)
            lc=self._layer_cfgs[i];H=lc['n_heads'];Hkv=lc['n_kv_heads'];Hd=lc.get('head_dim',self._head_dim)
            ws[f'{i}kc']=eng.alloc((mc*Hkv*Hd,));ws[f'{i}vc']=eng.alloc((mc*Hkv*Hd,))
            ws[f'{i}sc']=eng.alloc((H*mc,));ws[f'{i}ao']=eng.alloc((H*Hd,))
        self._f32_eng=eng;self._f32_gpu_ws=ws;self._kv_pos=0;return True
    def free_gpu_f32(self):
        if not self._f32_gpu_ws or not self._f32_eng:return
        for v in self._f32_gpu_ws.values():self._f32_eng.free(v)
        self._f32_gpu_ws=None
    def _load_b17_layer(self,li:int)->dict:
        bd=self._b17_src
        if bd is None:return {}
        nd=bd/f"layer_{li:03d}"
        if nd.is_dir() and (nd/"q.npy").exists():
            return {k:np.load(str(nd/f"{k}.npy")) for k in ('q','k','v','o','gate','up','down') if (nd/f"{k}.npy").exists()}
        nf=bd/f"layer_{li:03d}.npz"
        if nf.exists():
            d=np.load(str(nf));return {k:d[k] for k in ('q','k','v','o','gate','up','down') if k in d.files}
        return {}
    def _init_hip_b17_streaming(self):
        try:eng=_f32_eng_get()
        except Exception:return False
        ws={};mc=self.max_ctx
        ws['hd']=eng.upload_u8(self.head)
        for i in range(self.n_blocks):
            lc=self._layer_cfgs[i];H=lc['n_heads'];Hkv=lc['n_kv_heads'];Hd=lc.get('head_dim',self._head_dim)
            ws[f'{i}kc']=eng.alloc((mc*Hkv*Hd,));ws[f'{i}vc']=eng.alloc((mc*Hkv*Hd,))
            ws[f'{i}sc']=eng.alloc((H*mc,));ws[f'{i}ao']=eng.alloc((H*Hd,))
        self._f32_eng=eng;self._b17_stream_ws=ws;self._kv_pos=0;return True
    def free_b17_streaming(self):
        if not self._b17_stream_ws or not self._f32_eng:return
        for v in self._b17_stream_ws.values():self._f32_eng.free(v)
        self._b17_stream_ws=None
    def _forward_hip_b17_streaming(self,token_id:int)->np.ndarray:
        from concurrent.futures import ThreadPoolExecutor
        if self._b17_stream_ws is None:self._init_hip_b17_streaming()
        eng=self._f32_eng;ws=self._b17_stream_ws;T=self._kv_pos+1
        raw=self.embed[token_id];d=raw.reshape(-1,4).astype(np.uint32)
        emb_f=(d[:,0]+d[:,1]*17+d[:,2]*289+d[:,3]*4913).astype(np.uint32)
        emb_f=((emb_f<<16).view(np.float32)).reshape(self.hidden)
        xd=eng.upload(emb_f)
        _pf_cache={}
        def _pf(li):
            if li<self.n_blocks and li not in _pf_cache:_pf_cache[li]=self._load_b17_layer(li)
        pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='b17_pf')
        pool.submit(_pf,0)
        for i in range(self.n_blocks):
            if i+1<self.n_blocks:pool.submit(_pf,i+1)
            lc=self._layer_cfgs[i];s=self._dq_scales[i];H=lc['n_heads'];Hkv=lc['n_kv_heads'];Hd=lc.get('head_dim',self._head_dim);inter=lc.get('inter',self.inter)
            ld=_pf_cache.pop(i,None) or self._load_b17_layer(i)
            wq_d=eng.upload_u8(ld['q']);wk_d=eng.upload_u8(ld['k']);wv_d=eng.upload_u8(ld['v']);wo_d=eng.upload_u8(ld['o'])
            gw_d=eng.upload_u8(ld['gate']);uw_d=eng.upload_u8(ld['up']);dw_d=eng.upload_u8(ld['down'])
            del ld
            xnd=eng.rms_norm(xd,1,self.hidden)
            qd=eng.dq_gemv_b17(wq_d,xnd,self.hidden,H*Hd)
            kd=eng.dq_gemv_b17(wk_d,xnd,self.hidden,Hkv*Hd)
            vd=eng.dq_gemv_b17(wv_d,xnd,self.hidden,Hkv*Hd)
            eng.free(xnd);eng.free(wq_d);eng.free(wk_d);eng.free(wv_d)
            eng.mqa_attn(qd,ws[f'{i}kc'],ws[f'{i}vc'],kd,vd,ws[f'{i}sc'],ws[f'{i}ao'],H,Hkv,Hd,T,float(Hd)**-0.5)
            eng.free(qd);eng.free(kd);eng.free(vd)
            od=eng.dq_gemv_b17(wo_d,ws[f'{i}ao'],H*Hd,self.hidden);eng.free(wo_d)
            r1d=eng.add(xd,od);eng.free(od)
            r1nd=eng.rms_norm(r1d,1,self.hidden)
            gd=eng.dq_gemv_b17(gw_d,r1nd,self.hidden,inter)
            ud=eng.dq_gemv_b17(uw_d,r1nd,self.hidden,inter)
            eng.free(r1nd);eng.free(gw_d);eng.free(uw_d)
            hid=eng.silu(gd,ud);eng.free(gd);eng.free(ud)
            dd=eng.dq_gemv_b17(dw_d,hid,inter,self.hidden);eng.free(dw_d);eng.free(hid)
            xd_new=eng.add(r1d,dd);eng.free(r1d);eng.free(dd);eng.free(xd);xd=xd_new
        pool.shutdown(wait=False)
        xnd=eng.rms_norm(xd,1,self.hidden);eng.free(xd)
        logits_d=eng.dq_gemv_b17(ws['hd'],xnd,self.hidden,self.vocab);eng.free(xnd)
        eng.sync();logits=logits_d.to_host().reshape(1,-1);eng.free(logits_d)
        self._kv_pos=T;return logits
    def _forward_hip_f32_cached(self,token_id:int)->np.ndarray:
        if self._f32_gpu_ws is None:self.upload_gpu_f32()
        eng=self._f32_eng;ws=self._f32_gpu_ws;b17=self._b17
        T=self._kv_pos+1
        if b17:
            raw=self.embed[token_id];d=raw.reshape(-1,4).astype(np.uint32)
            emb_f=(d[:,0]+d[:,1]*17+d[:,2]*289+d[:,3]*4913).astype(np.uint32)
            emb_f=((emb_f<<16).view(np.float32)).reshape(self.hidden)
        else:
            emb_f=((self.embed[token_id].astype(np.float32)/8.0-1.0)*2.0).reshape(self.hidden)
        xd=eng.upload(emb_f)
        _dq=eng.dq_gemv_b17 if b17 else eng.dq_gemv
        for i,blk in enumerate(self.blocks):
            lc=self._layer_cfgs[i];s=self._dq_scales[i];H=lc['n_heads'];Hkv=lc['n_kv_heads'];Hd=lc.get('head_dim',self._head_dim);inter=lc.get('inter',self.inter)
            xnd=eng.rms_norm(xd,1,self.hidden)
            qd=_dq(ws[f'{i}wq'],xnd,self.hidden,H*Hd) if b17 else _dq(ws[f'{i}wq'],xnd,self.hidden,H*Hd,s)
            kd=_dq(ws[f'{i}wk'],xnd,self.hidden,Hkv*Hd) if b17 else _dq(ws[f'{i}wk'],xnd,self.hidden,Hkv*Hd,s)
            vd=_dq(ws[f'{i}wv'],xnd,self.hidden,Hkv*Hd) if b17 else _dq(ws[f'{i}wv'],xnd,self.hidden,Hkv*Hd,s)
            eng.free(xnd)
            inv_sqrt=float(Hd)**-0.5
            eng.mqa_attn(qd,ws[f'{i}kc'],ws[f'{i}vc'],kd,vd,ws[f'{i}sc'],ws[f'{i}ao'],H,Hkv,Hd,T,inv_sqrt)
            eng.free(qd);eng.free(kd);eng.free(vd)
            od=_dq(ws[f'{i}wo'],ws[f'{i}ao'],H*Hd,self.hidden) if b17 else _dq(ws[f'{i}wo'],ws[f'{i}ao'],H*Hd,self.hidden,s)
            r1d=eng.add(xd,od);eng.free(od)
            r1nd=eng.rms_norm(r1d,1,self.hidden)
            gd=_dq(ws[f'{i}gw'],r1nd,self.hidden,inter) if b17 else _dq(ws[f'{i}gw'],r1nd,self.hidden,inter,s)
            ud=_dq(ws[f'{i}uw'],r1nd,self.hidden,inter) if b17 else _dq(ws[f'{i}uw'],r1nd,self.hidden,inter,s)
            eng.free(r1nd)
            hid=eng.silu(gd,ud);eng.free(gd);eng.free(ud)
            dd=_dq(ws[f'{i}dw'],hid,inter,self.hidden) if b17 else _dq(ws[f'{i}dw'],hid,inter,self.hidden,s)
            eng.free(hid)
            xd_new=eng.add(r1d,dd);eng.free(r1d);eng.free(dd);eng.free(xd);xd=xd_new
        xnd=eng.rms_norm(xd,1,self.hidden);eng.free(xd)
        logits_d=_dq(ws['hd'],xnd,self.hidden,self.vocab) if b17 else _dq(ws['hd'],xnd,self.hidden,self.vocab,2.0)
        eng.free(xnd)
        eng.sync();logits=logits_d.to_host().reshape(1,-1);eng.free(logits_d)
        self._kv_pos=T;return logits
    def _forward_gpu_streaming(self,token_ids):
        eng=self._hip or _hip_eng();self._hip=eng
        B=token_ids.shape[0] if token_ids.ndim==2 else 1
        S=token_ids.shape[-1] if token_ids.ndim==2 else token_ids.shape[0]
        D=self.hidden
        tids=token_ids if token_ids.ndim==2 else token_ids[np.newaxis]
        uids=np.unique(tids.ravel())
        if self._emb_buf is None:
            emb_np=self.embed if self.embed is not None else self.load_embed()
            if emb_np is None:emb_np=gf17_init_weights((self.vocab,D),'centered')
            self._emb_buf=eng.upload(emb_np);self.embed=emb_np
        x=eng.embed(self._emb_buf,tids,B*S,D)
        for i in range(self.n_blocks):
            lc=self._layer_cfgs[i];H,Hkv,Hd=lc["n_heads"],lc["n_kv_heads"],lc.get("head_dim",self._head_dim)
            if self.blocks is not None:
                blk=self.blocks[i]
                wq_np,wk_np,wv_np,wo_np=blk.attn.q_proj.w,blk.attn.k_proj.w,blk.attn.v_proj.w,blk.attn.o_proj.w
                gw_np,uw_np,dw_np=blk.mlp.gate.w,blk.mlp.up.w,blk.mlp.down.w
            else:
                ld=self.load_layer_weights(i)
                if not ld:raise FileNotFoundError(f"SSD layer {i} not found in {self._layer_dir}")
                wq_np,wk_np,wv_np,wo_np=ld['q'],ld['k'],ld['v'],ld['o']
                gw_np,uw_np,dw_np=ld['gate'],ld['up'],ld['down']
            wq=eng.upload(wq_np);wk=eng.upload(wk_np);wv=eng.upload(wv_np);wo=eng.upload(wo_np)
            gw=eng.upload(gw_np);uw=eng.upload(uw_np);dw=eng.upload(dw_np)
            if self.blocks is None:del ld
            nx=eng.block_forward(x,[wq,wk,wv,wo],[gw,uw,dw],B,S,H,Hkv,Hd)
            eng.free(x);eng.free(wq);eng.free(wk);eng.free(wv);eng.free(wo);eng.free(gw);eng.free(uw);eng.free(dw)
            x=nx
        xn=eng.rms_norm(x.reshape(B*S,D));eng.free(x)
        if self._hd_buf is None:
            hd_np=self.head if self.head is not None else self.embed
            if hd_np is None:
                hd_np=self.load_embed()
                if hd_np is None:hd_np=gf17_init_weights((self.vocab,D),'centered')
            self._hd_buf=eng.upload(hd_np)
        logits=eng.matmul_t(xn,self._hd_buf);eng.free(xn)
        r=eng.download(logits);eng.free(logits)
        return r.reshape(B,S,self.vocab)[:,-1,:] if S>1 else r.reshape(B,self.vocab)
    def _forward_gpu(self,token_ids):
        if self._streaming:return self._forward_gpu_streaming(token_ids)
        eng=self._hip;ws=self._gpu_ws
        eng=self._hip;ws=self._gpu_ws
        B=token_ids.shape[0] if token_ids.ndim==2 else 1
        S=token_ids.shape[-1] if token_ids.ndim==2 else token_ids.shape[0]
        D=self.hidden
        tids=token_ids if token_ids.ndim==2 else token_ids[np.newaxis]
        x=eng.embed(ws['emb'],tids,B*S,D)
        for i in range(self.n_blocks):
            lc=self._layer_cfgs[i];H,Hkv,Hd=lc["n_heads"],lc["n_kv_heads"],lc.get("head_dim",self._head_dim)
            attn=[ws[f'{i}wq'],ws[f'{i}wk'],ws[f'{i}wv'],ws[f'{i}wo']]
            mlp=[ws[f'{i}gw'],ws[f'{i}uw'],ws[f'{i}dw']]
            nx=eng.block_forward(x,attn,mlp,B,S,H,Hkv,Hd);eng.free(x);x=nx
        xn=eng.rms_norm(x.reshape(B*S,D));eng.free(x)
        logits=eng.matmul_t(xn,ws['hd']);eng.free(xn)
        r=eng.download(logits);eng.free(logits)
        return r.reshape(B,S,self.vocab)[:,-1,:] if S>1 else r.reshape(B,self.vocab)
    def load_ari_textures(self,atex_dir=None):
        if not _ari_ok():return False
        import json
        try:eng=self._ari or _ari_eng();self._ari=eng
        except:return False
        ld=Path(atex_dir) if atex_dir else (Path(self._layer_dir) if self._layer_dir else self._tex_root/"models"/self.config_name)/'atex'
        mp=ld/'manifest.json'
        if not mp.exists():return False
        mf=json.loads(mp.read_text());ws={}
        if 'embed' in mf:
            data,w,h,nw=load_atex(str(ld/mf['embed']['file']))
            ws['emb']=eng.bind_texture(data);ws['emb'].n_weights=nw
        if 'head' in mf:
            data,w,h,nw=load_atex(str(ld/mf['head']['file']))
            ws['hd']=eng.bind_texture(data);ws['hd'].n_weights=nw
        for li_s,layer_info in mf.get('layers',{}).items():
            li=int(li_s)
            has_vq=any(v.get('vq',False) for v in layer_info.values())
            if has_vq:
                from amni.compute.ari_vq import load_vq_atex
                for k,info in layer_info.items():
                    fp=ld/info['file']
                    cb,labels,nw,page=load_vq_atex(str(fp))
                    if not eng._cb_loaded:eng.upload_codebook(cb)
                    ws[f'{li}{k}']=eng.bind_vq_texture(page,(nw+3)//4)
                    ws[f'{li}{k}'].n_weights=nw
            else:
                for k,info in layer_info.items():
                    data,w,h,nw=load_atex(str(ld/info['file']))
                    ws[f'{li}{k}']=eng.bind_texture(data)
                    ws[f'{li}{k}'].n_weights=nw
        self._ari_ws=ws;self._ari_manifest=mf;return True
    def _forward_gpu_ari(self,token_ids):
        eng=self._ari;ws=self._ari_ws
        B=token_ids.shape[0] if token_ids.ndim==2 else 1
        S=token_ids.shape[-1] if token_ids.ndim==2 else token_ids.shape[0]
        D=self.hidden;tids=token_ids if token_ids.ndim==2 else token_ids[np.newaxis]
        x=eng.tex_embed(ws['emb'],tids,B*S,D)
        for i in range(self.n_blocks):
            lc=self._layer_cfgs[i];H,Hkv,Hd=lc['n_heads'],lc['n_kv_heads'],lc.get('head_dim',self._head_dim)
            attn=[ws[f'{i}q'],ws[f'{i}k'],ws[f'{i}v'],ws[f'{i}o']]
            mlp=[ws[f'{i}gate'],ws[f'{i}up'],ws[f'{i}down']]
            nx=eng.tex_block_forward(x,attn,mlp,B,S,H,Hkv,Hd);eng.free(x);x=nx
        xn=eng.rms_norm(x.reshape(B*S,D));eng.free(x)
        emb_or_hd=ws.get('hd',ws['emb'])
        logits=eng.tex_matmul_t(xn,emb_or_hd,B*S,D,self.vocab);eng.free(xn)
        r=eng.download(logits);eng.free(logits)
        return r.reshape(B,S,self.vocab)[:,-1,:] if S>1 else r.reshape(B,self.vocab)
    def free_ari(self):
        if not self._ari_ws or not self._ari:return
        for v in self._ari_ws.values():self._ari.free_texture(v)
        self._ari_ws=None
    def load_prismtex(self,ptex_dir=None):
        if not _ptex_ok:return False
        pd=Path(ptex_dir) if ptex_dir else Path(self._layer_dir or '.')/'ptex'
        mp=pd/'manifest.json'
        if not mp.exists():return False
        import json
        with open(str(mp)) as f:mf=json.load(f)
        ws={}
        for key in('embed','head'):
            if key in mf:
                fp=pd/mf[key]['file']
                if fp.exists():ws[key]=_ptex_load(str(fp))
        for li_str,layer_info in mf.get('layers',{}).items():
            for tk,ti in layer_info.items():
                fp=pd/ti['file']
                if fp.exists():ws[f'l{li_str}_{tk}']=_ptex_load(str(fp))
        self._ptex_ws=ws;self._ptex_manifest=mf;return True
    def free_prismtex(self):self._ptex_ws=None;self._ptex_manifest=None;self._ptex_eng=None;self._ptex_gpu_ws=None
    def load_prismtex_gpu(self,ptex_dir=None,gen_blk=256):
        if not _ptex_gpu_ok():return False
        pd=Path(ptex_dir) if ptex_dir else Path(self._layer_dir or '.')/'ptex'
        mp=pd/'manifest.json'
        if not mp.exists():return False
        eng=_ptex_gpu_eng()
        result=_ptex_gpu_load(eng,str(pd),gen_blk)
        if result is None:return False
        ws,mf=result
        self._ptex_eng=eng;self._ptex_gpu_ws=ws;self._ptex_manifest=mf;return True
    def _forward_gpu_prismtex(self,token_ids):
        eng=self._ptex_eng;ws=self._ptex_gpu_ws;mf=self._ptex_manifest
        B,S,D=1,len(token_ids.ravel()),self.hidden
        ids=np.ascontiguousarray(token_ids.ravel(),dtype=np.int32)
        x=eng.embed(ws['embed'],ids,S,D)
        for li in range(mf.get('n_layers',0)):
            lc=self._layer_cfgs[li] if li<len(self._layer_cfgs) else {'n_heads':self.n_heads,'n_kv_heads':self.n_kv_heads,'head_dim':self._head_dim}
            tq=ws.get(f'l{li}_q');tk=ws.get(f'l{li}_k');tv=ws.get(f'l{li}_v');to=ws.get(f'l{li}_o')
            tg=ws.get(f'l{li}_gate');tu=ws.get(f'l{li}_up');td=ws.get(f'l{li}_down')
            if tq is None:continue
            H,Hkv,Hd=lc['n_heads'],lc['n_kv_heads'],lc['head_dim']
            x=eng.block_forward(x,(tq,tk,tv,to),(tg,tu,td),B,S,H,Hkv,Hd)
        xn=eng.rms_norm(x.reshape(B*S,D))
        hd=ws.get('head',ws.get('embed'))
        last=xn.reshape(B,S,D)
        last_tok=eng.alloc((1,D))
        eng._lib if hasattr(eng,'_lib') else None
        from amni.compute.prismtex_engine import _lib
        _lib.ptex_gen_d2d(last_tok.ptr,last.ptr.__class__(last.ptr.value+(S-1)*D) if S>1 else last.ptr,D)
        logits=eng.matmul_t(last_tok,hd,1,D,self.vocab)
        eng.free(xn);eng.free(last_tok)
        result=eng.download(logits);eng.free(logits)
        return result.reshape(1,-1)
    def _forward_cpu_prismtex(self,token_ids):
        from amni.compute.prismtex import decode_gf17_block
        ws=self._ptex_ws
        def _get_w(key):
            if key not in ws:return None
            primary,nw,mode,npx=ws[key]
            return decode_gf17_block({'mode':mode,'primary':primary,'n_weights':nw,'pixels':npx})
        emb_w=_get_w('embed')
        if emb_w is None:return self._forward_cpu(token_ids)
        emb_w=emb_w.reshape(self.vocab,self.hidden)
        x=emb_w[token_ids.ravel()].reshape(-1,self.hidden)
        for li in range(self._ptex_manifest.get('n_layers',0)):
            lc=self._layer_cfgs[li] if li<len(self._layer_cfgs) else {'n_heads':self.n_heads,'n_kv_heads':self.n_kv_heads,'head_dim':self._head_dim}
            wq=_get_w(f'l{li}_q');wk=_get_w(f'l{li}_k');wv=_get_w(f'l{li}_v');wo=_get_w(f'l{li}_o')
            wg=_get_w(f'l{li}_gate');wu=_get_w(f'l{li}_up');wd=_get_w(f'l{li}_down')
            if wq is None:continue
            H,Hkv,Hd=lc['n_heads'],lc['n_kv_heads'],lc['head_dim']
            xn=gf17_rms_norm(x)
            q=gf17_matmul_t(xn,wq.reshape(H*Hd,self.hidden))
            k=gf17_matmul_t(xn,wk.reshape(Hkv*Hd,self.hidden))
            v=gf17_matmul_t(xn,wv.reshape(Hkv*Hd,self.hidden))
            a=gf17_matmul_t(q,k.T) if q is not None and k is not None else xn
            a=gf17_rms_norm(a)
            o=gf17_matmul_t(a,v) if v is not None else a
            o=gf17_matmul_t(o,wo.reshape(self.hidden,H*Hd)) if wo is not None else o
            x=gf17_add(x,o[:x.shape[0],:x.shape[1]])
            xn2=gf17_rms_norm(x)
            if wg is not None and wu is not None and wd is not None:
                li_inter=lc.get('inter',self.inter);m=gf17_fused_mlp(xn2,wg.reshape(li_inter,self.hidden),wu.reshape(li_inter,self.hidden),wd.reshape(self.hidden,li_inter))
            else:m=xn2
            x=gf17_add(x,m[:x.shape[0],:x.shape[1]])
        x=gf17_rms_norm(x)
        hd_w=_get_w('head')
        logits=gf17_matmul_t(x,hd_w.reshape(self.vocab,self.hidden)) if hd_w is not None else gf17_matmul_t(x,emb_w)
        return logits[-1:] if x.shape[0]>1 else logits
    def _forward_raw(self,token_ids:np.ndarray)->np.ndarray:
        if self._ptex_eng and self._ptex_gpu_ws:return self._forward_gpu_prismtex(token_ids)
        if self._ptex_ws and self._ptex_manifest:return self._forward_cpu_prismtex(token_ids)
        if self._ari_ws and self._ari:return self._forward_gpu_ari(token_ids)
        if self._rg_ws and self._hip:return self._forward_gpu_railgun(token_ids)
        if self._streaming and _hip_ok():return self._forward_gpu_streaming_prefetch(token_ids)
        if self._gpu_ws is None and _hip_ok() and not self._streaming:self.upload_gpu()
        if self._gpu_ws:
            try:return self._forward_gpu(token_ids)
            except:pass
        if self.embed is None:e=self.load_embed();self.embed=e if e is not None else np.zeros((self.vocab,self.hidden),dtype=np.uint8)
        if self.blocks is None:raise RuntimeError("No blocks loaded and no GPU path available")
        x=self.embed[token_ids]
        for blk in self.blocks:x=blk.forward(x)
        x=gf17_rms_norm(x);last=x[:,-1,:]if x.ndim==3 else x[-1:];return gf17_matmul_t(last,self.head)
    def _forward_raw_f32(self,token_ids):
        if self.embed is None:e=self.load_embed();self.embed=e if e is not None else np.zeros((self.vocab,self.hidden),dtype=np.uint8)
        if self.blocks is None:raise RuntimeError("No blocks loaded")
        x=self._embed_f32[token_ids] if self._embed_f32 is not None else gf17_to_float(self.embed[token_ids],2.0)
        for i,blk in enumerate(self.blocks):x=blk.forward_f32(x,self._dq_scales[i])
        x=f32_rms_norm(x);last=x[:,-1,:]if x.ndim==3 else x[-1:]
        hw=self._head_f32 if self._head_f32 is not None else ((self.head.astype(np.float32)/8.0-1.0)*2.0)
        return last.reshape(1,-1)@hw.T
    def forward(self,token_ids:np.ndarray)->np.ndarray:
        if self._mode=='hip':
            _fwd=self._forward_hip_b17_streaming if (self._b17 and self._b17_src is not None and self._f32_gpu_ws is None) else self._forward_hip_f32_cached
            tids=token_ids.ravel()
            for tid in tids[:-1]:_fwd(int(tid))
            return _fwd(int(tids[-1]))
        return self._forward_raw_f32(token_ids)if self._mode=='f32' else self._forward_raw(token_ids)
    def predict(self,token_ids:np.ndarray)->np.ndarray:return self.forward(token_ids).argmax(axis=-1)
    def forward_next(self,token_ids:np.ndarray)->int:logits=self.forward(token_ids if token_ids.ndim==2 else token_ids[np.newaxis]);return int(logits[0].argmax())
    def forward_cached(self,token_ids:np.ndarray)->np.ndarray:
        if self._mode=='hip':
            _fwd=self._forward_hip_b17_streaming if (self._b17 and self._b17_src is not None and self._f32_gpu_ws is None) else self._forward_hip_f32_cached
            return _fwd(int(token_ids.ravel()[-1]))
        if self._mode=='f32':return self._forward_f32_cached(token_ids)
        x=self.embed[token_ids];[x:=blk.forward_cached(x)for blk in self.blocks];x=gf17_rms_norm(x);last=x[:,-1,:]if x.ndim==3 else x[-1:];return gf17_matmul_t(last,self.head)
    def _forward_f32_cached(self,token_ids):
        x=self._embed_f32[token_ids] if self._embed_f32 is not None else gf17_to_float(self.embed[token_ids],2.0)
        for i,blk in enumerate(self.blocks):x=blk.forward_f32_cached(x,self._dq_scales[i])
        x=f32_rms_norm(x);last=x[:,-1,:]if x.ndim==3 else x[-1:]
        hw=self._head_f32 if self._head_f32 is not None else ((self.head.astype(np.float32)/8.0-1.0)*2.0)
        return last.reshape(1,-1)@hw.T
    def forward_next_cached(self, token_id: int) -> int:
        tid = np.array([[token_id]], dtype=np.int64)
        return int(self.forward_cached(tid)[0].argmax())
    def clear_cache(self):
        if self.blocks:
            for blk in self.blocks: blk.clear_cache()
        self._kv_cached = False
        self._kv_pos = 0
    def generate_cached(self, prompt_ids: np.ndarray, max_len: int = 40) -> np.ndarray:
        self.clear_cache()
        prompt = prompt_ids[np.newaxis] if prompt_ids.ndim == 1 else prompt_ids
        logits = self.forward_cached(prompt)
        out = [int(logits[0].argmax())]
        for _ in range(max_len - 1):
            out.append(self.forward_next_cached(out[-1]))
        self.clear_cache()
        return np.array(out, dtype=np.int64)
    def generate_sampled(self, prompt_ids: np.ndarray, max_len: int = 40, temp: float = 0.8, top_p: float = 0.9, rep_penalty: float = 1.2, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        self.clear_cache()
        prompt = prompt_ids[np.newaxis] if prompt_ids.ndim == 1 else prompt_ids
        logits = self.forward_cached(prompt)
        out = [self._sample_token(logits[0], [], temp, top_p, rep_penalty, rng)]
        for _ in range(max_len - 1):
            tid = np.array([[out[-1]]], dtype=np.int64)
            logits = self.forward_cached(tid)
            out.append(self._sample_token(logits[0], out, temp, top_p, rep_penalty, rng))
        self.clear_cache()
        return np.array(out, dtype=np.int64)
    @staticmethod
    def _sample_token(logits: np.ndarray, history: list, temp: float, top_p: float, rep_penalty: float, rng) -> int:
        scores = logits.astype(np.float32).ravel()
        if rep_penalty != 1.0 and history:
            seen = set(history[-64:])
            for tid in seen:
                scores[tid] = scores[tid] / rep_penalty if scores[tid] > 0 else scores[tid] * rep_penalty
        if temp <= 0:
            return int(scores.argmax())
        scores = scores / temp
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()
        if top_p < 1.0:
            idx = np.argsort(probs)[::-1]
            csum = np.cumsum(probs[idx])
            cutoff = np.searchsorted(csum, top_p) + 1
            mask = np.zeros_like(probs, dtype=bool)
            mask[idx[:cutoff]] = True
            probs[~mask] = 0.0
            psum = probs.sum()
            probs = probs / psum if psum > 0 else probs
        return int(rng.choice(len(probs), p=probs))
    def param_count(self) -> int:
        return self.vocab * self.hidden * 2 + sum(b.param_count() for b in self.blocks)
    def get_all_weights(self) -> list:
        return [self.embed, self.head] + [w for b in self.blocks for w in b.get_weights()]
    def save_gf17(self, path: str):
        arrays = {"embed": self.embed, "head": self.head}
        for i, blk in enumerate(self.blocks):
            arrays[f"b{i}_aq"] = blk.attn.q_proj.w
            arrays[f"b{i}_ak"] = blk.attn.k_proj.w
            arrays[f"b{i}_av"] = blk.attn.v_proj.w
            arrays[f"b{i}_ao"] = blk.attn.o_proj.w
            arrays[f"b{i}_mg"] = blk.mlp.gate.w
            arrays[f"b{i}_mu"] = blk.mlp.up.w
            arrays[f"b{i}_md"] = blk.mlp.down.w
        np.savez_compressed(path, **arrays)
    def load_gf17(self, path: str):
        d = np.load(path)
        self.embed = d["embed"]
        self.head = d["head"]
        for i, blk in enumerate(self.blocks):
            blk.attn.q_proj.w = d[f"b{i}_aq"]
            blk.attn.k_proj.w = d[f"b{i}_ak"]
            blk.attn.v_proj.w = d[f"b{i}_av"]
            blk.attn.o_proj.w = d[f"b{i}_ao"]
            blk.mlp.gate.w = d[f"b{i}_mg"]
            blk.mlp.up.w = d[f"b{i}_mu"]
            blk.mlp.down.w = d[f"b{i}_md"]
    def save_railgun(self, path: str) -> Dict:
        try:
            from demo.railgun_core import railgun_encode, railgun_vram_bytes, railgun_ssd_bytes
            self.save_gf17(path)
            all_w = self.get_all_weights()
            flat = np.concatenate([a.ravel() for a in all_w])
            enc = railgun_encode(flat.astype(np.float32) / 16.0 * 2.0 - 1.0)
            rg_path = str(Path(path).with_suffix('.railgun.npz'))
            np.savez_compressed(rg_path,
                                r_packed=enc["vram"]["r_packed"],
                                r_lut=enc["vram"]["r_lut"],
                                g_packed=enc["ssd"]["g"]["packed"],
                                g_lut=enc["ssd"]["g"]["lut"],
                                n=np.array([enc["n"]]))
            vram_b = railgun_vram_bytes(enc)
            ssd_b = railgun_ssd_bytes(enc, tier=1)
            return {"npz": path, "railgun": rg_path, "raw_bytes": flat.nbytes,
                    "vram_bytes": vram_b, "ssd_tier1_bytes": ssd_b,
                    "compression": round(flat.nbytes / max(vram_b, 1), 1)}
        except ImportError:
            self.save_gf17(path)
            return {"npz": path, "railgun": None, "raw_bytes": sum(w.nbytes for w in self.get_all_weights())}
    def stats(self) -> Dict:
        tp = self.param_count()
        return {"config": self.config_name, "vocab": self.vocab, "hidden": self.hidden,
                "n_blocks": self.n_blocks, "n_heads": self.n_heads, "n_kv_heads": self.n_kv_heads,
                "inter": self.inter, "act": self._act, "total_params": tp,
                "gf17_mb": round(tp / 1e6, 2), "fp16_equiv_mb": round(tp * 2 / 1e6, 2),
                "compression": "2.0x vs fp16", "field": f"GF({P})", "bits": 5}
    def __repr__(self):
        s = self.stats()
        return f"AdamModel({s['config']}, {s['total_params']:,} params, {s['gf17_mb']} MB GF17)"
    def setup_streaming(self, max_resident: int = 4):
        tex_dir = self._tex_root / "models" / self.config_name
        if not (tex_dir / "meta.json").exists(): return False
        try:
            from amni.compute.tex_stream import TexStreamEngine
            self._stream_engine = TexStreamEngine(str(tex_dir), max_resident)
            return True
        except Exception: return False
    def _forward_streaming(self, token_ids: np.ndarray) -> np.ndarray:
        eng = getattr(self, '_stream_engine', None)
        x = self.embed[token_ids]
        for i, blk in enumerate(self.blocks):
            if eng and i + 1 < len(self.blocks):
                eng.prefetch_block_weights(i + 1)
            x = blk.forward(x)
        x = gf17_rms_norm(x)
        last = x[:, -1, :] if x.ndim == 3 else x[-1:]
        return gf17_matmul_t(last, self.head)
    def stream_context(self, page_start: int = 0, n_pages: int = 1) -> Optional[np.ndarray]:
        ctx_dir = self._tex_root / "context"
        if not (ctx_dir / "meta.json").exists(): return None
        try:
            from amni.compute.tex_stream import TexStreamEngine
            eng = TexStreamEngine(str(ctx_dir), max_resident=2)
            pages = [eng.get_page(page_start + i) for i in range(n_pages)]
            eng.close()
            return np.concatenate([p.reshape(-1) for p in pages if p is not None])
        except Exception: return None
    def close_streaming(self):
        eng = getattr(self, '_stream_engine', None)
        if eng: eng.close(); self._stream_engine = None
    def vram_bytes(self,engine:str='auto')->Dict:
        emb_b=self.vocab*self.hidden;hd_b=self.vocab*self.hidden
        blk_b=[self._block_param_count(i) for i in range(self.n_blocks)]
        total=emb_b+hd_b+sum(blk_b);work_buf=max(self.hidden*self.max_ctx*4,1<<20)
        rg_total=(emb_b+sum(b//4 for b in blk_b)+sum(4*7 for _ in range(self.n_blocks)))if engine=='railgun'else 0
        return{'total':total,'emb':emb_b,'head':hd_b,'blocks':sum(blk_b),'work_buf':work_buf,'per_block':blk_b,'full_vram':total+work_buf,'streaming_vram':emb_b+hd_b+max(blk_b)+work_buf,'railgun_vram':rg_total+work_buf if rg_total else 0,'full_mb':round((total+work_buf)/1e6,1),'streaming_mb':round((emb_b+hd_b+max(blk_b)+work_buf)/1e6,1)}
    def init_best_gpu(self,vram_budget:int=14*1024**3,prefer:str='auto')->str:
        vb=self.vram_bytes()
        if prefer!='auto':
            _init_map={'prismtex':self._try_init_prismtex,'ari':self._try_init_ari,'railgun':self._try_init_railgun,'streaming':self._try_init_streaming,'full':self._try_init_full}
            fn=_init_map.get(prefer)
            return prefer if fn and fn(vram_budget) else 'cpu'
        for name,fn in[('prismtex',self._try_init_prismtex),('ari',self._try_init_ari),('railgun',self._try_init_railgun)]:
            if fn(vram_budget):return name
        return'streaming'if vb['streaming_vram']<=vram_budget and self._try_init_streaming(vram_budget)else('full'if vb['full_vram']<=vram_budget and self._try_init_full(vram_budget)else'cpu')
    def _try_init_prismtex(self,budget:int)->bool:
        if not _ptex_gpu_ok():return False
        pd=Path(self._layer_dir or'.')/'ptex' if self._layer_dir else self._tex_root/'models'/self.config_name/'ptex'
        return self.load_prismtex_gpu(str(pd))if(pd/'manifest.json').exists()else False
    def _try_init_ari(self,budget:int)->bool:
        if not _ari_ok():return False
        ad=Path(self._layer_dir or'.')/'atex' if self._layer_dir else self._tex_root/'models'/self.config_name/'atex'
        return self.load_ari_textures(str(ad))if(ad/'manifest.json').exists()else False
    def _try_init_railgun(self,budget:int)->bool:
        vb=self.vram_bytes('railgun')
        if vb['railgun_vram']>budget or vb['railgun_vram']==0:return False
        return self.load_railgun_vram()
    def _try_init_streaming(self,budget:int)->bool:
        if not _hip_ok():return False
        self._hip=self._hip or _hip_eng()
        return True
    def _try_init_full(self,budget:int)->bool:
        if not _hip_ok() or self.embed is None:return False
        return self.upload_gpu()
    def gpu_engine_name(self)->str:
        if self._ptex_eng and self._ptex_gpu_ws:return'prismtex_gpu'
        if self._ptex_ws and self._ptex_manifest:return'prismtex_cpu'
        if self._ari_ws and self._ari:return'ari'
        if self._rg_ws and self._hip:return'railgun'
        if self._hip and self._streaming:return'streaming'
        if self._gpu_ws and self._hip:return'full_gpu'
        return'cpu'
    def generate_gpu(self,prompt_ids:np.ndarray,max_len:int=40,temperature:float=0.0)->np.ndarray:
        ids=prompt_ids[np.newaxis]if prompt_ids.ndim==1 else prompt_ids
        ctx=ids.copy();out=[]
        for _ in range(max_len):
            window=ctx[:,-self.max_ctx:]if ctx.shape[1]>self.max_ctx else ctx
            logits=self.forward(window)
            nxt=int(logits[0].argmax())if temperature==0.0 else int(np.random.choice(self.vocab,p=_softmax_gf17(logits[0],temperature)))
            out.append(nxt);ctx=np.concatenate([ctx,np.array([[nxt]],dtype=ctx.dtype)],axis=1)
        return np.array(out,dtype=np.int64)
    def _forward_gpu_streaming_prefetch(self,token_ids):
        from concurrent.futures import ThreadPoolExecutor
        eng=self._hip or _hip_eng();self._hip=eng
        B=token_ids.shape[0]if token_ids.ndim==2 else 1;S=token_ids.shape[-1]if token_ids.ndim==2 else token_ids.shape[0];D=self.hidden
        tids=token_ids if token_ids.ndim==2 else token_ids[np.newaxis]
        if self._emb_buf is None:
            emb_np=self.embed if self.embed is not None else self.load_embed()
            if emb_np is None:emb_np=gf17_init_weights((self.vocab,D),'centered')
            self._emb_buf=eng.upload(emb_np);self.embed=emb_np
        x=eng.embed(self._emb_buf,tids,B*S,D)
        _prefetch_cache={}
        def _prefetch_layer(li):
            if li>=self.n_blocks or li in _prefetch_cache:return
            ld=self.load_layer_weights(li)
            _prefetch_cache[li]=ld if ld else None
        pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='ssd_pf')
        pool.submit(_prefetch_layer,0)
        for i in range(self.n_blocks):
            if i+1<self.n_blocks:pool.submit(_prefetch_layer,i+1)
            lc=self._layer_cfgs[i];H,Hkv,Hd=lc['n_heads'],lc['n_kv_heads'],lc.get('head_dim',self._head_dim)
            if self.blocks is not None:
                blk=self.blocks[i]
                wq_np,wk_np,wv_np,wo_np=blk.attn.q_proj.w,blk.attn.k_proj.w,blk.attn.v_proj.w,blk.attn.o_proj.w
                gw_np,uw_np,dw_np=blk.mlp.gate.w,blk.mlp.up.w,blk.mlp.down.w
            else:
                ld=_prefetch_cache.pop(i,None)or self.load_layer_weights(i)
                if not ld:raise FileNotFoundError(f"SSD layer {i} not found")
                wq_np,wk_np,wv_np,wo_np=ld['q'],ld['k'],ld['v'],ld['o']
                gw_np,uw_np,dw_np=ld['gate'],ld['up'],ld['down']
                del ld
            wq=eng.upload(wq_np);wk=eng.upload(wk_np);wv=eng.upload(wv_np);wo=eng.upload(wo_np)
            gw=eng.upload(gw_np);uw=eng.upload(uw_np);dw=eng.upload(dw_np)
            nx=eng.block_forward(x,[wq,wk,wv,wo],[gw,uw,dw],B,S,H,Hkv,Hd)
            eng.free(x);[eng.free(w) for w in(wq,wk,wv,wo,gw,uw,dw)];x=nx
        pool.shutdown(wait=False)
        xn=eng.rms_norm(x.reshape(B*S,D));eng.free(x)
        if self._hd_buf is None:
            hd_np=self.head if self.head is not None else self.embed
            if hd_np is None:hd_np=self.load_embed()or gf17_init_weights((self.vocab,D),'centered')
            self._hd_buf=eng.upload(hd_np)
        logits=eng.matmul_t(xn,self._hd_buf);eng.free(xn)
        r=eng.download(logits);eng.free(logits)
        return r.reshape(B,S,self.vocab)[:,-1,:]if S>1 else r.reshape(B,self.vocab)
    def free_all_gpu(self):
        self.free_gpu()
        self.free_b17_streaming()
        self.free_gpu_f32()
        if self._ari_ws:self.free_ari()
        if self._ptex_eng:self.free_prismtex()
        if self._rg_ws and self._hip:
            for v in self._rg_ws.values():
                try:self._hip.free(v)
                except:pass
            self._rg_ws=None
        if self._emb_buf and self._hip:
            try:self._hip.free(self._emb_buf)
            except:pass
            self._emb_buf=None
        if self._hd_buf and self._hip:
            try:self._hip.free(self._hd_buf)
            except:pass
            self._hd_buf=None
def _softmax_gf17(logits:np.ndarray,temperature:float=1.0)->np.ndarray:
    x=logits.astype(np.float32)/16.0
    x=x/max(temperature,1e-8)
    e=np.exp(x-x.max())
    return e/e.sum()
