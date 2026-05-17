import numpy as np,json,gc,time,sys,os,types,importlib,importlib.util
from pathlib import Path
from typing import Dict,Tuple,Optional
_root=Path(__file__).resolve().parent
sys.path.insert(0,str(_root))
LAYER_KEYS=["q","k","v","o","gate","up","down"]
NORM_KEYS=["input_ln","post_attn_ln","pre_ff_ln","post_ff_ln","q_norm","k_norm","layer_scalar"]
GEMMA_NORM_MAP={"input_ln":"input_layernorm.weight","post_attn_ln":"post_attention_layernorm.weight","pre_ff_ln":"pre_feedforward_layernorm.weight","post_ff_ln":"post_feedforward_layernorm.weight","q_norm":"self_attn.q_norm.weight","k_norm":"self_attn.k_norm.weight","layer_scalar":"layer_scalar"}
GEMMA_PROJ_MAP={"q":"self_attn.q_proj.weight","k":"self_attn.k_proj.weight","v":"self_attn.v_proj.weight","o":"self_attn.o_proj.weight","gate":"mlp.gate_proj.weight","up":"mlp.up_proj.weight","down":"mlp.down_proj.weight"}
GEMMA_PLI_GLOBAL={"embed_tokens_per_layer":"embed_tokens_per_layer.weight","per_layer_model_projection":"per_layer_model_projection.weight","per_layer_projection_norm":"per_layer_projection_norm.weight"}
GEMMA_PLI_LAYER={"pli_gate":"per_layer_input_gate.weight","pli_proj":"per_layer_projection.weight","pli_norm":"post_per_layer_input_norm.weight"}
GEMMA_LAYER_TYPES=['sliding_attention']*4+['full_attention']+['sliding_attention']*4+['full_attention']+['sliding_attention']*4+['full_attention']+['sliding_attention']*4+['full_attention']+['sliding_attention']*4+['full_attention']+['sliding_attention']*4+['full_attention']+['sliding_attention']*4+['full_attention']
P=17
def _pf(*a,**k):print(*a,**k,flush=True)
def _ensure_modules():
    if 'amni.model.adam' in sys.modules:return
    try:import taichi
    except ImportError:
        _ti=types.ModuleType('taichi');_ti.gpu=_ti.cuda=_ti.vulkan=_ti.cpu=_ti.metal=_ti.f32=None;_ti.init=lambda **k:None
        _ti.field=lambda dtype,shape:type('F',(object,),{'from_numpy':lambda s,a:None})();_ti.i32=_ti.f32;_ti.kernel=_ti.func=lambda f:f
        sys.modules['taichi']=_ti
    _stub=types.ModuleType('amni');_stub.__path__=[os.path.join(str(_root),'amni')];_stub.__package__='amni';sys.modules['amni']=_stub
    for _sub in['amni.compute','amni.model','amni.training','amni.core','amni.a1','amni.pipeline']:
        _m=types.ModuleType(_sub);_m.__path__=[os.path.join(str(_root),_sub.replace('.',os.sep))];_m.__package__=_sub;sys.modules[_sub]=_m
    def _ld(n,r):
        sp=importlib.util.spec_from_file_location(n,os.path.join(str(_root),*r.split('/')));md=importlib.util.module_from_spec(sp);sys.modules[n]=md;sp.loader.exec_module(md);return md
    for n,r in[('amni.core.tex_atlas','amni/core/tex_atlas.py'),('amni.compute.gf17_ops','amni/compute/gf17_ops.py'),('amni.compute.kernels','amni/compute/kernels.py'),('amni.a1.asimov','amni/a1/asimov.py'),('amni.compute.fused_dsp','amni/compute/fused_dsp.py'),('amni.compute.prismtex','amni/compute/prismtex.py'),('amni.compute.prismtex_lut','amni/compute/prismtex_lut.py'),('amni.compute.prismtex_encoder','amni/compute/prismtex_encoder.py'),('amni.compute.prismtex_converter','amni/compute/prismtex_converter.py'),('amni.model.adam','amni/model/adam.py'),('amni.training.memory_texture','amni/training/memory_texture.py'),('amni.training.adam_gf17','amni/training/adam_gf17.py'),('amni.training.adam_distill','amni/training/adam_distill.py')]:
        _ld(n,r)
def _decode_ptex_to_fp16(ptex_dir:Path,info:dict)->np.ndarray:
    from amni.compute.prismtex import load_ptex,decode_gf17_block,decode_fp16_base17
    from amni.compute.prismtex_converter import load_prism_enc
    m,fn,nw=info["mode"],info["file"],info["n_weights"]
    prism_files=info.get("prism_files",{})
    enc=(load_prism_enc(str(ptex_dir/fn),name_prefix=fn.replace(".ptex","_")) if m==26 and prism_files else (lambda p,n,md,px:{"mode":md,"primary":p,"n_weights":n,"pixels":px})(*load_ptex(str(ptex_dir/fn))))
    digits=decode_gf17_block(enc)[:nw]
    shape=tuple(info["shape"])
    n_fp16=1
    for s in shape:n_fp16*=s
    return decode_fp16_base17(digits.reshape(-1,4),n_fp16).reshape(shape)
def _fp16_to_gf17(fp16:np.ndarray,scale:float=2.0,s=None)->np.ndarray:s=s if s is not None else np.full(fp16.shape[0],scale);return np.clip(np.round((np.clip(fp16.astype(np.float32)/s[:,None],-1,1)+1)*8),0,16).astype(np.uint8)
def _get_adam1_layer_shapes(model,li:int)->Dict[str,Tuple[int,int]]:lc=model._layer_cfgs[li];H,Hkv,Hd=lc["n_heads"],lc["n_kv_heads"],lc.get("head_dim",model._head_dim);inter=lc.get("inter",model.inter);hidden=model.hidden;return {"q":(H*Hd,hidden),"k":(Hkv*Hd,hidden),"v":(Hkv*Hd,hidden),"o":(hidden,H*Hd),"gate":(inter,hidden),"up":(inter,hidden),"down":(hidden,inter)}
def translate_gemma_ptex_to_adam1(ptex_dir:str,out_path:str=None,scale:float=2.0,verbose:bool=True)->Dict:
    _ensure_modules()
    from amni.model.adam import AdamModel,ADAM_CONFIGS
    ptex_dir=Path(ptex_dir)
    manifest=json.load(open(str(ptex_dir/"manifest.json")))
    if out_path is None:out_path=str(_root/"textures"/"models"/"adam-1"/"adam-1.safetensors")
    out_p=Path(out_path);out_p.parent.mkdir(parents=True,exist_ok=True)
    cfg=ADAM_CONFIGS["adam-1"]
    model=AdamModel('adam-1',vocab=cfg.get("vocab",262144),act='cube',auto_load=False)
    t0=time.perf_counter()
    result={"loaded":0,"skipped":0,"exact":0,"total_weights":0,"layers":{}}
    all_gf17=[]
    if verbose:
        _pf(f"[translator] Gemma ptex -> Adam-1 GF17 native (exact dims)")
        _pf(f"  source: {ptex_dir}")
        _pf(f"  target: {out_path}")
        _pf(f"  adam-1: hidden={model.hidden}, blocks={model.n_blocks}, vocab={model.vocab}")
        _pf(f"  per-layer overrides: {len([lc for lc in model._layer_cfgs if lc.get('inter',model.inter)!=model.inter or lc['n_heads']!=model.n_heads])} layers with variant dims")
    if "embed" in manifest:
        if verbose:_pf(f"  [embed] decoding ptex...",end="")
        fp16=_decode_ptex_to_fp16(ptex_dir,manifest["embed"])
        gf17=_fp16_to_gf17(fp16,scale)[:model.vocab,:model.hidden]
        all_gf17.append(gf17)
        if verbose:_pf(f" {fp16.shape} -> {gf17.shape} OK")
        result["loaded"]+=1;result["total_weights"]+=gf17.size
        del fp16;gc.collect()
    else:
        gf17=np.full((model.vocab,model.hidden),8,dtype=np.uint8)
        all_gf17.append(gf17)
        if verbose:_pf(f"  [embed] init ({gf17.shape})")
    head_gf17=gf17.copy()
    all_gf17.append(head_gf17)
    result["loaded"]+=1;result["total_weights"]+=head_gf17.size
    if verbose:_pf(f"  [head] tied to embed: {head_gf17.shape}")
    layers=manifest.get("layers",{});n_layers=min(len(layers),model.n_blocks)
    for li in range(n_layers):
        li_data=layers.get(str(li),{});li_shapes=_get_adam1_layer_shapes(model,li);li_info={};rs=3.0 if li in(19,24,29,34) else scale
        for wkey in LAYER_KEYS:
            tgt=li_shapes[wkey];fp16=_decode_ptex_to_fp16(ptex_dir,li_data[wkey]) if wkey in li_data else None;gf17=_fp16_to_gf17(fp16,rs) if fp16 is not None else np.full(tgt,8,dtype=np.uint8);src=fp16.shape if fp16 is not None else tgt;match=(src[0]==tgt[0]and src[1]==tgt[1]);tag="EXACT"if match else("MISMATCH"if wkey in li_data else"INIT");result["skipped"]+=(1 if not match and wkey in li_data else 0);result["exact"]+=(1 if match and wkey in li_data else 0);all_gf17.append(gf17 if match or wkey not in li_data else np.full(tgt,8,dtype=np.uint8));result["loaded"]+=1;result["total_weights"]+=(tgt[0]*tgt[1]);li_info[wkey]={"shape":list(tgt),"tag":tag};gc.collect()
        result["layers"][str(li)]=li_info
    flat=np.concatenate([w.ravel() for w in all_gf17]).astype(np.uint8)
    expected_ws=model.get_all_weights()
    expected_size=sum(w.size for w in expected_ws)
    if flat.size!=expected_size:
        if verbose:_pf(f"  WARNING: flat size {flat.size:,} != expected {expected_size:,}")
    if verbose:_pf(f"  total flat: {flat.size:,} GF17 values ({flat.nbytes/1e6:.1f} MB)")
    try:
        from safetensors.numpy import save_file
        save_file({"weights":flat},out_path)
        if verbose:_pf(f"  saved: {out_path} ({Path(out_path).stat().st_size/1e6:.1f} MB)")
    except ImportError:
        np.savez_compressed(out_path.replace('.safetensors','.npz'),weights=flat)
        if verbose:_pf(f"  saved npz: {out_path.replace('.safetensors','.npz')}")
    dt=time.perf_counter()-t0
    result["time_s"]=round(dt,2);result["output"]=out_path
    if verbose:
        _pf(f"\n  === Translator Summary ===")
        _pf(f"  loaded:    {result['loaded']} tensors")
        _pf(f"  exact:     {result['exact']} (shape matched)")
        _pf(f"  skipped:   {result['skipped']} (shape mismatch)")
        _pf(f"  weights:   {result['total_weights']:,}")
        _pf(f"  time:      {dt:.1f}s")
    with open(str(out_p.parent/"translator_manifest.json"),"w") as f:json.dump(result,f,indent=2)
    del model;gc.collect()
    return result
def load_translated_adam1(stn_path:str=None,vocab:int=262144,act:str="cube"):
    _ensure_modules()
    from amni.model.adam import AdamModel
    if stn_path is None:stn_path=str(_root/"textures"/"models"/"adam-1"/"adam-1.safetensors")
    model=AdamModel('adam-1',vocab=vocab,act=act,auto_load=False)
    try:
        from safetensors.numpy import load_file
        flat=load_file(stn_path)['weights']
    except ImportError:flat=np.load(stn_path.replace('.safetensors','.npz'))['weights']
    ws=model.get_all_weights()
    cum=np.cumsum([0]+[w.size for w in ws]).tolist()
    for i,w in enumerate(ws):w.flat[:]=flat[cum[i]:cum[i+1]]
    _pf(f"[translator] loaded {len(ws)} weight tensors ({flat.size:,} values) into adam-1")
    return model
def generate_torch(model,tokenizer,prompt:str,max_len:int=50,seq_len:int=4)->str:
    import torch
    from amni.compute.gf17_ops import gf17_to_float
    ids=tokenizer.encode(prompt)
    if len(ids)<seq_len:ids=[0]*(seq_len-len(ids))+ids
    ws=model.get_all_weights()
    embed_f=torch.from_numpy(gf17_to_float(ws[0])).float()
    head_f=torch.from_numpy(gf17_to_float(ws[1])).float()
    blk_weights=[]
    for bi in range(model.n_blocks):
        base=2+bi*7
        blk_weights.append([torch.from_numpy(gf17_to_float(ws[base+j])).float() for j in range(7)])
    for _ in range(max_len):
        ctx=ids[-seq_len:]
        x=embed_f[ctx].unsqueeze(0)
        for bi in range(model.n_blocks):
            wq,wk,wv,wo,wg,wu,wd=blk_weights[bi]
            lc=model._layer_cfgs[bi]
            H,Hkv,Hd=lc["n_heads"],lc["n_kv_heads"],lc.get("head_dim",model._head_dim)
            xn=torch.nn.functional.rms_norm(x,x.shape[-1:])
            q=xn@wq.T;k=xn@wk.T;v=xn@wv.T
            B,S,_=x.shape
            q=q.view(B,S,H,Hd).transpose(1,2)
            k=k.view(B,S,Hkv,Hd).transpose(1,2)
            v=v.view(B,S,Hkv,Hd).transpose(1,2)
            if Hkv<H:
                reps=H//Hkv
                k=k.repeat(1,reps,1,1);v=v.repeat(1,reps,1,1)
            attn=torch.softmax(torch.matmul(q,k.transpose(-2,-1))/np.sqrt(Hd),dim=-1)
            out=torch.matmul(attn,v).transpose(1,2).contiguous().view(B,S,-1)
            x=x+out@wo.T
            xn2=torch.nn.functional.rms_norm(x,x.shape[-1:])
            g=torch.nn.functional.silu(xn2@wg.T)
            x=x+(g*(xn2@wu.T))@wd.T
        logits=torch.nn.functional.rms_norm(x,x.shape[-1:])[:,-1,:]@head_f.T
        ids.append(int(logits[0].argmax()))
    return tokenizer.decode(ids)
def translate_gemma_to_fp16_ptex(ptex_dir:str,out_dir:str=None,verbose:bool=True)->Dict:
    _ensure_modules()
    from amni.model.adam import AdamModel,ADAM_CONFIGS
    from amni.compute.prismtex import encode_fp16_base17,save_ptex,MODE_FP16_BASE17
    ptex_dir=Path(ptex_dir)
    manifest=json.load(open(str(ptex_dir/"manifest.json")))
    if out_dir is None:out_dir=str(_root/"textures"/"models"/"adam-1-fp16")
    od=Path(out_dir);od.mkdir(parents=True,exist_ok=True)
    cfg=ADAM_CONFIGS["adam-1"]
    model=AdamModel('adam-1',vocab=cfg.get("vocab",262144),act='cube',auto_load=False)
    t0=time.perf_counter()
    result={"loaded":0,"exact":0,"skipped":0,"total_weights":0,"layers":{}}
    out_manifest={"hidden":model.hidden,"n_blocks":model.n_blocks,"vocab":model.vocab,"head_dim":model._head_dim,"layer_cfgs":[lc for lc in model._layer_cfgs]}
    if verbose:
        _pf(f"[fp16-ptex] Gemma ptex -> Adam-1 fp16 ptex (lossless)")
        _pf(f"  source: {ptex_dir}")
        _pf(f"  target: {od}")
    if "embed" in manifest:
        if verbose:_pf(f"  [embed] decoding...",end="")
        fp16=_decode_ptex_to_fp16(ptex_dir,manifest["embed"])[:model.vocab,:model.hidden]
        px,nw=encode_fp16_base17(fp16)
        save_ptex(str(od/"embed.ptex"),px,nw,MODE_FP16_BASE17,aux_data={"shape":list(fp16.shape),"dtype":"float16"})
        out_manifest["embed"]={"file":"embed.ptex","shape":list(fp16.shape),"n_weights":int(nw),"mode":int(MODE_FP16_BASE17)}
        if verbose:_pf(f" {fp16.shape} -> embed.ptex ({fp16.nbytes/1e6:.1f} MB)")
        result["loaded"]+=1;result["exact"]+=1;result["total_weights"]+=fp16.size
        save_ptex(str(od/"head.ptex"),px,nw,MODE_FP16_BASE17,aux_data={"shape":list(fp16.shape),"dtype":"float16"})
        out_manifest["head"]={"file":"head.ptex","shape":list(fp16.shape),"n_weights":int(nw),"mode":int(MODE_FP16_BASE17)}
        if verbose:_pf(f"  [head] tied -> head.ptex")
        result["loaded"]+=1;result["exact"]+=1;result["total_weights"]+=fp16.size
        del fp16,px;gc.collect()
    layers=manifest.get("layers",{})
    n_layers=min(len(layers),model.n_blocks)
    out_manifest["layers"]={}
    for li in range(n_layers):
        li_data=layers.get(str(li),{})
        li_shapes=_get_adam1_layer_shapes(model,li)
        li_out={}
        for wkey in LAYER_KEYS:
            tgt=li_shapes[wkey]
            fn=f"layer_{li:03d}_{wkey}.ptex"
            if wkey in li_data:
                fp16=_decode_ptex_to_fp16(ptex_dir,li_data[wkey])
                match=(fp16.shape[0]==tgt[0] and fp16.shape[1]==tgt[1])
                if match:
                    px,nw=encode_fp16_base17(fp16)
                    save_ptex(str(od/fn),px,nw,MODE_FP16_BASE17,aux_data={"shape":list(tgt),"dtype":"float16"})
                    result["exact"]+=1
                    if verbose:_pf(f"  [L{li:02d}.{wkey:4s}] {str(fp16.shape):>16s} [EXACT] -> {fn}")
                else:
                    if verbose:_pf(f"  [L{li:02d}.{wkey:4s}] MISMATCH {fp16.shape} != {tgt}")
                    result["skipped"]+=1
                    fp16=np.zeros(tgt,dtype=np.float16)
                    px,nw=encode_fp16_base17(fp16)
                    save_ptex(str(od/fn),px,nw,MODE_FP16_BASE17,aux_data={"shape":list(tgt),"dtype":"float16"})
                del fp16,px;gc.collect()
            else:
                fp16=np.zeros(tgt,dtype=np.float16)
                px,nw=encode_fp16_base17(fp16)
                save_ptex(str(od/fn),px,nw,MODE_FP16_BASE17,aux_data={"shape":list(tgt),"dtype":"float16"})
                if verbose:_pf(f"  [L{li:02d}.{wkey:4s}] INIT -> {fn}")
                del fp16,px;gc.collect()
            li_out[wkey]={"file":fn,"shape":list(tgt),"n_weights":int(tgt[0]*tgt[1]),"mode":int(MODE_FP16_BASE17)}
            result["loaded"]+=1;result["total_weights"]+=(tgt[0]*tgt[1])
        out_manifest["layers"][str(li)]=li_out
    dt=time.perf_counter()-t0
    result["time_s"]=round(dt,2);result["output"]=str(od)
    with open(str(od/"manifest.json"),"w") as f:json.dump(out_manifest,f,indent=2)
    if verbose:
        _pf(f"\n  === FP16 Ptex Summary ===")
        _pf(f"  loaded:  {result['loaded']} tensors")
        _pf(f"  exact:   {result['exact']} | skipped: {result['skipped']}")
        _pf(f"  weights: {result['total_weights']:,}")
        _pf(f"  time:    {dt:.1f}s")
    del model;gc.collect()
    return result
def export_norms_to_ptex(stn_path:str,ptex_dir:str,verbose:bool=True)->Dict:
    _ensure_modules()
    from safetensors import safe_open
    from amni.compute.prismtex import encode_fp16_base17,save_ptex,MODE_FP16_BASE17
    from amni.model.adam import ADAM_CONFIGS
    import torch
    od=Path(ptex_dir)
    manifest=json.load(open(str(od/"manifest.json")))
    f=safe_open(stn_path,'pt')
    pfx="model.language_model."
    cfg_path=Path(stn_path).parent/"config.json"
    gcfg=json.load(open(str(cfg_path))).get("text_config",{}) if cfg_path.exists() else {}
    manifest["rms_norm_eps"]=gcfg.get("rms_norm_eps",1e-6)
    manifest["hidden_activation"]=gcfg.get("hidden_activation","gelu_pytorch_tanh")
    manifest["final_logit_softcapping"]=gcfg.get("final_logit_softcapping",30.0)
    manifest["sliding_window"]=gcfg.get("sliding_window",512)
    manifest["rope_params"]=gcfg.get("rope_parameters",{"full_attention":{"partial_rotary_factor":0.25,"rope_theta":1000000.0},"sliding_attention":{"rope_theta":10000.0}})
    manifest["layer_types"]=gcfg.get("layer_types",GEMMA_LAYER_TYPES)
    acfg=ADAM_CONFIGS.get("adam-1",{})
    from amni.model.adam import AdamModel
    mdl=AdamModel('adam-1',vocab=acfg.get("vocab",262144),act='cube',auto_load=False)
    manifest["layer_cfgs"]=[lc for lc in mdl._layer_cfgs]
    del mdl
    def _save_w(w_tensor,name):
        w=w_tensor.float().half().numpy()
        px,nw=encode_fp16_base17(w)
        save_ptex(str(od/name),px,nw,MODE_FP16_BASE17,aux_data={"shape":list(w.shape),"dtype":"float16"})
        return {"file":name,"shape":list(w.shape),"n_weights":int(nw),"mode":int(MODE_FP16_BASE17)}
    fn_key=pfx+"norm.weight"
    count=0
    if fn_key in f.keys():
        manifest["final_norm"]=_save_w(f.get_tensor(fn_key),"final_norm.ptex")
        count+=1
        if verbose:_pf(f"  [final_norm] {f.get_tensor(fn_key).shape}")
    for li in range(manifest["n_blocks"]):
        li_data=manifest["layers"].get(str(li),{})
        for nk,gk in GEMMA_NORM_MAP.items():
            full_key=f"{pfx}layers.{li}.{gk}"
            if full_key in f.keys():
                fn=f"layer_{li:03d}_{nk}.ptex"
                li_data[nk]=_save_w(f.get_tensor(full_key),fn)
                count+=1
        manifest["layers"][str(li)]=li_data
    with open(str(od/"manifest.json"),"w") as fout:json.dump(manifest,fout,indent=2)
    if verbose:_pf(f"[norms] exported {count} tensors to {od}")
    gc.collect()
    return manifest
def translate_safetensors_to_fp16_ptex(stn_path:str,ptex_dir:str,verbose:bool=True)->Dict:
    _ensure_modules()
    from safetensors import safe_open
    from amni.compute.prismtex import encode_fp16_base17,save_ptex,MODE_FP16_BASE17
    from amni.model.adam import AdamModel,ADAM_CONFIGS
    import torch
    od=Path(ptex_dir);od.mkdir(parents=True,exist_ok=True)
    f=safe_open(stn_path,'pt');pfx="model.language_model."
    cfg_path=Path(stn_path).parent/"config.json"
    gcfg=json.load(open(str(cfg_path))).get("text_config",{}) if cfg_path.exists() else {}
    acfg=ADAM_CONFIGS.get("adam-1",{})
    mdl=AdamModel('adam-1',vocab=acfg.get("vocab",262144),act='cube',auto_load=False)
    layer_cfgs=[lc for lc in mdl._layer_cfgs];del mdl
    manifest={"hidden":1536,"n_blocks":35,"vocab":262144,"head_dim":256,"layer_cfgs":layer_cfgs,
        "rms_norm_eps":gcfg.get("rms_norm_eps",1e-6),"hidden_activation":gcfg.get("hidden_activation","gelu_pytorch_tanh"),
        "final_logit_softcapping":gcfg.get("final_logit_softcapping",30.0),"sliding_window":gcfg.get("sliding_window",512),
        "rope_params":gcfg.get("rope_parameters",{"full_attention":{"partial_rotary_factor":0.25,"rope_theta":1000000.0},"sliding_attention":{"rope_theta":10000.0}}),
        "layer_types":gcfg.get("layer_types",GEMMA_LAYER_TYPES),"layers":{}}
    result={"exact":0,"total_weights":0}
    t0=time.perf_counter()
    def _save_w(tensor,name,save_fp16_cache=False):
        w=tensor.float().half().numpy();sh=list(w.shape)
        px,nw=encode_fp16_base17(w)
        save_ptex(str(od/name),px,nw,MODE_FP16_BASE17,aux_data={"shape":sh,"dtype":"float16"})
        if save_fp16_cache:
            cache_name=name.replace(".ptex",".fp16.bin")
            w.ravel().tofile(str(od/cache_name))
            if verbose:_pf(f"    [cache] {cache_name} ({w.nbytes/1e9:.2f} GB)")
        result["exact"]+=1;result["total_weights"]+=int(nw)
        del w,px;gc.collect()
        return {"file":name,"shape":sh,"n_weights":int(nw),"mode":int(MODE_FP16_BASE17)}
    if verbose:_pf(f"[stn->fp16] Gemma safetensors -> fp16 ptex (lossless)")
    if verbose:_pf(f"  source: {stn_path}")
    if verbose:_pf(f"  target: {od}")
    ek=pfx+"embed_tokens.weight"
    if verbose:_pf(f"  [embed] {f.get_tensor(ek).shape}...",end="")
    manifest["embed"]=_save_w(f.get_tensor(ek),"embed.ptex")
    manifest["head"]=_save_w(f.get_tensor(ek),"head.ptex")
    if verbose:_pf(f" done")
    fnk=pfx+"norm.weight"
    if fnk in f.keys():
        manifest["final_norm"]=_save_w(f.get_tensor(fnk),"final_norm.ptex")
        if verbose:_pf(f"  [final_norm] {f.get_tensor(fnk).shape}")
    for li in range(35):
        li_key=f"{pfx}layers.{li}."
        li_data={}
        for wkey,gkey in GEMMA_PROJ_MAP.items():
            full_key=li_key+gkey
            if full_key in f.keys():
                fn=f"layer_{li:03d}_{wkey}.ptex"
                li_data[wkey]=_save_w(f.get_tensor(full_key),fn)
                if verbose:_pf(f"  [L{li:02d}.{wkey:4s}] {str(f.get_tensor(full_key).shape):>16s} -> {fn}")
        for nk,gk in GEMMA_NORM_MAP.items():
            full_key=li_key+gk
            if full_key in f.keys():
                fn=f"layer_{li:03d}_{nk}.ptex"
                li_data[nk]=_save_w(f.get_tensor(full_key),fn)
        manifest["layers"][str(li)]=li_data
    pli_cfg={"hidden_size_per_layer_input":gcfg.get("hidden_size_per_layer_input",256),"vocab_size_per_layer_input":gcfg.get("vocab_size_per_layer_input",262144),"num_kv_shared_layers":gcfg.get("num_kv_shared_layers",20)}
    manifest["pli_config"]=pli_cfg
    for gk,sk in GEMMA_PLI_GLOBAL.items():
        full_key=pfx+sk
        if full_key in f.keys():
            fn=f"pli_{gk}.ptex"
            is_big=gk=="embed_tokens_per_layer"
            manifest[gk]=_save_w(f.get_tensor(full_key),fn,save_fp16_cache=is_big)
            if verbose:_pf(f"  [PLI.{gk}] {str(f.get_tensor(full_key).shape):>20s} -> {fn}")
    for li in range(35):
        li_key=f"{pfx}layers.{li}."
        li_data2=manifest["layers"].get(str(li),{})
        for pk,sk in GEMMA_PLI_LAYER.items():
            full_key=li_key+sk
            if full_key in f.keys():
                fn=f"layer_{li:03d}_{pk}.ptex"
                li_data2[pk]=_save_w(f.get_tensor(full_key),fn)
                if verbose:_pf(f"  [L{li:02d}.{pk}] {str(f.get_tensor(full_key).shape):>16s} -> {fn}")
        manifest["layers"][str(li)]=li_data2
    dt=time.perf_counter()-t0
    with open(str(od/"manifest.json"),"w") as fout:json.dump(manifest,fout,indent=2)
    if verbose:
        _pf(f"\n  === Direct FP16 Ptex Summary ===")
        _pf(f"  tensors: {result['exact']} | weights: {result['total_weights']:,}")
        _pf(f"  time: {dt:.1f}s")
    gc.collect()
    return result
def pack_tokenizer_ptex(tok_dir:str,out_path:str,verbose:bool=True):
    from amni.compute.prismtex import save_ptex
    td=Path(tok_dir)
    tok_json=td/"tokenizer.json"
    tok_cfg=td/"tokenizer_config.json"
    chat_tpl=td/"chat_template.jinja"
    payload={}
    if tok_json.exists():payload["tokenizer.json"]=tok_json.read_text(encoding="utf-8")
    if tok_cfg.exists():payload["tokenizer_config.json"]=tok_cfg.read_text(encoding="utf-8")
    if chat_tpl.exists():payload["chat_template.jinja"]=chat_tpl.read_text(encoding="utf-8")
    raw=json.dumps(payload,ensure_ascii=False).encode("utf-8")
    n_bytes=len(raw)
    padded=raw+b'\x00'*((-n_bytes)%4)
    pixels=np.frombuffer(padded,dtype=np.uint8).reshape(-1,4)
    save_ptex(out_path,pixels,n_bytes,255,aux_data={"type":"tokenizer","src_dir":str(td),"n_bytes":n_bytes})
    if verbose:_pf(f"[tokenizer] packed {n_bytes:,} bytes -> {out_path} ({Path(out_path).stat().st_size/1e6:.1f} MB)")
def load_tokenizer_ptex(ptex_path:str):
    from amni.compute.prismtex import load_ptex
    import tempfile
    pri,n_w,mode,n_px=load_ptex(ptex_path)
    raw_bytes=pri.ravel().tobytes()[:n_w]
    payload=json.loads(raw_bytes.decode("utf-8"))
    td=tempfile.mkdtemp(prefix="adam1_tok_")
    for fn,content in payload.items():
        with open(os.path.join(td,fn),"w",encoding="utf-8") as f:f.write(content)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(td,local_files_only=True)
def load_fp16_model(ptex_dir:str,verbose:bool=True)->Dict:
    _ensure_modules()
    from amni.compute.prismtex import load_ptex,decode_fp16_base17
    import torch
    od=Path(ptex_dir)
    manifest=json.load(open(str(od/"manifest.json")))
    result={"hidden":manifest["hidden"],"n_blocks":manifest["n_blocks"],"vocab":manifest["vocab"],"head_dim":manifest.get("head_dim",256),"layer_cfgs":manifest["layer_cfgs"]}
    result["rms_norm_eps"]=manifest.get("rms_norm_eps",1e-6)
    result["hidden_activation"]=manifest.get("hidden_activation","gelu_pytorch_tanh")
    result["softcap"]=manifest.get("final_logit_softcapping",30.0)
    result["sliding_window"]=manifest.get("sliding_window",512)
    result["layer_types"]=manifest.get("layer_types",GEMMA_LAYER_TYPES)
    rope_params=manifest.get("rope_params",{})
    def _load_ptex_fp16(info,keep_fp16=False):
        cache_bin=str(od/info["file"].replace(".ptex",".fp16.bin"))
        if os.path.exists(cache_bin):
            fp16=np.fromfile(cache_bin,dtype=np.float16).reshape(info["shape"])
            if keep_fp16:return torch.from_numpy(fp16).contiguous()
            return torch.from_numpy(fp16.astype(np.float32)).contiguous()
        pri,n_w,mode,n_px=load_ptex(str(od/info["file"]))
        fp16=decode_fp16_base17(pri,info["n_weights"])
        if keep_fp16:return torch.from_numpy(fp16.reshape(info["shape"])).contiguous()
        return torch.from_numpy(fp16.reshape(info["shape"]).astype(np.float32)).contiguous()
    if verbose:_pf(f"[fp16] loading embed...",end="")
    result["embed"]=_load_ptex_fp16(manifest["embed"])
    if verbose:_pf(f" {result['embed'].shape}")
    if verbose:_pf(f"[fp16] loading head...",end="")
    result["head"]=_load_ptex_fp16(manifest["head"])
    if verbose:_pf(f" {result['head'].shape}")
    result["blocks"]=[];result["norms"]=[]
    for li in range(manifest["n_blocks"]):
        li_data=manifest["layers"][str(li)]
        bw=[_load_ptex_fp16(li_data[k]) for k in LAYER_KEYS]
        result["blocks"].append(bw)
        nm={}
        for nk in NORM_KEYS:
            if nk in li_data:nm[nk]=_load_ptex_fp16(li_data[nk])
        result["norms"].append(nm)
        if verbose and li%5==0:_pf(f"  layer {li}: q={bw[0].shape} gate={bw[4].shape} norms={list(nm.keys())}")
    if "final_norm" in manifest:
        result["final_norm"]=_load_ptex_fp16(manifest["final_norm"])
        if verbose:_pf(f"[fp16] final_norm: {result['final_norm'].shape}")
    else:result["final_norm"]=None
    max_pos=4096
    result["rope_caches"]={}
    for lt_name,rp in rope_params.items():
        theta=rp.get("rope_theta",10000.0)
        pf_val=rp.get("partial_rotary_factor",1.0)
        hd=512 if lt_name=="full_attention" else 256
        rot_dim=int(hd*pf_val)
        half=rot_dim//2
        freqs=1.0/(theta**(torch.arange(0,rot_dim,2,dtype=torch.float32)/rot_dim))
        t=torch.arange(max_pos,dtype=torch.float32)
        angles=torch.outer(t,freqs)
        emb=torch.cat([angles,angles],dim=-1)
        result["rope_caches"][lt_name]={"cos":emb.cos(),"sin":emb.sin(),"rot_dim":rot_dim}
    pli_cfg=manifest.get("pli_config",{})
    result["num_kv_shared_layers"]=pli_cfg.get("num_kv_shared_layers",20)
    result["pli"]=None
    if "embed_tokens_per_layer" in manifest:
        if verbose:_pf(f"[fp16] loading PLI weights...")
        pli={"n_layers":manifest["n_blocks"],"hd_pli":pli_cfg.get("hidden_size_per_layer_input",256)}
        pli["embed_per_layer"]=_load_ptex_fp16(manifest["embed_tokens_per_layer"],keep_fp16=True)
        if verbose:_pf(f"  embed_tokens_per_layer: {pli['embed_per_layer'].shape}")
        pli["model_projection"]=_load_ptex_fp16(manifest["per_layer_model_projection"])
        pli["projection_norm"]=_load_ptex_fp16(manifest["per_layer_projection_norm"])
        pli["model_proj_scale"]=manifest["hidden"]**-0.5
        pli["input_scale"]=2.0**-0.5
        pli["embed_scale"]=pli_cfg.get("hidden_size_per_layer_input",256)**0.5
        pli["layers"]=[]
        for li in range(manifest["n_blocks"]):
            li_data=manifest["layers"][str(li)]
            lpli={}
            for pk in GEMMA_PLI_LAYER.keys():
                if pk in li_data:lpli[pk]=_load_ptex_fp16(li_data[pk])
            pli["layers"].append(lpli)
        result["pli"]=pli
        if verbose:_pf(f"  PLI: {len(pli['layers'])} layers loaded")
    if verbose:_pf(f"[fp16] loaded {manifest['n_blocks']} blocks, RoPE caches: {list(result['rope_caches'].keys())}")
    return result
def _rms_norm(x,w,eps=1e-6):
    import torch
    rms=x.pow(2).mean(-1,keepdim=True).add(eps).rsqrt()
    return (x*rms*w) if w is not None else (x*rms)
def _rotate_half(x):
    import torch
    x1,x2=x[...,:x.shape[-1]//2],x[...,x.shape[-1]//2:]
    return torch.cat([-x2,x1],dim=-1)
def _apply_rope(x,cos,sin,rot_dim):
    import torch
    xr,xp=x[...,:rot_dim],x[...,rot_dim:]
    c=cos.unsqueeze(0).unsqueeze(0)
    s=sin.unsqueeze(0).unsqueeze(0)
    xr=xr*c+_rotate_half(xr)*s
    return torch.cat([xr,xp],dim=-1) if xp.shape[-1]>0 else xr
def generate_fp16_torch(model_data:Dict,tokenizer,prompt:str,max_len:int=50,seq_len:int=512)->str:
    import torch
    ids=tokenizer.encode(prompt)
    embed=model_data["embed"];head=model_data["head"]
    blocks=model_data["blocks"];norms=model_data["norms"]
    lcfgs=model_data["layer_cfgs"];fn_norm=model_data.get("final_norm")
    eps=model_data.get("rms_norm_eps",1e-6)
    softcap=model_data.get("softcap",30.0)
    ltypes=model_data.get("layer_types",[])
    rope_caches=model_data.get("rope_caches",{})
    hidden=model_data["hidden"]
    n_blocks=model_data["n_blocks"]
    scale_embed=np.sqrt(hidden)
    act_fn=torch.nn.functional.gelu if model_data.get("hidden_activation","gelu_pytorch_tanh")=="gelu_pytorch_tanh" else torch.nn.functional.silu
    act_kw={"approximate":"tanh"} if act_fn==torch.nn.functional.gelu else {}
    n_kv_shared=model_data.get("num_kv_shared_layers",20)
    first_shared=n_blocks-n_kv_shared if n_kv_shared>0 else n_blocks
    kv_src={}
    for bi in range(first_shared,n_blocks):
        lt_bi=ltypes[bi] if bi<len(ltypes) else "sliding_attention"
        for src in range(first_shared-1,-1,-1):
            if (ltypes[src] if src<len(ltypes) else "sliding_attention")==lt_bi:
                kv_src[bi]=src;break
    pli=model_data.get("pli")
    with torch.no_grad():
        for _ in range(max_len):
            ctx=ids[-seq_len:]
            seq_start=max(0,len(ids)-seq_len)
            x=embed[ctx].unsqueeze(0)*scale_embed
            B,S,D=x.shape
            positions=torch.arange(seq_start,seq_start+S,dtype=torch.long)
            pli_inputs=None
            if pli is not None:
                pli_emb=pli["embed_per_layer"][ctx].float()*pli["embed_scale"]
                pli_emb=pli_emb.view(S,n_blocks,pli["hd_pli"]).unsqueeze(0)
                pli_proj=(x@pli["model_projection"].T)*pli["model_proj_scale"]
                pli_proj=pli_proj.view(B,S,n_blocks,pli["hd_pli"])
                pli_proj=_rms_norm(pli_proj,pli["projection_norm"],eps)
                pli_inputs=(pli_proj+pli_emb)*pli["input_scale"]
            shared_kv={}
            for bi in range(n_blocks):
                wq,wk,wv,wo,wg,wu,wd=blocks[bi]
                lc=lcfgs[bi]
                H,Hkv,Hd=lc["n_heads"],lc["n_kv_heads"],lc.get("head_dim",256)
                nm=norms[bi] if bi<len(norms) else {}
                lt=ltypes[bi] if bi<len(ltypes) else "sliding_attention"
                xn=_rms_norm(x,nm.get("input_ln"),eps)
                q=(xn@wq.T).view(B,S,H,Hd).transpose(1,2)
                qn_w=nm.get("q_norm")
                if qn_w is not None:q=_rms_norm(q,qn_w,eps)
                if bi in kv_src:
                    k,v=shared_kv[kv_src[bi]]
                else:
                    k=(xn@wk.T).view(B,S,Hkv,Hd).transpose(1,2)
                    v=(xn@wv.T).view(B,S,Hkv,Hd).transpose(1,2)
                    kn_w=nm.get("k_norm")
                    if kn_w is not None:k=_rms_norm(k,kn_w,eps)
                    v=_rms_norm(v,None,eps)
                    rope=rope_caches.get(lt)
                    if rope:
                        rc,rs,rd=rope["cos"][positions],rope["sin"][positions],rope["rot_dim"]
                        k=_apply_rope(k,rc,rs,rd)
                    shared_kv[bi]=(k,v)
                rope=rope_caches.get(lt)
                if rope:
                    rc,rs,rd=rope["cos"][positions],rope["sin"][positions],rope["rot_dim"]
                    q=_apply_rope(q,rc,rs,rd)
                if Hkv<H:
                    reps=H//Hkv
                    k=k.repeat(1,reps,1,1);v=v.repeat(1,reps,1,1)
                attn=torch.matmul(q,k.transpose(-2,-1))
                if S>1:
                    causal=torch.tril(torch.ones(S,S,dtype=torch.bool))
                    if lt=="sliding_attention":
                        sw=model_data.get("sliding_window",512)
                        for i in range(S):
                            for j in range(S):
                                if i-j>sw:causal[i,j]=False
                    attn=attn.masked_fill(~causal.unsqueeze(0).unsqueeze(0),float('-inf'))
                attn=torch.softmax(attn,dim=-1)
                out=torch.matmul(attn,v).transpose(1,2).contiguous().view(B,S,-1)
                out=out@wo.T
                paln=nm.get("post_attn_ln")
                if paln is not None:out=_rms_norm(out,paln,eps)
                x=x+out
                xn2=_rms_norm(x,nm.get("pre_ff_ln"),eps)
                g=act_fn(xn2@wg.T,**act_kw)
                ffn=(g*(xn2@wu.T))@wd.T
                pfln=nm.get("post_ff_ln")
                if pfln is not None:ffn=_rms_norm(ffn,pfln,eps)
                x=x+ffn
                if pli_inputs is not None and bi<len(pli["layers"]):
                    lpli=pli["layers"][bi]
                    if "pli_gate" in lpli:
                        residual=x
                        gout=x@lpli["pli_gate"].T
                        gact=act_fn(gout,**act_kw)
                        x=gact*pli_inputs[:,:,bi,:]
                        x=x@lpli["pli_proj"].T
                        x=_rms_norm(x,lpli.get("pli_norm"),eps)
                        x=residual+x
                ls=nm.get("layer_scalar")
                if ls is not None:x=x*ls
            xf=_rms_norm(x,fn_norm,eps)
            logits=xf[:,-1,:]@head.T
            if softcap>0:logits=softcap*torch.tanh(logits/softcap)
            nxt=int(logits[0].argmax())
            ids.append(nxt)
            if nxt==tokenizer.eos_token_id:break
    return tokenizer.decode(ids,skip_special_tokens=True)
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(description="GF17 Translator: Gemma ptex -> Adam-1 native")
    ap.add_argument("--ptex-dir",default=str(_root/"textures"/"gemma-e2b-prism"),help="Gemma ptex directory")
    ap.add_argument("--output",default=None,help="Output safetensors path")
    ap.add_argument("--scale",type=float,default=2.0,help="GF17 quantization scale")
    ap.add_argument("--test",action="store_true",help="Run generation test after translation")
    ap.add_argument("--fp16",action="store_true",help="Translate to fp16 ptex (lossless)")
    ap.add_argument("--tokenizer-dir",default=str(_root/"models"/"TrevorJS"/"gemma-4-E4B-it-uncensored"),help="Tokenizer source dir")
    args=ap.parse_args()
    if args.fp16:
        out_dir=args.output or str(_root/"textures"/"models"/"adam-1-fp16")
        result=translate_gemma_to_fp16_ptex(args.ptex_dir,out_dir)
        stn_path=Path(args.tokenizer_dir)/"model.safetensors"
        if stn_path.exists():export_norms_to_ptex(str(stn_path),out_dir)
        pack_tokenizer_ptex(args.tokenizer_dir,str(Path(out_dir)/"tokenizer.ptex"))
        if args.test:
            weights=load_fp16_model(out_dir)
            tok=load_tokenizer_ptex(str(Path(out_dir)/"tokenizer.ptex"))
            _pf("\n=== Generation Test (fp16 torch) ===")
            for p in["What is Python?","def hello():","The capital of France is"]:
                out=generate_fp16_torch(weights,tok,p,max_len=30)
                _pf(f"  '{p}' -> '{out}'")
    else:
        result=translate_gemma_ptex_to_adam1(args.ptex_dir,args.output,args.scale)
        if args.test:
            from amni.training.adam_distill import CharTokenizer
            model=load_translated_adam1()
            tok=CharTokenizer(' abcdefghijklmnopqrstuvwxyz0123456789(){}[].:;,!?=+-*/<>_#')
            _pf("\n=== Generation Test (torch matmul) ===")
            for p in["def hello():","def fib(n):","class Node:","for i in range(10):"]:
                t0=time.perf_counter()
                out=generate_torch(model,tok,p,max_len=30,seq_len=4)
                dt=(time.perf_counter()-t0)*1000
                _pf(f"  [{dt:6.0f}ms] '{p}' -> '{out}'")
