import os,json
from pathlib import Path
import numpy as np,torch,torch.nn as nn
from amni.compute.reffelt4 import encode_fp16_to_rgba4
from amni.training.reffelt_shadow import ReffeltShadowLinear
def swap_linears_to_shadow(model,verbose=False):
    n=0
    for parent_name,parent in model.named_modules():
        for cname,child in list(parent.named_children()):
            if isinstance(child,nn.Linear) and not isinstance(child,ReffeltShadowLinear):
                shadow=ReffeltShadowLinear(child.in_features,child.out_features,bias=child.bias is not None,device=child.weight.device,dtype=child.weight.dtype)
                with torch.no_grad():
                    shadow.weight.copy_(child.weight)
                    if child.bias is not None and shadow.bias is not None:shadow.bias.copy_(child.bias)
                setattr(parent,cname,shadow)
                n+=1
                if verbose:print(f'  swapped {parent_name}.{cname} ({child.in_features}x{child.out_features})')
    return n
def _encode_param_to_page(t,page_w=None):
    a=t.detach().contiguous().cpu()
    if a.dtype==torch.bfloat16:a=a.view(torch.float16)
    elif a.dtype not in (torch.float16,torch.uint16,torch.float32,torch.float64):a=a.to(torch.float16)
    arr=a.numpy()
    if a.dtype==torch.float32:upe=2;u16=arr.view(np.uint16).reshape(-1)
    elif a.dtype==torch.float64:upe=4;u16=arr.view(np.uint16).reshape(-1)
    else:upe=1;u16=arr.view(np.uint16).reshape(-1)
    fp16_view=u16.view(np.float16)
    rgba=encode_fp16_to_rgba4(fp16_view)
    n=rgba.shape[0]
    shape=tuple(int(x) for x in t.shape)
    if page_w is None:page_w=int(shape[-1])*upe if len(shape)>=2 else min(4096,n)
    if page_w<=0:page_w=min(4096,n) if n>0 else 1
    h=(n+page_w-1)//page_w
    page=np.zeros((h,page_w,4),dtype=np.uint8)
    page.reshape(-1,4)[:n]=rgba
    return page,n,h,page_w,upe,shape
def save_native_bake(model,out_dir,model_name,verbose=False):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);(out/'tensors').mkdir(exist_ok=True)
    manifest={'model_name':model_name,'bake_version':'v5.4.1_native','reffelt_scheme':'rgba4','tensors':{}}
    n_ok=0;total_params=0;ptex_total=0;fp16_baseline=0;n_skipped=0
    for name,p in model.named_parameters():
        if not isinstance(p,torch.Tensor):n_skipped+=1;continue
        if p.numel()==0:n_skipped+=1;continue
        try:
            page,n,h,page_w,upe,shape=_encode_param_to_page(p)
        except Exception as e:
            if verbose:print(f'  [skip] {name}: {e}')
            n_skipped+=1;continue
        sname=name.replace('.','_')+'.ptex'
        fpath=out/'tensors'/sname
        with open(fpath,'wb') as f:f.write(np.ascontiguousarray(page).tobytes())
        ptex_bytes=fpath.stat().st_size
        ptex_total+=ptex_bytes
        params=int(np.prod(shape));total_params+=params;fp16_baseline+=params*2
        manifest['tensors'][name]={'shape':list(shape),'source_dtype':str(p.dtype).replace('torch.',''),'ptex_path':f'tensors/{sname}','ptex_bytes':int(ptex_bytes),'params':params,'page_h':int(h),'page_w':int(page_w),'n_pixels':int(n),'u16_per_elem':int(upe)}
        n_ok+=1
        if verbose and n_ok%50==0:print(f'  [{n_ok}] {name:60s} shape={shape}')
    for name,b in model.named_buffers():
        if b is None or b.numel()==0:continue
        try:
            page,n,h,page_w,upe,shape=_encode_param_to_page(b)
        except Exception:continue
        sname=name.replace('.','_')+'.ptex'
        fpath=out/'tensors'/sname
        with open(fpath,'wb') as f:f.write(np.ascontiguousarray(page).tobytes())
        ptex_bytes=fpath.stat().st_size;ptex_total+=ptex_bytes
        params=int(np.prod(shape));total_params+=params;fp16_baseline+=params*2
        manifest['tensors'][name]={'shape':list(shape),'source_dtype':str(b.dtype).replace('torch.',''),'ptex_path':f'tensors/{sname}','ptex_bytes':int(ptex_bytes),'params':params,'page_h':int(h),'page_w':int(page_w),'n_pixels':int(n),'u16_per_elem':int(upe),'is_buffer':True}
        n_ok+=1
    manifest['tensor_count']=n_ok;manifest['total_params']=total_params;manifest['fp16_baseline_bytes']=fp16_baseline;manifest['ptex_total_bytes']=ptex_total
    manifest['compression_ratio']=fp16_baseline/ptex_total if ptex_total>0 else 0.0
    with open(out/'manifest.json','w') as f:json.dump(manifest,f,indent=2)
    return manifest
class AdamA1NativeBuilder:
    @staticmethod
    def from_qwen2(qwen2_model,verbose=False):
        n=swap_linears_to_shadow(qwen2_model,verbose=verbose)
        return qwen2_model,n
    @staticmethod
    def save_bake(model,out_dir,model_name,verbose=False):
        return save_native_bake(model,out_dir,model_name,verbose=verbose)
