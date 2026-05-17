import os,json
from pathlib import Path
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from amni.compute.reffelt4 import encode_fp16_to_rgba4,decode_rgba4_to_fp16
def encode_tensor_to_rgba(t):
    a=t.detach().to(torch.float16).contiguous().cpu().numpy()
    return encode_fp16_to_rgba4(a)
def decode_rgba_to_tensor(rgba,shape,device='cpu',dtype=torch.float16):
    fp16=decode_rgba4_to_fp16(rgba).reshape(shape)
    return torch.from_numpy(np.ascontiguousarray(fp16)).to(device=device,dtype=dtype)
def roundtrip_bit_exact(t):
    a=t.detach().to(torch.float16).contiguous().cpu().numpy().view(np.uint16)
    rgba=encode_fp16_to_rgba4(a.view(np.float16))
    decoded=decode_rgba4_to_fp16(rgba).view(np.uint16)
    return bool(np.array_equal(a.reshape(-1),decoded[:a.size]))
class ReffeltShadowLinear(nn.Module):
    def __init__(self,in_features,out_features,bias=True,device=None,dtype=torch.bfloat16):
        super().__init__()
        self.in_features=in_features;self.out_features=out_features
        self.weight=nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))
        nn.init.kaiming_uniform_(self.weight,a=5**0.5)
        if bias:
            self.bias=nn.Parameter(torch.zeros(out_features,device=device,dtype=dtype))
        else:
            self.register_parameter('bias',None)
    def forward(self,x):return F.linear(x,self.weight,self.bias)
    def to_atlas_bytes(self):
        rgba=encode_tensor_to_rgba(self.weight)
        return rgba.tobytes(),tuple(self.weight.shape)
    def to_atlas_page(self,page_w=None):
        rgba=encode_tensor_to_rgba(self.weight)
        if page_w is None:page_w=self.in_features
        n=rgba.shape[0]
        h=(n+page_w-1)//page_w
        page=np.zeros((h,page_w,4),dtype=np.uint8)
        page.reshape(-1,4)[:n]=rgba
        return page,n,h,page_w
    def save_atlas(self,path):
        page,n,h,page_w=self.to_atlas_page()
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        with open(path,'wb') as f:f.write(np.ascontiguousarray(page).tobytes())
        return {'shape':list(self.weight.shape),'n_pixels':int(n),'page_h':int(h),'page_w':int(page_w),'ptex_bytes':int(page.nbytes)}
    def load_atlas(self,path,shape):
        raw=np.fromfile(path,dtype=np.uint8)
        n=int(np.prod(shape))
        rgba=raw.reshape(-1,4)[:n]
        decoded=decode_rgba4_to_fp16(rgba).reshape(shape)
        with torch.no_grad():
            self.weight.copy_(torch.from_numpy(np.ascontiguousarray(decoded)).to(self.weight.device,self.weight.dtype))
    def assert_bit_exact_roundtrip(self):
        return roundtrip_bit_exact(self.weight)
def save_model_as_bake(model,out_dir,model_name):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);(out/'tensors').mkdir(exist_ok=True)
    manifest={'model_name':model_name,'bake_version':'v5.4.0_native','reffelt_scheme':'rgba4','tensors':{}}
    n_ok=0;total_params=0;ptex_total=0;fp16_baseline=0
    for name,mod in model.named_modules():
        if isinstance(mod,ReffeltShadowLinear):
            sname=name.replace('.','_')+'_weight.ptex'
            page,n,h,page_w=mod.to_atlas_page()
            fpath=out/'tensors'/sname
            with open(fpath,'wb') as f:f.write(np.ascontiguousarray(page).tobytes())
            shape=tuple(mod.weight.shape)
            params=int(np.prod(shape));total_params+=params;fp16_baseline+=params*2
            ptex_bytes=fpath.stat().st_size;ptex_total+=ptex_bytes
            manifest['tensors'][f'{name}.weight']={'shape':list(shape),'source_dtype':'bfloat16','ptex_path':f'tensors/{sname}','ptex_bytes':int(ptex_bytes),'params':params,'page_h':int(h),'page_w':int(page_w),'n_pixels':int(n),'u16_per_elem':1}
            n_ok+=1
            if mod.bias is not None:
                bname=name.replace('.','_')+'_bias.ptex'
                bpage,bn,bh,bpw=ReffeltShadowLinear.__dict__['to_atlas_page'](type('T',(),{'weight':mod.bias,'in_features':int(mod.bias.shape[0])})())
                bpath=out/'tensors'/bname
                with open(bpath,'wb') as f:f.write(np.ascontiguousarray(bpage).tobytes())
                bshape=tuple(mod.bias.shape)
                bparams=int(np.prod(bshape));total_params+=bparams;fp16_baseline+=bparams*2
                bptex_bytes=bpath.stat().st_size;ptex_total+=bptex_bytes
                manifest['tensors'][f'{name}.bias']={'shape':list(bshape),'source_dtype':'bfloat16','ptex_path':f'tensors/{bname}','ptex_bytes':int(bptex_bytes),'params':bparams,'page_h':int(bh),'page_w':int(bpw),'n_pixels':int(bn),'u16_per_elem':1}
                n_ok+=1
    manifest['tensor_count']=n_ok;manifest['total_params']=total_params;manifest['fp16_baseline_bytes']=fp16_baseline;manifest['ptex_total_bytes']=ptex_total
    with open(out/'manifest.json','w') as f:json.dump(manifest,f,indent=2)
    return manifest
