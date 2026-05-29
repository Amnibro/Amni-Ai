import os,sys,json,time,hashlib,argparse,numpy as np,torch
from pathlib import Path
from safetensors import safe_open
_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_ROOT))
from amni.compute.reffelt4 import encode_fp16_to_rgba4,decode_rgba4_to_fp16,pack_rgba4_page
def to_uint16_np(t):
    t=t.contiguous().cpu()
    if t.dtype==torch.bfloat16:return t.view(torch.float16).numpy().view(np.uint16)
    if t.dtype==torch.float16:return t.numpy().view(np.uint16)
    if t.dtype==torch.uint16:return t.numpy()
    if t.dtype==torch.float32:return t.numpy().view(np.uint16)
    if t.dtype==torch.float64:return t.numpy().view(np.uint16)
    if t.dtype in (torch.float8_e4m3fn,torch.float8_e5m2):return t.to(torch.bfloat16).view(torch.float16).numpy().view(np.uint16)
    raise NotImplementedError(f'unsupported source dtype {t.dtype} — extend bake to handle')
def _u16_per_element(dtype_str):
    return {'bfloat16':1,'float16':1,'uint16':1,'float32':2,'float64':4}.get(dtype_str,1)
def encode_uint16_as_rgba_natural(u16,shape,u16_per_elem=1):
    fp16_view=u16.view(np.float16).reshape(-1)
    rgba=encode_fp16_to_rgba4(fp16_view)
    n=rgba.shape[0]
    page_w=int(shape[-1])*u16_per_elem if len(shape)>=2 else min(4096,n)
    h=(n+page_w-1)//page_w
    page=np.zeros((h,page_w,4),dtype=np.uint8)
    page.reshape(-1,4)[:n]=rgba
    return page,n,h,page_w
def decode_rgba_to_uint16(page,n):
    return decode_rgba4_to_fp16(page.reshape(-1,4)[:n]).view(np.uint16)
def sha256_of_array(arr):
    h=hashlib.sha256();h.update(np.ascontiguousarray(arr).tobytes());return h.hexdigest()
def sha256_of_file(path,chunk=1<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b:break
            h.update(b)
    return h.hexdigest()
def sanitize_key(k):return k.replace('.','_').replace('/','_')
def bake(src,out,model_name):
    src=Path(src);out=Path(out)
    src_files=[src] if src.is_file() else sorted(src.glob('*.safetensors'))
    assert src_files,f'no safetensors found at {src}'
    out.mkdir(parents=True,exist_ok=True);(out/'tensors').mkdir(exist_ok=True)
    t0=time.time()
    src_sha=hashlib.sha256()
    for sf in src_files:
        with open(sf,'rb') as f:
            while True:
                b=f.read(1<<20)
                if not b:break
                src_sha.update(b)
    src_sha_hex=src_sha.hexdigest()
    print(f'[v5.0.3a] bake start src={src} files={len(src_files)} sha256={src_sha_hex[:16]}... model={model_name}')
    manifest={'model_name':model_name,'source_files':[str(p.relative_to(src.parent if src.is_file() else src)) for p in src_files],'source_sha256':src_sha_hex,'bake_version':'v5.0.3','reffelt_scheme':'rgba4','tensors':{}}
    total_params=0;ptex_total=0;fp16_baseline=0
    n_ok=0;n_fail=0
    for sf in src_files:
        with safe_open(str(sf),framework='pt') as f:
            keys=list(f.keys())
            for ki,k in enumerate(keys):
                t=f.get_tensor(k)
                src_dtype=str(t.dtype).replace('torch.','')
                if t.dtype in (torch.float8_e4m3fn,torch.float8_e5m2):src_dtype='bfloat16'
                shape=tuple(int(x) for x in t.shape)
                u16=to_uint16_np(t)
                upe=_u16_per_element(src_dtype)
                if u16.size>50_000_000:
                    fp16_view=u16.view(np.float16).reshape(-1)
                    u16_flat=u16.reshape(-1)
                    chunk=20_000_000
                    n=fp16_view.size
                    MAX_TEX=16384
                    page_w=int(shape[-1])*upe if len(shape)>=2 else min(4096,n)
                    page_h=(n+page_w-1)//page_w
                    if page_h>MAX_TEX:
                        page_w=(n+MAX_TEX-1)//MAX_TEX
                        page_h=(n+page_w-1)//page_w
                    page=np.zeros((page_h,page_w,4),dtype=np.uint8)
                    page_flat=page.reshape(-1,4)
                    ok=True
                    for off in range(0,n,chunk):
                        end=min(off+chunk,n)
                        c_rgba=encode_fp16_to_rgba4(fp16_view[off:end])
                        page_flat[off:end]=c_rgba
                        sd=decode_rgba4_to_fp16(c_rgba).view(np.uint16)
                        if not np.array_equal(u16_flat[off:end],sd):ok=False;del c_rgba,sd;break
                        del c_rgba,sd
                else:
                    page,n,page_h,page_w=encode_uint16_as_rgba_natural(u16,shape,u16_per_elem=upe)
                    MAX_TEX=16384
                    if page_h>MAX_TEX or page_w>MAX_TEX:
                        new_w=(n+MAX_TEX-1)//MAX_TEX if page_h>MAX_TEX else page_w
                        if new_w>MAX_TEX:new_w=MAX_TEX
                        new_h=(n+new_w-1)//new_w
                        new_page=np.zeros((new_h,new_w,4),dtype=np.uint8)
                        new_page.reshape(-1,4)[:n]=page.reshape(-1,4)[:n]
                        page,page_h,page_w=new_page,new_h,new_w
                    u16_dec=decode_rgba_to_uint16(page,n).reshape(u16.shape)
                    ok=np.array_equal(u16,u16_dec)
                if not ok:
                    n_fail+=1
                    print(f'  [FAIL] {k} shape={shape} src_dtype={src_dtype}')
                    continue
                n_ok+=1
                fname=sanitize_key(k)+'.ptex'
                fpath=out/'tensors'/fname
                src_check=sha256_of_array(u16)
                with open(fpath,'wb') as fout:fout.write(np.ascontiguousarray(page).tobytes())
                ptex_bytes=fpath.stat().st_size
                ptex_total+=ptex_bytes
                params=int(np.prod(shape));total_params+=params;fp16_baseline+=params*2
                manifest['tensors'][k]={'shape':list(shape),'source_dtype':src_dtype,'ptex_path':f'tensors/{fname}','ptex_bytes':int(ptex_bytes),'src_sha256':src_check,'params':params,'page_h':int(page_h),'page_w':int(page_w),'n_pixels':int(n),'u16_per_elem':int(upe)}
                if (ki+1)%20==0 or ki+1==len(keys):print(f'  [{ki+1}/{len(keys)}] {k:60s} shape={str(shape):20s} page=({page_h}x{page_w}) ptex={ptex_bytes/1024:.1f}KB ok={ok}')
    manifest['tensor_count']=n_ok
    manifest['total_params']=total_params
    manifest['fp16_baseline_bytes']=fp16_baseline
    manifest['ptex_total_bytes']=ptex_total
    manifest['compression_ratio']=fp16_baseline/ptex_total if ptex_total>0 else 0.0
    manifest['bake_seconds']=time.time()-t0
    tmp=out/'manifest.json.tmp'
    with open(tmp,'w') as f:json.dump(manifest,f,indent=2)
    os.replace(tmp,out/'manifest.json')
    print(f'\n[v5.0.3a] DONE ok={n_ok} fail={n_fail} params={total_params:,} fp16={fp16_baseline/1024/1024:.1f}MB ptex={ptex_total/1024/1024:.1f}MB ratio={manifest["compression_ratio"]:.3f}x time={manifest["bake_seconds"]:.1f}s')
    return 0 if n_fail==0 else 1
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--src',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--model-name',required=True)
    args=ap.parse_args()
    sys.exit(bake(args.src,args.out,args.model_name))
if __name__=='__main__':main()
