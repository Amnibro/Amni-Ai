import argparse,json,sys,time,hashlib,numpy as np
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_ROOT))
from amni.compute.reffelt4 import REFFELT_K4
def _sha256(arr):
    h=hashlib.sha256();h.update(np.ascontiguousarray(arr).tobytes());return h.hexdigest()
def _load_ptex(path,n_pixels):
    raw=np.fromfile(path,dtype=np.uint8)
    return raw.reshape(-1,4)[:n_pixels]
def _split_to_digits(rgba):
    return rgba[:,0].copy(),rgba[:,1].copy(),rgba[:,2].copy(),rgba[:,3].copy()
def _reconstruct(d0,d1,d2,d3):
    a=d0.astype(np.uint32)+d1.astype(np.uint32)*17+d2.astype(np.uint32)*289+d3.astype(np.uint32)*4913
    return a.astype(np.uint16)
def _validate_digits(d0,d1,d2,d3):
    for i,d in enumerate((d0,d1,d2,d3)):
        if d.max(initial=0)>16:raise ValueError(f'digit plane d{i} contains value > 16: invalid GF(17) element')
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--src-bake',required=True)
    ap.add_argument('--out-bake',required=True)
    args=ap.parse_args()
    src=Path(args.src_bake);out=Path(args.out_bake)
    assert (src/'manifest.json').exists(),f'no manifest at {src}'
    out.mkdir(parents=True,exist_ok=True);(out/'tensors').mkdir(exist_ok=True)
    src_man=json.loads((src/'manifest.json').read_text())
    new_man={'model_name':src_man.get('model_name','?'),'source_files':src_man.get('source_files',[]),'source_sha256':src_man.get('source_sha256',''),'bake_version':'v5.0.3-gf17','reffelt_scheme':'gf17_digit_planes','reffelt_k4':list(REFFELT_K4),'derived_from':str(src.name),'tensors':{}}
    t0=time.time()
    n_ok=0;n_fail=0
    total_params=0;gf17_total=0
    keys=list(src_man['tensors'].keys())
    for ki,k in enumerate(keys):
        info=src_man['tensors'][k]
        ptex_path=src/info['ptex_path']
        n_pixels=int(info['n_pixels'])
        rgba=_load_ptex(ptex_path,n_pixels)
        d0,d1,d2,d3=_split_to_digits(rgba)
        _validate_digits(d0,d1,d2,d3)
        u16_recon=_reconstruct(d0,d1,d2,d3)
        src_sha=info.get('src_sha256','')
        u16_recon_sha=_sha256(u16_recon)
        ok=(u16_recon_sha==src_sha) if src_sha else True
        if not ok:
            n_fail+=1
            print(f'  [FAIL] {k} sha mismatch src={src_sha[:16]} recon={u16_recon_sha[:16]}',flush=True)
            continue
        n_ok+=1
        sk=k.replace('.','_').replace('/','_')
        gf17_path=out/'tensors'/f'{sk}.gf17'
        planes=np.stack([d0,d1,d2,d3],axis=0)
        with open(gf17_path,'wb') as f:f.write(planes.tobytes())
        gf17_bytes=gf17_path.stat().st_size
        gf17_total+=gf17_bytes
        params=int(info.get('params',0));total_params+=params
        new_man['tensors'][k]={'shape':info['shape'],'source_dtype':info.get('source_dtype','float16'),'gf17_path':f'tensors/{sk}.gf17','gf17_bytes':int(gf17_bytes),'src_sha256':src_sha,'params':params,'n_pixels':n_pixels,'n_digits_per_plane':n_pixels,'planes':4,'plane_offsets':{'d0':0,'d1':int(n_pixels),'d2':int(2*n_pixels),'d3':int(3*n_pixels)}}
        if (ki+1)%20==0 or ki+1==len(keys):print(f'  [{ki+1}/{len(keys)}] {k:60s} planes=4 gf17={gf17_bytes/1024:.1f}KB ok=True',flush=True)
    new_man['tensor_count']=n_ok
    new_man['total_params']=total_params
    new_man['gf17_total_bytes']=gf17_total
    new_man['fp16_baseline_bytes']=src_man.get('fp16_baseline_bytes',total_params*2)
    new_man['compression_ratio']=new_man['fp16_baseline_bytes']/gf17_total if gf17_total>0 else 0.0
    new_man['convert_seconds']=time.time()-t0
    tmp=out/'manifest.json.tmp'
    with open(tmp,'w') as f:json.dump(new_man,f,indent=2)
    tmp.replace(out/'manifest.json')
    print(f'\n[ptex->gf17] DONE ok={n_ok} fail={n_fail} params={total_params:,} fp16={new_man["fp16_baseline_bytes"]/1024/1024:.1f}MB gf17={gf17_total/1024/1024:.1f}MB ratio={new_man["compression_ratio"]:.3f}x time={new_man["convert_seconds"]:.1f}s',flush=True)
    return 0 if n_fail==0 else 1
if __name__=='__main__':sys.exit(main())
