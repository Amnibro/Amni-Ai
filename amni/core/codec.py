import numpy as np,scipy.sparse as sp,json as _json
from typing import Tuple,Optional
def encode_f32(v:float)->Tuple[int,int,int,int]:
    b=np.float32(v).tobytes()
    return b[0],b[1],b[2],b[3]
def decode_f32(r:int,g:int,b:int,a:int)->float:
    return np.frombuffer(bytes([r,g,b,a]),dtype=np.float32)[0]
def weights_to_rgba(w:np.ndarray,wd:int,ht:int)->np.ndarray:
    f=w.astype(np.float32).flatten()
    return np.pad(f,(0,max(0,wd*ht-f.shape[0])))[:wd*ht].view(np.uint8).reshape(ht,wd,4)
def rgba_to_weights(r:np.ndarray)->np.ndarray:
    return r.reshape(-1,4).view(np.float32).flatten()
def optimal_dimensions(n:int,m:int=4096)->Tuple[int,int]:
    w=min(m,int(np.ceil(np.sqrt(n))))
    return w,int(np.ceil(n/w))
def partition_weights(w:np.ndarray,p:int=65536)->list:
    f=w.astype(np.float32).flatten()
    return [f[i:i+p] for i in range(0,len(f),p)]
def merge_pages(p:list)->np.ndarray:
    return np.concatenate([x.astype(np.float32) for x in p])
def quantize_f32(w:np.ndarray,s:Optional[float]=None,b:Optional[float]=None)->Tuple[np.ndarray,float,float]:
    w=w.astype(np.float32)
    mn,mx=w.min(),w.max()
    s=(mx-mn)/255.0 if s is None else s
    s=1.0 if s==0 else s
    b=mn if b is None else b
    return np.clip(np.round((w-b)/s),0,255).astype(np.uint8),s,b
def dequantize_u8(q:np.ndarray,s:float,b:float)->np.ndarray:
    return q.astype(np.float32)*s+b
def weights_to_u8_image(w:np.ndarray,wd:int,ht:int,s:Optional[float]=None,b:Optional[float]=None)->Tuple[np.ndarray,float,float]:
    q,s,b=quantize_f32(w,s,b)
    f=q.flatten()
    return np.pad(f,(0,max(0,wd*ht-f.shape[0])))[:wd*ht].reshape(ht,wd),s,b
def u8_image_to_weights(i:np.ndarray,s:float,b:float)->np.ndarray:
    return dequantize_u8(i,s,b).flatten()
def generate_nonce()->int:
    return np.random.randint(0,10000000)
def nonce_distance(n1:int,n2:int)->int:
    return abs(n1-n2)
def dense_to_csr(d:np.ndarray,t:float=1e-5)->Tuple[np.ndarray,np.ndarray,np.ndarray,tuple]:
    c=sp.csr_matrix(np.where(np.abs(d)<t,0,d).astype(np.float32))
    return c.indptr.astype(np.int32),c.indices.astype(np.int32),c.data.astype(np.float16),d.shape
def csr_to_dense(p:np.ndarray,i:np.ndarray,v:np.ndarray,s:tuple)->np.ndarray:
    return sp.csr_matrix((v.astype(np.float32),i,p),shape=s).toarray().astype(np.float16)
def save_csr(f:str,p:np.ndarray,i:np.ndarray,v:np.ndarray,s:tuple):
    with open(f,'wb') as o:
        np.array([s[0],s[1],len(v)],dtype=np.int32).tofile(o)
        p.tofile(o)
        i.tofile(o)
        v.tofile(o)
def load_csr(f:str)->Tuple[np.ndarray,np.ndarray,np.ndarray,tuple]:
    with open(f,'rb') as i:
        h=np.fromfile(i,dtype=np.int32,count=3)
        return np.fromfile(i,dtype=np.int32,count=h[0]+1),np.fromfile(i,dtype=np.int32,count=h[2]),np.fromfile(i,dtype=np.float16,count=h[2]),(h[0],h[1])
def dense_to_bsr(d:np.ndarray,bs:int=64,t:float=1e-5)->Tuple[np.ndarray,np.ndarray,np.ndarray,tuple,int]:
    r,c=d.shape
    rb,cb=(r+bs-1)//bs,(c+bs-1)//bs
    pad_d=np.zeros((rb*bs,cb*bs),dtype=np.float32)
    pad_d[:r,:c]=d.astype(np.float32)
    in_idx,out_idx,blocks=[],[],[]
    for oi in range(rb):
        for ii in range(cb):
            blk=pad_d[oi*bs:(oi+1)*bs,ii*bs:(ii+1)*bs]
            if np.max(np.abs(blk))>t:
                out_idx.append(oi)
                in_idx.append(ii)
                blocks.append(blk.astype(np.float16))
    return np.array(in_idx,dtype=np.int32),np.array(out_idx,dtype=np.int32),np.stack(blocks).astype(np.float16) if blocks else np.zeros((0,bs,bs),dtype=np.float16),d.shape,bs
def bsr_to_dense(in_idx:np.ndarray,out_idx:np.ndarray,blocks:np.ndarray,shape:tuple,bs:int)->np.ndarray:
    r,c=shape
    d=np.zeros((((r+bs-1)//bs)*bs,((c+bs-1)//bs)*bs),dtype=np.float32)
    for i in range(len(in_idx)):
        d[out_idx[i]*bs:(out_idx[i]+1)*bs,in_idx[i]*bs:(in_idx[i]+1)*bs]=blocks[i].astype(np.float32)
    return d[:r,:c].astype(np.float16)
def save_bsr(f:str,in_idx:np.ndarray,out_idx:np.ndarray,blocks:np.ndarray,shape:tuple,bs:int):
    with open(f,'wb') as o:
        np.array([shape[0],shape[1],len(in_idx),bs],dtype=np.int32).tofile(o)
        in_idx.tofile(o)
        out_idx.tofile(o)
        blocks.tofile(o)
def load_bsr(f:str)->Tuple[np.ndarray,np.ndarray,np.ndarray,tuple,int]:
    with open(f,'rb') as i:
        h=np.fromfile(i,dtype=np.int32,count=4)
        n=h[2]
        bs=h[3]
        return np.fromfile(i,dtype=np.int32,count=n),np.fromfile(i,dtype=np.int32,count=n),np.fromfile(i,dtype=np.float16,count=n*bs*bs).reshape(n,bs,bs),(h[0],h[1]),bs
def dense_to_lut(d:np.ndarray,t:float=1e-5)->Tuple[np.ndarray,np.ndarray,np.ndarray,float,tuple]:
    d=d.astype(np.float32)
    r,c=d.shape
    w_max=np.abs(d).max()
    w_scale=w_max/7.0 if w_max>0 else 1.0
    mask=np.abs(d)>t
    rows,cols=np.where(mask)
    vals=d[mask]
    wq=np.clip(np.round(vals/w_scale),-8,7).astype(np.int32)+8
    counts=np.bincount(rows,minlength=r)
    rp=np.zeros(r+1,dtype=np.int32)
    rp[1:]=np.cumsum(counts)
    return rp,cols.astype(np.int32),wq.astype(np.uint8),w_scale,d.shape
def save_lut(f:str,rp:np.ndarray,ci:np.ndarray,wq:np.ndarray,w_scale:float,shape:tuple):
    with open(f,'wb') as o:
        np.array([shape[0],shape[1],len(ci)],dtype=np.int32).tofile(o)
        np.array([w_scale],dtype=np.float32).tofile(o)
        rp.tofile(o)
        ci.tofile(o)
        wq.tofile(o)
def load_lut(f:str)->Tuple[np.ndarray,np.ndarray,np.ndarray,float,tuple]:
    with open(f,'rb') as i:
        h=np.fromfile(i,dtype=np.int32,count=3)
        r,c,nnz=int(h[0]),int(h[1]),int(h[2])
        w_scale=float(np.fromfile(i,dtype=np.float32,count=1)[0])
        rp=np.fromfile(i,dtype=np.int32,count=r+1)
        ci=np.fromfile(i,dtype=np.int32,count=nnz)
        wq=np.fromfile(i,dtype=np.uint8,count=nnz)
        return rp,ci,wq,w_scale,(r,c)
def save_reffelt_nonces(path:str,nonces:np.ndarray):
    with open(path,'wb') as o:
        np.array([nonces.shape[0],nonces.shape[1]],dtype=np.int32).tofile(o)
        nonces.astype(np.float16).tofile(o)
def load_reffelt_nonces(path:str)->np.ndarray:
    with open(path,'rb') as i:
        h=np.fromfile(i,dtype=np.int32,count=2)
        return np.fromfile(i,dtype=np.float16,count=h[0]*h[1]).reshape(h[0],h[1])
def save_4d_atlas(path:str,nonce_ptr:np.ndarray,bucket_idx:np.ndarray,head_idx:np.ndarray,vals:np.ndarray,shape:tuple,val_scale:float=1.0):
    with open(path,'wb') as o:
        np.array(list(shape),dtype=np.int32).tofile(o)
        np.array([len(vals)],dtype=np.int32).tofile(o)
        np.array([val_scale],dtype=np.float32).tofile(o)
        nonce_ptr.astype(np.int32).tofile(o)
        bucket_idx.astype(np.int32).tofile(o)
        head_idx.astype(np.int16).tofile(o)
        vals.astype(np.float16).tofile(o)
def load_4d_atlas(path:str)->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,tuple,float]:
    with open(path,'rb') as i:
        sh=np.fromfile(i,dtype=np.int32,count=4)
        nnz=int(np.fromfile(i,dtype=np.int32,count=1)[0])
        vs=float(np.fromfile(i,dtype=np.float32,count=1)[0])
        np_=np.fromfile(i,dtype=np.int32,count=sh[0]+1)
        bi=np.fromfile(i,dtype=np.int32,count=nnz)
        hi=np.fromfile(i,dtype=np.int16,count=nnz)
        vl=np.fromfile(i,dtype=np.float16,count=nnz*sh[3])
        return np_,bi,hi,vl.reshape(nnz,sh[3]),(int(sh[0]),int(sh[1]),int(sh[2]),int(sh[3])),vs
def dense_to_4d_atlas(w4d:np.ndarray,sparsity:float=0.95)->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,tuple,float]:
    N,B,H,D=w4d.shape
    norms=np.linalg.norm(w4d.reshape(N*B*H,D),axis=1)
    thresh=np.percentile(norms,sparsity*100) if sparsity>0 else 0
    nonce_ptr=np.zeros(N+1,dtype=np.int32)
    b_list,h_list,v_list=[],[],[]
    for n in range(N):
        for b in range(B):
            for h in range(H):
                nm=np.linalg.norm(w4d[n,b,h])
                if nm>thresh:
                    b_list.append(b)
                    h_list.append(h)
                    v_list.append(w4d[n,b,h].astype(np.float16))
        nonce_ptr[n+1]=len(b_list)
    bi=np.array(b_list,dtype=np.int32) if b_list else np.zeros(0,dtype=np.int32)
    hi=np.array(h_list,dtype=np.int16) if h_list else np.zeros(0,dtype=np.int16)
    vl=np.stack(v_list).astype(np.float16) if v_list else np.zeros((0,D),dtype=np.float16)
    return nonce_ptr,bi,hi,vl,(N,B,H,D),1.0
def save_truth_atlas(path:str,nonce_ptr:np.ndarray,term_idx:np.ndarray,rel_idx:np.ndarray,vals:np.ndarray,shape:tuple,val_scale:float=1.0):
    with open(path,'wb') as o:
        np.array([0x54525448],dtype=np.uint32).tofile(o)
        np.array(list(shape),dtype=np.int32).tofile(o)
        np.array([len(term_idx)],dtype=np.int32).tofile(o)
        np.array([val_scale],dtype=np.float32).tofile(o)
        nonce_ptr.astype(np.int32).tofile(o)
        term_idx.astype(np.int32).tofile(o)
        rel_idx.astype(np.uint8).tofile(o)
        vals.astype(np.float16).tofile(o)
def load_truth_atlas(path:str)->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,tuple,float]:
    with open(path,'rb') as i:
        mg=np.fromfile(i,dtype=np.uint32,count=1)[0]
        assert mg==0x54525448,f"bad truth atlas magic: {mg:#x}"
        sh=np.fromfile(i,dtype=np.int32,count=4)
        nnz=int(np.fromfile(i,dtype=np.int32,count=1)[0])
        vs=float(np.fromfile(i,dtype=np.float32,count=1)[0])
        np_=np.fromfile(i,dtype=np.int32,count=sh[0]+1)
        ti=np.fromfile(i,dtype=np.int32,count=nnz)
        ri=np.fromfile(i,dtype=np.uint8,count=nnz)
        vl=np.fromfile(i,dtype=np.float16,count=nnz*sh[3])
        return np_,ti,ri,vl.reshape(nnz,sh[3]),(int(sh[0]),int(sh[1]),int(sh[2]),int(sh[3])),vs
def save_term_ontology(path:str,term_names:list,term_vecs:np.ndarray,rel_names:list):
    import json
    d={"n_terms":len(term_names),"n_rels":len(rel_names),"dim":int(term_vecs.shape[1]),"terms":term_names,"relations":rel_names}
    with open(path,'w') as f: json.dump(d,f)
    vp=path.replace('.json','.vecs.bin')
    term_vecs.astype(np.float16).tofile(vp)
def load_term_ontology(path:str)->Tuple[list,np.ndarray,list]:
    import json
    with open(path) as f: d=json.load(f)
    vp=path.replace('.json','.vecs.bin')
    vecs=np.fromfile(vp,dtype=np.float16).reshape(d["n_terms"],d["dim"])
    return d["terms"],vecs,d["relations"]
def save_delta_weights(path:str,weights:dict):
    with open(path,'wb') as o:
        np.array([len(weights)],dtype=np.int32).tofile(o)
        for nid in sorted(weights.keys()):
            w,b=weights[nid]
            np.array([nid,w.shape[0],w.shape[1]],dtype=np.int32).tofile(o)
            w.astype(np.float16).tofile(o)
            b.astype(np.float16).tofile(o)
def load_delta_weights(path:str)->dict:
    weights={}
    with open(path,'rb') as i:
        n=int(np.fromfile(i,dtype=np.int32,count=1)[0])
        for _ in range(n):
            h=np.fromfile(i,dtype=np.int32,count=3)
            nid,rows,cols=int(h[0]),int(h[1]),int(h[2])
            w=np.fromfile(i,dtype=np.float16,count=rows*cols).reshape(rows,cols)
            b=np.fromfile(i,dtype=np.float16,count=rows)
            weights[nid]=(w,b)
    return weights
def dense_to_truth_atlas(w4d:np.ndarray,sparsity:float=0.95)->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,tuple,float]:
    N,T,R,D=w4d.shape
    norms=np.linalg.norm(w4d.reshape(N*T*R,D),axis=1)
    thresh=np.percentile(norms,sparsity*100) if sparsity>0 else 0
    nonce_ptr=np.zeros(N+1,dtype=np.int32)
    t_list,r_list,v_list=[],[],[]
    for n in range(N):
        for t in range(T):
            for r in range(R):
                nm=np.linalg.norm(w4d[n,t,r])
                if nm>thresh:
                    t_list.append(t)
                    r_list.append(r)
                    v_list.append(w4d[n,t,r].astype(np.float16))
        nonce_ptr[n+1]=len(t_list)
    ti=np.array(t_list,dtype=np.int32) if t_list else np.zeros(0,dtype=np.int32)
    ri=np.array(r_list,dtype=np.uint8) if r_list else np.zeros(0,dtype=np.uint8)
    vl=np.stack(v_list).astype(np.float16) if v_list else np.zeros((0,D),dtype=np.float16)
    return nonce_ptr,ti,ri,vl,(N,T,R,D),1.0
def save_expert_registry(path:str,meta:dict,affinity:Optional[np.ndarray]=None):
    with open(path,'w') as f: _json.dump(meta,f)
    if affinity is not None:
        ap=path.replace('.json','.affinity.bin')
        with open(ap,'wb') as o:
            np.array(affinity.shape,dtype=np.int32).tofile(o)
            affinity.astype(np.float32).tofile(o)
def load_expert_registry(path:str)->Tuple[dict,Optional[np.ndarray]]:
    with open(path) as f: meta=_json.load(f)
    ap=path.replace('.json','.affinity.bin')
    mat=None
    import os
    if os.path.exists(ap):
        with open(ap,'rb') as i:
            sh=np.fromfile(i,dtype=np.int32,count=2)
            mat=np.fromfile(i,dtype=np.float32,count=int(sh[0])*int(sh[1])).reshape(int(sh[0]),int(sh[1]))
    return meta,mat
