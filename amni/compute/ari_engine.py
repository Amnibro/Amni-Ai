import ctypes as ct,numpy as np,os,sys
from pathlib import Path
from amni.compute.ternary5 import pack_ternary5_page
P=17
_dir=Path(__file__).resolve().parent
_so=(_dir/'hip'/'libari_hip.dll') if sys.platform=='win32' else (_dir/'hip'/'libari_hip.so')
_lib=None
_ok=False
ATEX_MAGIC=0x41544558
ATEX_VER=1
ATEX_F_RGBA16=1
ATEX_F_TERNARY=2
def _load():
    global _lib,_ok
    if _ok:return True
    if not _so.exists():return False
    try:
        rocm=os.environ.get('ROCM_HOME',r'C:\Program Files\AMD\ROCm\7.1' if sys.platform=='win32' else '/opt/rocm')
        hip_rt=(Path(rocm)/'bin'/'amdhip64_7.dll') if sys.platform=='win32' else (Path(rocm)/'lib'/'libamdhip64.so')
        if hip_rt.exists():ct.CDLL(str(hip_rt),mode=ct.RTLD_GLOBAL)
        _lib=ct.CDLL(str(_so),mode=ct.RTLD_GLOBAL if sys.platform!='win32' else 0)
        for nm,res,args in[
            ('ari_init',ct.c_int,[ct.c_int]),
            ('ari_shutdown',None,[]),
            ('ari_bind_texture',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_int]),
            ('ari_bind_texture_u16',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_int]),
            ('ari_free_texture',None,[ct.c_int]),
            ('ari_tex_update_rect',ct.c_int,[ct.c_int,ct.c_int,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_upload_codebook',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_int]),
            ('ari_alloc',ct.c_void_p,[ct.c_size_t]),
            ('ari_free',None,[ct.c_void_p]),
            ('ari_h2d',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_size_t]),
            ('ari_d2h',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_size_t]),
            ('ari_d2d',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_size_t]),
            ('ari_sync',ct.c_int,[]),
            ('ari_tex_matmul_t',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_tex_matmul_vq',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_tex_matmul_fp16',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_rg_gemv_fused',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int]),
            ('ari_rg_gemv_fused_offset',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_rgba_gemv_fused',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int]),
            ('ari_gemv_rgba_fp16',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int]),
            ('ari_gemv_rgba_bf16',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int]),
            ('ari_tex_matmul_bf16',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_tex_matmul_fp8',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_gemv_rgba16_fp16',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int]),
            ('ari_gemv_ternary_fp16',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int]),
            ('ari_gemv_rtier_fp16',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_gemv_rgba16_fp16_tiled',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_gemv_rgba16_fp16_batched',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_fetch_rgba16_indices',ct.c_int,[ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int]),
            ('ari_dispatch_fused_rgba16',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_tex_embed',ct.c_int,[ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int]),
            ('ari_vq_decode_tex',ct.c_int,[ct.c_int,ct.c_void_p,ct.c_int]),
            ('ari_elem_add',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int]),
            ('ari_elem_mul',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int]),
            ('ari_activate',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_int]),
            ('ari_rms_norm',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int]),
            ('ari_neg_score',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_attn_norm',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_apply_v',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_xpose_bshd',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_xpose_bhsd',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_repeat_kv',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('ari_get_tex_w',ct.c_int,[ct.c_int]),
            ('ari_get_tex_h',ct.c_int,[ct.c_int]),
        ]:
            fn=getattr(_lib,nm)
            if res is not None:fn.restype=res
            fn.argtypes=args
        rc=_lib.ari_init(0)
        _ok=(rc==0)
    except Exception as e:
        print(f"[ari_engine] load failed: {e}",file=sys.stderr)
        _ok=False
    return _ok
class DevBuf:
    __slots__=('ptr','shape','size','dtype')
    def __init__(self,ptr,shape,dtype=np.uint8):
        self.ptr=ct.c_void_p(ptr) if not isinstance(ptr,ct.c_void_p) else ptr
        self.shape=tuple(shape)
        self.size=int(np.prod(shape))
        self.dtype=np.dtype(dtype)
    @property
    def nbytes(self):return self.size*self.dtype.itemsize
    def reshape(self,*sh):
        n=int(np.prod(sh))
        assert n==self.size,f"reshape {self.shape}->{sh}: {self.size}!={n}"
        return DevBuf(self.ptr,sh,self.dtype)
    def to_host(self):
        h=np.empty(self.shape,dtype=self.dtype)
        _lib.ari_d2h(h.ctypes.data_as(ct.c_void_p),self.ptr,self.nbytes)
        return h
    @staticmethod
    def from_host(arr,dtype=None):
        dt=np.dtype(dtype) if dtype is not None else np.asarray(arr).dtype
        a=np.ascontiguousarray(arr,dtype=dt)
        ptr=_lib.ari_alloc(a.nbytes)
        _lib.ari_h2d(ct.c_void_p(ptr),a.ctypes.data_as(ct.c_void_p),a.nbytes)
        return DevBuf(ptr,a.shape,dt)
class TexBuf:
    __slots__=('idx','w','h','n_weights','vq','flags')
    def __init__(self,idx,w,h,n_weights=0,vq=False,flags=0):
        self.idx=idx;self.w=w;self.h=h;self.n_weights=n_weights;self.vq=vq;self.flags=flags
class RgBundle:
    __slots__=('r_tb','g_tb','b_tb','r_lut_ptr','g_lut_ptr','b_lut_ptr','routing_k','N','K')
    def __init__(self,r_tb,g_tb,b_tb,r_lut_ptr,g_lut_ptr,b_lut_ptr,routing_k,N,K):
        self.r_tb=r_tb;self.g_tb=g_tb;self.b_tb=b_tb
        self.r_lut_ptr=r_lut_ptr;self.g_lut_ptr=g_lut_ptr;self.b_lut_ptr=b_lut_ptr
        self.routing_k=routing_k;self.N=N;self.K=K
def weight_to_rgba_page(arr,page_w=4096):
    flat=np.ascontiguousarray(arr.ravel(),dtype=np.uint8)
    n=flat.size
    px_needed=(n+3)//4
    h_needed=(px_needed+page_w-1)//page_w
    expected=h_needed*page_w*4
    if n==expected:return flat.reshape(h_needed,page_w,4),n
    page=np.zeros((h_needed,page_w,4),dtype=np.uint8)
    page.reshape(-1)[:n]=flat
    return page,n
def pack_ternary_atex_page(arr,page_w=4096):
    page,n,_=pack_ternary5_page(arr,page_w)
    return page,n
def save_atex(path,page_data,n_weights,meta=None,flags=None):
    import struct,json
    p=Path(path)
    data=np.ascontiguousarray(page_data)
    fl=int(ATEX_F_RGBA16 if flags is None and data.dtype==np.uint16 else (0 if flags is None else flags))
    with open(str(p),'wb') as f:
        f.write(struct.pack('<I',ATEX_MAGIC))
        f.write(struct.pack('<H',ATEX_VER))
        f.write(struct.pack('<H',fl))
        h,w=data.shape[:2]
        f.write(struct.pack('<II',w,h))
        f.write(struct.pack('<I',n_weights))
        data.tofile(f)
    if meta:
        mp=p.with_suffix('.meta.json')
        with open(str(mp),'w') as f:json.dump(meta,f)
def load_atex_info(path):
    import struct
    with open(str(path),'rb') as f:
        mg=struct.unpack('<I',f.read(4))[0]
        assert mg==ATEX_MAGIC,f"bad atex magic: {mg:#x}"
        ver=struct.unpack('<H',f.read(2))[0]
        flags=struct.unpack('<H',f.read(2))[0]
        w,h=struct.unpack('<II',f.read(8))
        n_weights=struct.unpack('<I',f.read(4))[0]
        dtype=np.uint16 if flags&ATEX_F_RGBA16 else np.uint8
        data=np.fromfile(f,dtype=dtype,count=w*h*4).reshape(h,w,4)
    return data,w,h,n_weights,flags
def load_atex(path):
    data,w,h,n_weights,_=load_atex_info(path)
    return data,w,h,n_weights
class ARIEngine:
    def __init__(self):
        if not _load():raise RuntimeError("ARI HIP engine not available")
        self._bufs=set()
        self._texs=[]
        self._cb_loaded=False
    def bind_texture(self,rgba_page,n_weights=0):
        h,w=rgba_page.shape[:2]
        data=np.ascontiguousarray(rgba_page,dtype=np.uint8)
        idx=_lib.ari_bind_texture(data.ctypes.data_as(ct.c_void_p),w,h)
        if idx<0:raise RuntimeError(f"ari_bind_texture failed: w={w} h={h}")
        tb=TexBuf(idx,w,h,int(n_weights or h*w*4),False,0)
        self._texs.append(tb)
        return tb
    def bind_texture16(self,rgba16_page,n_weights=0):
        h,w=rgba16_page.shape[:2]
        data=np.ascontiguousarray(rgba16_page,dtype=np.uint16)
        idx=_lib.ari_bind_texture_u16(data.ctypes.data_as(ct.c_void_p),w,h)
        if idx<0:raise RuntimeError(f"ari_bind_texture_u16 failed: w={w} h={h}")
        tb=TexBuf(idx,w,h,int(n_weights or h*w*4),False,ATEX_F_RGBA16)
        self._texs.append(tb)
        return tb
    def fetch_texture16_indices(self,tb,indices):
        if not tb.flags&ATEX_F_RGBA16:raise NotImplementedError('fetch_texture16_indices is only wired for RGBA16 textures')
        idx=np.ascontiguousarray(indices,dtype=np.int32).reshape(-1)
        ib=self.upload(idx);out=self.alloc((idx.size,4),dtype=np.uint16)
        try:
            rc=_lib.ari_fetch_rgba16_indices(tb.idx,ib.ptr,out.ptr,int(idx.size))
            if rc!=0:raise RuntimeError(f'ari_fetch_rgba16_indices failed: rc={rc} tex={tb.idx} n={idx.size}')
            return self.download(out)
        finally:
            self.free(ib);self.free(out)
    def dispatch_fused_rgba16(self,x_fp16_dev_ptr,y_fp32_dev_ptr,num_tokens,hidden,token_ids_dev,weights_dev,gu_tex_host,dn_tex_host,gu_offsets_dev,dn_offsets_dev,N_gu,K_gu,N_dn,K_dn):
        gu=np.ascontiguousarray(gu_tex_host,dtype=np.int32).reshape(-1)
        dn=np.ascontiguousarray(dn_tex_host,dtype=np.int32).reshape(-1)
        nr=int(gu.size)
        if dn.size!=nr:raise ValueError('tex idx size mismatch')
        rc=_lib.ari_dispatch_fused_rgba16(ct.c_void_p(int(x_fp16_dev_ptr)),ct.c_void_p(int(y_fp32_dev_ptr)),int(num_tokens),int(hidden),nr,ct.c_void_p(int(token_ids_dev)),ct.c_void_p(int(weights_dev)),gu.ctypes.data_as(ct.c_void_p),dn.ctypes.data_as(ct.c_void_p),ct.c_void_p(int(gu_offsets_dev)),ct.c_void_p(int(dn_offsets_dev)),int(N_gu),int(K_gu),int(N_dn),int(K_dn))
        if rc!=0:raise RuntimeError(f'ari_dispatch_fused_rgba16 failed: rc={rc} routes={nr}')
        return rc
    def gemv_exact_fp16_batched(self,x_dev_ptr,tb,y_dev_ptr,M,N,K):
        if not tb.flags&ATEX_F_RGBA16:raise NotImplementedError('batched exact GEMV is RGBA16-only')
        rc=_lib.ari_gemv_rgba16_fp16_batched(ct.c_void_p(int(x_dev_ptr)),tb.idx,ct.c_void_p(int(y_dev_ptr)),int(M),int(N),int(K))
        if rc!=0:raise RuntimeError(f'ari_gemv_rgba16_fp16_batched failed: rc={rc} M={M} N={N} K={K}')
        return rc
    def bind_atex(self,path):
        data,_,_,n_weights,flags=load_atex_info(path)
        if flags&ATEX_F_RGBA16:return self.bind_texture16(data,n_weights)
        tb=self.bind_texture(data,n_weights);tb.flags=flags;return tb
    def bind_vq_texture(self,idx_page,n_blocks):
        h,w=idx_page.shape[:2]
        data=np.ascontiguousarray(idx_page,dtype=np.uint8)
        idx=_lib.ari_bind_texture(data.ctypes.data_as(ct.c_void_p),w,h)
        if idx<0:raise RuntimeError("ari_bind_texture(vq) failed")
        tb=TexBuf(idx,w,h,n_blocks,vq=True)
        self._texs.append(tb)
        return tb
    def upload_codebook(self,cb):
        cb=np.ascontiguousarray(cb,dtype=np.uint8)
        n_entries,dim=cb.shape
        _lib.ari_upload_codebook(cb.ctypes.data_as(ct.c_void_p),n_entries,dim)
        self._cb_loaded=True
    def free_texture(self,tb):
        _lib.ari_free_texture(tb.idx)
        self._texs=[t for t in self._texs if t.idx!=tb.idx]
    def upload(self,arr):
        b=DevBuf.from_host(arr);self._bufs.add(b.ptr.value);return b
    def download(self,buf):
        _lib.ari_sync();return buf.to_host()
    def alloc(self,shape,dtype=np.uint8):
        dt=np.dtype(dtype);n=int(np.prod(shape))*dt.itemsize;ptr=_lib.ari_alloc(n)
        self._bufs.add(ptr);return DevBuf(ptr,shape,dt)
    def free(self,buf):
        if buf.ptr and buf.ptr.value in self._bufs:
            _lib.ari_free(buf.ptr);self._bufs.discard(buf.ptr.value);buf.ptr=None
    def sync(self):_lib.ari_sync()
    def tex_matmul_t(self,a,tb,M,K,N):
        out=self.alloc((M,N))
        fn=_lib.ari_tex_matmul_vq if tb.vq else _lib.ari_tex_matmul_t
        fn(a.ptr,tb.idx,out.ptr,M,K,N)
        return out
    def tex_matmul_fp16(self,a,tb,M,K,N):
        if tb.flags&ATEX_F_RGBA16:raise NotImplementedError('tex_matmul_fp16 is only wired for RGBA8 exact pages; use gemv_exact_fp16 for exact64 today')
        out=self.alloc((M,N),dtype=np.uint16)
        _lib.ari_tex_matmul_fp16(a.ptr,tb.idx,out.ptr,M,K,N)
        return out
    def gemv_exact_fp16(self,x,tb,N,K):
        out=self.alloc((N,),dtype=np.uint16)
        (_lib.ari_gemv_ternary_fp16 if tb.flags&ATEX_F_TERNARY else (_lib.ari_gemv_rgba16_fp16 if tb.flags&ATEX_F_RGBA16 else _lib.ari_gemv_rgba_fp16))(x.ptr,tb.idx,out.ptr,N,K)
        return out
    def gemv_exact_fp16_tiled(self,x,tb,N,K,tile_h,tile_w):
        if not tb.flags&ATEX_F_RGBA16:raise NotImplementedError('tiled exact GEMV is only wired for exact64 pages')
        out=self.alloc((N,),dtype=np.uint16)
        _lib.ari_gemv_rgba16_fp16_tiled(x.ptr,tb.idx,out.ptr,N,K,int(tile_h),int(tile_w))
        return out
    def bind_rg_tensor(self,r_packed,g_packed,b_q,r_lut,g_lut,b_lut,routing_k,N,K):
        r_page,_=weight_to_rgba_page(r_packed);g_page,_=weight_to_rgba_page(g_packed);b_page,_=weight_to_rgba_page(b_q)
        r_tb=self.bind_texture(r_page);g_tb=self.bind_texture(g_page);b_tb=self.bind_texture(b_page)
        rl=np.ascontiguousarray(r_lut,dtype=np.float32);gl=np.ascontiguousarray(g_lut,dtype=np.float32);bl=np.ascontiguousarray(b_lut,dtype=np.float32)
        rp=_lib.ari_alloc(rl.nbytes);_lib.ari_h2d(ct.c_void_p(rp),rl.ctypes.data_as(ct.c_void_p),rl.nbytes);self._bufs.add(rp)
        gp=_lib.ari_alloc(gl.nbytes);_lib.ari_h2d(ct.c_void_p(gp),gl.ctypes.data_as(ct.c_void_p),gl.nbytes);self._bufs.add(gp)
        bp=_lib.ari_alloc(bl.nbytes);_lib.ari_h2d(ct.c_void_p(bp),bl.ctypes.data_as(ct.c_void_p),bl.nbytes);self._bufs.add(bp)
        return RgBundle(r_tb,g_tb,b_tb,rp,gp,bp,routing_k,N,K)
    def free_rg_bundle(self,bundle):
        self.free_texture(bundle.r_tb);self.free_texture(bundle.g_tb);self.free_texture(bundle.b_tb)
        for p in (bundle.r_lut_ptr,bundle.g_lut_ptr,bundle.b_lut_ptr):
            if p in self._bufs:_lib.ari_free(ct.c_void_p(p));self._bufs.discard(p)
    def rg_matmul_fused(self,x,bundle):
        out=self.alloc((bundle.N,),dtype=np.uint16)
        _lib.ari_rg_gemv_fused(x.ptr,bundle.r_tb.idx,bundle.g_tb.idx,bundle.b_tb.idx,ct.c_void_p(bundle.r_lut_ptr),ct.c_void_p(bundle.g_lut_ptr),ct.c_void_p(bundle.b_lut_ptr),bundle.routing_k,out.ptr,bundle.N,bundle.K)
        return out
    def rg_gemv_into(self,x_ptr,bundle,y_ptr):
        _lib.ari_rg_gemv_fused(ct.c_void_p(x_ptr),bundle.r_tb.idx,bundle.g_tb.idx,bundle.b_tb.idx,ct.c_void_p(bundle.r_lut_ptr),ct.c_void_p(bundle.g_lut_ptr),ct.c_void_p(bundle.b_lut_ptr),bundle.routing_k,ct.c_void_p(y_ptr),bundle.N,bundle.K)
    def elem_add(self,a,b):
        out=self.alloc(a.shape)
        _lib.ari_elem_add(a.ptr,b.ptr,out.ptr,a.size)
        return out
    def elem_mul(self,a,b):
        out=self.alloc(a.shape)
        _lib.ari_elem_mul(a.ptr,b.ptr,out.ptr,a.size)
        return out
    def activate(self,x):
        out=self.alloc(x.shape)
        _lib.ari_activate(x.ptr,out.ptr,x.size)
        return out
    def rms_norm(self,x):
        rows=x.shape[0] if len(x.shape)>=2 else 1
        cols=x.shape[-1]
        out=self.alloc(x.shape)
        _lib.ari_rms_norm(x.ptr,out.ptr,rows,cols)
        return out
    def tex_embed(self,tb,ids_np,S,D):
        ids=np.ascontiguousarray(ids_np.ravel(),dtype=np.int32)
        id_ptr=_lib.ari_alloc(ids.nbytes)
        _lib.ari_h2d(ct.c_void_p(id_ptr),ids.ctypes.data_as(ct.c_void_p),ids.nbytes)
        out=self.alloc((S,D))
        _lib.ari_tex_embed(tb.idx,ct.c_void_p(id_ptr),out.ptr,S,D)
        _lib.ari_free(ct.c_void_p(id_ptr))
        return out
    def neg_score(self,Q,K_,B,H,S,T,Hd):
        out=self.alloc((B,H,S,T))
        _lib.ari_neg_score(Q.ptr,K_.ptr,out.ptr,B,H,S,T,Hd)
        return out
    def attn_norm(self,sc,B,H,S,T):
        _lib.ari_attn_norm(sc.ptr,B,H,S,T)
        return sc
    def apply_v(self,sc,V,B,H,S,T,Hd):
        out=self.alloc((B,H,S,Hd))
        _lib.ari_apply_v(sc.ptr,V.ptr,out.ptr,B,H,S,T,Hd)
        return out
    def xpose_bshd(self,buf,B,S,H,Hd):
        out=self.alloc((B,H,S,Hd))
        _lib.ari_xpose_bshd(buf.ptr,out.ptr,B,S,H,Hd)
        return out
    def xpose_bhsd(self,buf,B,H,S,Hd):
        out=self.alloc((B,S,H,Hd))
        _lib.ari_xpose_bhsd(buf.ptr,out.ptr,B,H,S,Hd)
        return out
    def repeat_kv(self,buf,B,Hkv,H,T,Hd):
        out=self.alloc((B,H,T,Hd))
        _lib.ari_repeat_kv(buf.ptr,out.ptr,B,Hkv,H,T,Hd)
        return out
    def tex_attention(self,x,tq,tk,tv,to,B,S,H,Hkv,Hd):
        D=x.shape[-1];r=H//Hkv
        q=self.tex_matmul_t(x.reshape(B*S,D),tq,B*S,D,H*Hd)
        k=self.tex_matmul_t(x.reshape(B*S,D),tk,B*S,D,Hkv*Hd)
        v=self.tex_matmul_t(x.reshape(B*S,D),tv,B*S,D,Hkv*Hd)
        qt=self.xpose_bshd(q.reshape(B,S,H,Hd),B,S,H,Hd);self.free(q)
        kt=self.xpose_bshd(k.reshape(B,S,Hkv,Hd),B,S,Hkv,Hd);self.free(k)
        vt=self.xpose_bshd(v.reshape(B,S,Hkv,Hd),B,S,Hkv,Hd);self.free(v)
        T=S
        if r>1:
            kr=self.repeat_kv(kt,B,Hkv,H,T,Hd);self.free(kt);kt=kr
            vr=self.repeat_kv(vt,B,Hkv,H,T,Hd);self.free(vt);vt=vr
        sc=self.neg_score(qt,kt,B,H,S,T,Hd);self.free(qt);self.free(kt)
        sc=self.attn_norm(sc,B,H,S,T)
        out=self.apply_v(sc,vt,B,H,S,T,Hd);self.free(sc);self.free(vt)
        out2=self.xpose_bhsd(out,B,H,S,Hd);self.free(out)
        o=self.tex_matmul_t(out2.reshape(B*S,H*Hd),to,B*S,H*Hd,D);self.free(out2)
        return o.reshape(B,S,D)
    def fused_mlp(self,x,tg,tu,td):
        M=x.shape[0];D=x.shape[-1];inter=tg.n_weights//D
        gate=self.tex_matmul_t(x,tg,M,D,inter)
        gate_act=self.activate(gate);self.free(gate)
        up=self.tex_matmul_t(x,tu,M,D,inter)
        h=self.elem_mul(gate_act,up);self.free(gate_act);self.free(up)
        out=self.tex_matmul_t(h,td,M,inter,D);self.free(h)
        return out
    def tex_block_forward(self,x,attn_texs,mlp_texs,B,S,H,Hkv,Hd):
        tq,tk,tv,to=attn_texs
        tg,tu,td=mlp_texs
        D=x.shape[-1]
        xn=self.rms_norm(x.reshape(B*S,D))
        a=self.tex_attention(xn.reshape(B,S,D),tq,tk,tv,to,B,S,H,Hkv,Hd);self.free(xn)
        r1=self.elem_add(x.reshape(B*S,D),a.reshape(B*S,D));self.free(a)
        r1n=self.rms_norm(r1)
        m=self.fused_mlp(r1n,tg,tu,td);self.free(r1n)
        r2=self.elem_add(r1,m);self.free(r1);self.free(m)
        return r2.reshape(B,S,D)
    def shutdown(self):
        for ptr in list(self._bufs):
            try:_lib.ari_free(ct.c_void_p(ptr))
            except:pass
        self._bufs.clear()
        for tb in self._texs:
            try:_lib.ari_free_texture(tb.idx)
            except:pass
        self._texs.clear()
        _lib.ari_shutdown()
_engine=None
def get_engine():
    global _engine
    if _engine is None:_engine=ARIEngine()
    return _engine
def is_available():
    try:return _load()
    except:return False
