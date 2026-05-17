import ctypes as ct,numpy as np,os,sys
from pathlib import Path
P=17
_dir=Path(__file__).resolve().parent
_WIN=sys.platform=='win32'
_so=_dir/'hip'/('libgf17_hip.dll'if _WIN else'libgf17_hip.so')
_lib=None
_ok=False
def _rocm_home():
    if os.environ.get('ROCM_HOME'):return Path(os.environ['ROCM_HOME'])
    if _WIN:
        base=Path('C:/Program Files/AMD/ROCm')
        if base.exists():
            vers=sorted(base.iterdir(),reverse=True)
            return vers[0]if vers else base/'6.2'
        return Path('C:/Program Files/AMD/ROCm/6.2')
    return Path('/opt/rocm')
def _load():
    global _lib,_ok
    if _ok:return True
    if not _so.exists():return False
    try:
        rocm=_rocm_home()
        hip_rt=rocm/'bin'/'amdhip64.dll'if _WIN else rocm/'lib'/'libamdhip64.so'
        if hip_rt.exists():
            ct.CDLL(str(hip_rt),mode=ct.RTLD_GLOBAL)if not _WIN else ct.WinDLL(str(hip_rt))
        _lib=ct.WinDLL(str(_so))if _WIN else ct.CDLL(str(_so),mode=ct.RTLD_GLOBAL)
        for nm,res,args in[
            ('gf17_init',ct.c_int,[ct.c_int]),
            ('gf17_shutdown',None,[]),
            ('gf17_alloc',ct.c_void_p,[ct.c_size_t]),
            ('gf17_free',None,[ct.c_void_p]),
            ('gf17_h2d',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_size_t]),
            ('gf17_d2h',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_size_t]),
            ('gf17_d2d',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_size_t]),
            ('gf17_sync',ct.c_int,[]),
            ('gf17_matmul_t',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int]),
            ('gf17_elem_add',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int]),
            ('gf17_elem_mul',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int]),
            ('gf17_activate',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_int]),
            ('gf17_rms_norm',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int]),
            ('gf17_neg_score',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('gf17_attn_norm',ct.c_int,[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('gf17_apply_v',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('gf17_xpose_bshd',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('gf17_xpose_bhsd',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('gf17_repeat_kv',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int]),
            ('gf17_embed',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int]),
            ('gf17_unpack2b',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int]),
            ('gf17_alloc_f32',ct.c_void_p,[ct.c_size_t]),
            ('gf17_h2d_f32',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_size_t]),
            ('gf17_d2h_f32',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_size_t]),
            ('gf17_dq_gemv_f32',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_float]),
            ('gf17_dq_gemv_b17_f32',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int]),
            ('gf17_rms_norm_f32',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_float]),
            ('gf17_elem_add_f32',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int]),
            ('gf17_silu_inp_f32',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int]),
            ('gf17_mqa_attn_f32',ct.c_int,[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_float]),
        ]:
            fn=getattr(_lib,nm)
            if res is not None:fn.restype=res
            fn.argtypes=args
        rc=_lib.gf17_init(0)
        _ok=(rc==0)
    except Exception as e:
        print(f"[gf17_engine] load failed: {e}",file=sys.stderr)
        _ok=False
    return _ok
class DevBuf:
    __slots__=('ptr','shape','size')
    def __init__(self,ptr,shape):
        self.ptr=ct.c_void_p(ptr) if not isinstance(ptr,ct.c_void_p) else ptr
        self.shape=tuple(shape)
        self.size=int(np.prod(shape))
    def reshape(self,*sh):
        n=int(np.prod(sh))
        assert n==self.size,f"reshape {self.shape}->{sh}: size mismatch {self.size}!={n}"
        return DevBuf(self.ptr,sh)
    def to_host(self):
        h=np.empty(self.shape,dtype=np.uint8)
        _lib.gf17_d2h(h.ctypes.data_as(ct.c_void_p),self.ptr,self.size)
        return h
    @staticmethod
    def from_host(arr):
        a=np.ascontiguousarray(arr,dtype=np.uint8)
        ptr=_lib.gf17_alloc(a.size)
        _lib.gf17_h2d(ct.c_void_p(ptr),a.ctypes.data_as(ct.c_void_p),a.size)
        return DevBuf(ptr,a.shape)
class GF17Engine:
    def __init__(self):
        if not _load():raise RuntimeError("GF17 HIP engine not available — build libgf17_hip.so first")
        self._bufs=set()
    def upload(self,arr):
        b=DevBuf.from_host(arr);self._bufs.add(b.ptr.value);return b
    def download(self,buf):
        _lib.gf17_sync();return buf.to_host()
    def alloc(self,shape):
        n=int(np.prod(shape));ptr=_lib.gf17_alloc(n)
        self._bufs.add(ptr);return DevBuf(ptr,shape)
    def free(self,buf):
        if buf.ptr and buf.ptr.value in self._bufs:
            _lib.gf17_free(buf.ptr);self._bufs.discard(buf.ptr.value);buf.ptr=None
    def sync(self):_lib.gf17_sync()
    def matmul_t(self,a,w):
        M,K=a.shape[0],a.shape[-1];N=w.shape[0]
        out=self.alloc((M,N))
        _lib.gf17_matmul_t(a.ptr,w.ptr,out.ptr,M,K,N)
        return out
    def elem_add(self,a,b):
        out=self.alloc(a.shape)
        _lib.gf17_elem_add(a.ptr,b.ptr,out.ptr,a.size)
        return out
    def elem_mul(self,a,b):
        out=self.alloc(a.shape)
        _lib.gf17_elem_mul(a.ptr,b.ptr,out.ptr,a.size)
        return out
    def activate(self,x):
        out=self.alloc(x.shape)
        _lib.gf17_activate(x.ptr,out.ptr,x.size)
        return out
    def rms_norm(self,x):
        rows=x.shape[0] if len(x.shape)>=2 else 1
        cols=x.shape[-1]
        out=self.alloc(x.shape)
        _lib.gf17_rms_norm(x.ptr,out.ptr,rows,cols)
        return out
    def neg_score(self,Q,K_,B,H,S,T,Hd):
        out=self.alloc((B,H,S,T))
        _lib.gf17_neg_score(Q.ptr,K_.ptr,out.ptr,B,H,S,T,Hd)
        return out
    def attn_norm(self,sc,B,H,S,T):
        _lib.gf17_attn_norm(sc.ptr,B,H,S,T)
        return sc
    def apply_v(self,sc,V,B,H,S,T,Hd):
        out=self.alloc((B,H,S,Hd))
        _lib.gf17_apply_v(sc.ptr,V.ptr,out.ptr,B,H,S,T,Hd)
        return out
    def xpose_bshd(self,buf,B,S,H,Hd):
        out=self.alloc((B,H,S,Hd))
        _lib.gf17_xpose_bshd(buf.ptr,out.ptr,B,S,H,Hd)
        return out
    def xpose_bhsd(self,buf,B,H,S,Hd):
        out=self.alloc((B,S,H,Hd))
        _lib.gf17_xpose_bhsd(buf.ptr,out.ptr,B,H,S,Hd)
        return out
    def repeat_kv(self,buf,B,Hkv,H,T,Hd):
        out=self.alloc((B,H,T,Hd))
        _lib.gf17_repeat_kv(buf.ptr,out.ptr,B,Hkv,H,T,Hd)
        return out
    def embed(self,emb,ids_np,S,D):
        ids=np.ascontiguousarray(ids_np.ravel(),dtype=np.int32)
        id_ptr=_lib.gf17_alloc(ids.nbytes)
        _lib.gf17_h2d(ct.c_void_p(id_ptr),ids.ctypes.data_as(ct.c_void_p),ids.nbytes)
        out=self.alloc((S,D))
        _lib.gf17_embed(emb.ptr,ct.c_void_p(id_ptr),out.ptr,S,D)
        _lib.gf17_free(ct.c_void_p(id_ptr))
        return out
    def unpack2b(self,packed,lut,n):
        out=self.alloc((n,))
        _lib.gf17_unpack2b(packed.ptr,lut.ptr,out.ptr,n)
        return out
    def fused_mlp(self,x,gw,uw,dw):
        M=x.shape[0];D=x.shape[-1];inter=gw.shape[0]
        gate=self.matmul_t(x,gw)
        gate_act=self.activate(gate);self.free(gate)
        up=self.matmul_t(x,uw)
        h=self.elem_mul(gate_act,up);self.free(gate_act);self.free(up)
        out=self.matmul_t(h,dw);self.free(h)
        return out
    def attention(self,x,wq,wk,wv,wo,B,S,H,Hkv,Hd):
        id=x.shape[-1];ad=H*Hd;r=H//Hkv
        q=self.matmul_t(x.reshape(B*S,id),wq)
        k=self.matmul_t(x.reshape(B*S,id),wk)
        v=self.matmul_t(x.reshape(B*S,id),wv)
        qt=self.xpose_bshd(q.reshape(B,S,H,Hd),B,S,H,Hd);self.free(q)
        kt=self.xpose_bshd(k.reshape(B,S,Hkv,Hd),B,S,Hkv,Hd);self.free(k)
        vt=self.xpose_bshd(v.reshape(B,S,Hkv,Hd),B,S,Hkv,Hd);self.free(v)
        T=S
        if r>1:kr=self.repeat_kv(kt,B,Hkv,H,T,Hd);self.free(kt);kt=kr;vr=self.repeat_kv(vt,B,Hkv,H,T,Hd);self.free(vt);vt=vr
        sc=self.neg_score(qt,kt,B,H,S,T,Hd);self.free(qt);self.free(kt)
        sc=self.attn_norm(sc,B,H,S,T)
        out=self.apply_v(sc,vt,B,H,S,T,Hd);self.free(sc);self.free(vt)
        out2=self.xpose_bhsd(out,B,H,S,Hd);self.free(out)
        o=self.matmul_t(out2.reshape(B*S,ad),wo);self.free(out2)
        return o.reshape(B,S,id)
    def block_forward(self,x,attn_ws,mlp_ws,B,S,H,Hkv,Hd):
        wq,wk,wv,wo=attn_ws
        gw,uw,dw=mlp_ws
        D=x.shape[-1]
        xn=self.rms_norm(x.reshape(B*S,D))
        a=self.attention(xn.reshape(B,S,D),wq,wk,wv,wo,B,S,H,Hkv,Hd);self.free(xn)
        r1=self.elem_add(x.reshape(B*S,D),a.reshape(B*S,D));self.free(a)
        r1n=self.rms_norm(r1)
        m=self.fused_mlp(r1n,gw,uw,dw);self.free(r1n)
        r2=self.elem_add(r1,m);self.free(r1);self.free(m)
        return r2.reshape(B,S,D)
    def shutdown(self):
        for ptr in list(self._bufs):
            try:_lib.gf17_free(ct.c_void_p(ptr))
            except:pass
        self._bufs.clear()
        _lib.gf17_shutdown()
_engine=None
def get_engine():
    global _engine
    if _engine is None:_engine=GF17Engine()
    return _engine
def is_available():
    try:return _load()
    except:return False
class DevBufF32:
    __slots__=('ptr','shape','size')
    def __init__(self,ptr,shape):
        self.ptr=ct.c_void_p(ptr) if not isinstance(ptr,ct.c_void_p) else ptr
        self.shape=tuple(shape)
        self.size=int(np.prod(shape))
    def to_host(self):
        h=np.empty(self.shape,dtype=np.float32)
        _lib.gf17_d2h_f32(h.ctypes.data_as(ct.c_void_p),self.ptr,self.size)
        return h
    @staticmethod
    def from_host(arr):
        a=np.ascontiguousarray(arr,dtype=np.float32)
        ptr=_lib.gf17_alloc_f32(a.size)
        _lib.gf17_h2d_f32(ct.c_void_p(ptr),a.ctypes.data_as(ct.c_void_p),a.size)
        return DevBufF32(ptr,a.shape)
class F32Engine:
    def __init__(self):
        if not _load():raise RuntimeError("GF17 HIP engine not available")
        self._bufs=set()
    def alloc(self,shape):
        n=int(np.prod(shape));ptr=_lib.gf17_alloc_f32(n)
        self._bufs.add(ptr);return DevBufF32(ptr,shape)
    def upload(self,arr):
        b=DevBufF32.from_host(arr);self._bufs.add(b.ptr.value);return b
    def upload_u8(self,arr):
        b=DevBuf.from_host(arr);self._bufs.add(b.ptr.value);return b
    def download(self,buf):
        _lib.gf17_sync();return buf.to_host()
    def free(self,buf):
        if buf is None or buf.ptr is None:return
        v=buf.ptr.value if isinstance(buf.ptr,ct.c_void_p) else buf.ptr
        if v and v in self._bufs:
            fn=_lib.gf17_free if isinstance(buf,DevBuf) else _lib.gf17_free
            fn(ct.c_void_p(v));self._bufs.discard(v);buf.ptr=None
    def sync(self):_lib.gf17_sync()
    def dq_gemv(self,W_u8,x_f32,K,N,scale):
        out=self.alloc((N,))
        _lib.gf17_dq_gemv_f32(W_u8.ptr,x_f32.ptr,out.ptr,K,N,ct.c_float(scale));return out
    def dq_gemv_b17(self,W_u8,x_f32,K,N):
        out=self.alloc((N,))
        _lib.gf17_dq_gemv_b17_f32(W_u8.ptr,x_f32.ptr,out.ptr,K,N);return out
    def rms_norm(self,x,rows,cols,eps=1e-6):
        out=self.alloc(x.shape)
        _lib.gf17_rms_norm_f32(x.ptr,out.ptr,rows,cols,ct.c_float(eps));return out
    def add(self,a,b):
        out=self.alloc(a.shape)
        _lib.gf17_elem_add_f32(a.ptr,b.ptr,out.ptr,a.size);return out
    def silu(self,gate,up):
        out=self.alloc(gate.shape)
        _lib.gf17_silu_inp_f32(gate.ptr,up.ptr,out.ptr,gate.size);return out
    def mqa_attn(self,q,kc,vc,new_k,new_v,sb,ao_buf,H,Hkv,Hd,T,inv_sqrt):
        _lib.gf17_mqa_attn_f32(q.ptr,kc.ptr,vc.ptr,new_k.ptr,new_v.ptr,sb.ptr,ao_buf.ptr,H,Hkv,Hd,T,ct.c_float(inv_sqrt))
_f32_engine=None
def get_f32_engine():
    global _f32_engine
    if _f32_engine is None:_f32_engine=F32Engine()
    return _f32_engine
def f32_available():
    try:return _load()
    except:return False
