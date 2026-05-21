import os,ctypes,torch
_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),"hip")
_DLL=os.path.join(_DIR,"libconv3x3_wmma_lds.dll")
_lib=None
def _load():
    global _lib
    if _lib is not None:return _lib
    _lib=ctypes.CDLL(_DLL)
    _lib.conv3x3_wmma_lds_init.argtypes=[ctypes.c_int];_lib.conv3x3_wmma_lds_init.restype=ctypes.c_int
    _lib.conv3x3_wmma_lds_run.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    _lib.conv3x3_wmma_lds_run.restype=ctypes.c_int
    _lib.conv3x3_wmma_lds_sync.restype=ctypes.c_int
    _lib.conv3x3_wmma_lds_init(0)
    return _lib
def conv3x3_wmma_lds_s1p1(x,w,b=None):
    assert x.dtype==torch.float16 and w.dtype==torch.float16 and x.is_cuda and w.is_cuda
    assert w.shape[2]==3 and w.shape[3]==3
    N,Cin,H,W=x.shape
    Cout=w.shape[0]
    assert w.shape[1]==Cin and H%4==0 and W%4==0
    lib=_load()
    x=x.contiguous();w=w.contiguous()
    y=torch.empty(N,Cout,H,W,device=x.device,dtype=torch.float16)
    bp=ctypes.c_void_p(b.data_ptr()) if b is not None else ctypes.c_void_p(0)
    rc=lib.conv3x3_wmma_lds_run(ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(w.data_ptr()),bp,ctypes.c_void_p(y.data_ptr()),N,Cin,Cout,H,W)
    if rc!=0:raise RuntimeError(f"conv3x3_wmma_lds_run rc={rc}")
    lib.conv3x3_wmma_lds_sync()
    return y
