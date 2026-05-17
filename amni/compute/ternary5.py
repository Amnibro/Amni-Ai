import numpy as np
TRIT_BASE=3
TRIT_PACK=5
TRIT_STATES=TRIT_BASE**TRIT_PACK
TRIT_POW=np.array([1,3,9,27,81],dtype=np.uint32)
TERNARY_GF17=np.array([16,0,1],dtype=np.uint8)
TERNARY_SIGNED=np.array([-1,0,1],dtype=np.int8)
TERNARY_FP16=np.array([-1.0,0.0,1.0],dtype=np.float16)
def is_ternary_compatible(arr)->bool:
    a=np.asarray(arr).reshape(-1)
    return bool(np.all(np.isin(a,(-1,0,1))) or np.all(np.isin(a,(16,0,1))) or np.all(np.isin(a,(0,1,2))))
def ternary_codes(arr):
    a=np.asarray(arr).reshape(-1)
    if np.all(np.isin(a,(0,1,2))):return a.astype(np.uint8,copy=False)
    if np.all(np.isin(a,(16,0,1))):return np.where(a==16,0,np.where(a==0,1,2)).astype(np.uint8)
    if np.all(np.isin(a,(-1,0,1))):return np.where(a<0,0,np.where(a==0,1,2)).astype(np.uint8)
    raise ValueError('ternary carrier expects values in {-1,0,1}, {16,0,1}, or {0,1,2}')
def pack_ternary5(arr):
    codes=ternary_codes(arr)
    n=codes.size
    pad=(-n)%TRIT_PACK
    if pad:codes=np.concatenate([codes,np.full(pad,1,dtype=np.uint8)])
    blocks=codes.reshape(-1,TRIT_PACK).astype(np.uint32)
    packed=(blocks*TRIT_POW[np.newaxis,:]).sum(axis=1).astype(np.uint8)
    px_pad=(-packed.size)%4
    if px_pad:packed=np.concatenate([packed,np.zeros(px_pad,dtype=np.uint8)])
    return packed.reshape(-1,4),n
def unpack_ternary5_codes(primary,n_weights):
    if n_weights<=0:return np.zeros((0,),dtype=np.uint8)
    packed=np.ascontiguousarray(primary,dtype=np.uint8).reshape(-1).astype(np.uint32)[:(n_weights+TRIT_PACK-1)//TRIT_PACK]
    rem=packed.copy()
    out=np.empty((packed.size,TRIT_PACK),dtype=np.uint8)
    for i in range(TRIT_PACK):out[:,i]=(rem%TRIT_BASE).astype(np.uint8);rem//=TRIT_BASE
    return out.reshape(-1)[:n_weights]
def unpack_ternary5_gf17(primary,n_weights):
    return TERNARY_GF17[unpack_ternary5_codes(primary,n_weights)]
def unpack_ternary5_signed(primary,n_weights):
    return TERNARY_SIGNED[unpack_ternary5_codes(primary,n_weights)]
def unpack_ternary5_fp16(primary,n_weights):
    return TERNARY_FP16[unpack_ternary5_codes(primary,n_weights)]
def pack_ternary5_page(arr,page_w=4096):
    primary,n=pack_ternary5(arr)
    n_px=primary.shape[0]
    h=(n_px+page_w-1)//page_w
    page=np.zeros((h,page_w,4),dtype=np.uint8)
    page.reshape(-1,4)[:n_px]=primary
    return page,n,n_px