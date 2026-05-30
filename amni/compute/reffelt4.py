import numpy as np
from amni.compute.ternary5 import pack_ternary5_page,unpack_ternary5_signed,TRIT_PACK
REFFELT_K4=(1,17,289,4913)
REFFELT_K8=(1,17,289,4913,83521,1419857,24137569,410338673)
_K8=np.array(REFFELT_K8,dtype=np.uint64)
def encode_ids_to_rgba2(ids):
    a=np.ascontiguousarray(ids,dtype=np.uint64).reshape(-1)
    d=np.stack([((a//k)%17).astype(np.uint8) for k in REFFELT_K8],axis=-1)
    return d.reshape(-1,4)
def decode_rgba2_to_ids(rgba,n_ids=None):
    p=np.ascontiguousarray(rgba,dtype=np.uint8).reshape(-1,8).astype(np.uint64)
    vals=(p*_K8[None,:]).sum(axis=1).astype(np.int64)
    return vals if n_ids is None else vals[:n_ids]
def roundtrip_check_ids(ids):
    a=np.ascontiguousarray(ids,dtype=np.int64).reshape(-1)
    return np.array_equal(a,decode_rgba2_to_ids(encode_ids_to_rgba2(a),a.size))
def encode_fp16_to_rgba4(w):
    a=np.ascontiguousarray(w,dtype=np.float16).view(np.uint16).reshape(-1).astype(np.uint32)
    r=(a%17).astype(np.uint8);g=((a//17)%17).astype(np.uint8);b=((a//289)%17).astype(np.uint8);al=((a//4913)%17).astype(np.uint8)
    return np.stack([r,g,b,al],axis=-1)
def decode_rgba4_to_fp16(rgba):
    p=np.ascontiguousarray(rgba,dtype=np.uint8).reshape(-1,4).astype(np.uint32)
    bits=(p[:,0]+p[:,1]*17+p[:,2]*289+p[:,3]*4913).astype(np.uint16)
    return bits.view(np.float16)
def pack_rgba4_page(w,page_w=4096):
    rgba=encode_fp16_to_rgba4(w)
    n=rgba.shape[0]
    h=(n+page_w-1)//page_w
    page=np.zeros((h,page_w,4),dtype=np.uint8)
    page.reshape(-1,4)[:n]=rgba
    return page,n
def encode_fp16_to_rgba16_quad(w):
    bits=np.ascontiguousarray(w,dtype=np.float16).view(np.uint16).reshape(-1)
    pad=(-bits.size)%4
    if pad:bits=np.concatenate([bits,np.zeros(pad,dtype=np.uint16)])
    return bits.reshape(-1,4)
def decode_rgba16_quad_to_fp16(rgba,n_weights=None):
    bits=np.ascontiguousarray(rgba,dtype=np.uint16).reshape(-1)
    return bits[:bits.size if n_weights is None else n_weights].view(np.float16)
def pack_rgba16_quad_page(w,page_w=4096):
    rgba=encode_fp16_to_rgba16_quad(w)
    n=np.ascontiguousarray(w,dtype=np.float16).size
    h=(rgba.shape[0]+page_w-1)//page_w
    page=np.zeros((h,page_w,4),dtype=np.uint16)
    page.reshape(-1,4)[:rgba.shape[0]]=rgba
    return page,n
def pack_rgba16_quad_tile_page(w,tile_h=16,tile_w=16):
    arr=np.ascontiguousarray(w,dtype=np.float16)
    assert arr.ndim==2 and tile_h>0 and tile_w>0 and tile_w%4==0
    bits=arr.view(np.uint16)
    n,k=bits.shape
    npad=((n+tile_h-1)//tile_h)*tile_h
    kpad=((k+tile_w-1)//tile_w)*tile_w
    tile_pw=tile_w//4
    pad=np.zeros((npad,kpad),dtype=np.uint16)
    pad[:n,:k]=bits
    page=pad.reshape(npad//tile_h,tile_h,kpad//tile_w,tile_w).reshape(npad//tile_h,tile_h,kpad//tile_w,tile_pw,4).reshape(npad,(kpad//tile_w)*tile_pw,4)
    return page,n*k
def decode_rgba16_quad_tile_to_fp16(page,shape,tile_h=16,tile_w=16):
    n,k=map(int,shape)
    assert tile_h>0 and tile_w>0 and tile_w%4==0
    npad=((n+tile_h-1)//tile_h)*tile_h
    kpad=((k+tile_w-1)//tile_w)*tile_w
    tile_pw=tile_w//4
    pad=np.ascontiguousarray(page,dtype=np.uint16).reshape(npad,tile_pw*(kpad//tile_w),4).reshape(npad//tile_h,tile_h,kpad//tile_w,tile_pw,4).reshape(npad,kpad)
    return pad[:n,:k].reshape(-1).view(np.float16)
def _tile2d(arr,tile_h,tile_w):
    n,k=arr.shape
    npad=((n+tile_h-1)//tile_h)*tile_h;kpad=((k+tile_w-1)//tile_w)*tile_w
    p=np.zeros((npad,kpad),dtype=arr.dtype);p[:n,:k]=arr
    return p,npad,kpad
def pack_rtier_page(w,tile_h=16,tile_w=16):
    arr=np.ascontiguousarray(w,dtype=np.float16);assert arr.ndim==2
    n,k=arr.shape;p,npad,kpad=_tile2d(arr,tile_h,tile_w)
    nt_n,nt_k=npad//tile_h,kpad//tile_w
    tiles=p.reshape(nt_n,tile_h,nt_k,tile_w).transpose(0,2,1,3).reshape(nt_n*nt_k,tile_h*tile_w).astype(np.float32)
    abs_t=np.abs(tiles);thresh=np.median(abs_t,axis=1,keepdims=True)
    mask=abs_t>thresh;denom=mask.sum(axis=1,keepdims=True).clip(min=1)
    s=(abs_t*mask).sum(axis=1,keepdims=True)/denom
    trit=np.where(mask,np.sign(tiles),0).astype(np.int8)
    scales=s.reshape(nt_n,nt_k).astype(np.float16)
    trit_blocks=trit.reshape(nt_n,nt_k,tile_h,tile_w).transpose(0,2,1,3).reshape(nt_n*tile_h,nt_k*tile_w)
    page,n_w,_=pack_ternary5_page(trit_blocks.reshape(-1))
    return page,scales,(n,k),(npad,kpad),n_w
def unpack_rtier_page(page,scales,shape,padded_shape,tile_h=16,tile_w=16):
    n,k=int(shape[0]),int(shape[1]);npad,kpad=int(padded_shape[0]),int(padded_shape[1])
    nt_n,nt_k=npad//tile_h,kpad//tile_w;n_w=npad*kpad
    trit=unpack_ternary5_signed(page.reshape(-1,4),n_w).astype(np.float32).reshape(npad,kpad)
    s=np.ascontiguousarray(scales,dtype=np.float16).astype(np.float32).reshape(nt_n,nt_k)
    blk=trit.reshape(nt_n,tile_h,nt_k,tile_w).transpose(0,2,1,3).reshape(nt_n,nt_k,tile_h*tile_w)
    rec=(blk*s[:,:,None]).reshape(nt_n,nt_k,tile_h,tile_w).transpose(0,2,1,3).reshape(npad,kpad)
    return rec[:n,:k].astype(np.float16)
def pack_rtier_correction(w,w_recon,page_w=4096):
    arr=np.ascontiguousarray(w,dtype=np.float16);rec=np.ascontiguousarray(w_recon,dtype=np.float16);assert arr.shape==rec.shape
    src_bits=arr.view(np.uint16);rec_bits=rec.view(np.uint16)
    delta=(src_bits.astype(np.int32)-rec_bits.astype(np.int32)).astype(np.int32)
    delta_u=(delta&0xFFFF).astype(np.uint16).reshape(-1)
    pad=(-delta_u.size)%4
    if pad:delta_u=np.concatenate([delta_u,np.zeros(pad,dtype=np.uint16)])
    rgba=delta_u.reshape(-1,4)
    h=(rgba.shape[0]+page_w-1)//page_w
    page=np.zeros((h,page_w,4),dtype=np.uint16);page.reshape(-1,4)[:rgba.shape[0]]=rgba
    return page,arr.size
def apply_rtier_correction(w_recon,correction_page,shape):
    n,k=int(shape[0]),int(shape[1]);total=n*k
    rec=np.ascontiguousarray(w_recon,dtype=np.float16).reshape(-1)
    delta=np.ascontiguousarray(correction_page,dtype=np.uint16).reshape(-1)[:total]
    rec_bits=rec.view(np.uint16).astype(np.int32)
    truth_bits=((rec_bits+delta.astype(np.int32))&0xFFFF).astype(np.uint16)
    return truth_bits.view(np.float16).reshape(n,k)
def roundtrip_check_rtier(w,tile_h=16,tile_w=16):
    page,scales,sh,psh,_=pack_rtier_page(w,tile_h,tile_w)
    rec=unpack_rtier_page(page,scales,sh,psh,tile_h,tile_w)
    cpage,_=pack_rtier_correction(w,rec)
    truth=apply_rtier_correction(rec,cpage,sh)
    src=np.ascontiguousarray(w,dtype=np.float16)
    return np.array_equal(src.view(np.uint16),truth.view(np.uint16))
def roundtrip_check(w):
    return np.array_equal(np.ascontiguousarray(w,dtype=np.float16).view(np.uint16).reshape(-1),decode_rgba4_to_fp16(encode_fp16_to_rgba4(w)).view(np.uint16))
def roundtrip_check_quad(w):
    src=np.ascontiguousarray(w,dtype=np.float16).view(np.uint16).reshape(-1)
    return np.array_equal(src,decode_rgba16_quad_to_fp16(encode_fp16_to_rgba16_quad(w),src.size).view(np.uint16))
def roundtrip_check_quad_tile(w,tile_h=16,tile_w=16):
    src=np.ascontiguousarray(w,dtype=np.float16).view(np.uint16).reshape(-1)
    return np.array_equal(src,decode_rgba16_quad_tile_to_fp16(pack_rgba16_quad_tile_page(w,tile_h,tile_w)[0],np.ascontiguousarray(w,dtype=np.float16).shape,tile_h,tile_w).view(np.uint16))
