import numpy as np
W_MAT=np.array([[pow(4,k*n,17) for n in range(4)] for k in range(4)],dtype=np.int64)
WI_MAT=np.array([[pow(13,k*n,17) for n in range(4)] for k in range(4)],dtype=np.int64)
INV_SCALE_2D=16
INV_SCALE_1D=13
def ntt2d_blocks(X:np.ndarray)->np.ndarray:
 orig_shape=X.shape
 if len(orig_shape)==2:
  h,w=orig_shape
  blocks=X.reshape(h//4,4,w//4,4).swapaxes(1,2).reshape(-1,4,4)
  res=((W_MAT@blocks@W_MAT.T)%17).astype(np.uint8)
  return res.reshape(h//4,w//4,4,4).swapaxes(1,2).reshape(h,w)
 elif len(orig_shape)==3 and orig_shape[-2:]==(4,4):
  return ((W_MAT@X@W_MAT.T)%17).astype(np.uint8)
 raise ValueError("Expected 2D array with dimensions divisible by 4, or (B, 4, 4)")
def intt2d_blocks(Y:np.ndarray)->np.ndarray:
 orig_shape=Y.shape
 if len(orig_shape)==2:
  h,w=orig_shape
  blocks=Y.reshape(h//4,4,w//4,4).swapaxes(1,2).reshape(-1,4,4)
  res=((INV_SCALE_2D*(WI_MAT@blocks@WI_MAT.T))%17).astype(np.uint8)
  return res.reshape(h//4,w//4,4,4).swapaxes(1,2).reshape(h,w)
 elif len(orig_shape)==3 and orig_shape[-2:]==(4,4):
  return ((INV_SCALE_2D*(WI_MAT@Y@WI_MAT.T))%17).astype(np.uint8)
 raise ValueError("Expected 2D array with dimensions divisible by 4, or (B, 4, 4)")
def ntt_channel4(X:np.ndarray)->np.ndarray:
 return ((X@W_MAT.T)%17).astype(np.uint8)
def intt_channel4(Y:np.ndarray)->np.ndarray:
 return (((Y@WI_MAT.T)*INV_SCALE_1D)%17).astype(np.uint8)
def build_mipmap_pyramid(page:np.ndarray,max_levels:int=5)->list[np.ndarray]:
 pyramid=[page]
 curr=page
 for lvl in range(1,max_levels):
  h,w=curr.shape[:2]
  if h<4 or w<4:break
  c=curr.shape[2] if curr.ndim==3 else 1
  reshaped=curr.reshape(h//4,4,w//4,4,c).swapaxes(1,2).reshape(-1,4,4,c)
  dc_sums=reshaped.sum(axis=(1,2))%17
  dc_means=(dc_sums*INV_SCALE_2D)%17
  next_lvl=dc_means.reshape(h//4,w//4,c).astype(np.uint8)
  pyramid.append(next_lvl)
  curr=next_lvl
 return pyramid
def sample_lod(pyramid:list[np.ndarray],u:float,v:float,lod:float)->np.ndarray:
 lod_idx=max(0,min(int(lod),len(pyramid)-1))
 lvl_map=pyramid[lod_idx]
 h,w=lvl_map.shape[:2]
 px=min(int(u*w),w-1)
 py=min(int(v*h),h-1)
 return lvl_map[py,px]
def analyze_ntt_sparsity(Y:np.ndarray)->dict:
 h,w=Y.shape[:2]
 blocks=Y.reshape(h//4,4,w//4,4).swapaxes(1,2).reshape(-1,4,4)
 ac=blocks.copy()
 ac[:,0,0]=0
 zero_ac=int((ac==0).sum()-(len(blocks)))
 total_ac=int(blocks.size-len(blocks))
 zero_rate=round(zero_ac/max(total_ac,1),4)
 raw_bits_per_element=4.087
 est_entropy_bits=round((1.0-zero_rate)*raw_bits_per_element+zero_rate*0.2,3)
 return {"total_blocks":len(blocks),"ac_elements":total_ac,"ac_zeros":zero_ac,"ac_zero_rate":zero_rate,"est_bits_per_weight":est_entropy_bits}
