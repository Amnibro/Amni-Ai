import os,sys,math,time
from collections import OrderedDict
import numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_H0)
from amni.compute.ptex_ntt_mipmap import ntt2d_blocks,intt2d_blocks,build_mipmap_pyramid,sample_lod
TOTAL_PARAMS_1T=1_000_000_000_000
PAGE_DIM=4096
CHANNELS=4
TRITS_PER_BYTE=5
PARAMS_PER_PAGE=PAGE_DIM*PAGE_DIM*CHANNELS*TRITS_PER_BYTE
TOTAL_PAGES_1T=int(math.ceil(TOTAL_PARAMS_1T/PARAMS_PER_PAGE))
DOMAIN_RANGES={'math':(0,745),'stem':(745,1490),'civics':(1490,2235),'code':(2235,TOTAL_PAGES_1T)}
LOD_RECEPTIVE_TOKENS={0:1,1:4,2:16,3:64,4:256,5:1024,6:4096}
class Ptex1TTileCache:
 def __init__(self,max_tiles:int=4,tile_size:int=64):
  self.max_tiles=max_tiles
  self.tile_size=tile_size
  self.cache=OrderedDict()
  self.hits=0
  self.misses=0
 def get_tile_key(self,page_idx:int,tx:int,ty:int,lod:int)->tuple:return (page_idx,tx,ty,lod)
 def fetch_tile(self,page_idx:int,tx:int,ty:int,lod:int=0)->np.ndarray:
  k=self.get_tile_key(page_idx,tx,ty,lod)
  if k in self.cache:
   self.hits+=1
   self.cache.move_to_end(k)
   return self.cache[k]
  self.misses+=1
  rng=np.random.default_rng(abs(hash((page_idx,tx,ty,lod)))%(2**32))
  tile=rng.integers(0,17,size=(self.tile_size,self.tile_size,4),dtype=np.uint8)
  if len(self.cache)>=self.max_tiles:self.cache.popitem(last=False)
  self.cache[k]=tile
  return tile
 def memory_footprint_bytes(self)->int:return sum(t.nbytes for t in self.cache.values())
class Ptex1TContinuumEngine:
 def __init__(self,cache_tiles:int=4):
  self.total_params=TOTAL_PARAMS_1T
  self.total_pages=TOTAL_PAGES_1T
  self.cache=Ptex1TTileCache(max_tiles=cache_tiles,tile_size=64)
 def virtual_to_page_coord(self,domain:str,u:float,v:float,lod_level:int=0)->tuple:
  dom=domain.lower() if domain.lower() in DOMAIN_RANGES else 'stem'
  p_start,p_end=DOMAIN_RANGES[dom]
  n_dom_pages=p_end-p_start
  u_clamped=max(0.0,min(1.0,u))
  v_clamped=max(0.0,min(1.0,v))
  page_offset=int((u_clamped*10000+v_clamped*100)%n_dom_pages)
  page_idx=p_start+page_offset
  dim=max(1,PAGE_DIM>>lod_level)
  px=int(u_clamped*(dim-1))
  py=int(v_clamped*(dim-1))
  return page_idx,px,py
 def sample_1t_continuum(self,domain:str,u:float,v:float,lod:float=0.0)->dict:
  lod_clamped=max(0.0,min(6.0,lod))
  lod_int=int(lod_clamped)
  page_idx,px,py=self.virtual_to_page_coord(domain,u,v,lod_level=lod_int)
  tx=px//self.cache.tile_size
  ty=py//self.cache.tile_size
  tile=self.cache.fetch_tile(page_idx,tx,ty,lod=lod_int)
  lx=px%self.cache.tile_size
  ly=py%self.cache.tile_size
  val=tile[lx,ly].copy()
  receptive=LOD_RECEPTIVE_TOKENS.get(lod_int,4096)
  return {'domain':domain,'page_idx':page_idx,'lod':lod_clamped,'receptive_tokens':receptive,'px':px,'py':py,'val':val.tolist()}
 def get_architecture_metrics(self)->dict:
  raw_fp16_gb=round((TOTAL_PARAMS_1T*2.0)/(1024**3),2)
  raw_8bit_gb=round((TOTAL_PARAMS_1T*1.0)/(1024**3),2)
  ternary5_gb=round((TOTAL_PARAMS_1T/5.0)/(1024**3),2)
  ntt_compressed_gb=round((TOTAL_PARAMS_1T*0.878/8.0)/(1024**3),2)
  return {
   'total_parameters':TOTAL_PARAMS_1T,
   'total_pages_required':TOTAL_PAGES_1T,
   'params_per_page':PARAMS_PER_PAGE,
   'page_dimension':f'{PAGE_DIM}x{PAGE_DIM} RGBA',
   'memory_footprint':{
    'fp16_dense_gb':raw_fp16_gb,
    'int8_dense_gb':raw_8bit_gb,
    'ternary5_uncompressed_gb':ternary5_gb,
    'ntt_continuum_compressed_gb':ntt_compressed_gb,
    'compression_ratio_vs_fp16':round(raw_fp16_gb/ntt_compressed_gb,2),
    'working_set_cache_bytes':self.cache.memory_footprint_bytes(),
    'working_set_cache_kb':round(self.cache.memory_footprint_bytes()/1024,2)
   },
   'receptive_fields_tokens':LOD_RECEPTIVE_TOKENS
  }
