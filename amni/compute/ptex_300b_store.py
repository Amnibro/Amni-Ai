import os,sys,struct,math,mmap
import numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_H0)
MAGIC=b"PTEX_300B_RESIDENT\0"
HEADER_FMT="<19sIQIIII"
PAGE_ENTRY_FMT="<IIIB"
TOTAL_PARAMS_300B=300_000_000_000
TOTAL_PAGES_300B=895
PAGE_DIM=4096
DOMAINS={'math':(0,224),'stem':(224,448),'code':(448,672),'civics':(672,895)}
CODE_SUBRANGES={'rust':(448,504),'cpp':(504,560),'fortran':(560,616),'python':(616,672)}
UNPACK_T5=np.array([[((i//(3**k))%3)-1 for k in range(5)] for i in range(243)],dtype=np.int8)
class Ptex300BResidentStore:
 def __init__(self,filepath:str):
  self.filepath=filepath
  self.mmap_obj=None
  self.file_handle=None
  self.tile_cache={}
  self.max_cache_tiles=16
  self.total_pages=TOTAL_PAGES_300B
  self.page_dim=PAGE_DIM
  self.last_candidate_pages=[]
  self.is_initialized=os.path.exists(filepath) and os.path.getsize(filepath)>=struct.calcsize(HEADER_FMT)
  if not self.is_initialized:self._init_store_file()
  self._open_mmap()
 def _init_store_file(self):
  os.makedirs(os.path.dirname(self.filepath),exist_ok=True)
  header=struct.pack(HEADER_FMT,MAGIC,0x0614010C,TOTAL_PARAMS_300B,TOTAL_PAGES_300B,PAGE_DIM,4,len(DOMAINS))
  dir_bytes=bytearray()
  for p in range(TOTAL_PAGES_300B):
   dom_id=0 if p<224 else 1 if p<448 else 2 if p<672 else 3
   dir_bytes.extend(struct.pack(PAGE_ENTRY_FMT,p,p*4096,4096,dom_id))
  total_size=struct.calcsize(HEADER_FMT)+TOTAL_PAGES_300B*struct.calcsize(PAGE_ENTRY_FMT)+TOTAL_PAGES_300B*4096
  with open(self.filepath,"wb") as f:
   f.write(header)
   f.write(dir_bytes)
   rem=total_size-f.tell()
   chunk=b"\x20"*65536
   while rem>0:
    w_size=min(rem,len(chunk))
    f.write(chunk[:w_size])
    rem-=w_size
  self.is_initialized=True
 def _open_mmap(self):
  if not os.path.exists(self.filepath):return
  self.file_handle=open(self.filepath,"r+b")
  self.mmap_obj=mmap.mmap(self.file_handle.fileno(),0,access=mmap.ACCESS_READ)
 def close(self):
  if self.mmap_obj:self.mmap_obj.close();self.mmap_obj=None
  if self.file_handle:self.file_handle.close();self.file_handle=None
 def read_header(self)->dict:
  with open(self.filepath,"rb") as f:
   raw=f.read(struct.calcsize(HEADER_FMT))
  magic,ver,params,pages,dim,ch,n_dom=struct.unpack(HEADER_FMT,raw)
  return {"magic":magic.decode("ascii",errors="ignore").strip("\0"),"version":hex(ver),"total_params":params,"total_pages":pages,"page_dim":dim,"channels":ch,"domains":DOMAINS,"code_subranges":CODE_SUBRANGES,"file_size_bytes":os.path.getsize(self.filepath)}
 def get_page_offset(self,page_idx:int)->int:
  return struct.calcsize(HEADER_FMT)+TOTAL_PAGES_300B*struct.calcsize(PAGE_ENTRY_FMT)+page_idx*4096
 def read_tile_mmap(self,page_idx:int,tile_size:int=4096)->bytes:
  if page_idx in self.tile_cache:return self.tile_cache[page_idx]
  offset=self.get_page_offset(page_idx)
  if self.mmap_obj and offset+tile_size<=self.mmap_obj.size():
   tile_bytes=self.mmap_obj[offset:offset+tile_size]
  else:
   with open(self.filepath,"rb") as f:
    f.seek(offset if offset<os.path.getsize(self.filepath) else 0)
    tile_bytes=f.read(min(tile_size,os.path.getsize(self.filepath)-f.tell()))
  if len(self.tile_cache)>=self.max_cache_tiles:
   self.tile_cache.pop(next(iter(self.tile_cache)))
  self.tile_cache[page_idx]=tile_bytes
  return tile_bytes
 def sample_weights(self,domain:str,lod:int,u:float,v:float)->np.ndarray:
  r=CODE_SUBRANGES.get(domain,DOMAINS.get(domain,(0,224)))
  page_idx=r[0]+int((u%1.0)*(r[1]-r[0]))%max(1,r[1]-r[0])
  tile=self.read_tile_mmap(page_idx,tile_size=4096)
  byte_idx=int((v%1.0)*(len(tile)-1)) if len(tile)>1 else 0
  val=tile[byte_idx] if byte_idx<len(tile) else 0
  return UNPACK_T5[val%243]
 def write_page_data(self,page_idx:int,data:bytes):
  offset=self.get_page_offset(page_idx)
  self.close()
  with open(self.filepath,"r+b") as f:
   f.seek(offset)
   f.write(data)
  self._open_mmap()
 def get_domain_range(self,domain:str)->tuple[int,int]:
  return CODE_SUBRANGES.get(domain,DOMAINS.get(domain,(0,224)))
 def read_domain_transitions(self,domain:str)->tuple[dict,dict,dict,list]:
  r=self.get_domain_range(domain)
  trans3,trans2,trans1,unigrams={},{},{},[]
  for p in range(r[0],min(r[1],r[0]+16)):
   tile=self.read_tile_mmap(p,tile_size=4096)
   for i in range(0,len(tile)-4,4):
    a,b,c,d=tile[i],tile[i+1],tile[i+2],tile[i+3]
    k3=(a<<16)|(b<<8)|c
    k2=(b<<8)|c
    k1=c
    if k3 not in trans3:trans3[k3]=[]
    if d not in trans3[k3]:trans3[k3].append(d)
    if k2 not in trans2:trans2[k2]=[]
    if d not in trans2[k2]:trans2[k2].append(d)
    if k1 not in trans1:trans1[k1]=[]
    if d not in trans1[k1]:trans1[k1].append(d)
    if len(unigrams)<64 and d not in unigrams:unigrams.append(d)
  return trans3,trans2,trans1,unigrams
 def scan_domain_tiles(self,domain:str=None,keywords:list=None,max_matches:int=4,phrase:str=None)->list:
  if not keywords:return []
  kws=[k.lower().encode("latin1","ignore") for k in keywords if len(k)>=2]
  if not kws:return []
  ph=phrase.lower().encode("latin1","ignore") if phrase else b" ".join(kws)
  r=self.get_domain_range(domain) if domain and (domain in DOMAINS or domain in CODE_SUBRANGES) else (0,min(self.total_pages,TOTAL_PAGES_300B))
  matches=[]
  for p in range(r[0],r[1]):
   tile=self.read_tile_mmap(p,tile_size=self.page_dim)
   low=tile.lower()
   co_occur=sum(1 for k in kws if k in low)
   if co_occur==0:continue
   freq=sum(low.count(k)*len(k) for k in kws)
   score=co_occur*1000+freq
   if ph and ph in low:score+=15000
   first_idx=min(low.find(k) for k in kws if k in low)
   st_idx=max(0,first_idx-64)
   end_idx=min(len(tile),first_idx+1024)
   matches.append((score,p,tile[st_idx:end_idx]))
  matches.sort(key=lambda x:x[0],reverse=True)
  self.last_candidate_pages=[m[1] for m in matches[:max_matches]]
  return [m[2] for m in matches[:max_matches]]
