"""1.2T virtual trits under 1.2 bit/param: T5 residual + per-chunk zlib, folded walk."""
from __future__ import annotations
import os,struct,zlib,json,time
import numpy as np
from amni.compute.ptex_1t_store import UNPACK_T5,TOTAL_PARAMS_1T,default_1t_path
MAGIC=b"PTEX_T5_1P2"+b"\0"*4
HDR="<16sQQII"
CHUNK_TRITS=1<<16
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=os.path.join(_H0,"exports","gf17_continuum","adam_1t_t5_1p2.bin")
OUT500=os.path.join(_H0,"exports","gf17_continuum","adam_500b_t5_1p2.bin")
SCORE=os.path.join(_H0,"exports","gf17_continuum","ptex_1t_1p2bit_scorecard.json")

def _codes_from_bytes(buf:bytes)->np.ndarray:
 a=np.frombuffer(buf,dtype=np.uint8)
 return (a%243).astype(np.uint8)

def _t5_stream(codes:np.ndarray)->np.ndarray:
 return UNPACK_T5[codes].reshape(-1).astype(np.int8)

def _signed_to_code(t:np.ndarray)->np.ndarray:
 return np.where(t<0,0,np.where(t==0,1,2)).astype(np.uint8)

def _residual(codes01:np.ndarray)->np.ndarray:
 if codes01.size==0:return codes01
 prev=np.empty_like(codes01)
 prev[0]=1
 prev[1:]=codes01[:-1]
 return ((codes01.astype(np.int16)-prev.astype(np.int16))%3).astype(np.uint8)

def _pack5(codes01:np.ndarray)->bytes:
 n=int(codes01.size)
 pad=(-n)%5
 if pad:codes01=np.concatenate([codes01,np.full(pad,1,dtype=np.uint8)])
 b=codes01.reshape(-1,5).astype(np.uint32)
 packed=(b[:,0]+3*b[:,1]+9*b[:,2]+27*b[:,3]+81*b[:,4]).astype(np.uint8)
 return packed.tobytes()

def _unpack5(blob:bytes,n:int)->np.ndarray:
 p=np.frombuffer(blob,dtype=np.uint8).astype(np.uint32)
 out=np.empty((p.size,5),dtype=np.uint8)
 rem=p.copy()
 for i in range(5):
  out[:,i]=(rem%3).astype(np.uint8);rem//=3
 return out.reshape(-1)[:n]

def pack_source(src_path:str,dest:str=OUT,chunk_trits:int=CHUNK_TRITS)->dict:
 t0=time.perf_counter()
 raw=open(src_path,"rb").read()
 # skip small header if this is a ptex
 if raw.startswith(b"PTEX_1T_RESIDENT"):
  raw=raw[struct.calcsize("<19sIQIIII")+3584*struct.calcsize("<IIIB"):]
 codes=_codes_from_bytes(raw)
 trits=_signed_to_code(_t5_stream(codes))
 res=_residual(trits)
 packed=_pack5(res)
 hist=np.bincount(res,minlength=3).astype(np.float64)
 p=hist/max(hist.sum(),1.0)
 p=np.clip(p,1e-12,1)
 H=float(-(p*np.log2(p)).sum())
 chunks=[]
 n=int(res.size)
 step=chunk_trits
 for i in range(0,n,step):
  sl=res[i:i+step]
  blob=zlib.compress(_pack5(sl),9)
  chunks.append(blob)
 index=bytearray()
 off=0
 blob=bytearray()
 for c in chunks:
  index.extend(struct.pack("<Q",off))
  blob.extend(c)
  off+=len(c)
 stored=16+struct.calcsize(HDR)+len(index)+len(blob)
 # HDR already includes magic; we write magic+fields
 body=struct.pack(HDR,MAGIC,TOTAL_PARAMS_1T,n,step,len(chunks))+bytes(index)+bytes(blob)
 os.makedirs(os.path.dirname(dest),exist_ok=True)
 open(dest,"wb").write(body)
 stored=os.path.getsize(dest)
 bp_virt=8.0*stored/TOTAL_PARAMS_1T
 bp_phys=8.0*stored/max(n,1)
 dt=time.perf_counter()-t0
 card={
  "virtual_params":TOTAL_PARAMS_1T,
  "physical_trits":n,
  "source_bytes":len(raw),
  "stored_bytes":stored,
  "chunk_trits":step,
  "n_chunks":len(chunks),
  "trit_entropy_bits":round(H,4),
  "p_res":[round(float(x),4) for x in p],
  "bits_per_virtual_param":bp_virt,
  "bits_per_physical_trit":round(bp_phys,6),
  "under_1_2_virtual":bp_virt<1.2,
  "under_1_2_physical":bp_phys<1.2,
  "path":dest,
  "elapsed_s":round(dt,3),
 }
 json.dump(card,open(SCORE,"w"),indent=1)
 return card

class T5FoldStore:
 def __init__(self,filepath:str=OUT):
  self.filepath=filepath
  self.fh=open(filepath,"rb")
  raw=self.fh.read(struct.calcsize(HDR))
  mag,self.n_virt,self.n_trits,self.chunk_trits,self.n_chunks=struct.unpack(HDR,raw)
  if not mag.startswith(b"PTEX_T5_1P2"):raise ValueError("bad T5 1.2-bit store")
  self.index=list(struct.unpack("<"+"Q"*self.n_chunks,self.fh.read(8*self.n_chunks)))
  self.blob_off=self.fh.tell()
  self._cache={}
 def close(self):
  if self.fh:self.fh.close();self.fh=None
 def _chunk(self,ci:int)->np.ndarray:
  if ci in self._cache:return self._cache[ci]
  if ci+1<self.n_chunks:end=self.index[ci+1]
  else:
   self.fh.seek(0,2);end=self.fh.tell()-self.blob_off
  self.fh.seek(self.blob_off+self.index[ci])
  blob=self.fh.read(end-self.index[ci])
  n=min(self.chunk_trits,self.n_trits-ci*self.chunk_trits)
  codes=_unpack5(zlib.decompress(blob),n)
  if len(self._cache)>=8:self._cache.pop(next(iter(self._cache)))
  self._cache[ci]=codes
  return codes
 def walk_virt(self,virt:int)->int:
  i=int(virt)%int(self.n_trits)
  ci=i//self.chunk_trits
  off=i%self.chunk_trits
  codes=self._chunk(ci)
  # undo residual for a local window so the byte is a real T5 pack
  # residual stream: r[0]=t[0]-1, r[k]=t[k]-t[k-1]
  # reconstruct t in this chunk from r, seed 1
  if not hasattr(self,"_recon") or self._recon_ci!=ci:
   t=np.empty_like(codes)
   acc=1
   for k,r in enumerate(codes):
    acc=(int(acc)+int(r))%3
    t[k]=acc
   self._recon=t;self._recon_ci=ci
  t=self._recon
  base=off-off%5
  sl=t[base:base+5]
  if sl.size<5:
   pad=np.full(5-sl.size,1,dtype=np.uint8);sl=np.concatenate([sl,pad])
  return int(sl[0]+3*sl[1]+9*sl[2]+27*sl[3]+81*sl[4])
