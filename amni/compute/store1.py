import os,ctypes as C,functools,numpy as np
from pathlib import Path
from amni.compute import wikifact as WF
ROOT=Path(__file__).resolve().parents[2];DIR=Path(os.environ.get('AMNI_STORE1_DIR',str(ROOT/'data'/'store1'/'drive')));LIB=os.environ.get('AMNI_STORE1_LIB',str(ROOT/'data'/'store1'/'lib'/'libadam_store1.so'));FD=int(os.environ.get('AMNI_STORE1_FD','0'))
P8,P16,P32,P64=C.POINTER(C.c_uint8),C.POINTER(C.c_uint16),C.POINTER(C.c_uint32),C.POINTER(C.c_uint64)
_ptr=lambda a,t:a.ctypes.data_as(t)
class Drive:
 def __init__(s,d=DIR,fd=FD,lib=LIB):
  L=s.L=C.CDLL(str(lib));V=C.c_void_p
  for n,a,r in (('s1_open',[C.c_char_p,C.c_uint32],V),('s1_close',[V],None),('s1_ne',[V],C.c_uint64),('s1_ents',[V,P64,C.c_size_t,P32,P32,P8,C.c_size_t],C.c_int64),('s1_groups',[V,C.c_uint32,P16,P8,C.c_size_t],C.c_int64),('s1_values',[V,C.c_uint32,C.c_uint16,P8,P64,P32,P8,P8,C.c_size_t],C.c_int64),('s1_quals',[V,C.c_uint32,C.c_uint16,C.c_uint64,C.c_uint8,P16,P8,P64,C.c_size_t],C.c_int64),('s1_text',[V,C.c_uint64,C.c_uint32,P8,C.c_size_t],C.c_int64),('s1_sitelinks',[V,C.c_uint32],C.c_uint64),('s1_qid',[V,C.c_uint32],C.c_uint32),('s1_ix',[V,C.c_uint32],C.c_int64)):f=getattr(L,n);f.argtypes,f.restype=a,r
  s.p=L.s1_open(str(d).encode(),fd);s.fd=fd
  if not s.p:raise OSError(f"store1: cannot open {d} (fd {fd})")
  s.ne=int(L.s1_ne(s.p));s.buf=np.zeros(1<<16,np.uint8);s.gb=(np.zeros(64,np.uint16),np.zeros(64,np.uint8));s.vb=(np.zeros(512,np.uint8),np.zeros(512,np.uint64),np.zeros(512,np.uint32),np.zeros(512,np.uint8),np.zeros(512,np.uint8))
 def ents_many(s,hs)->list:
  h=np.ascontiguousarray(np.asarray(hs,np.uint64));cap=max(64,8*len(h));O=[[] for _ in h]
  while True:
   i,e,k=np.zeros(cap,np.uint32),np.zeros(cap,np.uint32),np.zeros(cap,np.uint8);n=int(s.L.s1_ents(s.p,_ptr(h,P64),len(h),_ptr(i,P32),_ptr(e,P32),_ptr(k,P8),cap))
   if n<=cap:break
   cap=n
  [O[a].append((b,c)) for a,b,c in zip(i[:n].tolist(),e[:n].tolist(),k[:n].tolist())];return O
 def ents(s,x:str)->list:
  L=s.ents_many([WF.h64(x)])[0]
  if s.fd:return L
  o=[]
  for e,k in L:
   if not s.vfy(x,e):break
   o.append((e,k))
  return o
 def vfy(s,x:str,e:int)->bool:N=s.names(e);return x in N or 'the '+x in N or x==WF.nrm(__import__('re').sub(r"\s*\([^)]*\)\s*$","",s.title(e)))
 def groups(s,e:int)->list:
  p,f=s.gb;n=int(s.L.s1_groups(s.p,int(e),_ptr(p,P16),_ptr(f,P8),64));return list(zip(p[:n].tolist(),f[:n].tolist()))
 def values(s,e:int,pn:int)->list:
  cap=512;k,v,x,pf,nq=s.vb;n=int(s.L.s1_values(s.p,int(e),int(pn),_ptr(k,P8),_ptr(v,P64),_ptr(x,P32),_ptr(pf,P8),_ptr(nq,P8),cap));n=min(n,cap)
  return list(zip(k[:n].tolist(),v[:n].tolist(),x[:n].tolist(),pf[:n].tolist(),nq[:n].tolist()))
 def quals(s,e:int,pn:int,i:int,nq:int)->list:
  p,k,v=np.zeros(32,np.uint16),np.zeros(32,np.uint8),np.zeros(32,np.uint64);n=int(s.L.s1_quals(s.p,int(e),int(pn),int(i),int(nq),_ptr(p,P16),_ptr(k,P8),_ptr(v,P64),32));return list(zip(p[:n].tolist(),k[:n].tolist(),v[:n].tolist()))
 def _t(s,e:int,w:int)->bytes:return s._tc(int(e),w)
 @functools.lru_cache(maxsize=65536)
 def _tc(s,e:int,w:int)->bytes:n=int(s.L.s1_text(s.p,e,w,_ptr(s.buf,P8),len(s.buf)));return s.buf[:min(n,len(s.buf))].tobytes()
 def label(s,e:int)->str:return s._t(e,0).decode('utf-8','replace') if 0<=e<s.ne else ''
 def title(s,e:int)->str:return s._t(e,1).decode('utf-8','replace') if 0<=e<s.ne else ''
 def string(s,sid:int)->str:return s._t(sid,3).decode('utf-8','replace')
 def name(s,e:int)->str:return s.label(e) or __import__('re').sub(r"\s*\([^)]*\)$","",s.title(e))
 def names(s,e:int)->list:
  if not 0<=e<s.ne:return []
  b=s._t(e,2);l,t=s.label(e),s.title(e);nl=WF.nrm(l) if l else '';nt=WF.nrm(t) if t and t!=l else '';o=[];i=0
  while i<len(b):
   k=b[i];j=b.index(0xFF,i+1);o.append(nl if k==0 else nt if k==1 else b[i+1:j].decode('utf-8','replace'));i=j+1
  return list(dict.fromkeys(o))
 def sitelinks(s,e:int)->int:return int(s.L.s1_sitelinks(s.p,int(e)))
 def qid(s,e:int)->int:return int(s.L.s1_qid(s.p,int(e)))
 def ix(s,q:int)->int:return int(s.L.s1_ix(s.p,int(q)))
 def close(s):s.p and s.L.s1_close(s.p);s.p=None
@functools.lru_cache(maxsize=4)
def drive(fd:int=FD):
 try:return Drive(DIR,fd)
 except OSError:return None
