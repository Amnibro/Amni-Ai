import os,ctypes as C,functools,numpy as np
from pathlib import Path
from amni.compute import wikifact as WF
ROOT=Path(__file__).resolve().parents[2];DIR=Path(os.environ.get('AMNI_STORE1_DIR',str(ROOT/'data'/'store1'/'drive')));LIB=os.environ.get('AMNI_STORE1_LIB',str(ROOT/'data'/'store1'/'lib'/'libadam_store1.so'));FD=int(os.environ.get('AMNI_STORE1_FD','0'))
DIR2=Path(os.environ.get('AMNI_STORE2_DIR',str(ROOT/'data'/'store2'/'drive')));LIB2=os.environ.get('AMNI_STORE2_LIB',str(ROOT/'data'/'store2'/'lib'/'libadam_store1.so'));FD2=int(os.environ.get('AMNI_STORE2_FD','2'))
P8,P16,P32,P64=C.POINTER(C.c_uint8),C.POINTER(C.c_uint16),C.POINTER(C.c_uint32),C.POINTER(C.c_uint64)
_ptr=lambda a,t:a.ctypes.data_as(t)
class Drive:
 def __init__(s,d=DIR,fd=FD,lib=LIB,pre='s1'):
  L=s.L=C.CDLL(str(lib));V=C.c_void_p;s.pre=pre
  for n,a,r in (('open',[C.c_char_p,C.c_uint32],V),('close',[V],None),('ne',[V],C.c_uint64),('ents',[V,P64,C.c_size_t,P32,P32,P8,C.c_size_t],C.c_int64),('groups',[V,C.c_uint32,P16,P8,C.c_size_t],C.c_int64),('values',[V,C.c_uint32,C.c_uint16,P8,P64,P32,P8,P8,C.c_size_t],C.c_int64),('quals',[V,C.c_uint32,C.c_uint16,C.c_uint64,C.c_uint8,P16,P8,P64,C.c_size_t],C.c_int64),('text',[V,C.c_uint64,C.c_uint32,P8,C.c_size_t],C.c_int64),('sitelinks',[V,C.c_uint32],C.c_uint64),('qid',[V,C.c_uint32],C.c_uint32),('ix',[V,C.c_uint32],C.c_int64))+((('keys',[V,C.c_char_p,C.c_size_t,P64,C.c_size_t],C.c_int64),('pieces',[V,C.c_char_p,C.c_size_t,P32,C.c_size_t],C.c_int64),('clear',[V],None),('cold',[V],None)) if pre=='s2' else ()):f=getattr(L,f'{pre}_{n}');f.argtypes,f.restype=a,r;setattr(s,'_'+n,f)
  s.p=s._open(str(d).encode(),fd);s.fd=fd
  if not s.p:raise OSError(f"{pre}: cannot open {d} (fd {fd})")
  s.ne=int(s._ne(s.p));s.buf=np.zeros(1<<16,np.uint8);s.gb=(np.zeros(64,np.uint16),np.zeros(64,np.uint8));s.vb=(np.zeros(512,np.uint8),np.zeros(512,np.uint64),np.zeros(512,np.uint32),np.zeros(512,np.uint8),np.zeros(512,np.uint8));s.gp=(_ptr(s.gb[0],P16),_ptr(s.gb[1],P8));s.vp=tuple(_ptr(x,t) for x,t in zip(s.vb,(P8,P64,P32,P8,P8)));s.bp=_ptr(s.buf,P8)
 def keys_many(s,xs:list)->list:
  if s.pre!='s2':return [WF.h64(x) for x in xs]
  b='\n'.join(xs).encode();o=np.zeros(max(1,len(xs)),np.uint64);s._keys(s.p,b,len(b),_ptr(o,P64),len(o));return o[:len(xs)].tolist()
 def key(s,x:str)->int:return s.keys_many([x])[0]
 def pieces(s,x:str)->list:
  if s.pre!='s2':return []
  b=x.encode();o=np.zeros(4*len(b)+8,np.uint32);n=int(s._pieces(s.p,b,len(b),_ptr(o,P32),len(o)));return o[:n].tolist()
 def ents_many(s,hs)->list:
  h=np.ascontiguousarray(np.asarray(hs,np.uint64));cap=max(64,8*len(h));O=[[] for _ in h]
  while True:
   i,e,k=np.zeros(cap,np.uint32),np.zeros(cap,np.uint32),np.zeros(cap,np.uint8);n=int(s._ents(s.p,_ptr(h,P64),len(h),_ptr(i,P32),_ptr(e,P32),_ptr(k,P8),cap))
   if n<=cap:break
   cap=n
  [O[a].append((b,c)) for a,b,c in zip(i[:n].tolist(),e[:n].tolist(),k[:n].tolist())];return O
 def ents(s,x:str)->list:
  L=s.ents_many([s.key(x)])[0]
  if s.fd:return L
  o=[]
  for e,k in L:
   if not s.vfy(x,e):break
   o.append((e,k))
  return o
 def vfy(s,x:str,e:int)->bool:N=s.names(e);return x in N or 'the '+x in N or x==WF.nrm(__import__('re').sub(r"\s*\([^)]*\)\s*$","",s.title(e)))
 def groups(s,e:int)->list:
  p,f=s.gb;n=int(s._groups(s.p,int(e),*s.gp,64));return list(zip(p[:n].tolist(),f[:n].tolist()))
 def values(s,e:int,pn:int)->list:
  cap=512;k,v,x,pf,nq=s.vb;n=int(s._values(s.p,int(e),int(pn),*s.vp,cap));n=min(n,cap)
  return list(zip(k[:n].tolist(),v[:n].tolist(),x[:n].tolist(),pf[:n].tolist(),nq[:n].tolist()))
 def quals(s,e:int,pn:int,i:int,nq:int)->list:
  p,k,v=np.zeros(32,np.uint16),np.zeros(32,np.uint8),np.zeros(32,np.uint64);n=int(s._quals(s.p,int(e),int(pn),int(i),int(nq),_ptr(p,P16),_ptr(k,P8),_ptr(v,P64),32));return list(zip(p[:n].tolist(),k[:n].tolist(),v[:n].tolist()))
 def _t(s,e:int,w:int)->bytes:return s._tc(int(e),w)
 @functools.lru_cache(maxsize=65536)
 def _tc(s,e:int,w:int)->bytes:n=int(s._text(s.p,e,w,s.bp,len(s.buf)));return s.buf[:min(n,len(s.buf))].tobytes()
 def label(s,e:int)->str:return s._t(e,4 if s.pre=='s2' else 0).decode('utf-8','replace') if 0<=e<s.ne else ''
 def title(s,e:int)->str:return s._t(e,1).decode('utf-8','replace') if 0<=e<s.ne else ''
 def string(s,sid:int)->str:return s._t(sid,3).decode('utf-8','replace')
 def name(s,e:int)->str:return s.label(e) or __import__('re').sub(r"\s*\([^)]*\)$","",s.title(e))
 def names(s,e:int)->list:
  if not 0<=e<s.ne:return []
  b=s._t(e,2);l,t=s.label(e),s.title(e);nl=WF.nrm(l) if l else '';nt=WF.nrm(t) if t and t!=l else '';o=[];i=0
  while i<len(b):
   k=b[i];j=b.index(0xFF,i+1);o.append(nl if k==0 else nt if k==1 else b[i+1:j].decode('utf-8','replace'));i=j+1
  return list(dict.fromkeys(o))
 def sitelinks(s,e:int)->int:return int(s._sitelinks(s.p,int(e)))
 def qid(s,e:int)->int:return int(s._qid(s.p,int(e)))
 def ix(s,q:int)->int:return int(s._ix(s.p,int(q)))
 def clear(s,cold:bool=False):
  s._tc.cache_clear()
  if s.pre=='s2':(s._cold if cold else s._clear)(s.p)
 def close(s):s.p and s._close(s.p);s.p=None
@functools.lru_cache(maxsize=4)
def drive(fd:int=FD):
 try:return Drive(DIR,fd)
 except OSError:return None
@functools.lru_cache(maxsize=4)
def drive2(fd:int=FD2):
 try:return Drive(DIR2,fd,LIB2,'s2')
 except OSError:return None
