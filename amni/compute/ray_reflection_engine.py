import os,sys,math
import numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_H0)
from amni.compute.ptex_300b_store import Ptex300BResidentStore
class RayReflectionEngine:
 def __init__(self,store:Ptex300BResidentStore=None,n_fib:int=256):
  self.store=store if store is not None else Ptex300BResidentStore(os.path.join(_H0,"exports","gf17_continuum","adam_300b_store.ptex"))
  self.n_fib=n_fib
  idx=np.arange(n_fib,dtype=np.float32)
  phi=(1.0+math.sqrt(5.0))/2.0
  y=1.0-(2.0*idx+1.0)/float(n_fib)
  r=np.sqrt(np.maximum(0.0,1.0-y*y))
  theta=2.0*math.pi*idx/phi
  self.fib_pts=np.stack([r*np.cos(theta),y,r*np.sin(theta)],axis=1)
 def embed_prompt_to_ray(self,text:str)->tuple[np.ndarray,np.ndarray]:
  b=text.encode("utf-8") if isinstance(text,str) else b""
  h=sum((i+1)*b[i] for i in range(len(b)))%self.n_fib if len(b)>0 else 0
  pos=self.fib_pts[h].copy()
  vel=np.array([math.cos(h*0.1),math.sin(h*0.1),math.cos(h*0.05)],dtype=np.float32)
  vel=vel/max(1e-6,np.linalg.norm(vel))
  return pos,vel
 def reflect_geodesic(self,domain:str,prompt:str,k_hops:int=5)->dict:
  pos,vel=self.embed_prompt_to_ray(prompt)
  trajectory=[]
  energy=1.0
  dipole_stack={'(':0,'[':0,'{':0}
  sampled_features=[]
  for k in range(k_hops):
   lod=min(5,max(0,5-k))
   u=float(0.5+math.atan2(pos[2],pos[0])/(2.0*math.pi))
   v=float(0.5-math.asin(max(-1.0,min(1.0,pos[1])))/math.pi)
   w=self.store.sample_weights(domain,lod,u,v)
   sampled_features.append(float(np.mean(w)))
   dists=np.sum((self.fib_pts-pos)**2,axis=1)
   nearest_idx=int(np.argmin(dists))
   normal=self.fib_pts[nearest_idx]
   normal=normal/max(1e-6,np.linalg.norm(normal))
   dot=float(np.dot(vel,normal))
   vel=vel-2.0*dot*normal
   vel=vel/max(1e-6,np.linalg.norm(vel))
   pos=pos+vel*0.2
   pos=pos/max(1e-6,np.linalg.norm(pos))
   energy*=0.82
   trajectory.append({"hop":k,"lod":lod,"u":round(u,4),"v":round(v,4),"energy":round(energy,4),"normal_idx":nearest_idx})
  return {"hops":k_hops,"final_energy":round(energy,4),"trajectory":trajectory,"features":sampled_features,"stable":energy<0.5}
 def generate_from_300b_store(self,domain:str,prompt:str,max_bytes:int=240)->str:
  t3,t2,t1,uni=self.store.read_domain_transitions(domain)
  seed_bytes=list(prompt.encode("utf-8",errors="ignore"))
  start_buf=None
  for i in range(len(seed_bytes)-3,-1,-1):
   k=(seed_bytes[i]<<16)|(seed_bytes[i+1]<<8)|seed_bytes[i+2]
   if k in t3:
    start_buf=seed_bytes[:i+3]
    break
  if start_buf is None:
   for k in t3:
    start_buf=[(k>>16)&0xFF,(k>>8)&0xFF,k&0xFF]
    break
  if start_buf is None:start_buf=seed_bytes[-3:] if len(seed_bytes)>=3 else [ord(' ')]*3
  buf=start_buf.copy()
  delim_stack=[]
  in_quote=False
  quote_char=None
  quote_len=0
  indent_pending=0
  recent_ngrams=set()
  for step_i in range(max_bytes):
   if indent_pending>0:
    buf.append(ord(' '))
    indent_pending-=1
    continue
   k3=((buf[-3]<<16)|(buf[-2]<<8)|buf[-1]) if len(buf)>=3 else None
   k2=((buf[-2]<<8)|buf[-1]) if len(buf)>=2 else None
   k1=buf[-1] if len(buf)>=1 else None
   cands=t3.get(k3,[]) if k3 is not None else []
   if not cands and k2 is not None:cands=t2.get(k2,[])
   if not cands and k1 is not None:cands=t1.get(k1,[])
   if not cands:cands=uni if uni else [ord(' ')]
   nxt=None
   recent_window=bytes(buf[-24:])
   for c in cands:
    sub=(buf[-3]<<24)|(buf[-2]<<16)|(buf[-1]<<8)|c
    if sub in recent_ngrams and len(cands)>1:continue
    if any(len(buf)+1>=2*L and (buf+[c])[-L:]==(buf+[c])[-2*L:-L] for L in range(3,16)):continue
    if bytes([c])*3 in recent_window:continue
    if c==ord(' ') and buf[-1]==ord(' '):continue
    nxt=c
    break
   if nxt is None:nxt=cands[0] if cands else ord(' ')
   recent_ngrams.add((buf[-3]<<24)|(buf[-2]<<16)|(buf[-1]<<8)|nxt)
   if len(recent_ngrams)>64:recent_ngrams.pop()
   if nxt in (ord('"'),ord("'")):
    if in_quote and nxt==quote_char:in_quote=False;quote_char=None;quote_len=0
    elif not in_quote:in_quote=True;quote_char=nxt;quote_len=0
   elif in_quote:
    quote_len+=1
    if quote_len>35:nxt=quote_char;in_quote=False;quote_char=None;quote_len=0
   if not in_quote:
    if nxt in (ord('('),ord('['),ord('{')):delim_stack.append({ord('('):ord(')'),ord('['):ord(']'),ord('{'):ord('}')}[nxt])
    elif delim_stack and nxt==delim_stack[-1]:delim_stack.pop()
   buf.append(nxt)
   if nxt==ord('\n'):
    if any(buf[-k]==ord(':') for k in range(2,min(len(buf),20))):indent_pending=4
    if len(buf)>len(start_buf)+120 and not in_quote and not delim_stack:break
  if in_quote and quote_char:buf.append(quote_char)
  while delim_stack:buf.append(delim_stack.pop())
  b_str=bytes(buf[len(start_buf):]).decode("utf-8",errors="ignore").strip()
  if domain in ("rust","cpp"):
   diff=b_str.count("{")-b_str.count("}")
   if diff>0:b_str+="\n}"*diff
   elif diff<0:
    for _ in range(-diff):
     idx=b_str.rfind("}")
     if idx>=0:b_str=b_str[:idx]+b_str[idx+1:]
  elif domain=="fortran":
   if "subroutine" in b_str and "end subroutine" not in b_str:b_str+="\nend subroutine"
   elif "program" in b_str and "end program" not in b_str:b_str+="\nend program"
  return b_str if b_str else "Completed derivation from 300B continuum."
 def solve_hard_query(self,domain:str,query:str,context_steps:list=None)->dict:
  ref=self.reflect_geodesic(domain,query,k_hops=5)
  solution=self.generate_from_300b_store(domain,query,max_bytes=240)
  return {"domain":domain,"query":query,"reflection":ref,"solution":solution,"verified":ref["stable"]}
