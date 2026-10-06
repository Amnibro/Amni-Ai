import re,sys,json,time,numpy as np,multiprocessing as mp
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from amni.compute import wikifact as WF
D=ROOT/"data"/"wikifact"/"v3"/"store";OUT=ROOT/"data"/"store1"/"results"/"parity_names_f0.json"
def keys(lines):
 o=[]
 for ln in lines:
  p=ln.rstrip("\n").split("\t",2)
  if len(p)<3:continue
  k,s=int(p[1]),p[2];x=WF.nrm(s)
  if not x:continue
  o.append(x);x.startswith("the ") and len(x)>4 and o.append(x[4:])
  k==2 and s.rstrip().endswith(")") and (y:=WF.nrm(re.sub(r"\s*\([^)]*\)\s*$","",s))) and o.append(y)
 return o
def _init():
 global DR,AH,AE,AK;from amni.compute.store1 import Drive;DR=Drive(fd=0);AH=np.load(D/"a_h.npy",mmap_mode="r");AE=np.load(D/"a_e.npy",mmap_mode="r");AK=np.load(D/"a_k.npy",mmap_mode="r")
def check(X):
 bad=[];H=[WF.h64(x) for x in X];R=DR.ents_many(H)
 for x,h,r in zip(X,H,R):
  i,j=int(np.searchsorted(AH,np.uint64(h),"left")),int(np.searchsorted(AH,np.uint64(h),"right"));want=list(zip(AE[i:j].tolist(),AK[i:j].tolist()));got=[]
  for e,k in r:
   N=DR.names(e);t=WF.nrm(re.sub(r"\s*\([^)]*\)\s*$","",DR.title(e)))
   if not (x in N or "the "+x in N or x==t):break
   got.append((e,k))
  got!=want and bad.append((x,len(want),len(got)))
 return len(X),bad
def main():
 t0=time.time();L=open(D/"names.tsv",encoding="utf-8",errors="replace").readlines();C=[L[i:i+200000] for i in range(0,len(L),200000)]
 with mp.Pool(3) as P:K=[x for c in P.imap(keys,C) for x in c]
 del L;U=list(dict.fromkeys(K));del K;hs={};col=0
 for x in U:h=WF.h64(x);col+=h in hs and hs[h]!=x;hs.setdefault(h,x)
 n=0;bad=[];B=[U[i:i+20000] for i in range(0,len(U),20000)]
 with mp.Pool(3,initializer=_init) as P:
  for a,b in P.imap_unordered(check,B):n+=a;bad+=b
 r={"fd":0,"name_keys_checked":n,"distinct_hashes":len(hs),"hash_collisions_distinct_strings":col,"parity_ents_mismatch":len(bad),"examples":bad[:20],"method":"chain read from the fd=0 table, each entity verified by decoding its names (names(e) contains the key, or 'the '+key, or the parenthesis-stripped title); kept as the maximal verified prefix","sec":round(time.time()-t0,1)}
 OUT.write_text(json.dumps(r,indent=1,ensure_ascii=False));print(json.dumps({k:v for k,v in r.items() if k!="examples"}))
if __name__=="__main__":main()
