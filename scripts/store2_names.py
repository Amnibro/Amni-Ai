import re,sys,struct,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from amni.compute.wikifact import nrm,h64
D=ROOT/"data"/"wikifact"/"v3"/"store";OUT=ROOT/"data"/"store2"/"build"
def main():
 OUT.mkdir(parents=True,exist_ok=True);M={};n=0
 for ln in open(D/"names.tsv",encoding="utf-8",errors="replace"):
  p=ln.rstrip("\n").split("\t",2);n+=1
  if len(p)<3:continue
  k,s=int(p[1]),p[2];x=nrm(s)
  if not x:continue
  for v in [x]+([x[4:]] if x.startswith("the ") and len(x)>4 else [])+([nrm(re.sub(r"\s*\([^)]*\)\s*$","",s))] if k==2 and s.rstrip().endswith(")") else []):
   v and M.setdefault(h64(v),v)
 ah=np.load(D/"a_h.npy",mmap_mode="r");u=np.unique(np.asarray(ah));miss=0
 with open(OUT/"namekeys.bin","wb") as f:
  for h in u.tolist():
   s=M.get(h);miss+=s is None;b=(s or "").encode();f.write(struct.pack("<QH",h,len(b))+b)
 print({"lines":n,"strings":len(M),"keys":len(u),"missing":miss})
if __name__=="__main__":main()
