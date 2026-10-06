import os,sys,struct,numpy as np,multiprocessing as mp
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from amni.compute.wikifact import nrm
D=ROOT/"data"/"wikifact"/"v3"/"store";OUT=ROOT/"data"/"store1"/"build"
def _ld():
 global LO,TO,NO,LB,TB,NM;LO=np.load(D/"e_lo.npy");TO=np.load(D/"e_to.npy");NO=np.load(D/"nm_o.npy");LB=np.memmap(D/"e_lab.bin",np.uint8,"r");TB=np.memmap(D/"e_tit.bin",np.uint8,"r");NM=np.memmap(D/"nm.bin",np.uint8,"r")
def enc(rng):
 a,b=rng;o=bytearray();bad=0
 for e in range(a,b):
  l=bytes(LB[LO[e]:LO[e+1]]).decode("utf-8","replace");t=bytes(TB[TO[e]:TO[e+1]]).decode("utf-8","replace");L=[x for x in bytes(NM[NO[e]:NO[e+1]]).decode("utf-8","replace").split("\n") if x]
  nl=nrm(l) if l else "";nt=nrm(t) if t and t!=l else "";E=[(0,b"") if x==nl and nl else (1,b"") if x==nt and nt else (2,x.encode()) for x in L]
  rb=list(dict.fromkeys([nl if k==0 else nt if k==1 else x.decode() for k,x in E]));bad+=rb!=L
  o+=struct.pack("<H",len(E))
  for k,x in E:o+=struct.pack("<B",k)+(struct.pack("<H",len(x))+x if k==2 else b"")
 return bytes(o),bad
def main():
 _ld();n=len(LO)-1;OUT.mkdir(parents=True,exist_ok=True);R=[(i,min(n,i+200000)) for i in range(0,n,200000)];bad=0
 with mp.Pool(4,initializer=_ld) as P,open(OUT/"nmenc.bin","wb") as f:
  for b,x in P.imap(enc,R):f.write(b);bad+=x
 print({"entities":n,"reconstruct_mismatch":bad,"bytes":os.path.getsize(OUT/"nmenc.bin")})
if __name__=="__main__":main()
