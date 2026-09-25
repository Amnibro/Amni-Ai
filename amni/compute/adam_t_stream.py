import os,sys,json,glob
import numpy as np
ROOT=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"data","adam_t","stream")
class Stream:
 def __init__(s,root:str=None,split:str="train",shards:int=None,concat:bool=False):
  s.root=root or os.environ.get("ADAM_T_STREAM",ROOT);mp=os.path.join(s.root,"manifest.json");s.man=json.load(open(mp)) if os.path.exists(mp) else {}
  fs=["train.bin"] if concat and split=="train" else [x["file"] for x in s.man.get("shards",[])] or sorted(os.path.basename(x) for x in glob.glob(os.path.join(s.root,"train_*.bin"))) if split=="train" else ["hold.bin"]
  s.files=fs[:shards] if shards else fs;s.mm=[np.memmap(os.path.join(s.root,f),np.uint8,"r") for f in s.files];s.n=np.array([len(m) for m in s.mm],np.int64);s.w=s.n/max(1,s.n.sum())
 def __len__(s)->int:return int(s.n.sum())
 def window(s,g:np.random.Generator,bs:int,t:int)->tuple:
  k=g.choice(len(s.mm),bs,p=s.w);o=(g.random(bs)*(s.n[k]-t-1)).astype(np.int64);x=np.stack([np.asarray(s.mm[a][b:b+t+1]) for a,b in zip(k,o)]);return x[:,:-1],x[:,1:]
 def windows(s,bs:int,t:int,seed:int=0,steps:int=None):
  g=np.random.default_rng(seed);i=0
  while steps is None or i<steps:yield s.window(g,bs,t);i+=1
 def docs(s,shard:int=0,meta:bool=False):
  f=s.files[shard];x=np.load(os.path.join(s.root,f[:-4]+".idx.npy"));m=s.mm[shard];pd={r.get("file"):r.get("pad",0) for r in s.man.get("shards",[])+[s.man.get("hold") or {}]};e=np.r_[x["off"][1:],len(m)-pd.get(f,0)].astype(np.int64);sr=s.man.get("sources",[])
  for j in range(len(x)):d=bytes(m[int(x["off"][j]):int(e[j])-2]);yield (d,{"src":sr[x["s"][j]] if sr else int(x["s"][j]),"store":int(x["st"][j]),"doc":int(x["d"][j])}) if meta else d
if __name__=="__main__":
 S=Stream(sys.argv[1] if len(sys.argv)>1 else None);H=Stream(S.root,"hold");x,y=next(S.windows(4,96,17));print(json.dumps({"root":S.root,"status":S.man.get("status"),"train_shards":len(S.files),"train_bytes":len(S),"hold_bytes":len(H),"mix":S.man.get("mix",{}).get("by_group")},indent=1));[print(repr(bytes(r).decode("utf-8","replace"))) for r in x]
