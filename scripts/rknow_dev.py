import os,re,sys,json,argparse
import numpy as np,pyarrow as pa
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from amni.compute import decontam as DC
HF=Path.home()/".cache"/"huggingface"/"datasets"/"google-research-datasets___nq_open"/"nq_open"/"0.0.0"/"5dd9790a83002ad084ddeb7c420dc716852c6f28";OUT=ROOT/"data"/"concepts"/"eval"/"rknow";LIC="CC BY-SA 3.0 (Natural Questions open / google-research-datasets/nq_open, train split)"
NORM=lambda s:" ".join(DC.words(s))
def arrow(s:str)->list:return pa.ipc.open_stream(str(HF/f"nq_open-{s}.arrow")).read_all().to_pylist()
def locked()->list:
 m=json.loads((ROOT/"data"/"concepts"/"eval"/"nightly"/"nightly_lock.json").read_text());return [ROOT/p for p in m["splits"] if str(p).endswith((".json",".jsonl"))]
def build(n:int,seed:int,gram:int,fit:int=0,h2:int=0,h3:int=0,fx:int=0)->dict:
 tr=arrow("train");va=arrow("validation");L=locked();lq={NORM(x) for f in L for x in DC.leaves(DC.load(f))};vq={NORM(r["question"]) for r in va};ix=DC.index(n=gram,files=L,cache=None);dx=DC.index(files=[f for f in DC.eval_sources() if "rknow" not in f.parts],cache=None)
 o=np.random.default_rng(seed).permutation(len(tr)).tolist();keep=[];rej={"exact_locked":0,"exact_val":0,"gram_locked":0,"gram13_evalpool":0}
 for i in o:
  r=tr[i];q=r["question"];k=NORM(q);why="exact_locked" if k in lq else "exact_val" if k in vq else "gram_locked" if DC.dirty(q,None,ix) else "gram13_evalpool" if DC.dirty(q,None,dx) else None
  if why:rej[why]+=1;continue
  r["answer"] and keep.append({"q":q+"?" if not q.endswith("?") else q,"answers":r["answer"],"license":LIC,"source":f"nq_open/train#{i}"})
  if len(keep)>=2*n+fit+h2+h3+fx:break
 OUT.mkdir(parents=True,exist_ok=True);S={"tune":keep[:n],"hold":keep[n:2*n],**({"fit":keep[2*n:2*n+fit]} if fit else {}),**({"hold2":keep[2*n+fit:2*n+fit+h2]} if h2 else {}),**({"hold3":keep[2*n+fit+h2:2*n+fit+h2+h3]} if h3 else {}),**({"fitx":keep[2*n+fit+h2+h3:]} if fx else {})};[(OUT/f"nq_dev_{k}.json").write_text(json.dumps(v,indent=0)) for k,v in S.items()]
 chk={k:{"exact_locked":sum(NORM(x["q"]) in lq for x in v),"gram_locked":sum(bool(DC.dirty(x["q"],None,ix)) for x in v)} for k,v in S.items()}
 meta={"seed":seed,"n":n,"gram":gram,"pool":len(tr),"locked_files":[DC.rel(f) for f in L],"rejected":rej,"recheck":chk,"fit":fit,"hold2":h2,"hold3":h3,"fitx":fx,"disjoint":all(not ({NORM(x["q"]) for x in S[a]}&{NORM(x["q"]) for x in S[b]}) for a in S for b in S if a<b),"license":LIC}
 (OUT/"nq_dev_meta.json").write_text(json.dumps(meta,indent=1));return meta
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--n",type=int,default=400);ap.add_argument("--seed",type=int,default=7);ap.add_argument("--gram",type=int,default=5);ap.add_argument("--fit",type=int,default=2000);ap.add_argument("--hold2",type=int,default=400);ap.add_argument("--hold3",type=int,default=400);ap.add_argument("--fitx",type=int,default=6400);a=ap.parse_args();print(json.dumps(build(a.n,a.seed,a.gram,a.fit,a.hold2,a.hold3,a.fitx),indent=1))
