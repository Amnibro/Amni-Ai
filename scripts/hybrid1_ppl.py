import os,sys,json,gc,argparse,math,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)]
from amni.serve import blender_atex as B
D=ROOT/"data"/"adam_t"/"eval"/"hybrid1_dev"
def text():return " ".join(" ".join(json.loads(l)["material"])+" "+json.loads(l)["q"] for l in open(D/"blend_probe.jsonl") if json.loads(l)["material"])+" "+" ".join(json.loads(l)["q"] for l in open(D/"greet_vent_probe.jsonl"))
@torch.no_grad()
def ppl(e,n=1536):
    ids=e.tok(text(),return_tensors="pt").input_ids[:,:n];h=e.prefill(ids,e.cache(),allh=True)[:-1];y=ids[0,1:].to(h.device);nll=sum(torch.nn.functional.cross_entropy(e.logits(h[i:i+64]),y[i:i+64],reduction="sum").item() for i in range(0,h.shape[0],64))/y.numel()
    return {"nll":round(nll,4),"ppl":round(math.exp(nll),3),"tokens":int(ids.shape[1]-1),"through":"engine prefill with the configured cache, chunking and tables"}
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--bakes",nargs="+",required=True);ap.add_argument("--opts",default="{}");ap.add_argument("--out",required=True);a=ap.parse_args();R={}
    for b in a.bakes:
        e=B.Atex(b,**json.loads(a.opts));R[os.path.basename(b.rstrip("/"))]=ppl(e);print(b,R[os.path.basename(b.rstrip("/"))],flush=True);del e;gc.collect();torch.cuda.empty_cache()
    json.dump(R,open(a.out,"w"),indent=1)
