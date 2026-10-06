import os,sys,json,gc,argparse,math,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)]
ap=argparse.ArgumentParser();ap.add_argument("--bakes",nargs="+",required=True);ap.add_argument("--opts",default="{}");ap.add_argument("--out",required=True);a=ap.parse_args()
from amni.serve import blender_atex as B
D=ROOT/"data"/"adam_t"/"eval"/"hybrid1_dev";T=" ".join(" ".join(json.loads(l)["material"])+" "+json.loads(l)["q"] for l in open(D/"blend_probe.jsonl") if json.loads(l)["material"])+" "+" ".join(json.loads(l)["q"] for l in open(D/"greet_vent_probe.jsonl"))
R={}
for b in a.bakes:
    e=B.Atex(b,**json.loads(a.opts));ids=e.tok(T,return_tensors="pt").input_ids[:,:1536]
    with torch.no_grad():
        h=e.base_model(inputs_embeds=e.emb(ids[0]).unsqueeze(0),use_cache=False).last_hidden_state[0]
        lg=torch.cat([torch.cat([e.logits(h[i+j]) for j in range(min(64,h.shape[0]-1-i))]) for i in range(0,h.shape[0]-1,64)])
        nll=torch.nn.functional.cross_entropy(lg[:ids.shape[1]-1],ids[0,1:].to(lg.device)).item()
    R[os.path.basename(b.rstrip("/"))]={"nll":round(nll,4),"ppl":round(math.exp(nll),3),"tokens":int(ids.shape[1]-1)};print(b,R[os.path.basename(b.rstrip("/"))],flush=True)
    del e,h,lg;gc.collect();torch.cuda.empty_cache()
json.dump(R,open(a.out,"w"),indent=1)
