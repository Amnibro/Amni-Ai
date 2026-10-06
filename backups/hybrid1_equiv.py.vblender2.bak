import os,sys,json,gc,argparse,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)];os.environ["AMNI_BLENDER_NOTHR"]="1"
ap=argparse.ArgumentParser();ap.add_argument("--bake",required=True);ap.add_argument("--cfgs",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
from amni.serve import blender_atex as B
P=["User: Marlowe Bridge: Marlowe Bridge is a stone arch bridge over the River Tane in the county of Eastmere. It carries a single lane of traffic. What is Marlowe Bridge?\nAdam:","User: Calculator: The answer is 84. Working: 12 x 7 = 84. What is 12 times 7?\nAdam:","User: Hi Adam, long day at work and I am wiped out.\nAdam:"]
R={};base=None
for name,o in json.loads(a.cfgs).items():
    b=B.Blender("atex:"+a.bake,{**o,"sys":"split4"});out=[b.e.generate(b.prompt_ids(p),48,(),0.0) and list(b.e.last_ids) for p in P];txt=[b.text(p,144) for p in P]
    base=base or out;m=[sum(x==y for x,y in zip(u,v))/max(1,max(len(u),len(v))) for u,v in zip(out,base)]
    R[name]={"opts":o,"identical_to_first":all(u==v for u,v in zip(out,base)),"token_match_frac":[round(x,3) for x in m],"sample":txt[0]};print(name,R[name]["identical_to_first"],R[name]["token_match_frac"],flush=True)
    del b;gc.collect();torch.cuda.empty_cache()
json.dump(R,open(a.out,"w"),indent=1)
