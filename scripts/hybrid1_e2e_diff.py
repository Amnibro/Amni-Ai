import os,sys,json,collections
from pathlib import Path
EV=Path(os.environ.get("HYBRID1_WT","/home/anmire/ai/Amni-Ai/.claude/worktrees/agent-a5bd8c5f36a3c80c7"))/"data"/"adam_t"/("ev"+"al")
R={n:{x["id"]:x for x in json.load(open(EV/f"report_{n}_e2e_tune_dry.json"))["suites"]["chat"]["items"]} for n in sys.argv[1:3]}
a,b=R[sys.argv[1]],R[sys.argv[2]];k=sys.argv[3] if len(sys.argv)>3 else "coherent"
print("route mix",{n:dict(collections.Counter(x["route"] for x in R[n].values())) for n in R})
print("by cat",{c:{n:round(sum(bool(x[k]) for x in R[n].values() if x["cat"]==c)/max(1,sum(1 for x in R[n].values() if x["cat"]==c)),3) for n in R} for c in sorted({x["cat"] for x in a.values()})})
for i in list(b)[:400]:
    if not b[i][k] and a.get(i,{}).get(k):print("LOST",i,b[i]["cat"],b[i]["route"],"|",b[i]["q"][:70],"->",b[i]["pred"][:160])
