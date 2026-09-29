import os,sys,json,time,argparse,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import adam_t_eval as AE
CS=ROOT/"data"/"adam_t"/"eval"/"commonsense"
K=("answered","coherent","decline","echo","on_topic","kw_any","fact","grounded")
def summ(G:list)->dict:
    f=lambda X,k:round(sum(bool(x[k]) for x in X if x.get(k) is not None)/max(1,sum(x.get(k) is not None for x in X)),4)
    return {"n":len(G),**{k:f(G,k) for k in K if any(x.get(k) is not None for x in G)},"routes":dict(collections.Counter(x["route"] for x in G)),"median_ms":round(float(np.median([x["ms"] for x in G])),1) if G else None}
def main():
    ap=argparse.ArgumentParser(description="ADAM-T-COMMONSENSE dev set through the live route (gguf_runtime.chat with AMNI_ADAM_T)");ap.add_argument("--split",default="tune",choices=("tune","test"));ap.add_argument("--proto-cmd",required=True);ap.add_argument("--label",required=True);ap.add_argument("--limit",type=int,default=0);ap.add_argument("--ids",default="");ap.add_argument("--out",default="");a=ap.parse_args()
    it=[json.loads(l) for l in open(CS/f"cs_{a.split}.jsonl")];ids=set(filter(None,a.ids.split(",")));it=[r for r in it if not ids or r["id"] in ids][:a.limit or None];M=AE.E2E(a.proto_cmd);H=__import__("amni.serve.adam_t_harness",fromlist=["run"]);orun=H.run;H.run=lambda *z,**k:(lambda o:(M.cap.__setitem__("slots",(o or {}).get("slots") or []),o)[1])(orun(*z,**k));t0=time.time();G=[]
    for j,r in enumerate(it):
        x=M.item(r);sl=(M.cap.get("slots") or []);x.update(kw_any=bool(AE.kwhit(AE.CHT(x["pred"]),r["kw"])) if r["kw"] else None,grounded=any(s.get("grounded") for s in sl) if r["cat"]=="explain" else None,prov=[p for s in sl for p in (s.get("prov") or [])][:4]);G.append(x);(j+1)%20 or print(f"  {j+1}/{len(it)} {time.time()-t0:.0f}s",flush=True)
    M.close();R={"label":a.label,"split":a.split,"env":{k:os.environ.get(k) for k in ("AMNI_ADAM_T_HARNESS","AMNI_COGNITION","AMNI_ADAM_T_HARNESS_EXPLAIN","AMNI_ADAM_T_HARNESS_CS")},"adapter":M.info,"all":summ(G),"by_cat":{c:summ([x for x in G if x["cat"]==c]) for c in ("explain","know")},"wall_s":round(time.time()-t0,1),"items":[{k:x.get(k) for k in ("id","cat","q","pred","route","answered","coherent","fact","kw_any","grounded","decline","prov","ms")} for x in G]}
    p=Path(a.out) if a.out else CS/f"report_{a.label}_{a.split}.json";p.write_text(json.dumps(R,indent=1));print(json.dumps({k:R[k] for k in ("label","split","all","by_cat","wall_s")},indent=1))
if __name__=="__main__":main()
