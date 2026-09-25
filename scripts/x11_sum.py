import json,sys,collections
from pathlib import Path
D=Path(__file__).resolve().parents[1]/"data"/"code_eval"
def summ(tag,ds="x11_dev"):
 it=json.load(open(D/f"report_{tag}_{ds}.json"))["items"];g=lambda r:r["id"].split("/")[0]
 out={}
 for k,rs in [("all",it)]+sorted(collections.defaultdict(list,{}).items()):pass
 grp=collections.defaultdict(list)
 for r in it:grp["all"].append(r);grp[g(r)].append(r)
 for k,rs in grp.items():
  n=len(rs);out[k]={"n":n,"pass":sum(r["ok"] for r in rs),"x9":sum(bool(r.get("oks")) and r["oks"][r["ts_x9_pick"]] if r.get("ts_x9_pick") is not None and r.get("oks") else r["ok"] for r in rs),"found":sum(bool(r.get("ts_found")) for r in rs),"no_cand":sum(not r.get("ts_found") for r in rs),"oracle":sum(bool(r.get("any_ok")) for r in rs),"reasons":dict(collections.Counter(r.get("ts_reason") for r in rs if not r.get("ts_found")))}
 return out
if __name__=="__main__":
 for t in sys.argv[1:]:
  tg,ds=(t.split(":")+["x11_dev"])[:2];print(tg,json.dumps(summ(tg,ds)))
