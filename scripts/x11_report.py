import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from x0_code_eval import DATA
from amni.compute import eval_lock
MANI=ROOT/"docs"/"x11_locked_manifest.json";CODE=[ROOT/"amni"/"compute"/f for f in ("tile_synth.py","code_rank.py","code_steer.py","eval_lock.py","decontam.py")]+[ROOT/"scripts"/f for f in ("x0_code_eval.py","x8_sketch.py","x5_mine.py","x11_mine.py","x11_steer.py","x11_report.py")]
def mcnemar(b,c):n=b+c;return 1.0 if n==0 else min(1.0,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n)
def items(tag,ds):return {r["id"]:r for r in json.load(open(DATA/f"report_{tag}_{ds}.json"))["items"]}
def x9ok(r):p=r.get("ts_x9_pick");return bool(r.get("oks")[p]) if p is not None and r.get("oks") else bool(r["ok"])
def part(A,B,ids):
 a=[A[i] for i in ids];b=[B[i] for i in ids];w=sum(y["ok"] and not x["ok"] for x,y in zip(a,b));l=sum(x["ok"] and not y["ok"] for x,y in zip(a,b))
 return {"n":len(ids),"x9":sum(map(x9ok,a)),"x10":sum(x["ok"] for x in a),"x11":sum(y["ok"] for y in b),"x11_wins":w,"x11_losses":l,"mcnemar_p":round(mcnemar(w,l),5),"no_cand_x10":sum(not x.get("ts_found") for x in a),"no_cand_x11":sum(not y.get("ts_found") for y in b),"oracle_x10":sum(bool(x.get("any_ok")) for x in a),"oracle_x11":sum(bool(y.get("any_ok")) for y in b),"found_wrong_x10":sum(bool(x.get("ts_found")) and not x["ok"] for x in a),"found_wrong_x11":sum(bool(y.get("ts_found")) and not y["ok"] for y in b),**({"he_plus_x10":sum(bool(x.get("ok_plus")) for x in a),"he_plus_x11":sum(bool(y.get("ok_plus")) for y in b)} if any("ok_plus" in x for x in a) else {})}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--x10",required=True);ap.add_argument("--x11",required=True);ap.add_argument("--sets",default="apps_x11_lock,lcd_easy_lock");ap.add_argument("--out",required=True);ap.add_argument("--kind",default="prod");ap.add_argument("--cfg",default="");a=ap.parse_args()
 res={}
 for ds in a.sets.split(","):A,B=items(a.x10,ds),items(a.x11,ds);assert set(A)==set(B),ds;res[ds]=part(A,B,sorted(A))
 if all(ds in res for ds in ("apps_x11_lock","lcd_easy_lock")):
  A={**items(a.x10,"apps_x11_lock"),**items(a.x10,"lcd_easy_lock")};B={**items(a.x11,"apps_x11_lock"),**items(a.x11,"lcd_easy_lock")};res["locked_total"]=part(A,B,sorted(A))
 print(json.dumps(res,indent=1));eval_lock.record(DATA/a.out,{"results":res,"x10_tag":a.x10,"x11_tag":a.x11,"x11_cfg":a.cfg},code_files=CODE,splits=json.loads(MANI.read_text()) if "lock" in a.sets else [DATA/f"{s}.jsonl" for s in a.sets.split(",")],kind=a.kind)
if __name__=="__main__":main()
