import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.serve import pipe1 as P
def main():
    ap=argparse.ArgumentParser(description="PIPE-1 Proving Ground quarantine: list, promote (admin endorse), reject, compact or forget global corrections");ap.add_argument("cmd",choices=("list","promote","reject","compact","forget"));ap.add_argument("id",nargs="?",default="");a=ap.parse_args()
    if a.cmd=="list":[print(json.dumps({k:x.get(k) for k in ("id","tier","coord","v","old","safety","ts")})) for x in P.qload()];return
    r=P.promote(a.id) if a.cmd=="promote" else P.reject(a.id) if a.cmd=="reject" else {"touched":P.gcompact()} if a.cmd=="compact" else {"ok":any(P.gforget(k) for x in P.qload() if x["id"]==a.id for k in x["keys"][:1])};print(json.dumps(r))
if __name__=="__main__":main()
