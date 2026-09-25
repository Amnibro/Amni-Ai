import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from x11_report import items,part
if __name__=="__main__":
 base=items(sys.argv[1],"x11_dev")
 for t in sys.argv[2:]:
  B=items(t,"x11_dev");ids=sorted(set(base)&set(B));r=part(base,B,ids);g=lambda p:{k:v for k,v in part({i:base[i] for i in ids if i.startswith(p)},{i:B[i] for i in ids if i.startswith(p)},[i for i in ids if i.startswith(p)]).items() if k in ("n","x10","x11","no_cand_x10","no_cand_x11","oracle_x10","oracle_x11")}
  print(t,{k:r[k] for k in ("n","x10","x11","x11_wins","x11_losses","no_cand_x10","no_cand_x11","oracle_x10","oracle_x11")},"| apps",g("apps"),"| lcb",g("lcb"))
