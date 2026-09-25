import json,sys,warnings
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from x0_code_eval import DATA,load_jsonl,selfchecks,problem_prompt
from amni.compute import tile_synth as TS,code_rank as CR,eval_lock
warnings.simplefilter("ignore");rows={r["id"]:r for f in ("mbpp_holdout","apps_fn_dev") for r in load_jsonl(f)};fail,tot=Counter(),Counter();nf=nt=0
for f in ("report_x10_dev16_apps_fn_dev.json","report_x10_dev16_mbpp_holdout.json"):
 for it in json.load(open(DATA/f))["items"]:
  r=rows.get(it["id"])
  if r is None or eval_lock.eval_only(r):continue
  p=problem_prompt(r,"code")
  try:ops=CR.memory(TS.Spec(p,selfchecks(p))).get("op","want",[])
  except Exception:continue
  bad=it.get("ts_reason")!="found";nt+=1;nf+=bad
  for o in ops:tot[o]+=1;fail[o]+=bad
base=nf/nt;w={o:round(max(0.0,fail[o]/tot[o]-base)/base*min(1,tot[o]/30),3) for o in tot}
(DATA/"x11_opw.json").write_text(json.dumps(w,indent=1));print(base,sorted(w.items(),key=lambda x:-x[1])[:15])
