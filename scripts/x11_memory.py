import ast,json,re,sys,warnings
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from x0_code_eval import load_jsonl,problem_prompt,selfchecks
from x5_mine import pure,tile_src
from amni.compute import tile_synth as TS,code_steer as CS,eval_lock
TRAIN=("mbpp_holdout","apps_fn_dev","apps_fn_lock")
def main():
 warnings.simplefilter("ignore");rows=[r for f in TRAIN for r in load_jsonl(f)];eval_lock.assert_trainable(rows,"the X11 program memory");st=Counter();out=[]
 for r in rows:
  try:t=ast.parse(r.get("canonical") or "");fs=[n for n in t.body if isinstance(n,ast.FunctionDef)];assert len(fs)==1 and all(isinstance(n,(ast.FunctionDef,ast.Import,ast.ImportFrom)) for n in t.body)
  except Exception:st["not_single_function"]+=1;continue
  f=fs[0];f.body=[n for n in t.body if isinstance(n,(ast.Import,ast.ImportFrom))]+f.body;why,imp=pure(f)
  if why:st["impure:"+why.split(":")[0]]+=1;continue
  try:nm,src=tile_src(f,imp);TS.safe_check(TS._strip_imports(src))
  except Exception:st["unsafe"]+=1;continue
  p=problem_prompt(r,"code")
  try:sp=TS.Spec(p,selfchecks(p))
  except Exception:st["spec_fail"]+=1;continue
  st["kept"]+=1;out.append({"id":r["id"],"src":src,"words":CS.mwords(sp.desc+" "+sp.name),"g13":sorted(CS.shingles(sp.desc)),"fk":[sp.name,re.sub(r"\s+"," ",sp.desc)[:200]],"licence":"MIT (APPS)" if r["id"].startswith("apps") else "CC-BY-4.0 (MBPP)"})
 CS.MEMP.write_text("".join(json.dumps(o)+"\n" for o in out),encoding="utf-8");print(dict(st),len(out))
if __name__=="__main__":main()
