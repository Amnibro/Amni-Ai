from __future__ import annotations
import hashlib,json,time
from pathlib import Path
GATE_ACC=0.30
GATE_WRONG=0.10
HOLDOUT_N=800
HOLDOUT_CONTRACT="last_N_train_rows"
SOLVER_ID="amni.math.solve"
MATH_PY_NAMES=("__init__.py","harness.py","rules.py","planner.py","protocol.py","retrieve.py","sympy_tool.py","expr_search.py","metrics.py")
def norm_answer(a)->str:
 s=str(a or "").strip().replace(",","")
 try:
  f=float(s);return str(int(round(f))) if abs(f-round(f))<1e-8 else ("%.12g"%f)
 except Exception:return s
def empty_metrics()->dict:
 return {"n":0,"correct":0,"wrong":0,"declined":0,"accuracy":0.0,"wrong_rate":0.0,"decline_rate":0.0,"wall_s":0.0}
def finalize_counts(correct:int,wrong:int,declined:int,wall_s:float,n:int|None=None)->dict:
 n=correct+wrong+declined if n is None else n
 d=n if n else 1
 return {"n":n,"correct":correct,"wrong":wrong,"declined":declined,"accuracy":correct/d if n else 0.0,"wrong_rate":wrong/d if n else 0.0,"decline_rate":declined/d if n else 0.0,"wall_s":round(float(wall_s),2)}
def classify(pred,gold)->str:
 if pred is None:return "declined"
 return "correct" if norm_answer(pred)==norm_answer(gold) else "wrong"
def run_solver_rows(solve_fn,rows,*,get_q=None,get_gold=None,limit=None,progress_every=0,tag="")->dict:
 get_q=get_q or (lambda r:r.get("question") or r.get("problem") or "")
 get_gold=get_gold or (lambda r:r.get("gold") or r.get("answer") or "")
 rows=list(rows[:limit] if limit else rows)
 correct=wrong=declined=0;t0=time.time()
 for i,r in enumerate(rows):
  q=get_q(r);g=norm_answer(get_gold(r));out=solve_fn(q)
  pred=None if out is None else (out[0] if isinstance(out,tuple) else out)
  st=classify(pred,g)
  if st=="declined":declined+=1
  elif st=="correct":correct+=1
  else:wrong+=1
  if progress_every and (i+1)%progress_every==0:
   print(f"{tag} {i+1}/{len(rows)} acc={correct/(i+1):.3f} wrong={wrong/(i+1):.3f} dec={declined/(i+1):.3f}",flush=True)
 return finalize_counts(correct,wrong,declined,time.time()-t0,n=len(rows))
def sha256_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def sha256_file(path:Path)->str:return sha256_bytes(Path(path).read_bytes())
def fingerprint_math_dir(math_dir:Path,names:tuple[str,...]|None=None)->dict:
 math_dir=Path(math_dir);names=names or MATH_PY_NAMES
 files=[];h=hashlib.sha256()
 for name in sorted(names):
  p=math_dir/name
  if not p.is_file():continue
  raw=p.read_bytes();digest=sha256_bytes(raw)
  files.append({"name":name,"sha256":digest,"bytes":len(raw)})
  h.update(name.encode());h.update(b"\0");h.update(raw);h.update(b"\0")
 return {"math_dir":str(math_dir),"files":files,"fingerprint":h.hexdigest(),"n_files":len(files)}
def holdout_split(train_rows:list,holdout_n:int=HOLDOUT_N)->tuple[list,list]:
 if holdout_n<=0:raise ValueError("holdout_n must be >0")
 if len(train_rows)<holdout_n:raise ValueError(f"train n={len(train_rows)} < holdout_n={holdout_n}")
 return train_rows[:-holdout_n],train_rows[-holdout_n:]
def holdout_question_hash(hold_rows:list,get_q=None)->str:
 get_q=get_q or (lambda r:r.get("question") or r.get("problem") or "")
 h=hashlib.sha256()
 for r in hold_rows:
  h.update(get_q(r).encode());h.update(b"\0")
 return h.hexdigest()
def refuse_accidental_test(rows:list,*,label:str,train_n:int|None=None,claimed_split:str="")->None:
 n=len(rows);bad=[]
 if claimed_split.lower() in ("test","gsm8k_test") and train_n and n>=train_n:
  bad.append(f"{label}: refused full-train-sized set labeled test (n={n}, train_n={train_n})")
 if bad:raise SystemExit("REFUSE accidental test look: "+", ".join(bad))
def gate_probe(metrics:dict)->dict:
 acc=float(0 if metrics.get("accuracy") is None else metrics["accuracy"])
 wr=float(0 if metrics.get("wrong_rate") is None else metrics["wrong_rate"])
 return {"accuracy_ge_30":acc>=GATE_ACC,"wrong_le_10":wr<=GATE_WRONG,"gate_ready_candidate":acc>=GATE_ACC and wr<=GATE_WRONG}
def report_gate_fields(*,holdout:dict|None,test:dict|None,test_touched:bool,gate_ready_file:bool=False)->dict:
 hold_ok=bool(holdout) and float(0 if holdout.get("accuracy") is None else holdout["accuracy"])>=GATE_ACC and float(1 if holdout.get("wrong_rate") is None else holdout["wrong_rate"])<=GATE_WRONG
 gate_ready=bool(hold_ok);gate_pass=None
 if test_touched and test is not None:
  gate_pass=bool(float(0 if test.get("accuracy") is None else test["accuracy"])>=GATE_ACC and float(1 if test.get("wrong_rate") is None else test["wrong_rate"])<=GATE_WRONG)
 return {"gate_ready":gate_ready,"gate_pass":gate_pass,"test_touched":bool(test_touched),"gate_ready_file_present":bool(gate_ready_file),"thresholds":{"accuracy_min":GATE_ACC,"wrong_rate_max":GATE_WRONG,"holdout_n":HOLDOUT_N,"holdout_contract":HOLDOUT_CONTRACT}}
def load_gate_ready(path:Path)->dict:
 path=Path(path)
 if not path.is_file():raise SystemExit(f"REFUSE --test-run: missing gate_ready file at {path} (Keyak must set gate_ready.json after holdout ≥30% acc with wrong≤10%)")
 data=json.loads(path.read_text(encoding="utf-8"))
 if not data.get("gate_ready"):raise SystemExit(f"REFUSE --test-run: {path} has gate_ready!=true")
 if not data.get("authorized_by"):raise SystemExit(f"REFUSE --test-run: {path} missing authorized_by")
 return data
def require_test_run_authorized(*,test_run:bool,gate_ready_path:Path)->dict|None:
 if not test_run:return None
 return load_gate_ready(gate_ready_path)
def build_preregister(*,solver_id:str,math_fp:dict,train_path:Path|None,train_sha256:str|None,holdout_n:int,holdout_q_sha256:str|None,thresholds:dict|None=None,note:str="")->dict:
 thr=thresholds or {"accuracy_min":GATE_ACC,"wrong_rate_max":GATE_WRONG}
 return {"schema":"m08_preregister_v1","frozen_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"solver_id":solver_id,"math_fingerprint":math_fp.get("fingerprint"),"math_files":math_fp.get("files"),"holdout":{"contract":HOLDOUT_CONTRACT,"n":holdout_n,"train_path":str(train_path) if train_path else None,"train_sha256":train_sha256,"holdout_question_sha256":holdout_q_sha256},"thresholds":thr,"test_policy":{"one_look":True,"requires_gate_ready_json":True,"test_touched":False},"note":note or "Frozen BEFORE any gate claim. GSM8K test reserved until Keyak writes gate_ready.json."}
def write_json(path:Path,obj:dict)->Path:
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2),encoding="utf-8");return path
