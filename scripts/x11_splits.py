import argparse,ast,glob,json,random,re,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from x0_code_eval import DATA,save_jsonl,load_jsonl
from x10_splits import apps,_texts,_sig,_check,_cut,_oracle,_okp,prompt_of
from amni.compute import decontam,eval_lock
RAW10=Path.home()/".cache"/"amni_x10";RAW=Path.home()/".cache"/"amni_x11";TACO=str(RAW/"taco"/"*.parquet")
LOCKED=("apps_x11_lock","lcd_easy_lock");DEV=("mbpp_holdout","apps_fn_dev","lcb_easy_fn","apps_fn_lock");PRIOR=("humaneval","mbpp","mbpp_fresh","mbpp_holdout","apps_fn_dev","apps_fn_lock","lcb_easy_fn");MANI=ROOT/"docs"/"x11_locked_manifest.json";DEVMANI=ROOT/"docs"/"x11_dev_manifest.json"
SEED,LCD=1711,RAW/"lcd"
def kata(u):u=(u or "").lower().rstrip("/");m=re.search(r"codewars\.com/kata/([0-9a-f]{24}|[\w-]+)",u);return m.group(1) if m else u.split("://")[-1]
def apps_test_fn():
 for l in open(RAW10/"apps_test.jsonl",encoding="utf-8"):
  r=json.loads(l)
  try:io=json.loads(r["input_output"]) if r.get("input_output") else {}
  except Exception:io={}
  (r.pop("input_output",None),r.update(src_diff=r.get("difficulty"))) if io.get("fn_name") else None
  yield from ([(r,io)] if io.get("fn_name") else [])
def apps_urls():return {kata(json.loads(l).get("url")) for f in ("apps_train.jsonl","apps_test.jsonl") for l in open(RAW10/f,encoding="utf-8")}-{""}
def taco_rows(skip):
 import pyarrow.parquet as pq
 st={"seen":0,"fn":0,"dup_apps_url":0,"dup_url":0};seen=set();k=0
 for f in sorted(glob.glob(TACO)):
  for b in pq.ParquetFile(f).iter_batches(batch_size=256,columns=["question","solutions","starter_code","input_output","difficulty","source","url"]):
   for r in b.to_pylist():
    k+=1;st["seen"]+=1
    try:io=json.loads(r["input_output"]) if r.get("input_output") else {}
    except Exception:io={}
    if not io.get("fn_name"):continue
    st["fn"]+=1;u=kata(r.get("url"))
    if u in skip:st["dup_apps_url"]+=1;continue
    if u in seen:st["dup_url"]+=1;continue
    seen.add(u);r.pop("input_output",None);yield {**r,"id":k,"shard":Path(f).stem},io
 taco_rows.st=st
def _kw(src,params):
 c=ast.parse("f("+re.sub(r"\bnull\b","None",re.sub(r"\bfalse\b","False",re.sub(r"\btrue\b","True",src)))+")",mode="eval").body;d={k.arg:ast.literal_eval(k.value) for k in c.keywords};return [d[p] for p,_ in params] if set(d)==set(p for p,_ in params) and not c.args else None
def _hidden(test,params):
 out=[]
 for st in ast.parse(test).body[0].body if test.strip().startswith("def check") else []:
  try:
   t=st.test;c=t.left;assert isinstance(st,ast.Assert) and isinstance(t,ast.Compare) and isinstance(t.ops[0],ast.Eq) and isinstance(c,ast.Call) and getattr(c.func,"id","")=="candidate" and not c.args;d={k.arg:ast.literal_eval(k.value) for k in c.keywords};assert set(d)==set(p for p,_ in params);out.append(([d[p] for p,_ in params],ast.literal_eval(t.comparators[0])))
  except Exception:pass
 return out
def lcd(jobs=4):
 st={"seen":0,"skip_node":0,"skip_sig":0,"skip_ex":0,"skip_tests":0,"skip_oracle":0,"skip_prompt":0};todo=[]
 for f in ("LeetCodeDataset-train.jsonl","LeetCodeDataset-test.jsonl"):
  for l in open(LCD/f,encoding="utf-8"):
   r=json.loads(l)
   if r["difficulty"]!="Easy":continue
   st["seen"]+=1;sc=r["starter_code"]
   if re.search(r"ListNode|TreeNode|Node\b",sc):st["skip_node"]+=1;continue
   s=_sig(sc)
   if not s or s[3]:st["skip_sig"]+=1;continue
   exs=[]
   for m in re.finditer(r"Input:\s*(.+?)\s*\n\s*Output:",r["problem_description"]):
    try:a=_kw(m.group(1),s[1]);(exs.append(a) if a is not None and len(repr(a))<=300 else None)
    except Exception:pass
   if not exs:st["skip_ex"]+=1;continue
   ts=_hidden(r["test"],s[1])
   if len(ts)<5:st["skip_tests"]+=1;continue
   todo.append((r,s,_cut(ts),exs[:3]))
 with ThreadPoolExecutor(jobs) as ex:res=list(ex.map(lambda t:_oracle((t[0]["prompt"]+"\n"+t[0]["completion"],t[1][0],True,t[2],t[3])),todo))
 rows=[]
 for (r,s,ts,exs),(ok,outs) in zip(todo,res):
  if not ok or outs is None:st["skip_oracle"]+=1;continue
  name,params,ret,_=s;desc=re.sub(r"(?s)\n\s*Example\s*\d*:.*?(?=\n\s*Constraints:|\Z)","\n",r["problem_description"]).strip();pr=prompt_of(name,params,ret,desc,list(zip(exs,outs)))
  if not _okp(pr,len(exs)):st["skip_prompt"]+=1;continue
  rows.append({"id":f"lcd/{r['question_id']}","prompt":pr,"canonical":r["completion"],"test":_check(ts,False),"entry_point":name,"kind":"humaneval","src":"newfacade/LeetCodeDataset Easy (Apache-2.0)","title":r["task_id"],"date":r["estimated_date"],"n_public":len(exs),"n_tests":len(ts)})
 st["kept"]=len(rows);return rows,st
def fk(r):q=re.split(r'"""|\'\'\'',r["prompt"]);return (r["entry_point"],re.sub(r"\s+"," ",(q[1] if len(q)>1 else q[0]).split(">>>")[0]).strip()[:200])
def build(a):
 t0=time.time();pf=[DATA/f"{s}.jsonl" for s in PRIOR];base=decontam.index(files=pf,cache=None);pk={fk(r) for s in PRIOR if s not in ("mbpp","mbpp_holdout","mbpp_fresh") for r in load_jsonl(s)}
 meta={"built":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"seed":SEED,"prior_sets":list(PRIOR)}
 A,sa=apps(RAW10/"apps_train.jsonl","iv",a.jobs,diff="interview");B,sb=apps(None,"test",a.jobs,rows=apps_test_fn(),src="codeparrot/apps test call-based, all difficulties (MIT)")
 def clean(rs,ix,keys):
  out,ks=[],set()
  for r,t in zip(rs,_texts(rs)):
   k=fk(r)
   if k in keys or k in ks or decontam.dirty(t,ix=ix):continue
   ks.add(k);out.append(r)
  return out
 od=lambda r:(r["id"].split("/")[0],int(r["id"].split("/")[1]));AL=clean(A+B,base,pk);L,sl=lcd(a.jobs);LL=clean(L,base,pk|{fk(r) for r in AL})
 save_jsonl("_x11_tmp_locked",AL+LL);lk=decontam.index(files=[DATA/"_x11_tmp_locked.jsonl"],cache=None);(DATA/"_x11_tmp_locked.jsonl").unlink();dv=[r for s in DEV for r in load_jsonl(s)];dd=[r for r,t in zip(dv,_texts(dv)) if decontam.dirty(t,ix=lk)]
 (save_jsonl("_x11_tmp_dd",dd),setattr(build,"dx",decontam.index(files=[DATA/"_x11_tmp_dd.jsonl"],cache=None)),(DATA/"_x11_tmp_dd.jsonl").unlink()) if dd else None;rm=[r["id"] for r,t in zip(AL+LL,_texts(AL+LL)) if dd and decontam.dirty(t,ix=build.dx)];AL,LL=sorted([r for r in AL if r["id"] not in rm],key=od),sorted([r for r in LL if r["id"] not in rm],key=od)
 save_jsonl("apps_x11_lock",AL);save_jsonl("lcd_easy_lock",LL);(DATA/"lcd_easy_dev.jsonl").unlink(missing_ok=True)
 meta.update(apps_interview_train={**sa},apps_test_callbased={**sb},apps_locked=len(AL),lcd_easy={**sl,"locked":len(LL)},dev_rows_hitting_locked_13gram=[r["id"] for r in dd],locked_rows_removed_for_dev_overlap=rm,taco_note="BAAI/TACO (Apache-2.0) checked: 3,098 of its 3,298 call-based rows share a problem URL with APPS; the other 200 call-based rows (geeksforgeeks) give tests as C-style text with only the 2 statement examples, so no usable hidden tests; TACO not used",secs=round(time.time()-t0,1))
 (DATA/"x11_splits_meta.json").write_text(json.dumps(meta,indent=1));print(json.dumps(meta,indent=1))
def lock(a):m=eval_lock.lock([DATA/f"{s}.jsonl" for s in LOCKED],MANI);d=eval_lock.lock([DATA/f"{s}.jsonl" for s in DEV],DEVMANI);print(json.dumps({"locked":m,"dev":d},indent=1))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("cmd",choices=["build","lock","verify"]);ap.add_argument("--jobs",type=int,default=4);a=ap.parse_args()
 build(a) if a.cmd=="build" else lock(a) if a.cmd=="lock" else print(json.dumps({"locked_changed":eval_lock.verify(MANI),"dev_changed":eval_lock.verify(DEVMANI)}))
if __name__=="__main__":main()
