import os,re,sys,json,time,shlex,socket,argparse,resource,tempfile,subprocess,datetime as dt
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
from amni.compute import eval_lock as EL
from amni.compute.ray_field import DEFAULT_PACK,BASE_PACK,DEFAULT_LEX
EV=ROOT/"data"/"concepts"/"eval";NB=EV/"nightly";MAN=NB/"nightly_lock.json";HIST=EV/"nightly_history.json";F7=NB/"f7_smoke_transcript.json";NQ=EV/"v20_m13"/"nq_v20_m13.json"
SEED=20260925;NQ_N=200;BPB_BATCHES=64;THREADS=6;RSS_MAX=6<<30;AVAIL_MIN=12<<30;BUDGET=88*60;GB=1<<30
SPLITS=[ROOT/"data"/"chat3"/"hold_eval.jsonl",EV/"recall.jsonl",EV/"twohop.jsonl",ROOT/"data"/"clean1"/"hold.bin",ROOT/"data"/"clean2"/"hold.bin",NQ,F7]
CODE=[Path(__file__).resolve(),*[ROOT/p for p in ("scripts/v19_rust_eval.py","scripts/v19_eval_run.py","amni/serve/gguf_runtime.py","amni/serve/user_memory.py","amni/compute/ray_field.py","amni/compute/ray_learn.py","amni/compute/ray_qa.py","amni/compute/ks_retrieve.py","amni/compute/eval_lock.py")],*sorted(ROOT.glob("adam_ray.cpython-*.so"))]
MEM_ENV={"DEC":"stream","MAXB":"240","LOCK":"1","SS":"1","REP":"1.0","VETO":"0"};PHASES=(("mem",1800),("bpb",900),("chat",2700))
LOWER={"bpb_clean1":0.02,"bpb_clean2":0.02,"garbles":0};NOALERT={"nq_decline"}
FUNC=set("of the a an to and or with for in on at from by than instead as but is was".split());CITE=re.compile(r"\s*\((?:Source:|I learned)[^)]*\)\.?\s*$")
NORM=lambda s:" ".join(re.sub(r"\b(a|an|the)\b"," ",re.sub(r"[^\w\s]"," ",(s or "").lower())).split())
packs=lambda p:[p[:-8]+s for s in ("_sid.npy","_sw.npy","_dp.npy","_bias.npy")]
def netguard()->list:
 bl=[];oc,og=socket.socket.connect,socket.getaddrinfo;loc=lambda h:h is None or str(h).split("%")[0] in ("127.0.0.1","::1","localhost","")
 def deny(h):bl.append(str(h));raise ConnectionRefusedError(f"adam_nightly: network off ({h})")
 socket.socket.connect=lambda s,a,*k:oc(s,a,*k) if s.family==socket.AF_UNIX or loc(a[0] if isinstance(a,tuple) else a) else deny(a)
 socket.getaddrinfo=lambda h,*a,**k:og(h,*a,**k) if loc(h) else deny(h);return bl
def score_qa(pred:str,golds:list,decline)->dict:
 b=CITE.sub("",pred or "").strip();core=re.split(r"(?<=[.!?])\s+",b)[0] if b else "";d=bool(decline.search((pred or "").lower()));nb=NORM(b);gs=[g for g in map(NORM,golds) if g]
 return {"decline":d,"em":not d and NORM(core) in gs,"hit":not d and any(re.search(r"(?<!\S)"+re.escape(g)+r"(?!\S)",nb) for g in gs)}
def lexok(w:str,lex:set)->bool:return all(p in lex for p in re.split(r"-",w.lower().split("'")[0].split("’")[0]) if p)
def garbles(reply:str,values:list,lex:set)->list:
 out=[]
 for v in values:
  for m in re.finditer(r"(?<![A-Za-z0-9])"+re.escape(v),reply,re.I):
   nx=reply[m.end():m.end()+1];tail=re.split(r"[.!?\n]",reply[m.end():],maxsplit=1)[0];tk=re.findall(r"[A-Za-z][A-Za-z'’-]*",tail);bad=[w for w in tk if not lexok(w,lex)]
   why="glued" if re.match(r"[A-Za-z0-9]",nx) else "nonword:"+",".join(bad) if bad else "dangling:"+tk[-1] if tk and tk[-1].lower() in FUNC else None
   why and out.append({"value":v,"why":why,"tail":tail[:60]})
 return out
def phase_mem(limit:int)->dict:
 sys.argv=[sys.argv[0],DEFAULT_PACK,"nightly"];os.environ.update({**MEM_ENV,"LEX":DEFAULT_LEX})
 import v19_rust_eval as R
 c3=R.rows(R.C3);d=R.rows(os.path.join(str(R.ev.EVAL),"recall.jsonl"),lambda r:"unknown" if r["answer"]=="UNKNOWN" else "known");th=R.rows(os.path.join(str(R.ev.EVAL),"twohop.jsonl"),lambda r:"twohop");cut=lambda x:x[:limit] if limit else x
 return {"chat3":R.suite(cut(c3),"chat3_hold"),"D":R.suite(cut(d)+cut(th),"D"),"pack":R.P,"engine":f"adam_ray {R.e.backend} L={R.e.L} copy={R.e.copy}","decode":{"dec":R.DEC,"maxb":R.MAXB,"lock":R.LOCK,"stop_sentence":R.SS,"rep":R.REP,"beam_w":R.BW,"veto":R.VETO,"ctx":R.CTX,"lex":DEFAULT_LEX}}
def phase_bpb(limit:int)->dict:
 from amni.compute.ray_field import engine
 e=engine(BASE_PACK).load();nb=min(limit,BPB_BATCHES) if limit else BPB_BATCHES
 return {"pack":BASE_PACK,"arch":e.arch,"batches":nb,"ctx":128,"bs":256,"seed":SEED,"bytes":nb*128*256,**{f"bpb_{h}":round(float(e.bpb(hold=str(ROOT/"data"/h/"hold.bin"),batches=nb,seed=SEED)),4) for h in ("clean1","clean2")}}
def phase_chat(limit:int)->dict:
 from amni.serve import gguf_runtime as g
 from v19_eval_run import DECLINE
 nq=json.loads(NQ.read_text());ix=sorted(np.random.default_rng(SEED).choice(len(nq),min(NQ_N,len(nq)),replace=False).tolist());ix=ix[:limit] if limit else ix;qa=[];t0=time.time()
 for i in ix:t=time.time();a=g.chat(nq[i]["q"],do_sample=False,user="nightly-qa",session="nightly-qa")["answer"];qa.append({"i":i,"q":nq[i]["q"],"gold":nq[i]["answers"],"pred":a,**score_qa(a,nq[i]["answers"],DECLINE),"s":round(time.time()-t,2)})
 qs=time.time()-t0;tr=json.loads(F7.read_text());lex={l.strip().lower() for l in open(ROOT/"data"/DEFAULT_LEX/"words.txt",encoding="utf-8",errors="replace")};h=[];sm=[]
 for t in tr["turns"]:
  a=g.chat(t["u"],history=h,do_sample=False,user="nightly-f7",session="nightly-f7")["answer"];h.append((t["u"],a));e=t.get("expect")
  sm.append({"u":t["u"],"a":a,"expect":e,"hit":None if e is None else bool(DECLINE.search(a.lower())) if e=="UNKNOWN" else bool(re.search(r"(?<![A-Za-z0-9])"+re.escape(e.lower())+r"(?![A-Za-z0-9])",a.lower())),"garbles":garbles(a,tr["values"],lex),"nonwords":[w for w in re.findall(r"[A-Za-z][A-Za-z'’-]*",a) if not lexok(w,lex)]})
 n=max(1,len(qa));qn=[x for x in sm if x["hit"] is not None]
 return {"model":g.model_id(),"backend":g.backend_name(),"nq":{"split":EL.rel(NQ),"n":len(qa),"seed":SEED,"acc":sum(x["hit"] for x in qa)/n,"em":sum(x["em"] for x in qa)/n,"decline":sum(x["decline"] for x in qa)/n,"s_per_q":round(qs/n,2),"items":qa},"smoke":{"id":tr["id"],"turns":len(sm),"garbles":sum(len(x["garbles"]) for x in sm),"garbled_turns":sum(bool(x["garbles"]) for x in sm),"nonwords":sum(len(x["nonwords"]) for x in sm),"hits":sum(bool(x["hit"]) for x in qn),"asked":len(qn),"hit_rate":sum(bool(x["hit"]) for x in qn)/max(1,len(qn)),"items":sm}}
def flat(r:dict)->dict:
 c,d,b,q,s=(r.get(k) or {} for k in ("chat3","D","bpb","nq","smoke"))
 m={"chat3_strict":c.get("strict"),"D_strict":d.get("strict"),**{f"D_{k}":v for k,v in (d.get("strict_by_kind") or {}).items()},"nq_acc":q.get("acc"),"nq_em":q.get("em"),"nq_decline":q.get("decline"),"bpb_clean1":b.get("bpb_clean1"),"bpb_clean2":b.get("bpb_clean2"),"garbles":s.get("garbles"),"smoke_hits":s.get("hit_rate")}
 return {k:v for k,v in m.items() if v is not None}
def alerts(cur:dict,prev:dict)->list:return [{"metric":k,"prev":p,"curr":c,"delta":round(c-p,4)} for k,c in cur.items() for p in [(prev or {}).get(k)] if p is not None and k not in NOALERT and (round(c-p,6)>LOWER[k] if k in LOWER else round((p-c)*100,6)>2.0)]
def load_hist(p)->dict:
 try:h=json.loads(Path(p).read_text());return h if isinstance(h.get("runs"),list) else {**h,"runs":[]}
 except (OSError,ValueError,AttributeError):return {"runs":[]}
def save_hist(p,h:dict):p=Path(p);tmp=p.with_name(p.name+".tmp");tmp.write_text(json.dumps(h,indent=1));os.replace(tmp,p)
def finalize(rep:dict,man:dict,out_dir,hist,date:str,forced:bool=False,blockers=(),code=(),cmd=None,extra=None)->tuple:
 ch=EL.verify(man);kind="dry" if ch or blockers or forced else "prod";H=load_hist(hist);prev=next((r for r in reversed(H["runs"]) if r.get("kind","prod")=="prod" and r.get("date","")<date),None);m=flat(rep)
 al=alerts(m,prev and prev.get("m"))+[{"metric":"split_changed","files":sorted(ch)}]*bool(ch)+[{"metric":"blocker","why":b} for b in blockers]
 p=Path(out_dir)/f"report_nightly_{date}{'' if kind=='prod' else '_dry'}.json";r=EL.record(p,{**rep,"date":date,"m":m,"alerts":al,"blockers":list(blockers),"prev_date":prev and prev.get("date")},cmd,code,man,kind,extra);lk=r["eval_lock"]
 kind=="prod" and save_hist(hist,{**H,"runs":[x for x in H["runs"] if x.get("date")!=date]+[{"date":date,"kind":"prod","ts":lk["timestamp"],"report":EL.rel(p),"m":m,"alerts":al,"wall_s":rep.get("wall_s"),"pack":Path(rep.get("pack") or "").name,"base_pack":Path((rep.get("bpb") or {}).get("pack") or "").name,"git":lk["git"]["head"][:9],"code_hash":lk["code_hash"][:16],"splits_hash":lk["splits_hash"][:16]}]})
 return r,p
def _status(pid:int,key:str)->int:
 try:return next((int(l.split()[1])<<10 for l in open(f"/proc/{pid}/status") if l.startswith(key+":")),0)
 except OSError:return 0
def avail()->int:return next(int(l.split()[1])<<10 for l in open("/proc/meminfo") if l.startswith("MemAvailable:"))
def run_phase(name:str,env:dict,limit:int,deadline:float,timeout:int,log)->tuple:
 out=Path(env["NIGHTLY_TMP"])/f"{name}.json";t=time.time();why=None;peak=0;log.write(f"== phase {name} {time.strftime('%H:%M:%S')}\n");log.flush()
 p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--phase",name,"--out",str(out),"--limit",str(limit)],cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
 while p.poll() is None:
  peak=max(peak,_status(p.pid,"VmRSS"));why=why or ("rss>6GB" if peak>RSS_MAX else "MemAvailable<12GB" if avail()<AVAIL_MIN else "run budget 88min" if time.time()>deadline else f"phase timeout {timeout}s" if time.time()-t>timeout else None)
  why and p.poll() is None and (p.kill(),p.wait());time.sleep(1)
 res=json.loads(out.read_text()) if p.returncode==0 and out.exists() else None
 return res,{"s":round(time.time()-t,1),"peak_rss_mb":round(max(peak,(res or {}).get("maxrss_mb",0)*(1<<20))/(1<<20)),"rc":p.returncode,**({"killed":why} if why else {})}
def child_env(td:str)->dict:
 e={k:v for k,v in os.environ.items() if not k.startswith(("AMNI_","ADAM_")) and k not in ("TS_NAT_PACK","EVAL_N","C3_N","LEX","DEC","MAXB","LOCK","SS","REP","VETO","BEAM_W","CTX")}
 return {**e,"NIGHTLY_TMP":td,"AMNI_GEN_BACKEND":"ray","AMNI_RAY_ARCH":"cpu","AMNI_UMEM_DIR":os.path.join(td,"umem"),"AMNI_RAY_LEARNED":os.path.join(td,"learned.jsonl"),"AMNI_WIKI_CACHE":os.path.join(td,"wiki_cache"),"AMNI_LEARN_WEB":"0","AMNI_LEARN_LOCAL":"1","PYTHONUNBUFFERED":"1",**{k:str(THREADS) for k in ("RAYON_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_MAX_THREADS","TI_NUM_THREADS","POLARS_MAX_THREADS")}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--phase");ap.add_argument("--out");ap.add_argument("--limit",type=int,default=0);ap.add_argument("--dry",action="store_true");ap.add_argument("--date",default=dt.date.today().isoformat());ap.add_argument("--skip",default="");a=ap.parse_args()
 if a.phase:
  bl=netguard();r={"mem":phase_mem,"bpb":phase_bpb,"chat":phase_chat}[a.phase](a.limit);r["net_blocked"]=bl;r["maxrss_mb"]=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);Path(a.out).write_text(json.dumps(r));return
 os.nice(max(0,10-os.nice(0)));t0=time.time();NB.mkdir(parents=True,exist_ok=True);MAN.exists() or EL.lock(SPLITS,MAN);man=json.loads(MAN.read_text());deadline=t0+BUDGET;blockers=[];res={};stats={}
 next((1 for _ in range(60) if avail()>=AVAIL_MIN+RSS_MAX or time.sleep(10)),0)
 avail()<AVAIL_MIN+RSS_MAX and blockers.append(f"MemAvailable {avail()/GB:.1f}GB < 18GB after 10 min wait")
 with tempfile.TemporaryDirectory(prefix="adam_nightly_") as td,open(NB/f"run_{a.date}.log","w") as log:
  env=child_env(td)
  for name,to in ([] if blockers else PHASES):
   r,st=(None,{"skipped":True}) if name in a.skip.split(",") else run_phase(name,env,a.limit,deadline,to,log);stats[name]=st;res[name]=r or {};r is None and blockers.append(f"phase {name} failed: "+json.dumps(st))
 mem,bp,ch=res.get("mem",{}),res.get("bpb",{}),res.get("chat",{});nb=sorted({h for r in res.values() for h in r.get("net_blocked",[])})
 nb and blockers.append("network attempts blocked: "+",".join(nb[:5]))
 rep={"chat3":mem.get("chat3"),"D":mem.get("D"),"pack":mem.get("pack") or DEFAULT_PACK,"engine":mem.get("engine"),"decode":mem.get("decode"),"bpb":bp or None,"nq":ch.get("nq"),"smoke":ch.get("smoke"),"serve":{"model":ch.get("model"),"backend":ch.get("backend"),"pack":DEFAULT_PACK,"base_pack":BASE_PACK,"lex":DEFAULT_LEX,"web":"off (AMNI_LEARN_WEB=0 + socket guard)","do_sample":False},"resources":{"threads":THREADS,"nice":os.nice(0),"phases":stats,"peak_rss_mb":max([s.get("peak_rss_mb",0) for s in stats.values()] or [0]),"mem_avail_gb_end":round(avail()/GB,1)},"wall_s":round(time.time()-t0)}
 cmd=shlex.join([sys.executable]+sys.argv);r,p=finalize(rep,man,EV,HIST,a.date,a.dry or a.limit>0 or bool(a.skip),blockers,CODE,cmd,{"packs":EL.file_hashes(packs(DEFAULT_PACK)+packs(BASE_PACK))})
 print(json.dumps({"report":EL.rel(p),"kind":r["kind"],"wall_s":rep["wall_s"],"m":r["m"],"alerts":r["alerts"],"peak_rss_mb":rep["resources"]["peak_rss_mb"]},indent=1))
if __name__=="__main__":main()
