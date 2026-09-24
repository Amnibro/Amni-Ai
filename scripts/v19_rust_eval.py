import os,sys,json,time,re,numpy as np
from collections import Counter
_H0=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));H=os.environ.get("AMNI_HOME",_H0);sys.path[:0]=[_H0,os.path.join(_H0,"scripts"),os.path.join(H,"scripts")]
import adam_ray
import v19_eval_run as ev
P=sys.argv[1];P=P if P.endswith("_sid.npy") else P+"_sid.npy";P=P if os.path.isabs(P) or os.path.exists(P) else os.path.join(H,P);TAG=sys.argv[2] if len(sys.argv)>2 else "v19_rust";N=int(os.environ.get("EVAL_N","0"));CTX=int(os.environ.get("CTX","1048576"));B=os.path.join(H,"data",os.environ.get("LEX","big2"))
DEC,MAXB,LOCK,SS,REP,BW,VETO=os.environ.get("DEC","greedy"),int(os.environ.get("MAXB","90")),os.environ.get("LOCK","0")=="1",os.environ.get("SS","0")=="1",float(os.environ.get("REP","0")),int(os.environ.get("BEAM_W","4")),os.environ.get("VETO","0")=="1"
C3=os.path.join(H,"data","chat3","hold_eval.jsonl")
e=adam_ray.RayField(P,os.path.join(B,"synsets.txt"),os.path.join(B,"words.txt"),os.path.join(B,"wordcounts.txt"),os.environ.get("AMNI_RAY_ARCH","cpu"),H).load()
def gen(prompt:str,maxb:int=MAXB)->str:
 s=prompt.encode("ascii","replace")[-(CTX-maxb):];out=b""
 if DEC=="beam":return e.beam(s.decode("ascii"),BW,4,maxb,16,VETO,lock=LOCK,stop_sentence=SS).split("\n")[0].strip()
 if DEC=="stream":return "".join(e.stream(s.decode("ascii"),max_bytes=maxb,temp=1.0,topk=1,stop=["\n"],veto=VETO,lock=LOCK,stop_sentence=SS,rep=REP)).strip()
 lg=e.prime(s)[0]
 for _ in range(maxb):
  c=int(np.argmax(lg))
  if c==10 or c==30:break
  out+=bytes([c]);lg=e._advance([0],[c],True)[0]
 return out.decode("ascii","replace").strip()
_n=lambda s:re.sub(r"\s+"," ",(s or "").strip().lower())
bw=lambda v:r"(?<![A-Za-z0-9-])"+re.escape(v)+r"(?![A-Za-z0-9-])";ww=lambda v,t:re.search(bw(v),t)
SENT=re.compile(r"[.!?](?=\s|$)");NEG=re.compile(r"(?:\bnot|\binstead of|\brather than|\b(?:changed|switched|moved) (?:it )?from)\s+$")
FW=set("the a an and or of to in on for with is are was were be been being do does did what why how whom which when where who can could would should will may might must i i'm you you're your me my it it's its this that there their they them he he's she she's his her we our us as at by from about than then so if not no yes just also some any have has had haven't don't told know only".split())
VOC=sorted({r["answer"].strip() for f in (C3,os.path.join(str(ev.EVAL),"recall.jsonl"),os.path.join(str(ev.EVAL),"twohop.jsonl")) for r in map(json.loads,open(f)) if r["answer"]!="UNKNOWN" and len(r["answer"].strip())>=3})
def strict(pred:str,answer:str,dialogue:str,kind:str)->bool:
 p=_n(pred);m=SENT.search(p);f=p[:m.end()] if m else p;a=_n(answer)
 us=[l[5:].strip() for l in dialogue.split("\n") if l.startswith("User:")][:-1]
 cs=[_n(v) for v in VOC if any(re.search(bw(v),u) for u in us)];cs=[v for v in cs if a=="unknown" or (v not in a and a not in v)]
 ok=len(pred.encode())<=200 and max(Counter(w for w in re.findall(r"[a-z0-9']+",p) if w not in FW).values(),default=0)<3
 if a=="unknown":return ok and bool(ev.DECLINE.search(f)) and not any(ww(v,f) for v in cs)
 ex=set() if kind=="correction" else {v for v in cs if any(ww(v,_n(u)) and ww(a,_n(u)) for u in us)}
 return ok and bool(ww(a,f)) and not any(not NEG.search(p[:x.start()]) for v in cs if v not in ex for x in re.finditer(bw(v),p))
def rows(name:str,kind=None)->list:return [{**r,"kind":kind(r) if kind else r.get("kind")} for r in (json.loads(l) for l in open(name))]
def suite(rs:list,name:str)->dict:
 out=[];by={}
 for r in rs:
  p=gen(r["dialogue"]);h,s=ev.score_contains(p,r["answer"]),strict(p,r["answer"],r["dialogue"],r["kind"]);out.append({"kind":r["kind"],"pred":p,"answer":r["answer"],"hit":h,"strict":s});k=by.setdefault(r["kind"],[0,0,0]);k[0]+=int(h);k[1]+=int(s);k[2]+=1
 return {"suite":name,"n":len(out),"acc":sum(x["hit"] for x in out)/max(1,len(out)),"strict":sum(x["strict"] for x in out)/max(1,len(out)),"by_kind":{k:round(a/c,4) for k,(a,b,c) in by.items()},"strict_by_kind":{k:round(b/c,4) for k,(a,b,c) in by.items()},"n_by_kind":{k:c for k,(a,b,c) in by.items()},"items":out}
if __name__=="__main__":
 t=time.time();c3n=int(os.environ.get("C3_N","0"));c3=rows(C3);d=rows(os.path.join(str(ev.EVAL),"recall.jsonl"),lambda r:"unknown" if r["answer"]=="UNKNOWN" else "known")
 d=d[:N] if N else d;th=rows(os.path.join(str(ev.EVAL),"twohop.jsonl"),lambda r:"twohop");th=th[:N] if N else th
 rep={"chat3":suite(c3[:c3n] if c3n else c3,"chat3_hold"),"D":suite(d+th,"D"),"pack":P,"engine":f"adam_ray {e.backend} L={e.L} copy={e.copy}","decode":{"dec":DEC,"maxb":MAXB,"lock":LOCK,"stop_sentence":SS,"rep":REP,"beam_w":BW,"veto":VETO}}
 rep["wall_s"]=round(time.time()-t);dst=os.path.join(str(ev.EVAL),f"report_{TAG}.json");json.dump(rep,open(dst,"w"),indent=1)
 print(json.dumps({"chat3":[round(rep["chat3"]["acc"],4),round(rep["chat3"]["strict"],4)],"chat3_by_kind":{k:[v,rep["chat3"]["strict_by_kind"][k]] for k,v in rep["chat3"]["by_kind"].items()},"D":{k:[v,rep["D"]["strict_by_kind"][k]] for k,v in rep["D"]["by_kind"].items()},"decode":rep["decode"],"wall_s":rep["wall_s"],"dst":dst}))
