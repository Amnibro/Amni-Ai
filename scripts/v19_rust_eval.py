import os,sys,json,time,re,numpy as np
from collections import Counter
_H0=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));H=os.environ.get("AMNI_HOME",_H0);sys.path[:0]=[_H0,os.path.join(_H0,"scripts"),os.path.join(H,"scripts")]
import adam_ray
import v19_eval_run as ev
P=sys.argv[1];P=P if P.endswith("_sid.npy") else P+"_sid.npy";P=P if os.path.isabs(P) or os.path.exists(P) else os.path.join(H,P);TAG=sys.argv[2] if len(sys.argv)>2 else "v19_rust";N=int(os.environ.get("EVAL_N","0"));CTX=int(os.environ.get("CTX","1048576"));B=os.path.join(H,"data",os.environ.get("LEX","clean1"))
DEC,MAXB,LOCK,SS,REP,BW,VETO=os.environ.get("DEC","greedy"),int(os.environ.get("MAXB","90")),os.environ.get("LOCK","0")=="1",os.environ.get("SS","0")=="1",float(os.environ.get("REP","0")),int(os.environ.get("BEAM_W","4")),os.environ.get("VETO","0")=="1"
C3=os.path.join(H,"data","chat3","hold_eval.jsonl");D2F=os.path.join(str(ev.EVAL),"d2","d2.jsonl");D2L=os.path.join(str(ev.EVAL),"d2","d2_lock.json");SUITES=os.environ.get("SUITES","chat3,D").split(",")
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
def strict(pred:str,answer:str,dialogue:str,kind:str,voc=None)->bool:
 p=_n(pred);m=SENT.search(p);f=p[:m.end()] if m else p;a=_n(answer)
 us=[l[5:].strip() for l in dialogue.split("\n") if l.startswith("User:")][:-1]
 cs=[_n(v) for v in (voc or VOC) if any(re.search(bw(v),u) for u in us)];cs=[v for v in cs if a=="unknown" or (v not in a and a not in v)]
 ok=len(pred.encode())<=200 and max(Counter(w for w in re.findall(r"[a-z0-9']+",p) if w not in FW).values(),default=0)<3
 if a=="unknown":return ok and bool(ev.DECLINE.search(f)) and not any(ww(v,f) for v in cs)
 ex=set() if kind=="correction" else {v for v in cs if any(ww(v,_n(u)) and ww(a,_n(u)) for u in us)}
 return ok and bool(ww(a,f)) and not any(not NEG.search(p[:x.start()]) for v in cs if v not in ex for x in re.finditer(bw(v),p))
def rows(name:str,kind=None)->list:return [{**r,"kind":kind(r) if kind else r.get("kind")} for r in (json.loads(l) for l in open(name))]
def suite(rs:list,name:str,voc=None)->dict:
 out=[];by={}
 for r in rs:
  p=gen(r["dialogue"]);h,s=ev.score_contains(p,r["answer"]),strict(p,r["answer"],r["dialogue"],r["kind"],voc);out.append({"kind":r["kind"],"pred":p,"answer":r["answer"],"hit":h,"strict":s});k=by.setdefault(r["kind"],[0,0,0]);k[0]+=int(h);k[1]+=int(s);k[2]+=1
 G={"known":[k for k in by if k not in ("unknown","twohop")],"unknown":["unknown"],"twohop":["twohop"]}
 return {"suite":name,"n":len(out),"strict_groups":{g:round(sum(by[k][1] for k in ks)/max(1,sum(by[k][2] for k in ks)),4) for g,ks in G.items() if any(k in by for k in ks)},"acc":sum(x["hit"] for x in out)/max(1,len(out)),"strict":sum(x["strict"] for x in out)/max(1,len(out)),"by_kind":{k:round(a/c,4) for k,(a,b,c) in by.items()},"strict_by_kind":{k:round(b/c,4) for k,(a,b,c) in by.items()},"n_by_kind":{k:c for k,(a,b,c) in by.items()},"items":out}
def d2()->tuple:
 from amni.compute import eval_lock as EL
 bad=EL.verify(D2L);assert not bad,f"D2 split changed since lock: {bad}";rs=rows(D2F);return rs,sorted(set(VOC)|{v for r in rs for v in r["values"] if len(v)>=3})
if __name__=="__main__":
 t=time.time();c3n=int(os.environ.get("C3_N","0"));c3=rows(C3);d=rows(os.path.join(str(ev.EVAL),"recall.jsonl"),lambda r:"unknown" if r["answer"]=="UNKNOWN" else "known")
 d=d[:N] if N else d;th=rows(os.path.join(str(ev.EVAL),"twohop.jsonl"),lambda r:"twohop");th=th[:N] if N else th
 rep={**({"chat3":suite(c3[:c3n] if c3n else c3,"chat3_hold")} if "chat3" in SUITES else {}),**({"D":suite(d+th,"D")} if "D" in SUITES else {}),**({"D2":(lambda rv:suite(rv[0][:N] if N else rv[0],"D2",rv[1]))(d2())} if "D2" in SUITES else {}),"pack":P,"engine":f"adam_ray {e.backend} L={e.L} copy={e.copy}","decode":{"dec":DEC,"maxb":MAXB,"lock":LOCK,"stop_sentence":SS,"rep":REP,"beam_w":BW,"veto":VETO}}
 rep["wall_s"]=round(time.time()-t);dst=os.path.join(str(ev.EVAL),f"report_{TAG}.json");json.dump(rep,open(dst,"w"),indent=1)
 print(json.dumps({**{k:{"acc_strict":[round(rep[k]["acc"],4),round(rep[k]["strict"],4)],"by_kind":{c:[v,rep[k]["strict_by_kind"][c]] for c,v in rep[k]["by_kind"].items()},"groups":rep[k]["strict_groups"]} for k in ("chat3","D","D2") if k in rep},"decode":rep["decode"],"wall_s":rep["wall_s"],"dst":dst}))
