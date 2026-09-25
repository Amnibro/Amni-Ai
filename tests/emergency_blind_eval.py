import sys,json,time,tracemalloc
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from amni.serve import emergency as E
from amni.compute import eval_lock as EL
D=ROOT/"tests"/"data"/"emergency";LK=D/(sys.argv[2] if len(sys.argv)>2 else "lock_blind567.json");NM=LK.stem.replace("lock_","");OUT=Path(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1]!="-" else ROOT/"data"/"safety"/f"report_emergency_{NM}.json"
M=json.loads(LK.read_text());assert not EL.verify(M),EL.verify(M)
R=[(Path(f).name.split("_")[0],"pos" in Path(f).name,*l.rstrip("\n").split("\t",1)) for f in sorted(M["splits"]) for l in open(ROOT/f,encoding="utf-8") if l.strip()]
tracemalloc.start();E._lut();mem=tracemalloc.get_traced_memory();tracemalloc.stop();tm=json.loads(str(__import__("numpy").load(E.LUTP)["meta"]));TV=tm["th_veto"]
t=time.perf_counter();O=[(E.rules(m),E.classify(m)) for *_,m in R];ms=(time.perf_counter()-t)*1000/len(R)
t=time.perf_counter();DT=[E.detect.__wrapped__(m) for *_,m in R];ms_d=(time.perf_counter()-t)*1000/len(R)
def sc(m):z=E._lut();return E._sc(z[0],E._ix(E.efeats(m)))[0]
S=[sc(m) for *_,m in R]
SY={"rules":lambda r,c,s:bool(r),"lut":lambda r,c,s:bool(c),"rules_or_lut":lambda r,c,s:bool(r or c),"veto":lambda r,c,s:bool(c or r and s>=TV)}
def cat(r,c):return (r or c or {}).get("category")
PR="first look: code, LUT and thresholds frozen in git before these sets were read; primary system = detect() as shipped (rules with veto on, suicide exempt from veto, OR the LUT with its concern band); a concern-tier reply counts as a hit for recall AND as a false positive on near-misses; secondary = emergency-tier-only FP, rules alone, LUT alone, rules_or_lut"
rep={"protocol":PR if NM!="blind567" else "first look: model and thresholds frozen in git before these sets were read; primary system = rules_or_lut; veto = lut OR (rules AND lut_score>=th_veto), pre-registered secondary","lut_meta":tm,"n_pos":sum(p for _,p,*_ in R),"n_neg":sum(not p for _,p,*_ in R),"ms_per_msg_rules_plus_lut":round(ms,3),"ms_per_msg_detect":round(ms_d,3),"lut_mb_resident":round(E._lut()[0].nbytes/1e6,2),"lut_mb_peak_load":round(mem[1]/1e6,2),"lut_file_kb":round(E.LUTP.stat().st_size/1e3,1)}
SY={"detect":None,**SY} if NM!="blind567" else SY
for k,f in SY.items():
    h=[bool(d) for d in DT] if f is None else [f(r,c,s) for (r,c),s in zip(O,S)]
    per={st:{"recall":round(sum(x for x,(g,p,*_) in zip(h,R) if g==st and p)/sum(1 for g,p,*_ in R if g==st and p),4),"fp":round(sum(x for x,(g,p,*_) in zip(h,R) if g==st and not p)/sum(1 for g,p,*_ in R if g==st and not p),4)} for st in sorted({g for g,*_ in R})}
    pc=Counter(c.split("-")[0] for (_,p,c,_) in R if p);hc=Counter(c.split("-")[0] for x,(_,p,c,_) in zip(h,R) if p and x);nc=Counter(c for (_,p,c,_) in R if not p);fc=Counter(c for x,(_,p,c,_) in zip(h,R) if not p and x)
    rep[k]={"recall":round(sum(x for x,(_,p,*_) in zip(h,R) if p)/rep["n_pos"],4),"fp":round(sum(x for x,(_,p,*_) in zip(h,R) if not p)/rep["n_neg"],4),"per_set":per,"recall_by_cat":{c:round(hc[c]/pc[c],4) for c in sorted(pc)},"fp_by_nearmiss":{c:f"{fc[c]}/{nc[c]}" for c in sorted(nc)},"cat_acc_on_hits":round(sum((d["category"] if f is None else cat(r,c))==lab.split("-")[0] for x,(r,c),d,(_,p,lab,_) in zip(h,O,DT,R) if p and x)/max(1,sum(x for x,(_,p,*_) in zip(h,R) if p)),4)}
    ne=[x for x,(_,p,_,m) in zip(h,R) if p and not m.isascii()];rep[k]|={"recall_suicide_direct":round(sum(x for x,(_,p,c,_) in zip(h,R) if p and c=="suicide")/max(1,sum(1 for _,p,c,_ in R if p and c=="suicide")),4),"recall_suicide_indirect":round(sum(x for x,(_,p,c,_) in zip(h,R) if p and c=="suicide-indirect")/max(1,sum(1 for _,p,c,_ in R if p and c=="suicide-indirect")),4),"recall_non_ascii":f"{sum(ne)}/{len(ne)}"} if NM!="blind567" else {}
import numpy as np
SA=np.array(S);YA=np.array([p for _,p,*_ in R]);RU=np.array([bool(r) for r,_ in O]);rep["oracle_threshold_curves"]={f"fp{f}":{"lut_alone_recall":round(float((SA[YA]>=np.quantile(SA[~YA],1-f)).mean()),4),"rules_or_lut_recall":round(float((RU|(SA>=np.quantile(SA[~YA],1-f)))[YA].mean()),4),"rules_or_lut_fp":round(float((RU|(SA>=np.quantile(SA[~YA],1-f)))[~YA].mean()),4)} for f in (0.01,0.03,0.05,0.1)}
P0=[(d,(g,p,c,m)) for d,(g,p,c,m) in zip(DT,R)] if NM!="blind567" else []
rep|={"detect_fp_emergency_only":round(sum(1 for d,(g,p,c,m) in P0 if not p and d and d["tier"]=="emergency")/rep["n_neg"],4),"detect_fp_concern_only":round(sum(1 for d,(g,p,c,m) in P0 if not p and d and d["tier"]=="concern")/rep["n_neg"],4),"detect_nearmiss_confusion":{c:dict(Counter(d["kind"] for d,(g,p,cc,m) in P0 if not p and cc==c and d)) for c in sorted({c for _,p,c,_ in R if not p})},"detect_tier_on_positives":{f"{c}|{t}":n for (c,t),n in Counter((c,d["tier"] if d else "miss") for d,(g,p,c,m) in P0 if p).items()},"detect_fn":[(g,c,m) for d,(g,p,c,m) in P0 if p and not d],"detect_fp":[(g,c,m,d["kind"],d["tier"],d.get("src","rules")) for d,(g,p,c,m) in P0 if not p and d]} if P0 else {}
rep["fn_primary"]=[(g,c,m) for (r,cc),(g,p,c,m) in zip(O,R) if p and not (r or cc)];rep["fp_primary"]=[(g,c,m,"rules:"+r["kind"] if r else "lut:%.2f"%cc["score"]) for (r,cc),(g,p,c,m) in zip(O,R) if not p and (r or cc)]
EL.record(OUT,rep,cmd="python tests/emergency_blind_eval.py "+" ".join(sys.argv[1:]),code_files=[E.__file__,E.LUTP,__file__],splits=M,kind="prod")
print(json.dumps({k:(v if not isinstance(v,dict) or k=="lut_meta" else {a:b for a,b in v.items() if a!="per_set"}) for k,v in rep.items() if k not in ("fn_primary","fp_primary","detect_fn","detect_fp")},indent=1))
