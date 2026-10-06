import re,sys,json,random,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from amni.compute import wikifact as WF
OUT=ROOT/"data"/"store2"/"props_mined.json";PJ=ROOT/"data"/"store1"/"props_en.json"
STOP=set("a an the of in on at to for by with from as is are was were be been being am do does did done has have had having will would shall should can could may might must what which who whom whose when where why how that this these those it its they them their there here i you he she we me my your his her our us and or but not no yes if then than so very just also about into onto over under up down out off again ever any some all each every much many more most other such only own same few one ones s use used e g eg see item items subject object entity value property statement wikidata".split())
GEN=set("person place thing work date time number name type kind part member list other specific known most main".split())
def lem(w:str)->str:
 wn=WF._wn();return ((wn.morphy(w,'v') or wn.morphy(w,'n')) if wn and len(w)>3 else None) or w
def lg(s:str)->list:return [lem(x) for x in WF.nrm(re.sub(r"\([^)]*\)","",s)).split()]
def wcls(S,p:int,dt:str)->str:
 if dt=="time":return "when"
 if dt=="quantity":return "how"
 if dt!="wikibase-item":return "what"
 rng=random.Random(p);hum=plc=og=n=0
 import numpy as np
 idx=np.flatnonzero(np.asarray(S.f_p)==p)
 for f in (rng.sample(list(idx),min(400,len(idx))) if len(idx) else []):
  v=int(S.f_v[f]);n+=1;G={int(x) for x in S.f_p[int(S.e_fs[v]):int(S.e_fs[v])+int(S.e_fn[v])]}
  org=bool(G&{159,452,749,112,169,1128,1454,856,127});hum+=5 in S.inst(v) or 569 in G or 21 in G;og+=org;plc+=bool(G&{131,30,36,1082,2044,206,625,1376,17}) and not org
 return "who" if n and hum/n>0.5 else "org" if n and og/n>0.4 else "where" if n and plc/n>0.5 else "what"
def wnx(phrase:list,wn)->list:
 o=[];ws=" ".join(phrase)
 if not wn or not phrase:return o
 heads=[w for w in phrase if w not in STOP and w not in GEN]
 for w in heads[-2:]:
  for pos in ("v","n"):
   for sy in wn.synsets(w,pos)[:2]:
    for l in sy.lemmas():
     n=l.name().lower().replace("_"," ")
     if n!=w and n not in WF.LIGHT and len(n)>2:o.append((ws.replace(w,n),"syn"));' ' in n and o.append((n,"syn"))
     for d in l.derivationally_related_forms():
      dn=d.name().lower().replace("_"," ")
      if dn!=w and dn not in WF.LIGHT and len(dn)>2:o.append((ws.replace(w,dn),"der"));len(phrase)>1 and o.append((dn,"der"))
 return o
def gloss(w:str,wn)->list:
 o=[]
 for sy in (wn.synsets(w,"n")[:1] if wn else []):
  t=[lem(x) for x in WF.nrm(sy.definition()).split()]
  o+=[" ".join(t[i:i+n]) for n in (2,3) for i in range(len(t)-n+1) if t[i] not in STOP and t[i+n-1] not in STOP and t[i] not in GEN]
 return o
def main():
 import numpy as np
 S=WF.store();wn=WF._wn();J=json.loads(PJ.read_text())["props"];P={};DF=collections.Counter();PC=np.bincount(np.asarray(S.f_p).astype(np.int64))
 desc={}
 for k,v in J.items():
  d=[x for x in lg(v["description"]) if x not in STOP and x not in GEN and len(x)>2];desc[k]=sorted(set(d));DF.update(set(d))
 for k,v in J.items():
  p=int(k[1:]);wh=wcls(S,p,v["datatype"]);E=[]
  for i,a in enumerate([v["label"]]+v["aliases"]):
   t=lg(a)
   if not t or all(x in STOP for x in t) or len(t)>4:continue
   E.append((" ".join(t),1.0 if i==0 else 0.8,"label" if i==0 else "alias"))
   [E.append((x,0.6,src)) for x,src in wnx(t,wn)]
   len(t)>1 and [E.append((x,0.45,"uni")) for x in t if (x not in STOP or wn and wn.synsets(x,"v") and x not in WF.LIGHT) and x not in GEN and len(x)>2]
  [E.append((g,0.35,"gloss")) for g in gloss(lg(v["label"])[-1] if lg(v["label"]) else "",wn)] if v["label"] else None
  [E.append((d,round(0.3*min(1.0,3.0/DF[d]),3),"desc")) for d in desc[k]]
  B={}
  for x,w,src in E:
   if w>B.get(x,(0,""))[0]:B[x]=(w,src)
  P[k]={"label":v["label"],"wh":wh,"facts":int(PC[p]) if p<len(PC) else 0,"datatype":v["datatype"],"keys":[[x,w,src] for x,(w,src) in sorted(B.items())]}
 OUT.write_text(json.dumps({"license":"CC0 1.0 (Wikidata labels/aliases/descriptions) + WordNet 3.0 licence (synonyms, derivations, glosses)","source":str(PJ.relative_to(ROOT)),"props":P},indent=0,ensure_ascii=False));print(len(P),sum(len(v["keys"]) for v in P.values()),collections.Counter(v["wh"] for v in P.values()))
if __name__=="__main__":main()
