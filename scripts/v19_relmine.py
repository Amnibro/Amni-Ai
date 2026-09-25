import os,re,sys,json,time,collections,multiprocessing as mp
from pathlib import Path
import numpy as np
import nltk
H=Path(os.environ.get("AMNI_HOME","/home/anmire/ai/Amni-Ai"));W=Path(__file__).resolve().parents[1];OUT=Path(os.environ.get("RELMINE_OUT",W/"crates"/"adam-ray"/"src"/"v19_rel.tsv"));CORP=H/"data"/os.environ.get("RELMINE_CORPUS","clean2")/"train.bin";MB=int(os.environ.get("RELMINE_MB","480"));NW=min(6,int(os.environ.get("RELMINE_WORKERS","6")));SEED=int(os.environ.get("RELMINE_SEED","1911"))
nltk.data.path.insert(0,str(H/"data"/"concepts"/"nltk_data"));from nltk.corpus import wordnet as wn
TOK=re.compile(r"[A-Za-z]+|[0-9]+|[,;:()\"]");SENT=re.compile(r"(?<=[.!?])\s+|\n+")
FUNC=set("a an the and or but of to in on for with as at by from into about over under is are was were be been being am do does did has have had my your his her their our its it this that these those i you he she they we them him us me s t m re ve d ll not no never so if then than also just still now really very".split())
PRON=set("i he she they we you".split());POSS=set("my his her their our your".split());DET=set("a an the some his her their my our your its".split());NEG=set("not never t no".split());PREP=set("in at to from as of for with on into about by".split())
SEEDV={4:"live dwell reside inhabit settle",5:"work employ",7:"like love enjoy adore relish",2:"hate dislike detest loathe abhor despise",8:"allergic",9:"drive",11:"eat",12:"study major",13:"graduate attend"}
LEAD=set("is was are were am has have had been also now still just really currently recently always often usually will would can could does did do actually finally even truly".split())
def infl(w):return {w,w+"s",w+"es",w+"ed",w+"d",w+"ing",w[:-1]+"ing",w[:-1]+"ies",w[:-1]+"ied"} if w[-1:]=="e" or w[-1:]=="y" else {w,w+"s",w+"es",w+"ed",w+"ing",w+w[-1]+"ed",w+w[-1]+"ing"}
SEEDW={r:{x for w in ws.split() for x in infl(w)} for r,ws in SEEDV.items()};ALLSEED=set().union(*SEEDW.values())
def hy(ss,inst=False,cap=None):
 out=set()
 for s in ss:
  S=wn.synset(s)
  for x in [S]+list(S.closure(lambda y:y.hyponyms()+(y.instance_hyponyms() if inst else []))):
   out|={w for l in x.lemmas() for w in [l.name().replace("_"," ")] if re.fullmatch(r"[A-Za-z][A-Za-z -]{1,19}",w) and len(w.split())<=3 and (cap is None or w[0].isupper()==cap)}
 return out
def types():
 T={4:hy("city.n.01 town.n.01 national_capital.n.01 state_capital.n.01 port.n.01 country.n.02 american_state.n.01 province.n.01".split(),True,True),5:hy("worker.n.01 professional.n.01 skilled_worker.n.01 creator.n.02 performer.n.01 scientist.n.01".split(),False,False),7:hy("sport.n.01 diversion.n.01 hobby.n.01".split(),False,False),11:hy("dish.n.02 dessert.n.01 baked_goods.n.01 edible_fruit.n.01 vegetable.n.01 cheese.n.01 sandwich.n.01 soup.n.01".split(),False,None),10:hy("chromatic_color.n.01 achromatic_color.n.01".split(),False,False),12:hy(["discipline.n.01"],False,None)|hy(["natural_language.n.01"],True,True),8:hy("edible_nut.n.01 shellfish.n.01 pollen.n.01 antibiotic.n.01 dander.n.01".split(),False,False),9:hy("car.n.01 truck.n.01 motorcycle.n.01".split(),False,False),13:hy(["university.n.01","university.n.03","college.n.01"],True,True)}
 amb=collections.Counter(w.lower() for v in T.values() for w in v)
 return {r:{w for w in v if amb[w.lower()]==1 and w.lower() not in FUNC} for r,v in T.items()}
def vindex(T):
 ix={}
 for r,vs in T.items():
  for v in vs:
   t=tuple(TOK.findall(v));ix.setdefault(t[0] if v[0].isupper() else t[0].lower(),[]).append((t if v[0].isupper() else tuple(x.lower() for x in t),r,v[0].isupper()))
 return {k:sorted(v,key=lambda x:-len(x[0])) for k,v in ix.items()}
def occ(tk,ix):
 lo=[x.lower() for x in tk];i=0
 while i<len(tk):
  hit=next(((i,i+len(t),r) for t,r,cp in (ix.get(tk[i],[])+ix.get(lo[i],[]) if tk[i]!=lo[i] else ix.get(lo[i],[])) if (tuple(tk[i:i+len(t)]) if cp else tuple(lo[i:i+len(t)]))==t),None)
  if hit:yield hit;i=hit[1]
  else:i+=1
def left(tk,i):
 j=i;lo=[x.lower() for x in tk]
 while j>0 and lo[j-1] in DET and i-j<2:j-=1
 k=j;mid=[]
 while k>0 and len(mid)<5:
  w,lw=tk[k-1],lo[k-1]
  if not w.isalpha():return None
  if lw in PRON or (w[0].isupper() and k-1>0):return (tk[k-1],tuple(reversed(mid)),"p" if lw in PRON else "n")
  if k>=2 and lo[k-2] in POSS and w.islower() and lw not in FUNC:return (lo[k-2]+" "+lw,tuple(reversed(mid)),"o")
  mid.append(lw);k-=1
 return None
def scan(a):
 off,n,T,pairs=a;ix=vindex(T);C=collections.Counter();N=collections.Counter();S=collections.defaultdict(set);P=set();CN=collections.Counter()
 with open(CORP,"rb") as f:f.seek(off);raw=f.read(n).decode("utf-8","replace")
 for s in SENT.split(raw):
  if len(s)<12 or len(s)>400:continue
  tk=TOK.findall(s)
  for i,j,r in occ(tk,ix):
   L=left(tk,i)
   if not L:continue
   subj,mid,kind=L
   if not 1<=len(mid)<=4:continue
   (kind!="n") and (C.update([(mid,r)]),N.update([mid]))
   v=" ".join(tk[i:j]).lower()
   if kind=="n" and ALLSEED.intersection(mid):P.add((subj.lower(),mid,v,r))
   if pairs and kind=="n" and (subj.lower(),v) in pairs:S[(mid,pairs[(subj.lower(),v)])].add((subj.lower(),v))
  for m in re.finditer(r"\b(?:my|his|her|their|your) ((?:[a-z]+ )?[a-z]+) (?:is|was) (?:a |an |the )?([A-Za-z]+(?: [A-Za-z]+)?)",s):
   nv=m.group(1)
   for i,j,r in occ(TOK.findall(m.group(2)),ix):CN[(nv,r)]+=1;break
 return C,N,{k:len(v) for k,v in S.items()},P,CN
def run(T,pairs=None):
 sz=CORP.stat().st_size;R=np.random.default_rng(SEED);per=MB*(1<<20)//(NW*8);offs=sorted(int(x) for x in R.integers(0,sz-per-1,NW*8))
 with mp.Pool(NW) as p:rs=p.map(scan,[(o,per,T,pairs) for o in offs])
 C,N,S,CN=collections.Counter(),collections.Counter(),collections.Counter(),collections.Counter();P=set()
 for c,n,s,pp,cn in rs:C.update(c);N.update(n);S.update(s);P|=pp;CN.update(cn)
 return C,N,S,P,CN
RN={4:("residence.n.01 hometown.n.01 abode.n.01","populate.v.01 reside.v.01",512,""),5:("occupation.n.01 profession.n.02 career.n.01 trade.n.02","work.v.01",512,""),7:("pastime.n.01 avocation.n.01 diversion.n.01","like.v.02 love.v.01 enjoy.v.01 adore.v.01 love.v.02 care_for.v.02",1024,""),2:("peeve.n.01 dislike.n.02 aversion.n.01","dislike.v.01 hate.v.01 contemn.v.01 abhor.v.01 resent.v.01",1024,""),8:("allergy.n.01 allergen.n.01 hypersensitivity_reaction.n.01","",512,"allergic.s.01"),9:("vehicle.n.01 car.n.01 motor_vehicle.n.01 bicycle.n.01 motorcycle.n.01 truck.n.01","drive.v.01",1024,""),10:("color.n.01 shade.n.02 hue.n.01","",256,""),11:("food.n.01 food.n.02 dish.n.02 meal.n.01 dainty.n.01 cuisine.n.01 bite.n.04","eat.v.01 devour.v.03 crave.v.01",1024,""),12:("discipline.n.01","study.v.02 major.v.01",1024,""),13:("university.n.01 college.n.01 alma_mater.n.01","graduate.v.01 enroll.v.01 matriculate.v.01",1024,""),6:("name.n.01 nickname.n.01 appellation.n.01","name.v.01 dub.v.01 baptize.v.01",1024,"")}
VT={11:"food.n.02 dish.n.02 dainty.n.01 edible_fruit.n.01 vegetable.n.01 beverage.n.01",5:"worker.n.01 professional.n.01 skilled_worker.n.01 creator.n.02 performer.n.01 scientist.n.01",7:"sport.n.01 diversion.n.01 pastime.n.01",12:"discipline.n.01",10:"chromatic_color.n.01 achromatic_color.n.01"}
def wordnet(out,T):
 top=lambda w,pos:wn.synsets(w,pos)[:2]
 for r,(nr,vr,vc,ar) in RN.items():
  ns,vs,as_=({wn.synset(s) for s in x.split()} for x in (nr,vr,ar))
  for S0 in ns:
   for l in S0.lemmas():
    w=l.name().lower();ws=w.split("_")
    if len(ws)==1 and w.isalpha() and w not in FUNC and any(x in ns for x in top(w,"n")):[out.append(("wordnet","",x,256,r,0,0,S0.name())) for x in (w,w+"s")]
    elif len(ws)==2 and all(x.isalpha() for x in ws):out.append(("wordnet",ws[0],ws[1],256,r,0,0,S0.name()))
  for S0 in vs:
   for l in S0.lemmas():
    w=l.name().lower();ws=w.split("_")
    if len(ws)==1 and w.isalpha() and w not in FUNC and any(x in vs for x in top(w,"v")):[out.append(("wordnet","",x,vc if vc!=256 else 1024,r,0,0,S0.name())) for x in sorted(infl(w))]
  for S0 in as_:
   for l in S0.lemmas():
    w=l.name().lower()
    if w.isalpha():out.append(("wordnet","",w,512,r,0,0,S0.name()))
 for c in sorted({w.lower() for w in T[10] if w.isalpha()}):out.append(("wncolor","",c,2048,10,0,0,"color.n.01"))
 for c in sorted({w.lower() for w in T[10] if len(w.split())==2 and all(x.isalpha() for x in w.split())}):out.append(("wncolor",c.split()[0],c.split()[1],2048,10,0,0,"color.n.01"))
 for l in wn.synset("digest.v.03").lemmas():
  w=l.name().lower()
  if w.isalpha():[out.append(("wnneg","",x,1024,102,0,0,"digest.v.03")) for x in sorted(infl(w))]
 for s0 in ("fond.s.03","obsessed.s.02","haunted.s.01"):
  for l in wn.synset(s0).lemmas():
   w=l.name().lower()
   if w.isalpha():out.append(("wnadj","",w,512,7,0,0,s0))
 for w in ("spend","spends","spent","spending"):out.append(("seed","",w,512,7,0,0,"spend.v.01"))
 for l in wn.synset("relax.v.01").lemmas():
  w=l.name().lower()
  if w.isalpha():[out.append(("wnrelax","",y,1024,7,0,0,"relax.v.01")) for y in sorted(infl(w))]
 for p0,w in (("for","fun"),("free","time"),("spare","time"),("downtime",""),("leisure","time")):out.append(("seed",p0 if w else "",w or p0,256,7,0,0,"leisure.n.01"))
 for s0 in ("inhabitant.n.01","native.n.01","resident.n.01"):
  for l in wn.synset(s0).lemmas():
   w=l.name().lower()
   if w.isalpha():[out.append(("wnres","",y,512,4,0,0,s0)) for y in (w,w+"s")]
 for x in [wn.synset("administrative_district.n.01")]+list(wn.synset("administrative_district.n.01").closure(lambda z:z.hyponyms(),depth=1))+[wn.synset(s0) for s0 in ("region.n.03","city.n.01","town.n.01","island.n.01","area.n.01")]:
  for l in x.lemmas():
   w=l.name().lower()
   if w.isalpha():[out.append(("wnmeas","",y,32768,0,0,0,x.name())) for y in (w,w+"s")]
 for x in [wn.synset("container.n.01")]+list(wn.synset("container.n.01").closure(lambda z:z.hyponyms(),depth=2))+[wn.synset(s0) for s0 in ("part.n.02","lot.n.02","kind.n.01","bit.n.02","slice.n.01","helping.n.01","bunch.n.01","handful.n.01")]:
  for l in x.lemmas():
   w=l.name().lower()
   if w.isalpha():[out.append(("wnmeas","",y,32768,0,0,0,x.name())) for y in (w,w+"s")]
 for r,ss in VT.items():
  for s in ss.split():
   for x in [wn.synset(s)]+list(wn.synset(s).closure(lambda z:z.hyponyms())):
    for l in x.lemmas():
     w=l.name().lower().split("_")[-1]
     if w.isalpha() and len(w)>=3 and w not in FUNC and x in wn.synsets(w,"n")[:2]:[out.append(("wnvtype","",y,8192,r,0,0,x.name())) for y in (w,w+"s",w+"es",w[:-1]+"ies" if w.endswith("y") else w+"s")]
 for l in open(H/"data"/"clean2"/"wordcounts.txt",encoding="utf-8",errors="replace"):
  a=l.split(" ")
  if len(a)>=2 and a[1].strip().isdigit() and int(a[1])>=2000 and a[0].isalpha() and a[0].islower() and a[0] not in FUNC:
   v,n=wn.synsets(a[0],"v"),wn.synsets(a[0],"n")
   rr=wn.synsets(a[0],"r")
   if rr and len(rr)>=len(n)+len(v)+len(wn.synsets(a[0],"a"))+len(wn.synsets(a[0],"s")) and int(a[1])>=2000:out.append(("wnadv","",a[0],65536,0,0,0,rr[0].name()))
   if n and n[0].lexname() in ("noun.person","noun.animal","noun.artifact") and int(a[1])>=1000 and not (v and len(v)>len(n)):out.append(("wnnoun","",a[0],16384,1,0,0,n[0].name()))
   if v and wn.morphy(a[0],"v") and wn.morphy(a[0],"v")!=a[0]+"s" and len(v)>=len(n) and (a[0].endswith("s") or a[0].endswith("ed")):out.append(("wnverb","",a[0],4096,1,0,0,v[0].name()))
def main():
 t0=time.time();T=types();print(json.dumps({"types":{r:len(v) for r,v in T.items()}}),flush=True)
 C,N,_,P,CN=run(T);print(json.dumps({"pass1_s":round(time.time()-t0),"mids":len(N),"pairs":len(P)}),flush=True)
 seedw=SEEDW;MOT=lambda h:any(x.lexname() in ('verb.motion','verb.competition','verb.contact') for b in [wn.morphy(h,'v')] if b for x in wn.synsets(b,'v')[:1]);FLD={r:{y for w in ws.split() for s0 in wn.synsets(w) if s0.pos() in 'va' for y in [s0]+s0.hypernyms()+list(s0.closure(lambda z:z.hyponyms(),depth=2))} for r,ws in SEEDV.items()};infld=lambda h,r:any(x in FLD.get(r,()) for pos in 'va' for b in [wn.morphy(h,pos)] if b for x in wn.synsets(b,pos))
 prec=lambda m,r:C[(m,r)]/max(1,N[m])
 seedm={(m,r) for (m,r),c in C.items() if c>=3 and prec(m,r)>=.5 and any(w in seedw.get(r,()) for w in m)}
 pairs={(s,v):r for s,m,v,r in P if (m,r) in seedm}
 C2,N2,S,_,_=run(T,pairs);print(json.dumps({"pass2_s":round(time.time()-t0),"seed_mids":len(seedm),"seed_pairs":len(pairs),"boot_mids":len(S)}),flush=True)
 os.environ.get("RELMINE_DEBUG") and Path(os.environ["RELMINE_DEBUG"]).write_text(json.dumps({r:[(" ".join(m),c,round(prec(m,r),2),S.get((m,r),0)) for (m,rr),c in C.most_common() if rr==r and c>=5][:80] for r in T},indent=0))
 rows={}
 for (m,r),c in C.items():
  mm=list(m)
  while mm and mm[0] in LEAD:mm=mm[1:]
  neg=any(w in NEG for w in m);mm=[w for w in mm if w not in NEG and w not in ("doesn","don","didn","isn","wasn","can","cannot")]
  cont=[i for i,w in enumerate(mm) if w not in FUNC];sup=S.get((m,r),0)
  if len(cont)!=1 or c<6 or prec(m,r)<.6 or not ((sup>=3 and infld(mm[cont[-1]],r)) or mm[cont[-1]] in seedw.get(r,())) or MOT(mm[cont[-1]]):continue
  hd=cont[-1];tail=mm[hd+1:];rr=2 if neg and r in (7,11) else r
  if neg and rr==r:continue
  if tail and any(w not in PREP for w in tail):continue
  key=(mm[hd-1] if hd>0 else "",mm[hd]);cls=512 if tail else 1024;k=(key,cls,rr);x=rows.setdefault(k,[0,0,set()]);x[0]+=c;x[1]=max(x[1],sup);x[2].add(" ".join(m))
 TN=collections.Counter()
 for (nv,r),c in CN.items():TN[nv]+=c
 for (nv,r),c in CN.items():
  tot=TN[nv]
  if c>=30 and c/tot>=.6 and nv.split()[-1] not in FUNC and not any(x.lexname()=='noun.person' for x in wn.synsets(nv.split()[-1],'n')):k=((nv.split()[0] if " " in nv else "",nv.split()[-1]),256,r);x=rows.setdefault(k,[0,0,set()]);x[0]+=c;x[2].add("my "+nv+" is")
 hp=collections.defaultdict(collections.Counter)
 for (key,cls,r),(c,_,_) in rows.items():hp[key[1]][(cls,r)]+=c
 out=[]
 for (key,cls,r),(c,sup,ms) in sorted(rows.items(),key=lambda x:(x[0][2],-x[1][0])):
  h=hp[key[1]];uni=h[(cls,r)]/sum(h.values())>=.8
  out.append(("mined","" if uni else key[0],key[1],cls,r,c,sup,"|".join(sorted(ms))[:200]))
 wordnet(out,T)
 cnt={a[0] for l in open(H/"data"/"clean2"/"wordcounts.txt",encoding="utf-8",errors="replace") for a in [l.split(" ")] if len(a)>=2 and a[1].strip().isdigit() and int(a[1])>=50}
 seen=set();fin=[]
 for o in out:
  k=(o[1],o[2],o[3])
  if k in seen or o[2] not in cnt:continue
  seen.add(k);fin.append(o)
 OUT.write_text("".join("\t".join(str(x) for x in o)+"\n" for o in fin));print(json.dumps({"rows":len(fin),"by_src":collections.Counter(o[0] for o in fin),"by_rel":collections.Counter(o[4] for o in fin),"secs":round(time.time()-t0),"out":str(OUT)}),flush=True)
if __name__=="__main__":main()
