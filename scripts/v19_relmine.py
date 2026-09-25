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
G2CUE="""4::live~ in|live~ over in|live~ out in|live~ way out in|live~ all the way out in|live~ all the way over in|live~ way over in|live~ clear across in|am in|are in|was in|am based in|am from|are from|reside~ in|dwell~ in|settle~ in|settle~ down in|move~ to|move~ out to|move~ over to|relocate~ to|base~ in|based out of|rent~ a place in|flat in|apartment in|condo in|house in|place in|home in|stay~ in|is from|come~ from|hail~ from|grew up in|raised in|is over in|is out in|is in|made a home in|home for|local in|a local in|is a local in|live~ near|live~ just outside|live~ outside
4:64:live~|base~|settle~|call home
4:16:live~|base~|settle~|stay~|move~|call home|reside~|stationed|is located
5::work~ as|employ~ as|earn~ a living as|make~ a living as|train~ as|qualified as|serve~ as|for a living|for work|day job|by day|job title|line of work|work~ full-time as|work~ part-time as|freelance~ as|job as|career as
5:64:by trade|by profession
5:16:earn~ a living|make~ a living|do~ for work|do~ for a living|do~ professionally|do~ for money|work~|employed|do~ all day at work
6::go~ by|answer~ to|respond~ to the name|respond~ to|known as|go~ by the name|name on|called|is called|goes by the name of
6:1:call~|nickname~|christen~|dub~|name~|baptise~|baptize~|rename~
11:8:we would have|happier than|crazy about|nuts about|mad about|go~ crazy for|could eat|love~ to eat
7:8:like~|love~|enjoy~|adore~|relish~|is into|are into|am into|really into|more into|totally into|super into|so into|got into|gotten into|get~ into|getting into|is big on|big on|obsessed with|hooked on|pick~ up|took up|take~ up|fan of|keen on|passionate about|addicted to|to unwind|to relax|for fun|free time|spare time|spare hour|downtime|spend~ all day|spend~ hours|spend~ weekends|spend~ most weekends|nothing more than|does a lot of|do~ a lot of|lot of time|enjoy~ doing|love~ doing|dabble~ in|devoted to|a sucker for|live~ for|all about|is all about|mostly into|big into|very into|quite into|pretty into|heavily into
7:16:into|happiest|busy|keep~ busy|do~ for fun|do~ in|do~ to unwind|do~ to relax|enjoy~|like~ to do|love~ to do|pastime
7:136:get enough of
8::allergic to|allergy to|allergic reaction to|allergies to|hives around|hives from|hives when|hives if|rash from|react~ badly to|react~ to|epipen because of|epipen for|itchy from|itchy around|sneez~ around|sneez~ near|swell~ up around|intolerance to|sensitive to|allergic|trigger~ allergy|trigger~ allergies
8:64:because of an allergy|because of allergies|due to an allergy|it is an allergy|is an allergy
2:32:avoid~|keep~ away from|stay~ away from|keep~ clear of|steer~ clear of
2:36:go near|go anywhere near|eat|be around|be near|go around
9::drive~|drove|driven|get~ around in|commute~ in|commute~ to work in|behind the wheel of|zip~ around in|cruise~ around in|get~ around town in|parked in|drive~ around in|driving around in
9:2:got|have got|has got|traded it in for|trade~ it in for|swapped it for|swap~ it for|replaced it with|proud owner of|own~|bought|buy~|has|have|had|keep~|swear~ by|just got|picked up|lease~|test-drove|upgraded to|traded up to
10:3:paint~|repaint~|dye~
10:2:went with|came in|comes in|everything in|dressed in|wear~|choose~|chose|decorated in|paint~ in
10:8:pick~|go~ for|opt~ for
10::shade of
11::eat~|could eat|order~|crave~|craving|beg~ for|go~ crazy for|go~ nuts for|go~ wild for|weakness for|only eat|snack~ on|munch~ on|feast~ on|devour~|live~ on|like~ eating|love~ eating|like~ to eat|love~ to eat|cook~ for|makes for dinner|always order~|go-to order|plate of
12::go~ to school for|went to school for|study~|major~ in|major~|degree in|background in|diploma in|masters in|master in|phd in|doctorate in|specialize~ in|specialise~ in|qualification in|certificate in|trained in|lectures in|teach~|studying
12:2:read~|switch~ to|changed to
13:64:grad|alum|alumnus|alumna|graduate
13::went to|go~ to school at|graduate~ from|graduate of|alum of|alumna of|alumnus of|alumni of|years at|student at|studying at|enrolled at|attend~|studied at|undergrad at|degree from|diploma from|transfer~ to|went to college at|went to university at|went to school at|degree at|studies at|study~ at|did undergrad at|went to school|go~ to school
13:16:enrolled|graduate~|go~ to|attend~
14:2:turn~|hit|aged|built|made|is|am|are|was
14:64:years old|year old|years ago|years young
15:66:birthday|birthday month|birth month
15::year older every|years older every|older every|born in|born on|born one|born|celebrate~ birthday in|celebrate~ birthday
15:66:baby
2::hate~|dislike~|detest~|loathe~|despise~|abhor~|resent~|zero patience for|no patience for|grumpy about|freak~ out at|freak~ out over|freak~ out about|freak~ out when|freaks out at|berserk over|go~ berserk over|go~ berserk at|terrified of|scared of|afraid of|frightened of|fear~|pet peeve|pet hate|hide~ from|hide~ under the bed because of|under the bed because of|run~ from|bark~ at|hiss~ at|tired of|sick of|fed up with|can not stand|cannot stand|nerves|allergic to criticism
2:4:stand|bear|tolerate~|handle|stomach|mention~|abide|patience for
2:1:annoy~|irritate~|bother~|bug~|drive~ crazy|drive~ up the wall|get~ on nerves
2:16:set~|set~ off|get~ on nerves"""
G2RN="""4:home|house|place|flat|apartment|city|town|hometown|country|state|province|region|prefecture|county|continent|address|neighbourhood|neighborhood|home base|home town|condo|base
5:job|occupation|profession|career|trade|line of work|day job|job title|work|role|gig|vocation|livelihood|title|line of business
6:name|nickname|full name|first name|real name
7:hobby|hobbies|pastime|passion|thing to do|thing to do for fun|activity|sport|free time|spare time|downtime|way to relax|way to unwind|interest|interests
8:allergy|allergies|allergen
9:car|ride|vehicle|wheels|daily driver|motor|garage|daily ride|set of wheels
10:colour|color|shade|hue|colour of choice|color of choice|colours|colors
11:anything|everything|food|dish|meal|snack|comfort food|cuisine|treat|thing to eat|dinner|order|food of choice|meal of choice|dish of choice
12:major|degree|field|subject|academic background|background|studies|area of study|field of study|diploma|masters|phd|doctorate|specialty|speciality|discipline|degree subject|academic field
13:school|college|university|uni|alma mater|old school|undergrad|campus
14:age
15:birthday|birth month|birthday month|big day|bday|birth day
2:pet peeve|pet hate|biggest fear|fear|phobia|nemesis|least favourite thing|least favorite thing|nerves
11:food-wise
13:school-wise|college-wise|academically
5:work-wise|job-wise
10:colour-wise|color-wise
0:thing|one"""
G2MOD="favourite favorite fave fav go-to all-time biggest big top usual current new main real actual true one only preferred absolute very first own chosen current usual regular"
G2FUNC={1:"a an the some any this that these those every each another",2:"my your his her their its our",3:"i you he she they it we",4:"me you him her them it us mine myself yours hers his theirs ours",5:"is are was were am be been being",6:"do does did don doesn didn can could will would shall should may might must has have had hasn haven hadn isn aren wasn weren won wouldn couldn shouldn cannot ain",7:"in at on to from for with by of about into onto over under around near as than between through during after before since until without within across behind beside along among toward towards upon via per off down up out back away",8:"and or but so because while whereas although though if when whenever unless until since nor yet then",9:"what where who when which how why whose whom",10:"not never no none",11:"oh okay ok well hey hi hello yeah yes um uh hmm wow anyway btw fyi actually wait sorry oops please thanks alright"}
G2SKIP="thing things something stuff old best least most a an the some any this that these those my your his her their its our is are was were am be been being really just also always still even truly absolutely totally mostly only now currently usually generally basically definitely actually literally lately recently more than nothing one good big huge little lovely whole plate bowl cup glass slice piece bag bunch handful serving portion of lot bit kind sort type doing having the"
G2STOP="every each whenever lately recently now anymore again still too ago forever together instead rather yet whatsoever most later soon last next today tonight tomorrow yesterday often always usually sometimes never daily weekly really just also very so quite pretty super totally absolutely definitely honestly actually literally basically seriously mostly probably maybe perhaps anyway either neither else more less than over all these those days years months weeks lot times hands down though no-one either"
G2META="say said says tell told tells mention mentioned mentions forget forgot forgotten think thought remember remembered recall recalled guess mean meant wonder wondering wondered know knew bet suppose believe misspoke misspoken wrote typed"
G2HYPO="almost nearly wish wishes wished want wants wanted hope hopes hoped dream dreams dreamed dreamt plan plans planning planned considering consider considers thinking might someday eventually wanna gonna"
G2JOIN="of de la del di da van von du le der den st am upon es el al bin ben dos das do"
G2CAR="Acura,Alfa,Aston,Audi,Bentley,BMW,Bugatti,Buick,Cadillac,Chevrolet,Chevy,Chrysler,Citroen,Cupra,Dacia,Daewoo,Daihatsu,Datsun,DeLorean,Dodge,Ferrari,Fiat,Fisker,Ford,Genesis,GMC,Holden,Honda,Hummer,Hyundai,Infiniti,Isuzu,Jaguar,Jeep,Kia,Koenigsegg,Lada,Lamborghini,Lancia,Land,Lexus,Lincoln,Lotus,Lucid,Maserati,Maybach,Mazda,McLaren,Mercedes,Mercury,MG,Mini,Mitsubishi,Morgan,Nissan,Oldsmobile,Opel,Pagani,Peugeot,Plymouth,Polestar,Pontiac,Porsche,Proton,Ram,Renault,Rivian,Rolls,Rover,Saab,Saturn,Scion,Seat,Skoda,Smart,Subaru,Suzuki,Tata,Tesla,Toyota,Trabant,Vauxhall,Volkswagen,VW,Volvo,Harley,Ducati,Vespa,Yamaha,Kawasaki,Triumph,Prius,Corolla,Camry,Civic,Accord,Mustang,Beetle,Golf,Jetta,Passat,Outback,Forester,Impreza,Wrangler,Cherokee,Tacoma,Tundra,Silverado,Miata,Corvette,Camaro"
G2COL="red orange yellow green blue purple pink brown black white gray grey silver gold golden beige teal navy maroon violet scarlet crimson indigo turquoise magenta lavender burgundy charcoal cerulean chartreuse cobalt emerald fuchsia ivory khaki lilac mauve ochre periwinkle sapphire sepia taupe terracotta ultramarine vermilion aquamarine azure bronze copper cyan mahogany chestnut gunmetal pewter platinum amethyst goldenrod seafoam aqua celadon puce umber russet cornflower heliotrope plum rust peach mint lime lemon salmon coral cream sand slate olive mustard wine rose ruby jade pearl tan aubergine apricot tangerine saffron sage denim mulberry honey eggplant lemon cherry raspberry blush oatmeal khaki bone ebony jet ash smoke stone butter caramel cinnamon chocolate coffee mocha orchid lilac lavender rosewood ochre"
G2SCH="University,College,Institute,Academy,Polytechnic,Conservatory,School,Uni,Tech,Seminary,Politecnico,Universidad,Universite,Hochschule"
def gen2():
 t0=time.time();rows=[];cnt={a[0]:int(a[1]) for l in open(H/"data"/"clean1"/"wordcounts.txt",encoding="utf-8",errors="replace") for a in [l.split(" ")] if len(a)>=2 and a[1].strip().isdigit()}
 ex=lambda p:[" ".join(ws) for ws in __import__("itertools").product(*[sorted(infl(w[:-1])) if w.endswith("~") else [w] for w in p.split()])]
 for ln in G2CUE.splitlines():
  r,f,ps=ln.split(":",2);[rows.append(("seed",x.lower(),1,int(r),int(f or 0))) for p in ps.split("|") for x in ex(p)]
 for ln in G2RN.splitlines():
  r,ps=ln.split(":",1);[rows.append(("seed",p,2,int(r),0)) for p in ps.split("|")]
 T={11:hy("food.n.02 dish.n.02 dainty.n.01 edible_fruit.n.01 vegetable.n.01 beverage.n.01 baked_goods.n.01 dessert.n.01 cheese.n.01 sandwich.n.01 soup.n.01 meat.n.01 seafood.n.01 snack_food.n.01 candy.n.01 pasta.n.02 bread.n.01 salad.n.01".split(),False,False),10:hy("chromatic_color.n.01 achromatic_color.n.01".split(),False,False),5:hy("worker.n.01 professional.n.01 skilled_worker.n.01 creator.n.02 performer.n.01 scientist.n.01 craftsman.n.03 technician.n.01 engineer.n.01 employee.n.01 expert.n.01 artist.n.01 health_professional.n.01".split(),False,False),12:hy(["discipline.n.01","science.n.01","humanistic_discipline.n.01"],False,None)|hy(["natural_language.n.01"],True,True),7:hy("sport.n.01 diversion.n.01 hobby.n.01 pastime.n.01 game.n.01 exercise.n.01 art.n.03 dancing.n.01 music.n.01 handicraft.n.01".split(),False,False),8:hy("allergen.n.01 pollen.n.01 dander.n.01 edible_nut.n.01 shellfish.n.01 antibiotic.n.01 analgesic.n.01 drug.n.01 mold.n.05 mite.n.02 latex.n.01 sting.n.01".split(),False,False),9:hy("car.n.01 truck.n.01 motorcycle.n.01 van.n.05".split(),False,False),4:hy("city.n.01 town.n.01 national_capital.n.01 state_capital.n.01 port.n.01 country.n.02 american_state.n.01 province.n.01 island.n.01".split(),True,True),13:hy(["university.n.01","university.n.03","college.n.01"],True,True)}
 seen=set();[seen.add(c) or rows.append(("seed",c,3,10,0)) for c in dict.fromkeys(G2COL.split()) if c not in seen]
 pl=lambda w:[w]+([w+"es"] if w.endswith(("s","x","ch","sh")) else [w[:-1]+"ies"] if w.endswith("y") and w[-2:-1] not in "aeiou" else [w+"s"])
 for r in (10,5,12,7,11,8,9,13,4):
  for w in sorted(T[r]):
   k=w.lower() if not w[0].isupper() else w
   for k2 in (pl(k) if r in (11,8,7,5) and not w[0].isupper() else [k]):
    if k2.lower() not in seen and cnt.get(k.split()[-1],cnt.get(k.split()[-1].lower(),0))>=20:seen.add(k2.lower());rows.append(("wordnet",k2.lower(),3,r,1 if w[0].isupper() else 0))
 for c in G2CAR.split(","):c.lower() not in seen and rows.append(("seed",c.lower(),3,9,1))
 for c in G2SCH.split(","):rows.append(("seed",c.lower(),3,13,3))
 for i,m in enumerate("january february march april may june july august september october november december".split()):rows.append(("seed",m,3,15,1))
 kin="sister mom aunt grandma niece daughter wife girlfriend stepmom stepsister godmother goddaughter brother dad uncle grandpa nephew son husband boyfriend stepdad stepbrother godfather godson cousin coworker boss landlord landlady neighbor neighbour roommate housemate flatmate bandmate classmate teammate mentor tutor partner friend manager buddy pal mate fiance fiancee parents parent kids kid twin twins grandparents in-laws folks family supervisor colleague intern babysitter dentist therapist physio barber hairdresser accountant doctor coach teacher".split()
 heads=set(kin)|{l.name().lower() for S in wn.all_synsets("n") if S.lexname() in ("noun.person","noun.animal","noun.artifact") for l in S.lemmas() if l.name().isalpha() and cnt.get(l.name().lower(),0)>=50}
 [rows.append(("wordnet",w,4,0,0)) for w in sorted(heads)]
 vb=sorted(w for w,c in cnt.items() if c>=200 and w.isalpha() and w.islower() and w not in heads and wn.morphy(w,"v") and len(wn.synsets(w,"v"))>=len(wn.synsets(w,"n")))
 [rows.append(("wordnet",w,5,0,0)) for w in vb]
 [rows.append(("wordnet",w,13,0,0)) for w,c in sorted(cnt.items()) if c>=200 and w.isalpha() and w.islower() and any(l.name()==w for S in wn.synsets(w,"v") for l in S.lemmas())]
 [rows.append(("seed",w,6,0,0)) for w in G2MOD.split()]
 [rows.append(("seed",w,7,0,0)) for w in G2STOP.split()]
 [rows.append(("seed",w,8,0,0)) for w in G2META.split()]
 [rows.append(("seed",w,9,0,0)) for w in G2HYPO.split()]
 [rows.append(("seed",w,10,0,0)) for w in G2SKIP.split()]
 [rows.append(("seed",w,11,0,0)) for w in G2JOIN.split()]
 [rows.append(("seed",w,12,k,0)) for k,ws in G2FUNC.items() for w in ws.split()]
 for f in (l.split("\t") for l in (W/"crates"/"adam-ray"/"src"/"v19_rel.tsv").read_text().splitlines()):
  c,r,ph=int(f[3]),int(f[4]),(f[1]+" "+f[2]).strip()
  if c==256 and r>1:rows.append(("rel1",ph,2,r,0))
  elif c in (512,1024) and r==102:rows.append(("rel1",ph,1,2,4))
  elif c in (512,1024) and 1<r<16:rows+=[("rel1",ph,1,r,8 if r==7 else 0)]+([("rel1",ph+" of",1,r,0)] if f[0]=="wnres" else [])
  elif c==2048:rows.append(("rel1",ph,3,10,0))
 mined=os.environ.get("RELMINE_G2MINE")=="1" and g2mine() or []
 rows+=mined;dd=set();fin=[]
 for o in rows:
  if (o[1],o[2],o[3]) not in dd:dd.add((o[1],o[2],o[3]));fin.append(o)
 out=W/"crates"/"adam-ray"/"src"/"v19_gen2.tsv";out.write_text("".join("\t".join(str(x) for x in o)+"\n" for o in fin));print(json.dumps({"rows":len(fin),"by_class":collections.Counter(o[2] for o in fin),"mined":len(mined),"secs":round(time.time()-t0),"out":str(out)}),flush=True)
G2LEM={4:set("live dwell reside settle move relocate emigrate immigrate grow raise stay remain base come hail".split()),5:set("work serve employ appoint promote train qualify".split()),12:set("study major specialize specialise".split()),13:set("graduate attend enroll enrol".split()),7:set("like love enjoy adore relish".split()),11:set("eat crave devour".split()),9:set("drive".split()),2:set("hate dislike detest loathe despise abhor resent".split()),8:set("react".split())}
def g2mine():
 T=types();C,N,_,_,_=run(T);out=[]
 for (m,r),c in sorted(C.items(),key=lambda x:-x[1]):
  mm=list(m)
  while mm and mm[0] in LEAD:mm=mm[1:]
  while mm and mm[-1] in DET:mm=mm[:-1]
  if 2<=len(mm)<=4 and c>=20 and c/max(1,N[m])>=.75 and mm[0] not in FUNC and mm[-1] in PREP and not any(w in NEG or w in ("s","re","ll","ve","d","m","t") for w in mm) and wn.morphy(mm[0],"v") in G2LEM.get(r,()):out.append(("mined2"," ".join(mm),1,r,8 if r==7 else 0))
 os.environ.get("RELMINE_DEBUG") and Path(os.environ["RELMINE_DEBUG"]).write_text("\n".join("\t".join(map(str,o)) for o in out))
 return out
if __name__=="__main__":gen2() if os.environ.get("RELMINE_GEN2")=="1" else main()
