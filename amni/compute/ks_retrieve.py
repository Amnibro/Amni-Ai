import os,re,glob,math,shutil,hashlib,collections,urllib.parse
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
import amni.compute.ray_learn as RL
from amni.compute.ray_qa import words
from amni.compute.ks_codec import KStore,paginate,memguard
from scripts.neutral_filter import DEFAULT_FILTER
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KS=os.path.join(_H0,"data","ks");STORE=os.environ.get("AMNI_KS_STORE","store_clean");KT=64;FETCH=os.path.join(KS,"fetch");PAREN=re.compile(r"\s*\((?:[^()]|\([^()]*\))*\)");COP=re.compile(r"\b(?:is|are|was|were|refers? to|became|has been)\b");NAVX=re.compile(r"(?i)\b(?:may|can|might) (?:also )?refer to|\bis a list of\b|\bis the name of\b|\bis a surname\b|\bis a given name\b")
BORN=re.compile(r"\((?:[^()]*?[;,]\s*)?(?:born|b\.)\s|\((?:[^()]*?[;,]\s*)?(?:c\.\s*)?(?:\d{1,2}\s+)?(?:[A-Z][a-z]+\s+)?(?:\d{1,2},?\s+)?\d{3,4}\s*[–-]\s*(?:c\.\s*)?(?:\d{1,2}\s+)?(?:[A-Z][a-z]+\s+)?(?:\d{1,2},?\s+)?\d{3,4}\)")
OCC=re.compile(r"\b(?:is|was)\s+(?:an?|the)\s+(?:(?!(?:by|of|from|for|in|on|at|with|about|featuring|and|to)\b)[\w'-]+\s+){0,5}?(?:actor|actress|singer|musician|songwriter|rapper|politician|writer|author|novelist|poet|playwright|screenwriter|composer|conductor|director|producer|footballer|cricketer|athlete|sprinter|swimmer|cyclist|golfer|businessman|businesswoman|entrepreneur|engineer|scientist|physicist|chemist|biologist|botanist|zoologist|mathematician|philosopher|painter|artist|sculptor|architect|comedian|journalist|broadcaster|king|queen|prince|princess|emperor|empress|monarch|saint|bishop|pope|lawyer|jurist|judge|admiral|soldier|explorer|inventor|designer|dancer|wrestler|boxer|coach|player|economist|historian|priest|activist|diplomat|guitarist|drummer|pianist|violinist|vocalist|personality|statesman|stateswoman|theologian|astronomer|physician|surgeon|psychologist|sociologist|anthropologist|linguist|photographer|illustrator|cartoonist|filmmaker|chef|pilot|astronaut|missionary|nun|monk|rabbi|cleric|scholar|professor|educator|officer|mayor|governor|senator|minister|mayor)s?\b")
def th(w:bytes)->int:return int.from_bytes(hashlib.blake2b(w,digest_size=4).digest(),"little")
def terms(t:str,ps:list)->tuple:
 tw={w:2.0*RL.idf(w) for w in words(t.encode())};out=[]
 for i,pg in enumerate(ps):
  c=collections.Counter(words(pg));s={w:(1+math.log(n))*RL.idf(w) for w,n in c.items()};[s.__setitem__(w,s.get(w,0.0)+v) for w,v in tw.items()];out.extend((th(w),i,min(255,int(v*8))) for w,v in sorted(s.items(),key=lambda kv:-kv[1])[:KT])
 a=np.array(out,np.int64).reshape(-1,3);return a[:,0].astype(np.uint32),a[:,1].astype(np.uint32),a[:,2].astype(np.uint8)
def save_index(d:str,**A):os.makedirs(d+".tmp",exist_ok=True);[np.save(os.path.join(d+".tmp",k+".npy"),v) for k,v in A.items()];os.path.isdir(d) and os.replace(d,d+".old");os.replace(d+".tmp",d);os.path.isdir(d+".old") and shutil.rmtree(d+".old")
def load_index(d:str)->dict:
 d[-4:]==".npz" and not os.path.isdir(d[:-4]) and save_index(d[:-4],**dict(np.load(d)));d[-4:]==".npz" and os.remove(d);d=d[:-4] if d[-4:]==".npz" else d
 return {k:np.load(os.path.join(d,k+".npy"),mmap_mode="r") for k in ("terms","ptr","pages","w","npages")}
class KSIndex:
 def __init__(self,root:str=KS,codec=None,srcs:tuple=("wikipedia",)):
  self.root=root;self.codec=codec;[load_index(f) for s in srcs for f in glob.glob(os.path.join(root,"index",s,"*.npz"))];self.seg=[(s,os.path.basename(f),load_index(f)) for s in srcs for f in sorted(glob.glob(os.path.join(root,"index",s,"[0-9]*"))) if os.path.isdir(f)];self._st={}
 def search(self,q:str,k:int=5)->list:
  qw=list(dict.fromkeys(words(q.encode())));hs=np.array([th(w) for w in qw],np.uint32);iw=np.array([RL.idf(w) for w in qw]);out=[]
  for g,(_,_,z) in enumerate(self.seg):
   j=np.searchsorted(z["terms"],hs);ok=(j<len(z["terms"]))&(z["terms"][np.minimum(j,len(z["terms"])-1)]==hs);n=int(np.asarray(z["npages"]).reshape(-1)[0]);sc=np.zeros(n);hit=np.zeros(n)
   for jj,i in zip(j[ok],iw[ok]):sl=slice(z["ptr"][jj],z["ptr"][jj+1]);np.add.at(sc,z["pages"][sl],z["w"][sl]*i/8.0);np.add.at(hit,z["pages"][sl],i)
   sc*=(hit/max(iw.sum(),1e-9))**2;t=np.argpartition(-sc,min(k,len(sc)-1))[:k] if len(sc) else [];out+=[(float(sc[p]),g,int(p)) for p in t if sc[p]>0]
  return sorted(out,reverse=True)[:k]
 def _clean(self,g:int,want:set)->dict:
  s,k,_=self.seg[g];pf=pq.ParquetFile(os.path.join(self.root,"clean",s,k+".parquet"));n=0;out={}
  for b in pf.iter_batches(batch_size=1024,columns=["title","text"]):
   for t,x in zip(b.column(0).to_pylist(),b.column(1).to_pylist()):ps=paginate(x.encode());[out.__setitem__(n+i,(t,pg.decode("utf-8","replace"),False)) for i,pg in enumerate(ps) if n+i in want];n+=len(ps)
   if len(out)==len(want):break
  return out
 def fetch(self,hits:list)->list:
  by=collections.defaultdict(list);[by[g].append(p) for _,g,p in hits];res={}
  for g,ps in by.items():
   s,k,_=self.seg[g];sp=os.path.join(self.root,STORE,s,k);fp=os.path.join(self.root,"fetch",s,k);f=self._st.setdefault(("f",g),fstore(fp)) if os.path.exists(os.path.join(fp,"pg.npy")) else None;f is None and memguard()
   if f is not None:[res.__setitem__((g,p),(t,DEFAULT_FILTER.clean(paginate(x.encode())[p-int(f["pg"][d])].decode("utf-8","replace"))[0],False)) for p in ps for d in [int(np.searchsorted(f["pg"],p,"right")-1)] for t,x in [fdoc(f,d)]]
   elif os.path.exists(os.path.join(sp,"meta.json")):st=self._st.setdefault(g,KStore(sp,self.codec));[res.__setitem__((g,p),(st.title[st.doc_of(p)],x.decode("utf-8","replace"),True)) for p,x in zip(ps,st.get(ps))]
   else:[res.__setitem__((g,p),v) for p,v in self._clean(g,set(ps)).items()]
  return [res[(g,p)] for _,g,p in hits]
def answer(q:str,texts:list)->tuple:
 qw=set(words(q.encode()));W={w:RL.idf(w) for w in qw};best=max(((sum(W[w] for w in qw&set(words(x.encode()))),x) for _,t,_ in texts for x in RL.sents(t)),default=(0.0,""))[1];c=RL.cands(RL.qtype(q),best,q) if best else []
 return best,(c[0][0] if c else "")
def lead(t:str,x:str,cap:int=320)->tuple:
 p=DEFAULT_FILTER.clean(next((y.strip() for y in x.split("\n")[:4] if len(y.strip())>=40),""))[0];pr=bool(BORN.search(p[:300]) or OCC.search(p[:250]));g=" ".join(m.group(0) for m in [BORN.search(p[:300])] if m);p=re.sub(r",(?:\s*,)+",",",re.sub(r"\s+([,.;:])",r"\1",re.sub(r"\(\s*(?:[;,]\s*)+","(",re.sub(r"\s*\([^()\d]*\)","",p)) if pr else PAREN.sub("",p)));s=RL.sents(p);bt=re.sub(r"\s*\([^()]*\)$","",t);tw=set(words(bt.encode()))
 ok=bool(s) and len(s[0])<=cap and bool(COP.search(s[0][:250])) and not NAVX.search(s[0]) and bool(tw) and len(tw&set(words(s[0].encode())))*2>=len(tw) and not re.match(r"(?:Lists? of|Index of|Outline of|Timeline of|Glossary of)\b",t)
 m=re.search(r"\b(?:is|was|are|were)\s+((?:an?|the|one of)\s[^.;:]{2,120})",s[0]) if s else None;return (s[0]+(" "+s[1] if len(s)>1 and len(s[0])+len(s[1])<cap else "") if ok else ""),pr,(g+" "+re.split(r",| who | which | that | known | best | whose ",m.group(1))[0] if m else g).strip(),bt
def fstore(g:str)->dict:return {k:np.load(os.path.join(g,k+".npy"),mmap_mode="r") for k in ("off","raw","pg")}|{"blob":np.memmap(os.path.join(g,"blob.bin"),np.uint8,"r")}
def fdoc(f:dict,d:int)->tuple:
 b=pa.decompress(f["blob"][f["off"][d]:f["off"][d+1]].tobytes(),int(f["raw"][d]),codec="zstd",asbytes=True).decode("utf-8","replace");t,_,x=b.partition("\x00");return t,x
class Titles:
 def __init__(self,root:str=FETCH,srcs:tuple=("wikipedia",)):
  self.z=[(s,dict(np.load(os.path.join(root,s,"titles.npz")))) for s in srcs if os.path.exists(os.path.join(root,s,"titles.npz"))];self.f={}
 def doc(self,t:str)->tuple:
  h=np.uint64(th64(t))
  for s,z in self.z:
   j=int(np.searchsorted(z["h"],h));D=[fdoc(self.f.setdefault(g,fstore(g)),int(z["doc"][i])) for i in range(j,int(np.searchsorted(z["h"],h,"right"))) for g in [os.path.join(FETCH,s,"%05d"%int(z["seg"][i]))]]
   if D:return next((d for d in D if d[0]==t),next((d for d in D if d[0].lower()==t.lower()),None))
  return None
 def gloss(self,t:str)->str:
  d=self.doc(t);return (lead(*d)[2] or "?") if d else None
def wurl(t:str,s:str="wikipedia")->str:return "https://en.wikipedia.org/wiki/"+urllib.parse.quote(t.replace(" ","_"))
def th64(t:str)->int:return int.from_bytes(hashlib.blake2b(norm(t).encode(),digest_size=8).digest(),"little")
def norm(s:str)->str:return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9 ]"," ",s.lower())).strip()
