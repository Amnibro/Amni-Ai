import os,re,json,math,time,array
import numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_QA=os.path.join(_H0,"exports","gf17_continuum","ray_qa_index.npz");STOP=set("the a an and or of to in on for with is are was were be been being do does did what why how whom which when where can could would should will shall may might must i you me my your it its this that these those there their they them he she his her we our us as at by from about into than then so if not no yes please tell give explain describe there here just also some any".split())
_TOK=re.compile(rb"[a-z][a-z']{1,}");BAD=re.compile(rb"(?i)(i'?m sorry|i am sorry|as an ai|language model|i cannot|i can't|i don't have|not sure what you)");NAME=re.compile(rb"(?i)open ?assistant");JUNK=re.compile(rb"(?i)(\bum+\b|\blol\b|\blmao\b|\bhaha|\bidk\b|dunno|'s answer|answer above|\basker\b|you people|\bu r\b|!!!|\?\?)");CHORE=re.compile(rb"(?i)(\b(divide|multiply|subtract|calculate|compute|solve for|convert)\b.{0,20}\d|next (number|term) in (the|this) (sequence|series)|\d+\s*[-+*/x\^]\s*\d+)");PROPER=re.compile(r"(?<!^)(?<![.?!] )\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)");ASSIST=re.compile(rb"(?i)(how (may|can) i (help|assist)|what can i do for you)");HARD=re.compile(r"(?i)\b(\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|last|latest|earliest|oldest|newest|youngest|most|least|largest|smallest|biggest|tallest|highest|lowest|longest|shortest|best|worst|fastest|slowest)\b|\b(?:war|part|chapter|phase|stage|henry|louis|george|edward|elizabeth|charles|richard|james|william)\s+([ivx]{1,4})\b")
def stem(w:bytes)->bytes:return w[:-2] if w.endswith(b"oes") else (w[:-3]+b"y" if w.endswith(b"ies") and len(w)>4 else (w[:-1] if w.endswith(b"s") and not w.endswith((b"ss",b"us",b"is")) and len(w)>3 else w))
def words(b:bytes)->list:return [stem(w) for w in _TOK.findall(b.lower()) if w.decode() not in STOP]
def pairs(p:str):
 lo="yahoo" in os.path.basename(p)
 if p.endswith(".jsonl"):
  for l in open(p,"rb"):r=json.loads(l);yield r["q"].encode(),r["a"].encode(),int(r.get("s",2)),(r.get("t") or "").encode()
  return
 for m in re.finditer(rb"User: ([^\n]{4,400})\nAdam: ([^\n]{8,700})\n\n",open(p,"rb").read()):
  a=re.sub(rb"\s+",b" ",re.sub(rb"https?://\S+|www\.\S+",b"",NAME.sub(b"Adam",m.group(2)).replace(b"\\n",b" "))).strip()
  if not BAD.search(a[:160]) and not (lo and (len(a)<40 or JUNK.search(a) or not re.match(rb"[A-Z0-9\"]",a))) and not ASSIST.search(m.group(1)) and not CHORE.search(m.group(1)) and m.group(1).count(b"?")<=1 and not re.search(rb"\?\s*\S.{8,}",m.group(1)):yield m.group(1),a,1 if lo else 0,b""
def build(paths:list,out:str=DEFAULT_QA)->dict:
 vocab={};seen=set();qs=[];src=array.array("b");ans=bytearray();off=array.array("q",[0]);ref=bytearray();roff=array.array("q",[0]);qlen=array.array("f");P=[(array.array("q"),array.array("i")) for _ in range(3)];t0=time.time();pk=0
 for p in paths:
  for q,a,s,r in pairs(p):
   k=q.lower()
   if k in seen:continue
   seen.add(k);n=len(qs);qs.append(q);src.append(s);ans+=a;off.append(len(ans));ref+=r;roff.append(len(ref));ww=words(q);qw=set(ww);qlen.append(max(1,len(qw)))
   for j,d in enumerate((qw,set(words(a[:240])),{x+b" "+y for x,y in zip(ww,ww[1:])})):P[j][0].extend(vocab.setdefault(w,len(vocab)) for w in d);P[j][1].extend([n]*len(d))
   n%65536==0 and (pk:=max(pk,int(open("/proc/self/statm").read().split()[1])*os.sysconf("SC_PAGE_SIZE")))
 n=len(qs);V=len(vocab);seen=None;S=[];m=np.frombuffer(src,np.int8)!=2
 for r,c in P:r=np.frombuffer(r,np.int64);c=np.frombuffer(c,np.int32);o=np.argsort(r,kind="stable");S.append((np.searchsorted(r[o],np.arange(V+1)).astype(np.int64),c[o],np.bincount(r[m[c]],minlength=V)))
 (p1,c1,d1),(p2,c2,d2),(p3,c3,_)=S;P=None;idf=np.log((int(m.sum())+1)/(d1+d2*0.3+0.5)).astype(np.float32);widf=np.log((n+1)/(np.diff(p1)+np.diff(p2)*0.3+0.5)).astype(np.float32);pk=max(pk,int(open("/proc/self/statm").read().split()[1])*os.sysconf("SC_PAGE_SIZE"))
 np.savez(out,vocab=np.array(list(vocab.keys()),dtype=object),ptr=p1,cols=c1,aptr=p2,acols=c2,bptr=p3,bcols=c3,idf=idf,qlen=np.frombuffer(qlen,np.float32),ans=np.frombuffer(bytes(ans),np.uint8),off=np.frombuffer(off,np.int64),qs=np.array(qs,dtype=object),src=np.frombuffer(src,np.int8),**({"ref":np.frombuffer(bytes(ref),np.uint8),"roff":np.frombuffer(roff,np.int64),"widf":widf} if len(ref) else {}))
 return {"pairs":n,"vocab":V,"postings":int(c1.size+c2.size+c3.size),"answer_bytes":int(off[-1]),"ref_pairs":int(np.count_nonzero(np.diff(np.frombuffer(roff,np.int64)))),"peak_rss_gb":round(pk/2**30,2),"s":round(time.time()-t0,1)}
class QA:
 def __init__(self,path:str=None):
  z=np.load(path or os.environ.get("AMNI_RAY_QA",DEFAULT_QA),allow_pickle=True);self.vocab={w:i for i,w in enumerate(z["vocab"])};self.ptr=z["ptr"];self.cols=z["cols"];self.aptr=z["aptr"];self.acols=z["acols"];self.bptr=z["bptr"];self.bcols=z["bcols"];self.idf=z["idf"];self.qlen=z["qlen"];self.ans=z["ans"];self.off=z["off"];self.qs=z["qs"];self.src=z["src"] if "src" in z.files else np.zeros(len(self.qlen),np.int8);self.rb,self.ro=(z["ref"],z["roff"]) if "roff" in z.files else (np.zeros(0,np.uint8),np.zeros(len(self.qlen)+1,np.int64));self.wk=self.src==2;self.widf=z["widf"] if "widf" in z.files else self.idf
 def overlap(self,q:str,c:str,wk:bool=False)->float:
  a=set(words(q.encode()));b=set(words(c.encode()));w=lambda x:float((self.widf if wk else self.idf)[self.vocab[x]]) if x in self.vocab else 1.0
  i=sum(w(x) for x in a&b);pn=0.6 if any(p.lower() not in q.lower() for p in PROPER.findall(c)) else 1.0;return pn*i/max(sum(w(x) for x in a|b),1e-9),i/max(sum(w(x) for x in a),1e-9),pn*i/max(sum(w(x) for x in b),1e-9)
 def answer(self,i:int)->bytes:return self.ans[self.off[i]:self.off[i+1]].tobytes()
 def ref(self,i:int)->str:return self.rb[self.ro[i]:self.ro[i+1]].tobytes().decode("utf-8","replace")
 def search(self,q:str,k:int=3)->list:
  qw=words(q.encode());ids=[self.vocab[w] for w in set(qw) if w in self.vocab];bi=[self.vocab[a+b" "+b] for a,b in zip(qw,qw[1:]) if a+b" "+b in self.vocab]
  if not ids or any(w not in self.vocab and len(w)>=4 for w in qw):return []
  sc=np.zeros(len(self.qlen),np.float32);tot=float(sum(self.idf[i] for i in ids))
  for i in ids:np.add.at(sc,self.cols[self.ptr[i]:self.ptr[i+1]],self.idf[i]);np.add.at(sc,self.acols[self.aptr[i]:self.aptr[i+1]],0.4*self.idf[i])
  for b in bi:np.add.at(sc,self.bcols[self.bptr[b]:self.bptr[b+1]],1.5*float(np.mean([self.idf[i] for i in ids])))
  r=max(ids,key=lambda i:self.idf[i]);must=np.zeros(len(self.qlen),bool);must[self.cols[self.ptr[r]:self.ptr[r+1]]]=True;must[self.acols[self.aptr[r]:self.aptr[r+1]]]=True;sc=sc/np.sqrt(self.qlen*len(ids));sc=np.where(must,sc,sc*0.6) if must.any() else sc;T=lambda s:(lambda i:i[np.lexsort((i,-s[i]))][:k])(np.nonzero((s>=-np.partition(-s,min(k,len(s))-1)[min(k,len(s))-1])&(s>0))[0]);top=np.union1d(T(np.where(self.wk,0,sc)),T(np.where(self.wk,sc,0))) if self.wk.any() else T(sc);top=top[np.lexsort((top,-sc[top]))]
  return [(float(sc[t]),float(sc[t]*math.sqrt(self.qlen[t]*len(ids))/max(tot,1e-9)),self.qs[t].decode("utf-8","replace"),self.answer(int(t)),int(self.src[t]),self.ref(int(t))) for t in top if sc[t]>0]
def hard(q:str,c:str,a:bytes=b"")->bool:
 n=lambda t:{(x or y).lower() for x,y in HARD.findall(t)};t=a.decode("utf-8","replace");ql=q.strip().lower()
 need=re.search(r"\d{3,4}|\b(january|february|march|april|may|june|july|august|september|october|november|december|century|decade|bc|ad|ago)\b",t,re.I) if re.match(r"(when|what year|which year|what date|in what year|in which year)\b",ql) else (re.search(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|million|billion|dozen|half|several|few)\b",t,re.I) if re.match(r"how (many|much|old|long|far|tall|big|high|deep|fast)\b|what (percentage|percent|number)\b",ql) else (re.search(r"(?<![.!?] )(?<!^)\b[A-Z][a-z]+",t) if re.match(r"who(m|se)?\b",ql) else True))
 return n(q)==n(c) and bool(need)
GEN={b"who",b"mean",b"meaning",b"definition",b"define",b"defined",b"explain",b"know",b"called",b"thing",b"term",b"word",b"stand"}
FRAME=re.compile(r"(?i)^(?:(?:what|who) (?:is|are|was|were) (?:the |an? )?(.+?)|what does (?:the |an? )?(.+?) mean)\?$")
def toks(t:str)->list:return [stem(w) for w in re.findall(rb"[a-z0-9']+",t.lower().encode())]
def wgate(q:str,c:str,a:bytes)->bool:
 m=FRAME.match(c.strip());t=toks(m.group(1) or m.group(2)) if m else [];u=toks(q);cw=set(words(c.encode()));aw=set(words(a))
 return bool(t) and any(u[i:i+len(t)]==t for i in range(len(u)-len(t)+1)) and all(w in aw or (len(w)>=5 and any(x[:5]==w[:5] for x in aw)) for w in set(words(q.encode()))-cw-GEN)
def pick(e,qa:QA,q:str)->list:
 c=qa.search(q,int(os.environ.get("QA_K","20")))
 if not c:return []
 a=[r[3][:int(os.environ.get("QA_SPAN","160"))] for r in c];n=np.array([max(1,len(x)) for x in a],float);pc=e.score([f"User: {q}\nAdam: ".encode()]*len(c),a);pu=e.score([b"User: \nAdam: "]*len(c),a);pmi=(pc-pu)/n;ov=np.array([qa.overlap(q,r[2],r[4]==2) for r in c]);ok=(ov[:,1]>=float(os.environ.get("QA_REC","0.8")))&(ov[:,2]>=float(os.environ.get("QA_PREC","0.55")))&np.array([hard(q,r[2],r[3][:300]) and (r[4]!=2 or wgate(q,r[2],r[3])) for r in c]);f=np.where(ok,ov[:,0]+float(os.environ.get("QA_LAM","0.5"))*pmi-float(os.environ.get("QA_YAHOO","0.15"))*np.array([r[4] for r in c]),-9.0);i=int(np.argmax(f))
 return [(c[i][0],float(ov[i,0]) if ok[i] else 0.0)+tuple(c[i][2:])]
def guide(e,qa:QA,q:str,cov:float=None,full:bool=False):
 h=pick(e,qa,q);h=h[0] if h and h[0][1]>=(cov if cov is not None else float(os.environ.get("QA_COV","0.3"))) else None;return h if full or h is None else h[3]
