import os,re,math
import numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_QA=os.path.join(_H0,"exports","gf17_continuum","ray_qa_index.npz");STOP=set("the a an and or of to in on for with is are was were be been being do does did what why how whom which when where can could would should will shall may might must i you me my your it its this that these those there their they them he she his her we our us as at by from about into than then so if not no yes please tell give explain describe there here just also some any".split())
_TOK=re.compile(rb"[a-z][a-z']{1,}");BAD=re.compile(rb"(?i)(i'?m sorry|i am sorry|as an ai|language model|i cannot|i can't|i don't have|not sure what you)");NAME=re.compile(rb"(?i)open ?assistant");JUNK=re.compile(rb"(?i)(\bum+\b|\blol\b|\blmao\b|\bhaha|\bidk\b|dunno|'s answer|answer above|\basker\b|you people|\bu r\b|!!!|\?\?)");CHORE=re.compile(rb"(?i)(\b(divide|multiply|subtract|calculate|compute|solve for|convert)\b.{0,20}\d|next (number|term) in (the|this) (sequence|series)|\d+\s*[-+*/x\^]\s*\d+)");PROPER=re.compile(r"(?<!^)(?<![.?!] )\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)");ASSIST=re.compile(rb"(?i)(how (may|can) i (help|assist)|what can i do for you)")
def stem(w:bytes)->bytes:return w[:-2] if w.endswith(b"oes") else (w[:-3]+b"y" if w.endswith(b"ies") and len(w)>4 else (w[:-1] if w.endswith(b"s") and not w.endswith((b"ss",b"us",b"is")) and len(w)>3 else w))
def words(b:bytes)->list:return [stem(w) for w in _TOK.findall(b.lower()) if w.decode() not in STOP]
def build(paths:list,out:str=DEFAULT_QA)->dict:
 qs=[];ans=[];src=[]
 for si,p in enumerate(paths):
  lo="yahoo" in os.path.basename(p)
  for m in re.finditer(rb"User: ([^\n]{4,400})\nAdam: ([^\n]{8,700})\n\n",open(p,"rb").read()):a=re.sub(rb"\s+",b" ",re.sub(rb"https?://\S+|www\.\S+",b"",NAME.sub(b"Adam",m.group(2)).replace(b"\\n",b" "))).strip();(qs.append(m.group(1)),ans.append(a),src.append(1 if lo else 0)) if not BAD.search(a[:160]) and not (lo and (len(a)<40 or JUNK.search(a) or not re.match(rb"[A-Z0-9\"]",a))) and not ASSIST.search(m.group(1)) and not CHORE.search(m.group(1)) and m.group(1).count(b"?")<=1 and not re.search(rb"\?\s*\S.{8,}",m.group(1)) else None
 seen={};keep=[i for i,q in enumerate(qs) if seen.setdefault(q.lower(),i)==i];qs=[qs[i] for i in keep];ans=[ans[i] for i in keep];src=[src[i] for i in keep]
 vocab={};n=len(qs)
 def inv(docs:list)->tuple:
  rows=[];cols=[]
  for i,d in enumerate(docs):
   for w in d:rows.append(vocab.setdefault(w,len(vocab)));cols.append(i)
  rows=np.asarray(rows,np.int64);cols=np.asarray(cols,np.int32);o=np.argsort(rows,kind="stable");return rows[o],cols[o]
 qw=[set(words(q)) for q in qs];bg=[{a+b" "+b for a,b in zip(w,w[1:])} for w in (words(q) for q in qs)];aw=[set(words(a[:240])) for a in ans];r1,c1=inv(qw);r2,c2=inv(aw);r3,c3=inv(bg);V=len(vocab)
 ptr=lambda r:np.searchsorted(r,np.arange(V+1)).astype(np.int64);p1,p2,p3=ptr(r1),ptr(r2),ptr(r3);idf=np.log((n+1)/(np.diff(p1)+np.diff(p2)*0.3+0.5)).astype(np.float32)
 qlen=np.array([max(1,len(w)) for w in qw],np.float32);off=np.cumsum([0]+[len(a) for a in ans]).astype(np.int64)
 np.savez(out,vocab=np.array(list(vocab.keys()),dtype=object),ptr=p1,cols=c1,aptr=p2,acols=c2,bptr=p3,bcols=c3,idf=idf,qlen=qlen,ans=np.frombuffer(b"".join(ans),np.uint8),off=off,qs=np.array(qs,dtype=object),src=np.array(src,np.int8))
 return {"pairs":n,"vocab":V,"postings":int(c1.size+c2.size+c3.size),"answer_bytes":int(off[-1])}
class QA:
 def __init__(self,path:str=None):
  z=np.load(path or os.environ.get("AMNI_RAY_QA",DEFAULT_QA),allow_pickle=True);self.vocab={w:i for i,w in enumerate(z["vocab"])};self.ptr=z["ptr"];self.cols=z["cols"];self.aptr=z["aptr"];self.acols=z["acols"];self.bptr=z["bptr"];self.bcols=z["bcols"];self.idf=z["idf"];self.qlen=z["qlen"];self.ans=z["ans"];self.off=z["off"];self.qs=z["qs"];self.src=z["src"] if "src" in z.files else np.zeros(len(self.qlen),np.int8)
 def overlap(self,q:str,c:str)->float:
  a=set(words(q.encode()));b=set(words(c.encode()));w=lambda x:float(self.idf[self.vocab[x]]) if x in self.vocab else 1.0
  i=sum(w(x) for x in a&b);pn=0.6 if any(p.lower() not in q.lower() for p in PROPER.findall(c)) else 1.0;return pn*i/max(sum(w(x) for x in a|b),1e-9),i/max(sum(w(x) for x in a),1e-9),pn*i/max(sum(w(x) for x in b),1e-9)
 def answer(self,i:int)->bytes:return self.ans[self.off[i]:self.off[i+1]].tobytes()
 def search(self,q:str,k:int=3)->list:
  qw=words(q.encode());ids=[self.vocab[w] for w in set(qw) if w in self.vocab];bi=[self.vocab[a+b" "+b] for a,b in zip(qw,qw[1:]) if a+b" "+b in self.vocab]
  if not ids or any(w not in self.vocab and len(w)>=4 for w in qw):return []
  sc=np.zeros(len(self.qlen),np.float32);tot=float(sum(self.idf[i] for i in ids))
  for i in ids:np.add.at(sc,self.cols[self.ptr[i]:self.ptr[i+1]],self.idf[i]);np.add.at(sc,self.acols[self.aptr[i]:self.aptr[i+1]],0.4*self.idf[i])
  for b in bi:np.add.at(sc,self.bcols[self.bptr[b]:self.bptr[b+1]],1.5*float(np.mean([self.idf[i] for i in ids])))
  r=max(ids,key=lambda i:self.idf[i]);must=np.zeros(len(self.qlen),bool);must[self.cols[self.ptr[r]:self.ptr[r+1]]]=True;must[self.acols[self.aptr[r]:self.aptr[r+1]]]=True;sc=sc/np.sqrt(self.qlen*len(ids));sc=np.where(must,sc,sc*0.6) if must.any() else sc;top=np.argpartition(-sc,min(k,len(sc)-1))[:k];top=top[np.argsort(-sc[top])]
  return [(float(sc[t]),float(sc[t]*math.sqrt(self.qlen[t]*len(ids))/max(tot,1e-9)),self.qs[t].decode("utf-8","replace"),self.answer(int(t)),int(self.src[t])) for t in top if sc[t]>0]
def pick(e,qa:QA,q:str)->list:
 c=qa.search(q,int(os.environ.get("QA_K","20")))
 if not c:return []
 a=[r[3][:int(os.environ.get("QA_SPAN","160"))] for r in c];n=np.array([max(1,len(x)) for x in a],float);pc=e.score([f"User: {q}\nAdam: ".encode()]*len(c),a);pu=e.score([b"User: \nAdam: "]*len(c),a);pmi=(pc-pu)/n;ov=np.array([qa.overlap(q,r[2]) for r in c]);ok=(ov[:,1]>=float(os.environ.get("QA_REC","0.8")))&(ov[:,2]>=float(os.environ.get("QA_PREC","0.55")));f=np.where(ok,ov[:,0]+float(os.environ.get("QA_LAM","0.5"))*pmi-float(os.environ.get("QA_YAHOO","0.15"))*np.array([r[4] for r in c]),-9.0);i=int(np.argmax(f))
 return [(c[i][0],float(ov[i,0]) if ok[i] else 0.0)+tuple(c[i][2:])]
def guide(e,qa:QA,q:str,cov:float=None)->bytes:
 h=pick(e,qa,q);return h[0][3] if h and h[0][1]>=(cov if cov is not None else float(os.environ.get("QA_COV","0.3"))) else None
