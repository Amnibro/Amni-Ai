import os,json,zlib,hashlib,importlib.util
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from amni.compute.ray_field import RayField,LEGACY_PACK,BASE_PACK,MAXB,_H0
from amni.compute.reffelt4 import encode_ids_to_rgba2,decode_rgba2_to_ids
PAGE=4096;CMAX=2*PAGE+8;TOT=1<<16;TOP=1<<24;BOT=1<<16;M32=(1<<32)-1;QK=64;DMAX=QK*24;EXPC=1057095000;START=10;MAGIC=b"AKS1";ATW=4096;SEPS=(b"\n\n",b"\n",b". ",b" ");CODER="ksc1";TIERS=["verified","endorsed","unverified","rejected"];DEF={"score":0.5,"stale":False,"date":""};CW=np.arange(1,8);LEXF=("synsets.txt","words.txt","wordcounts.txt")
CODERS={"ks1-v16":(LEGACY_PACK,"big2"),"ks2-clean1":(BASE_PACK,"clean1")};DEFAULT_VER=os.environ.get("AMNI_KS_CODER","ks2-clean1")
LUT=np.array([1<<30]+[0]*DMAX,np.int64);[LUT.__setitem__(d,int(LUT[d-1])*EXPC>>30) for d in range(1,DMAX+1)]
def paginate(b:bytes,cap:int=PAGE)->list:
 out=[];i=0
 while len(b)-i>cap:j=next((k+len(s) for s in SEPS for k in [b.rfind(s,i+cap//4,i+cap)] if k>=0),None) or next(t for t in range(i+cap,i,-1) if b[t]&0xC0!=0x80);out.append(b[i:j]);i=j
 return out+[b[i:]] if i<len(b) or not out else out
def table(lg:np.ndarray)->np.ndarray:
 q=np.floor(np.nan_to_num(lg,nan=-1e4,posinf=1e4,neginf=-1e4)*QK).astype(np.int64);e=LUT[np.minimum(q.max(axis=1,keepdims=True)-q,DMAX)];return 1+(e*(TOT-256))//e.sum(axis=1,keepdims=True)
def memguard(min_avail_gb:float=None,max_rss_gb:float=None):
 m={l.split(":")[0]:int(l.split()[1]) for l in open("/proc/meminfo")};r=int(open("/proc/self/statm").read().split()[1])*os.sysconf("SC_PAGE_SIZE");lo=float(min_avail_gb if min_avail_gb is not None else os.environ.get("KS_MIN_AVAIL_GB","12"));hi=float(max_rss_gb if max_rss_gb is not None else os.environ.get("KS_MAX_RSS_GB","3"))
 if m["MemAvailable"]/2**20<lo or r/2**30>hi:raise MemoryError(f"KS memguard: MemAvailable {m['MemAvailable']/2**20:.1f} GB (min {lo}), own RSS {r/2**30:.2f} GB (max {hi})")
def crc16(b:bytes)->int:return zlib.crc32(b)&0xFFFF
def _flush(low:int,rng:int)->bytes:return next(v.to_bytes(4,"big")[:k] for k in range(5) for u in [1<<(32-8*k)] for v in [-(-low//u)*u] if v<low+rng and v<=M32)
def digest(fs:list)->str:
 h=hashlib.sha256();[h.update(hashlib.file_digest(open(f,"rb"),"sha256").digest()) for f in fs];return h.hexdigest()[:32]
def lex_files(lex:str="big2",sub:tuple=LEXF)->list:return [f for f in (os.path.join(_H0,"data",lex,x) for x in sub) if os.path.exists(f)]
def pack_files(path:str,lex:str="big2")->list:
 pre=path[:-8] if path.endswith("_sid.npy") else path.replace("_pk_q5_i8.npz","");return [f for f in ([pre+"_dp.npy",path,pre+"_sw.npy",pre+"_bias.npy"] if path.endswith("_sid.npy") else [path,pre+"_bias.npy"]) if os.path.exists(f)]+lex_files(lex,("synsets.txt","wordcounts.txt"))
def pack_id(path:str,lex:str="big2")->str:return digest(pack_files(path,lex))
class KSCodec:
 def __init__(self,pack:str=None,arch:str=None,nb:int=MAXB,impl:str=None,ver:str=None,lex:str=None,ks:str=None):
  ver=ver or (None if pack or lex else DEFAULT_VER);self.pack,self.lex=CODERS[ver] if ver else (pack or LEGACY_PACK,lex or os.environ.get("AMNI_RAY_LEX","big2"));self.ver=ver or next((k for k,v in CODERS.items() if v==(self.pack,self.lex)),"custom");self.arch=arch or os.environ.get("AMNI_RAY_ARCH","cpu");self.impl=impl or os.environ.get("AMNI_RAY_IMPL","rust" if importlib.util.find_spec("adam_ray") else "taichi");self.nb=min(nb,MAXB);self.pid=None;self.lid=None;self.stats=False;self.nll=[0.0];self._e=None;self.ks=ks or os.environ.get("AMNI_KS_IMPL","rust" if importlib.util.find_spec("adam_ks") else "python");self._k=None
 def ident(self)->dict:self.pid=self.pid or pack_id(self.pack,self.lex);self.lid=self.lid or digest(lex_files(self.lex));return {"coder":f"{CODER}:{self.impl}-{self.arch}:{os.path.basename(self.pack)}:{self.pid[:12]}","ver":self.ver,"pack":os.path.basename(self.pack),"pack_id":self.pid,"lex":self.lex,"lex_id":self.lid,"impl":self.impl,"arch":self.arch,"qk":QK,"dmax":DMAX,"expc":EXPC,"tot":TOT,"start":START,"page":PAGE}
 def eng(self):
  f=[os.path.join(_H0,"data",self.lex,x) for x in LEXF];self._e=self._e or (__import__("adam_ray").RayField(self.pack,*f,self.arch,_H0) if self.impl=="rust" else (lambda r:(setattr(r,"wcnt_path",f[2]),r)[1])(RayField(self.pack,f[0],f[1],self.arch))).load();return self._e
 def run(self,lens:list,data:list,dec:bool=False)->list:
  r=self.eng();L=np.asarray(lens,np.int64);res=[b""]*len(L);od=[int(i) for i in np.argsort(-L,kind="stable") if L[i]>0]
  for w in range(0,len(od),self.nb):
   g=np.array(od[w:w+self.nb]);k=len(g);n=L[g];W=int(n[0]);src=np.zeros((k,(CMAX if dec else W)+8),np.uint8);[src.__setitem__((j,slice(0,len(data[i]))),np.frombuffer(data[i],np.uint8)) for j,i in enumerate(g)]
   out=np.zeros((k,W if dec else CMAX),np.uint8);z=np.zeros(k,np.int64);low,op,row=z.copy(),z.copy(),np.arange(k);rng=z+M32;ip=z+4;code=(src[:,:4].astype(np.int64)*np.array([1<<24,1<<16,1<<8,1])).sum(axis=1);lg=np.asarray(r.prime(b"",k),np.float64)
   for t in range(W):
    a=np.arange(len(row));cs=np.cumsum(table(lg),axis=1);T=cs[:,-1];rr=rng//T
    s=(cs<=np.minimum(((code-low)&M32)//rr,T-1)[:,None]).sum(axis=1) if dec else src[row,t].astype(np.int64);hi=cs[a,s];f=hi-np.where(s>0,cs[a,np.maximum(s-1,0)],0);low=(low+rr*(hi-f))&M32;rng=rr*f
    dec and out.__setitem__((row,t),s);self.stats and self.nll.__setitem__(0,self.nll[0]+float((np.log(np.exp(lg-lg.max(axis=1,keepdims=True)).sum(axis=1))+lg.max(axis=1)-lg[a,s]).sum())/np.log(2))
    while True:
     u=(low^(low+rng))<TOP;b=~u&(rng<BOT);rng=np.where(b,(-low)&(BOT-1),rng);mm=u|b
     if not mm.any():break
     x=np.nonzero(mm)[0]
     if dec:code[x]=((code[x]<<8)|src[row[x],ip[x]])&M32;ip[x]+=1
     else:out[row[x],op[x]]=low[x]>>24;op[x]+=1
     low[x]=(low[x]<<8)&M32;rng[x]=(rng[x]<<8)&M32
    fin=n[row]==t+1
    for j in np.nonzero(fin)[0]:res[g[row[j]]]=out[row[j],:n[row[j]]].tobytes() if dec else out[row[j],:op[j]].tobytes()+_flush(int(low[j]),int(rng[j]))
    kp=np.nonzero(~fin)[0]
    if not len(kp):break
    row,low,rng,op,ip,code=row[kp],low[kp],rng[kp],op[kp],ip[kp],code[kp];lg=np.asarray(r._advance(kp.tolist(),(out if dec else src)[row,t].tolist(),True),np.float64)
  return res
 def rs(self):
  self._k=self._k or (self.ks=="rust" and self.impl=="rust" and self.arch=="cpu" and importlib.util.find_spec("adam_ks") is not None and __import__("adam_ks").Codec(self.pack,os.path.join(_H0,"data",self.lex),self.ver,int(os.environ.get("AMNI_KS_THREADS","0")))) or None;return self._k
 def encode(self,pages:list)->list:k=self.rs();return self.run([len(p) for p in pages],pages) if k is None else (lambda r:(self.nll.__setitem__(0,self.nll[0]+r[1]),r[0])[1])(k.encode([bytes(p) for p in pages],self.stats))
 def decode(self,lens:list,codes:list)->list:k=self.rs();return self.run(lens,codes,True) if k is None else k.decode([int(x) for x in lens],[bytes(c) for c in codes])
def _dir_pack(clen:np.ndarray,nby:np.ndarray,crc:np.ndarray)->np.ndarray:
 a=encode_ids_to_rgba2(clen.astype(np.int64)+83521*nby.astype(np.int64)).reshape(-1,8);b=encode_ids_to_rgba2(crc.astype(np.int64)).reshape(-1,8)[:,:4];a[:,7]=(-(a[:,:7].astype(np.int64)*CW).sum(axis=1)-(b.astype(np.int64)*np.arange(8,12)).sum(axis=1))%17;return np.concatenate([a,b],axis=1)
def _dir_unpack(d:np.ndarray)->tuple:
 bad=(d>16).any() or ((d[:,:7].astype(np.int64)*CW).sum(axis=1)+d[:,7].astype(np.int64)+(d[:,8:].astype(np.int64)*np.arange(8,12)).sum(axis=1))%17!=0
 if np.any(bad):raise ValueError("KS index corrupt: GF(17) check digit mismatch")
 v=decode_rgba2_to_ids(np.concatenate([d[:,:7],np.zeros((len(d),1),np.uint8)],axis=1));w=decode_rgba2_to_ids(np.concatenate([d[:,8:],np.zeros((len(d),4),np.uint8)],axis=1));return v%83521,v//83521,w
def atlas(d:np.ndarray)->np.ndarray:
 px=d.reshape(-1,4);h=-(-len(px)//ATW);a=np.zeros((max(h,1),ATW,4),np.uint8);a.reshape(-1,4)[:len(px)]=px;return a
def write_store(out:str,docs,codec:KSCodec,chunk:int=4096,log=None,src:str="wikipedia")->dict:
 if isinstance(docs,str) and codec.rs() is not None:return json.loads(codec.rs().encode_store(docs,out,src,chunk))
 docs=({"src":src,"title":r["title"],"text":r["text"].encode(),"tier":"endorsed","score":r["score"],"stale":r["stale"],"date":r["date"],"evidence":{"urls":[r["url"]],"agree":1,"refs":r["refs"],**({"license":r["license"]} if "license" in r else {}),"signals":{},"code":None}} for b in pq.ParquetFile(docs).iter_batches(batch_size=2048) for r in b.to_pylist()) if isinstance(docs,str) else docs
 os.makedirs(out,exist_ok=True);pf=open(os.path.join(out,"pages.bin"),"wb");clen=[];nby=[];crc=[];D={k:[] for k in ("src","title","first")};P={k:[] for k in ("tier","score","stale","date","evidence")};srcs={};buf=[];tb=0
 def flush():
  cs=codec.encode(buf);[pf.write(c) for c in cs];clen.extend(len(c) for c in cs);nby.extend(len(p) for p in buf);crc.extend(crc16(p) for p in buf);buf.clear();log and log(len(clen))
 for d in docs:
  ps=paginate(d["text"]);D["src"].append(srcs.setdefault(d["src"],len(srcs)));D["title"].append(d["title"]);D["first"].append(len(clen)+len(buf));buf.extend(ps);tb+=len(d["text"]);[P[k].extend([TIERS.index(d.get("tier","unverified")) if k=="tier" else (json.dumps(d.get("evidence",{}),separators=(",",":")) if k=="evidence" else d.get(k,DEF[k]))]*len(ps)) for k in P]
  len(buf)>=chunk and (memguard(),flush())
 buf and flush();pf.close();d=_dir_pack(np.array(clen,np.int64),np.array(nby,np.int64),np.array(crc,np.int64));pay=pa.Codec("zstd",compression_level=19).compress(np.ascontiguousarray(d.T).tobytes(),asbytes=True)
 hd=json.dumps({"n":len(clen),"crc32":zlib.crc32(pay),"raw":d.size,"atlas":list(atlas(d).shape),"code_bytes":int(sum(clen))}).encode();open(os.path.join(out,"dir.ptex"),"wb").write(MAGIC+len(hd).to_bytes(4,"little")+hd+pay)
 pq.write_table(pa.table({"src":np.array(D["src"],np.uint8),"title":D["title"],"first":np.array(D["first"],np.uint32)}),os.path.join(out,"docs.parquet"),compression="zstd",compression_level=19)
 pq.write_table(pa.table({"coder":np.zeros(len(clen),np.uint8),"tier":np.array(P["tier"],np.uint8),"score":np.array(P["score"],np.float32),"stale":np.array(P["stale"],bool),"date":P["date"],"evidence":P["evidence"]}),os.path.join(out,"pagemeta.parquet"),compression="zstd",compression_level=19)
 meta={"coders":[codec.ident()],"tiers":TIERS,"sources":list(srcs),"docs":len(D["title"]),"pages":len(clen),"src_bytes":tb,"code_bytes":int(sum(clen))};json.dump(meta,open(os.path.join(out,"meta.json"),"w"),indent=1);return meta
class KStore:
 def __init__(self,path:str,codec:KSCodec=None,check:bool=True):
  self.path=path;self.meta=json.load(open(os.path.join(path,"meta.json")));raw=open(os.path.join(path,"dir.ptex"),"rb").read()
  if raw[:4]!=MAGIC:raise ValueError("KS index corrupt: bad magic")
  n=int.from_bytes(raw[4:8],"little");hd=json.loads(raw[8:8+n]);pay=raw[8+n:]
  if zlib.crc32(pay)!=hd["crc32"]:raise ValueError("KS index corrupt: directory crc32 mismatch")
  d=np.frombuffer(pa.decompress(pay,hd["raw"],codec="zstd",asbytes=True),np.uint8).reshape(12,-1).T;self.clen,self.nby,self.crc=_dir_unpack(d);self.off=np.r_[0,np.cumsum(self.clen)];self.pages=np.memmap(os.path.join(path,"pages.bin"),np.uint8,"r") if self.off[-1] else np.zeros(0,np.uint8)
  if len(self.clen)!=hd["n"] or self.off[-1]!=len(self.pages) or (self.nby>PAGE).any() or (self.clen>CMAX).any():raise ValueError("KS index corrupt: directory does not match pages.bin")
  dt=pq.read_table(os.path.join(path,"docs.parquet"));self.src=dt["src"].to_numpy();self.title=dt["title"].to_pylist();self.first=dt["first"].to_numpy().astype(np.int64);self.pm=pq.read_table(os.path.join(path,"pagemeta.parquet"));self.coder=self.pm["coder"].to_numpy();self.codecs=list(codec) if isinstance(codec,(list,tuple)) else [codec or KSCodec()];self.codec=self.codecs[0];self.check=check
  if len(self.coder)!=len(self.clen) or (self.coder>=len(self.meta["coders"])).any():raise ValueError("KS index corrupt: page metadata does not match directory")
 def __len__(self)->int:return len(self.clen)
 def doc_of(self,i:int)->int:return int(np.searchsorted(self.first,i,"right")-1)
 def code(self,i:int)->bytes:return self.pages[self.off[i]:self.off[i+1]].tobytes()
 def match(self,k:int):
  m=self.meta["coders"][k];return next((c for c in self.codecs if c is not None for i in [c.ident()] if all(m.get(x,i[x] if x=="lex_id" else None)==i[x] for x in ("coder","pack_id","lex_id")+(("impl","arch") if self.check else ()))),None)
 def get(self,ids:list)->list:
  C={k:self.match(k) for k in {int(self.coder[i]) for i in ids}};bad=[i for i in ids if C[int(self.coder[i])] is None]
  if bad:raise ValueError(f"KS frozen-coder mismatch on pages {bad[:8]}: stored {self.meta['coders'][self.coder[bad[0]]]['coder']} (lex {self.meta['coders'][self.coder[bad[0]]].get('lex','big2')}) vs codecs {[c.ident()['coder']+' lex '+c.lex for c in self.codecs if c is not None]}")
  out=[b""]*len(ids);[out.__setitem__(n,x) for k,c in C.items() for j in [[n for n,i in enumerate(ids) if self.coder[i]==k]] for n,x in zip(j,c.decode([int(self.nby[ids[n]]) for n in j],[self.code(ids[n]) for n in j]))];bad=[i for i,x in zip(ids,out) if crc16(x)!=self.crc[i]]
  if bad:raise ValueError(f"KS decode crc check failed on pages {bad[:8]} (model/arch mismatch?)")
  return out
 def sizes(self)->dict:return {k:os.path.getsize(os.path.join(self.path,k)) for k in ("pages.bin","dir.ptex","docs.parquet","pagemeta.parquet","meta.json")}
