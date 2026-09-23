import os,math,codecs,threading
import numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PACK=os.path.join(_H0,"exports","gf17_continuum","rayhash_v16_big2_L21_e20_hop4_q5_s150k_pk_q5_i8.npz")
V=256;WM=(1<<20)-1;NI=30;BASE=21;HOPS=4;KS=3;MAXB=256;TOPIC_A=0.99
_lock=threading.Lock();_eng=None
def _unpack5(b:np.ndarray,n:int)->np.ndarray:
 u=np.packbits(np.concatenate([np.zeros((b.size//7,8,1),np.uint8),np.unpackbits(b.reshape(-1,7),axis=1).reshape(-1,8,7)],axis=2),axis=2).reshape(-1)[:n].astype(np.int16);return np.stack([u//25-2,(u//5)%5-2,u%5-2],axis=1).astype(np.int8)
def load_pack(path:str)->dict:
 z=np.load(path);L,T,K=[int(v) for v in z["sshape"]];dsh=[int(v) for v in z["dshape"]];n=int(z["trit_n"]);cells=L*T
 q=(z["dense_digits"].astype(np.int16)-128).astype(np.float32).reshape(-1,8);sc=z["dense_scale"].astype(np.float32).reshape(-1,1);dp=(q/127.0*sc).reshape(-1,V).astype(np.float16).astype(np.float32)
 t=_unpack5(z["trits"],n).reshape(L,T,K).astype(np.float32);ub=np.unpackbits(z["scale4"])[:cells*4].reshape(cells,4);s4=(ub*np.array([8,4,2,1],np.uint8)).sum(axis=1).astype(np.float32);lo=float(z["scale_lo"]);hi=float(z["scale_hi"])
 scale=np.where(s4>0,np.power(2.0,lo+(s4-1)/14.0*(hi-lo)),0.0).reshape(L,T,1);sw=(t*(0.5*scale)).astype(np.float16)
 bp=path.replace("_pk_q5_i8.npz","_bias.npy");bias=np.load(bp).astype(np.float32) if os.path.exists(bp) else np.zeros(V,np.float32)
 return {"dp":dp,"sid":z["ids"].reshape(L,T,K),"sw":sw,"bias":bias,"L":L,"T":T,"K":K,"dshape":dsh}
def load_syn(path:str)->tuple:
 ts=np.zeros(WM+1,dtype=np.int32);th=np.zeros(WM+1,dtype=np.int32)
 for l in (open(path,"rb").read().splitlines() if os.path.exists(path) else []):
  parts=l.split(b" ");h=0
  if len(parts)!=3:continue
  for c in parts[0]:h=(h*83492791+c+1)&0xFFFFFFFF
  ts[h&WM]=int(parts[1])&0x7FFFFFFF;th[h&WM]=int(parts[2])&0x7FFFFFFF
 return ts,th
class RayField:
 def __init__(self,path:str=None,syn:str=None,words:str=None,arch:str=None):
  self.path=path or os.environ.get("AMNI_RAY_PACK",DEFAULT_PACK);self.syn=syn or os.path.join(_H0,"data","big2","synsets.txt");self.words_path=words or os.path.join(_H0,"data","big2","words.txt");self.arch=arch or os.environ.get("AMNI_RAY_ARCH","vulkan");self.ready=False;self.words=set();self.pref=set();self.ok=np.array([(32<=c<127) or c in (9,10) for c in range(V)])
 def load(self):
  if self.ready:return self
  import taichi as ti
  ti.init(arch=getattr(ti,self.arch),log_level=ti.ERROR,offline_cache=True);self.ti=ti;pk=load_pack(self.path);L,T,K=pk["L"],pk["T"],pk["K"];self.L,self.T=L,T;TE=T
  DP=ti.field(ti.f32,shape=(pk["dp"].shape[0],V));SID=ti.field(ti.u8,shape=(L,T,K));SW=ti.field(ti.f16,shape=(L,T,K));BI=ti.field(ti.f32,shape=V);SYN=ti.field(ti.i32,shape=WM+1);HYP=ti.field(ti.i32,shape=WM+1);PUNCT=ti.field(ti.i32,shape=V);WORDCH=ti.field(ti.i32,shape=V)
  ST=ti.field(ti.u32,shape=(2,MAXB,NI));TF=ti.field(ti.f32,shape=(2,MAXB,2));LOG=ti.field(ti.f32,shape=(MAXB,V));PAR=ti.field(ti.i32,shape=MAXB);BYT=ti.field(ti.i32,shape=MAXB)
  DP.from_numpy(pk["dp"]);SID.from_numpy(pk["sid"]);SW.from_numpy(pk["sw"]);BI.from_numpy(pk["bias"]);ts,th=load_syn(self.syn);SYN.from_numpy(ts);HYP.from_numpy(th)
  pn=np.zeros(V,np.int32);pn[[ord(c) for c in ".,;:!?()[]{}<>=\"'`#-"]]=1;PUNCT.from_numpy(pn);wn=np.zeros(V,np.int32);wn[[ord(c) for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"]]=1;WORDCH.from_numpy(wn)
  @ti.func
  def hmix2(a:ti.u32,b:ti.u32,salt:ti.u32)->ti.u32:
   h=(a*ti.u32(73856093))^(b*ti.u32(19349663))^(salt*ti.u32(83492791));h^=h>>ti.u32(13);h*=ti.u32(0x5BD1E995);h^=h>>ti.u32(15);return h
  @ti.func
  def slot(u:ti.u32,v:ti.u32,l:ti.i32)->ti.i32:
   return ti.cast(hmix2(u,v,ti.cast(l,ti.u32)+ti.u32(1))%ti.u32(TE),ti.i32)
  @ti.func
  def cell(b:ti.i32,l:ti.i32,e:ti.i32,w:ti.f32):
   for s in ti.static(range(KS)):LOG[b,ti.cast(SID[l,e,s],ti.i32)]+=w*ti.cast(SW[l,e,s],ti.f32)
  @ti.kernel
  def reset(nb:ti.i32,dst:ti.i32):
   for b,i in ti.ndrange(nb,NI):ST[dst,b,i]=ti.u32(1) if i==24 else ti.u32(0)
   for b,i in ti.ndrange(nb,2):TF[dst,b,i]=0.0
  @ti.kernel
  def step(nb:ti.i32,src:ti.i32,dst:ti.i32,want:ti.i32):
   for b in range(nb):
    pa=PAR[b];bb=BYT[b]
    for i in ti.static(range(NI)):ST[dst,b,i]=ST[src,pa,i]
    tx=TF[src,pa,0];ty=TF[src,pa,1];wh=ST[dst,b,16];pw1=ST[dst,b,17];pw2=ST[dst,b,18];pw3=ST[dst,b,19];pw4=ST[dst,b,20];s1=ST[dst,b,21];s2=ST[dst,b,22];g1=ST[dst,b,23];g2=ST[dst,b,26];atstart=ti.cast(ST[dst,b,24],ti.i32);indent=ti.cast(ST[dst,b,25],ti.i32);col=ti.cast(ST[dst,b,27],ti.i32);incode=ti.cast(ST[dst,b,28],ti.i32);lp=ST[dst,b,29];lastp=ti.cast(lp&ti.u32(0xFFFF),ti.i32);tick=ti.cast(lp>>ti.u32(16),ti.i32)
    isw=WORDCH[bb]==1;ends=(not isw) and wh!=ti.u32(0)
    if ends:
     pw4=pw3;pw3=pw2;pw2=pw1;pw1=wh;s2=s1;g2=g1;s1=ti.cast(SYN[ti.cast(pw1&ti.u32(WM),ti.i32)],ti.u32);g1=ti.cast(HYP[ti.cast(pw1&ti.u32(WM),ti.i32)],ti.u32)
     a=ti.cast(pw1%ti.u32(360),ti.f32)*(2*math.pi/360.0);tx=TOPIC_A*tx+ti.cos(a)*(1-TOPIC_A)*8;ty=TOPIC_A*ty+ti.sin(a)*(1-TOPIC_A)*8
    wh=(wh*ti.u32(83492791)+ti.cast(bb,ti.u32)+ti.u32(1)) if isw else ti.u32(0)
    nl=bb==10;col=0 if nl else col+1
    if nl:atstart=1;indent=0
    elif bb==32:
     if atstart==1:indent+=1
    else:atstart=0
    tick=tick+1 if bb==96 else (0 if nl else tick)
    if tick==3:incode=1-incode;tick=0
    if PUNCT[bb]==1:lastp=bb
    elif nl:lastp=10
    for i in ti.static(range(15,0,-1)):ST[dst,b,i]=ST[dst,b,i-1]
    ST[dst,b,0]=ti.cast(bb,ti.u32);ST[dst,b,16]=wh;ST[dst,b,17]=pw1;ST[dst,b,18]=pw2;ST[dst,b,19]=pw3;ST[dst,b,20]=pw4;ST[dst,b,21]=s1;ST[dst,b,22]=s2;ST[dst,b,23]=g1;ST[dst,b,26]=g2;ST[dst,b,24]=ti.cast(atstart,ti.u32);ST[dst,b,25]=ti.cast(indent,ti.u32);ST[dst,b,27]=ti.cast(col,ti.u32);ST[dst,b,28]=ti.cast(incode,ti.u32);ST[dst,b,29]=ti.cast(lastp,ti.u32)|(ti.cast(tick,ti.u32)<<ti.u32(16));TF[dst,b,0]=tx;TF[dst,b,1]=ty
    if want==1:
     b1=ST[dst,b,0];b2=ST[dst,b,1];b3=ST[dst,b,2];b4=ST[dst,b,3];b5=ST[dst,b,4];b6=ST[dst,b,5];b7=ST[dst,b,6];b8=ST[dst,b,7];b9=ST[dst,b,8];b10=ST[dst,b,9];b11=ST[dst,b,10];b12=ST[dst,b,11];b13=ST[dst,b,12];b14=ST[dst,b,13];b15=ST[dst,b,14];b16=ST[dst,b,15]
     r1=ti.cast(b1,ti.i32);r2=256+ti.cast(b2*ti.u32(256)+b1,ti.i32)
     for c in range(V):LOG[b,c]=BI[c]+DP[r1,c]+DP[r2,c]
     cell(b,2,slot(hmix2(b3,b2,ti.u32(7)),b1,2),1.0);cell(b,3,slot(hmix2(hmix2(b4,b3,ti.u32(9)),b2,ti.u32(11)),b1,3),1.0);cell(b,4,slot(wh,pw1&ti.u32(255),4),1.0);cell(b,5,slot(pw1&ti.u32(4095),pw2&ti.u32(4095),5),1.0)
     cell(b,6,slot(hmix2(ti.cast(ti.min(indent,32),ti.u32),ti.cast(incode*512+lastp,ti.u32),ti.u32(13)),ti.cast(ti.min(col,255),ti.u32),6),1.0)
     h5=hmix2(hmix2(b5,b4,ti.u32(17)),hmix2(b3,b2,ti.u32(19)),ti.u32(21));cell(b,7,slot(h5,b1,7),1.0);h6=hmix2(h5,b6,ti.u32(23));cell(b,8,slot(h6,b1,8),1.0);h8=hmix2(hmix2(h6,b7,ti.u32(29)),b8,ti.u32(31));cell(b,9,slot(h8,b1,9),1.0)
     cell(b,10,slot(hmix2(pw1,pw2,ti.u32(37))&ti.u32(65535),pw3&ti.u32(4095),10),1.0);h10=hmix2(hmix2(h8,b9,ti.u32(41)),b10,ti.u32(43));cell(b,11,slot(h10,b1,11),1.0);h12=hmix2(hmix2(h10,b11,ti.u32(47)),b12,ti.u32(53));cell(b,12,slot(h12,b1,12),1.0)
     h16=hmix2(hmix2(hmix2(hmix2(h12,b13,ti.u32(59)),b14,ti.u32(61)),b15,ti.u32(67)),b16,ti.u32(71));cell(b,13,slot(h16,b1,13),1.0);cell(b,14,slot(hmix2(hmix2(pw1,pw2,ti.u32(73)),hmix2(pw3,pw4,ti.u32(79)),ti.u32(83)),wh&ti.u32(65535),14),1.0)
     cell(b,15,slot(wh,ti.u32(0),15),1.0);cell(b,16,slot(hmix2(wh,pw1,ti.u32(89)),ti.u32(0),16),1.0);cell(b,17,slot(s1,s2,17),1.0);cell(b,18,slot(g1,g2,18),1.0);cell(b,19,slot(wh,s1,19),1.0)
     ang=ti.atan2(ty,tx)/(2*math.pi);ang=ang-ti.floor(ang);uu=ang*256.0;vv=ti.min(ti.sqrt(tx*tx+ty*ty),7.99)/8.0*256.0;u0=ti.floor(uu);v0=ti.floor(vv);fu=uu-u0;fv=vv-v0;i0=ti.cast(u0,ti.i32);j0=ti.cast(v0,ti.i32)
     for k in ti.static(range(4)):
      ii=i0+(k&1);jj=j0+(k>>1);w=(fu if k&1 else 1-fu)*(fv if k>>1 else 1-fv)
      if w!=0.0:cell(b,BASE-1,slot(ti.cast(ii%256,ti.u32),ti.cast(ti.min(jj,255),ti.u32),BASE-1),w)
     for h in ti.static(range(HOPS)):
      a0=0;a1=0;a2=0;m0=-1e30;m1=-1e30;m2=-1e30
      for c in range(V):
       v=LOG[b,c]
       if v>m0:m2=m1;a2=a1;m1=m0;a1=a0;m0=v;a0=c
       elif v>m1:m2=m1;a2=a1;m1=v;a1=c
       elif v>m2:m2=v;a2=c
      lo=ti.min(a0,ti.min(a1,a2));hi=ti.max(a0,ti.max(a1,a2));mid=a0+a1+a2-lo-hi
      cell(b,BASE+h,slot(hmix2(hmix2(ti.cast(lo*65536+mid*256+hi,ti.u32),b1,ti.u32(97+h)),b2,ti.u32(103+h)),ti.cast(h,ti.u32),BASE+h),1.0)
  self._reset,self._step,self.PAR,self.BYT,self.LOG=reset,step,PAR,BYT,LOG;self.cur=0;self.ready=True;return self
 def _advance(self,par:list,byt:list,want:bool=True)->np.ndarray:
  nb=len(byt);src=self.cur;dst=1-src;self.PAR.from_numpy(np.pad(np.asarray(par,np.int32),(0,MAXB-nb)));self.BYT.from_numpy(np.pad(np.asarray(byt,np.int32),(0,MAXB-nb)));self._step(nb,src,dst,1 if want else 0);self.cur=dst
  return self.LOG.to_numpy()[:nb].astype(np.float64) if want else None
 def prime(self,data:bytes,nb:int=1)->np.ndarray:
  self._reset(nb,self.cur);lg=None;seq=list(data) or [10]
  for i,c in enumerate(seq):lg=self._advance(list(range(nb)),[c]*nb,i==len(seq)-1)
  return lg
 def logits_for(self,data:bytes)->np.ndarray:
  self._reset(1,self.cur);return np.stack([self._advance([0],[c],True)[0] for c in data])
 def bpb(self,hold:str=None,batches:int=60,ctx:int=128,bs:int=256,seed:int=1234)->float:
  d=np.memmap(hold or os.path.join(_H0,"data","big2","hold.bin"),dtype=np.uint8,mode="r");g=np.random.default_rng(seed);tot=0.0;n=0
  for _ in range(batches):
   ix=g.integers(0,len(d)-ctx-1,bs);x=np.stack([np.asarray(d[i:i+ctx]) for i in ix]).astype(np.int64);y=np.stack([np.asarray(d[i+1:i+1+ctx]) for i in ix]).astype(np.int64);self._reset(bs,self.cur)
   for t in range(ctx):
    lg=self._advance(list(range(bs)),x[:,t].tolist(),True);m=lg.max(axis=1,keepdims=True);lz=np.log(np.exp(lg-m).sum(axis=1))+m[:,0];tot+=float((lz-lg[np.arange(bs),y[:,t]]).sum());n+=bs
  return tot/n/math.log(2)
 def _load_words(self):
  if self.words or not os.path.exists(self.words_path):return
  for w in open(self.words_path,"rb").read().splitlines():self.words.add(w);[self.pref.add(w[:k]) for k in range(1,len(w)+1)]
 def word_ok(self,seq:list,c:int)->bool:
  if not self.words:return True
  j=len(seq)
  while j>0 and (65<=seq[j-1]<=90 or 97<=seq[j-1]<=122 or seq[j-1]==39):j-=1
  cur=bytes(seq[j:]);isl=65<=c<=90 or 97<=c<=122 or c==39
  return ((cur+bytes([c])) in self.pref or len(cur)>=24) if isl else (len(cur)<=1 or cur in self.words or cur.lower() in self.words)
 def stream(self,prompt:str,max_bytes:int=400,temp:float=0.7,topk:int=12,stop:tuple=("\n\n",),veto:bool=True,seed:int=None):
  with _lock:
   self.load();veto and self._load_words();g=np.random.default_rng(seed);seq=list(prompt.encode());lg=self.prime(bytes(seq))[0];dec=codecs.getincrementaldecoder("utf-8")("replace");out=b""
   try:
    for _ in range(max_bytes):
     z=np.where(self.ok,lg/temp,-1e9);z[seq[-1]]-=6.0 if len(seq)>=4 and len(set(seq[-4:]))==1 else 0.0;order=np.argsort(z)[::-1];keep=[int(c) for c in order[:topk*6] if not veto or self.word_ok(seq,int(c))][:topk] or [int(c) for c in order[:topk]];zz=z[keep]-z[keep].max();pr=np.exp(zz);pr/=pr.sum();c=keep[int(g.choice(len(keep),p=pr))]
     seq.append(c);out+=bytes([c]);s=dec.decode(bytes([c]))
     if any(out.endswith(t.encode()) for t in stop):break
     s and (yield s);lg=self._advance([0],[c],True)[0]
   except GeneratorExit:return
 def beam(self,prompt:str,width:int=8,expand:int=4,maxlen:int=300,rep:int=16,veto:bool=True)->str:
  with _lock:
   self.load();veto and self._load_words();p0=list(prompt.encode());lg0=self.prime(bytes(p0));lp=np.log(np.clip(self.ok.astype(np.float64),1e-300,1.0));beams=[(0.0,p0,0)];lgs=lg0;norm=lambda t:t[0]/(len(t[1])-len(p0)+1)**0.7
   for _ in range(maxlen):
    cand=[]
    for b,(sc,seq,_) in enumerate(beams):
     z=lgs[b]+lp;z[seq[-1]]-=6.0 if len(seq)>=4 and len(set(seq[-4:]))==1 else 0.0;z-=z.max();pr=np.exp(z);pr/=pr.sum();order=np.argsort(pr)[::-1];top=[int(c) for c in order[:expand*6] if not veto or self.word_ok(seq,int(c))][:expand] or [int(c) for c in order[:expand]]
     for c in top:nseq=seq+[c];pen=0.6 if rep>0 and len(nseq)>2*rep and bytes(nseq[-rep:]) in bytes(nseq[:-rep]) else 0.0;cand.append((sc+float(np.log(pr[c]+1e-12))-pen,nseq,b))
    cand.sort(key=norm,reverse=True);beams=cand[:width];lgs=self._advance([b for _,_,b in beams],[s[-1] for _,s,_ in beams],True)
    if all(s[-1]==10 for _,s,_ in beams):break
   return bytes(max(beams,key=norm)[1][len(p0):]).decode("utf-8","replace")
def engine()->RayField:
 global _eng
 _eng=_eng or RayField();return _eng
