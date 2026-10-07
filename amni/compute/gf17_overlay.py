import struct,hashlib,numpy as np
VD=6;TOMB=17**VD-1;MAG=b'GF17OV01';HD=struct.Struct('<8sIQIIIIII');INV=np.array([0,1,9,6,13,7,3,5,15,2,12,14,10,4,11,8,16],np.uint8);MUL=(np.arange(17)[:,None]*np.arange(17)[None,:]%17).astype(np.uint8);M17=(np.arange(17*64)%17).astype(np.uint8);P17=17**np.arange(VD,dtype=np.int64)
def hk(key:int,seed:int)->tuple:return struct.unpack('<4Q',hashlib.blake2b(struct.pack('<Q',key&(1<<64)-1),digest_size=32,key=struct.pack('<Q',seed)).digest())
def nonce(x:str)->int:return struct.unpack('<Q',hashlib.blake2b(x.encode('utf-8'),digest_size=8,person=b'amni-pipe1-key').digest())[0]
def px(d:np.ndarray)->np.ndarray:return np.concatenate([d.reshape(-1),np.zeros(-d.size%4,np.uint8)]).reshape(-1,4)
def b17(b:bytes)->np.ndarray:a=np.frombuffer(b,np.uint8);return np.stack([a%17,a//17],1).reshape(-1).astype(np.uint8)
def u17(d:np.ndarray)->bytes:d=d.reshape(-1,2).astype(np.uint16);return (d[:,0]+17*d[:,1]).astype(np.uint8).tobytes()
class Ov:
    def __init__(s,nsh:int=1,seed:int=17,w:int=24,fd:int=6,eps:float=0.15):s.nsh,s.seed,s.w,s.fd,s.eps,s.R=nsh,seed,w,fd,eps,VD+fd;s.log={};s.pend={};s.heap=[];s.cells=[np.zeros((0,s.R),np.uint8)]*nsh;s.pc=np.zeros((0,s.R),np.uint8)
    def _h(s,k:int)->tuple:h=hk(k,s.seed);return h,((h[0]>>32)*s.nsh)>>32
    def _fp(s,h)->np.ndarray:return np.array([((h[2 if j<8 else 3]>>(8*(j%8)))&255)*17>>8 for j in range(s.fd)],np.uint8)
    def _row(s,h,vid:int)->np.ndarray:return np.concatenate([(vid//P17%17).astype(np.uint8),s._fp(h)])
    def _solve(s,items:list,tag:int)->np.ndarray:
        n=len(items);m=int(np.ceil(n*(1+s.eps)))+s.w if n else 0;H=[s._h(k)[0] for k,_ in items]
        while n:
            pc=np.zeros((m,s.w),np.uint8);pv=np.zeros((m,s.R),np.uint8);oc=np.zeros(m,bool);ok=True
            for i,c,v in sorted(((int((h[0]&0xFFFFFFFF)*(m-s.w+1)>>32),np.array([(((h[1]&((1<<s.w)-1))|1)>>j)&1 for j in range(s.w)],np.uint8),s._row(h,vid)) for h,(_,vid) in zip(H,items)),key=lambda r:r[0]):
                while ok:
                    if not oc[i]:iv=INV[c[0]];pc[i]=MUL[iv,c];pv[i]=MUL[iv,v];oc[i]=True;break
                    f=17-int(c[0]);c=M17[c+MUL[f,pc[i]]];v=M17[v+MUL[f,pv[i]]];z=np.flatnonzero(c)
                    if not len(z):ok=not v.any();break
                    c=np.concatenate([c[z[0]:],np.zeros(z[0],np.uint8)]);i+=int(z[0])
                if not ok:break
            if ok:break
            m+=max(1,n//200)
        if not n:return np.zeros((0,s.R),np.uint8)
        g=np.random.default_rng([s.seed,tag+1,m]);Z=np.zeros((m,s.R),np.uint8)
        for i in range(m-1,-1,-1):j=min(s.w,m-i);Z[i]=((pv[i].astype(np.int64)-MUL[pc[i,1:j][:,None],Z[i+1:i+j]].sum(0,dtype=np.int64))%17).astype(np.uint8) if oc[i] else g.integers(0,17,s.R,dtype=np.uint8)
        return Z
    def _look(s,Z:np.ndarray,h)->int:
        m=len(Z)
        if not m:return -1
        i=int((h[0]&0xFFFFFFFF)*(m-s.w+1)>>32);b=np.flatnonzero([(((h[1]&((1<<s.w)-1))|1)>>j)&1 for j in range(s.w)]);d=M17[Z[i+b].sum(0,dtype=np.int64)]
        return int((d[:VD].astype(np.int64)*P17).sum()) if (d[VD:]==s._fp(h)).all() else -1
    def get(s,key:int):
        h,sh=s._h(key);v=s._look(s.pc,h);v=s._look(s.cells[sh],h) if v<0 else v
        return None if v<0 or v==TOMB or v>=len(s.heap) else s.heap[v]
    def items(s)->dict:return {k:s.heap[v] for k,v in {**s.log,**s.pend}.items() if v!=TOMB}
    def learn(s,key:int,payload:bytes,compact:bool=False)->int:
        v=s.pend.get(key,s.log.get(key,TOMB));v!=TOMB and v<len(s.heap) and s.heap.__setitem__(v,b'');s.heap.append(bytes(payload));s.pend[key]=len(s.heap)-1;s._sp();compact and s.compact();return len(s.heap)-1
    def forget(s,key:int,compact:bool=False)->bool:
        v=s.pend.get(key,s.log.get(key,TOMB))
        if v==TOMB:return False
        s.heap[v]=b'';s.pend[key]=TOMB;s._sp();compact and s.compact();return True
    def _trim(s):
        u=set(s.log.values())|set(s.pend.values())
        while s.heap and s.heap[-1]==b'' and len(s.heap)-1 not in u:s.heap.pop()
    def _sp(s):s.pc=s._solve(sorted(s.pend.items()),s.nsh)
    def compact(s)->list:
        T=sorted({s._h(k)[1] for k in s.pend});[s.log.pop(k,None) if v==TOMB else s.log.__setitem__(k,v) for k,v in s.pend.items()];s.pend={};s.pc=np.zeros((0,s.R),np.uint8)
        for t in T:s.cells[t]=s._solve(sorted((k,v) for k,v in s.log.items() if s._h(k)[1]==t),t)
        s._trim();return T
    def rebuild(s)->'Ov':o=Ov(s.nsh,s.seed,s.w,s.fd,s.eps);o.log=dict(s.log);o.heap=list(s.heap);o.cells=[o._solve(sorted((k,v) for k,v in o.log.items() if o._h(k)[1]==t),t) for t in range(o.nsh)];return o
    def dumps(s)->bytes:
        L=sorted(s.log.items());P=sorted(s.pend.items());cel=[px(c) for c in s.cells]+[px(s.pc)];hp=px(b17(b''.join(struct.pack('<I',len(x))+x for x in s.heap)))
        h=HD.pack(MAG,1,s.seed,s.nsh,s.w,s.fd,len(L),len(P),len(s.heap));fl=struct.pack(f'<{s.nsh+1}I{s.nsh+1}I',*[len(c) for c in s.cells+[s.pc]],*[len(c) for c in cel])
        return h+fl+struct.pack(f'<{2*len(L)+2*len(P)}Q',*[x for kv in L+P for x in kv])+b''.join(c.tobytes() for c in cel)+struct.pack('<Q',len(hp))+hp.tobytes()
    @staticmethod
    def loads(b:bytes)->'Ov':
        mg,ver,seed,nsh,w,fd,nl,npn,nh=HD.unpack_from(b);(mg!=MAG or ver!=1) and (_ for _ in ()).throw(ValueError('overlay corrupt'));o=Ov(nsh,seed,w,fd);q=HD.size;F=struct.unpack_from(f'<{2*nsh+2}I',b,q);q+=8*nsh+8
        K=struct.unpack_from(f'<{2*nl+2*npn}Q',b,q);q+=16*(nl+npn);o.log={K[2*i]:K[2*i+1] for i in range(nl)};o.pend={K[2*i]:K[2*i+1] for i in range(nl,nl+npn)};C=[]
        for j in range(nsh+1):a=np.frombuffer(b,np.uint8,4*F[nsh+1+j],q).reshape(-1);q+=4*F[nsh+1+j];C.append(a[:F[j]*o.R].reshape(-1,o.R).copy())
        o.cells,o.pc=C[:nsh],C[nsh];n=struct.unpack_from('<Q',b,q)[0];q+=8;r=u17(np.frombuffer(b,np.uint8,4*n,q));H=[];i=0
        while len(H)<nh:z=struct.unpack_from('<I',r,i)[0];H.append(r[i+4:i+4+z]);i+=4+z
        o.heap=H;return o
