import re,sys,gzip,json,time,hashlib,argparse,urllib.request
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)]
from amni.compute.wikifact import h64
URL="https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
RELS=("IsA","UsedFor","CapableOf","PartOf","HasA","MadeOf","AtLocation","Causes","Desires","ReceivesAction","HasProperty","Antonym","CreatedBy","HasPrerequisite","CausesDesire","MotivatedByGoal","DefinedAs","LocatedNear")
DS=("conceptnet","wordnet","wiktionary","opencyc","dbpedia","verbosity","other")
LIC="CC BY-SA 4.0"
ATTR="This work includes data from ConceptNet 5, which was compiled by the Commonsense Computing Initiative. ConceptNet 5 is freely available under the Creative Commons Attribution-ShareAlike license (CC BY SA 4.0) from https://conceptnet.io. The included data was created by contributors to Commonsense Computing projects, contributors to Wikimedia projects, Games with a Purpose, Princeton University's WordNet, DBPedia, OpenCyc, and Umbra Research."
key=lambda t:' '.join(re.sub(r"[^a-z0-9' ]+",' ',t.lower().replace('_',' ')).split())
def g17(v:np.ndarray,n:int)->np.ndarray:return np.stack([(v.astype(np.int64)//17**k)%17 for k in range(n)],1).astype(np.uint8)
def u17(d:np.ndarray)->np.ndarray:return (d.astype(np.int64)*(17**np.arange(d.shape[1],dtype=np.int64))).sum(1)
def build(src:str,out:Path,minw:float)->dict:
    t0=time.time();out.mkdir(parents=True,exist_ok=True);fh=urllib.request.urlopen(src) if src.startswith('http') else open(src,'rb');hs=hashlib.sha256();nb=[0]
    class R:
        def read(s,n=-1):b=fh.read(n);hs.update(b);nb[0]+=len(b);return b
    RI={r:i for i,r in enumerate(RELS)};N={};E={};seen=0;kept=0
    nid=lambda t:N.setdefault(t,len(N))
    for l in gzip.open(R(),'rt',encoding='utf-8'):
        p=l.split('\t',4)
        if len(p)<5 or not (p[2].startswith('/c/en/') and p[3].startswith('/c/en/')):continue
        seen+=1;r=p[1][3:]
        if r not in RI:continue
        j=json.loads(p[4]);w=float(j.get('weight',1.0));a,b=key(p[2].split('/')[3]),key(p[3].split('/')[3])
        if w<minw or not a or not b or a==b or len(a)>60 or len(b)>60 or len(a.split())>6 or len(b.split())>6:continue
        ds=(j.get('dataset') or '').split('/')[2:3];f=(1<<DS.index(ds[0]) if ds and ds[0] in DS else 1<<6)|(128 if 'by-sa' in (j.get('license') or '') else 0);k=(nid(a),RI[r],nid(b));x=E.get(k);E[k]=(max(w,x[0]),x[1]|f) if x else (w,f);kept+=1
    fh.close();T=list(N);H=np.array([h64(t) for t in T],np.uint64);o=np.argsort(H,kind='stable');rk=np.empty(len(o),np.int64);rk[o]=np.arange(len(o));H=H[o];assert len(np.unique(H))==len(H)
    K=np.array(list(E.keys()),np.int64).reshape(-1,3);V=np.array(list(E.values()),np.float64).reshape(-1,2);s,r,d=rk[K[:,0]],K[:,1],rk[K[:,2]];w=np.minimum(np.round(V[:,0]*100),65535).astype(np.uint16);fl=V[:,1].astype(np.uint8)
    q=np.lexsort((-w.astype(np.int64),r,s));s,r,d,w,fl=s[q],r[q],d[q],w[q],fl[q];nn=len(T);ptr=np.searchsorted(s,np.arange(nn+1)).astype(np.uint32);ro=np.lexsort((-w.astype(np.int64),d));rp=np.searchsorted(d[ro],np.arange(nn+1)).astype(np.uint32)
    tb=[T[i].encode() for i in o.tolist()];off=np.zeros(nn+1,np.uint64);off[1:]=np.cumsum([len(x) for x in tb]);(out/'n_txt.bin').write_bytes(b''.join(tb))
    A={'n_h':H,'n_o':off.astype(np.uint32),'e_p':ptr,'e_r':r.astype(np.uint8),'e_d':d.astype(np.uint32),'e_w':w,'e_f':fl,'r_p':rp,'r_e':ro.astype(np.uint32)}
    [np.save(out/f"{k}.npy",v) for k,v in A.items()]
    G=np.concatenate([g17(s,8),g17(r,2),g17(d,8),g17(w,4)],1);G.tofile(out/'edges.gf17');Z=np.fromfile(out/'edges.gf17',np.uint8).reshape(-1,22);ex=bool((u17(Z[:,:8])==s).all() and (u17(Z[:,8:10])==r).all() and (u17(Z[:,10:18])==d).all() and (u17(Z[:,18:])==w).all())
    rc={x:int((r==i).sum()) for i,x in enumerate(RELS)};sz={p.name:p.stat().st_size for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}
    man={'name':'ConceptNet 5.7 English commonsense edges','license':LIC,'attribution':ATTR,'source':URL if not src.startswith('http') else src,'source_sha256':hs.hexdigest(),'source_bytes':nb[0],'filter':{'languages':'both ends /c/en/','relations':list(RELS),'min_weight':minw,'max_words':6,'max_chars':60,'dedupe':'(src,rel,dst) max weight, dataset flags OR-ed'},'en_edges_seen':seen,'edges_matched':kept,'edges':int(len(s)),'nodes':nn,'by_relation':rc,'flags':{'bits':{d_:1<<i for i,d_ in enumerate(DS)},'by_sa':128},'gf17':{'file':'edges.gf17','digits':'s:8 r:2 d:8 w:4 (base 17, little-endian)','exact':ex},'bytes':sz,'total_bytes':int(sum(sz.values())),'built':time.strftime('%Y-%m-%d %H:%M:%S'),'sec':round(time.time()-t0,1)}
    (out/'manifest.json').write_text(json.dumps(man,indent=1));return man
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--src',default=URL);ap.add_argument('--out',default=str(ROOT/'data'/'concepts'/'conceptnet'));ap.add_argument('--minw',type=float,default=1.0);a=ap.parse_args();m=build(a.src,Path(a.out),a.minw);print(json.dumps({k:v for k,v in m.items() if k!='attribution'},indent=1))
