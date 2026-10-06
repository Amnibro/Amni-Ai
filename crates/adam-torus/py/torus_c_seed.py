import os,json,argparse,numpy as np
ROOT=os.environ.get("AMNI_ROOT","/home/anmire/ai/Amni-Ai");TC=os.path.join(ROOT,"data","torus","c");NRT=24
def build(H,T,src,SPL,K,coarse=None,NRT=NRT):
 N=len(src);tr=SPL==0;eh,et=H[tr],T[tr];din=np.bincount(et,minlength=N);dout=np.bincount(eh,minlength=N);dd=np.full(N,3,np.int64);dd[(dout==0)&(din>0)]=0
 for r in (1,2):np.minimum.at(dd,eh[dd[et]==r-1],r)
 f=np.stack([src.astype(np.int64),np.digitize(din,[1,4,64]),np.digitize(dout,[1,2,5]),dd]+([coarse] if coarse is not None else []),1);nf=f.shape[1];rg=np.array([3,4,4,4]+([int(coarse.max())+1] if coarse is not None else []));mul=np.concatenate([np.cumprod(rg[::-1])[::-1][1:],[1]]);code=(f*mul[None,:]).sum(1);nc=int(np.prod(rg));w=np.bincount(code,weights=(din+dout).astype(np.float64),minlength=nc);cf=np.stack([(np.arange(nc)//m)%g for m,g in zip(mul,rg)],1)
 kept=np.argsort(-w,kind="stable")[:NRT];ham=(cf[:,None,:]!=cf[kept][None,:,:]).sum(2);rmap=np.argmin(ham*1e9-w[kept][None,:],axis=1);rmap[kept]=np.arange(NRT);role=np.where(din+dout>0,rmap[code],-1).astype(np.int8);rf=cf[kept]
 pc=np.bincount(role[eh].astype(np.int64)*NRT+role[et],minlength=NRT*NRT);top=np.argsort(-pc,kind="stable")[:K];pf=np.concatenate([np.repeat(rf,NRT,0),np.tile(rf,(NRT,1))],1);ph=(pf[:,None,:]!=pf[top][None,:,:]).sum(2);tab=np.argmin(ph*1e9-pc[top][None,:],axis=1);tab[top]=np.arange(K)
 return dict(role=role,table=tab.astype(np.int16),rolef=rf,pair_count=pc,seed=tab[role[eh].astype(np.int64)*NRT+role[et]].astype(np.int16))
def apply(S,h,t):return S["table"][S["role"][h].astype(np.int64)*int(round(len(S["table"])**0.5))+S["role"][t]].astype(np.int64)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--K",type=int,default=32);ap.add_argument("--coarse",type=int,default=0);ap.add_argument("--roles",type=int,default=NRT);ap.add_argument("--data",default="c_data.npz");A=ap.parse_args();d=np.load(os.path.join(TC,A.data),allow_pickle=True);H,T,src,SPL=d["h"],d["t"],d["src"].astype(np.int64),d["split"];co=None
 if A.coarse:
  sp=np.load(os.path.join(TC,"sphere_noleak.npz"));li=d["lift"];c64=sp["cell_id_64"].astype(np.int64);co=np.where(li>=0,c64[np.maximum(li,0)]//4096+1,0)
 S=build(H,T,src,SPL,A.K,co,A.roles);np.savez(os.path.join(TC,f"seed_K{A.K}{'c' if A.coarse else ''}{'r'+str(A.roles) if A.roles!=NRT else ''}.npz"),**S)
 print(json.dumps({"K":A.K,"roles":int(NRT),"pairs_seen":int((S["pair_count"]>0).sum()),"top_pair_share":round(float(S["pair_count"][np.argsort(-S["pair_count"])[:A.K]].sum()/S["pair_count"].sum()),4),"clusters_used":int(len(np.unique(S["seed"]))),"largest_share":round(float(np.bincount(S["seed"]).max()/len(S["seed"])),4)}),flush=True)
if __name__=="__main__":main()
