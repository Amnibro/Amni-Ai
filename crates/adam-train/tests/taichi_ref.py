import os,sys,numpy as np
H=os.environ.get("AMNI_HOME","/home/anmire/ai/Amni-Ai");sys.path.insert(0,os.path.join(H,"scripts"))
R=__import__(os.environ.get("REF_SCRIPT","ray_hash_field_ti_v18"))
out=sys.argv[1];os.makedirs(out,exist_ok=True)
corp=R.load_multi() if R.MULTI else [R.load_corpus()]
def batch(d,g):
 ix=g.integers(0,len(d)-R.Tn-1,R.BS);return np.stack([np.asarray(d[i:i+R.Tn]) for i in ix]).astype(np.int64),np.stack([np.asarray(d[i+1:i+1+R.Tn]) for i in ix]).astype(np.int64)
def evalv(nb=20,ci=0):
 ge=np.random.default_rng(1234);return float(np.mean([R.run_batch(*batch(corp[ci][1],ge),False,0,ci) for _ in range(nb)]))
R.init_page(0.0);P=os.environ.get("PAGES","")
if P:R.DP.from_numpy(np.load(P.replace("_sid.npy","_dp.npy")).astype(np.float32));R.SID.from_numpy(np.load(P).astype(np.int32));R.SW.from_numpy(np.load(P.replace("_sid.npy","_sw.npy")).astype(np.float32));R.bias.from_numpy(np.load(P.replace("_sid.npy","_bias.npy")).astype(np.float32))
g=np.random.default_rng(R.SEED);f=open(os.path.join(out,"steps.txt"),"w")
for it in range(1,R.STEPS+1):
 ci=it%len(corp);x,y=batch(corp[ci][0],g);l=R.run_batch(x,y,True,it,ci);f.write(f"{it} {l!r}\n")
 if it%200==0 or it==1 or it==R.STEPS:f.write(f"eval {it} {evalv()!r}\n");f.flush()
f.close()
if R.TERN:R.refresh_levels(R.TERN_TH)
dp=lambda a:np.concatenate([a[0,:256],a[1]]).astype(np.float32)
for k,v in {"dp":dp(R.DP.to_numpy()),"dm":dp(R.DM.to_numpy()),"dv":dp(R.DV.to_numpy()),"sid":R.SID.to_numpy().astype(np.float32),"sw":R.SW.to_numpy(),"sm":R.SM.to_numpy(),"sv":R.SV.to_numpy(),"bias":R.bias.to_numpy(),"lq":np.concatenate([R.LTH.to_numpy(),R.LMAG.to_numpy()])}.items():np.save(os.path.join(out,k+".npy"),v.astype(np.float32))
print("reference written",out,flush=True)
