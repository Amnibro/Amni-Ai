import os,sys,importlib.util,numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
[os.environ.setdefault(k,v) for k,v in {"LEVELS":"21","HOPS":"4","LOG2_ENTRIES":"20","QLEV":"5","TERN":"1","TERN_SCOPE":"cell","BS":"1","CORPUS":"big2","STEPS":"0","CTX":"256"}.items()]
spec=importlib.util.spec_from_file_location("v16",os.path.join(_H0,"scripts",os.environ.get("REF_SCRIPT","ray_hash_field_ti_v16.py")));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
P=os.environ.get("REF_PAGES",os.path.join(_H0,"exports","gf17_continuum","rayhash_v16_big2_L21_e20_hop4_q5_s150k_dq_q5_i8"));m.init_page(0.0);m.DP.from_numpy(np.load(P+"_dp.npy").astype(np.float32));m.SID.from_numpy(np.load(P+"_sid.npy").astype(np.int32));m.SW.from_numpy(np.load(P+"_sw.npy").astype(np.float32));m.bias.from_numpy(np.load(P+"_bias.npy").astype(np.float32));m.SALT[None]=0
out={}
for i,t in enumerate(open(sys.argv[1],"rb").read().split(b"\x00")):
 x=np.zeros((m.BS,m.Tn),np.int32);L=len(t);x[0,:L]=list(t);m.X.from_numpy(x);m.drivers(1,L,m.TOPIC_A);m.forward(L);[(m.hop(L,h),m.forward_lvl(L,m.BASE+h)) for h in range(m.HOPS)];out[f"t{i}"]=m.LOG.to_numpy()[:L].copy()
np.savez(sys.argv[2],**out)
