import os,sys,importlib.util,numpy as np
ROOT=os.environ.get("AMNI_ROOT","/home/anmire/ai/Amni-Ai");A2B="--a2b" in sys.argv;TD=os.path.join(ROOT,"data","torus","a2b") if A2B else os.path.join(ROOT,"data","torus");os.environ["AMNI_ROOT"]=ROOT;os.environ["SPHERE_OUT"]=os.path.join(TD,"sphere_noleak.npz")
d=np.load(os.path.join(TD,"a2b_data.npz" if A2B else "a2_data.npz"),allow_pickle=True);NS=int(d["n_sphere"]);m=(d["split"]>0)&((d["h"]<NS)&(d["t"]<NS) if A2B else (d["r"]<3));N=len(d["names"]);ho=np.sort(np.stack([d["h"][m],d["t"][m]],1),1);hk=np.unique(ho[:,0]*N+ho[:,1])
w=np.load(os.path.join(ROOT,"data","concepts","wordnet_graph.npz"),allow_pickle=True);es,ed=w["edge_src"].astype(np.int64),w["edge_dst"].astype(np.int64);pk=np.minimum(es,ed)*N+np.maximum(es,ed);keep=~np.isin(pk,hk)
out=os.path.join(TD,"wordnet_graph_noleak.npz");np.savez(out,**{k:(w[k][keep] if k.startswith("edge_") and k!="edge_names" else w[k]) for k in w.files})
print(f"held_out_pairs={len(hk)} wn_edges={len(es)} removed={int((~keep).sum())}",flush=True)
spec=importlib.util.spec_from_file_location("cs_noleak",os.path.join(ROOT,"scripts","concept_sphere.py"));cs=importlib.util.module_from_spec(spec);spec.loader.exec_module(cs)
cs.WN=out;cs.OUT=os.environ["SPHERE_OUT"];cs.TMP=os.path.join(TD,"sphere_tmp");cs.main()
