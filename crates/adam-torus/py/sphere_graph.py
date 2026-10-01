import os,sys,importlib.util,numpy as np
ROOT=os.environ.get("AMNI_ROOT","/home/anmire/ai/Amni-Ai");os.environ["AMNI_ROOT"]=ROOT;os.environ["SPHERE_OUT"]=os.path.join(ROOT,"data","torus","_never_written.npz")
def grab(path,out):
 spec=importlib.util.spec_from_file_location("cs_"+os.path.basename(path).replace(".",""),path);cs=importlib.util.module_from_spec(spec);spec.loader.exec_module(cs)
 def stop(*a,**k):
  f=sys._getframe(1).f_locals;np.savez(out,edge_src=f["edge_src"],edge_dst=f["edge_dst"],wn_src=f["es"],wn_dst=f["ed"],hyp_src=f["hyp_src"],hyp_dst=f["hyp_dst"],N=np.int64(f["N"]),Ns=np.int64(f["Ns"]));raise SystemExit(0)
 cs.ti.init=stop
 try:cs.main()
 except SystemExit:pass
 return cs
if __name__=="__main__":
 os.makedirs(os.path.join(ROOT,"data","torus"),exist_ok=True)
 grab(os.path.join(ROOT,"scripts","concept_sphere.py"),os.path.join(ROOT,"data","torus","sphere_graph_cur.npz"))
 grab(os.path.join(ROOT,"backups","concept_sphere_2026-09-30","concept_sphere_C_bak.py"),os.path.join(ROOT,"data","torus","sphere_graph_C.npz"))
 for n in ("cur","C"):
  g=np.load(os.path.join(ROOT,"data","torus",f"sphere_graph_{n}.npz"));print(n,{k:(g[k].shape if g[k].ndim else int(g[k])) for k in g.files},flush=True)
