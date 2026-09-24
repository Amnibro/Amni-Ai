import os,sys,json,numpy as np
W=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))));sys.path[:0]=[W,os.path.join(W,"crates","adam-ray-py","tests")]
import adam_ray
from amni.compute.ray_field import RayField
from ray_ref import TEXTS,HOME
B=os.path.join(HOME,"data","big2");S,WD,WC=os.path.join(B,"synsets.txt"),os.path.join(B,"words.txt"),os.path.join(B,"wordcounts.txt")
for pack in sys.argv[1:]:
 ref=RayField(path=pack,syn=S,words=WD,arch="cpu");ref.wcnt_path=WC;ref.load();lr=[ref.logits_for(t) for t in TEXTS]
 for arch in os.environ.get("ARCHS","cpu,vulkan").split(","):
  e=adam_ray.RayField(pack,S,WD,WC,arch,HOME).load();la=[e.logits_for(t) for t in TEXTS]
  d=max(float(np.abs(a-b).max()) for a,b in zip(la,lr));ex=sum(int((a==b).all(1).sum()) for a,b in zip(la,lr));ag=sum(int((a.argmax(1)==b.argmax(1)).sum()) for a,b in zip(la,lr));n=sum(len(t) for t in TEXTS)
  print(json.dumps({"pack":os.path.basename(pack),"levels":ref.L,"entries":ref.T,"qt":bool(ref.qt),"rust":e.backend,"positions":n,"bit_identical_rows":f"{ex}/{n}","top1_agree":f"{ag}/{n}","max_abs_logit_diff":d}),flush=True)
