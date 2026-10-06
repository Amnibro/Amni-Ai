import numpy as np,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/"data"/"wikifact"/"v3"/"store";OUT=ROOT/"data"/"store1"/"results"/"reverse_estimate.json"
L=lambda n:np.asarray(np.load(D/f"{n}.npy",mmap_mode="r"))
fs,fp,fk,fv=L("f_s"),L("f_p"),L("f_k"),L("f_v");m=fk==1;o=fv[m].astype(np.int64);p=fp[m].astype(np.int64);s=fs[m].astype(np.int64);del fs,fp,fk,fv
k=np.lexsort((s,p,o));o,p,s=o[k],p[k],s[k];g=np.r_[True,(o[1:]!=o[:-1])|(p[1:]!=p[:-1])];d=np.where(g,s,np.r_[0,np.diff(s)])
def evd(v):v=v.astype(np.float64);return np.where(v<16,1,2+np.ceil(np.log(np.maximum(v-15,1))/np.log(17)+1e-12))
dig=evd(d).sum()+2*g.sum()+evd(np.diff(np.r_[np.flatnonzero(g),len(g)])).sum()
r={"entity_facts":int(len(o)),"object_property_groups":int(g.sum()),"est_digits":int(dig),"est_bytes":int(dig*17/32),"note":"lower-bound style estimate of a reverse index (object -> property -> sorted subjects) coded as base-17 EV deltas plus a 2-digit property and a count per group; excludes the object -> group directory; v3 r_v/r_f use 344 MB for the same mapping"}
OUT.write_text(json.dumps(r,indent=1));print(r)
