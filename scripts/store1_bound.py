import numpy as np,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/"data"/"wikifact"/"v3"/"store";OUT=ROOT/"data"/"store1"/"results"/"bounds.json"
L=lambda n:np.asarray(np.load(D/f"{n}.npy",mmap_mode="r"))
fp,fk,fr,fv,fx,fqo=L("f_p"),L("f_k"),L("f_r"),L("f_v"),L("f_x"),L("f_qo")
qc=np.diff(fqo.astype(np.int64));aux=fr.astype(np.uint64)|(qc.astype(np.uint64)<<np.uint64(1));N=len(fp);H=0.0;Hv=0.0;CB=0.0
for k in (1,2,3,4,5):
 m=fk==k;p=fp[m].astype(np.uint64);v=fv[m] if k<4 else np.arange(int(m.sum()),dtype=np.uint64);x=fx[m].astype(np.uint64);a=aux[m]
 for keyset,acc in (((p,v,x,a),"H"),((p,v,x),"Hv")):
  K=np.stack(keyset,1);o=np.lexsort(K.T[::-1]);K=K[o];b=np.r_[True,(K[1:]!=K[:-1]).any(1)];cnt=np.diff(np.r_[np.flatnonzero(b),len(K)]).astype(np.float64);pu=K[b,0]
  t=0.0;cb=0.0
  for q in np.unique(pu):c=cnt[pu==q];n=c.sum();t+=-(c*np.log2(c/n)).sum();cb+=n*math.log2(max(len(c),1))
  if acc=="H":H+=t;CB+=cb
  else:Hv+=t
r={"facts":int(N),"H0_fact_symbol_bits_per_fact":round(H/N,4),"H0_value_only_bits_per_fact":round(Hv/N,4),"codebook_bound_bits_per_fact":round(CB/N,4),"note":"0-order entropy of the stored fact symbol (value, unit, preferred flag, qualifier count) given the property; strings counted as unique (their text lives in the string codec). codebook bound = log2 of the per-property distinct-symbol count, the A1-style bound for fixed-width retrieval. Codebook storage itself is not in these bounds."}
OUT.write_text(json.dumps(r,indent=1));print(r)
