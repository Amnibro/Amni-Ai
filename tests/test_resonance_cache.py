import sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from amni.inference.resonance_cache import ResonanceCache
def test_build_and_lookup_reconstructs():
    rng=np.random.RandomState(0);centers=rng.randn(8,32)
    Xtr=(np.repeat(centers,200,0)+rng.randn(1600,32)*0.05).astype(np.float32);W=rng.randn(48,32).astype(np.float32)
    c=ResonanceCache.build(Xtr,W,pcs=4,resid_pcs=8,levels=17,min_support=4)
    Xte=(np.repeat(centers,20,0)+rng.randn(160,32)*0.05).astype(np.float32);true=Xte@W.T
    y,hit=c.lookup(Xte)
    cos=[float(np.dot(y[i],true[i])/(np.linalg.norm(y[i])*np.linalg.norm(true[i])+1e-9)) for i in range(len(Xte)) if hit[i]]
    assert hit.mean()>0.5,hit.mean();assert np.mean(cos)>0.95,np.mean(cos)
def test_ood_falls_back():
    rng=np.random.RandomState(1);centers=rng.randn(8,32)
    Xtr=(np.repeat(centers,200,0)+rng.randn(1600,32)*0.05).astype(np.float32);W=rng.randn(48,32).astype(np.float32)
    c=ResonanceCache.build(Xtr,W,pcs=4,resid_pcs=8,levels=17,min_support=4)
    Xood=(rng.randn(200,32)*5).astype(np.float32);_,hit=c.lookup(Xood)
    assert hit.mean()<0.6,hit.mean()
def test_empty_cache_signals_fallback():
    rng=np.random.RandomState(2);X=(rng.randn(50,16)).astype(np.float32);W=rng.randn(20,16).astype(np.float32)
    c=ResonanceCache.build(X,W,pcs=4,resid_pcs=4,levels=17,min_support=999);y,hit=c.lookup(X)
    assert hit.sum()==0 and y.shape==(50,20)
if __name__=='__main__':
    fns=[v for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v)];ok=0
    for fn in fns:
        try:fn();print(f'PASS {fn.__name__}');ok+=1
        except Exception:print(f'FAIL {fn.__name__}');traceback.print_exc()
    print(f'{ok}/{len(fns)} PASS');sys.exit(0 if ok==len(fns) else 1)
