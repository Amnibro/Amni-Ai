import sys,os,numpy as np
a,b=sys.argv[1],sys.argv[2]
def steps(d):
 tr={};ev={}
 for l in open(os.path.join(d,"steps.txt")):
  p=l.split()
  if p[0]=="eval":ev[int(p[1])]=float(p[2])
  else:tr[int(p[0])]=float(p[1])
 return tr,ev
(ta,ea),(tb,eb)=steps(a),steps(b);ks=sorted(set(ta)&set(tb))
d=np.array([abs(ta[k]-tb[k]) for k in ks]);r=d/np.array([abs(tb[k]) for k in ks])
print(f"train loss: {len(ks)} steps | exact {int((d==0).sum())} | max|d| {d.max():.3e} | max rel {r.max():.3e} | first {ta[ks[0]]:.8f} vs {tb[ks[0]]:.8f} | last {ta[ks[-1]]:.8f} vs {tb[ks[-1]]:.8f}")
for k in sorted(set(ea)&set(eb)):print(f"  eval@{k}: {ea[k]:.10f} vs {eb[k]:.10f} d={abs(ea[k]-eb[k]):.3e}")
ok=True
for k in ["dp","dm","dv","sid","sw","sm","sv","bias","lq"]:
 x,y=np.load(os.path.join(a,k+".npy")),np.load(os.path.join(b,k+".npy"))
 if x.shape!=y.shape:print(f"{k}: shape {x.shape} vs {y.shape}");ok=False;continue
 dd=np.abs(x.astype(np.float64)-y);ne=int((x!=y).sum());sc=max(float(np.abs(y).max()),1e-30)
 print(f"{k:>4}: {x.size:>10} values | differ {ne:>8} | max|d| {dd.max():.3e} | max|d|/max|ref| {dd.max()/sc:.3e}")
 ok&=ne==0
si=np.load(os.path.join(b,"sid.npy"));ini=np.array([32,101,116],np.float32)
print(f"slot ids replaced vs init (reference): {int((si!=ini).sum())} slots in {int((si!=ini).any(axis=2).sum())} cells; live |sw|>=0.1: {int((np.abs(np.load(os.path.join(b,'sw.npy')))>=0.1).sum())}")
print("EXACT" if ok and (d==0).all() else "NOT EXACT")
