import os,sys,subprocess,tempfile,numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,_H0)
from amni.compute.ray_field import RayField
TEXTS=[b"The weather today is sunny, and the kaolinite clay mineral forms in warm, humid soils.",b"def add(a, b):\n    return a + b\n\n```python\nprint(add(2, 3))\n```\nDone.",b"  Indented line; with (punct) {braces} and \"quotes\" -- plus a\ttab.\nNext line!",b"Photosynthesis converts light energy into chemical energy stored in glucose molecules within the plant.",b"User: Why do volcanoes erupt near tectonic boundaries?\nAdam: Volcanoes erupt where magma rises through cracks in the crust.\n\nUser: Who is Adam?\nAdam: I am Adam."]
def test_parity():
 d=tempfile.mkdtemp();src=os.path.join(d,"t.bin");ref=os.path.join(d,"ref.npz");open(src,"wb").write(b"\x00".join(TEXTS))
 subprocess.run([sys.executable,os.path.join(_H0,"tests","ray_field_ref.py"),src,ref],check=True,cwd=_H0);r=np.load(ref);e=RayField(path=os.environ.get('PAR_PACK')).load();worst=0.0;agree=0;n=0
 for i,t in enumerate(TEXTS):
  a=e.logits_for(t);b=r[f"t{i}"].astype(np.float64);worst=max(worst,float(np.abs(a-b).max()));agree+=int((a.argmax(1)==b.argmax(1)).sum());n+=len(t)
 print({"max_abs_logit_diff":worst,"top1_agree":f"{agree}/{n}"});assert worst<1e-3 and agree==n
if __name__=="__main__":test_parity()
