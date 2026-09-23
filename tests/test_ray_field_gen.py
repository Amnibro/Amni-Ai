import os,sys,time
_H0=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,_H0)
from amni.compute.ray_field import RayField
def test_gen():
 e=RayField().load()
 for p in ["The water cycle begins when","How do volcanoes form?\n","def fibonacci(n):\n    "]:
  t=time.time();s="".join(e.stream(p,max_bytes=240,seed=7));dt=time.time()-t;print({"prompt":p,"mode":"stream","bytes":len(s.encode()),"bytes_per_s":round(len(s.encode())/dt)});print(p+s+"\n")
  t=time.time();b=e.beam(p,maxlen=200);dt=time.time()-t;print({"prompt":p,"mode":"beam8","bytes":len(b.encode()),"bytes_per_s":round(len(b.encode())/dt)});print(p+b+"\n");assert s and b
if __name__=="__main__":test_gen()
