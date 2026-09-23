import os,sys,time
_H0=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,_H0)
from amni.compute.ray_field import RayField
def test_bpb(batches:int=int(os.environ.get("BPB_BATCHES","60"))):
 e=RayField().load();t=time.time();b=e.bpb(batches=batches);dt=time.time()-t;print({"bits_per_byte":round(b,4),"batches":batches,"positions":batches*256*128,"seconds":round(dt,1),"positions_per_s":round(batches*256*128/dt)});assert b<1.62
if __name__=="__main__":test_bpb()
