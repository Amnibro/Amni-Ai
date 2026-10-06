import os,sys,json,time
from amni.serve import adam_t_client as C,blender_atex as B
assert B.active(),"AMNI_BLENDER must be atex:<bake>"
t=time.time();c=C.client();ld=round(time.time()-t,1);v=B.proc_vram()
R={"bake":os.environ["AMNI_BLENDER"],"opts":json.loads(os.environ.get("AMNI_BLENDER_OPTS") or "{}"),"client_type":type(c).__name__,"same_instance":C.client() is c,"load_s":ld,"status":c.status(),"replies":{"blend":c.gen("What is Marlowe Bridge?",rc=["Marlowe Bridge: Marlowe Bridge is a stone arch bridge over the River Tane in the county of Eastmere. It carries a single lane of traffic."]),"chat":c.gen("Hi Adam, long day at work and I am wiped out."),"uncovered":c.gen("Who designed the Pelham Tower?")},"proc_vram_GB_after":round(B.proc_vram()/1e9,3),"proc_vram_GB_after_load":round(v/1e9,3)}
print(json.dumps(R,indent=1));sys.exit(0 if all(R["replies"].values()) and R["same_instance"] else 1)
