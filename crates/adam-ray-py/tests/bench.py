import os,sys,json,time,subprocess
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))));sys.path[:0]=[_H0,os.path.dirname(os.path.abspath(__file__))]
HOME=os.environ.get("AMNI_HOME",_H0);EX=os.path.join(HOME,"exports","gf17_continuum");D=os.path.join(HOME,"data","big2")
PACKS={"v16":os.path.join(EX,"rayhash_v16_big2_L21_e20_hop4_q5_s150k_pk_q5_i8.npz"),"v18":os.path.join(EX,"rayhash_v18_chat2_qt_s20k_sid.npy")}
def engine(impl:str,arch:str,pk:str):
 if impl=="rust":import adam_ray;return adam_ray.RayField(pk,os.path.join(D,"synsets.txt"),os.path.join(D,"words.txt"),os.path.join(D,"wordcounts.txt"),arch,HOME).load()
 from amni.compute.ray_field import RayField
 e=RayField(path=pk,syn=os.path.join(D,"synsets.txt"),words=os.path.join(D,"words.txt"),arch=arch);e.wcnt_path=os.path.join(D,"wordcounts.txt");return e.load()
def one(impl:str,arch:str,tag:str,reps:int=3)->dict:
 t=time.time();e=engine(impl,arch,PACKS[tag]);lt=time.time()-t;q=b"User: What is the capital of France?\nAdam: ";c=[(b"Paris is the capital and largest city of France, known for art, fashion, food and culture. "*2)[:150+i] for i in range(20)]
 run=lambda:"".join(e.stream("User: Tell me about volcanoes.\nAdam:",max_bytes=400,temp=0.7,topk=12,stop=(),seed=7)).encode();run();e.score([q]*20,c)
 t=time.time();nb=sum(len(run()) for _ in range(reps));ts=time.time()-t;t=time.time();[e.score([q]*20,c) for _ in range(reps)];tc=time.time()-t;T=max(len(q)+len(x) for x in c)
 return {"impl":impl,"arch":arch,"pack":tag,"load_s":round(lt,2),"stream_Bps":round(nb/ts,1),"score20_Bps":round(reps*20*T/tc,1),"score20_call_ms":round(1000*tc/reps,1)}
def main():
 if sys.argv[1:2]==["one"]:print(json.dumps(one(*sys.argv[2:5])));return
 rows=[]
 for tag in sys.argv[1:] or ["v16","v18"]:
  for impl,arch in [("rust","vulkan"),("rust","cpu"),("taichi","cpu"),("taichi","vulkan")]:
   r=subprocess.run([sys.executable,os.path.abspath(__file__),"one",impl,arch,tag],capture_output=True,text=True,env={**os.environ,"AMNI_HOME":HOME});rows.append(json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode==0 else {"impl":impl,"arch":arch,"pack":tag,"error":r.stderr[-300:]});print(json.dumps(rows[-1]),flush=True)
 print("| pack | impl | arch | stream B/s (1 beam) | score B/s (batch 20) | score call ms | load s |\n|---|---|---|---|---|---|---|");[print(f"| {r['pack']} | {r['impl']} | {r['arch']} | {r.get('stream_Bps')} | {r.get('score20_Bps')} | {r.get('score20_call_ms')} | {r.get('load_s')} |") for r in rows]
if __name__=="__main__":main()
