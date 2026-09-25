import os,sys,time,numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))));sys.path[:0]=[_H0,os.path.dirname(os.path.abspath(__file__))];os.environ.setdefault("AMNI_RAY_ARCH","cpu")
from amni.compute.ray_field import RayField
HOME=os.environ.get("AMNI_HOME",_H0);EX=os.path.join(HOME,"exports","gf17_continuum")
PACKS={"v16":os.path.join(EX,"rayhash_v16_big2_L21_e20_hop4_q5_s150k_pk_q5_i8.npz"),"v18":os.path.join(EX,"rayhash_v18_chat2_qt_s20k_sid.npy")}
TEXTS=[b"The weather today is sunny, and the kaolinite clay mineral forms in warm, humid soils.",b"def add(a, b):\n    return a + b\n\n```python\nprint(add(2, 3))\n```\nDone.",b"  Indented line; with (punct) {braces} and \"quotes\" -- plus a\ttab.\nNext line!",b"Photosynthesis converts light energy into chemical energy stored in glucose molecules within the plant.",b"User: Why do volcanoes erupt near tectonic boundaries?\nAdam: Volcanoes erupt where magma rises through cracks in the crust.\n\nUser: Who is Adam?\nAdam: I am Adam."]
PROMPTS=[b"User: What is the capital of France?\nAdam: ",b"User: \nAdam: ",b"User: How do bees make honey?\nAdam: "]
CONTS=[b"Paris is the capital of France.",b"Bees collect nectar from flowers and store it in the hive.",b"I am not sure."]
def field(tag:str,**kw)->RayField:return RayField(path=PACKS[tag],syn=os.path.join(HOME,"data","big2","synsets.txt"),words=os.path.join(HOME,"data","big2","words.txt"),**kw)
def fix(e:RayField)->RayField:e.wcnt_path=os.path.join(HOME,"data","big2","wordcounts.txt");return e
def qa_ref(out:str,tag:str="v18"):
 from amni.compute import ray_qa
 from scripts.ray_chat_eval import QS,QS2,QS3
 e=fix(field(tag)).load();qa=ray_qa.QA(os.path.join(EX,"ray_qa_index.npz"));f=open(out,"w")
 for q in QS+QS2+QS3:p=ray_qa.pick(e,qa,q);f.write("\x1f".join([q,",".join(repr(x[0]) for x in qa.search(q,20)),p[0][3].hex() if p else "",repr(p[0][1]) if p else ""])+"\n")
def main():
 tag,out=sys.argv[1],sys.argv[2]
 if tag=="qa":return qa_ref(out)
 if tag.startswith("tables_"):
  from amni.compute.ray_field import load_pack
  pk=load_pack(PACKS[tag[7:]]);[np.save(f"{out}_{k}.npy",np.ascontiguousarray(pk[k])) for k in ("dp","sid","sw","bias")];return
 e=fix(field(tag)).load();r={}
 for i,t in enumerate(TEXTS):r[f"t{i}"]=e.logits_for(t).astype(np.float32)
 P=[p for p in PROMPTS for _ in CONTS];C=[c for _ in PROMPTS for c in CONTS];r["score"]=e.score(P,C);t=time.time()
 r["stream"]=np.frombuffer("".join(e.stream("User: What is the capital of France?\nAdam:",max_bytes=160,temp=0.6,topk=8,stop=("\n\n","\nUser:","\nAdam:"),seed=11)).encode(),np.uint8)
 r["guided"]=np.frombuffer("".join(e.stream("User: How do bees make honey?\nAdam:",max_bytes=160,temp=0.6,topk=8,stop=("\n\n","\nUser:","\nAdam:"),seed=11,guide=b" Bees collect nectar from flowers and store it in the hive.",beta=6.0)).encode(),np.uint8)
 r["beam"]=np.frombuffer(e.beam("User: What is gravity?\nAdam:",width=4,expand=3,maxlen=40).encode(),np.uint8);r["bpb"]=np.array([e.bpb(hold=os.path.join(HOME,"data","clean1","hold.bin"),batches=1,ctx=32,bs=16)])
 np.savez(out,**r);print(tag,{k:v.shape for k,v in r.items()},bytes(r["stream"]).decode(),"|",bytes(r["guided"]).decode(),"|",bytes(r["beam"]).decode(),r["bpb"])
if __name__=="__main__":main()
