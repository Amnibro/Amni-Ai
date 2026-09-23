import os,sys
_H0=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,_H0);os.environ["AMNI_GEN_BACKEND"]="ray"
from amni.serve import gguf_runtime as g
def test_backend():
 assert g.enabled() and g.backend_name()=="ray" and g.model_id()=="adam:ray-v16";h=g.health();assert h["ok"] and h["backend"]=="ray"
 r=g.chat("Tell me about rivers.",max_new_tokens=120);assert r["tier"]=="tier_ray" and r["answer"] and r["tokens"]>0
 parts=list(g.chat_stream("Tell me about mountains.",max_new_tokens=120));assert parts and len("".join(parts))>10
 s=g.svc();t,n=s.chat("What is a star?",max_new_tokens=80);assert t and n>0 and "------" not in t
 gen=g.chat_stream("Tell me about forests.",max_new_tokens=400);next(gen);gen.close()
 print({"chat":r["answer"][:80],"stream_parts":len(parts),"svc":t[:60]})
if __name__=="__main__":test_backend()
