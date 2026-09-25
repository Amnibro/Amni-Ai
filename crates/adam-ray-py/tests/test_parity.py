import os,sys,json,pytest,numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))));sys.path[:0]=[_H0,os.path.dirname(os.path.abspath(__file__))];os.environ.setdefault("AMNI_RAY_ARCH","cpu")
import adam_ray
from amni.compute.ray_field import RayField
from amni.compute import ray_qa
from scripts.ray_chat_eval import QS,QS2,QS3
from ray_ref import TEXTS,PACKS,PROMPTS,CONTS,HOME,LX,field,fix
ARCH=os.environ.get("ADAM_RAY_TEST_ARCH","vulkan");STOP=("\n\n","\nUser:","\nAdam:")
def rust(tag:str)->"adam_ray.RayField":return adam_ray.RayField(PACKS[tag],os.path.join(LX,"synsets.txt"),os.path.join(LX,"words.txt"),os.path.join(LX,"wordcounts.txt"),ARCH,HOME).load()
def engine_parity(tag:str,ref:RayField=None)->dict:
 ref=ref or fix(field(tag)).load();e=rust(tag);worst=0.0;agree=0;n=0
 for t in TEXTS:a=e.logits_for(t);b=ref.logits_for(t);worst=max(worst,float(np.abs(a-b).max()));agree+=int((a.argmax(1)==b.argmax(1)).sum());n+=len(t)
 P=[p for p in PROMPTS for _ in CONTS];C=[c for _ in PROMPTS for c in CONTS];sd=float(np.abs(e.score(P,C)-ref.score(P,C)).max())
 g=lambda x,**k:"".join(x.stream("User: How do bees make honey?\nAdam:",max_bytes=160,temp=0.6,topk=8,stop=STOP,seed=11,**k))
 out={"pack":tag,"backend":e.backend,"max_abs_logit_diff":worst,"top1_agree":f"{agree}/{n}","max_abs_score_diff":sd,"stream_equal":g(e)==g(ref),"guided_equal":g(e,guide=b" Bees make honey from nectar.",beta=6.0)==g(ref,guide=b" Bees make honey from nectar.",beta=6.0)}
 print(json.dumps(out));assert worst<1e-3 and agree==n;return out
def qa_parity(tag:str="v18",qs:list=None)->dict:
 qs=qs or QS+QS2+QS3;path=os.path.join(HOME,"exports","gf17_continuum","ray_qa_index.npz");pq=ray_qa.QA(path);rq=adam_ray.QA(path);ref=fix(field(tag)).load();e=rust(tag);r={"n":len(qs),"search_exact":0,"search_tie_equal":0,"pick_equal":0,"pick_q_equal":0,"guide_equal":0,"rows":[]}
 for q in qs:
  a=pq.search(q,20);b=rq.search(q,20);ka=[(x[2],x[3]) for x in a];kb=[(x[2],x[3]) for x in b];sa=np.array([x[0] for x in a]);sb=np.array([x[0] for x in b])
  ex=ka==kb;tie=len(a)==len(b) and bool(np.allclose(sa,sb,rtol=1e-5,atol=1e-7)) and all(x in kb or abs(s-sa[-1])<=1e-6*max(1,abs(sa[-1])) for x,s in zip(ka,sa))
  pa=ray_qa.pick(ref,pq,q);pb=adam_ray.pick(e,rq,q);pe=len(pa)==len(pb) and all(x[3]==y[3] and abs(x[0]-y[0])<=1e-5*max(1,abs(x[0])) and abs(x[1]-y[1])<=1e-9 for x,y in zip(pa,pb));pqe=[x[2] for x in pa]==[x[2] for x in pb];cv=float(os.environ.get("QA_COV","0.3"));ge=bool(pa and pa[0][1]>=cv)==bool(pb and pb[0][1]>=cv)
  r["search_exact"]+=ex;r["search_tie_equal"]+=ex or tie;r["pick_equal"]+=pe;r["pick_q_equal"]+=pqe;r["guide_equal"]+=ge;r["rows"].append({"q":q,"search_exact":ex,"tie_equal":ex or tie,"pick_equal":pe,"py_pick":pa[0][2] if pa else None,"rs_pick":pb[0][2] if pb else None,"py_cov":pa[0][1] if pa else None,"rs_cov":pb[0][1] if pb else None})
 print(json.dumps({k:v for k,v in r.items() if k!="rows"}));[print(x) for x in r["rows"] if not (x["tie_equal"] and x["pick_equal"])];assert r["search_tie_equal"]==r["n"] and r["pick_equal"]==r["n"];return r
@pytest.mark.skip(reason="legacy_grok retired: v16 big2 pack moved to exports/legacy_grok; no clean v16-format pack exists")
def test_v16():engine_parity("v16")
def test_v18():engine_parity("v18")
def test_qa():qa_parity()
if __name__=="__main__":[f() for f in (test_v16,test_v18,test_qa) if not sys.argv[1:] or f.__name__ in sys.argv[1:]]
