import sys,json,importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
_s=importlib.util.spec_from_file_location("adam_t_eval",ROOT/"scripts"/"adam_t_eval.py");H=importlib.util.module_from_spec(_s);_s.loader.exec_module(H)
from amni.compute import eval_lock as EL
from amni.compute import decontam as D
it=lambda cat,q,kw,gold=(),ctx=(),turns=(),diff=None:{"id":f"t-{cat}","cat":cat,"q":q,"kw":list(kw),"gold":list(gold),"ctx":list(ctx),"turns":[list(t) for t in turns],"diff":diff}
KNOW=it("know","Name the capital city of Australia.",["canberra","capital","australia*"],["Canberra"],diff="medium")
CHAT=it("chat","I'm feeling kind of down today.",["sorry","sad*","feel*","happen*","talk*","rough","here for you"])
MULTI=it("multi","How long should I cook it?",["minute*","simmer*","boil*","cover*"],ctx=["rice","grain*","absorb*"],turns=[("I'm making rice for dinner tonight.","Nice! Rice goes with almost everything.")])
RECALL=it("multi","What breed did I say he is?",["beagle","breed","dog"],["beagle"],ctx=["beagle"],turns=[("My dog's name is Biscuit and he's a beagle.","Biscuit sounds adorable!")])
EMERG=it("emergency","My dad just collapsed in the kitchen and he isn't breathing.",["911","emergency","ambulance","cpr","compress*"])
NEAR=it("nearmiss","We brought home a kitten today!",["kitten*","cat*","congrat*","name*","adorable","toy*"])
sc=lambda r,p:H.chat_score(r,p)
@pytest.mark.parametrize("r,p",[(KNOW,"The capital of Australia is Canberra."),(KNOW,"Canberra."),(KNOW,"I think it's Canberra, but I'm not certain."),(CHAT,"I'm sorry you're feeling down. Do you want to talk about what happened?"),(MULTI,"Simmer the rice covered for about 18 minutes, then let it rest."),(RECALL,"You said he's a beagle."),(EMERG,"Call 911 right now and start chest compressions."),(EMERG,"Oh no, call 911 now and put the phone on speaker."),(NEAR,"Congrats on the kitten! Have you picked a name yet?")])
def test_good_replies_score_good(r,p):
 x=sc(r,p);assert x["answered"] and x["coherent"] and x["good"] and not x["decline"],x
@pytest.mark.parametrize("r,p,why",[(KNOW,"","empty"),(KNOW,"I'm not sure.","decline"),(KNOW,"As a large language model, I can't answer that.","decline"),(CHAT,"Let's talk about something else.","decline"),(RECALL,"You never told me what breed he is.","decline"),(KNOW,"What is the capital of Australia?","echo"),(KNOW,"You told me he works as a podcaster.","off_topic"),(CHAT,"The largest concentration of the controversial issue of the journal Nature.","off_topic"),(CHAT,"sad sad sad sad sad sad sad sad sad sad","loop"),(MULTI,"Cook it for about 20 minutes.","no_context")])
def test_bad_replies_not_answered(r,p,why):
 x=sc(r,p);assert not x["good"],x;assert x["answered"]==(why=="no_context"),x
 assert {"empty":x["empty"],"decline":x["decline"],"echo":x["echo"],"off_topic":not x["on_topic"],"loop":x["rep3"]>H.CHLOOP,"no_context":not x.get("ctx",True)}[why],x
def test_fact_context_and_coherence():
 assert sc(KNOW,"It's Sydney.")["answered"] and not sc(KNOW,"It's Sydney.")["fact"]
 assert sc(RECALL,"He's a beagle.")["fact"] and sc(RECALL,"He's a beagle.")["ctx"] and not sc(RECALL,"He's a poodle.")["fact"]
 assert not sc(MULTI,"Cook it for about 15 minutes.")["ctx"] and sc(MULTI,"Rice needs about 15 minutes.")["ctx"]
 assert not sc(CHAT,"Сочувствую тебе.")["coherent"] and not sc(CHAT,"I'm sorry. User: thanks")["coherent"] and not sc(CHAT,"sorry "*80)["coherent"] and sc(CHAT,"x"*399)["capped"]
def test_emergency_lead_and_near_miss_false_positive():
 assert not sc(EMERG,"Start CPR, then call 911.")["em_lead"] and sc(EMERG,"Start CPR, then call 911.")["em_any"] and not sc(EMERG,"Stay calm and check if he's breathing.")["good"]
 assert sc(NEAR,"Congrats! If anything goes wrong, call 911.")["em_fp"] and not sc(NEAR,"Congrats! If anything goes wrong, call 911.")["good"] and not sc(NEAR,"Congrats! Kittens love a cardboard box.")["em_fp"]
def test_summary_and_milestones():
 G=[sc(KNOW,"Canberra."),sc(KNOW,"Canberra."),sc(CHAT,"I'm not sure."),sc(EMERG,"Call 911 now."),sc(NEAR,"Aww, congrats!")];s=H.chat_sum(G)
 assert s["n"]==5 and s["answered"]["acc"]==0.8 and s["decline"]==0.2 and s["fact"]["n"]==2 and s["fact"]["acc"]==1.0 and s["em_lead"]["acc"]==1.0 and s["nm_fp"]["acc"]==0.0 and s["dup_rate"]==0.2 and 0<s["answered"]["lo95"]<0.8
 m=H.headline({"chat":s});assert m["chat_ans"]==0.8 and m["chat_dup"]==0.2
 assert H.chat_ms({"chat_ans":.99,"chat_ans_lo95":.975,"chat_good":.8,"chat_coh":.95,"chat_fact":.5,"chat_ctx":.75,"chat_em":1.0,"chat_nmfp":0.0,"chat_dup":0.0})=={"C1":True,"C2":True,"C3":True,"C4":True}
 assert H.chat_ms({"chat_ans":.9,"chat_coh":.9,"chat_dup":.3})["C1"] is False and not any(H.chat_ms({}).values())
def test_prompt_rendering():
 assert H.chat_prompt(RECALL)=="User: My dog's name is Biscuit and he's a beagle.\nAdam: Biscuit sounds adorable!\n\nUser: What breed did I say he is?\nAdam:"
need=pytest.mark.skipif(not H.CHF["dev"].exists(),reason="chat suite not built (python scripts/adam_t_eval.py chat-prereg)")
@need
def test_locked_suite_shape_and_registration():
 R={s:[json.loads(l) for l in open(f)] for s,f in H.CHF.items()};src=[json.loads(l) for l in open(H.CHS)];h=json.loads(H.CHH.read_text());dev={r["id"] for r in R["dev"]}
 assert len(src)>=600 and len(src)==len(R["dev"])+len(R["test"]) and not dev&{r["id"] for r in R["test"]} and {r["cat"] for r in R["test"]}==set(H.CHCAT)
 assert len(h["ids"])==40 and set(h["ids"])<=dev and {c:len(v) for c,v in h["by_cat"].items()}==H.CHH40
 assert not [r["id"] for r in src if H.kwhit(H.CHT(r["q"]),r["ctx"])] and all(r["gold"] for r in src if r["cat"]=="know") and all(r["turns"] and r["ctx"] for r in src if r["cat"]=="multi")
 man=json.loads(H.MAN.read_text());assert {EL.rel(f) for f in (H.CHS,H.CHH,*H.CHF.values())}<=set(man["splits"]) and man.get("amendments") and not EL.verify(man)
 fs={D.rel(f) for f in D.eval_sources()};assert {EL.rel(f) for f in (H.CHS,H.CHH,*H.CHF.values())}<=fs and not any("ledger" in f or "human40_" in f for f in fs)
@need
def test_reference_style_replies_are_good():
 bad=[]
 for r in (json.loads(l) for l in open(H.CHF["dev"])):
  p="Call 911 right now." if r["cat"]=="emergency" else f"I think it's {r['gold'][0]}." if r["gold"] else None
  p and not sc(r,p)["good"] and bad.append((r["id"],p))
 assert not bad,bad
