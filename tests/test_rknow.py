import os,sys,json,subprocess,urllib.request
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np,pytest
import amni.compute.rknow as RK
_H0=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));BIN=os.path.join(_H0,"crates","target","rknow","release","adam-rknow");HAVE=os.path.exists(os.path.join(RK.IXD,"meta.json"))
def _nonet(*a,**k):raise AssertionError("network touched")
def test_tokenizer_matches_rust():
 if not os.path.exists(BIN):pytest.skip("adam-rknow not built")
 for t in ["Toby’s wife, Beyoncé's 5D Mark IV — Nations stories","Ærø Ōsaka straße ﬁnance ½ x² Łódź İstanbul ΣΊΣΥΦΟΣ Москва 東京 3rd 1990s","the cities' bus's glass flies ties analysis"]:
  r=json.loads(subprocess.run([BIN,"tok",t],capture_output=True,text=True).stdout);assert [str(RK.fnv(w.encode())) for w in RK.toks(t)]==r
def test_entities_and_places():
 e=[v for v,_,_ in RK.ents("The first trip was made by Dwight D. Eisenhower in 1952 to the Jacob K. Javits Convention Center.")];assert "Dwight D. Eisenhower" in e and "Jacob K. Javits Convention Center" in e and "The" not in e
 assert "Beijing, China" in [c[0] for c in RK.gen("place","Headquartered in Beijing, China, the bank has 109 members.","where is it")]
@pytest.mark.skipif(not HAVE,reason="index_rknow missing")
def test_search_and_cited_answer(monkeypatch):
 monkeypatch.setattr(urllib.request,"urlopen",_nonet);I=RK.ix();h=I.search("hot tub time machine film",5);assert h and "Hot Tub Time Machine" in I.text(h[0][1])[0]
 r=RK.ask("who directed the film hot tub time machine?");assert r and r["tier"]=="high" and r["reply"].endswith(").") and "(Source: Wikipedia, " in r["reply"] and r["license"]=="CC BY-SA 4.0"
@pytest.mark.skipif(not HAVE,reason="index_rknow missing")
def test_personal_declined_temporal_hedged():
 for q in ["who is my sister's boss?","what is my bank password?"]:assert RK.ask(q) is None
 r=RK.ask("who is the current president of france?");assert r and r["tier"]!="high" and r["reply"].startswith("As of my Wikipedia snapshot") and "(Source: Wikipedia, " in r["reply"]
def test_route_order(monkeypatch):
 import amni.serve.gguf_runtime as g,amni.serve.user_memory as um,amni.compute.ray_field as rf
 calls=[];monkeypatch.setattr(rf,"engine",lambda *a,**k:type("E",(),{"impl":"py","v19":True})());monkeypatch.setattr(g,"_ray_guide",lambda e,m:None);monkeypatch.setattr(g,"_ray_learned",lambda m:None);monkeypatch.setattr(g,"_rknow",lambda m:(calls.append(m) or {"reply":"Steve Pink (Source: Wikipedia, Hot Tub Time Machine)."},True));monkeypatch.setenv("AMNI_RAY_LEARN","0");monkeypatch.setenv("AMNI_RCHAT","0")
 mem=lambda route,reply:(lambda *a,**k:{"pre":"","hist":[],"reply":reply,"route":route})
 monkeypatch.setattr(um,"turn",mem(False,"Your sister is Ana."));assert "".join(g._ray_stream("who is my sister?"))=="Your sister is Ana." and calls==[]
 monkeypatch.setattr(um,"turn",mem(False,None));assert "".join(g._ray_stream("who is my sister's boss?"))!="Steve Pink (Source: Wikipedia, Hot Tub Time Machine)." and calls==[]
 assert "".join(g._ray_stream("who directed hot tub time machine?"))=="Steve Pink (Source: Wikipedia, Hot Tub Time Machine)." and calls==["who directed hot tub time machine?"]
 monkeypatch.setenv("AMNI_RAY_RKNOW","0");calls.clear();"".join(g._ray_stream("who directed hot tub time machine?"));assert calls==[]
def test_fact_gate():
 for q in ["Do you have a favorite season?","I started learning guitar last month.","hi there, how's it going?","What's a good hobby I can do at home?","what should i name my dog?","how old are you?","who is my sister's boss?","who is the current president of france?"]:assert not RK.factq(q),q
 for q in ["who plays raj koothrappali on big bang theory?","when was the first harry potter movie released?","where is the danube river located?","how many episodes are in season 1 of the office?"]:assert RK.factq(q),q
@pytest.mark.skipif(not HAVE,reason="index_rknow missing")
def test_chit_chat_never_answered():
 for q in ["Do you have a favorite season?","I started learning guitar last month.","what is the meaning of life?","who is the best singer ever?","what's the best pizza topping?"]:assert RK.ask(q) is None,q
def test_fact_guard_skips_rchat(monkeypatch):
 import amni.serve.gguf_runtime as g,amni.serve.user_memory as um,amni.compute.ray_field as rf
 rc=[];monkeypatch.setattr(rf,"engine",lambda *a,**k:type("E",(),{"impl":"py","v19":True})());monkeypatch.setattr(g,"_ray_guide",lambda e,m:None);monkeypatch.setattr(g,"_ray_learned",lambda m:None);monkeypatch.setattr(g,"_rchat",lambda *a,**k:rc.append(a[0]) or "chat reply");monkeypatch.setenv("AMNI_RAY_LEARN","0");monkeypatch.setattr(um,"turn",lambda *a,**k:{"pre":"","hist":[],"reply":None,"route":False})
 monkeypatch.setattr(g,"_rknow",lambda m:(None,True));assert "".join(g._ray_stream("who played sax on the heat is on?"))=="I'm not sure about that one yet." and rc==[]
 monkeypatch.setattr(g,"_rknow",lambda m:(None,False));assert "".join(g._ray_stream("I started learning guitar last month."))=="chat reply" and rc==["I started learning guitar last month."]
 monkeypatch.setenv("AMNI_RKNOW_FACT_GUARD","0");monkeypatch.setattr(g,"_rknow",lambda m:(None,True));assert "".join(g._ray_stream("who played sax?"))=="chat reply"
def test_relational_facts():
 F=lambda y:[(f["k"],{k:[z[0] for z in v] for k,v in f["S"].items()}) for f in RK.facts(y)]
 assert ("par",{"agent":["Tina Majorino"],"char":["Dr. Heather Brooks"]}) in F("In the season 9 premiere, interns Dr. Jo Wilson (Camilla Luddington) and Dr. Heather Brooks (Tina Majorino) are introduced.")
 assert ("resp",{"agent":["Tina Majorino"],"char":["Heather Brooks"]}) in F("It was announced that Camilla Luddington, Gaius Charles, and Tina Majorino had been cast as Jo Wilson, Shane Ross, and Heather Brooks, respectively.")
 assert ("as",{"agent":["Betsy Sodaro"],"char":["Dabby"]}) in F("Betsy Sodaro as Dabby")
 assert ("loc",{"place":["Jacob K. Javits Convention Center"]}) in F("The New York Comic Con is held annually at the Jacob K. Javits Convention Center in New York City.")
 assert any(k=="pas" and s.get("agent")==["David Swift"] for k,s in F("The Parent Trap is a 1961 American comedy film written and directed by David Swift."))
 assert not any(k=="as" for k,_ in F("Sasha Pieterse continued to portray Alison DiLaurentis as Alison struggles."))
def test_question_parse_and_slots():
 qp=RK.qparse("who plays emily's father on pretty little liars?");assert qp[0]==1 and qp[1]=="agent" and qp[6]==({"emily","father"},True)
 assert RK.qparse("who did tina majorino play in grey's anatomy?")[1]=="char" and RK.qparse("when was elegy written in a country churchyard published?")[0]==12
 f=lambda c,d="":{"r":1,"S":{"agent":[("X",0,1)],"char":[(c,0,1)]},"c":d,"k":"as"}
 assert RK.slot(f("Emily Fields"),"agent",{"emily","father"},True)==0.5 and RK.slot(f("Wayne Fields",", Emily's father"),"agent",{"emily","father"},True)==1.0
 assert RK.slot(f("John Paul Getty III's mother"),"agent",{"john","paul","getty"})==0.5
def test_wide_gate():
 for q in ["the name given to the noble class of russia is?","who sings i'm never going to give you up?","when did if you wannabe my lover come out?","who is the current president of france?"]:assert RK.factq(q,True),q
 for q in ["Do you have a favorite season?","what time is it?","what are we having for dinner tonight?","who is my sister's boss?","how old are you?","I started learning guitar last month."]:assert not RK.factq(q,True),q
 assert not RK.factq("who is the current president of france?") and RK.askq("who is the current president of france?") and RK.askq("the name given to the noble class of russia is?")
def test_scorer_normalization():
 import importlib,sys as _s
 _s.path.insert(0,os.path.join(_H0,"scripts"));AN=importlib.import_module("adam_nightly");D=__import__("re").compile("not sure")
 assert AN.score_qa("Derek Jacobi (full name Sir Derek George Jacobi) (Source: Wikipedia, X).",["Sir Derek Jacobi"],D)["hit"] and AN.score_qa("1990 (Source: Wikipedia, X).",["in 1990"],D)["em"] and AN.score_qa("Zoe Saldaña.",["Zoe Saldana"],D)["em"] and AN.score_qa("Dr. Heather Brooks (Source: Wikipedia, X).",["Dr. Heather Brooks"],D)["em"]
def test_factual_skips_guide_and_learn(monkeypatch):
 import amni.serve.gguf_runtime as g,amni.serve.user_memory as um,amni.compute.ray_field as rf,amni.compute.ray_learn as RL
 seen=[];monkeypatch.setattr(rf,"engine",lambda *a,**k:type("E",(),{"impl":"py","v19":True})());monkeypatch.setattr(g,"_ray_guide",lambda e,m:seen.append("guide"));monkeypatch.setattr(g,"_ray_learned",lambda m:seen.append("learned"));monkeypatch.setattr(RL,"learn",lambda *a:seen.append("learn"));monkeypatch.setattr(g,"_rchat",lambda *a,**k:seen.append("rchat") or "chat reply");monkeypatch.setattr(g,"_rknow",lambda m:(None,False));monkeypatch.setattr(um,"turn",lambda *a,**k:{"pre":"","hist":[],"reply":None,"route":True})
 assert "".join(g._ray_stream("the name given to the noble class of russia is?"))=="I'm not sure about that one yet." and seen==[]
 assert "".join(g._ray_stream("who is the current chief justice of the supreme court?"))=="I'm not sure about that one yet." and seen==[]
def test_statement_gate_and_chat_negatives():
 for q in ["the name given to the noble class of russia is","largest producer of mica in the world is","voice of toph in avatar the last airbender","names of characters in back to the future","how many us aircraft carriers do we have","who sings i wanna be like you jungle book 2016","where is the continental divide on i 70","is the vena cava a vein or artery"]:assert RK.askq(q),q
 for q in ["long day at work","that movie was amazing","the kids are driving me crazy","who made you","how old are you?","tell me a joke","what do you think about camping?","is pineapple on pizza okay","which is the best season","who do you think will win the game tonight?"]:assert not RK.askq(q),q
 E=os.path.join(_H0,"data","concepts","eval","rknow");bad=[x["q"] for f in ("chat_neg_dev.json","chat_neg_dev2.json") if os.path.exists(os.path.join(E,f)) for x in json.load(open(os.path.join(E,f))) if RK.askq(x["q"])];assert not bad,bad
@pytest.mark.skipif(not HAVE,reason="index_rknow missing")
def test_always_answer_tiers(monkeypatch):
 import re,importlib,sys as _s
 monkeypatch.setattr(urllib.request,"urlopen",_nonet);_s.path.insert(0,os.path.join(_H0,"scripts"));DC=importlib.import_module("v19_eval_run").DECLINE
 for q in ["who plays general gage in sons of liberty","competition commission of india comes under which ministry","when was the first harry potter movie released","who directed the film hot tub time machine"]:
  r=RK.ask(q);assert r and r["tier"] in ("high","medium","low") and "(Source: Wikipedia, " in r["reply"] and not DC.search(r["reply"].lower()),q
 assert RK.ask("who directed the film hot tub time machine")["tier"]=="high" and RK.ask("qwxz zzkq blorf flarn") is None
 assert RK.alias("Peter Ostrum")>={"Peter Ostrum","Peter Gardner Ostrum"}
def test_gbdt_ranker_loaded():
 if RK.GBM is None:pytest.skip("rknow_rank.json missing")
 X=np.zeros((3,len(RK.GBM["feat"])+len(RK.GBM["rel"])));p=RK.gbp(X);assert p.shape==(3,) and np.all((p>0)&(p<1)) and RK.CWTS["bias"]==RK.GBM["cw"][0]
