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
 r=RK.ask("who directed the film hot tub time machine?",gate=False);assert r and r["reply"].endswith(").") and "(Source: Wikipedia, " in r["reply"] and r["license"]=="CC BY-SA 4.0"
@pytest.mark.skipif(not HAVE,reason="index_rknow missing")
def test_personal_and_temporal_declined():
 for q in ["who is my sister's boss?","what is my bank password?","who is the current president of france?"]:assert RK.ask(q) is None
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
