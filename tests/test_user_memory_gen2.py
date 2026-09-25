import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amni.serve import user_memory as U
CASES=[("My study buddy rents a little flat in Napier.",[(4,"Napier")]),("My goddaughter breaks out in hives around strawberries.",[(8,"strawberries")]),("If you ever go to Tainan, that's where my partner lives.",[(4,"Tainan")]),("My manager's boyfriend's daily driver is a Fiat Panda.",[(9,"Fiat Panda")]),("If my brother were a pilot, he would be thrilled.",[]),("My sister doesn't live in Oslo.",[]),("We went with lavender for my twin brother's teapot.",[(10,"lavender")]),("The one thing my gecko can't stand is the hairdryer.",[(2,"the hairdryer")])]
def facts(t,**k):return [(f[1],f[2].decode()) for f in U._P(**k).turn(t.encode())]
def test_gen2_rules():
    for t,e in CASES:assert facts(t,lt=True,gn=True,g2=True)==e,(t,facts(t,lt=True,gn=True,g2=True))
def test_gen2_off_unchanged():
    assert facts("My study buddy rents a little flat in Napier.",lt=True,gn=True)!=[(4,"Napier")] and facts("My goddaughter breaks out in hives around strawberries.",lt=True,gn=True)!=[(8,"strawberries")]
def test_gen2_question_entity_and_relation():
    p=U._P(lt=True,gn=True,g2=True);a=p.turn(b"My stepsister did her masters in microbiology.");p.turn(b"Pop quiz: what's my stepsister's academic background? I forget.");assert (p.qent,p.qrel)==(a[0][0],12)
    p=U._P(lt=True,gn=True,g2=True);p.turn(b"My pen pal keeps a Mini Cooper in the garage.");p.turn(b"Out of curiosity, what's parked in my landlady's garage?");assert p.qrel==9 and p.qent not in (0,1) and not [f for f in p.facts if f[0]==p.qent]
def test_gen2_pending_same_as_link():
    p=U._P(lt=True,gn=True,g2=True);p.turn(b"My tortoise shares a name with my gecko.");p.turn(b"My gecko is called Biscuit.");p.turn(b"What do we call my tortoise?");assert [f[2] for f in p.facts if f[0]==p.qent and f[1]==6]==[b"Biscuit"]
def test_gen2_region_hop_and_names():
    p=U._P(lt=True,gn=True,g2=True);f=p.turn(b"My lab partner lives over in Kalamazoo, a town in Michigan.");assert [(x[1],x[2]) for x in f]==[(4,b"Kalamazoo"),(3,b"Michigan")] and p.names[f[0][0]]=="lab partner"
def test_gen2_flag_follows_pack(monkeypatch):
    monkeypatch.setenv("AMNI_UMEM_G2","0");assert not U._g2();monkeypatch.setenv("AMNI_UMEM_G2","1");assert U._g2()
def test_gen2_migrates_old_entity_keys(monkeypatch):
    monkeypatch.setenv("AMNI_UMEM_G2","1");monkeypatch.setenv("AMNI_UMEM_LT","1");U._EID.clear()
    old,new=U._eid("sister-in-law",2),U._eid("sister-in-law",'cur');assert old and new and old!=new
    assert U._mig([U.F(old,4,1,0.0,0,b'\0'*8,b'Oslo','sister-in-law')])[0].ent==new;U._EID.clear()
