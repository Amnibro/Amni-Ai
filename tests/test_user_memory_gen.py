import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amni.serve import user_memory as U
CASES=[("My aunt lives in Tromso. My cousin lives in the same town as my aunt.",[(4,"Tromso"),(4,"Tromso")]),("My horse can't stand fireworks.",[(2,"fireworks")]),("My study buddy is a court reporter.",[(5,"court reporter")]),("My uncle adores bonsai.",[(7,"bonsai")]),("My neighbour lives in Ipoh, a city in Malaysia.",[(4,"Ipoh"),(3,"Malaysia")]),("Kotor is part of Montenegro.",[(3,"Montenegro")]),("My gecko is 18 now. My accountant's cockatiel is exactly as old as my gecko.",[(14,"18"),(14,"18")]),("My boss's dog is called Biscuit.",[(6,"Biscuit")]),("My babysitter spends most weekends on bowling.",[(7,"bowling")]),("My goat loves nothing more than blueberries.",[(7,"blueberries")]),("My teammate's twin has just hit 50.",[(14,"50")])]
def facts(t,**k):return [(f[1],f[2].decode()) for f in U._P(**k).turn(t.encode())]
def test_gen_rules():
    for t,e in CASES:assert facts(t,lt=True,gn=True)==e,(t,facts(t,lt=True,gn=True))
def test_gen_off_unchanged():
    assert facts("My cousin lives in the same town as my aunt.",lt=True,gn=False)!=[(4,"Tromso")] and facts("My uncle adores bonsai.",lt=True,gn=False)==[] and facts("My horse can't stand fireworks.",lt=True,gn=False)==[]
def test_gen_entities_distinct_and_stable():
    p=U._P(lt=True,gn=True);a=p.turn(b"My sister's husband lives in Oslo.");b=p.turn(b"My husband lives in Rome.");assert a[0][0]!=b[0][0]
    q=U._P(lt=True,gn=True);q.turn(b"My sister's husband lives in Oslo.");q.turn(b"Where does my sister's husband live?");assert q.qent==a[0][0] and q.qrel==4
def test_question_freeze():
    p=U._P(lt=True,gn=True);p.turn(b"My sister lives in Oslo.");p.turn(b"Where does my sister live? I forget.");s=U._P(lt=True,gn=True).turn(b"My sister lives in Oslo.")[0][0];assert p.qent==s and p.qrel==4
def test_gen_flag_follows_pack(monkeypatch):
    monkeypatch.delenv("AMNI_UMEM_GN",raising=False);monkeypatch.setenv("AMNI_UMEM_GN","0");assert not U._gn();monkeypatch.setenv("AMNI_UMEM_GN","1");assert U._gn()
