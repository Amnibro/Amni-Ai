import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from amni.serve import user_memory as U
SEED=["My sister lives in Rome.","I live in Paris.","My name is Anna.","I'm allergic to peanuts.","My sister's husband lives in Lyon.","My dog is called Rex.","My brother lives in Oslo.","My brother works as a nurse.","My brother drives a Volvo."]
B=("Oslo","nurse","Volvo")
FORGET=[("Forget what I told you about my brother.",B),("Please forget where my brother lives.",("Oslo",)),("Forget about my brother.",B),("Forget my brother.",B),("Forget everything I said about my brother.",B),("Forget anything I've told you about my brother.",B),("Forget what I said about my brother.",B),("Forget what I mentioned about my brother.",B),("Please forget all the stuff I told you about my brother.",B),("Don't remember my brother anymore.",B),("Don't remember anything about my brother anymore.",B),("You don't need to remember my brother anymore.",B),("No need to remember my brother's car.",("Volvo",)),("Stop remembering things about my brother.",B),("Delete what you know about my brother.",B),("Delete everything you know about my brother.",B),("Erase what you have on my brother.",B),("Remove my brother from your memory.",B),("Wipe what you know about my brother.",B),("Clear everything you have saved about my brother.",B),("Never mind what I said about my brother.",B),("Nevermind what I told you about my brother.",B),("Disregard what I told you about my brother.",B),("Ignore what I said about my brother.",B),("Scratch what I said about my brother.",B),("Can you forget what I told you about my brother?",B),("Could you please forget about my brother?",B),("I want you to forget about my brother.",B),("I'd like you to forget what I told you about my brother.",B),("Okay, forget about my brother.",B),("Hey Adam, please delete my brother's info.",B),("Forget where my brother lives.",("Oslo",)),("Forget my brother's job.",("nurse",)),("Please unlearn what I told you about my brother.",B),("Forget about him.",B),("Forget what I told you about my sister's husband.",("Lyon",)),("Forget my name.",("Anna",)),("Forget what I told you about my dog.",("Rex",)),("Actually, forget what I just told you about my brother.",B),("Forget that my brother lives in Oslo.",("Oslo",)),("Forget that I'm allergic to peanuts.",("peanuts",)),("Don't store anything about my brother.",B),("Forget everything about my sister.",("Rome",)),("Please don't keep my brother's details anymore.",B)]
NEAR=[("I forgot my keys.",0),("Don't forget to call mom.",0),("I always forget where I parked.",0),("Did you forget my name?",0),("Forget it.",0),("Forget about it.",0),("Never mind.",0),("Never mind, it doesn't matter.",0),("Delete the file on my desktop.",0),("Remove the tomatoes from my salad recipe.",0),("Erase the whiteboard.",0),("Clear my schedule for Friday.",0),("How do I delete my account?",0),("Can you help me remember my brother's birthday?",0),("I can't forget what my brother said.",0),("My brother forgets everything.",0),("Stop talking about my brother.",0),("Ignore my brother, he's joking.",0),("Let's forget about work tonight.",0),("Forgetting things is normal.",0),("Don't remember where I parked.",0),("I don't remember my brother's address.",0),("Please don't forget my brother lives in Oslo.",0),("Never forget where you came from.",0),("I just got a kitten!",0),("I got a kitten named Biscuit.",0),("My sister got a puppy.",0),("I have a hamster.",0),("We got a dog last week.",0),("My mom just got a parrot!",0),("I got a promotion.",0),("I got a job.",1),("I got a new job as a teacher.",1),("I just got a Tesla.",0),("My brother has a kitten.",0),("I have a new kitten.",0),("Wipe the table before dinner.",0),("Scratch that, I'm tired.",0),("Why do people forget names?",0),("I got a new laptop.",0),("My sister has a gecko.",0)]
MODES={"prod":"1","legacy":"0"}
def _env(mode,d):
    os.environ["AMNI_UMEM_DIR"]=d;[os.environ.__setitem__(f"AMNI_UMEM_{k}",MODES[mode]) for k in ("LT","GN","G2")];U._EID.clear();U._rcache.clear()
def _seed(u):
    h=[]
    for s in SEED:U.turn(s,h,user=u);h.append((s,"ok"))
    return h
vals=lambda u:sorted(f.val.decode() for f in U.load(U.user_key(u)))
def score(mode,d):
    _env(mode,d);bad=[];base=None
    for i,(t,gone) in enumerate(FORGET):
        u=f"f{i}";h=_seed(u);base=base or vals(u);r=U.turn(t,h,user=u);got=vals(u);want=sorted(v for v in base if v not in gone)
        if got!=want or not r["reply"] or "forgotten" not in r["reply"]:bad.append(("forget",t,r["reply"],sorted(set(base)-set(got))))
    for i,(t,ok5) in enumerate(NEAR):
        u=f"n{i}";h=_seed(u);r=U.turn(t,h,user=u);got=U.load(U.user_key(u));new=[(f.rel,f.val.decode()) for f in got if f.val.decode() not in base]
        if r["reply"] or sorted(f.val.decode() for f in got if f.val.decode() in base)!=base or any(x[0]==5 for x in new) and not ok5:bad.append(("near",t,r["reply"],new))
    return bad
@pytest.mark.parametrize("mode",sorted(MODES))
def test_forget_phrasings_and_near_misses(mode,tmp_path,monkeypatch):
    [monkeypatch.setenv(k,os.environ.get(k,"")) for k in ("AMNI_UMEM_DIR","AMNI_UMEM_LT","AMNI_UMEM_GN","AMNI_UMEM_G2")];bad=score(mode,str(tmp_path));assert bad==[],bad
@pytest.mark.parametrize("mode",sorted(MODES))
def test_forget_all_and_history(mode,tmp_path,monkeypatch):
    [monkeypatch.setenv(k,os.environ.get(k,"")) for k in ("AMNI_UMEM_DIR","AMNI_UMEM_LT","AMNI_UMEM_GN","AMNI_UMEM_G2")];_env(mode,str(tmp_path))
    for i,t in enumerate(("Forget everything.","Delete everything you know about me.","Please erase everything I told you.")):
        u=f"a{i}";h=_seed(u);r=U.turn(t,h,user=u);assert vals(u)==[] and "everything" in r["reply"],(t,r)
    h=[(s,"ok") for s in SEED]+[("Forget what I told you about my brother.","Okay.")];p,mf,hf,hents,drop=U.parse("Where does my brother live?",h)
    assert not {b"Oslo",b"nurse",b"Volvo"}&{f[2] for f in hf} and b"Rome" in {f[2] for f in hf} and not any(b"orget" in f[2] for f in hf)
@pytest.mark.parametrize("mode",sorted(MODES))
def test_pet_never_occupation(mode,tmp_path,monkeypatch):
    [monkeypatch.setenv(k,os.environ.get(k,"")) for k in ("AMNI_UMEM_DIR","AMNI_UMEM_LT","AMNI_UMEM_GN","AMNI_UMEM_G2")];_env(mode,str(tmp_path))
    for t in ("I just got a kitten!","I have a kitten.","My brother got a puppy.","We just got a bunny!"):assert not [f for f in U._P().turn(t.encode()) if f[1]==5],t
    assert [(f[1],f[2]) for f in U._P().turn(b"I am a nurse.")]==[(5,b"nurse")] or mode=="prod"
