import os,pytest,torch
from amni.serve import blender_atex as B,adam_t_client as C
def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("AMNI_BLENDER",raising=False);assert not B.active()
def test_flag_on(monkeypatch):
    monkeypatch.setenv("AMNI_BLENDER","atex:/x");assert B.active()
def test_msgs_single_and_history():
    p=C.template("What is 2+2?",[("hi","hello")],0,(),["Calculator: The answer is 4."]," ")
    M=B.msgs(p);assert M[-1]=={"role":"user","content":"Calculator: The answer is 4. What is 2+2?"} and M[0]=={"role":"user","content":"hi"} and M[1]["content"]=="hello"
def test_notes_router():
    assert B.NOTE.match("Marlowe Bridge: It is a bridge. What is it?") and B.NOTE.match("Calculator: The answer is 4. What?") and not B.NOTE.match("Hi again, Adam, I finished painting.") and not B.NOTE.match("What is 2+2?")
def test_system_prompts_frozen():
    assert set(B.SYSM)>={"alone","split4"} and isinstance(B.SYSM["split4"],tuple) and "not sure" in B.SYSM["split4"][0]
@pytest.mark.parametrize("bits",[4,8])
def test_kv_quant_roundtrip(bits):
    L=B.kvlayer(bits)();torch.manual_seed(0);k=torch.randn(1,2,5,64);v=torch.randn(1,2,5,64);ko,vo=L.update(k,v)
    assert L.get_seq_length()==5 and (ko-k).norm()/k.norm()<(0.15 if bits==4 else 0.02)
    ko2,_=L.update(k[:,:,:1],v[:,:,:1]);assert ko2.shape[-2]==6
