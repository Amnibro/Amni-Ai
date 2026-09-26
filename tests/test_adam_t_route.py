import os,sys,json,time,subprocess
import pytest
from amni.serve import adam_t_client as C
from amni.serve import gguf_runtime as G
STUB=r'''
import sys,json,base64,os,time
M=os.environ.get("FAKE_MODE","ok");D=os.environ["FAKE_DIR"];CTX=int(os.environ.get("FAKE_CTX","3328"));b=lambda s:base64.b64decode(s);e=lambda x:base64.b64encode(x).decode()
open(os.path.join(D,"starts"),"a").write(f"{os.getpid()} {os.environ.get('BACKEND','')}\n")
M=="banner" and print("loading model...",flush=True)
M=="die_at_start" and sys.exit(2)
for l in sys.stdin:
    q=json.loads(l);op=q.get("op")
    if op=="hello":
        M=="hang_hello" and time.sleep(60);r={"model":"adam-t-fake","step":7,"fmt":"chat","ctx":CTX,"ctx_symbols":1024,"vocab":4096,"proto":1}
    elif op=="gen":
        p=b(q["prompt"]).decode();st=[b(s).decode() for s in q["stop"]];open(os.path.join(D,"prompts.jsonl"),"a").write(json.dumps({"p":p,"max":q["max"],"stop":st,"pid":os.getpid()})+"\n")
        M=="slow" and time.sleep(60)
        (M=="crash" or M=="crash_once" and not os.path.exists(os.path.join(D,"crashed"))) and (open(os.path.join(D,"crashed"),"w").close(),sys.exit(3))
        t=(" Kittens are fun. If it stops breathing call 911." if M=="say911" else "" if M=="empty" else " ADAM-T heard: "+p.rsplit("User: ",1)[-1].split("\n")[0]+"\n\nUser: leaked turn")[:int(q["max"])]
        t=min([t]+[t[:t.index(s)] for s in st if s and s in t],key=len);r={"error":"boom"} if M=="error" else {"text":e(t.encode())}
    elif op=="lp":r={"lp":[-float(len(base64.b64decode(c))) for _,c in q["items"]]}
    elif op=="bye":r={"bye":True}
    else:r={"error":f"bad op {op}"}
    print(json.dumps(r),flush=True)
    if op=="bye":break
'''
EM="My mom and my dad are laying on the floor and do not answer me. I'm scared."
@pytest.fixture
def fake(tmp_path,monkeypatch):
    s=tmp_path/"fake_serve.py";s.write_text(STUB);b=tmp_path/"adam-t";b.write_text(f"#!/bin/sh\nshift\nexec {sys.executable} {s} \"$@\"\n");b.chmod(0o755)
    [monkeypatch.delenv(k,raising=False) for k in ("AMNI_ADAM_T_BACKEND","AMNI_ADAM_T_CTX","AMNI_ADAM_T_MAX_BYTES")]
    monkeypatch.setenv("FAKE_DIR",str(tmp_path));monkeypatch.setenv("AMNI_ADAM_T_BIN",str(b));monkeypatch.setenv("AMNI_ADAM_T_TIMEOUT","2");monkeypatch.setenv("AMNI_ADAM_T_LOAD_TIMEOUT","3");monkeypatch.setenv("AMNI_ADAM_T",str(tmp_path/"run"))
    yield tmp_path
    [c.close() for c in C._C.values()];C._C.clear()
prompts=lambda d:[json.loads(l) for l in open(d/"prompts.jsonl")] if (d/"prompts.jsonl").exists() else []
starts=lambda d:open(d/"starts").read().split("\n")[:-1] if (d/"starts").exists() else []
def test_proto_roundtrip(fake):
    p=C.Proto([os.environ["AMNI_ADAM_T_BIN"],"serve","x"],2,None,subprocess.DEVNULL)
    assert p.info["model"]=="adam-t-fake" and p.info["fmt"]=="chat" and p.info["step"]==7
    assert p.text("User: hi\nAdam:",200,["\n\n"])==" ADAM-T heard: hi" and p.lp([(b"a",b"xyz")]*70)==[-3.0]*70
    with pytest.raises(C.ProtoError):p.rpc(op="nope")
    assert p.alive();p.close();assert not p.alive()
def test_client_template_stops_and_cpu_env(fake):
    c=C.AdamT();assert c.gen("hello  there",120)=="ADAM-T heard: hello  there"
    r=prompts(fake)[-1];assert r["p"]=="User: hello  there\nAdam:" and r["max"]==120 and r["stop"]==list(C.STOPS)
    assert starts(fake)[0].endswith(" ndarray") and c.status()["alive"] and c.status()["step"]==7
    c.gen("and now?",120,history=[("hi","hey!\n\nwhat's up"),["my name is Ann","Nice to meet you, Ann."]])
    assert prompts(fake)[-1]["p"]=="User: hi\nAdam: hey!\nwhat's up\n\nUser: my name is Ann\nAdam: Nice to meet you, Ann.\n\nUser: and now?\nAdam:" and len(starts(fake))==1
def test_history_trimmed_to_budget(fake,monkeypatch):
    monkeypatch.setenv("FAKE_CTX","400");c=C.AdamT();H=[(f"question number {i} "+"x"*40,f"answer {i} "+"y"*40) for i in range(20)]
    c.gen("latest?",200,history=H);p=prompts(fake)[-1]["p"];assert len(p.encode())<=400-200 and p.endswith("User: latest?\nAdam:") and "question number 19" in p and "question number 0 " not in p
    assert C.template("q",H,0).count("User:")==21 and C.template("q"*500,H,100)=="User: "+"q"*500+"\nAdam:"
def test_timeout_kills_and_restarts(fake,monkeypatch):
    monkeypatch.setenv("FAKE_MODE","slow");c=C.AdamT();t=time.monotonic()
    with pytest.raises(C.ProtoError):c.gen("hi",64)
    assert time.monotonic()-t<5 and c.p is None and c.fails==1
    with pytest.raises(C.ProtoError,match="backoff"):c.gen("hi",64)
    monkeypatch.setenv("FAKE_MODE","ok");c.until=0;assert c.gen("again",64)=="ADAM-T heard: again" and c.fails==0 and len(starts(fake))==2
def test_crash_auto_restart(fake,monkeypatch):
    monkeypatch.setenv("FAKE_MODE","crash_once");c=C.AdamT()
    with pytest.raises(C.ProtoError,match="exited"):c.gen("hi",64)
    c.until=0;assert c.gen("hi",64)=="ADAM-T heard: hi" and len(starts(fake))==2
def test_load_failures(fake,monkeypatch):
    monkeypatch.setenv("FAKE_MODE","hang_hello");t=time.monotonic()
    with pytest.raises(C.ProtoError,match="timeout"):C.AdamT().gen("hi",64)
    assert time.monotonic()-t<6;monkeypatch.setenv("FAKE_MODE","die_at_start")
    with pytest.raises(C.ProtoError):C.AdamT().gen("hi",64)
    monkeypatch.setenv("FAKE_MODE","banner");assert C.AdamT().gen("hi",64)=="ADAM-T heard: hi"
def test_target_prefers_gf17(tmp_path):
    (tmp_path/"model.gf17").write_bytes(b"x");assert C.AdamT(str(tmp_path)).target()==str(tmp_path/"model.gf17") and C.AdamT(str(tmp_path/"r")).target()==str(tmp_path/"r")
@pytest.fixture
def rules(monkeypatch):
    monkeypatch.setenv("AMNI_GEN_BACKEND","ray");calls=[]
    monkeypatch.setattr(G,"_ray_stream",lambda m,*a,**k:(calls.append(m),iter(["RULE reply"]))[1]);return calls
def test_flag_off_unchanged(rules,monkeypatch,tmp_path):
    monkeypatch.delenv("AMNI_ADAM_T",raising=False);monkeypatch.setenv("AMNI_ADAM_T_BIN",str(tmp_path/"missing"))
    assert not C.on() and C.answer("hi") is None and G.chat("tell me about rivers")=={"answer":"RULE reply","tier":"tier_ray","tokens":10}
    assert "".join(G.chat_stream("tell me about rivers"))=="RULE reply" and rules==["tell me about rivers"]*2 and C._C=={}
def test_flag_on_routes_to_adam_t(fake,rules):
    r=G.chat("tell me about rivers",history=[("hi","hello!")]);assert r["tier"]=="tier_adam_t" and r["answer"]=="ADAM-T heard: tell me about rivers" and rules==[]
    assert "".join(G.chat_stream("what about lakes?"))=="ADAM-T heard: what about lakes?" and G.Svc().chat("and seas?")[0]=="ADAM-T heard: and seas?" and rules==[]
    assert prompts(fake)[0]["p"].startswith("User: hi\nAdam: hello!\n\n") and prompts(fake)[0]["max"]==400 and prompts(fake)[2]["max"]==320
@pytest.mark.parametrize("mode",["error","empty","crash","slow","die_at_start"])
def test_fallback_to_rules(fake,rules,monkeypatch,mode):
    monkeypatch.setenv("FAKE_MODE",mode);r=G.chat("tell me about rivers");assert r["tier"]=="tier_ray" and r["answer"]=="RULE reply" and rules==["tell me about rivers"]
    assert "".join(G.chat_stream("tell me about lakes"))=="RULE reply"
def test_emergency_first_and_guarded(fake,rules,monkeypatch):
    r=G.chat(EM);assert r["tier"]=="tier_emergency" and "911" in r["answer"] and "911" in "".join(G.chat_stream("my baby is choking"))
    assert prompts(fake)==[] and starts(fake)==[] and rules==[]
    monkeypatch.setenv("FAKE_MODE","say911");r=G.chat("I just got a kitten!");assert r["tier"]=="tier_adam_t" and r["answer"]=="Kittens are fun. " and "911" not in "".join(G.chat_stream("I just got a kitten!"))
def test_route_logged_only_to_debug(fake,rules,monkeypatch,caplog):
    caplog.set_level("DEBUG",logger="amni.serve.adam_t");G.chat("my secret plan");monkeypatch.setenv("FAKE_MODE","error");C.client().close();G.chat("another secret")
    recs=[r for r in caplog.records if r.name=="amni.serve.adam_t"];assert [r.levelname for r in recs]==["DEBUG","DEBUG"] and "route=adam_t" in recs[0].getMessage() and "route=fallback:ProtoError" in recs[1].getMessage() and not any("secret" in r.getMessage() for r in caplog.records)
def test_harness_uses_shared_client():
    src=open(os.path.join(os.path.dirname(C.__file__),"..","..","scripts","adam_t_eval.py")).read();assert "class Proto(ProtoClient)" in src and "Popen" not in src and "amni/serve/adam_t_client.py" in src
