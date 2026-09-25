import re,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import adam_nightly as N
from amni.compute import eval_lock as EL
DECL=re.compile(r"not sure|don'?t know")
M={"chat3_strict":0.90,"D_known":0.75,"D_unknown":0.84,"nq_acc":0.10,"nq_decline":0.50,"bpb_clean1":1.81,"bpb_clean2":1.92,"garbles":2,"smoke_hits":0.8}
REP={"chat3":{"strict":0.9},"D":{"strict":0.78,"strict_by_kind":{"known":0.75,"unknown":0.84}},"bpb":{"bpb_clean1":1.81,"bpb_clean2":1.92},"nq":{"acc":0.1,"em":0.05,"decline":0.5},"smoke":{"garbles":2,"hit_rate":0.8},"wall_s":5}
def setup(tmp_path):
 s=tmp_path/"split.jsonl";s.write_text('{"q":"a"}\n');return s,EL.lock([s]),tmp_path/"nightly_history.json"
def test_alerts_thresholds():
 assert N.alerts(M,None)==[] and N.alerts(M,M)==[]
 cur={**M,"chat3_strict":0.879,"D_known":0.73,"bpb_clean1":1.831,"bpb_clean2":1.94,"garbles":3,"nq_decline":0.9,"smoke_hits":0.8}
 got={a["metric"]:a for a in N.alerts(cur,M)}
 assert set(got)=={"chat3_strict","bpb_clean1","garbles"} and got["chat3_strict"]["prev"]==0.9 and got["bpb_clean1"]["delta"]==0.021
 assert N.alerts({**M,"nq_acc":0.0},M)[0]["metric"]=="nq_acc" and N.alerts({**M,"garbles":0,"bpb_clean1":1.5},M)==[] and N.alerts({"new_metric":0.1},M)==[]
def test_flat_carries_d2_groups():
 m=N.flat({**REP,"D2":{"strict":0.3,"strict_groups":{"known":0.31,"unknown":0.7,"twohop":0.0}}});assert m["D2_strict"]==0.3 and m["D2_known"]==0.31 and m["D2_unknown"]==0.7 and m["D2_twohop"]==0.0 and "D2_strict" not in N.flat(REP)
 assert [a["metric"] for a in N.alerts({**m,"D2_known":0.25},m)]==["D2_known"] and any(str(x).endswith("d2/d2.jsonl") for x in N.SPLITS)
def test_prod_appends_history_and_alerts_vs_previous_night(tmp_path):
 s,man,h=setup(tmp_path);r,p=N.finalize(REP,man,tmp_path,h,"2026-09-25")
 H=json.loads(h.read_text());assert r["kind"]=="prod" and p.name=="report_nightly_2026-09-25.json" and len(H["runs"])==1 and H["runs"][0]["m"]["chat3_strict"]==0.9 and r["alerts"]==[]
 bad={**REP,"chat3":{"strict":0.85}};r2,_=N.finalize(bad,man,tmp_path,h,"2026-09-26");H=json.loads(h.read_text())
 assert r2["kind"]=="prod" and [a["metric"] for a in r2["alerts"]]==["chat3_strict"] and r2["prev_date"]=="2026-09-25" and [x["date"] for x in H["runs"]]==["2026-09-25","2026-09-26"]
 N.finalize(bad,man,tmp_path,h,"2026-09-26");assert [x["date"] for x in json.loads(h.read_text())["runs"]]==["2026-09-25","2026-09-26"]
def test_dry_on_split_change_no_history(tmp_path):
 s,man,h=setup(tmp_path);s.write_text('{"q":"changed"}\n');r,p=N.finalize(REP,man,tmp_path,h,"2026-09-25")
 assert r["kind"]=="dry" and p.name=="report_nightly_2026-09-25_dry.json" and not h.exists() and r["eval_lock"]["splits_changed"]
 assert any(a["metric"]=="split_changed" and a["files"]==[EL.rel(s)] for a in r["alerts"]) and str(p) not in EL.prod_reports(str(tmp_path/"report_*.json"))
def test_forced_dry_and_blockers_never_touch_history(tmp_path):
 s,man,h=setup(tmp_path);N.finalize(REP,man,tmp_path,h,"2026-09-24");before=h.read_text()
 r,_=N.finalize(REP,man,tmp_path,h,"2026-09-25",forced=True);r2,_=N.finalize(REP,man,tmp_path,h,"2026-09-25",blockers=["phase chat failed"])
 assert r["kind"]=="dry" and r2["kind"]=="dry" and {"metric":"blocker","why":"phase chat failed"} in r2["alerts"] and h.read_text()==before
def test_prev_is_last_prod_before_date(tmp_path):
 h=tmp_path/"h.json";N.save_hist(h,{"runs":[{"date":"2026-09-23","m":{"chat3_strict":0.95}},{"date":"2026-09-24","kind":"prod","m":{"chat3_strict":0.91}}]});s,man,_=setup(tmp_path)
 r,_=N.finalize(REP,man,tmp_path,h,"2026-09-25");assert r["prev_date"]=="2026-09-24" and r["alerts"]==[]
def test_qa_scoring_and_garbles():
 assert N.score_qa("Freddie Highmore. (Source: Wikipedia — The Good Doctor.)",["Freddie Highmore"],DECL)=={"decline":False,"em":True,"hit":True}
 assert N.score_qa("Jenny Slate. Jenny Slate as Harley Quinn, The Joker's girlfriend. (Source: Wikipedia — The Lego Batman Movie. I'll remember this.)",["Jenny Slate"],DECL)["em"] and not N.score_qa("Ted Danson played him. Ted Danson.",["Ted Danson"],DECL)["em"]
 assert N.score_qa("I'm not sure about that one yet.",["Ted Danson"],DECL)["decline"] and not N.score_qa("1931",["19"],DECL)["hit"] and not N.score_qa("I",["I Love Lucy"],DECL)["hit"]
 lex={"he's","a","pilot","instead","of","you","told","me","lives","in","the","if","i","remember","right"}
 assert N.garbles("He's a pilot, if I remember right.",["pilot"],lex)==[] and N.garbles("San Diego.",["San Diego"],lex)==[]
 assert [g["why"] for g in N.garbles("He's a pilot, instead of.",["pilot"],lex)]==["dangling:of"] and N.garbles("a pilotxq",["pilot"],lex)[0]["why"]=="glued" and N.garbles("a pilot zorblat",["pilot"],lex)[0]["why"]=="nonword:zorblat"
def test_child_env_is_local_cpu_and_clean(monkeypatch,tmp_path):
 monkeypatch.setenv("AMNI_LEARN_WEB","1");monkeypatch.setenv("AMNI_RAY_PACK","/x.npy");monkeypatch.setenv("SSH_AUTH_SOCK","keep");e=N.child_env(str(tmp_path))
 assert e["AMNI_LEARN_WEB"]=="0" and "AMNI_RAY_PACK" not in e and e["AMNI_RAY_ARCH"]=="cpu" and e["RAYON_NUM_THREADS"]=="6" and e["SSH_AUTH_SOCK"]=="keep" and e["AMNI_RAY_LEARNED"].startswith(str(tmp_path))
def test_netguard_blocks_remote(monkeypatch):
 import socket
 monkeypatch.setattr(socket.socket,"connect",socket.socket.connect);monkeypatch.setattr(socket,"getaddrinfo",socket.getaddrinfo);bl=N.netguard()
 try:socket.getaddrinfo("en.wikipedia.org",443);ok=False
 except ConnectionRefusedError:ok=True
 assert ok and bl==["en.wikipedia.org"]
