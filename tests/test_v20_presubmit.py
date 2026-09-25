import importlib.util,json,os
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];_s=importlib.util.spec_from_file_location("v20_presubmit",ROOT/"scripts"/"v20_presubmit.py");V=importlib.util.module_from_spec(_s);_s.loader.exec_module(V)
CTO=[ROOT/"amni"/"compute"/"decontam.py",ROOT/"amni"/"compute"/"eval_lock.py",ROOT/"amni"/"serve"/"self_debug.py",ROOT/"scripts"/"x0_code_eval.py"]
def mk(tmp,name,src):p=tmp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(src);return p.resolve()
def T(*ps):return {p:V.text(p) for p in ps}
def sevs(fs,rule=None):return [f["sev"] for f in fs if rule is None or rule in f["rule"]]
def test_box_paths_fail_env_default_home_ok_docstring_warns(tmp_path):
 p=mk(tmp_path,"a.py",'def f():\n    """Never binds /home/box."""\n    return "/workspace/m09/x"\nH=os.environ.get("H","/home/anmire/ai")\nG="/home/anmire/ai/x"\nW="C:\\\\Users\\\\antho\\\\x"\n');fs=V.c_paths(T(p))
 assert sorted((f["line"],f["sev"]) for f in fs)==[(2,"WARN"),(3,"FAIL"),(5,"FAIL"),(6,"FAIL")]
def test_rust_env_default_workspace_still_fails(tmp_path):
 p=mk(tmp_path,"e.rs",'fn r()->String{std::env::var("AMNI_ROOT").unwrap_or_else(|_|"/workspace/m09/amni_root".into())}\n');assert sevs(V.c_paths(T(p)))==["FAIL"]
def test_tainted_refs_fail_and_guard_context_warns(tmp_path):
 p=mk(tmp_path,"t.py",'A="data/big2/lex.json"\nB=load("data/chat4e/train.jsonl")\nC="grokipedia"\nD=[x for x in S if x!="yahoo_answers"] if refuse else S\nE={"chat3":1,"big20":2,"chat5":3}\n');fs=V.c_taint(T(p))
 assert [(f["line"],f["sev"]) for f in fs]==[(1,"FAIL"),(2,"FAIL"),(3,"FAIL"),(4,"WARN")]
@pytest.mark.parametrize("src,want",[
 ('import subprocess\nsubprocess.run(["python3","-c",code])\n',"FAIL"),
 ('import subprocess,sys\nsubprocess.run([sys.executable,"-c",code],timeout=5)\n',"FAIL"),
 ('import subprocess\nsubprocess.run(cmd,shell=True)\n',"FAIL"),
 ('import os\nos.system("ls "+d)\n',"FAIL"),
 ('x=eval(s)\n',"FAIL"),
 ('import subprocess\nsubprocess.run(["timeout","5","python3","-c",c])\n',"FAIL"),
 ('import subprocess\nsubprocess.run(["python3"],input=code)\n',"FAIL"),
 ('import subprocess\nsubprocess.run(["bash","-c","echo hi"])\n',"FAIL"),
 ('from subprocess import run\nrun(["node","-e",js])\n',"FAIL"),
 ('import subprocess\nBWRAP="/usr/bin/bwrap"\ncmd=[BWRAP,"--unshare-net",*argv]\nsubprocess.run(cmd)\n',None),
 ('import subprocess,sys\nsubprocess.run([sys.executable,str(ROOT/"scripts"/"x.py"),"--p","1"])\n',None),
 ('import subprocess\nsubprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"])\n',None),
 ('import subprocess\nsubprocess.run(_bwrap_cmd(wd,[py,"main.py"]),input=s)\n',None),
 ('x=eval("1+1")\nm.eval()\n',None),
 ('import subprocess\nsubprocess.run([str(binp)])\n',"WARN"),
])
def test_python_exec_classification(tmp_path,src,want):
 p=mk(tmp_path,"x.py",src);fs=V.py_exec(p,src);assert V.worst(fs)==(want or "PASS"),fs
@pytest.mark.parametrize("src,want",[
 ('let c=Command::new("python3").arg("-c").arg(py).spawn();',"FAIL"),
 ('let o=Command::new("timeout")\n  .args(["5","python3","-c",code]).output();',"FAIL"),
 ('let o=Command::new("sh").arg("-c").arg(s).output();',"FAIL"),
 ('let o=Command::new("bwrap").args(["--unshare-all","python3","-c",c]).output();',None),
 ('let o=Command::new("git").arg("status").output();',None),
 ('let o=Command::new(&prog).output();',"WARN"),
])
def test_rust_exec_classification(tmp_path,src,want):
 p=mk(tmp_path,"x.rs",src);assert V.worst(V.rs_exec(p,src))==(want or "PASS")
def test_private_decontam_list_fails_shared_import_passes(tmp_path):
 bad=mk(tmp_path,"ing.py",'from pathlib import Path\ndef _eval_files():return list((ROOT/"data"/"concepts"/"eval").rglob("*.json"))\ndef grams(t):return {t[i:i+13] for i in range(len(t))}\n')
 ok=mk(tmp_path,"ing2.py",'from amni.compute import decontam\ndef _eval_files():return list((ROOT/"data"/"concepts"/"eval").rglob("*.json"))\ndef grams(t):return decontam.dirty(t)\n')
 plain=mk(tmp_path,"runner.py",'import json\nrows=[json.loads(l) for l in open("data/webeval/nq.jsonl")]\n')
 assert sevs(V.c_decontam_static(T(bad)))==["FAIL"] and V.c_decontam_static(T(ok,plain))==[]
def test_gate_tuning_warns_only(tmp_path):
 p=mk(tmp_path,"hc.py",'import json\nQ=[json.loads(l) for l in open("data/concepts/eval/related.jsonl")]\ndef tune(th):return sum(score(q,th) for q in Q)\nbest=max(G,key=tune)\n');fs=V.c_tuning(T(p))
 assert sevs(fs)==["WARN"] and V.c_tuning(T(mk(tmp_path,"dev.py",'import json\nQ=json.load(open("data/x_dev.json"))\ndef tune(t):return t\n')))==[]
def test_tautological_tests_detected(tmp_path):
 t=mk(tmp_path,"test_t.py",'from pathlib import Path\nSRC=Path(__file__).read_text()\ndef test_a():\n assert True\n assert 1==1\ndef test_b():\n assert "def f" in SRC\n');r=mk(tmp_path,"test_r.py",'from m import f\ndef test_a():\n assert f(2)==4\n')
 assert V.taut_py(t.read_text())[:2]==(3,3) and V.taut_py(r.read_text())[:2]==(1,0)
 assert V.taut_rs('#[test] fn a(){assert!(true);assert_eq!(1,1);}')==(2,2) and V.taut_rs('#[test] fn a(){assert_eq!(f(2),4);}')==(1,0)
def test_servers_cors_and_unauthenticated_routes(tmp_path):
 a=mk(tmp_path,"main.rs",'let app=Router::new().route("/chat",post(chat)).layer(CorsLayer::permissive());')
 b=mk(tmp_path,"srv.py",'from fastapi import FastAPI\napp=FastAPI()\napp.add_middleware(CORSMiddleware,allow_origins=["*"])\n@app.post("/run")\ndef run(x):return x\n')
 c=mk(tmp_path,"ok.rs",'let app=Router::new().route("/chat",post(chat)).layer(from_fn(require_token)).layer(CorsLayer::new().allow_origin(origin));')
 fs=V.c_servers(T(a,b,c));assert {(Path(f["file"]).name,f["rule"]) for f in fs}=={("main.rs","open CORS"),("main.rs","HTTP routes without an auth check"),("srv.py","open CORS"),("srv.py","HTTP routes without an auth check")}
def test_cto_modules_pass_every_static_check():
 t=T(*CTO);fs=V.c_paths(t)+V.c_taint(t)+V.c_exec(t)+V.c_decontam_static(t)+V.c_servers(t)+V.c_tuning(t);assert "FAIL" not in sevs(fs),fs
def test_claims_missing_artifact_box_path_and_perfect_score(tmp_path):
 c=mk(tmp_path,"c.json",json.dumps([{"metric":"precision","value":1.0,"command":"/workspace/x/run.sh","artifact":"data/nope/m.json"},{"metric":"f1"}]));fs,_=V.c_claims(str(c),[],False,5e-4,60)
 rules={f["rule"] for f in fs};assert {"claim precision: box path","claim precision: artifact not on Antman","claim precision: command not runnable here","claim precision: perfect score","claim f1 missing keys"}<=rules
def test_trivial_claim_command_fails(tmp_path):
 c=mk(tmp_path,"c.json",json.dumps([{"metric":"acc","value":0.5,"command":"true","artifact":"docs/tasks/v20/README.md"}]));fs,_=V.c_claims(str(c),[],False,5e-4,60);assert "claim acc: trivial command regenerates nothing" in {f["rule"] for f in fs}
def test_no_claims_is_warn():assert sevs(V.c_claims(None,[],False,5e-4,60)[0])==["WARN"]
@pytest.mark.skipif(not (os.path.exists("/usr/bin/bwrap") and os.path.exists("/usr/bin/prlimit")),reason="needs the bwrap jail")
def test_claim_rerun_in_jail_compares_values(tmp_path):
 art="docs/tasks/v20/README.md";c=mk(tmp_path,"c.json",json.dumps([{"metric":"acc","value":0.5,"command":"/bin/sh -c 'echo acc=0.5'","artifact":art},{"metric":"acc2","value":0.7,"command":"/bin/sh -c 'echo acc2=0.5'","artifact":art}]))
 fs,notes=V.c_claims(str(c),[],True,5e-4,30);assert [n["rerun_value"] for n in notes]==[0.5,0.5] and all(n["jailed"] for n in notes)
 assert [f["rule"] for f in fs if f["sev"]=="FAIL"]==["claim acc2: rerun differs"]
def test_main_writes_report_and_exit_code(tmp_path):
 bad=mk(tmp_path,"m.py",'import subprocess\nsubprocess.run(["python3","-c",src])\n');od=tmp_path/"out"
 assert V.main(["--mission","MX1","--files",str(bad),"--out-dir",str(od),"--no-status"])==1
 r=json.loads((od/"MX1_presubmit.json").read_text());assert r["verdict"]=="FAIL" and r["checks"]["3_exec_safety"]["status"]=="FAIL" and len(r["code_hash"])==64 and r["git"]["head"] and (od/"MX1_presubmit.md").exists()
 assert V.main(["--mission","MX2","--files",*map(str,CTO),"--out-dir",str(od),"--no-status"])==0
def test_missing_declared_file_fails(tmp_path):assert V.main(["--mission","MX3","--files",str(tmp_path/"nope.py"),"--out-dir",str(tmp_path),"--no-status"])==1
