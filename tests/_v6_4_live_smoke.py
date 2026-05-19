"""Live probe v6.4: amni CLI launched server + ReAct goal + PII publish dry-run."""
import os,sys,json,time,urllib.request,urllib.error
BASE=os.environ.get('AMNI_BASE_URL','http://127.0.0.1:8002')
def post(p,b,t=180):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:return e.code,json.loads(e.read().decode())
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=20) as resp:return json.loads(resp.read().decode())
print('=== v6.4 LIVE SMOKE (amni-CLI-launched server) ===',flush=True)
boot=get('/stats')
print(f'\nlessons={boot["lessons_n"]} skills={boot["skills"]}',flush=True)
print(f'goal skill present: {"goal" in boot["skills"]}',flush=True)
print('\n--- ReAct goal: "tell me the time" ---',flush=True)
t0=time.time()
code,r=post('/skills/goal',{'args':{'goal':'tell me the current time','max_steps':3}},t=300)
wall=time.time()-t0
print(f'  HTTP {code} wall={wall:.1f}s',flush=True)
out=r.get('output',{})
print(f'  final={out.get("final")!r}',flush=True)
print(f'  stop_reason={out.get("stop_reason")} n_steps={out.get("n_steps")}',flush=True)
for s in (out.get('steps') or [])[:3]:print(f'    step: {json.dumps(s,default=str)[:200]}',flush=True)
print('\n--- ReAct goal: multi-step "search memory for SkillRegistry then tell me what it is" ---',flush=True)
t0=time.time()
code,r=post('/skills/goal',{'args':{'goal':'search adam memory for the term SkillRegistry then summarize what it is','max_steps':3}},t=180)
wall=time.time()-t0
print(f'  HTTP {code} wall={wall:.1f}s',flush=True)
out=r.get('output',{})
print(f'  final={(out.get("final") or "(none)")[:200]!r}',flush=True)
print(f'  stop_reason={out.get("stop_reason")} n_steps={out.get("n_steps")}',flush=True)
for s in (out.get('steps') or [])[:5]:
    if 'plan_raw' in s:print(f'    plan_raw={s["plan_raw"][:200]!r}',flush=True)
    else:print(f'    step: {s.get("tool")} -> {str(s.get("result"))[:140]}',flush=True)
print('\n=== amni CLI: invoke `amni stats` via subprocess ===',flush=True)
import subprocess
r=subprocess.run([sys.executable,'-m','amni.cli','--help'],capture_output=True,text=True,timeout=30)
print(f'  amni --help: stdout first line: {r.stdout.splitlines()[0]!r}',flush=True)
print(f'  exit code: {r.returncode}',flush=True)
print('\n=== final stats ===',flush=True)
s=get('/stats')
print(f'lessons={s["lessons_n"]} sessions={s["sessions_n"]}',flush=True)
