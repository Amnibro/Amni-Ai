import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=120):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:return e.code,json.loads(e.read().decode())
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=20) as resp:return json.loads(resp.read().decode())
print('=== boot state ===',flush=True)
s=get('/stats')
print(f'lessons_n={s["lessons_n"]} (loaded from disk after prior scan) skills={s["skills"]}',flush=True)
print('\n=== unrestricted file_read ===',flush=True)
code,r=post('/skills/file_read',{'args':{'path':'C:/Windows/System32/drivers/etc/hosts'}},t=20)
print(f'HTTP {code} ok={r.get("ok")} bytes={r.get("output",{}).get("bytes")}',flush=True)
print('\n=== chat-driven natural-language scan (now bulk-fit at end) ===',flush=True)
t0=time.time()
code,r=post('/chat',{'message':'scan the directory C:/Users/antho/Documents/ai/Amni-Ai/amni/serve'},t=180)
wall=time.time()-t0
print(f'wall={wall:.1f}s tier={r.get("tier")} answer={r.get("answer","")[:200]!r}',flush=True)
print(f'skill_calls={[c["skill"] for c in (r.get("skill_calls") or [])]}',flush=True)
sc=r.get('skill_calls',[{}])[0].get('result',{}).get('output',{}) if r.get('skill_calls') else {}
print(f'  files_scanned={sc.get("files_scanned")} lessons_added={sc.get("lessons_added")} bulk_fit={sc.get("bulk_fit")}',flush=True)
print('\n=== mem skill query against ingested content ===',flush=True)
code,r=post('/skills/mem',{'args':{'query':'What is SkillRegistry?'}},t=30)
print(f'mem result: {json.dumps(r,indent=2,default=str)[:600]}',flush=True)
print('\n=== final stats ===',flush=True)
s=get('/stats')
print(f'lessons_n={s["lessons_n"]} sessions_n={s["sessions_n"]}',flush=True)
nz={k:v for k,v in s["tier_counts"].items() if v>0}
print(f'tier hits since boot: {nz}',flush=True)
