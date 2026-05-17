import json,os,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(path,body):
    r=urllib.request.Request(f'{BASE}{path}',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(r,timeout=300) as resp:return json.loads(resp.read().decode())
def get(path):
    with urllib.request.urlopen(f'{BASE}{path}',timeout=30) as resp:return json.loads(resp.read().decode())
def delete(path):
    r=urllib.request.Request(f'{BASE}{path}',method='DELETE')
    with urllib.request.urlopen(r,timeout=30) as resp:return json.loads(resp.read().decode())
print('=== MULTI-TURN CONVERSATION ===',flush=True)
r1=post('/chat',{'message':'Hello, my name is the maintainer.'})
sid=r1['session_id']
print(f'Turn 1 [sid={sid}]: tier={r1["tier"]} answer={r1["answer"][:80]!r}',flush=True)
r2=post('/chat',{'message':'What was my name again?','session_id':sid})
print(f'Turn 2 [sid={sid}]: tier={r2["tier"]} answer={r2["answer"][:80]!r}',flush=True)
r3=post('/chat',{'message':'Now what is 7 times 8?','session_id':sid})
print(f'Turn 3 [sid={sid}]: tier={r3["tier"]} answer={r3["answer"][:80]!r}',flush=True)
sessions=get('/sessions')['sessions']
my_session=[s for s in sessions if s['session_id']==sid]
print(f'Session persisted: {len(my_session)>0} (size_bytes={my_session[0]["size"] if my_session else "n/a"})',flush=True)
print('\n=== DIRECT SKILL CALLS VIA HTTP ===',flush=True)
sep='/'
testfile=os.path.abspath(os.path.join(os.getcwd(),'tests','v6_smoke_artifact.txt')).replace(os.sep,sep)
print(f'target file: {testfile}',flush=True)
r=post('/skills/file_write',{'args':{'path':testfile,'content':'v6.0.0 lives!'}})
print(f'file_write: ok={r["ok"]} bytes={r["output"].get("bytes_written") if r["ok"] else r}',flush=True)
r=post('/skills/file_read',{'args':{'path':testfile}})
print(f'file_read:  ok={r["ok"]} content={r["output"].get("content")!r}' if r['ok'] else f'file_read failed: {r}',flush=True)
print('\n=== ASIMOV GATE FROM HTTP LAYER ===',flush=True)
try:r=post('/skills/file_read',{'args':{'path':'C:/Windows/win.ini'}})
except urllib.error.HTTPError as e:print(f'file outside workdir gated (HTTP {e.code}): {e.read().decode()[:160]}',flush=True)
try:r=post('/skills/shell',{'args':{'cmd':'rm -rf /'}})
except urllib.error.HTTPError as e:print(f'shell rm -rf gated (HTTP {e.code}): {e.read().decode()[:160]}',flush=True)
try:
    r=post('/skills/shell',{'args':{'cmd':'git status'}})
    print(f'shell git status: ok={r["ok"]} stdout_first_line={r["output"]["stdout"].splitlines()[0] if r["output"].get("stdout") else "(empty)"}',flush=True)
except urllib.error.HTTPError as e:print(f'shell git unexpectedly gated: {e.read().decode()[:200]}',flush=True)
print('\n=== TEACH + LOOKUP ROUNDTRIP ===',flush=True)
r=post('/teach',{'question':'What is the secret v6 test phrase?','answer':'amnibro-ships-deployables'})
print(f'teach: lessons_n={r["lessons_n"]}',flush=True)
r=post('/chat',{'message':'What is the secret v6 test phrase?'})
print(f'lookup after teach: tier={r["tier"]} answer={r["answer"][:80]!r}',flush=True)
print('\n=== CLEANUP ===',flush=True)
print('delete artifact session:',delete(f'/sessions/{sid}'),flush=True)
try:os.unlink(testfile);print(f'removed {testfile}',flush=True)
except Exception as e:print(f'cleanup: {e}',flush=True)
print('\n=== FINAL STATS ===',flush=True)
stats=get('/stats')
print(f'lessons_n={stats["lessons_n"]} sessions_n={stats["sessions_n"]} svc_boot_s={stats.get("svc_boot_s")}',flush=True)
nz={k:v for k,v in stats['tier_counts'].items() if v>0}
print(f'tier hits: {nz}',flush=True)
