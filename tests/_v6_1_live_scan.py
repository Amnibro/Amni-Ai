import os,json,urllib.request,urllib.error
from pathlib import Path
BASE=os.environ.get('AMNI_BASE_URL','http://127.0.0.1:8002')
_REPO=Path(__file__).resolve().parents[1]
_SCAN_DIR=os.environ.get('AMNI_SCAN_DIR',str(_REPO/'amni'/'serve'))
_READ_FILE=os.environ.get('AMNI_READ_FILE',str(_REPO/'changelog.md'))
def post(p,b,t=300):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:return e.code,json.loads(e.read().decode())
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=20) as resp:return json.loads(resp.read().decode())
print('=== chat-driven natural-language scan ===',flush=True)
code,r=post('/chat',{'message':f'scan the directory {_SCAN_DIR}'},t=240)
print(f'tier={r.get("tier")} answer={r.get("answer","")[:200]!r}',flush=True)
print(f'skill_calls={[c["skill"] for c in (r.get("skill_calls") or [])]}',flush=True)
print('\n=== mem skill query against ingested content ===',flush=True)
code,r=post('/skills/mem',{'args':{'query':'What is SkillRegistry?'}},t=60)
print(f'mem result: {json.dumps(r,indent=2,default=str)[:500]}',flush=True)
print('\n=== unrestricted file_read on a non-workdir file ===',flush=True)
code,r=post('/skills/file_read',{'args':{'path':_READ_FILE,'max_bytes':800}},t=20)
print(f'HTTP {code} ok={r.get("ok")} bytes={r.get("output",{}).get("bytes")} preview={r.get("output",{}).get("content","")[:120]!r}',flush=True)
print('\n=== final stats ===',flush=True)
s=get('/stats')
print(f'lessons_n={s["lessons_n"]} sessions_n={s["sessions_n"]}',flush=True)
nz={k:v for k,v in s["tier_counts"].items() if v>0}
print(f'tier hits: {nz}',flush=True)
