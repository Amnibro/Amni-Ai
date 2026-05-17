"""Probe iter8 additions."""
import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=8):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except Exception as e:return None,{'error':str(e)}
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=20) as resp:return json.loads(resp.read().decode())
def ask(msg,t=8):
    t0=time.time()
    code,r=post('/chat',{'message':msg},t=t)
    wall=time.time()-t0
    print(f'[{wall:5.2f}s {r.get("tier","?"):<27}] {msg!r}',flush=True)
print('='*80,flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}\n',flush=True)
print('--- LEETCODE MEDIUM ---',flush=True)
for q in ['What is the rotate image problem?','What is the merge intervals problem?','What is the house robber problem?','What is the LRU cache design?','What is the daily temperatures problem?']:ask(q)
print('--- DOMAINS ---',flush=True)
for q in ['What is a stock exchange?','What is a derivative in finance?','What is Bitcoin?','What is Ethereum?','What is a smart contract?','What is HIPAA?','What is FHIR?','What is procedural generation?']:ask(q)
print('--- LESS COMMON LANGS ---',flush=True)
for q in ['What is Ruby?','What is Rails?','What is Erlang?','What is the let it crash philosophy?','What is Lua?','What is set -euo pipefail?']:ask(q)
print('--- PRACTICAL RECIPES ---',flush=True)
for q in ['How do I deploy a Python web app?','How do I add HTTPS?','How do I implement authentication?','How do I avoid [REDACTED]?','How do I roll back a bad deploy?']:ask(q)
print('\n=== STATS ===',flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}',flush=True)
