"""Probe CoT scaffold — should appear on novel reasoning/code queries, skip creative/greeting/factual-cached."""
import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=180):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except Exception as e:return None,{'error':str(e)}
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=20) as resp:return json.loads(resp.read().decode())
def ask(msg,t=180):
    t0=time.time()
    code,r=post('/chat',{'message':msg},t=t)
    wall=time.time()-t0
    ans=(r.get("answer") or "")[:500].encode("ascii","replace").decode("ascii")
    print(f'[{wall:5.2f}s tier={r.get("tier","?"):<32} cat={r.get("category","?")}]',flush=True)
    print(f'Q: {msg!r}',flush=True)
    print(f'A: {ans}',flush=True)
    print('---',flush=True)
print('='*80,flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}\n',flush=True)
print('=== NOVEL REASONING (should show CoT) ===',flush=True)
ask('Design a system to detect duplicate transactions across two databases in real time')
ask('Why might my Postgres query be slow even though I added an index')
print('\n=== CREATIVE (should skip CoT) ===',flush=True)
ask('Write a haiku about rain')
print('\n=== GREETING (skip) ===',flush=True)
ask('Howdy')
print('\n=== FACTUAL CACHED (corpus hit, no CoT needed) ===',flush=True)
ask('What is binary search?')
