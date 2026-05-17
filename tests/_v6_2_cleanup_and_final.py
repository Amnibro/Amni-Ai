"""Clean poisoned '18' lessons via DELETE /lessons + final probe of fixed multi-turn + flat-cosine."""
import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=180):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:return e.code,json.loads(e.read().decode())
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=20) as resp:return json.loads(resp.read().decode())
def delete(p):
    r=urllib.request.Request(f'{BASE}{p}',method='DELETE')
    try:
        with urllib.request.urlopen(r,timeout=30) as resp:return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:return {'error':e.read().decode()}
def ask(msg,sid=None,t=120):
    t0=time.time()
    code,r=post('/chat',{'message':msg,'session_id':sid} if sid else {'message':msg},t=t)
    wall=time.time()-t0
    print(f'  [{wall:5.2f}s][{r.get("tier","?"):<25}] Q:{msg[:50]!r:<55} A:{(r.get("answer") or "")[:200]!r}',flush=True)
    return r
print('=== STEP 1: find and delete poisoned "18" lessons ===',flush=True)
r=get('/lessons?q=18&limit=20')
print(f'  searching for "18" in lessons -> {r["total"]} matches',flush=True)
bad=[]
for it in r['items']:
    if it['a'].strip()=='18' and 'name' in it['q'].lower() or 'game' in it['q'].lower() or 'favorite' in it['q'].lower() or 'love' in it['q'].lower():
        print(f'    POISONED [#{it["idx"]}] Q:{it["q"][:80]!r} A:{it["a"]!r}',flush=True)
        bad.append(it['idx'])
print(f'\n  Deleting {len(bad)} poisoned lessons (in reverse idx order to keep indices stable)...',flush=True)
for idx in sorted(bad,reverse=True):
    r=delete(f'/lessons/{idx}')
    print(f'    deleted #{idx}: {r}',flush=True)
print('\n=== STEP 2: re-probe multi-turn — should be CLEAN now ===',flush=True)
sid=ask('My name is the maintainer and I love Final Fantasy 10.')['session_id']
ask('What is my name?',sid=sid)
ask('What game did I say I love?',sid=sid)
ask('What is 9 + 9?',sid=sid)
ask('What was my favorite game again?',sid=sid)
print('\n=== STEP 3: mem flat-cosine fallback ===',flush=True)
code,r=post('/skills/mem',{'args':{'query':'What is SkillRegistry?','k':3}},t=30)
print(json.dumps(r,indent=2,default=str)[:1500],flush=True)
print('\n=== STEP 4: chat-based mem use ===',flush=True)
ask('search memory for SkillRegistry')
print('\n=== FINAL STATS ===',flush=True)
s=get('/stats')
print(f'lessons={s["lessons_n"]} sessions={s["sessions_n"]}',flush=True)
