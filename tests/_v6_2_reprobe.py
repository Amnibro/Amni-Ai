"""Re-probe of v6.2.0 fixes. Targets the bugs found in the v6.1 capability probe."""
import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=180):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:return e.code,json.loads(e.read().decode())
    except Exception as e:return None,{'error':str(e)}
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=20) as resp:return json.loads(resp.read().decode())
def ask(msg,sid=None,t=120):
    t0=time.time()
    code,r=post('/chat',{'message':msg,'session_id':sid} if sid else {'message':msg},t=t)
    wall=time.time()-t0
    ans=(r.get('answer') or '')[:300]
    print(f'  [{wall:5.2f}s][{r.get("tier","?"):<25}] Q: {msg[:60]!r}',flush=True)
    print(f'                                            A: {ans}',flush=True)
    return r
print('=== v6.2.0 RE-PROBE — targeted at v6.1 bugs ===',flush=True)
boot=get('/stats');print(f'Boot: lessons={boot["lessons_n"]} sessions={boot["sessions_n"]}\n',flush=True)
print('--- BUG 1 FIX: calc skill should be FAST now (Python eval, not Adam) ---',flush=True)
ask('What is 2 + 2?')
ask('What is 17 * 23?')
ask('calculate 144 / 12')
ask('What is seven times eight?')
ask('What is 99 + 1?')
print('\n--- BUG 2 FIX: multi-turn no longer poisoned by prior math ---',flush=True)
sid=ask('My name is the maintainer and I love Final Fantasy 10.')['session_id']
ask('What is my name?',sid=sid)
ask('What game did I say I love?',sid=sid)
ask('What is 9 + 9?',sid=sid)
ask('What was my favorite game again?',sid=sid)
print('\n--- BUG 3 FIX: post-scan recall should find real content ---',flush=True)
print('  [first scan amni/serve so we have known content]',flush=True)
code,r=post('/skills/scan',{'args':{'path':'C:/Users/antho/Documents/ai/Amni-Ai/amni/serve','glob':'*.py','max_files':10}},t=180)
print(f'  scan -> ok={r.get("ok")} added={r.get("output",{}).get("lessons_added")}',flush=True)
print('  [now query mem skill with high-recall flat-cosine fallback]',flush=True)
code,r=post('/skills/mem',{'args':{'query':'What is SkillRegistry?'}},t=30)
print(f'  mem result:\n  {json.dumps(r,indent=2,default=str)[:1200]}',flush=True)
print('\n--- BUG 4 FIX: self-introspection actually describes Adam ---',flush=True)
ask('What can you do?')
ask('Who are you?')
ask('list your skills')
print('\n--- NEW: knowledge browser endpoint ---',flush=True)
r=get('/lessons?limit=3')
print(f'  /lessons?limit=3 -> total={r["total"]} items_n={len(r["items"])}',flush=True)
for it in r['items']:print(f'    [#{it["idx"]}] {it["q"][:80]!r} -> {it["a"][:60]!r}',flush=True)
r=get('/lessons?q=PROBE&limit=5')
print(f'  /lessons?q=PROBE -> {r["total"]} matches',flush=True)
for it in r['items']:print(f'    [#{it["idx"]}] {it["q"][:80]!r} -> {it["a"][:60]!r}',flush=True)
print('\n--- FINAL STATS ---',flush=True)
s=get('/stats')
print(f'lessons={s["lessons_n"]} (delta={s["lessons_n"]-boot["lessons_n"]}) sessions={s["sessions_n"]}',flush=True)
nz={k:v for k,v in s["tier_counts"].items() if v>0}
print(f'tier hits this run: {nz}',flush=True)
