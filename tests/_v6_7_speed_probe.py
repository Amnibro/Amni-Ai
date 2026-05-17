"""Speed probe — verify GPU encoder + 810-lesson bank. Haiku should hit creative corpus instantly."""
import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=180):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:return e.code,json.loads(e.read().decode())
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=20) as resp:return json.loads(resp.read().decode())
def ask(msg,sid=None,t=180):
    t0=time.time()
    code,r=post('/chat',{'message':msg,'session_id':sid} if sid else {'message':msg},t=t)
    wall=time.time()-t0
    ans=(r.get("answer") or "")[:280].encode("ascii","replace").decode("ascii")
    print(f'[{wall:5.2f}s tier={r.get("tier","?"):<27}] {msg!r}',flush=True)
    print(f'  -> {ans}\n',flush=True)
    return r
print('='*80,flush=True)
print('=== SPEED PROBE: GPU ENCODER + 810-LESSON BANK ===',flush=True)
print('='*80,flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]} skills={len(s["skills"])}\n',flush=True)
print('--- HAIKU (should hit creative corpus instantly) ---',flush=True)
ask('Write a haiku about AI')
ask('Write a haiku about code')
ask('Write a haiku about the ocean')
ask('Write a haiku about a cat')
print('\n--- WORLD FACTS (should hit facts corpus) ---',flush=True)
ask('What is the largest country by area?')
ask('How many continents are there?')
ask('How fast is the speed of light?')
ask('What is absolute zero?')
print('\n--- JS / DEVOPS / SQL (new corpus) ---',flush=True)
ask('What is the difference between let, const, and var?')
ask('What is the difference between WHERE and HAVING in SQL?')
ask('What is the difference between CMD and ENTRYPOINT?')
ask('What does git bisect do?')
ask('What are React hooks?')
print('\n--- ADVANCED COT (statistical / systems) ---',flush=True)
ask('What is overfitting?')
ask('What is Bayes theorem in plain English?')
ask('How would you design a URL shortener?')
print('\n=== STATS ===',flush=True)
s=get('/stats')
print(f'lessons={s["lessons_n"]} sessions={s["sessions_n"]}',flush=True)
