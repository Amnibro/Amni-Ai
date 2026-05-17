"""Probe iter6 additions: FP, distributed DBs, scientific, war stories, short paraphrases."""
import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=10):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except Exception as e:return None,{'error':str(e)}
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=20) as resp:return json.loads(resp.read().decode())
def ask(msg,t=10):
    t0=time.time()
    code,r=post('/chat',{'message':msg},t=t)
    wall=time.time()-t0
    ans=(r.get("answer") or "")[:100].encode("ascii","replace").decode("ascii")
    print(f'[{wall:5.2f}s {r.get("tier","?"):<27}] {msg[:48]!r:<55} -> {ans}',flush=True)
print('='*100,flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}\n',flush=True)
print('--- FUNCTIONAL ---',flush=True)
for q in ['What is a pure function?','What is currying?','What is a monad?','What is the Maybe monad?']:ask(q)
print('--- DISTRIBUTED DBs ---',flush=True)
for q in ['What is Cassandra?','What is Google Spanner?','What is CockroachDB?','What is Redis?','When NOT to use a distributed DB?']:ask(q)
print('--- SCIENTIFIC ---',flush=True)
for q in ['What is the central limit theorem?','What is Monte Carlo simulation?','What is PCA?','What is the FFT?']:ask(q)
print('--- WAR STORIES ---',flush=True)
for q in ['What is a thundering herd problem?','What is cache stampede?','What is Conway\'s Law?','What is the second-system effect?','What is the Knight Capital incident?']:ask(q)
print('--- SHORT PARAPHRASES ---',flush=True)
for q in ['What is RAG?','What is chunking in RAG?','What is a closure?','What is async/await?','What is Big O notation?']:ask(q)
print('\n=== STATS ===',flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}',flush=True)
