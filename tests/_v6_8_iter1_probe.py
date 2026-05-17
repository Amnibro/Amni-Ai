"""Probe iter1 additions: Rust, concurrency, advanced algos, python deep dive."""
import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=120):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:return e.code,json.loads(e.read().decode())
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=20) as resp:return json.loads(resp.read().decode())
def ask(msg,t=60):
    t0=time.time()
    code,r=post('/chat',{'message':msg},t=t)
    wall=time.time()-t0
    ans=(r.get("answer") or "")[:240].encode("ascii","replace").decode("ascii")
    print(f'[{wall:5.2f}s tier={r.get("tier","?"):<27}] {msg!r}',flush=True)
    print(f'  -> {ans}\n',flush=True)
print('='*80,flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}',flush=True)
print('--- RUST ---',flush=True)
ask('What is ownership in Rust?')
ask('What is the Result type in Rust?')
ask('What does the ? operator do in Rust?')
ask('What is the difference between Box, Rc, and Arc?')
ask('How do I share state between threads in Rust?')
print('\n--- CONCURRENCY ---',flush=True)
ask('What is the difference between concurrency and parallelism?')
ask('What is a deadlock?')
ask('What is a channel for inter-thread communication?')
ask('What is backpressure?')
print('\n--- ADVANCED ALGORITHMS ---',flush=True)
ask('What is topological sort?')
ask('What is Union-Find data structure?')
ask('What is the longest common subsequence problem?')
ask('How do I find the kth largest element in an array?')
ask('How does Dijkstra handle a re-update of a node already popped?')
print('\n--- PYTHON DEEP DIVE ---',flush=True)
ask('What is asyncio.gather vs asyncio.wait_for?')
ask('What is dataclasses.dataclass and when use it?')
ask('What is functools.lru_cache?')
ask('What is the walrus operator?')
ask('Why is mutable default argument a bug?')
print('\n=== STATS ===',flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]} sessions={s["sessions_n"]}',flush=True)
