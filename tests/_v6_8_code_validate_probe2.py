"""Fresh probe with tightened code scaffold + larger token budget."""
import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=400):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except Exception as e:return None,{'error':str(e)}
def ask(msg,t=400):
    t0=time.time()
    code,r=post('/chat',{'message':msg},t=t)
    wall=time.time()-t0
    ans=(r.get("answer") or "").encode("ascii","replace").decode("ascii")
    print('='*80,flush=True)
    print(f'Q: {msg}',flush=True)
    print(f'[wall={wall:.1f}s tier={r.get("tier","?")}]',flush=True)
    print(f'A:\n{ans[:2500]}',flush=True);print('',flush=True)
ask('Write a Python function that returns the nth Fibonacci number, both recursive with memoization and iterative versions')
ask('Write a Python class for an LRU cache with get/put in O(1)')
