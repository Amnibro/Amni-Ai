"""Fresh queries to test full propose-critique-refine output with bumped token budget."""
import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=300):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except Exception as e:return None,{'error':str(e)}
def ask(msg,t=300):
    t0=time.time()
    code,r=post('/chat',{'message':msg},t=t)
    wall=time.time()-t0
    ans=(r.get("answer") or "").encode("ascii","replace").decode("ascii")
    print(f'='*80,flush=True)
    print(f'Q: {msg}',flush=True)
    print(f'[wall={wall:.1f}s tier={r.get("tier","?")} cat={r.get("category")}]',flush=True)
    print(f'A:\n{ans[:2500]}',flush=True)
    print('',flush=True)
print('Fresh CoT probe — propose/critique/refine with bigger token budget...\n',flush=True)
ask('Design a rate limiter that handles 50k requests per second and supports per-user and per-IP limits')
ask('How would you debug a websocket connection that drops every 90 seconds in production but works in dev')
