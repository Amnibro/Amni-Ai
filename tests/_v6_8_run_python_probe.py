"""Probe code execution sandbox + auto-run."""
import json,time,urllib.request
BASE='http://127.0.0.1:8002'
def post(p,b,t=400):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except Exception as e:return None,{'error':str(e)}
print('=== direct run_python skill ===',flush=True)
code,r=post('/skills/run_python',{'args':{'code':'for i in range(5): print(f\"item {i}: {i**2}\")'}},t=20)
print(f'  HTTP {code}',flush=True)
print(f'  {json.dumps(r,default=str)[:600]}',flush=True)
print()
print('=== chat: write Python that auto-runs ===',flush=True)
t0=time.time()
code,r=post('/chat',{'message':'Write a Python script that prints the first 10 Fibonacci numbers'},t=400)
wall=time.time()-t0
ans=(r.get('answer','') or '').encode('ascii','replace').decode('ascii')
print(f'[wall={wall:.1f}s tier={r.get("tier")}]',flush=True)
print(ans[:2500],flush=True)
