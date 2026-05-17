"""Probe new propose-critique-refine scaffold. Watch for FIRST SHOT + CRITIQUE + REFINE structure."""
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
    print(f'A:\n{ans[:1500]}',flush=True)
    print('',flush=True)
print('Probing propose-critique-refine CoT scaffold...\n',flush=True)
ask('How do I deduplicate users by email when emails may have different cases and trailing spaces?')
ask('Why might my Python script use 4GB of RAM for a 100MB CSV file?')
