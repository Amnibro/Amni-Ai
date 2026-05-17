"""Force CoT code mode + auto-run by asking for novel script."""
import json,time,urllib.request
BASE='http://127.0.0.1:8002'
def post(p,b,t=500):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except Exception as e:return None,{'error':str(e)}
t0=time.time()
code,r=post('/chat',{'message':'Write a Python script that generates 20 random integers between 1 and 100, finds their median, and prints both the list and the median'},t=500)
wall=time.time()-t0
ans=(r.get('answer','') or '').encode('ascii','replace').decode('ascii')
print(f'[wall={wall:.1f}s tier={r.get("tier")} skills={[c.get("skill") for c in r.get("skill_calls",[])]}]',flush=True)
print(ans[:3000],flush=True)
