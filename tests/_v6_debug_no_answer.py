"""Probe the live server for the exact response shape on the failing inputs."""
import json,urllib.request
BASE='http://127.0.0.1:8002'
def post(path,body,timeout=300):
    r=urllib.request.Request(f'{BASE}{path}',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(r,timeout=timeout) as resp:return resp.status,json.loads(resp.read().decode())
for msg in ['howdy!',"what's 14!",'hello','tell me a joke']:
    print(f'\n=== {msg!r} ===',flush=True)
    code,j=post('/chat',{'message':msg})
    print(f'  HTTP {code}',flush=True)
    print(f'  {json.dumps(j,indent=2,default=str)[:1500]}',flush=True)
