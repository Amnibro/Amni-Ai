"""Probe iter3 additions."""
import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=30):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except Exception as e:return None,{'error':str(e)}
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=20) as resp:return json.loads(resp.read().decode())
def ask(msg,t=15):
    t0=time.time()
    code,r=post('/chat',{'message':msg},t=t)
    wall=time.time()-t0
    ans=(r.get("answer") or "")[:180].encode("ascii","replace").decode("ascii")
    print(f'[{wall:5.2f}s {r.get("tier","?"):<27}] {msg!r}',flush=True)
    print(f'  -> {ans}\n',flush=True)
print('='*80,flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}\n',flush=True)
print('--- ML FRAMEWORKS ---',flush=True)
for q in ['What is autograd in PyTorch?','What is the training loop in PyTorch?','What is mixed precision training (AMP)?','What is HuggingFace transformers?','What is PEFT?','What is jax.jit?']:ask(q)
print('--- SECURITY DEEP ---',flush=True)
for q in ['What is the difference between symmetric and asymmetric encryption?','What is OAuth 2.0?','What is a JWT?','What is SSRF and how to prevent?','What is constant-time comparison?']:ask(q)
print('--- DISTRIBUTED SYSTEMS ---',flush=True)
for q in ['What is Raft consensus?','What is the consensus problem?','What is a CRDT?','What is eventual consistency?','What is a correlation ID?']:ask(q)
print('--- PERFORMANCE ---',flush=True)
for q in ['What is the first rule of performance optimization?','What is a flame graph?','What is vectorization?','What is cache locality?']:ask(q)
print('--- ARCHITECTURE ---',flush=True)
for q in ['What is hexagonal architecture?','What is CQRS?','What is event sourcing?','What is the strangler fig pattern?','What is a bounded context?']:ask(q)
print('\n=== STATS ===',flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}',flush=True)
