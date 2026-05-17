"""Probe iter9 — previously-missed short queries + cloud + ML deep."""
import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=8):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except Exception as e:return None,{'error':str(e)}
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=20) as resp:return json.loads(resp.read().decode())
def ask(msg,t=8):
    t0=time.time()
    code,r=post('/chat',{'message':msg},t=t)
    wall=time.time()-t0
    print(f'[{wall:5.2f}s {r.get("tier","?"):<27}] {msg!r}',flush=True)
print('='*80,flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}\n',flush=True)
print('--- PREVIOUSLY-MISSED (now should hit) ---',flush=True)
for q in ['What is Bitcoin?','What is HIPAA?','What is FHIR?','What is Erlang?','What is Lua?','What is HashMap?','What is LRU cache?','What is Postgres?','What is MongoDB?','What is Redis?','What is Kafka?','What is React?','What is Next.js?','What is Tailwind?','What is npm?']:ask(q)
print('--- CLOUD ---',flush=True)
for q in ['What is EC2?','What is S3?','What is Lambda?','What is Cloud Run?','What is Terraform?','What is GitOps?','What is FinOps?']:ask(q)
print('--- ML DEEP ---',flush=True)
for q in ['What is RoPE?','What is LoRA?','What is QLoRA?','What is flash attention?','What is RLHF?','What is DPO?','What is in-context learning?','What is the KV cache in LLM?']:ask(q)
print('--- ALGO CLASSICS ---',flush=True)
for q in ['What is heapsort?','What is QuickSelect?','What is counting sort?','What is Aho-Corasick?','What is segment tree?','What is reservoir sampling?']:ask(q)
print('\n=== STATS ===',flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}',flush=True)
