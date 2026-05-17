"""Probe iter4 additions."""
import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=20):
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
    ans=(r.get("answer") or "")[:150].encode("ascii","replace").decode("ascii")
    print(f'[{wall:5.2f}s {r.get("tier","?"):<27}] {msg!r}',flush=True)
    print(f'  -> {ans}\n',flush=True)
print('='*80,flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}\n',flush=True)
print('--- NETWORKING ---',flush=True)
for q in ['What is the TCP three-way handshake?','What is the OSI 7-layer model?','What are common HTTP status codes?','What is QUIC and why does it matter?','What is WebRTC?']:ask(q)
print('--- GAME DEV ---',flush=True)
for q in ['What is a game loop?','What is delta time?','What is the difference between Unity and Unreal?','What is collision detection vs collision response?','What is a shader?']:ask(q)
print('--- EMBEDDED ---',flush=True)
for q in ['What is the difference between malloc and calloc?','What is the stack vs the heap?','What is the difference between a microcontroller and a microprocessor?','What is an RTOS?','What is volatile in C?']:ask(q)
print('--- ADVANCED MATH ---',flush=True)
for q in ['How do I check if a point is inside a polygon?','How do I compute GCD efficiently?','What is the sieve of Eratosthenes?','What is a dot product?','What is the chain rule?']:ask(q)
print('--- FACTS EXT ---',flush=True)
for q in ['What is the central dogma of molecular biology?','What is CRISPR?','What is pH?','What are the four fundamental forces?','What is Moore\'s Law?']:ask(q)
print('\n=== STATS ===',flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}',flush=True)
