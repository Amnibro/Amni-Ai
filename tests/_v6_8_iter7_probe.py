"""Probe iter7 additions — focus on short-form retrieval that previously missed."""
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
    tier=r.get("tier","?")
    print(f'[{wall:5.2f}s {tier:<27}] {msg!r}',flush=True)
print('='*80,flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}\n',flush=True)
print('--- SHORT-FORM (previously missed) ---',flush=True)
for q in ['What is PCA?','What is FFT?','What is Cassandra?','What is async/await?','What is cache stampede?','What is REST?','What is GraphQL?','What is gRPC?','What is OAuth?','What is JWT?','What is HTTPS?','What is Docker?','What is Kubernetes?','What is BFS?','What is DFS?','What is the GIL?','What is Big O notation?','What is recursion?','What is OOP?','What is functional programming?']:ask(q)
print('--- C++ / JAVA / KOTLIN ---',flush=True)
for q in ['What is RAII in C++?','What is std::unique_ptr?','What is move semantics?','What is the JVM?','What is HashMap?','What is Spring Boot?','What is Kotlin?','What is a data class in Kotlin?']:ask(q)
print('\n=== STATS ===',flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}',flush=True)
