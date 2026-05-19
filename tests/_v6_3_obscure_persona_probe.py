"""Probe the persona web-learn pipeline against several genuinely obscure historical figures.
Goal: verify Adam can scrape DDG -> Wikipedia -> distill into usable persona without manual description."""
import os,json,time,urllib.request,urllib.error
from pathlib import Path
BASE=os.environ.get('AMNI_BASE_URL','http://127.0.0.1:8002')
_REPO=Path(__file__).resolve().parents[1]
def post(p,b,t=180):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:return e.code,json.loads(e.read().decode())
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=30) as resp:return json.loads(resp.read().decode())
def delete(p):
    r=urllib.request.Request(f'{BASE}{p}',method='DELETE')
    try:
        with urllib.request.urlopen(r,timeout=30) as resp:return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:return None
print('=== OBSCURE PERSONA WEB-LEARN PROBE ===',flush=True)
print('\nForgetting any cached fallback "Sherlock Holmes" so we relearn fresh:',flush=True)
import json as _json
pf=os.environ.get('AMNI_PERSONAS_PATH',str(_REPO/'experiences'/'personas.json'))
if os.path.exists(pf):
    d=_json.load(open(pf,encoding='utf-8'))
    before=len(d.get('personas',[]))
    d['personas']=[p for p in d.get('personas',[]) if p.get('source')=='user']
    open(pf,'w',encoding='utf-8').write(_json.dumps(d,indent=2))
    print(f'  cleared {before - len(d["personas"])} non-user personas from disk store',flush=True)
print('\n--- Web-learn 5 genuinely obscure historical figures ---',flush=True)
TARGETS=['Hildegard von Bingen','Murasaki Shikibu','Hypatia of Alexandria','Ibn Battuta','Rosalind Franklin']
for name in TARGETS:
    print(f'\n>>> Learning persona: {name!r}',flush=True)
    t0=time.time()
    code,r=post('/persona',{'name':name,'learn_via_web':True},t=300)
    wall=time.time()-t0
    if code!=200:
        print(f'  FAILED HTTP {code}: {r}',flush=True);continue
    p=r.get('persona',{})
    print(f'  [{wall:5.1f}s] source: {p.get("source")}',flush=True)
    print(f'  description: {p.get("description","")[:400]}',flush=True)
    print(f'  voice_hints: {p.get("voice_hints",[])}',flush=True)
print('\n--- Test one of them in actual chat ---',flush=True)
t0=time.time()
code,r=post('/persona',{'name':'Hildegard von Bingen'},t=30)
print(f'  set persona -> ok={code==200}',flush=True)
sid=f'obscure_test_{int(time.time())}'
post('/persona',{'name':'Hildegard von Bingen','session_id':sid},t=30)
for q in ['Hi!','What is the capital of France?','Tell me about music.','What is the meaning of life?']:
    code,r=post('/chat',{'message':q,'session_id':sid},t=180)
    print(f'  Q: {q!r}',flush=True)
    print(f'    A: {(r.get("answer") or "")[:300]}',flush=True)
    print(f'    [tier={r.get("tier")} persona={r.get("persona")}]',flush=True)
print('\n--- final stats ---',flush=True)
s=get('/stats')
print(f'lessons={s["lessons_n"]} sessions={s["sessions_n"]}',flush=True)
print(f'\nKnown personas now:',flush=True)
for p in get('/personas')['known']:
    print(f'  {p["name"]:<28} src={p["source"]}',flush=True)
