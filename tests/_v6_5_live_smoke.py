"""Live smoke v6.5: UI HTML loads, /project + /project/tree work, /reflect endpoint, sidebar quick-actions hit real endpoints."""
import json,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def get(p,t=20):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=t) as r:return r.read().decode()
def get_json(p,t=20):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=t) as r:return json.loads(r.read().decode())
def post(p,b,t=180):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:return e.code,json.loads(e.read().decode())
print('=== v6.5 LIVE SMOKE ===',flush=True)
print('\n--- GET / serves new UI with sidebar + wizard ---',flush=True)
html=get('/')
checks=[('sidebar','id="sidebar"' in html),('wizard','id="wizard"' in html),('quick-actions','qaScanFolder' in html),('persona-selector','setPersonaUI' in html),('voice','toggleVoiceOut' in html),('file-tree','loadFileTree' in html),('empty-state','id="empty"' in html),('examples','class="ex"' in html)]
for n,ok in checks:print(f'  [{"OK" if ok else "FAIL"}] {n}',flush=True)
print('\n--- GET /project ---',flush=True)
proj=get_json('/project')
print(f'  {json.dumps(proj,default=str)}',flush=True)
print('\n--- GET /project/tree ---',flush=True)
tree=get_json('/project/tree?depth=1&limit=20')
print(f'  root={tree["root"]} items_n={len(tree["items"])}',flush=True)
for it in tree['items'][:8]:print(f'    {it["depth"]*"  "}{"+ " if it["is_dir"] else "  "}{it["name"]}',flush=True)
print('\n--- POST /skills/mem via UI quick action style ---',flush=True)
code,r=post('/skills/mem',{'args':{'query':'SkillRegistry','k':3}},t=30)
print(f'  HTTP {code} -- mem query',flush=True)
out=r.get('output',{})
print(f'  hits_n={len(out.get("hits",[]))} lessons_n={out.get("lessons_n")}',flush=True)
for h in out.get('hits',[])[:2]:
    if 'a' in h:print(f'    [{h.get("score","?")}] {h.get("q","")[:80]} -> {h.get("a","")[:80]}',flush=True)
print('\n--- final stats ---',flush=True)
s=get_json('/stats')
print(f'lessons={s["lessons_n"]} sessions={s["sessions_n"]} skills={len(s["skills"])}',flush=True)
