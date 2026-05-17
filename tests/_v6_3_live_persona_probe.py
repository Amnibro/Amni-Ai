"""Live probe — does the persona system actually produce ORGANIC, varied responses?
Same question, three different personas: Rikku, Yoda, Scientist."""
import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=180):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:return e.code,json.loads(e.read().decode())
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=20) as resp:return json.loads(resp.read().decode())
print('=== v6.3.0 LIVE PERSONA PROBE ===',flush=True)
print('--- /personas (list known) ---',flush=True)
r=get('/personas')
print(f'  default: {r["default"]}',flush=True)
for p in r['known']:print(f'    {p["name"]:<14} (warmth={p["warmth"]} formality={p["formality"]} excitement={p["excitement"]} src={p["source"]})',flush=True)
def probe_with_persona(persona_name,questions):
    print(f'\n--- Persona: {persona_name.upper()} ---',flush=True)
    sid=f'probe_{persona_name}_{int(time.time())}'
    code,r=post('/persona',{'name':persona_name,'session_id':sid,'learn_via_web':False})
    print(f'  set persona -> ok={code==200} actual={r.get("persona",{}).get("name")}',flush=True)
    for q in questions:
        t0=time.time()
        code,r=post('/chat',{'message':q,'session_id':sid},t=180)
        wall=time.time()-t0
        print(f'  [{wall:5.1f}s][{r.get("category","?"):<10}][{r.get("tier","?"):<22}] Q:{q!r:<55}',flush=True)
        print(f'    A: {(r.get("answer") or "")[:240]}',flush=True)
qs=['Hi!','What is 2 + 2?','What is the capital of France?','Tell me about yourself.','Write a haiku about texture-native AI.']
probe_with_persona('rikku',qs)
probe_with_persona('yoda',qs)
probe_with_persona('scientist',qs)
print('\n--- Custom persona via web-learn (small risk: crawler may fail) ---',flush=True)
code,r=post('/persona',{'name':'Sherlock Holmes','learn_via_web':True},t=120)
print(f'  learn-result HTTP {code}',flush=True)
if code==200:
    p=r.get('persona',{})
    print(f'    name={p.get("name")} source={p.get("source")}',flush=True)
    print(f'    description: {p.get("description","")[:300]}',flush=True)
    print(f'    voice_hints: {p.get("voice_hints",[])}',flush=True)
print('\n--- MCP server smoke ---',flush=True)
code,r=post('/mcp',{'jsonrpc':'2.0','id':1,'method':'initialize','params':{}},t=20)
print(f'  initialize HTTP {code} -> protocol={r.get("result",{}).get("protocolVersion")} server={r.get("result",{}).get("serverInfo",{}).get("name")}',flush=True)
code,r=post('/mcp',{'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}},t=20)
tools=r.get('result',{}).get('tools',[])
print(f'  tools/list HTTP {code} -> {len(tools)} tools: {[t["name"] for t in tools[:6]]}...',flush=True)
code,r=post('/mcp',{'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'ask_adam','arguments':{'question':'What is 17 * 23?','persona':'yoda'}}},t=120)
print(f'  tools/call ask_adam HTTP {code}',flush=True)
print(f'    raw: {json.dumps(r.get("result",{}).get("_raw",{}),default=str)[:300]}',flush=True)
print('\n--- final stats ---',flush=True)
s=get('/stats')
print(f'lessons={s["lessons_n"]} sessions={s["sessions_n"]}',flush=True)
