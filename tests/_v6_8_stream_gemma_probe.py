"""Probe TRUE Gemma streaming — query not in cache."""
import time,json,urllib.request
BASE='http://127.0.0.1:8002'
def stream(msg,t=300):
    req=urllib.request.Request(f'{BASE}/chat/stream',data=json.dumps({'message':msg}).encode(),headers={'Content-Type':'application/json'},method='POST')
    t0=time.time();first=None;n=0;last_progress=0
    with urllib.request.urlopen(req,timeout=t) as r:
        buf=b''
        for chunk in r:
            buf+=chunk
            if b'\n\n' in buf:
                events=buf.split(b'\n\n');buf=events.pop()
                for evt in events:
                    if not evt.strip():continue
                    lines=evt.decode('utf-8',errors='replace').split('\n')
                    etype='msg';edata=''
                    for ln in lines:
                        if ln.startswith('event: '):etype=ln[7:]
                        elif ln.startswith('data: '):edata+=ln[6:]
                    if etype=='token':
                        elapsed=time.time()-t0;n+=1
                        if first is None:first=elapsed;print(f'  FIRST TOKEN at {first:.2f}s',flush=True)
                        if elapsed-last_progress>5:print(f'    [+{elapsed:.1f}s] tokens={n}',flush=True);last_progress=elapsed
                    elif etype=='done':print(f'  DONE at {time.time()-t0:.2f}s tokens={n}: {edata[:200]}',flush=True);return
                    elif etype=='meta':print(f'  META at {time.time()-t0:.2f}s: {edata[:160]}',flush=True)
                    elif etype=='error':print(f'  ERROR: {edata}',flush=True)
print('=== Gemma streaming probe (novel query) ===',flush=True)
stream('Why does ice float on water but most other solids sink in their liquid form?')
