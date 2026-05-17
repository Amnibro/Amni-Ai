"""Probe streaming endpoint — verify SSE events arrive progressively, not in one batch."""
import time,json,urllib.request
BASE='http://127.0.0.1:8002'
def stream(msg,t=300):
    req=urllib.request.Request(f'{BASE}/chat/stream',data=json.dumps({'message':msg}).encode(),headers={'Content-Type':'application/json'},method='POST')
    t0=time.time();first=None;tokens=[]
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
                        elapsed=time.time()-t0
                        if first is None:first=elapsed;print(f'  FIRST TOKEN at {first:.2f}s',flush=True)
                        try:tokens.append(json.loads(edata))
                        except:pass
                    elif etype=='done':print(f'  DONE at {time.time()-t0:.2f}s: {edata[:200]}',flush=True);return tokens
                    elif etype=='meta':print(f'  META at {time.time()-t0:.2f}s: {edata[:160]}',flush=True)
                    elif etype=='error':print(f'  ERROR: {edata}',flush=True)
    return tokens
print('='*60,flush=True)
print('Streaming "Write a Python function to reverse a string"',flush=True)
print('='*60,flush=True)
toks=stream('Write a Python function to reverse a string')
print(f'\nTOTAL TOKEN-EVENTS: {len(toks)}',flush=True)
print(f'JOINED OUTPUT (first 400 chars): {"".join(toks)[:400].encode("ascii","replace").decode("ascii")}',flush=True)
