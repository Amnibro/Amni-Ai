"""Truly novel query that must trigger Gemma generation."""
import time,json,urllib.request
BASE='http://127.0.0.1:8002'
def stream(msg,t=400):
    req=urllib.request.Request(f'{BASE}/chat/stream',data=json.dumps({'message':msg}).encode(),headers={'Content-Type':'application/json'},method='POST')
    t0=time.time();first=None;n=0;all_chunks=[]
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
                        try:all_chunks.append(json.loads(edata))
                        except:pass
                        if n in (5,15,30,60,100,200):print(f'    [+{elapsed:.1f}s] tokens={n}',flush=True)
                    elif etype=='done':
                        wall=time.time()-t0;rate=n/(wall-(first or 0))*60 if (wall-(first or 0))>0 else 0
                        print(f'  DONE at {wall:.2f}s tokens={n} chunks/min={rate:.0f}: {edata[:200]}',flush=True);return all_chunks
                    elif etype=='meta':print(f'  META at {time.time()-t0:.2f}s: {edata[:160]}',flush=True)
                    elif etype=='error':print(f'  ERROR: {edata}',flush=True)
    return all_chunks
print('=== Streaming a query that MUST use Gemma generation ===',flush=True)
chunks=stream('I want to write a small script that monitors a folder of CSV files and emails me when any column average drifts more than 3 sigma from baseline. Outline how I would build this in Python.')
print(f'\n--- joined output ({sum(len(c) for c in chunks)} chars total) ---',flush=True)
out=''.join(chunks)[:800].encode('ascii','replace').decode('ascii')
print(out,flush=True)
