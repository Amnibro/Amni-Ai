"""Iter14: probe /chat/stream with code query, verify validate + exec SSE events fire."""
import time,json,urllib.request
BASE='http://127.0.0.1:8001'
def stream(msg,t=400):
    req=urllib.request.Request(f'{BASE}/chat/stream',data=json.dumps({'message':msg}).encode(),headers={'Content-Type':'application/json'},method='POST')
    t0=time.time();first=None;n=0;all_chunks=[];events_seen={'meta':0,'token':0,'validate':0,'exec':0,'done':0,'error':0}
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
                    events_seen[etype]=events_seen.get(etype,0)+1
                    if etype=='token':
                        elapsed=time.time()-t0;n+=1
                        if first is None:first=elapsed;print(f'  FIRST TOKEN at {first:.2f}s',flush=True)
                        try:all_chunks.append(json.loads(edata))
                        except:pass
                        if n in (5,30,100,200):print(f'    [+{elapsed:.1f}s] tokens={n}',flush=True)
                    elif etype=='done':
                        wall=time.time()-t0
                        print(f'  DONE at {wall:.2f}s tokens={n}: {edata[:200]}',flush=True);return all_chunks,events_seen
                    elif etype=='meta':print(f'  META at {time.time()-t0:.2f}s: {edata[:200]}',flush=True)
                    elif etype=='validate':print(f'  VALIDATE at {time.time()-t0:.2f}s: {edata}',flush=True)
                    elif etype=='exec':print(f'  EXEC at {time.time()-t0:.2f}s: {edata[:300]}',flush=True)
                    elif etype=='error':print(f'  ERROR: {edata}',flush=True)
    return all_chunks,events_seen
print('=== Code query — should emit validate + exec ===',flush=True)
chunks,seen=stream('Write a python function that returns the Nth Fibonacci number using memoization, then print fib(10).')
print(f'\n--- output ({sum(len(c) for c in chunks)} chars) ---',flush=True)
out=''.join(chunks)[:1200].encode('ascii','replace').decode('ascii')
print(out,flush=True)
print(f'\n--- events seen ---\n{seen}',flush=True)
assert seen['exec']>=1 or seen['validate']>=1,'expected at least one validate/exec event'
print('PASS',flush=True)
