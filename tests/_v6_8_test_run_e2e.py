"""Iter17 e2e: probe /chat/stream with code query, verify test_run SSE event fires + tests_ok suffix."""
import json,time,urllib.request
BASE='http://127.0.0.1:8001'
def stream(msg,t=400):
    req=urllib.request.Request(f'{BASE}/chat/stream',data=json.dumps({'message':msg}).encode(),headers={'Content-Type':'application/json'},method='POST')
    t0=time.time();chunks=[];events={}
    with urllib.request.urlopen(req,timeout=t) as r:
        buf=b''
        for chunk in r:
            buf+=chunk
            if b'\n\n' in buf:
                evs=buf.split(b'\n\n');buf=evs.pop()
                for evt in evs:
                    if not evt.strip():continue
                    lines=evt.decode('utf-8',errors='replace').split('\n')
                    et='msg';ed=''
                    for ln in lines:
                        if ln.startswith('event: '):et=ln[7:]
                        elif ln.startswith('data: '):ed+=ln[6:]
                    events[et]=events.get(et,0)+1
                    if et=='token':
                        try:chunks.append(json.loads(ed))
                        except:pass
                    elif et=='test_run':print(f'  TEST_RUN at {time.time()-t0:.2f}s: {ed}',flush=True)
                    elif et=='exec':print(f'  EXEC at {time.time()-t0:.2f}s: {ed[:200]}',flush=True)
                    elif et=='perturb':print(f'  PERTURB at {time.time()-t0:.2f}s: {ed[:200]}',flush=True)
                    elif et=='done':print(f'  DONE at {time.time()-t0:.2f}s: {ed}',flush=True);return chunks,events
                    elif et=='meta':print(f'  META at {time.time()-t0:.2f}s: {ed[:160]}',flush=True)
                    elif et=='error':print(f'  ERROR: {ed}',flush=True)
    return chunks,events
print('=== code query that should produce TESTS section ===',flush=True)
chunks,events=stream('Write a python function is_prime(n) that returns True for prime numbers, then test it on print(is_prime(7)).')
print(f'\nevents={events}',flush=True)
print('\n--- output ---',flush=True)
out=''.join(chunks).encode('ascii','replace').decode('ascii')
print(out[:2000],flush=True)
