"""Iter19 e2e: novel code query with new scaffold should yield diverse asserts."""
import json,time,urllib.request
BASE='http://127.0.0.1:8001'
def stream(msg,t=400):
    req=urllib.request.Request(f'{BASE}/chat/stream',data=json.dumps({'message':msg}).encode(),headers={'Content-Type':'application/json'},method='POST')
    t0=time.time();chunks=[];done=None;trun=None
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
                    if et=='token':
                        try:chunks.append(json.loads(ed))
                        except:pass
                    elif et=='test_run':trun=json.loads(ed);print(f'  TEST_RUN at {time.time()-t0:.2f}s: {ed}',flush=True)
                    elif et=='done':done=json.loads(ed);print(f'  DONE at {time.time()-t0:.2f}s: {ed}',flush=True);return chunks,done,trun,time.time()-t0
                    elif et=='exec':print(f'  EXEC at {time.time()-t0:.2f}s: {ed[:160]}',flush=True)
                    elif et=='meta':print(f'  META at {time.time()-t0:.2f}s: {ed[:160]}',flush=True)
                    elif et=='error':print(f'  ERROR: {ed}',flush=True)
    return chunks,done,trun,time.time()-t0
Q='Write a python function reverse_string(s) that returns s reversed, then print(reverse_string("hello")).'
chunks,done,trun,wall=stream(Q)
tier=done.get('tier','') if done else ''
print(f'\ntier={tier}  wall={wall:.1f}s  trun={trun}',flush=True)
out=''.join(chunks).encode('ascii','replace').decode('ascii')
print(f'\n--- answer ---\n{out[:2500]}',flush=True)
if trun:
    div=trun.get('diversity',0.0)
    print(f'\nDIVERSITY SCORE: {div}',flush=True)
    print(f'TIER SUFFIX: {[s for s in ("_tests_thin","_tests_ok","_tests_diverse") if s in tier]}',flush=True)
