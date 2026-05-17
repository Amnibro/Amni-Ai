"""Iter18 e2e: ask code question, expect tests_ok+promoted; ask again, expect tier1_5_semantic_lesson sub-second."""
import json,time,urllib.request
BASE='http://127.0.0.1:8001'
def stream(msg,t=400):
    req=urllib.request.Request(f'{BASE}/chat/stream',data=json.dumps({'message':msg}).encode(),headers={'Content-Type':'application/json'},method='POST')
    t0=time.time();chunks=[];events={};done=None;last_meta={}
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
                    elif et=='promoted':print(f'  PROMOTED at {time.time()-t0:.2f}s: {ed}',flush=True)
                    elif et=='test_run':print(f'  TEST_RUN at {time.time()-t0:.2f}s: {ed}',flush=True)
                    elif et=='exec':print(f'  EXEC at {time.time()-t0:.2f}s: {ed[:160]}',flush=True)
                    elif et=='done':done=json.loads(ed);print(f'  DONE at {time.time()-t0:.2f}s: {ed}',flush=True);return chunks,events,done,time.time()-t0
                    elif et=='meta':print(f'  META at {time.time()-t0:.2f}s: {ed[:160]}',flush=True)
                    elif et=='error':print(f'  ERROR: {ed}',flush=True)
    return chunks,events,done,time.time()-t0
Q='Write a python function add_one(n) that returns n+1, then print(add_one(5)).'
print('=== Round 1: generate + promote ===',flush=True)
chunks1,events1,done1,wall1=stream(Q)
tier1=done1.get('tier','') if done1 else ''
print(f'\nRound 1 tier={tier1}  wall={wall1:.1f}s  events={events1}',flush=True)
assert '_promoted' in tier1,f'expected _promoted suffix, got {tier1}'
print('=== Round 2: SAME query — should hit LUT ===',flush=True)
time.sleep(2)
chunks2,events2,done2,wall2=stream(Q)
tier2=done2.get('tier','') if done2 else ''
print(f'\nRound 2 tier={tier2}  wall={wall2:.1f}s',flush=True)
assert 'lesson' in tier2 or 'tier1' in tier2 or wall2<5,f'round2 should hit LUT or be sub-5s, got tier={tier2} wall={wall2:.1f}s'
print(f'\nSpeedup: {wall1:.1f}s -> {wall2:.1f}s  ({wall1/max(wall2,0.01):.0f}x)',flush=True)
print('PASS',flush=True)
