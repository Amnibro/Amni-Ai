import os,sys,json,argparse,urllib.request
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANONICAL=[
('What language do the Al Bhed speak?','Al Bhed — spelled B-H-E-D — a one-to-one letter-substitution cipher used by the Al Bhed people in Final Fantasy X. It is NOT spelled "Al Beth".'),
('Who are you?','I am Adam, a GF(17) texture-native local AI assistant built by Anthony Reffelt.'),
('What is your name?','Adam.'),
('How is Adam spelled?','A-D-A-M.'),
('What are you?','Adam — a fully local, GF(17) texture-native AI: weights stored as lossless RGBA PTEX atlases, a closed-loop self-iterative memory (PTEX+ATEX) across learning, memory, context, knowledge, and wisdom.'),
]
def _valid(rows):
    bad=[(i,q,a) for i,(q,a) in enumerate(rows) if not isinstance(q,str) or not isinstance(a,str) or len(q.strip())<3 or len(a.strip())<3]
    return (not bad),bad
def seed_http(base,rows):
    ok=0
    for q,a in rows:
        data=json.dumps({'question':q,'answer':a}).encode('utf-8')
        req=urllib.request.Request(base.rstrip('/')+'/teach',data=data,headers={'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(req,timeout=8) as r:ok+=1 if r.status<400 else 0
        except Exception as e:print(f'[seed] HTTP teach failed for {q!r}: {e}',flush=True)
    return ok
def seed_local(root,rows):
    from amni.inference.answer_lut import AnswerLUT
    from amni.learning.knowledge_base import KnowledgeBase
    from amni.serve.memory_bus import MemoryBus
    lut=AnswerLUT(os.path.join(root,'answers'));kb=KnowledgeBase(os.path.join(root,'knowledge'))
    bus=MemoryBus(adam=None,answer_lut=lut,sem_lut=None,kb=kb,learning_atlas=None,ledger_path=os.path.join(root,'answers','corrections.jsonl'))
    ok=0
    for q,a in rows:
        r=bus.record_learning(q,a,kind='fact',provenance='user:Anthony',exactness='exact')
        ok+=1 if r.get('stored') and r.get('recall_ok') else 0
        print(f'[seed] {("OK " if r.get("recall_ok") else "?? ")}{q[:48]!r} -> homes={r.get("homes")}',flush=True)
    return ok
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=os.environ.get('AMNI_ADAM_ROOT','experiences/adam_lut'))
    ap.add_argument('--server',default='http://127.0.0.1:7700')
    ap.add_argument('--prefer-http',action='store_true')
    a=ap.parse_args()
    okv,bad=_valid(CANONICAL)
    if not okv:print(f'[seed] malformed rows: {bad}',flush=True);sys.exit(2)
    if a.prefer_http:
        n=seed_http(a.server,CANONICAL)
        if n==len(CANONICAL):print(f'[seed] seeded {n}/{len(CANONICAL)} via HTTP',flush=True);return
        print('[seed] HTTP incomplete, falling back to local on-disk seed',flush=True)
    n=seed_local(a.root,CANONICAL)
    print(f'[seed] seeded {n}/{len(CANONICAL)} canonical facts into {a.root} (ATEX-exact + KB; instant tier0 recall on restart)',flush=True)
    sys.exit(0 if n==len(CANONICAL) else 1)
if __name__=='__main__':main()
