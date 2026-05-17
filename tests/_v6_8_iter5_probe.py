"""Probe iter5 additions + verify previously-missed paraphrases now hit."""
import json,time,urllib.request,urllib.error
BASE='http://127.0.0.1:8002'
def post(p,b,t=15):
    r=urllib.request.Request(f'{BASE}{p}',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as resp:return resp.status,json.loads(resp.read().decode())
    except Exception as e:return None,{'error':str(e)}
def get(p):
    with urllib.request.urlopen(f'{BASE}{p}',timeout=20) as resp:return json.loads(resp.read().decode())
def ask(msg,t=10):
    t0=time.time()
    code,r=post('/chat',{'message':msg},t=t)
    wall=time.time()-t0
    ans=(r.get("answer") or "")[:130].encode("ascii","replace").decode("ascii")
    print(f'[{wall:5.2f}s {r.get("tier","?"):<27}] {msg!r}',flush=True)
    print(f'  -> {ans}\n',flush=True)
print('='*80,flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}\n',flush=True)
print('--- PYTHON LIBS ---',flush=True)
for q in ['What is numpy broadcasting?','What is the difference between loc and iloc?','What is FastAPI?','What is a pytest fixture?','What is Pydantic?']:ask(q)
print('--- AI/RAG ---',flush=True)
for q in ['What is a token?','What is temperature in LLM generation?','What is RAG?','What is chunking in RAG?','What is hybrid search in RAG?','What is a prompt injection attack?']:ask(q)
print('--- LEETCODE HARD ---',flush=True)
for q in ['What is the longest palindromic substring approach?','What is the merge K sorted lists problem?','How do I serialize and deserialize a binary tree?','What is the single number problem XOR trick?']:ask(q)
print('--- PARAPHRASE VARIANTS (previously missed) ---',flush=True)
for q in ['What is delta time?','What is volatile in C?','What is the difference between MCU and MPU?','What is the walrus operator?']:ask(q)
print('--- DEBUGGING ADVANCED ---',flush=True)
for q in ['How do I debug code I did not write?','What is rubber duck debugging?','What is a Heisenbug?','What is bisect technique?']:ask(q)
print('\n=== STATS ===',flush=True)
s=get('/stats');print(f'lessons={s["lessons_n"]}',flush=True)
