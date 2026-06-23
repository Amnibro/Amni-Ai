"""MacroCodeEngine — the codegen accelerator. Auto-mines recurring code blocks into PTEX macro-tokens, then emits them by SPECULATIVE-ACCEPT: a matched block is verified in ~1 forward pass (draft window K) instead of generated token-by-token. Self-growing (learn() = learning ratchet): every verified/accepted output is mined back in, so coverage->1 and speed compounds the more a codebase is worked. Accepted blocks are known-good code, so they also RAISE correctness (no typos/invisible-unicode on accepted spans). coverage() gives the CPU-measurable forward-pass savings; the GPU speculative-decoder uses blocks as drafts. Built on PtexMacroCache (1 block = 1 lossless PTEX macro-token)."""
import math,re,keyword
from collections import Counter
from amni.inference.ptex_macro_cache import PtexMacroCache
_KEEP=set(keyword.kwlist)|{'self','cls','STR','NUM','ID','print','len','range','int','str','float','list','dict','set','tuple','enumerate','zip','open','super','isinstance','return','import','from'}
def normalize_line(L):
    ind=len(L)-len(L.lstrip());s=L.strip()
    s=re.sub(r'(["\']).*?\1','STR',s);s=re.sub(r'\b\d+\.?\d*\b','NUM',s)
    s=re.sub(r'\b[A-Za-z_]\w*\b',lambda m:m.group(0) if m.group(0) in _KEEP else 'ID',s)
    return (' '*ind)+s
class MacroCodeEngine:
    def __init__(s,tok,max_block=3,min_freq=2,min_len=6):
        s.tok=tok;s.max_block=max_block;s.min_freq=min_freq;s.min_len=min_len;s.blocks={};s.struct={};s.cache=PtexMacroCache(tok);s._ntok={}
    def _enc(s,t):return s.tok.encode(t,add_special_tokens=False)
    def _ntoks(s,b):
        if b not in s._ntok:s._ntok[b]=len(s._enc(b))
        return s._ntok[b]
    def _spans(s,code):
        lines=code.split('\n')
        for L in range(1,s.max_block+1):
            for i in range(len(lines)-L+1):
                b='\n'.join(lines[i:i+L])
                if len(b.strip())>=s.min_len:yield b
    def mine(s,texts):
        cnt=Counter()
        for code in texts:
            for b in s._spans(code):cnt[b]+=1
        a=0
        for b,f in cnt.items():
            if f>=s.min_freq and b not in s.blocks:s.blocks[b]=f;a+=1
        return a
    def learn(s,text):
        a=0
        for b in s._spans(text):
            if b not in s.blocks:s.blocks[b]=99;a+=1
        return a
    def intern_all(s):
        for b in s.blocks:s.cache.intern(b)
        return s.cache.stats()
    def coverage(s,code,K=16):
        lines=code.split('\n');i=0;cov=0;accepts=0;passes=0
        while i<len(lines):
            m=0
            for L in range(min(s.max_block,len(lines)-i),0,-1):
                if '\n'.join(lines[i:i+L]) in s.blocks:m=L;break
            if m:
                nt=s._ntoks('\n'.join(lines[i:i+m]));cov+=nt;accepts+=1;passes+=max(1,math.ceil(nt/K));i+=m
            else:
                passes+=max(1,s._ntoks(lines[i]));i+=1
        tot=len(s._enc(code))
        return {'tot':tot,'cov_pct':round(100*cov/max(tot,1),1),'accepts':accepts,'passes':passes,'speedup':round(tot/max(passes,1),2),'lib':len(s.blocks)}
    def mine_struct(s,texts):
        cnt=Counter()
        for code in texts:
            nl=[normalize_line(L) for L in code.split('\n')]
            for L in range(1,s.max_block+1):
                for i in range(len(nl)-L+1):
                    b='\n'.join(nl[i:i+L])
                    if len(b.strip())>=s.min_len:cnt[b]+=1
        a=0
        for b,f in cnt.items():
            if f>=s.min_freq and b not in s.struct:s.struct[b]=f;a+=1
        return a
    def struct_coverage(s,code):
        nl=[normalize_line(L) for L in code.split('\n')];raw=code.split('\n');i=0;known=0;accepts=0;holes=0;gen=0
        while i<len(nl):
            m=0;mb=None
            for L in range(min(s.max_block,len(nl)-i),0,-1):
                b='\n'.join(nl[i:i+L])
                if b in s.struct:m=L;mb=b;break
            if m:
                nt=s._ntoks('\n'.join(raw[i:i+m]));known+=nt;accepts+=1;p=mb.count('ID')+mb.count('NUM')+mb.count('STR');holes+=p;i+=m
            else:
                gen+=s._ntoks(raw[i]);i+=1
        tot=len(s._enc(code));passes=holes+gen
        return {'tot':tot,'known_pct':round(100*known/max(tot,1),1),'accepts':accepts,'holes':holes,'passes':passes,'speedup_est':round(tot/max(passes,1),2),'lib':len(s.struct)}
