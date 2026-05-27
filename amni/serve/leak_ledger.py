"""leak_ledger — self-learning loop for thinking-process / tool-narration leaks.
Closed loop: regex stripper DETECTS a leak -> record() COMMITS the offending fragment to an append-only ledger + a dedicated PTEX file (errors-to-avoid) -> scrub_learned() CONSULTS accumulated signatures to catch repeats the regex never anticipated. Each leak makes the next one likelier to be caught.
Leaks are Adam's OWN scaffolding (not user PII), so the fragment text is safe to store locally; the ledger is gitignored."""
import time,json,re,threading,hashlib
from pathlib import Path
from typing import Dict,Any,List,Optional
_LOCK=threading.Lock()
_SIG_MIN=16
_sigs:Optional[set]=None
_SCAFFOLD_HINTS=re.compile(r'(?i)\b(?:thinking\s*process|thought\s*process|chain[\s-]of[\s-]thought|analyze\s+the\s+request|check\s+(?:context|tools)|determine\s+(?:the\s+)?strategy|formulate|self[- ]correction|refinement|recall\s+(?:the\s+)?persona|apply\s+(?:the\s+)?persona|final\s+output|restate|first\s+shot|critique|\bknowns\b|\bapproach\b|looked\s+it\s+up|search\s+(?:performed|returns?|completed)|presenting|simulating|outputting|the\s+system\s+(?:returns?|outputs?))\b')
def _is_scaffoldish(s:str)->bool:
    return bool(_SCAFFOLD_HINTS.search(s or ''))
def _ledger_path()->Path:
    p=Path(__file__).resolve().parents[2]/'data'/'thinking_leaks.jsonl';p.parent.mkdir(parents=True,exist_ok=True);return p
def _ptex_path()->Path:
    p=Path(__file__).resolve().parents[2]/'lessons'/'leak_avoidance_ptex';p.parent.mkdir(parents=True,exist_ok=True);return p
def _norm(s:str)->str:
    return re.sub(r'\s+',' ',(s or '')).strip()
def _signatures_from(fragment:str)->List[str]:
    frag=_norm(fragment)
    out=[]
    if len(frag)>=_SIG_MIN:out.append(frag[:160])
    for part in re.split(r'(?<=[.:;])\s+|\n+',frag):
        part=_norm(part)
        if len(part)>=_SIG_MIN:out.append(part[:160])
    seen=set();uniq=[]
    for s in out:
        if not _is_scaffoldish(s):continue
        k=s.lower()
        if k not in seen:seen.add(k);uniq.append(s)
    return uniq
def _load_sigs()->set:
    global _sigs
    if _sigs is not None:return _sigs
    _sigs=set()
    p=_ledger_path()
    if p.exists():
        try:
            for ln in p.read_text(encoding='utf-8').splitlines():
                ln=ln.strip()
                if not ln:continue
                try:r=json.loads(ln)
                except Exception:continue
                for s in (r.get('signatures') or []):
                    if len(s)>=_SIG_MIN:_sigs.add(s.lower())
        except Exception:pass
    return _sigs
def record(leaked:str,clean:str='',categories:Optional[List[str]]=None,source:str='wrap')->Dict[str,Any]:
    leaked=_norm(leaked)
    if len(leaked)<_SIG_MIN:return {'recorded':False,'reason':'fragment too short'}
    sigs=_signatures_from(leaked)
    rec={'ts':time.time(),'iso':time.strftime('%Y-%m-%dT%H:%M:%S'),'source':source,'categories':sorted(set(categories or [])),'leaked':leaked[:400],'clean_preview':_norm(clean)[:120],'signatures':sigs,'hash':hashlib.sha256(leaked.encode('utf-8','ignore')).hexdigest()[:12]}
    try:
        from amni.serve.reffelt_tag import tag_record
        _t=tag_record(_norm(clean) or leaked,extra_tags=['thinking_process_leak']);rec['tags']=_t['tags'];rec['nonce']=_t['nonce']
    except Exception:pass
    try:
        with _LOCK:
            with _ledger_path().open('a',encoding='utf-8') as fh:fh.write(json.dumps(rec,ensure_ascii=False)+'\n')
            s=_load_sigs()
            for sg in sigs:s.add(sg.lower())
    except Exception as e:return {'recorded':False,'error':str(e)}
    return {'recorded':True,'signatures':len(sigs),'hash':rec['hash']}
def scrub_learned(text:str)->str:
    if not text:return text
    sigs=_load_sigs()
    if not sigs:return text
    t=text
    for sg in sorted(sigs,key=len,reverse=True):
        if len(sg)<_SIG_MIN or not _is_scaffoldish(sg):continue
        t=re.sub(re.escape(sg),' ',t,flags=re.IGNORECASE)
    t=re.sub(r'[ \t]+',' ',t);t=re.sub(r'\n{3,}','\n\n',t)
    return t.strip() or text
def commit_to_ptex(lut=None,adam=None,max_pairs:int=300,save:bool=True)->Dict[str,Any]:
    p=_ledger_path()
    if not p.exists():return {'committed':0,'reason':'no ledger'}
    pairs=[]
    try:
        for ln in p.read_text(encoding='utf-8').splitlines():
            ln=ln.strip()
            if not ln:continue
            try:r=json.loads(ln)
            except Exception:continue
            q=r.get('leaked','');a=r.get('clean_preview','') or '[avoid: internal reasoning / tool narration leaked — emit only the final answer]'
            if q:pairs.append((q,a))
    except Exception as e:return {'committed':0,'error':str(e)}
    pairs=pairs[-int(max_pairs):]
    if not pairs:return {'committed':0,'reason':'no pairs'}
    if lut is None:
        try:
            from amni.inference.semantic_ptex_lut import SemanticPTEXLUT
            enc=getattr(getattr(adam,'sem_lut',None),'encoder',None) if adam is not None else None
            lut=SemanticPTEXLUT(grid=32,pca_dim=4,encoder=enc)
        except Exception as e:return {'committed':0,'error':f'lut unavailable: {e}'}
    for q,a in pairs:
        try:lut.add(q,a)
        except Exception:pass
    if save:
        try:lut.fit();lut.save(str(_ptex_path()))
        except Exception as e:return {'committed':len(pairs),'fit_or_save_error':str(e)[:160]}
    return {'committed':len(pairs),'ptex':str(_ptex_path())}
def stats(limit:int=30)->Dict[str,Any]:
    p=_ledger_path()
    if not p.exists():return {'total':0,'by_category':{},'by_source':{},'distinct_signatures':0,'recent':[]}
    rows=[]
    try:
        for ln in p.read_text(encoding='utf-8').splitlines():
            ln=ln.strip()
            if not ln:continue
            try:rows.append(json.loads(ln))
            except Exception:continue
    except Exception:return {'total':0,'by_category':{},'by_source':{},'distinct_signatures':0,'recent':[]}
    bycat={};bysrc={}
    for r in rows:
        for c in r.get('categories',[]):bycat[c]=bycat.get(c,0)+1
        s=r.get('source','?');bysrc[s]=bysrc.get(s,0)+1
    rows.sort(key=lambda r:-float(r.get('ts') or 0))
    return {'total':len(rows),'by_category':dict(sorted(bycat.items(),key=lambda kv:-kv[1])),'by_source':dict(sorted(bysrc.items(),key=lambda kv:-kv[1])),'distinct_signatures':len(_load_sigs()),'recent':[{k:r.get(k) for k in ('iso','source','categories','clean_preview','hash')} for r in rows[:int(max(1,limit))]]}
def _reset_cache():
    global _sigs;_sigs=None
