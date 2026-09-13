import os,json,time,hashlib,re
from pathlib import Path
from typing import Optional,Dict,Any,Tuple
try:from amni.inference.answer_lut import _normalize_query as _norm
except Exception:
    def _norm(q):return re.sub(r'\s+',' ',(q or '').strip().lower())
def _sig(s):return re.sub(r'\s+',' ',(s or '').strip().lower())
def _kbkey(kind,key):return f'{kind}::'+hashlib.sha256(_norm(key).encode('utf-8','ignore')).hexdigest()[:16]
_CONF_DIRECT=float(os.environ.get('AMNI_RECALL_DIRECT_GATE','0.90'))
_CONF_GROUND=float(os.environ.get('AMNI_RECALL_GROUND_GATE','0.84'))
_CONF_MARGIN=float(os.environ.get('AMNI_RECALL_MARGIN','0.04'))
_LEDGER_OVERLAP=float(os.environ.get('AMNI_LEDGER_OVERLAP','0.45'))
_STOP=set("a an the is are was were be been being do does did what who when where which whom whose how why that this these those it its of to for and or in on at by from with as if then than so not no yes i you we they he she them my your our their".split())
def _content_tokens(s):
    return [t for t in re.findall(r"[a-z0-9']+",_norm(s)) if t not in _STOP and len(t)>1]
def _token_overlap(a,b)->float:
    na,nb=_norm(a),_norm(b)
    if not na or not nb:return 0.0
    if len(na)>=12 and len(nb)>=12 and (na in nb or nb in na):return 1.0
    ta,tb=set(_content_tokens(a)),set(_content_tokens(b))
    if len(ta)<2 or len(tb)<2:return 0.0
    inter=len(ta&tb)
    if inter<2:return 0.0
    return inter/min(len(ta),len(tb))
class MemoryBus:
    def __init__(self,adam=None,answer_lut=None,sem_lut=None,kb=None,learning_atlas=None,ledger_path='data/corrections.jsonl'):
        self.adam=adam
        self.answer_lut=answer_lut if answer_lut is not None else getattr(getattr(adam,'adam',None),'lut',None)
        self.sem_lut=sem_lut if sem_lut is not None else getattr(adam,'sem_lut',None)
        self.kb=kb
        self.la=learning_atlas
        self.ledger=Path(ledger_path)
        self._antipattern=set()
        self._ledger_pairs=[]
        self._tier_counts={'tier0_atex_override':0,'tier0_ledger_overlap':0,'tier2_sem':0,'tier3_kb':0,'miss':0}
        self._load_antipattern()
    def _load_antipattern(self):
        try:
            if self.ledger.exists():
                for ln in self.ledger.read_text(encoding='utf-8').splitlines():
                    try:
                        obj=json.loads(ln)
                        self._antipattern.add(_sig(obj.get('wrong','')))
                        q,c=obj.get('q'),obj.get('corrected')
                        if q and c:self._ledger_pairs.append((q,c))
                    except Exception:pass
                self._ledger_pairs=self._ledger_pairs[-80:]
        except Exception:pass
    def _remember_ledger(self,q,corrected):
        if q and corrected:
            self._ledger_pairs.append((q,corrected))
            self._ledger_pairs=self._ledger_pairs[-80:]
    def record_learning(self,key,value,kind='fact',provenance='',exactness='semantic',supersedes=None)->Dict[str,Any]:
        if not key or not value:return {'stored':False,'homes':[],'recall_ok':False,'reason':'empty'}
        homes=[];conf=1.0 if str(provenance).startswith('user:') else 0.7;ts=time.time()
        meta={'kind':kind,'wrong':supersedes,'confidence':conf,'priority':100 if exactness=='exact' else 50,'ts':ts,'provenance':provenance,'federable':bool(kind in('fact','skill_lesson') and conf>=0.7 and 'personal' not in kind)}
        if exactness=='exact' and self.answer_lut is not None:
            try:self.answer_lut.store(key,value,source=provenance or 'memory_bus',meta=dict(meta),track_recent=True);homes.append('atex')
            except Exception:pass
        if self.kb is not None:
            try:
                self.kb.add(_kbkey(kind,key),value,meta=dict(meta));getattr(self.kb,'flush',lambda:None)();homes.append('kb')
            except Exception:pass
        if self.adam is not None and hasattr(self.adam,'teach'):
            try:self.adam.teach(key,value);homes.append('ptex_sem')
            except Exception:pass
        if self.la is not None:
            try:
                self.la.record(key,value,source=provenance or kind,confidence=conf,kind=kind)
                if supersedes:self.la.mark_debated(key,supersedes,value,source=provenance or 'correction')
            except Exception:pass
        if supersedes:
            self._antipattern.add(_sig(supersedes))
            try:
                self.ledger.parent.mkdir(parents=True,exist_ok=True)
                with self.ledger.open('a',encoding='utf-8') as f:f.write(json.dumps({'q':key,'wrong':supersedes,'corrected':value,'kind':kind,'provenance':provenance,'ts':ts})+'\n')
                self._remember_ledger(key,value)
                homes.append('ledger')
            except Exception:pass
        recall_ok=False
        try:
            v,_home,_c=self.recall(key)
            recall_ok=v is not None and (_sig(v)==_sig(value) if exactness=='exact' else (_sig(v)==_sig(value) or _sig(value) in _sig(v)))
        except Exception:recall_ok=False
        if exactness=='exact' and self.answer_lut is not None and 'atex' in homes:
            try:(self.answer_lut.commit_recent() if recall_ok else self.answer_lut.rollback_recent())
            except Exception:pass
            if not recall_ok:
                try:homes.remove('atex')
                except Exception:pass
        _durable=('kb' in homes) or ('atex' in homes)
        stored=bool(homes) and (recall_ok or (exactness!='exact' and _durable))
        return {'stored':stored,'homes':homes,'recall_ok':recall_ok}
    def is_suppressed(self,text)->bool:return _sig(text) in self._antipattern
    def suppress(self,wrong,reason='',q='')->bool:
        if not wrong:return False
        self._antipattern.add(_sig(wrong))
        try:
            self.ledger.parent.mkdir(parents=True,exist_ok=True)
            with self.ledger.open('a',encoding='utf-8') as f:f.write(json.dumps({'q':q,'wrong':wrong,'corrected':None,'kind':'suppression','provenance':reason,'ts':time.time()})+'\n')
        except Exception:pass
        return True
    def grounding_fact(self,query)->Optional[str]:
        try:
            v,home,c=self.recall(query,gate=_CONF_GROUND)
            return f'AUTHORITATIVE — a learned correction/fact is on file for this topic (recall confidence {round(float(c),3)}); ground your answer in it and do NOT contradict it: {str(v)[:400]}' if v else None
        except Exception:return None
    def _hit(self,home):self._tier_counts[home]=self._tier_counts.get(home,0)+1;return home
    def stats(self)->Dict[str,Any]:
        tc=dict(self._tier_counts);total=sum(tc.values());hits=total-tc.get('miss',0)
        return {'recall_total':total,'recall_hits':hits,'recall_hit_rate':round(hits/total,3) if total else 0.0,'by_tier':tc,'antipattern_n':len(self._antipattern)}
    def recall(self,query,gate=None)->Tuple[Optional[str],str,float]:
        gate=_CONF_DIRECT if gate is None else gate
        if self.answer_lut is not None:
            try:
                hit=self.answer_lut.lookup(query)
                if hit and hit.get('a') and not self.is_suppressed(hit.get('a')):return hit['a'],self._hit('tier0_atex_override'),1.0
            except Exception:pass
        led=self._ledger_overlap(query)
        if led:return led
        if self.sem_lut is not None and getattr(self.sem_lut,'_raw',None):
            try:
                res=self.sem_lut.lookup_soft(query,k=1,return_diag=True)
                a,_d,cos_top,cos_2nd=res if (isinstance(res,tuple) and len(res)==4) else (res,None,None,None)
                if a and not self.is_suppressed(a):
                    ct=float(cos_top) if cos_top is not None else 0.0
                    ok_margin=(cos_2nd is None) or ((ct-float(cos_2nd))>=_CONF_MARGIN)
                    if ct>=gate and ok_margin:return a,self._hit('tier2_sem'),round(ct,4)
            except Exception:pass
        if self.kb is not None:
            try:
                v=self.kb.lookup(_kbkey('correction',query)) or self.kb.lookup(_kbkey('fact',query))
                if v and not self.is_suppressed(v) and 0.95>=gate:return v,self._hit('tier3_kb'),0.95
            except Exception:pass
        return None,self._hit('miss'),0.0
    def _ledger_overlap(self,query):
        best=None;best_s=0.0
        for q,c in self._ledger_pairs:
            if not c or self.is_suppressed(c):continue
            s=_token_overlap(query,q)
            if s>best_s:best_s=s;best=c
        if best and best_s>=_LEDGER_OVERLAP:return best,self._hit('tier0_ledger_overlap'),round(best_s,4)
        return None
    def learned_override(self,query):
        try:v,home,c=self.recall(query)
        except Exception:return None,'miss',0.0
        if v and str(home).startswith('tier0_'):return v,home,c
        return None,home,c
