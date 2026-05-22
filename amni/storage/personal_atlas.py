"""PersonalAtlas — organic per-user PTEX cell-address LUT for personal facts.
No schema. No fixed fields. Every fact the user reveals gets embedded via MiniLM, projected to a grid cell, and stored AT that cell address. The cell IS the category — neighborhoods emerge naturally (family-talk clusters next to family-talk, hobby-talk near hobby-talk). New topics the user invents land at their own address with zero code change.
Confidentiality is three-state — public / confidential / UNCLEAR — and UNCLEAR is the safe default. UNCLEAR facts are held in a pending queue until the user explicitly confirms; they NEVER auto-resolve to public. Background daemon thread does extraction so chat turns don't block. Single `__local_user__` slot only — never per-session, never `__global__`, never federates.
Public API:
  enqueue(message,session_id)        — fire-and-forget; worker picks it up
  recall(query,k,max_radius)         — cell-walk; returns facts (filtered by confidentiality)
  pending_clarifications()           — list of UNCLEAR facts awaiting user confirm
  confirm_clarification(fact_id,is_confidential) — finalizes pending
  forget(pattern)                    — chat-driven erasure
  stats()                            — entries / pending / dead-letter sizes"""
import json,time,re,threading,queue,hashlib,uuid
from pathlib import Path
from typing import List,Dict,Any,Optional,Tuple,Callable
from amni.inference.semantic_ptex_lut import SemanticPTEXLUT
_LOCAL_KEY='__local_user__'
_REFIT_EVERY=5
_QUEUE_MAX=200
_WORKER_POLL_S=0.5
_EXTRACTION_PROMPT=(
    'Extract atomic personal facts the USER asserts about themselves in this single message. '
    'For each fact, output JSON:\n'
    '{"fact":"<short third-person statement, e.g. user\'s partner is named Maria>", '
    '"confidence":0.0-1.0, '
    '"confidentiality":"public" | "confidential" | "unclear"}\n\n'
    'CONFIDENTIALITY RULES (strict):\n'
    '- "confidential" — medical conditions, medications, allergies, addresses, phone, email, SSN, finances, names of family/children, employer details not publicly listed, religious or political beliefs, sexual orientation, immigration status, legal history, mental health, exact birthday.\n'
    '- "public" — favorite color/food/music/team, hobbies that do not reveal identity, programming languages or tools the user works with, broad city or timezone, general profession in broad strokes.\n'
    '- "unclear" — anything that could go either way. WHEN UNCERTAIN, ALWAYS PICK "unclear". Never default to "public" if you are not sure.\n\n'
    'Output ONLY a JSON array (or [] if no personal facts). No prose, no markdown fences.\n\n'
    'User message: "{MSG}"\n\n'
    'JSON array:'
)
_JSON_ARR_RE=re.compile(r'\[(?:[^\[\]]|\[[^\[\]]*\])*\]',re.DOTALL)
_CONFIRM_YES_RE=re.compile(r'\b(?:yes|yeah|yep|yup|confidential|private|don\'?t\s+share|keep\s+(?:it\s+)?(?:private|confidential|secret)|sensitive|hide|don\'?t\s+mention|do\s+not\s+(?:share|mention))\b',re.IGNORECASE)
_CONFIRM_NO_RE=re.compile(r'\b(?:no|nope|public|fine|share|okay\s+to\s+mention|okay\s+to\s+share|not\s+(?:secret|private|sensitive)|whatever|don\'?t\s+care)\b',re.IGNORECASE)
class _Slot:
    __slots__=('lut','meta','refit_due','lock')
    def __init__(self,lut,meta):self.lut=lut;self.meta=meta;self.refit_due=0;self.lock=threading.Lock()
class PersonalAtlas:
    def __init__(self,root:str='experiences/personal_atlas',encoder=None,adam=None,grid:int=64,pca_dim:int=8,enable_worker:bool=True):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.grid=grid;self.pca_dim=pca_dim;self.encoder=encoder;self.adam=adam
        self._pending_path=self.root/'pending_clarifications.jsonl'
        self._dead_letter_path=self.root/'dead_letter.jsonl'
        self._queue:queue.Queue=queue.Queue(maxsize=_QUEUE_MAX)
        self._stop_event=threading.Event()
        self._slot=self._load_slot()
        self._pending=self._load_pending()
        self._last_asked_id=None
        self._last_asked_ts=0.0
        self._ask_expire_s=300.0
        self._worker_thread=None
        if enable_worker and adam is not None:
            self._worker_thread=threading.Thread(target=self._worker_loop,name='PersonalAtlasWorker',daemon=True)
            self._worker_thread.start()
    def _slot_path(self)->Path:return self.root/f'personal_{_LOCAL_KEY}'
    def _meta_path(self)->Path:return self.root/f'personal_{_LOCAL_KEY}.meta.json'
    def _load_slot(self)->_Slot:
        sp=self._slot_path();mp=self._meta_path()
        if not mp.exists():return _Slot(SemanticPTEXLUT(grid=self.grid,pca_dim=self.pca_dim,encoder=self.encoder),{})
        try:lut=SemanticPTEXLUT.load(str(sp),encoder=self.encoder)
        except Exception:lut=SemanticPTEXLUT(grid=self.grid,pca_dim=self.pca_dim,encoder=self.encoder)
        try:
            raw=json.loads(mp.read_text(encoding='utf-8'))
            meta={tuple(k.split('::',1)):v for k,v in raw.items()} if isinstance(raw,dict) else {}
        except Exception:meta={}
        return _Slot(lut,meta)
    def _save_slot(self):
        with self._slot.lock:
            try:
                _embs=self._slot.lut._stored_embs
                _n_raw=len(self._slot.lut._raw)
                if _n_raw>0 and (_embs is None or len(_embs)!=_n_raw):
                    try:self._slot.lut.fit()
                    except Exception as fe:print(f'[PersonalAtlas] pre-save refit failed: {fe}',flush=True)
                if _n_raw>0 and self._slot.lut._stored_embs is not None:self._slot.lut.save(str(self._slot_path()))
            except Exception as e:print(f'[PersonalAtlas] save lut failed: {e}',flush=True)
            try:self._meta_path().write_text(json.dumps({'::'.join(k):v for k,v in self._slot.meta.items()},default=str),encoding='utf-8')
            except Exception as e:print(f'[PersonalAtlas] save meta failed: {e}',flush=True)
    def _load_pending(self)->List[Dict[str,Any]]:
        if not self._pending_path.exists():return []
        out=[]
        try:
            for ln in self._pending_path.read_text(encoding='utf-8').strip().splitlines():
                if ln.strip():
                    try:out.append(json.loads(ln))
                    except Exception:continue
        except Exception:pass
        return out
    def _save_pending(self):
        try:
            with self._pending_path.open('w',encoding='utf-8') as f:
                for p in self._pending:f.write(json.dumps(p,default=str)+'\n')
        except Exception as e:print(f'[PersonalAtlas] save pending failed: {e}',flush=True)
    def _dead_letter(self,record:Dict[str,Any]):
        try:
            with self._dead_letter_path.open('a',encoding='utf-8') as f:f.write(json.dumps({'ts':time.time(),**record},default=str)+'\n')
        except Exception:pass
    def enqueue(self,message:str,session_id:Optional[str]=None):
        if not message or not message.strip():return False
        try:self._queue.put_nowait({'message':message,'session_id':session_id,'ts':time.time()})
        except queue.Full:self._dead_letter({'kind':'queue_full','message':message[:300],'session_id':session_id});return False
        return True
    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:item=self._queue.get(timeout=_WORKER_POLL_S)
            except queue.Empty:continue
            try:self._process(item)
            except Exception as e:self._dead_letter({'kind':'process_exception','error':str(e)[:300],'item':{k:(v[:200] if isinstance(v,str) else v) for k,v in item.items()}})
            finally:
                try:self._queue.task_done()
                except Exception:pass
    def _extract_with_adam(self,message:str)->List[Dict[str,Any]]:
        if self.adam is None or not hasattr(self.adam,'chat_persona'):return []
        prompt=_EXTRACTION_PROMPT.replace('{MSG}',message[:600].replace('"','\\"'))
        try:r=self.adam.chat_persona(prompt,system='You are a strict JSON-only fact extractor. Output a JSON array, nothing else.',max_new_tokens=400,do_sample=False)
        except Exception as e:self._dead_letter({'kind':'adam_call_failed','error':str(e)[:300],'message':message[:200]});return []
        ans=(r or {}).get('answer','') if isinstance(r,dict) else ''
        if not ans:return []
        m=_JSON_ARR_RE.search(ans)
        raw=m.group(0) if m else ans.strip()
        try:parsed=json.loads(raw)
        except Exception:
            try:parsed=json.loads(raw.replace("'",'"'))
            except Exception:return []
        if not isinstance(parsed,list):return []
        out=[]
        for f in parsed:
            if not isinstance(f,dict):continue
            fact=str(f.get('fact','')).strip()
            if not fact:continue
            conf=float(f.get('confidence',0.5) or 0.5)
            cclass=str(f.get('confidentiality','unclear')).strip().lower()
            if cclass not in ('public','confidential','unclear'):cclass='unclear'
            out.append({'fact':fact,'confidence':conf,'confidentiality':cclass})
        return out
    def _process(self,item:Dict[str,Any]):
        msg=item.get('message','');sid=item.get('session_id')
        facts=self._extract_with_adam(msg)
        for f in facts:
            cclass=f['confidentiality']
            if cclass=='unclear':
                pid=uuid.uuid4().hex[:16]
                self._pending.append({'id':pid,'fact':f['fact'],'source_excerpt':msg[:240],'confidence':f['confidence'],'session_id':sid,'ts':time.time()})
                self._save_pending()
            else:
                is_conf=(cclass=='confidential')
                self._record(f['fact'],msg[:240],is_conf,f['confidence'])
    def _record(self,fact:str,source_excerpt:str,is_confidential:bool,confidence:float):
        ts=time.time()
        with self._slot.lock:
            self._slot.lut.add(fact,fact)
            self._slot.meta[(fact,fact)]={'ts':ts,'is_confidential':bool(is_confidential),'conf':float(confidence),'source':source_excerpt[:240]}
            self._slot.refit_due-=1
        if self._slot.refit_due<=0 and len(self._slot.lut._raw)>0:
            try:self._slot.lut.fit();self._slot.refit_due=_REFIT_EVERY
            except Exception as e:print(f'[PersonalAtlas] fit failed: {e}',flush=True)
        self._save_slot()
    def record_direct(self,fact:str,source_excerpt:str='',is_confidential:bool=True,confidence:float=1.0)->Dict[str,Any]:
        if not fact or not fact.strip():return {'recorded':False,'reason':'empty'}
        self._record(fact.strip(),source_excerpt,bool(is_confidential),float(confidence))
        return {'recorded':True,'is_confidential':bool(is_confidential)}
    def recall(self,query:str,k:int=5,max_radius:int=3,include_confidential:bool=True)->List[Dict[str,Any]]:
        if not query:return []
        if len(self._slot.lut._raw)==0 or self._slot.lut._stored_embs is None or self._slot.lut._pca_Vt is None:return []
        try:query_cell,_,_=self._slot.lut._project(query)
        except Exception:return []
        grid_cells=[(c,sum(abs(ci-qi) for ci,qi in zip(c,query_cell))) for c in self._slot.lut._cells.keys()]
        grid_cells.sort(key=lambda x:(x[1],x[0]))
        results=[];seen=set()
        for cell,radius in grid_cells:
            if radius>max_radius:break
            hit=self._slot.lut._cells.get(cell)
            if hit is None:continue
            fact=hit['q']
            if fact in seen:continue
            m=self._slot.meta.get((fact,fact),{})
            if m.get('is_confidential') and not include_confidential:continue
            seen.add(fact)
            results.append({'fact':fact,'is_confidential':bool(m.get('is_confidential',True)),'conf':m.get('conf',1.0),'cell_radius':radius,'ts':m.get('ts'),'source':m.get('source','')[:200]})
            if len(results)>=k:break
        return results
    def pending_clarifications(self,limit:int=3)->List[Dict[str,Any]]:
        return list(self._pending[:limit])
    def confirm_clarification(self,fact_id:str,is_confidential:bool)->Dict[str,Any]:
        idx=next((i for i,p in enumerate(self._pending) if p.get('id')==fact_id),-1)
        if idx<0:return {'confirmed':False,'reason':'not_found'}
        p=self._pending.pop(idx);self._save_pending()
        self._record(p['fact'],p.get('source_excerpt',''),bool(is_confidential),float(p.get('confidence',1.0)))
        return {'confirmed':True,'is_confidential':bool(is_confidential),'fact':p['fact']}
    def parse_clarification_reply(self,user_reply:str)->Optional[bool]:
        if not user_reply:return None
        yes=bool(_CONFIRM_YES_RE.search(user_reply));no=bool(_CONFIRM_NO_RE.search(user_reply))
        if yes and not no:return True
        if no and not yes:return False
        return None
    def build_clarification_question(self,pending:Dict[str,Any])->str:
        return (f"Quick check — earlier oui mentioned: \"{pending['fact']}\". Should I treat that as confidential (locked to this machine, never referenced in tool calls or shared), or is it regular context I can use freely? Just say \"confidential\" or \"public\".")
    def forget(self,fact_pattern:Optional[str]=None,forget_all:bool=False)->Dict[str,Any]:
        if forget_all:
            with self._slot.lock:
                self._slot=_Slot(SemanticPTEXLUT(grid=self.grid,pca_dim=self.pca_dim,encoder=self.encoder),{})
            self._save_slot()
            self._pending=[];self._save_pending()
            return {'forgot':'all'}
        if not fact_pattern:return {'forgot':'none','reason':'no_pattern'}
        pat=re.compile(fact_pattern,re.IGNORECASE)
        removed=[]
        with self._slot.lock:
            keep_meta={};keep_raw=[]
            for (q,a),m in self._slot.meta.items():
                if pat.search(q):removed.append(q);continue
                keep_meta[(q,a)]=m;keep_raw.append((q,a))
            self._slot.lut._raw=keep_raw
            self._slot.meta=keep_meta
            self._slot.lut._stored_embs=None;self._slot.lut._pca_Vt=None;self._slot.lut._cells={}
        if keep_raw:
            try:self._slot.lut.fit()
            except Exception:pass
        self._save_slot()
        return {'forgot':len(removed),'facts':removed[:10]}
    def stats(self)->Dict[str,Any]:
        n=len(self._slot.lut._raw)
        conf_n=sum(1 for m in self._slot.meta.values() if m.get('is_confidential'))
        return {'entries':n,'confidential':conf_n,'public':n-conf_n,'pending':len(self._pending),'queue_size':self._queue.qsize(),'unique_cells':len(self._slot.lut._cells)}
    def next_clarification_to_ask(self)->Optional[Dict[str,Any]]:
        if self._last_asked_id is not None and (time.time()-self._last_asked_ts)<self._ask_expire_s:return None
        if not self._pending:return None
        p=self._pending[0]
        self._last_asked_id=p['id'];self._last_asked_ts=time.time()
        return p
    def try_parse_pending_reply(self,user_message:str)->Optional[Dict[str,Any]]:
        if self._last_asked_id is None:return None
        if (time.time()-self._last_asked_ts)>=self._ask_expire_s:self._last_asked_id=None;return None
        decision=self.parse_clarification_reply(user_message)
        if decision is None:return None
        result=self.confirm_clarification(self._last_asked_id,decision)
        self._last_asked_id=None;self._last_asked_ts=0.0
        return result
    def shutdown(self,timeout:float=2.0):
        self._stop_event.set()
        if self._worker_thread is not None:
            try:self._worker_thread.join(timeout=timeout)
            except Exception:pass
