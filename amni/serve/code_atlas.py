"""CodeAtlas — PTEX cell-address LUT for (coding-intent → tool-sequence) memory.
Mirrors ConversationAtlas but stores tool sequences and outcomes instead of chat pairs. Cell-address lookup with L1 grid scan, never cosine-top-K. Per-session + __global__ slots; only non-personal entries flow to __global__ and the PRISM federation pump.
record(session_id, intent, tool_seq, outcome, is_personal) — writes to session slot and global if shareable.
recall(intent, k, max_radius)                              — neighbor-cell scan, returns prior tool sequences for similar tasks.
federation_pull(min_conf, limit)                            — federable subset for cross-Adam sharing."""
import json,time,threading
from pathlib import Path
from typing import List,Dict,Any,Optional,Tuple
from amni.inference.semantic_ptex_lut import SemanticPTEXLUT
from amni.serve.conversation import detect_personal
_REFIT_EVERY=5
_GLOBAL_KEY='__global__'
_LOCAL_USER_KEY='__local_user__'
class _AtlasSlot:
    __slots__=('lut','meta','refit_due','lock')
    def __init__(self,lut:SemanticPTEXLUT,meta:Dict[Tuple[str,str],Dict[str,Any]]):
        self.lut=lut;self.meta=meta;self.refit_due=0;self.lock=threading.Lock()
def _seq_signature(tool_seq:List[Dict[str,Any]])->str:
    return ' → '.join((c.get('name') or '?') for c in (tool_seq or [])) or '∅'
def _intent_contains_pii(intent:str,tool_seq:List[Dict[str,Any]])->bool:
    if detect_personal(intent or ''):return True
    for c in tool_seq or []:
        args=c.get('arguments') or {}
        if isinstance(args,str):
            if detect_personal(args):return True
        elif isinstance(args,dict):
            for v in args.values():
                if isinstance(v,str) and detect_personal(v):return True
    return False
class CodeAtlas:
    def __init__(self,root:str='experiences/code_atlas',grid:int=64,pca_dim:int=8,encoder=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.grid=grid;self.pca_dim=pca_dim;self.encoder=encoder
        self._slots:Dict[str,_AtlasSlot]={}
        self._load_all()
    def _slot_path(self,key:str)->Path:return self.root/f'code_{key}'
    def _meta_path(self,key:str)->Path:return self.root/f'code_{key}.meta.json'
    def _load_all(self):
        for mp in sorted(self.root.glob('code_*.meta.json')):
            key=mp.stem.removeprefix('code_').removesuffix('.meta')
            try:self._load_slot(key)
            except Exception as e:print(f'[CodeAtlas] failed to load {key}: {e}',flush=True)
    def _load_slot(self,key:str):
        sp=self._slot_path(key);mp=self._meta_path(key)
        if not mp.exists():
            self._slots[key]=_AtlasSlot(SemanticPTEXLUT(grid=self.grid,pca_dim=self.pca_dim,encoder=self.encoder),{});return
        try:lut=SemanticPTEXLUT.load(str(sp),encoder=self.encoder)
        except Exception:lut=SemanticPTEXLUT(grid=self.grid,pca_dim=self.pca_dim,encoder=self.encoder)
        meta_raw=json.loads(mp.read_text(encoding='utf-8'))
        meta={tuple(k.split('::',1)):v for k,v in meta_raw.items()} if isinstance(meta_raw,dict) else {}
        self._slots[key]=_AtlasSlot(lut,meta)
    def _save_slot(self,key:str):
        slot=self._slots[key]
        with slot.lock:
            try:
                _embs=slot.lut._stored_embs;_n_raw=len(slot.lut._raw)
                if _n_raw>0 and (_embs is None or len(_embs)!=_n_raw):
                    try:slot.lut.fit()
                    except Exception as fe:print(f'[CodeAtlas] pre-save refit {key} failed: {fe}',flush=True)
                if _n_raw>0 and slot.lut._stored_embs is not None:slot.lut.save(str(self._slot_path(key)))
            except Exception as e:print(f'[CodeAtlas] save lut {key} failed: {e}',flush=True)
            try:self._meta_path(key).write_text(json.dumps({'::'.join(k):v for k,v in slot.meta.items()},default=str),encoding='utf-8')
            except Exception as e:print(f'[CodeAtlas] save meta {key} failed: {e}',flush=True)
    def _ensure(self,key:str)->_AtlasSlot:
        if key not in self._slots:self._slots[key]=_AtlasSlot(SemanticPTEXLUT(grid=self.grid,pca_dim=self.pca_dim,encoder=self.encoder),{})
        return self._slots[key]
    def _refit_if_due(self,slot:_AtlasSlot,force:bool=False):
        if (force or slot.refit_due<=0) and len(slot.lut._raw)>0:
            try:slot.lut.fit();slot.refit_due=_REFIT_EVERY
            except Exception as e:print(f'[CodeAtlas] fit failed: {e}',flush=True)
    def record(self,session_id:str,intent:str,tool_seq:List[Dict[str,Any]],outcome:str='',is_personal:Optional[bool]=None,confidence:float=1.0)->Dict[str,Any]:
        if not intent:return {'recorded':False,'reason':'empty_intent'}
        sig=_seq_signature(tool_seq)
        is_personal=_intent_contains_pii(intent,tool_seq) if is_personal is None else bool(is_personal)
        ts=time.time();added=[]
        for key in (session_id,_LOCAL_USER_KEY) if is_personal else (session_id,_GLOBAL_KEY):
            slot=self._ensure(key)
            with slot.lock:
                slot.lut.add(intent,sig)
                slot.meta[(intent,sig)]={'ts':ts,'is_personal':is_personal,'conf':float(confidence),'session_id':session_id,'tool_seq':[{'name':c.get('name'),'arguments':c.get('arguments')} for c in (tool_seq or [])],'outcome':(outcome or '')[:800]}
                slot.refit_due-=1
            self._refit_if_due(slot)
            self._save_slot(key);added.append(key)
        return {'recorded':True,'is_personal':is_personal,'slots':added,'signature':sig}
    def recall(self,intent:str,session_id:Optional[str]=None,k:int=3,include_global:bool=True,include_local:bool=True,max_radius:int=3)->List[Dict[str,Any]]:
        if not intent:return []
        results:List[Dict[str,Any]]=[];seen:set=set()
        scopes=([session_id] if session_id else [])+([_LOCAL_USER_KEY] if include_local else [])+([_GLOBAL_KEY] if include_global else [])
        for key in scopes:
            slot=self._slots.get(key)
            if slot is None or len(slot.lut._raw)==0 or slot.lut._stored_embs is None or slot.lut._pca_Vt is None:continue
            try:query_cell,_,_=slot.lut._project(intent)
            except Exception:continue
            grid_cells=[(c,sum(abs(ci-qi) for ci,qi in zip(c,query_cell))) for c in slot.lut._cells.keys()]
            grid_cells.sort(key=lambda x:(x[1],x[0]))
            for cell,radius in grid_cells:
                if radius>max_radius:break
                hit=slot.lut._cells.get(cell)
                if hit is None:continue
                q=hit['q'];a=hit['a']
                if (q,a) in seen:continue
                seen.add((q,a));m=slot.meta.get((q,a),{})
                results.append({'intent':q,'signature':a,'tool_seq':m.get('tool_seq',[]),'outcome':m.get('outcome',''),'scope':key,'cell_radius':radius,'is_personal':m.get('is_personal',False),'ts':m.get('ts'),'conf':m.get('conf',1.0)})
                if len(results)>=k*len(scopes):break
        results.sort(key=lambda r:(r.get('cell_radius',9999),-(r.get('ts') or 0)))
        return results[:k]
    def hint_for_prompt(self,intent:str,session_id:Optional[str]=None,k:int=2)->str:
        hits=self.recall(intent,session_id=session_id,k=k)
        if not hits:return ''
        lines=['[Prior similar tasks I solved — tool sequences that worked, use as a hint, not a constraint]']
        for h in hits:lines.append(f"- intent: \"{h['intent'][:120]}\"\n  tool_seq: {h['signature']}\n  outcome: {h['outcome'][:200]}")
        return '\n'.join(lines)
    def federation_pull(self,min_confidence:float=0.8,limit:int=200)->List[Dict[str,Any]]:
        out=[];slot=self._slots.get(_GLOBAL_KEY)
        if slot is None:return out
        for (q,a),m in slot.meta.items():
            if m.get('is_personal'):continue
            if m.get('conf',1.0)<min_confidence:continue
            out.append({'intent':q,'signature':a,'tool_seq':m.get('tool_seq',[]),'outcome':m.get('outcome',''),'conf':m.get('conf'),'ts':m.get('ts'),'session_id':m.get('session_id')})
            if len(out)>=limit:break
        return out
    def stats(self)->Dict[str,Any]:
        out={}
        for key,slot in self._slots.items():
            n=len(slot.lut._raw);priv=sum(1 for v in slot.meta.values() if v.get('is_personal'))
            out[key]={'entries':n,'personal':priv,'federable':n-priv,'unique_cells':len(slot.lut._cells)}
        return out
    def forget_session(self,session_id:str)->bool:
        if session_id not in self._slots:return False
        self._slots.pop(session_id,None)
        for p in (self._slot_path(session_id).with_suffix('.npz'),self._slot_path(session_id).with_suffix('.json'),self._meta_path(session_id)):
            try:p.unlink(missing_ok=True)
            except Exception:pass
        return True
