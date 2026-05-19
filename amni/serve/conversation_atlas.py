"""ConversationAtlas — PTEX cell-address LUT for chat turns.
Each (user_msg -> assistant_reply) becomes one cell in a SemanticPTEXLUT grid (encode -> PCA -> discretize -> cell). The cell IS the nonce IS the address. Lookup is hit/miss on cell, with adjacent-cell tolerance via lookup_soft. Per-session + global atlases; only non-personal entries feed the global atlas and the PRISM federation pump."""
import json,time,threading
from pathlib import Path
from typing import List,Dict,Any,Optional,Tuple
from amni.inference.semantic_ptex_lut import SemanticPTEXLUT
from amni.serve.conversation import detect_personal
_REFIT_EVERY=5
_GLOBAL_KEY='__global__'
class _AtlasSlot:
    __slots__=('lut','meta','refit_due','lock')
    def __init__(self,lut:SemanticPTEXLUT,meta:Dict[Tuple[str,str],Dict[str,Any]]):
        self.lut=lut;self.meta=meta;self.refit_due=0;self.lock=threading.Lock()
class ConversationAtlas:
    def __init__(self,root:str='experiences/conversation_atlas',grid:int=64,pca_dim:int=8,encoder=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.grid=grid;self.pca_dim=pca_dim;self.encoder=encoder
        self._slots:Dict[str,_AtlasSlot]={}
        self._load_all()
    def _slot_path(self,key:str)->Path:return self.root/f'atlas_{key}'
    def _meta_path(self,key:str)->Path:return self.root/f'atlas_{key}.meta.json'
    def _load_all(self):
        for mp in sorted(self.root.glob('atlas_*.meta.json')):
            key=mp.stem.removeprefix('atlas_').removesuffix('.meta')
            try:self._load_slot(key)
            except Exception as e:print(f'[ConversationAtlas] failed to load {key}: {e}',flush=True)
    def _load_slot(self,key:str):
        sp=self._slot_path(key);mp=self._meta_path(key)
        if not mp.exists():
            lut=SemanticPTEXLUT(grid=self.grid,pca_dim=self.pca_dim,encoder=self.encoder)
            self._slots[key]=_AtlasSlot(lut,{});return
        try:lut=SemanticPTEXLUT.load(str(sp),encoder=self.encoder)
        except Exception:lut=SemanticPTEXLUT(grid=self.grid,pca_dim=self.pca_dim,encoder=self.encoder)
        meta_raw=json.loads(mp.read_text(encoding='utf-8'))
        meta={tuple(k.split('::',1)):v for k,v in meta_raw.items()} if isinstance(meta_raw,dict) else {}
        self._slots[key]=_AtlasSlot(lut,meta)
    def _save_slot(self,key:str):
        slot=self._slots[key]
        with slot.lock:
            try:
                if len(slot.lut._raw)>0 and slot.lut._stored_embs is not None:slot.lut.save(str(self._slot_path(key)))
            except Exception as e:print(f'[ConversationAtlas] save lut {key} failed: {e}',flush=True)
            try:self._meta_path(key).write_text(json.dumps({'::'.join(k):v for k,v in slot.meta.items()},default=str),encoding='utf-8')
            except Exception as e:print(f'[ConversationAtlas] save meta {key} failed: {e}',flush=True)
    def _ensure(self,key:str)->_AtlasSlot:
        if key not in self._slots:self._slots[key]=_AtlasSlot(SemanticPTEXLUT(grid=self.grid,pca_dim=self.pca_dim,encoder=self.encoder),{})
        return self._slots[key]
    def _refit_if_due(self,slot:_AtlasSlot,force:bool=False):
        if (force or slot.refit_due<=0) and len(slot.lut._raw)>0:
            try:slot.lut.fit();slot.refit_due=_REFIT_EVERY
            except Exception as e:print(f'[ConversationAtlas] fit failed: {e}',flush=True)
    def record(self,session_id:str,user_msg:str,assistant_reply:str,is_personal:Optional[bool]=None,confidence:float=1.0)->Dict[str,Any]:
        if not user_msg or not assistant_reply:return {'recorded':False,'reason':'empty'}
        is_personal=detect_personal(user_msg) or detect_personal(assistant_reply) if is_personal is None else bool(is_personal)
        ts=time.time();added=[]
        for key in (session_id,) if is_personal else (session_id,_GLOBAL_KEY):
            slot=self._ensure(key)
            with slot.lock:
                slot.lut.add(user_msg,assistant_reply)
                slot.meta[(user_msg,assistant_reply)]={'ts':ts,'is_personal':is_personal,'conf':float(confidence),'session_id':session_id}
                slot.refit_due-=1
            self._refit_if_due(slot)
            self._save_slot(key);added.append(key)
        return {'recorded':True,'is_personal':is_personal,'slots':added}
    def recall(self,query:str,session_id:str,k:int=3,include_global:bool=True,max_radius:int=3)->List[Dict[str,Any]]:
        if not query:return []
        results:List[Dict[str,Any]]=[];seen:set=set()
        scopes=[session_id]+([_GLOBAL_KEY] if include_global else [])
        for key in scopes:
            slot=self._slots.get(key)
            if slot is None or len(slot.lut._raw)==0 or slot.lut._stored_embs is None or slot.lut._pca_Vt is None:continue
            try:query_cell,_,_=slot.lut._project(query)
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
                results.append({'user':q,'assistant':a,'scope':key,'cell_radius':radius,'is_personal':m.get('is_personal',False),'ts':m.get('ts')})
                if len(results)>=k*len(scopes):break
        results.sort(key=lambda r:(r.get('cell_radius',9999),-(r.get('ts') or 0)))
        return results[:k]
    def history_pairs(self,session_id:str,n:int=12)->List[Tuple[str,str]]:
        slot=self._slots.get(session_id)
        if slot is None:return []
        items=sorted(slot.meta.items(),key=lambda kv:kv[1].get('ts',0))
        return [(k[0],k[1]) for k,_ in items[-n:]]
    def federation_pull(self,min_confidence:float=0.8,limit:int=200)->List[Dict[str,Any]]:
        out=[];slot=self._slots.get(_GLOBAL_KEY)
        if slot is None:return out
        for (q,a),m in slot.meta.items():
            if m.get('is_personal'):continue
            if m.get('conf',1.0)<min_confidence:continue
            out.append({'q':q,'a':a,'conf':m.get('conf'),'ts':m.get('ts'),'session_id':m.get('session_id')})
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
