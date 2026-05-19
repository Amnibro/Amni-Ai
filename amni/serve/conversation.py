"""Conversation — JSONL-persisted multi-turn session with PII flagging.
Each turn: {ts, role, content, is_personal?, tier?, tokens?, skill_calls?}. Local-only by design; never federated."""
import json,re,time,uuid
from pathlib import Path
from typing import List,Dict,Any,Optional,Tuple
_CAP=1000
_PII_MARKERS=re.compile(r"\b(?:my\s+(?:name|email|e-?mail|phone|address|password|favorite|birthday|family|wife|husband|kid|son|daughter|partner|company|employer|salary|ssn|work\s+number|home\s+number|cell)|i\s+(?:am\s+(?:called|named)|live\s+at|work\s+at|am\s+from)|call\s+me|i'?m\s+a\s+(?:doctor|nurse|teacher|engineer|lawyer|programmer)|email:\s*\S+@\S+|phone:?\s*\+?\d)",re.IGNORECASE)
_PII_NAME=re.compile(r"(?:(?i:\bnamed)\s+[A-Z][A-Za-z]+\b)|(?:(?i:\bthis\s+is)\s+[A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]+)?(?:\s+(?:speaking|here|calling))?\b)|(?:(?i:\b(?:mr|mrs|ms|dr|prof|mister|missus|doctor|professor))\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\b)")
_PII_HARD=re.compile(r"\b[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}\b|\+?\d{1,3}[ -.]?\(?\d{3}\)?[ -.]?\d{3}[ -.]?\d{4}\b|\b\d{3}[ -.]\d{3}[ -.]\d{4}\b|\b\d{3}-\d{2}-\d{4}\b|\b\d{16}\b|\b(?:\d{4}[ -]?){3}\d{4}\b|sk-[A-Za-z0-9_-]{20,}|(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{36,}|AKIA[A-Z0-9]{16}|AIza[\w-]{35}|xox[bopa]-[A-Za-z0-9-]{20,}")
def detect_personal(text:str)->bool:
    if not text:return False
    return bool(_PII_MARKERS.search(text) or _PII_NAME.search(text) or _PII_HARD.search(text))
class Conversation:
    def __init__(self,session_id:str,path:Path):
        self.session_id=session_id;self.path=path;self.turns:List[Dict[str,Any]]=[]
        self.path.parent.mkdir(parents=True,exist_ok=True);self._load()
    def _load(self):
        if not self.path.exists():return
        try:
            with open(self.path,'r',encoding='utf-8') as f:
                for line in f:
                    line=line.strip()
                    if line:self.turns.append(json.loads(line))
        except Exception:pass
    def append(self,role:str,content:str,meta:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
        turn={'ts':time.time(),'role':role,'content':content,'is_personal':detect_personal(content) if role in ('user','assistant') else False}
        if meta:turn.update(meta)
        self.turns.append(turn)
        try:
            with open(self.path,'a',encoding='utf-8') as f:f.write(json.dumps(turn)+'\n')
        except Exception:pass
        if len(self.turns)>=_CAP:self._rotate()
        return turn
    def _rotate(self):
        archive=self.path.with_suffix(f'.{int(time.time())}.archived.jsonl')
        try:self.path.rename(archive)
        except Exception:return
        self.turns=self.turns[-100:]
        try:
            with open(self.path,'w',encoding='utf-8') as f:
                for t in self.turns:f.write(json.dumps(t)+'\n')
        except Exception:pass
    def recent(self,n:int=10)->List[Dict[str,Any]]:return self.turns[-n:]
    def transcript(self,n:int=10)->str:
        ts=self.recent(n)
        return '\n'.join(f'{t["role"].upper()}: {t["content"]}' for t in ts if t.get('role') in ('user','assistant'))
    def history_pairs(self,n:int=12,exclude_last_user:bool=True)->List[Tuple[str,str]]:
        ua=[t for t in self.turns if t.get('role') in ('user','assistant')]
        if exclude_last_user and ua and ua[-1].get('role')=='user':ua=ua[:-1]
        pairs:List[Tuple[str,str]]=[];pending_u:Optional[str]=None
        for t in ua:
            if t.get('role')=='user':pending_u=t.get('content','')
            elif t.get('role')=='assistant' and pending_u is not None:pairs.append((pending_u,t.get('content','')));pending_u=None
        return pairs[-n:]
    def has_personal(self,n:int=20)->bool:return any(t.get('is_personal') for t in self.recent(n) if t.get('role') in ('user','assistant'))
    def clear(self):
        self.turns=[]
        try:self.path.unlink(missing_ok=True)
        except Exception:pass
class ConversationStore:
    def __init__(self,root:str='experiences/conversations'):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self._active:Dict[str,Conversation]={}
    def get(self,session_id:Optional[str]=None)->Conversation:
        sid=session_id or uuid.uuid4().hex[:12]
        if sid in self._active:return self._active[sid]
        c=Conversation(sid,self.root/f'{sid}.jsonl');self._active[sid]=c;return c
    def list_sessions(self)->List[Dict[str,Any]]:
        out=[]
        for p in sorted(self.root.glob('*.jsonl'),key=lambda x:x.stat().st_mtime,reverse=True):
            if '.archived.' in p.name:continue
            out.append({'session_id':p.stem,'mtime':p.stat().st_mtime,'size':p.stat().st_size})
        return out
    def delete(self,session_id:str)->bool:
        p=self.root/f'{session_id}.jsonl';ok=p.exists();self._active.pop(session_id,None)
        try:p.unlink(missing_ok=True)
        except Exception:return False
        return ok
