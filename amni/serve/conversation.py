"""Conversation — JSONL-persisted multi-turn session, capped + auto-rotated.
Each turn: {ts, role, content, tier?, tokens?, skill_calls?}. Roles: user/assistant/system/skill."""
import json,time,uuid
from pathlib import Path
from typing import List,Dict,Any,Optional
_CAP=1000
class Conversation:
    def __init__(self,session_id:str,path:Path):
        self.session_id=session_id
        self.path=path
        self.turns:List[Dict[str,Any]]=[]
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self._load()
    def _load(self):
        if not self.path.exists():return
        try:
            with open(self.path,'r',encoding='utf-8') as f:
                for line in f:
                    line=line.strip()
                    if line:self.turns.append(json.loads(line))
        except Exception:pass
    def append(self,role:str,content:str,meta:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
        turn={'ts':time.time(),'role':role,'content':content}
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
    def clear(self):
        self.turns=[]
        try:self.path.unlink(missing_ok=True)
        except Exception:pass
class ConversationStore:
    def __init__(self,root:str='experiences/conversations'):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self._active:Dict[str,Conversation]={}
    def get(self,session_id:Optional[str]=None)->Conversation:
        sid=session_id or uuid.uuid4().hex[:12]
        if sid in self._active:return self._active[sid]
        c=Conversation(sid,self.root/f'{sid}.jsonl')
        self._active[sid]=c
        return c
    def list_sessions(self)->List[Dict[str,Any]]:
        out=[]
        for p in sorted(self.root.glob('*.jsonl'),key=lambda x:x.stat().st_mtime,reverse=True):
            if '.archived.' in p.name:continue
            out.append({'session_id':p.stem,'mtime':p.stat().st_mtime,'size':p.stat().st_size})
        return out
    def delete(self,session_id:str)->bool:
        p=self.root/f'{session_id}.jsonl'
        ok=p.exists()
        self._active.pop(session_id,None)
        try:p.unlink(missing_ok=True)
        except Exception:return False
        return ok
