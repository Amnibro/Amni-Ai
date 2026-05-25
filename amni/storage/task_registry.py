"""TaskRegistry — in-memory tracker for long-running Adam operations so the user can see what's running and cancel anything that's wasting time.
Active tasks (running) + ring-buffer of recent finished. Each task gets a unique id, optional progress 0-100, status text, cancel flag pollable by the worker.
Pure in-memory — dies on restart, which is correct: stale active tasks shouldn't survive a crash. The user's persistent stuff lives in the scheduler/learning daemon already."""
import time,uuid,threading
from typing import Dict,Any,List,Optional
_RECENT_KEEP=20
class TaskRegistry:
    def __init__(self,recent_keep:int=_RECENT_KEEP):
        self._lock=threading.Lock()
        self._active:Dict[str,Dict[str,Any]]={}
        self._recent:List[Dict[str,Any]]=[]
        self.recent_keep=recent_keep
    def register(self,kind:str,label:str='',total:Optional[int]=None,meta:Optional[Dict[str,Any]]=None)->str:
        tid=f't_{uuid.uuid4().hex[:12]}'
        now=time.time()
        t={'id':tid,'kind':kind,'label':label or kind,'status':'running','progress_pct':0,'progress_msg':'','started_at':now,'finished_at':None,'total':total,'done':0,'outcome':None,'error':None,'cancel_requested':False,'meta':meta or {}}
        with self._lock:self._active[tid]=t
        return tid
    def update(self,task_id:str,done:Optional[int]=None,total:Optional[int]=None,progress_pct:Optional[float]=None,message:str='',meta_patch:Optional[Dict[str,Any]]=None)->bool:
        with self._lock:
            t=self._active.get(task_id)
            if t is None:return False
            if done is not None:t['done']=int(done)
            if total is not None:t['total']=int(total)
            if progress_pct is not None:t['progress_pct']=int(max(0,min(100,progress_pct)))
            elif t.get('total'):t['progress_pct']=int(max(0,min(100,(int(t['done'])/int(t['total']))*100)))
            if message:t['progress_msg']=str(message)[:200]
            if meta_patch:t['meta'].update(meta_patch)
            return True
    def complete(self,task_id:str,outcome:Any=None)->bool:
        with self._lock:
            t=self._active.pop(task_id,None)
            if t is None:return False
            t['status']='done';t['finished_at']=time.time();t['progress_pct']=100;t['outcome']=outcome
            self._recent.insert(0,t);self._recent=self._recent[:self.recent_keep]
            return True
    def fail(self,task_id:str,error:str)->bool:
        with self._lock:
            t=self._active.pop(task_id,None)
            if t is None:return False
            t['status']='failed';t['finished_at']=time.time();t['error']=str(error)[:400]
            self._recent.insert(0,t);self._recent=self._recent[:self.recent_keep]
            return True
    def request_cancel(self,task_id:str)->bool:
        with self._lock:
            t=self._active.get(task_id)
            if t is None:return False
            t['cancel_requested']=True
            t['progress_msg']='(cancel requested)'
            return True
    def cancel_requested(self,task_id:str)->bool:
        with self._lock:
            t=self._active.get(task_id)
            return bool(t and t.get('cancel_requested'))
    def mark_cancelled(self,task_id:str)->bool:
        with self._lock:
            t=self._active.pop(task_id,None)
            if t is None:return False
            t['status']='cancelled';t['finished_at']=time.time()
            self._recent.insert(0,t);self._recent=self._recent[:self.recent_keep]
            return True
    def get(self,task_id:str)->Optional[Dict[str,Any]]:
        with self._lock:
            t=self._active.get(task_id)
            if t is not None:return dict(t)
            for r in self._recent:
                if r['id']==task_id:return dict(r)
            return None
    def list_active(self)->List[Dict[str,Any]]:
        with self._lock:return [dict(t) for t in self._active.values()]
    def list_recent(self,limit:int=10)->List[Dict[str,Any]]:
        with self._lock:return [dict(t) for t in self._recent[:limit]]
    def stats(self)->Dict[str,Any]:
        with self._lock:
            done=sum(1 for r in self._recent if r['status']=='done')
            failed=sum(1 for r in self._recent if r['status']=='failed')
            cancelled=sum(1 for r in self._recent if r['status']=='cancelled')
            return {'active':len(self._active),'recent_done':done,'recent_failed':failed,'recent_cancelled':cancelled,'total_recent':len(self._recent)}
