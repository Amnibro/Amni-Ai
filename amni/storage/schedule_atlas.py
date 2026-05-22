"""ScheduleAtlas — persistent job store for Adam-driven recurring tasks.
One JSONL `jobs.jsonl` (full job rewrite on save), one `outcomes.jsonl` (append-only, capped per-job in memory). Job shape:
  {id, kind:'skill'|'prompt'|'webpoll', payload, cadence_s, next_fire_at, last_fired_at, runs, enabled, label, created_ts}
Thread-safe via a single lock; reads/writes serialized."""
import json,time,uuid,threading
from pathlib import Path
from typing import List,Dict,Any,Optional
_OUTCOME_KEEP=5
class ScheduleAtlas:
    def __init__(self,root:str='experiences/scheduler'):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self._jobs_path=self.root/'jobs.jsonl'
        self._outcomes_path=self.root/'outcomes.jsonl'
        self._lock=threading.Lock()
        self._jobs:Dict[str,Dict[str,Any]]={}
        self._outcomes:Dict[str,List[Dict[str,Any]]]={}
        self._load()
    def _load(self):
        if self._jobs_path.exists():
            try:
                for ln in self._jobs_path.read_text(encoding='utf-8').strip().splitlines():
                    if ln.strip():
                        try:j=json.loads(ln)
                        except Exception:continue
                        if 'id' in j:self._jobs[j['id']]=j
            except Exception as e:print(f'[ScheduleAtlas] load jobs failed: {e}',flush=True)
        if self._outcomes_path.exists():
            try:
                for ln in self._outcomes_path.read_text(encoding='utf-8').strip().splitlines():
                    if ln.strip():
                        try:o=json.loads(ln)
                        except Exception:continue
                        jid=o.get('job_id')
                        if jid:self._outcomes.setdefault(jid,[]).append(o)
                for jid in list(self._outcomes.keys()):self._outcomes[jid]=self._outcomes[jid][-_OUTCOME_KEEP:]
            except Exception as e:print(f'[ScheduleAtlas] load outcomes failed: {e}',flush=True)
    def _save_jobs(self):
        try:
            tmp=self._jobs_path.with_suffix('.jsonl.tmp')
            with tmp.open('w',encoding='utf-8') as f:
                for j in self._jobs.values():f.write(json.dumps(j,default=str)+'\n')
            tmp.replace(self._jobs_path)
        except Exception as e:print(f'[ScheduleAtlas] save jobs failed: {e}',flush=True)
    def _append_outcome(self,outcome:Dict[str,Any]):
        try:
            with self._outcomes_path.open('a',encoding='utf-8') as f:f.write(json.dumps(outcome,default=str)+'\n')
        except Exception as e:print(f'[ScheduleAtlas] append outcome failed: {e}',flush=True)
    def add(self,kind:str,payload:Any,cadence_s:int,label:str='',start_in_s:Optional[int]=None,enabled:bool=True)->Dict[str,Any]:
        if kind not in ('skill','prompt','webpoll'):return {'error':f'unknown kind "{kind}"'}
        if int(cadence_s)<10:return {'error':'cadence_s must be >= 10'}
        jid=f'job_{uuid.uuid4().hex[:12]}'
        now=time.time();start_in=int(start_in_s) if start_in_s is not None else 0
        job={'id':jid,'kind':kind,'payload':payload,'cadence_s':int(cadence_s),'next_fire_at':now+start_in,'last_fired_at':0.0,'runs':0,'enabled':bool(enabled),'label':label or '','created_ts':now}
        with self._lock:self._jobs[jid]=job;self._save_jobs()
        return {'id':jid,'job':job}
    def cancel(self,job_id:str)->bool:
        with self._lock:
            if job_id not in self._jobs:return False
            self._jobs.pop(job_id);self._save_jobs();return True
    def enable(self,job_id:str,enabled:bool=True)->bool:
        with self._lock:
            j=self._jobs.get(job_id)
            if j is None:return False
            j['enabled']=bool(enabled);self._save_jobs();return True
    def list_jobs(self)->List[Dict[str,Any]]:
        with self._lock:return [dict(j) for j in self._jobs.values()]
    def get(self,job_id:str)->Optional[Dict[str,Any]]:
        with self._lock:j=self._jobs.get(job_id);return dict(j) if j else None
    def outcomes(self,job_id:str)->List[Dict[str,Any]]:
        with self._lock:return list(self._outcomes.get(job_id,[]))
    def due_jobs(self,now:Optional[float]=None)->List[Dict[str,Any]]:
        now=now if now is not None else time.time()
        with self._lock:return [dict(j) for j in self._jobs.values() if j.get('enabled') and j.get('next_fire_at',0)<=now]
    def record_fire(self,job_id:str,outcome:Dict[str,Any])->bool:
        now=time.time()
        with self._lock:
            j=self._jobs.get(job_id)
            if j is None:return False
            j['last_fired_at']=now;j['runs']=int(j.get('runs',0))+1;j['next_fire_at']=now+int(j.get('cadence_s',60))
            self._save_jobs()
            entry={'job_id':job_id,'ts':now,'outcome':outcome}
            self._outcomes.setdefault(job_id,[]).append(entry)
            self._outcomes[job_id]=self._outcomes[job_id][-_OUTCOME_KEEP:]
        self._append_outcome(entry)
        return True
    def stats(self)->Dict[str,Any]:
        with self._lock:
            jobs=list(self._jobs.values())
            return {'n_jobs':len(jobs),'enabled':sum(1 for j in jobs if j.get('enabled')),'disabled':sum(1 for j in jobs if not j.get('enabled')),'total_runs':sum(int(j.get('runs',0)) for j in jobs),'kinds':{k:sum(1 for j in jobs if j.get('kind')==k) for k in ('skill','prompt','webpoll')}}
