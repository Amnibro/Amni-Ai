"""AdamScheduler — background thread that fires Adam's own recurring jobs.
Pulls from ScheduleAtlas, runs due jobs via the skill registry or chat_persona. Outcomes capped per job. schedule_loop skill is the user-facing surface: add/list/cancel/enable/disable/runs/run_now.
Job kinds:
  skill:    payload = {'name':'<skill>', 'args':{...}}  — invokes reg.call(name, args, ctx={adam,...})
  prompt:   payload = {'text':'...', 'system':'...optional'}  — adam.chat_persona, output stored in outcome
  webpoll:  payload = {'url':'https://...', 'query':'optional'}  — uses 'web' skill to fetch+distill
"""
import time,threading,traceback
from typing import Dict,Any,Optional
from amni.storage.schedule_atlas import ScheduleAtlas
_POLL_S=5.0
class AdamScheduler:
    def __init__(self,atlas:Optional[ScheduleAtlas]=None,skill_registry=None,adam=None,start_thread:bool=True,poll_s:float=_POLL_S):
        self.atlas=atlas or ScheduleAtlas()
        self.skills=skill_registry
        self.adam=adam
        self.poll_s=float(poll_s)
        self._stop=threading.Event()
        self._thread=None
        self._tick_lock=threading.Lock()
        if start_thread:self._start()
    def _start(self):
        if self._thread is not None and self._thread.is_alive():return
        self._thread=threading.Thread(target=self._loop,name='AdamScheduler',daemon=True)
        self._thread.start()
    def shutdown(self,timeout:float=2.0):
        self._stop.set()
        if self._thread is not None:
            try:self._thread.join(timeout=timeout)
            except Exception:pass
    def _loop(self):
        while not self._stop.is_set():
            try:self.tick()
            except Exception as e:print(f'[AdamScheduler] tick exception: {e}',flush=True)
            self._stop.wait(self.poll_s)
    def tick(self,now:Optional[float]=None)->int:
        with self._tick_lock:
            due=self.atlas.due_jobs(now=now)
            fired=0
            for j in due:
                outcome=self._fire(j)
                self.atlas.record_fire(j['id'],outcome)
                fired+=1
            return fired
    def _fire(self,job:Dict[str,Any])->Dict[str,Any]:
        t0=time.time()
        try:
            kind=job.get('kind');payload=job.get('payload') or {}
            if kind=='skill':
                name=payload.get('name');args=payload.get('args') or {}
                if not name or self.skills is None:return {'ok':False,'error':'skill registry not available or missing name','wall_s':round(time.time()-t0,3)}
                r=self.skills.call(name,args,ctx={'adam':self.adam,'scheduler':self})
                return {'ok':bool(r.ok),'tier':'skill','skill':name,'output_preview':(str(r.output)[:200] if r.output is not None else ''),'error':(r.error or '') if not r.ok else '','wall_s':round(time.time()-t0,3)}
            if kind=='prompt':
                if self.adam is None or not hasattr(self.adam,'chat_persona'):return {'ok':False,'error':'adam not available','wall_s':round(time.time()-t0,3)}
                text=payload.get('text') or '';system=payload.get('system') or 'You are Adam, summarize concisely.'
                r=self.adam.chat_persona(text,system=system,max_new_tokens=400,do_sample=False)
                ans=(r or {}).get('answer','') if isinstance(r,dict) else ''
                return {'ok':bool(ans),'tier':'prompt','output_preview':ans[:400],'wall_s':round(time.time()-t0,3)}
            if kind=='webpoll':
                if self.skills is None or not self.skills.has('web'):return {'ok':False,'error':'web skill not available','wall_s':round(time.time()-t0,3)}
                q=payload.get('query') or payload.get('url') or ''
                r=self.skills.call('web',{'query':q},ctx={'adam':self.adam})
                preview=(r.output or {}).get('answer','')[:300] if r.ok else ''
                return {'ok':bool(r.ok),'tier':'webpoll','output_preview':preview,'error':(r.error or '') if not r.ok else '','wall_s':round(time.time()-t0,3)}
            return {'ok':False,'error':f'unknown kind {kind}','wall_s':round(time.time()-t0,3)}
        except Exception as e:return {'ok':False,'error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc()[:500],'wall_s':round(time.time()-t0,3)}
    def run_now(self,job_id:str)->Dict[str,Any]:
        j=self.atlas.get(job_id)
        if j is None:return {'error':f'no job {job_id}'}
        outcome=self._fire(j)
        self.atlas.record_fire(job_id,outcome)
        return {'fired':True,'outcome':outcome}
def schedule_loop_skill(args:Dict[str,Any],ctx:Dict[str,Any],reg)->Dict[str,Any]:
    sched=ctx.get('scheduler') if ctx else None
    if sched is None:return {'error':'AdamScheduler not in skill context'}
    action=(args.get('action') or '').strip().lower()
    if action in ('add','create','new'):
        kind=(args.get('kind') or '').strip().lower()
        payload=args.get('payload')
        cadence_s=int(args.get('cadence_s') or 0)
        label=str(args.get('label') or '')
        start_in_s=args.get('start_in_s')
        if not kind:return {'error':'need kind: skill|prompt|webpoll'}
        if not payload:return {'error':'need payload (depends on kind)'}
        if not cadence_s:return {'error':'need cadence_s (seconds between fires, >= 10)'}
        return sched.atlas.add(kind,payload,cadence_s,label=label,start_in_s=start_in_s)
    if action=='list':
        return {'jobs':sched.atlas.list_jobs(),'stats':sched.atlas.stats()}
    if action=='get':
        jid=args.get('id') or args.get('job_id')
        if not jid:return {'error':'need id/job_id'}
        j=sched.atlas.get(jid);return {'job':j} if j else {'error':f'no job {jid}'}
    if action in ('cancel','delete','remove'):
        jid=args.get('id') or args.get('job_id')
        if not jid:return {'error':'need id/job_id'}
        return {'cancelled':sched.atlas.cancel(jid),'id':jid}
    if action=='enable':
        jid=args.get('id') or args.get('job_id')
        if not jid:return {'error':'need id/job_id'}
        return {'updated':sched.atlas.enable(jid,True),'id':jid,'enabled':True}
    if action=='disable':
        jid=args.get('id') or args.get('job_id')
        if not jid:return {'error':'need id/job_id'}
        return {'updated':sched.atlas.enable(jid,False),'id':jid,'enabled':False}
    if action in ('runs','outcomes','history'):
        jid=args.get('id') or args.get('job_id')
        if not jid:return {'error':'need id/job_id'}
        return {'id':jid,'outcomes':sched.atlas.outcomes(jid)}
    if action=='run_now':
        jid=args.get('id') or args.get('job_id')
        if not jid:return {'error':'need id/job_id'}
        return sched.run_now(jid)
    if action=='stats':return sched.atlas.stats()
    return {'error':f'unknown action "{action}"; valid: add|list|get|cancel|enable|disable|runs|run_now|stats'}
