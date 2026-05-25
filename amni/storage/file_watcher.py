"""FileWatcher — stdlib polling-based file/folder change watcher. No watchdog dependency.
Single daemon thread polls all registered watches every poll_s seconds (default 5). For each watch: walk the path, build {relpath: (mtime, size)} snapshot, diff against the prior snapshot, emit created/modified/deleted events. Each watch keeps a ring-buffer of recent events.
Optional on_change_action: when an event fires, invoke a named skill via the skill_registry with merged args. Lets users wire up "when X changes, run Y" workflows from the chat or scheduler."""
import os,time,uuid,threading,fnmatch
from pathlib import Path
from typing import Dict,Any,List,Optional,Tuple,Callable
_EVENT_KEEP=50
_DEFAULT_POLL_S=5.0
_MAX_FILES_PER_WATCH=2000
class FileWatcher:
    def __init__(self,skill_registry=None,adam=None,poll_s:float=_DEFAULT_POLL_S,start_thread:bool=True):
        self.skills=skill_registry;self.adam=adam;self.poll_s=float(poll_s)
        self._lock=threading.Lock()
        self._watches:Dict[str,Dict[str,Any]]={}
        self._stop=threading.Event()
        self._thread=None
        if start_thread:self._start()
    def _start(self):
        if self._thread is not None and self._thread.is_alive():return
        self._thread=threading.Thread(target=self._loop,name='FileWatcher',daemon=True);self._thread.start()
    def shutdown(self,timeout:float=2.0):
        self._stop.set()
        if self._thread is not None:
            try:self._thread.join(timeout=timeout)
            except Exception:pass
    def add(self,path:str,glob:str='*',recursive:bool=True,label:str='',on_change_skill:Optional[str]=None,on_change_args:Optional[Dict[str,Any]]=None,coalesce_s:float=2.0)->Dict[str,Any]:
        p=Path(path).expanduser()
        if not p.exists():return {'error':f'path not found: {path}'}
        wid=f'w_{uuid.uuid4().hex[:12]}'
        w={'id':wid,'path':str(p.resolve()),'glob':glob or '*','recursive':bool(recursive),'label':label or str(p.name) or str(p),'on_change_skill':on_change_skill or '','on_change_args':on_change_args or {},'coalesce_s':float(coalesce_s),'created_ts':time.time(),'last_poll_ts':0.0,'snapshot':{},'events':[],'is_dir':p.is_dir(),'enabled':True,'last_fired_at':0.0,'fire_count':0}
        with self._lock:
            self._snapshot(w)
            w['initialized']=True
            self._watches[wid]=w
        return {'id':wid,'watch':{k:v for k,v in w.items() if k!='snapshot'}}
    def cancel(self,watch_id:str)->bool:
        with self._lock:return self._watches.pop(watch_id,None) is not None
    def enable(self,watch_id:str,enabled:bool=True)->bool:
        with self._lock:
            w=self._watches.get(watch_id)
            if w is None:return False
            w['enabled']=bool(enabled);return True
    def list_watches(self)->List[Dict[str,Any]]:
        with self._lock:return [{k:v for k,v in w.items() if k!='snapshot'} for w in self._watches.values()]
    def get(self,watch_id:str)->Optional[Dict[str,Any]]:
        with self._lock:
            w=self._watches.get(watch_id)
            return {k:v for k,v in w.items() if k!='snapshot'} if w else None
    def events(self,watch_id:str,limit:int=20)->List[Dict[str,Any]]:
        with self._lock:
            w=self._watches.get(watch_id)
            return list((w or {}).get('events',[]))[-int(limit):]
    def _snapshot(self,w:Dict[str,Any])->Dict[str,Tuple[float,int]]:
        root=Path(w['path']);glob=w['glob'];recursive=w['recursive']
        snap:Dict[str,Tuple[float,int]]={}
        try:
            if root.is_file():
                try:st=root.stat();snap[str(root.name)]=(st.st_mtime,st.st_size)
                except Exception:pass
            else:
                iterator=root.rglob(glob) if recursive else root.glob(glob)
                count=0
                for item in iterator:
                    if count>=_MAX_FILES_PER_WATCH:break
                    try:
                        if not item.is_file():continue
                        st=item.stat()
                        snap[str(item.relative_to(root))]=(st.st_mtime,st.st_size)
                        count+=1
                    except Exception:continue
        except Exception as e:print(f'[FileWatcher] snapshot {w["id"]} failed: {e}',flush=True)
        w['snapshot']=snap
        return snap
    def _poll_one(self,w:Dict[str,Any])->List[Dict[str,Any]]:
        if not w.get('enabled'):return []
        old=w.get('snapshot',{});new=self._snapshot(w);events=[];ts=time.time()
        for relpath,(mtime,size) in new.items():
            if relpath not in old:events.append({'kind':'created','path':relpath,'mtime':mtime,'size':size,'ts':ts})
            else:
                old_mt,old_sz=old[relpath]
                if abs(mtime-old_mt)>0.5 or size!=old_sz:events.append({'kind':'modified','path':relpath,'mtime':mtime,'size':size,'ts':ts})
        for relpath in old:
            if relpath not in new:events.append({'kind':'deleted','path':relpath,'ts':ts})
        if events:
            ring=w.get('events',[]);ring.extend(events);w['events']=ring[-_EVENT_KEEP:]
            w['last_poll_ts']=ts
            now=ts
            if w.get('on_change_skill') and self.skills is not None and (now-w.get('last_fired_at',0))>=w.get('coalesce_s',2.0):
                try:
                    args=dict(w.get('on_change_args') or {})
                    args.setdefault('watch_id',w['id']);args.setdefault('events_n',len(events));args.setdefault('first_change',events[0])
                    self.skills.call(w['on_change_skill'],args,ctx={'adam':self.adam,'file_watcher':self})
                    w['last_fired_at']=now;w['fire_count']=int(w.get('fire_count',0))+1
                except Exception as e:print(f'[FileWatcher] on_change skill {w["on_change_skill"]} failed: {e}',flush=True)
        else:w['last_poll_ts']=ts
        return events
    def tick(self)->int:
        fired=0
        with self._lock:watches=list(self._watches.values())
        for w in watches:
            ev=self._poll_one(w)
            if ev:fired+=1
        return fired
    def _loop(self):
        while not self._stop.is_set():
            try:self.tick()
            except Exception as e:print(f'[FileWatcher] loop exception: {e}',flush=True)
            self._stop.wait(self.poll_s)
    def stats(self)->Dict[str,Any]:
        with self._lock:
            ws=list(self._watches.values())
            total_events=sum(len(w.get('events',[])) for w in ws)
            return {'n_watches':len(ws),'enabled':sum(1 for w in ws if w.get('enabled')),'total_recent_events':total_events,'auto_fires':sum(int(w.get('fire_count',0)) for w in ws)}
def watch_skill(args:Dict[str,Any],ctx:Dict[str,Any],reg)->Dict[str,Any]:
    fw=ctx.get('file_watcher') if ctx else None
    if fw is None:return {'error':'FileWatcher not in skill context'}
    action=(args.get('action') or '').strip().lower()
    if action in ('add','create','watch'):
        path=(args.get('path') or '').strip()
        if not path:return {'error':'need path'}
        return fw.add(path=path,glob=args.get('glob') or '*',recursive=bool(args.get('recursive',True)),label=args.get('label',''),on_change_skill=args.get('on_change_skill'),on_change_args=args.get('on_change_args') or {},coalesce_s=float(args.get('coalesce_s',2.0)))
    if action=='list':return {'watches':fw.list_watches(),'stats':fw.stats()}
    if action=='get':
        wid=args.get('id') or args.get('watch_id')
        if not wid:return {'error':'need id'}
        w=fw.get(wid);return {'watch':w} if w else {'error':f'no watch {wid}'}
    if action in ('cancel','remove','delete'):
        wid=args.get('id') or args.get('watch_id')
        if not wid:return {'error':'need id'}
        return {'cancelled':fw.cancel(wid),'id':wid}
    if action=='enable':
        wid=args.get('id') or args.get('watch_id')
        return {'updated':fw.enable(wid,True) if wid else False,'id':wid,'enabled':True}
    if action=='disable':
        wid=args.get('id') or args.get('watch_id')
        return {'updated':fw.enable(wid,False) if wid else False,'id':wid,'enabled':False}
    if action=='events':
        wid=args.get('id') or args.get('watch_id')
        if not wid:return {'error':'need id'}
        evs=fw.events(wid,limit=int(args.get('limit',20)))
        return {'id':wid,'events':evs,'n':len(evs),'widget':{'type':'watch','title':f'Watch · {wid[-6:]}','icon':'📁','data':{'watch_id':wid,'events':evs[-12:],'n':len(evs)}}}
    if action=='tick':return {'fired':fw.tick()}
    if action=='stats':
        s=fw.stats();return {**s,'widget':{'type':'watch','title':'File watchers','icon':'📁','data':s}}
    return {'error':f'unknown action "{action}"; valid: add|list|get|cancel|enable|disable|events|tick|stats'}
