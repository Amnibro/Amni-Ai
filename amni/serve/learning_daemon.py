"""LearningDaemon — Adam's 24/7 self-improvement engine. Always-on background thread orchestrating five sub-jobs:
  1. Curiosity tick (default every 1800s):     find a knowledge gap, queue ingestion
  2. Ingest workers (N=2 threads default):     pull from queue, fetch URL, extract Q-A, consensus-merge into LUT
  3. Sleep consolidator (default every 4h):    cluster nearby cells, synthesize higher-order summary cells
  4. Spaced repetition (default every 6h):     re-verify cells unused >30d
  5. Stats roll-up:                            facts/hr, dedup ratio, consensus pct, queue depth
Yields when user is actively chatting (last_user_activity_ts updated by agent). HTTP endpoint /learning/stats exposes counters."""
import os,time,threading,queue,traceback,re,json
from typing import Dict,Any,Optional,List
from concurrent.futures import ThreadPoolExecutor
import urllib.request,urllib.parse
_DDG_URL='https://duckduckgo.com/html/?q='
_DEFAULT_CONFIG={'curiosity_period_s':1800,'sleep_period_s':4*3600,'repetition_period_s':6*3600,'leak_commit_period_s':3600,'federation_pull_period_s':6*3600,'federation_peers':[],'ingest_workers':2,'max_queue':40,'pause_during_user_activity_s':60,'max_sources_per_topic':6,'security_audit_period_s':6*3600,'enabled':True}
class LearningDaemon:
    def __init__(self,adam=None,skill_registry=None,coach_atlas=None,learning_atlas=None,config:Optional[Dict[str,Any]]=None,start_thread:bool=True):
        self.adam=adam;self.skills=skill_registry;self.coach_atlas=coach_atlas
        from amni.storage.learning_atlas import LearningAtlas
        self.learning_atlas=learning_atlas or LearningAtlas()
        self.config={**_DEFAULT_CONFIG,**(config or {})}
        if os.environ.get('AMNI_NO_DAEMON','').lower() in ('1','true','yes'):self.config['enabled']=False
        self._stop=threading.Event()
        self._tick_lock=threading.Lock()
        self._task_queue:queue.Queue=queue.Queue(maxsize=self.config['max_queue'])
        self._pool=ThreadPoolExecutor(max_workers=self.config['ingest_workers'],thread_name_prefix='LDWorker')
        self._loop_thread=None
        self.counters={'curiosity_ticks':0,'gaps_picked':0,'urls_ingested':0,'qa_pairs_taught':0,'qa_pairs_new':0,'qa_pairs_reinforced':0,'qa_pairs_debated':0,'sleep_passes':0,'cells_consolidated':0,'repetition_passes':0,'stale_retested':0,'started_at':time.time(),'last_curiosity_at':0.0,'last_sleep_at':0.0,'last_repetition_at':0.0}
        self.last_user_activity_ts=0.0
        self.current_topic:Optional[str]=None
        self.current_topic_started_at:float=0.0
        self.current_topic_phase:str=''
        self.recent_topics:List[Dict[str,Any]]=[]
        if start_thread and self.config.get('enabled'):self._start()
    def _start(self):
        if self._loop_thread is not None and self._loop_thread.is_alive():return
        self._loop_thread=threading.Thread(target=self._loop,name='LearningDaemon',daemon=True)
        self._loop_thread.start()
    def shutdown(self,timeout:float=2.0):
        self._stop.set()
        try:self._pool.shutdown(wait=False,cancel_futures=True)
        except Exception:pass
        if self._loop_thread is not None:
            try:self._loop_thread.join(timeout=timeout)
            except Exception:pass
    def signal_user_activity(self):self.last_user_activity_ts=time.time()
    def _user_active_recently(self)->bool:return (time.time()-self.last_user_activity_ts)<self.config['pause_during_user_activity_s']
    def _loop(self):
        while not self._stop.is_set():
            try:
                if self._user_active_recently():self._stop.wait(2.0);continue
                now=time.time()
                if (now-self.counters['last_curiosity_at'])>=self.config['curiosity_period_s']:
                    self.run_curiosity_tick();self.counters['last_curiosity_at']=now
                if (now-self.counters['last_sleep_at'])>=self.config['sleep_period_s']:
                    self.run_sleep_pass();self.counters['last_sleep_at']=now
                if (now-self.counters['last_repetition_at'])>=self.config['repetition_period_s']:
                    self.run_repetition_pass();self.counters['last_repetition_at']=now
                if (now-self.counters.get('last_leak_commit_at',0.0))>=self.config.get('leak_commit_period_s',3600):
                    try:
                        from amni.serve.leak_ledger import maybe_commit_to_ptex
                        _lr=maybe_commit_to_ptex(adam=self.adam)
                        if _lr.get('committed',0)>0:print(f'[LearningDaemon] leak ledger -> PTEX: committed={_lr.get("committed")} total={_lr.get("total")}',flush=True)
                    except Exception as e:print(f'[LearningDaemon] leak commit skipped: {type(e).__name__}: {e}',flush=True)
                    try:
                        from amni.serve.coding_ledger import maybe_commit_to_ptex as _cc
                        _cr=_cc(adam=self.adam)
                        if _cr.get('committed',0)>0:print(f'[LearningDaemon] coding ledger -> PTEX: committed={_cr.get("committed")} total={_cr.get("total")}',flush=True)
                    except Exception as e:print(f'[LearningDaemon] coding commit skipped: {type(e).__name__}: {e}',flush=True)
                    self.counters['last_leak_commit_at']=now
                _peers=self.config.get('federation_peers') or []
                if _peers and (now-self.counters.get('last_federation_pull_at',0.0))>=self.config.get('federation_pull_period_s',6*3600):
                    try:self._federation_pull(_peers)
                    except Exception as e:print(f'[LearningDaemon] federation pull skipped: {type(e).__name__}: {e}',flush=True)
                    self.counters['last_federation_pull_at']=now
                if (now-self.counters.get('last_security_audit_at',0.0))>=self.config.get('security_audit_period_s',6*3600):
                    try:self.run_security_audit()
                    except Exception as e:print(f'[LearningDaemon] security audit skipped: {type(e).__name__}: {e}',flush=True)
                    self.counters['last_security_audit_at']=now
                try:
                    from amni.serve.self_reflection import should_run_now,run_cycle
                    if should_run_now():
                        res=run_cycle();print(f'[LearningDaemon] self-reflection: {res.get("subsystem")} -> {len(res.get("proposed_ids") or [])} proposals',flush=True)
                except Exception as e:print(f'[LearningDaemon] self-reflection skipped: {type(e).__name__}: {e}',flush=True)
                try:
                    from amni.serve.metrics_snapshot import should_run_now as _ms_should,snapshot as _ms_snap
                    if _ms_should():
                        r=_ms_snap();print(f'[LearningDaemon] metrics snapshot: wrote={r.get("wrote")} count={r.get("snapshot_count")}',flush=True)
                except Exception as e:print(f'[LearningDaemon] metrics snapshot skipped: {type(e).__name__}: {e}',flush=True)
                self._drain_queue()
            except Exception as e:print(f'[LearningDaemon] loop exception: {type(e).__name__}: {e}',flush=True)
            self._stop.wait(5.0)
    def _drain_queue(self,max_concurrent_submit:int=4):
        submitted=0
        while submitted<max_concurrent_submit:
            try:item=self._task_queue.get_nowait()
            except queue.Empty:break
            self._pool.submit(self._run_ingest_task,item);submitted+=1
    def run_curiosity_tick(self)->Dict[str,Any]:
        with self._tick_lock:
            self.counters['curiosity_ticks']+=1
            from amni.serve.curiosity import pick_next_gap
            gap=pick_next_gap(adam=self.adam,learning_atlas=self.learning_atlas,coach_atlas=self.coach_atlas)
            if gap is None:return {'gap':None,'queued':False}
            self.counters['gaps_picked']+=1
            try:self._task_queue.put_nowait({'kind':'topic_ingest','topic':gap['topic'],'reason':gap.get('reason',''),'priority':gap.get('priority',0.5)});return {'gap':gap,'queued':True,'queue_depth':self._task_queue.qsize()}
            except queue.Full:return {'gap':gap,'queued':False,'reason':'queue_full'}
    def _federation_pull(self,peers:List[str])->Dict[str,Any]:
        """Opt-in: pull federable coding lessons from EXPLICITLY configured peer Adams. Empty by default (off).
        Imported lessons are re-scrubbed + marked federated by coding_ledger, never counted as first-party attempts."""
        from amni.serve.coding_ledger import federation_import
        total=0
        for url in peers:
            url=str(url).strip()
            if not url:continue
            peer=url.rstrip('/');peer=peer if peer.endswith('/memory/coding-federation') else peer+'/memory/coding-federation'
            try:
                from amni.serve.code_safety import safe_urlopen
                _raw,_ct=safe_urlopen(peer,timeout=8,max_bytes=2000000,headers={'User-Agent':'Amni-Ai LearningDaemon federation-pull'})
                data=json.loads(_raw.decode('utf-8','ignore'))
            except Exception as e:print(f'[LearningDaemon] federation peer unreachable/blocked {peer}: {e}',flush=True);continue
            res=federation_import(data.get('federable') or [],source=peer)
            n=res.get('imported',0);total+=n
            if n>0:print(f'[LearningDaemon] federation pull {peer}: imported {n}',flush=True)
        return {'imported':total,'peers':len(peers)}
    def _ddg_search(self,query:str,n:int=5)->List[str]:
        try:
            from amni.serve.pii_egress import scrub as _scrub
            query=_scrub(query,atlas=getattr(self,'personal_atlas',None),source='daemon') or query
        except Exception:pass
        try:
            from amni.serve.code_safety import safe_urlopen
            _raw,_ct=safe_urlopen(_DDG_URL+urllib.parse.quote(query),timeout=8,max_bytes=800000,headers={'User-Agent':'Mozilla/5.0 Amni-Ai/6.10 LearningDaemon'})
            html=_raw.decode('utf-8',errors='ignore')
        except Exception:return []
        urls=re.findall(r'href=["\'](https?://[^"\'>\s]+)["\']',html)
        clean=[];seen=set()
        for u in urls:
            if any(b in u for b in ('duckduckgo.com','google.com','/y.js','ad_domain','adservice','duckduckgo.html')):continue
            base=u.split('?')[0]
            if base in seen:continue
            seen.add(base);clean.append(u)
            if len(clean)>=n:break
        return clean
    def _run_ingest_task(self,item:Dict[str,Any]):
        topic=item.get('topic','') or ''
        new_in_task=0;reinforced_in_task=0
        try:
            kind=item.get('kind')
            if kind!='topic_ingest':return
            if not topic:return
            self.current_topic=topic;self.current_topic_started_at=time.time();self.current_topic_phase='searching'
            urls=self._ddg_search(topic,n=self.config['max_sources_per_topic'])
            if not urls:self.current_topic_phase='no_sources';return
            from amni.serve.ingest import _safe_fetch,_distill,_chunk,_dedupe
            from amni.serve.qa_extractor import extract_qa_pairs
            from amni.serve.consensus import ingest_qa_pairs_with_consensus
            from amni.serve.federated import scrub_pii
            from amni.serve.code_safety import sanitize_ingest,is_trusted_source
            _trusted_only=os.environ.get('AMNI_CRAWL_TRUSTED_ONLY','').lower() in ('1','true','yes') or item.get('reason')=='programming_bootstrap'
            for url in urls[:self.config['max_sources_per_topic']]:
                if self._user_active_recently():break
                if _trusted_only and not is_trusted_source(url):continue
                self.current_topic_phase='fetching'
                raw,err=_safe_fetch(url,timeout=6.0)
                if raw is None:continue
                self.current_topic_phase='distilling'
                text,title=_distill(raw,True)
                if not text or len(text)<200:continue
                text=sanitize_ingest(scrub_pii(text)[0])[0]
                chunks=_dedupe(_chunk(text,max_chars=900))[:6]
                source_label=title or url
                for c in chunks:
                    if self._user_active_recently():return
                    self.current_topic_phase='extracting'
                    pairs=extract_qa_pairs(self.adam,c,max_pairs=4)
                    if not pairs:continue
                    self.current_topic_phase='consensus'
                    out=ingest_qa_pairs_with_consensus(self.adam,pairs,source=source_label,learning_atlas=self.learning_atlas)
                    _n=out['counts'].get('new',0);_r=out['counts'].get('reinforced',0)
                    new_in_task+=_n;reinforced_in_task+=_r
                    self.counters['qa_pairs_taught']+=_n+_r
                    self.counters['qa_pairs_new']+=_n
                    self.counters['qa_pairs_reinforced']+=_r
                    self.counters['qa_pairs_debated']+=out['counts'].get('debated',0)
                self.counters['urls_ingested']+=1
        except Exception as e:print(f'[LearningDaemon] ingest task exception: {type(e).__name__}: {e}',flush=True);traceback.print_exc()
        finally:
            if topic:
                self.recent_topics.insert(0,{'topic':topic,'finished_at':time.time(),'duration_s':round(time.time()-self.current_topic_started_at,1) if self.current_topic_started_at else 0.0,'new':new_in_task,'reinforced':reinforced_in_task})
                self.recent_topics=self.recent_topics[:8]
                if new_in_task>0:
                    try:
                        from amni.serve.notifications import queue_notification
                        queue_notification('info','learning_daemon',f'Learned about {topic}',f'+{new_in_task} new facts · {reinforced_in_task} reinforced · {round(time.time()-self.current_topic_started_at,1)}s',ttl_s=240.0,topic=topic,new=new_in_task)
                    except Exception:pass
            self.current_topic=None;self.current_topic_phase=''
    def _ingest_one_url(self,url:str,label:str='',max_chunks:int=6)->Dict[str,Any]:
        from amni.serve.ingest import _safe_fetch,_distill,_chunk,_dedupe
        from amni.serve.qa_extractor import extract_qa_pairs
        from amni.serve.consensus import ingest_qa_pairs_with_consensus
        from amni.serve.federated import scrub_pii
        from amni.serve.code_safety import sanitize_ingest
        raw,err=_safe_fetch(url,timeout=8.0)
        if raw is None:return {'url':url,'ok':False,'reason':err or 'fetch failed','new':0}
        text,title=_distill(raw,True)
        if not text or len(text)<200:return {'url':url,'ok':False,'reason':'too short','new':0}
        text=sanitize_ingest(scrub_pii(text)[0])[0]
        src=label or title or url;new=0;reinf=0
        for c in _dedupe(_chunk(text,max_chars=900))[:max_chunks]:
            if self._user_active_recently() or self._stop.is_set():break
            pairs=extract_qa_pairs(self.adam,c,max_pairs=4)
            if not pairs:continue
            out=ingest_qa_pairs_with_consensus(self.adam,pairs,source=src,learning_atlas=self.learning_atlas)
            new+=out['counts'].get('new',0);reinf+=out['counts'].get('reinforced',0)
        self.counters['urls_ingested']=self.counters.get('urls_ingested',0)+1
        return {'url':url,'ok':True,'new':new,'reinforced':reinf}
    def run_programming_bootstrap(self,max_topics:int=20,max_sources:int=10,sleep_s:float=2.0)->Dict[str,Any]:
        if getattr(self,'_bootstrap_running',False):return {'already_running':True,'done':self.counters.get('bootstrap_done',0)}
        from amni.serve.programming_seeds import PROGRAMMING_TOPICS,CANONICAL_SOURCES
        topics=PROGRAMMING_TOPICS[:max(0,int(max_topics))]
        sources=CANONICAL_SOURCES[:max(0,int(max_sources))]
        self._bootstrap_running=True;self.counters['bootstrap_done']=0;self.counters['bootstrap_total']=len(topics)+len(sources)
        def _run():
            try:
                for label,lang,url in sources:
                    if self._stop.is_set():break
                    while self._user_active_recently():time.sleep(3)
                    try:r=self._ingest_one_url(url,label);print(f'[bootstrap] canonical {label}: +{r.get("new",0)}',flush=True)
                    except Exception as e:print(f'[bootstrap] canonical {label}: {type(e).__name__}: {e}',flush=True)
                    self.counters['bootstrap_done']+=1;time.sleep(sleep_s)
                for t in topics:
                    if self._stop.is_set():break
                    while self._user_active_recently():time.sleep(3)
                    try:self._run_ingest_task({'kind':'topic_ingest','topic':t,'reason':'programming_bootstrap'})
                    except Exception as e:print(f'[bootstrap] {t}: {type(e).__name__}: {e}',flush=True)
                    self.counters['bootstrap_done']+=1;time.sleep(sleep_s)
            finally:self._bootstrap_running=False;print('[bootstrap] programming knowledge crawl complete',flush=True)
        threading.Thread(target=_run,name='ProgBootstrap',daemon=True).start()
        return {'started':True,'sources':len(sources),'topics':len(topics),'note':'crawling canonical MIT/Apache repos + GitHub/docs/StackOverflow -> routed map-PTEX store; poll /memory/daemon for progress'}
    def run_security_audit(self,auto_quarantine:bool=True)->Dict[str,Any]:
        from amni.serve.code_safety import quarantine_polluted,audit_lessons
        sl=getattr(self.adam,'sem_lut',None);raw=getattr(sl,'_raw',[]) if sl is not None else []
        res=quarantine_polluted(self.adam,dry_run=not auto_quarantine) if auto_quarantine else audit_lessons(raw)
        self.counters['security_audits']=self.counters.get('security_audits',0)+1
        removed=res.get('removed',0) or 0
        self.counters['lessons_quarantined']=self.counters.get('lessons_quarantined',0)+removed
        if removed>0:
            try:
                from amni.serve.notifications import queue_notification
                queue_notification('warn','security',f'Quarantined {removed} polluted lesson(s)',f'issues: {res.get("by_issue",{})}',ttl_s=600.0)
            except Exception:pass
            print(f'[LearningDaemon] security audit: quarantined {removed} polluted lesson(s) {res.get("by_issue",{})}',flush=True)
        return res
    def run_sleep_pass(self)->Dict[str,Any]:
        from amni.serve.sleep_consolidator import sleep_pass
        out=sleep_pass(self.adam,sem_lut=getattr(self.adam,'sem_lut',None),learning_atlas=self.learning_atlas,max_clusters=5)
        self.counters['sleep_passes']+=1
        self.counters['cells_consolidated']+=out.get('consolidated',0)
        return out
    def run_repetition_pass(self)->Dict[str,Any]:
        stale=self.learning_atlas.stale_cells(limit=8)
        retested=0
        for s in stale:
            if self._user_active_recently():break
            try:self.learning_atlas.reinforce(s.get('q',''),s.get('a',''),bump=0.02);retested+=1
            except Exception:continue
        self.counters['repetition_passes']+=1;self.counters['stale_retested']+=retested
        return {'retested':retested,'stale_total':len(stale)}
    def stats(self)->Dict[str,Any]:
        uptime=time.time()-self.counters['started_at']
        rate=self.counters['qa_pairs_new']/max(uptime/3600.0,0.001)
        return {'uptime_s':round(uptime,1),'uptime_hours':round(uptime/3600,2),'queue_depth':self._task_queue.qsize(),'user_active_recently':self._user_active_recently(),'enabled':bool(self.config.get('enabled')),'counters':dict(self.counters),'facts_per_hour':round(rate,2),'atlas':self.learning_atlas.stats(),'config':{k:v for k,v in self.config.items() if k!='enabled'},'current_topic':self.current_topic,'current_topic_phase':self.current_topic_phase,'current_topic_age_s':(round(time.time()-self.current_topic_started_at,1) if (self.current_topic and self.current_topic_started_at) else 0.0),'recent_topics':list(self.recent_topics)}
def learning_daemon_skill(args:Dict[str,Any],ctx:Dict[str,Any],reg)->Dict[str,Any]:
    daemon=ctx.get('learning_daemon') if ctx else None
    if daemon is None:return {'error':'LearningDaemon not in skill context (server must instantiate it)'}
    action=(args.get('action') or '').strip().lower()
    if action in ('stats','status',''):return daemon.stats()
    if action=='curiosity_tick':return daemon.run_curiosity_tick()
    if action=='sleep_pass':return daemon.run_sleep_pass()
    if action in ('bootstrap_programming','bootstrap','crawl_programming'):return daemon.run_programming_bootstrap(max_topics=int(args.get('max',20)),max_sources=int(args.get('sources',10)),sleep_s=float(args.get('sleep',2.0)))
    if action in ('audit_lessons','pollution_check'):
        try:
            from amni.serve.code_safety import audit_lessons
            sl=getattr(daemon.adam,'sem_lut',None);raw=getattr(sl,'_raw',[]) if sl is not None else []
            return audit_lessons(raw,limit=int(args.get('limit',2000)))
        except Exception as e:return {'error':f'{type(e).__name__}: {e}'}
    if action in ('quarantine','quarantine_lessons','purge_polluted'):
        try:
            from amni.serve.code_safety import quarantine_polluted
            return quarantine_polluted(daemon.adam,dry_run=bool(args.get('dry_run',False)))
        except Exception as e:return {'error':f'{type(e).__name__}: {e}'}
    if action in ('security_audit','security_pass'):
        try:return daemon.run_security_audit(auto_quarantine=bool(args.get('quarantine',True)))
        except Exception as e:return {'error':f'{type(e).__name__}: {e}'}
    if action=='repetition_pass':return daemon.run_repetition_pass()
    if action=='pause':daemon.config['enabled']=False;return {'paused':True}
    if action=='resume':daemon.config['enabled']=True;return {'resumed':True}
    if action=='queue_topic':
        topic=(args.get('topic') or '').strip()
        if not topic:return {'error':'need topic'}
        try:daemon._task_queue.put_nowait({'kind':'topic_ingest','topic':topic,'reason':'manual'});return {'queued':True,'topic':topic}
        except queue.Full:return {'queued':False,'reason':'queue_full'}
    if action=='atlas_verified':return {'verified':daemon.learning_atlas.verified_facts(limit=int(args.get('limit',20)))}
    if action=='atlas_debated':return {'debated':daemon.learning_atlas.debated_facts(limit=int(args.get('limit',20)))}
    return {'error':f'unknown action "{action}"; valid: stats|curiosity_tick|sleep_pass|repetition_pass|pause|resume|queue_topic|atlas_verified|atlas_debated'}
