"""metrics_snapshot — daily snapshots of Adam's behavioral metrics for trend tracking (v6.10.60).
Step 4 of the self-improvement trajectory. Once per ~24h the LearningDaemon writes a snapshot row to
data/metrics_snapshots.jsonl capturing:
  - skill_calls: total calls, ok_rate, p50/p90, top skills by use
  - daemon: qa_pairs_new + qa_pairs_reinforced, facts/hr, sleep_passes, repetition_passes
  - coach: current streak, longest streak, total topics
  - verification: pass/fail counts over the last 24h
  - proposals: counts by status (proposed/attempted/validated/deployed/declined/reverted)
  - reflection: cycle_count, last_subsystem
  - needs_testing: queue depth
Append-only log; trend() compares latest vs N days back to surface deltas. NO statistical claims here
— Adam stores raw numbers; humans (and Adam, eventually) interpret them."""
import json,time,os
from pathlib import Path
from typing import Dict,Any,List,Optional
_MIN_INTERVAL_S=20*3600
def _repo_root()->Path:return Path(__file__).resolve().parents[2]
def _data_dir()->Path:
    p=_repo_root()/'data';p.mkdir(parents=True,exist_ok=True);return p
def _log_path()->Path:return _data_dir()/'metrics_snapshots.jsonl'
def _state_path()->Path:return _data_dir()/'metrics_snapshot_state.json'
def _load_state()->Dict[str,Any]:
    p=_state_path()
    if not p.exists():return {'last_run_ts':0.0,'snapshot_count':0,'enabled':True}
    try:return json.loads(p.read_text(encoding='utf-8'))
    except Exception:return {'last_run_ts':0.0,'snapshot_count':0,'enabled':True}
def _save_state(s:Dict[str,Any])->None:
    try:_state_path().write_text(json.dumps(s,indent=2,default=str),encoding='utf-8')
    except Exception as e:print(f'[metrics_snapshot] state save failed: {e}',flush=True)
def _collect_skill_stats()->Dict[str,Any]:
    try:from amni.serve.skill_stats import skill_stats
    except Exception:return {}
    try:s=skill_stats() or {}
    except Exception as e:return {'error':str(e)}
    rows=s.get('per_skill') if isinstance(s,dict) else (s if isinstance(s,list) else [])
    if not isinstance(rows,list):rows=[]
    by_calls=sorted(rows,key=lambda r:-int(r.get('n_calls') or 0))[:8]
    total_calls=sum(int(r.get('n_calls') or 0) for r in rows);total_ok=sum(int(r.get('ok') or 0) for r in rows)
    return {'total_calls':total_calls,'total_ok':total_ok,'ok_rate':round(total_ok/total_calls,3) if total_calls else None,'n_skills':len(rows),'top':[{'name':r.get('skill') or r.get('name'),'n_calls':r.get('n_calls'),'ok_rate':r.get('ok_rate'),'p90_ms':r.get('p90_ms') or r.get('p90')} for r in by_calls]}
def _collect_daemon_stats()->Dict[str,Any]:
    try:
        import importlib;mod=importlib.import_module('amni.serve.learning_daemon')
        inst=getattr(mod,'_GLOBAL_DAEMON',None)
        if inst is None:return {'active':False}
        c=getattr(inst,'counters',{}) or {};started=float(c.get('started_at') or time.time())
        hours=max((time.time()-started)/3600.0,1e-6)
        new=int(c.get('qa_pairs_new') or 0);rein=int(c.get('qa_pairs_reinforced') or 0)
        return {'active':True,'uptime_hours':round(hours,2),'qa_pairs_new':new,'qa_pairs_reinforced':rein,'facts_per_hour':round((new+rein)/hours,3),'sleep_passes':int(c.get('sleep_passes') or 0),'repetition_passes':int(c.get('repetition_passes') or 0),'urls_ingested':int(c.get('urls_ingested') or 0)}
    except Exception as e:return {'active':False,'error':str(e)}
def _collect_coach_stats()->Dict[str,Any]:
    try:from amni.storage.coach_atlas import CoachAtlas
    except Exception:return {}
    try:
        atlas=CoachAtlas();topics=atlas.list_topics() if hasattr(atlas,'list_topics') else []
        streak=atlas.streak_stats() if hasattr(atlas,'streak_stats') else {}
        return {'n_topics':len(topics),'current_streak':int(streak.get('current') or 0),'longest_streak':int(streak.get('longest') or 0),'today_count':int(streak.get('today_count') or 0)}
    except Exception as e:return {'error':str(e)}
def _collect_verification_stats(window_s:int=24*3600)->Dict[str,Any]:
    log=_data_dir()/'verification_log.jsonl'
    if not log.exists():return {'pass':0,'fail':0,'unverified':0,'pass_rate':None}
    cutoff=time.time()-window_s;p=0;f=0;u=0
    try:
        with log.open('r',encoding='utf-8') as fh:
            for line in fh:
                line=line.strip()
                if not line:continue
                try:r=json.loads(line)
                except Exception:continue
                if float(r.get('ts') or 0)<cutoff:continue
                v=(r.get('verdict') or r.get('status') or '').lower()
                if v=='pass':p+=1
                elif v=='fail':f+=1
                else:u+=1
    except Exception as e:return {'error':str(e)}
    tot=p+f
    return {'pass':p,'fail':f,'unverified':u,'pass_rate':round(p/tot,3) if tot else None,'window_hours':round(window_s/3600,1)}
def _collect_proposal_stats()->Dict[str,Any]:
    try:from amni.serve.self_improvement import stats as _s
    except Exception:return {}
    try:return _s() or {}
    except Exception as e:return {'error':str(e)}
def _collect_reflection_stats()->Dict[str,Any]:
    try:from amni.serve.self_reflection import status as _s
    except Exception:return {}
    try:
        s=_s() or {}
        return {'cycle_count':int(s.get('cycle_count') or 0),'last_subsystem':s.get('last_subsystem'),'enabled':bool(s.get('enabled',True))}
    except Exception as e:return {'error':str(e)}
def _collect_needs_testing()->Dict[str,Any]:
    p=_data_dir()/'needs_testing.jsonl'
    if not p.exists():return {'queue_depth':0}
    try:return {'queue_depth':sum(1 for ln in p.read_text(encoding='utf-8').splitlines() if ln.strip())}
    except Exception as e:return {'queue_depth':None,'error':str(e)}
def collect()->Dict[str,Any]:
    """Build a snapshot dict without writing it to the log. Pure read."""
    return {'ts':time.time(),'iso':time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime()),'skill_calls':_collect_skill_stats(),'daemon':_collect_daemon_stats(),'coach':_collect_coach_stats(),'verification_24h':_collect_verification_stats(),'proposals':_collect_proposal_stats(),'reflection':_collect_reflection_stats(),'needs_testing':_collect_needs_testing()}
def should_run_now(force:bool=False)->bool:
    s=_load_state()
    if not s.get('enabled',True) and not force:return False
    if force:return True
    return (time.time()-float(s.get('last_run_ts') or 0))>=_MIN_INTERVAL_S
def snapshot(force:bool=False,notify:bool=False)->Dict[str,Any]:
    if not should_run_now(force=force):
        s=_load_state();last=float(s.get('last_run_ts') or 0)
        return {'wrote':False,'reason':'too soon','seconds_until_eligible':max(0,int(_MIN_INTERVAL_S-(time.time()-last)))}
    snap=collect()
    try:
        with _log_path().open('a',encoding='utf-8') as fh:fh.write(json.dumps(snap,default=str)+'\n')
    except Exception as e:return {'wrote':False,'error':f'log write failed: {e}'}
    st=_load_state();st['last_run_ts']=snap['ts'];st['snapshot_count']=int(st.get('snapshot_count') or 0)+1;_save_state(st)
    if notify:
        try:from amni.serve.notifications import push_notification
        except Exception:push_notification=None
        if push_notification:
            try:push_notification(kind='metrics_snapshot',title='Adam took a metrics snapshot',body=f'snapshot #{st["snapshot_count"]} written; check /memory/metrics for trend',meta={'snapshot_count':st['snapshot_count']})
            except Exception:pass
    return {'wrote':True,'snapshot':snap,'snapshot_count':st['snapshot_count']}
def history(limit:int=30)->List[Dict[str,Any]]:
    p=_log_path()
    if not p.exists():return []
    out=[]
    try:
        for line in p.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line:continue
            try:out.append(json.loads(line))
            except Exception:continue
    except Exception:return []
    return out[-int(max(1,limit)):]
def _delta(now:Optional[float],then:Optional[float])->Optional[float]:
    if now is None or then is None:return None
    try:return round(float(now)-float(then),4)
    except Exception:return None
def trend(days:int=7)->Dict[str,Any]:
    """Compare latest snapshot to the oldest one within the last `days` days. Returns key deltas."""
    h=history(limit=500)
    if not h:return {'available':False,'reason':'no snapshots yet'}
    latest=h[-1];cutoff=time.time()-int(days)*86400
    older=[r for r in h if float(r.get('ts') or 0)<=cutoff]
    base=older[-1] if older else h[0]
    if base is latest:return {'available':False,'reason':'only one snapshot or all within window'}
    def g(d,*ks):
        for k in ks:
            if not isinstance(d,dict):return None
            d=d.get(k)
        return d
    return {'available':True,'from_iso':base.get('iso'),'to_iso':latest.get('iso'),'days_span':round((float(latest.get('ts') or 0)-float(base.get('ts') or 0))/86400,2),'deltas':{'skill_calls.total_calls':_delta(g(latest,'skill_calls','total_calls'),g(base,'skill_calls','total_calls')),'skill_calls.ok_rate':_delta(g(latest,'skill_calls','ok_rate'),g(base,'skill_calls','ok_rate')),'daemon.qa_pairs_new':_delta(g(latest,'daemon','qa_pairs_new'),g(base,'daemon','qa_pairs_new')),'daemon.facts_per_hour':_delta(g(latest,'daemon','facts_per_hour'),g(base,'daemon','facts_per_hour')),'coach.current_streak':_delta(g(latest,'coach','current_streak'),g(base,'coach','current_streak')),'verification.pass_rate':_delta(g(latest,'verification_24h','pass_rate'),g(base,'verification_24h','pass_rate')),'reflection.cycle_count':_delta(g(latest,'reflection','cycle_count'),g(base,'reflection','cycle_count')),'needs_testing.queue_depth':_delta(g(latest,'needs_testing','queue_depth'),g(base,'needs_testing','queue_depth'))},'latest':latest,'baseline':base}
def status()->Dict[str,Any]:
    s=_load_state();h=history(limit=5);now=time.time();last=float(s.get('last_run_ts') or 0)
    return {'enabled':bool(s.get('enabled',True)),'last_run_ts':last,'last_run_iso':time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(last)) if last else '','snapshot_count':int(s.get('snapshot_count') or 0),'seconds_until_eligible':max(0,int(_MIN_INTERVAL_S-(now-last))),'min_interval_s':_MIN_INTERVAL_S,'recent':[{'iso':r.get('iso'),'skill_calls':(r.get('skill_calls') or {}).get('total_calls'),'qa_new':(r.get('daemon') or {}).get('qa_pairs_new')} for r in h]}
def set_enabled(enabled:bool)->Dict[str,Any]:
    s=_load_state();s['enabled']=bool(enabled);_save_state(s);return status()
