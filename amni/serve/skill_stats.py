"""skill_stats — aggregate per-skill latency + success-rate from the existing logs/agent_skill_calls.jsonl audit log.
Reuses the existing audit trail (no new storage). Read-only aggregator with optional time-window filter."""
import json,time
from pathlib import Path
from typing import Dict,Any,List,Optional
def _default_log_path()->Path:return Path(__file__).resolve().parents[2]/'logs'/'agent_skill_calls.jsonl'
def _percentile(sorted_vals:List[float],pct:float)->float:
    if not sorted_vals:return 0.0
    n=len(sorted_vals);idx=max(0,min(n-1,int(pct*n/100.0)));return float(sorted_vals[idx])
def aggregate(log_path:Optional[str]=None,hours:Optional[float]=None,limit_per_skill:int=2000)->Dict[str,Any]:
    """Walk the audit log, group by skill, return {skills:{name:{n_calls, ok, errors, avg_ms, p50, p90, p99, max_ms, last_ts}}, totals:{...}, window_hours, log_exists}."""
    p=Path(log_path) if log_path else _default_log_path()
    if not p.exists():return {'skills':{},'totals':{'n_calls':0,'n_ok':0,'n_err':0},'window_hours':hours,'log_exists':False}
    cutoff=(time.time()-hours*3600) if hours else 0
    by_skill:Dict[str,List[Dict[str,Any]]]={}
    try:
        for ln in p.read_text(encoding='utf-8',errors='ignore').splitlines():
            if not ln.strip():continue
            try:r=json.loads(ln)
            except Exception:continue
            if cutoff and float(r.get('ts') or 0)<cutoff:continue
            sk=r.get('skill') or '?'
            buf=by_skill.setdefault(sk,[])
            if len(buf)<limit_per_skill:buf.append(r)
            else:buf[len(buf)%limit_per_skill]=r
    except Exception:return {'skills':{},'totals':{'n_calls':0,'n_ok':0,'n_err':0},'window_hours':hours,'log_exists':True,'error':'log read failed'}
    out={};total_calls=0;total_ok=0;total_err=0;total_ms=0
    for sk,calls in by_skill.items():
        ms=sorted([float(c.get('elapsed_ms') or 0) for c in calls])
        ok=sum(1 for c in calls if c.get('ok'));n=len(calls);err=n-ok
        last_ts=max(float(c.get('ts') or 0) for c in calls) if calls else 0
        out[sk]={'n_calls':n,'ok':ok,'errors':err,'ok_rate':round(ok/n,3) if n else 0.0,'avg_ms':round(sum(ms)/n,1) if n else 0.0,'p50_ms':round(_percentile(ms,50),1),'p90_ms':round(_percentile(ms,90),1),'p99_ms':round(_percentile(ms,99),1),'max_ms':round(ms[-1],1) if ms else 0.0,'last_ts':last_ts,'last_ago_s':round(time.time()-last_ts,1) if last_ts else None}
        total_calls+=n;total_ok+=ok;total_err+=err;total_ms+=sum(ms)
    skills_sorted=dict(sorted(out.items(),key=lambda kv:-kv[1]['n_calls']))
    return {'skills':skills_sorted,'totals':{'n_calls':total_calls,'n_ok':total_ok,'n_err':total_err,'overall_ok_rate':round(total_ok/total_calls,3) if total_calls else 0.0,'avg_ms':round(total_ms/total_calls,1) if total_calls else 0.0},'window_hours':hours,'log_exists':True,'log_path':str(p)}
