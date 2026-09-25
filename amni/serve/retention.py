import os,sys,json,time,argparse,threading
from pathlib import Path
_REPO=Path(__file__).resolve().parents[2]
PROTECTED=('experiences/user_memory','experiences/user_profile.json','experiences/personal_atlas','experiences/conversation_atlas','experiences/adam_lessons.npz','experiences/adam_lut','experiences/personas.json')
_f=lambda k,d:float(os.environ.get(k,d) or d)
def policy():
    return {'days':_f('AMNI_RETENTION_DAYS',90),'max_mb':_f('AMNI_RETENTION_MAX_MB',1024),'audit_days':_f('AMNI_RETENTION_AUDIT_DAYS',365),'jsonl_mb':_f('AMNI_RETENTION_JSONL_MB',50),'interval_h':_f('AMNI_RETENTION_INTERVAL_H',24)}
def targets(root=None,conv_root=None):
    r=Path(root) if root else Path.cwd();c=Path(conv_root) if conv_root else Path('experiences')/'conversations';c=c if c.is_absolute() else r/c
    js=[r/'logs'/n for n in ('agent_skill_calls.jsonl','skill_failures.jsonl','reflection.jsonl')]+([] if root else [_REPO/'logs'/'pc_actions.jsonl',_REPO/'data'/'shell_history.jsonl'])
    return {'files':[(c,'*.jsonl'),(r/'logs','amni_serve_out_*.log'),(r/'logs','amni_serve_err_*.log')],'conv':c,'jsonl':list(dict.fromkeys(p.resolve() for p in js))}
def _ts(line):
    try:
        v=json.loads(line);t=v.get('ts') if isinstance(v,dict) else None
        return float(t) if isinstance(t,(int,float)) else None
    except Exception:return None
def _trim_jsonl(p,cutoff,max_bytes,dry):
    try:raw=p.read_text(encoding='utf-8',errors='replace').splitlines(keepends=True)
    except Exception:return 0
    keep=[l for l in raw if (lambda t:t is None or t>=cutoff)(_ts(l))];sz=[len(l.encode('utf-8')) for l in keep];tot=sum(sz);i=0
    while tot>max_bytes and i<len(keep):tot-=sz[i];i+=1
    keep=keep[i:];dropped=len(raw)-len(keep)
    if dropped and not dry:tmp=p.with_suffix(p.suffix+'.retention.tmp');tmp.write_text(''.join(keep),encoding='utf-8');os.chmod(tmp,0o600);os.replace(tmp,p)
    return dropped
def sweep(root=None,conv_root=None,days=None,max_mb=None,audit_days=None,jsonl_mb=None,dry_run=False,now=None):
    pol=policy();now=time.time() if now is None else now
    days=pol['days'] if days is None else float(days);max_mb=pol['max_mb'] if max_mb is None else float(max_mb)
    audit_days=pol['audit_days'] if audit_days is None else float(audit_days);jsonl_mb=pol['jsonl_mb'] if jsonl_mb is None else float(jsonl_mb)
    t=targets(root,conv_root);cut=now-days*86400;rm=[];kept=[]
    for d,g in t['files']:
        for p in (sorted(d.glob(g)) if d.is_dir() else []):
            st=p.stat();(rm if st.st_mtime<cut else kept).append((st.st_mtime,st.st_size,p,d==t['conv']))
    conv=sorted([k for k in kept if k[3]],key=lambda k:k[0]);tot=sum(k[1] for k in conv);cap=max_mb*1048576;i=0
    while tot>cap and i<len(conv):tot-=conv[i][1];rm.append(conv[i]);i+=1
    rmset={k[2] for k in rm}
    if not dry_run:
        for _,_,p,_ in rm:
            try:p.unlink()
            except Exception:pass
        for _,_,p,c in kept:
            try:os.chmod(p,0o600) if c and p not in rmset else None
            except Exception:pass
    trimmed={str(p):n for p in t['jsonl'] if p.is_file() for n in [_trim_jsonl(p,now-audit_days*86400,jsonl_mb*1048576,dry_run)] if n}
    return {'dry_run':bool(dry_run),'days':days,'max_mb':max_mb,'deleted':[str(k[2]) for k in rm],'trimmed_lines':trimmed,'protected':list(PROTECTED)}
def start_background(root=None,conv_root=None):
    if os.environ.get('AMNI_RETENTION_AUTO','1').strip().lower() in ('0','false','no','off') or policy()['days']<=0:return None
    def loop():
        while True:
            try:r=sweep(root,conv_root);print(f"[retention] deleted={len(r['deleted'])} trimmed={sum(r['trimmed_lines'].values())} (days={r['days']} max_mb={r['max_mb']})",flush=True) if (r['deleted'] or r['trimmed_lines']) else None
            except Exception as e:print(f'[retention] sweep failed: {e}',flush=True)
            time.sleep(max(1.0,policy()['interval_h'])*3600)
    th=threading.Thread(target=loop,daemon=True,name='amni-retention');th.start();return th
def add_args(ap):
    ap.add_argument('--root',default=None,help='Data root holding experiences/ and logs/ (default: current dir)');ap.add_argument('--conv-root',default=None)
    ap.add_argument('--days',type=float,default=None,help='Delete conversation transcripts and serve logs older than N days (0 = all). Default AMNI_RETENTION_DAYS or 90')
    ap.add_argument('--max-mb',type=float,default=None,help='Cap on total conversation transcript size');ap.add_argument('--audit-days',type=float,default=None)
    ap.add_argument('--jsonl-mb',type=float,default=None);ap.add_argument('--dry-run',action='store_true',help='Show what would be removed')
    return ap
def run_cli(a):
    r=sweep(a.root,a.conv_root,a.days,a.max_mb,a.audit_days,a.jsonl_mb,a.dry_run);print(json.dumps(r,indent=2),flush=True);return r
if __name__=='__main__':run_cli(add_args(argparse.ArgumentParser(prog='amni purge',description='Purge old conversation transcripts, serve logs and audit JSONL. Never touches user memory.')).parse_args(sys.argv[1:]))
