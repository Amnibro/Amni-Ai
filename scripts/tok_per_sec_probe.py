"""tok_per_sec_probe — drives Adam server's /complete with three workloads (short/medium/long) and reports tok/s
so a before/after streaming_linear fast-path comparison is honest. Usage: python scripts/tok_per_sec_probe.py [--url http://127.0.0.1:7700] [--runs 3]"""
import sys,json,argparse,time,urllib.request
from pathlib import Path
def _post(url,payload,timeout=240):
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'content-type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url',default='http://127.0.0.1:7700');ap.add_argument('--runs',type=int,default=3)
    a=ap.parse_args();url=a.url.rstrip('/')
    cases=[('short','Count to 5:\n1, 2,',64),('medium','Explain how an LRU cache works in one paragraph:\n',256),('long','Write a step-by-step solution: Janet has 12 apples, gives half to her brother, then eats 2. How many left?\nSolution:',512)]
    rows=[]
    for label,prompt,mt in cases:
        runs=[]
        for r in range(a.runs):
            t0=time.time();j=_post(url+'/complete',{'prefix':prompt,'max_tokens':mt,'stop':[]});dt=time.time()-t0
            ntok=int(j.get('tokens') or 0);runs.append((ntok,dt,(ntok/dt) if dt>0 else 0.0))
        best=max(r[2] for r in runs);avg=sum(r[2] for r in runs)/len(runs)
        rows.append({'workload':label,'max_tokens':mt,'runs':[{'tokens':n,'wall_s':round(d,2),'tok_per_s':round(t,2)} for n,d,t in runs],'best_tok_per_s':round(best,2),'avg_tok_per_s':round(avg,2)})
        print(f'[{label:6}] max_tokens={mt:>4}  best={best:6.2f} tok/s  avg={avg:6.2f} tok/s  runs={[round(t,2) for _,_,t in runs]}',flush=True)
    out_path=Path('eval_reports/tok_per_sec_probe.json');out_path.parent.mkdir(exist_ok=True);out_path.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(f'\nwrote {out_path}')
if __name__=='__main__':main()
