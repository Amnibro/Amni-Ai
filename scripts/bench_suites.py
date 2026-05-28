"""bench_suites — run Adam on the leaderboard benchmarks (ARC, MMLU, GSM8K, HellaSwag, Winogrande) and emit the
chart people read: a model-vs-model accuracy table (+ optional PNG) against published references. Drives a RUNNING
Adam server (python scripts/amni_serve.py).

Usage:
  python scripts/bench_suites.py --suites arc mmlu gsm8k --limit 100 --url http://127.0.0.1:7700 --chart adam_leaderboard.png
  python scripts/bench_suites.py --suites sample            # offline scoring smoke
"""
import sys,json,argparse,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.eval import suite_bench as sb
def _post(url,payload,timeout):
    req=urllib.request.Request(url,data=json.dumps(payload).encode('utf-8'),headers={'content-type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode('utf-8','ignore'))
def _make_generate_fn(url,timeout):
    url=url.rstrip('/')
    def gen(prompt):
        numeric=('####' in prompt) or ('step by step' in prompt)
        mt=8 if 'single letter' in prompt else (640 if numeric else 320)
        stops=['Problem:','Question:'] if numeric else ['\n\n','</task>','Question:']
        for path,payload,outk in ((url+'/complete',{'prefix':prompt,'max_tokens':mt,'stop':stops},('completion','text','output')),(url+'/chat',{'message':prompt,'max_new_tokens':mt},('answer','text','response'))):
            try:
                j=_post(path,payload,timeout)
                if isinstance(j,dict):
                    for k in outk:
                        if j.get(k):return j[k]
                    if isinstance(j.get('output'),dict):
                        for k in outk:
                            if j['output'].get(k):return j['output'][k]
            except Exception:continue
        return ''
    return gen
def main():
    ap=argparse.ArgumentParser(description='Run Adam on leaderboard benchmarks and chart it.')
    ap.add_argument('--suites',nargs='+',default=['arc','mmlu','gsm8k','hellaswag','winogrande'])
    ap.add_argument('--limit',type=int,default=None,help='cap items per suite')
    ap.add_argument('--url',default='http://127.0.0.1:7700')
    ap.add_argument('--req-timeout',type=int,default=120)
    ap.add_argument('--chart',default=None,help='output PNG path for the bar chart')
    ap.add_argument('--out',default=None,help='write full per-suite JSON results here')
    ap.add_argument('--label',default='Adam (this run)')
    a=ap.parse_args()
    gen=_make_generate_fn(a.url,a.req_timeout)
    scores={};full={}
    for s in a.suites:
        print(f'[suites] running {s} (limit={a.limit}) ...',flush=True)
        r=sb.run_suite(s,gen,limit=a.limit)
        full[s]=r
        if r['n']>0:scores[s]=r['accuracy'];print(f'[suites] {s}: {r["accuracy"]}% ({r["correct"]}/{r["n"]})',flush=True)
        else:print(f'[suites] {s}: skipped ({r.get("reason","no items")})',flush=True)
    print('\n'+sb.leaderboard(scores,adam_label=a.label)+'\n',flush=True)
    if a.chart:
        cr=sb.render_chart(scores,a.chart,adam_label=a.label.split(' ')[0])
        print(f'[suites] chart: {cr}',flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({'scores':scores,'full':full},indent=2,default=str),encoding='utf-8')
        print(f'[suites] wrote {a.out}',flush=True)
if __name__=='__main__':main()
