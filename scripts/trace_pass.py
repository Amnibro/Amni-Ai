"""trace_pass — drive Adam's /complete endpoint over a small representative corpus while the in-server
trace_hooks counter is enabled. Reports |S_l| (distinct residual-stream hashes) per layer to decide the
resonance-substrate path (Class A direct-address vs Class B Reffelt-on-activations vs Class B7 sub-vector).
Assumes the server already has /admin/trace/{start,stop,reset,status} endpoints wired (see
docs/checklists/trace_pass_design_v1.md). Single-process, ~30-40 min wall."""
import sys,json,argparse,urllib.request,time
from pathlib import Path
def _post(url,payload=None,timeout=240):
    data=json.dumps(payload or {}).encode()
    req=urllib.request.Request(url,data=data,headers={'content-type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())
def _get(url,timeout=10):
    req=urllib.request.Request(url,method='GET')
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())
def _load_corpus(limit_gsm:int,limit_arc:int):
    items=[]
    try:
        from datasets import load_dataset
        ds=load_dataset('gsm8k','main',split='test')
        for i,r in enumerate(ds):
            items.append({'src':'gsm8k','prompt':f'Problem: {r["question"]}\nSolution (work through it step by step, then write the final answer on its own line prefixed by "####"):\n'})
            if len(items)>=limit_gsm:break
        ds2=load_dataset('allenai/ai2_arc','ARC-Challenge',split='test')
        added=0
        for r in ds2:
            labels=r['choices']['label'];texts=r['choices']['text']
            if r['answerKey'] not in labels:continue
            opts='\n'.join(f'{chr(65+j)}. {t}' for j,t in enumerate(texts))
            items.append({'src':'arc','prompt':f'Answer the multiple-choice question with ONLY the single letter of the correct option.\n\n{r["question"]}\n{opts}\nAnswer:'})
            added+=1
            if added>=limit_arc:break
    except Exception as e:print(f'[trace] corpus load error: {e}',flush=True)
    return items
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url',default='http://127.0.0.1:7700')
    ap.add_argument('--limit-gsm',type=int,default=200)
    ap.add_argument('--limit-arc',type=int,default=200)
    ap.add_argument('--max-tokens',type=int,default=64)
    ap.add_argument('--req-timeout',type=int,default=120)
    ap.add_argument('--out',default='eval_reports/trace_state_counts.json')
    ap.add_argument('--summary',default='eval_reports/trace_pass_summary.md')
    ap.add_argument('--chart',default='eval_reports/trace_state_counts.png')
    a=ap.parse_args()
    url=a.url.rstrip('/')
    print(f'[trace] /admin/trace/status -> {_get(url+"/admin/trace/status")}',flush=True)
    print(f'[trace] reset -> {_post(url+"/admin/trace/reset")}',flush=True)
    print(f'[trace] start -> {_post(url+"/admin/trace/start")}',flush=True)
    items=_load_corpus(a.limit_gsm,a.limit_arc)
    print(f'[trace] corpus: {len(items)} items',flush=True)
    t0=time.time()
    for i,it in enumerate(items):
        try:_post(url+'/complete',{'prefix':it['prompt'],'max_tokens':a.max_tokens,'stop':[]},timeout=a.req_timeout)
        except Exception as e:print(f'[trace] item {i} skipped: {str(e)[:80]}',flush=True)
        if (i+1)%25==0:
            s=_get(url+'/admin/trace/status');print(f'[trace] {i+1}/{len(items)} total_obs={s.get("total_obs")} dt={time.time()-t0:.0f}s',flush=True)
    snap=_post(url+'/admin/trace/stop')
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(snap,indent=2),encoding='utf-8')
    print(f'[trace] wrote {a.out}',flush=True)
    layers=snap.get('layers',{})
    lines=[f'# Trace pass summary — {len(layers)} layers, {snap.get("total_obs",0)} observations','','| layer | distinct |S_l| | total obs | dominance ratio |','|---|---:|---:|---:|']
    for k in sorted(layers,key=lambda x:int(x)):
        L=layers[k];d=L.get('distinct',0);t=L.get('total',0);dom=(L.get('top_5',[[None,0]])[0][1]/t) if t else 0
        lines.append(f'| {k} | {d} | {t} | {dom:.2%} |')
    lines+=['','## Substrate decision',f'- Max |S_l| across layers: **{max((L.get("distinct",0) for L in layers.values()),default=0)}**','- If ≤2^32 → Class A direct-address fits in single-pixel RGBA8','- If ≤2^64 → Class A with 2-pixel composite addressing','- If >2^64 → Class B Reffelt-on-activations mandatory']
    Path(a.summary).write_text('\n'.join(lines),encoding='utf-8')
    print(f'[trace] wrote {a.summary}',flush=True)
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        xs=sorted([int(k) for k in layers]);ys=[layers[str(k)]['distinct'] for k in xs]
        fig,ax=plt.subplots(figsize=(12,5));ax.bar(xs,ys);ax.set_yscale('log');ax.set_xlabel('layer index');ax.set_ylabel('|S_l| distinct residual hashes (log)');ax.set_title('Distinct residual-stream states per layer — Adam Gemma-4-E2B GF17')
        fig.tight_layout();fig.savefig(a.chart,dpi=120);plt.close(fig)
        print(f'[trace] wrote {a.chart}',flush=True)
    except Exception as e:print(f'[trace] chart skipped: {e}',flush=True)
if __name__=='__main__':main()
