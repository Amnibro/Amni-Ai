"""adam_iteration_ledger — append a benchmark run to the per-iteration history (checklist B3) and render the
trend chart that makes stepwise improvement / regression visible. CPU-only; reads result JSONs, writes CSV + PNG.
Usage:
  python scripts/adam_iteration_ledger.py --add eval_reports/leaderboard_<bake>_<date>.json --iter 22 --bake v5.0.3-gf17 --tokps 27.4
  python scripts/adam_iteration_ledger.py --chart        # re-render trend from existing CSV"""
import sys,json,csv,argparse
from pathlib import Path
LEDGER=Path('eval_reports/adam_iteration_history.csv')
BENCHES=['mmlu_pro','gpqa_diamond','math500','humanevalplus','mbppplus','gsm8k','arc']
FIELDS=['iter','bake_version','date','harness_version','tokps']+BENCHES+['notes']
def _read_scores(result_json):
    j=json.loads(Path(result_json).read_text(encoding='utf-8'))
    scores=j.get('scores') or {}
    full=j.get('full') or {}
    out={}
    for b in BENCHES:
        if b in scores:out[b]=scores[b]
        elif b in full and isinstance(full[b],dict) and 'accuracy' in full[b]:out[b]=full[b]['accuracy']
    return out,j
def add(result_json,iteration,bake,date,tokps,harness_version,notes):
    scores,j=_read_scores(result_json)
    LEDGER.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    if LEDGER.exists():
        with open(LEDGER,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    row={'iter':iteration,'bake_version':bake,'date':date,'harness_version':harness_version,'tokps':tokps,'notes':notes}
    for b in BENCHES:row[b]=scores.get(b,'')
    rows=[r for r in rows if r.get('iter')!=str(iteration)]
    rows.append(row)
    rows.sort(key=lambda r:float(r['iter']) if str(r.get('iter','')).replace('.','',1).isdigit() else 0)
    with open(LEDGER,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
        for r in rows:w.writerow({k:r.get(k,'') for k in FIELDS})
    print(f'[ledger] wrote iter {iteration} to {LEDGER} ({len(rows)} total rows)')
    return rows
def chart(out='eval_reports/adam_iteration_trend.png'):
    if not LEDGER.exists():print('[ledger] no history yet');return
    with open(LEDGER,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if not rows:print('[ledger] empty');return
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        iters=[r['iter'] for r in rows]
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(11,9),height_ratios=[3,1])
        for b in BENCHES:
            ys=[]
            xs=[]
            for r in rows:
                v=r.get(b,'')
                if v not in ('',None):
                    try:ys.append(float(v));xs.append(r['iter'])
                    except Exception:pass
            if ys:ax1.plot(xs,ys,marker='o',label=b)
        ax1.set_ylabel('accuracy %');ax1.set_title('Adam accuracy by iteration (per benchmark)');ax1.legend(fontsize=8);ax1.grid(True,alpha=0.3);ax1.set_ylim(0,100)
        tps=[]
        txs=[]
        for r in rows:
            v=r.get('tokps','')
            if v not in ('',None):
                try:tps.append(float(v));txs.append(r['iter'])
                except Exception:pass
        if tps:ax2.plot(txs,tps,marker='s',color='tab:red');ax2.set_ylabel('tok/s');ax2.set_xlabel('iteration');ax2.set_title('Throughput by iteration');ax2.grid(True,alpha=0.3)
        fig.tight_layout();fig.savefig(out,dpi=120);plt.close(fig)
        print(f'[ledger] chart -> {out}')
    except Exception as e:print(f'[ledger] chart skipped: {e}')
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--add',default=None,help='result JSON to ingest')
    ap.add_argument('--iter',default=None);ap.add_argument('--bake',default='');ap.add_argument('--date',default='')
    ap.add_argument('--tokps',default='');ap.add_argument('--harness-version',default='h1.0.0');ap.add_argument('--notes',default='')
    ap.add_argument('--chart',action='store_true')
    a=ap.parse_args()
    if a.add and a.iter is not None:add(a.add,a.iter,a.bake,a.date,a.tokps,a.harness_version,a.notes)
    if a.chart or a.add:chart()
    if not a.add and not a.chart:ap.print_help()
if __name__=='__main__':main()
