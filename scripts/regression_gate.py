"""regression_gate — the "provably improving" mechanism (checklist B4/B5). Compares a candidate Adam iteration's
benchmark result against the previous canonical result, per bench, against a calibrated noise band. A candidate is
ADOPTED only if no bench regresses beyond the noise band AND aggregate >= previous. CPU-only.
Usage:
  python scripts/regression_gate.py --candidate eval_reports/leaderboard_new.json --baseline eval_reports/leaderboard_prev.json --noise-band 1.5
  python scripts/regression_gate.py --calibrate eval_reports/run1.json eval_reports/run2.json eval_reports/run3.json  # estimate noise band"""
import sys,json,argparse,statistics
from pathlib import Path
BENCHES=['mmlu_pro','gpqa_diamond','math500','humanevalplus','mbppplus','gsm8k','arc']
def _scores(p):
    j=json.loads(Path(p).read_text(encoding='utf-8'))
    s=j.get('scores') or {}
    full=j.get('full') or {}
    out={}
    for b in BENCHES:
        if b in s:out[b]=float(s[b])
        elif b in full and isinstance(full[b],dict) and 'accuracy' in full[b]:out[b]=float(full[b]['accuracy'])
    return out
def calibrate(paths):
    runs=[_scores(p) for p in paths]
    benches=set().union(*[set(r) for r in runs])
    print('[calibrate] per-bench run-to-run spread (same bake, repeated runs):')
    bands={}
    for b in sorted(benches):
        vals=[r[b] for r in runs if b in r]
        if len(vals)>=2:
            spread=max(vals)-min(vals);sd=statistics.pstdev(vals)
            bands[b]=round(spread,2)
            print(f'  {b:16} vals={vals} spread={spread:.2f}pp stdev={sd:.2f}')
    overall=max(bands.values()) if bands else 0
    print(f'[calibrate] recommended noise band = {overall:.2f}pp (max per-bench spread)')
    return bands
def gate(candidate,baseline,noise_band):
    c=_scores(candidate);b=_scores(baseline)
    benches=[x for x in BENCHES if x in c and x in b]
    print(f'[gate] candidate vs baseline, noise band = {noise_band}pp')
    print(f'{"bench":16} {"baseline":>9} {"candidate":>10} {"delta":>8}  verdict')
    regressions=[];deltas=[]
    for x in benches:
        d=c[x]-b[x];deltas.append(d)
        verdict='OK'
        if d<-noise_band:verdict='REGRESSION';regressions.append((x,d))
        elif d>noise_band:verdict='IMPROVED'
        else:verdict='within-noise'
        print(f'{x:16} {b[x]:9.1f} {c[x]:10.1f} {d:+8.1f}  {verdict}')
    agg_c=sum(c[x] for x in benches)/len(benches);agg_b=sum(b[x] for x in benches)/len(benches)
    print(f'{"AGGREGATE":16} {agg_b:9.2f} {agg_c:10.2f} {agg_c-agg_b:+8.2f}')
    adopt=(len(regressions)==0) and (agg_c>=agg_b)
    print()
    if adopt:print(f'[gate] ADOPT — no bench regressed beyond {noise_band}pp and aggregate improved ({agg_c-agg_b:+.2f}pp)')
    else:
        if regressions:print(f'[gate] BLOCK — {len(regressions)} bench(es) regressed beyond noise: '+', '.join(f'{x}({d:+.1f})' for x,d in regressions))
        if agg_c<agg_b:print(f'[gate] BLOCK — aggregate dropped ({agg_c-agg_b:+.2f}pp)')
    return {'adopt':adopt,'regressions':regressions,'agg_candidate':agg_c,'agg_baseline':agg_b,'deltas':dict(zip(benches,deltas))}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate');ap.add_argument('--baseline');ap.add_argument('--noise-band',type=float,default=1.5)
    ap.add_argument('--calibrate',nargs='+',default=None)
    ap.add_argument('--json',action='store_true')
    a=ap.parse_args()
    if a.calibrate:calibrate(a.calibrate);return
    if a.candidate and a.baseline:
        r=gate(a.candidate,a.baseline,a.noise_band)
        if a.json:print(json.dumps(r,indent=2,default=str))
        sys.exit(0 if r['adopt'] else 1)
    ap.print_help()
if __name__=='__main__':main()
