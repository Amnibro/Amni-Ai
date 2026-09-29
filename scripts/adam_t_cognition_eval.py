import os,re,sys,json,time,random,argparse,resource,statistics
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import amni.serve.cognition as C
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));DEV=os.path.join(ROOT,'data','adam_t','eval','cognition','dev.jsonl');SRC=os.path.join(ROOT,'data','adam_t','eval','chat','chat_src.jsonl')
def strs(o):return [o] if isinstance(o,str) else [z for v in (o.values() if isinstance(o,dict) else o) for z in strs(v)] if isinstance(o,(dict,list)) else []
def ng(t:str,n:int=6)->set:w=re.findall(r"[a-z0-9']+",t.lower());return {tuple(w[i:i+n]) for i in range(len(w)-n+1)}
def overlap(dev=DEV,src=SRC)->dict:
    S=set().union(*[ng(s) for l in open(src) for s in strs(json.loads(l))]);D=[json.loads(l) for l in open(dev)];H=[(o['id'],sorted(' '.join(g) for g in ng(' '.join(strs({k:v for k,v in o.items() if k!='license'})))&S)) for o in D]
    return dict(src_6grams=len(S),items=len(D),overlapping=[h for h in H if h[1]])
def an(x:str)->str:return C._sg(re.sub(r"(?i)^(?:the|an?)\s+",'',x.strip().rstrip('.')).lower())
def hit(txt:str,gold:list)->bool:t=txt.lower();return any(g.lower() in t for g in gold)
def ptxt(p:dict)->str:return ' '.join(e['line']+' '+e['r']+' '+e['b'] for e in p['e'])
def pct(x):return round(100*float(np.mean(x)),1) if len(x) else None
def lat(v):v=sorted(v);return dict(n=len(v),p50=round(statistics.median(v),2),p95=round(v[min(len(v)-1,int(0.95*len(v)))],2),max=round(v[-1],2),mean=round(float(np.mean(v)),2)) if v else {}
def run(dev=DEV,out=None)->dict:
    D=[json.loads(l) for l in open(dev)];t=time.perf_counter();C.warm();warm=time.perf_counter()-t;R=[];L={}
    for o in D:
        ty=o['type'];r=dict(id=o['id'],type=ty,q=o['q'],gold=o['gold'])
        if ty=='mode':m,cf=C.mode(o['q']);r.update(pred=m,ok=m==o['gold'][0],conf=cf);R.append(r);continue
        op,a=C.parse(o['q']);r['parse_op']=op;r['parse_args']=a;r['parse_ok']=op==ty and all(an(a.get(k,''))==an(o[k]) for k in ('a','b','c') if k in o)
        f={'relate':lambda:C.relate(o['a'],o['b']),'common':lambda:C.common(o['a'],o['b']),'analogy':lambda:C.analogy(o['a'],o['b'],o['c']),'abstract':lambda:C.abstract(o['a'])}[ty];x=f();L.setdefault(ty,[]).append(x['ms']);P=x['paths']
        r.update(ms=x['ms'],ok_any=x['ok'],lines=x['lines'],top=[dict(kind=p['kind'],q=p.get('q'),v=round(p['v'],3),src=[e['src'] for e in p['e']],refs=[e['ref'] for e in p['e']]) for p in P[:3]])
        if ty in ('relate','common'):r.update(hit1=bool(P) and hit(ptxt(P[0]),o['gold']),hit3=any(hit(ptxt(p),o['gold']) for p in P[:3]),hitl=hit(' '.join(x['lines']),o['gold']))
        elif ty=='analogy':r.update(answer=x.get('answer'),hit1=bool(x.get('answer')) and hit(x['answer'],o['gold']),hit3=any(hit(p.get('ans',''),o['gold']) for p in P[:3]))
        else:B=[e['b'] for e in (P[0]['e'] if P else [])];r.update(chain=B,hit1=bool(B) and hit(B[0],o['gold']),hit3=hit(' | '.join(B),o['gold']))
        e2=C.run(o['q']);r.update(e2e_op=e2.get('op'),e2e_mode=e2['mode'],e2e_lines=e2['lines'],e2e_hit=hit(' '.join(e2['lines'])+' '+str(e2.get('answer') or ''),o['gold']));L.setdefault('e2e',[]).append(e2['ms']);R.append(r)
    S={}
    for ty in ('relate','common','analogy','abstract'):
        X=[r for r in R if r['type']==ty];S[ty]=dict(n=len(X),answered=pct([r['ok_any'] for r in X]),hit1=pct([r['hit1'] for r in X]),hit3=pct([r['hit3'] for r in X]),hit_lines=pct([r.get('hitl',r['hit1']) for r in X]),parse=pct([r['parse_ok'] for r in X]),e2e_hit=pct([r['e2e_hit'] for r in X]),lat=lat(L.get(ty,[])))
    M=[r for r in R if r['type']=='mode'];S['mode']=dict(n=len(M),acc=pct([r['ok'] for r in M]),**{g:pct([r['ok'] for r in M if r['gold'][0]==g]) for g in ('factual','analytical','creative')},confusion={f"{g}->{p}":sum(1 for r in M if r['gold'][0]==g and r['pred']==p) for g in ('factual','analytical','creative') for p in ('factual','analytical','creative')})
    S['e2e_lat']=lat(L.get('e2e',[]));S['warm_s']=round(warm,2);S['rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024
    src={};[src.__setitem__(s,src.get(s,0)+1) for r in R if r['type'] in ('relate','common') and r.get('top') for s in r['top'][0]['src']];S['top_path_sources']=src
    o=dict(summary=S,items=R);out and open(out,'w').write(json.dumps(o,indent=1,default=str));return o
def sample(rep:str,n:int=30,seed:int=7)->None:
    R=[r for r in json.load(open(rep))['items'] if r['type'] in ('relate','common')];random.Random(seed).shuffle(R)
    for r in R[:n]:print(f"{r['id']} | {r['q']} | hit1={r['hit1']} | top={r['top'][0]['kind'] if r['top'] else None}");[print('    ',l) for l in r['lines']];[print('     ref:',x) for x in (r['top'][0]['refs'] if r['top'] else [])]
def creative(seeds:list,k:int=3)->list:
    O=[]
    for s in seeds:
        o=C.connect(s,full=True);A=o.get('all',[]);O.append(dict(seed=s,ms=o['ms'],n=len(A),top=A[:k],bottom=A[-k:],lines=o['lines']))
    return O
def calib()->dict:
    G=C._g();rng=random.Random(3);N=[i for i in range(len(G['nm'])) if G['pos'][i]==0];rnd=[C.sdist(rng.choice(N),rng.choice(N)) for _ in range(400)]
    sib=[C.sdist(a,b) for a,b in [(C.wsyn(x,0)[0],C.wsyn(y,0)[0]) for x,y in (('dog','cat'),('oak','maple'),('violin','cello'),('car','truck'),('apple','pear'),('shark','dolphin'),('lion','tiger'),('rose','tulip'),('copper','silver'),('coffee','tea'),('chair','table'),('river','lake'),('hammer','wrench'),('shirt','trousers'),('eagle','hawk'))]]
    rel=[C.sdist(C.wsyn(a,0)[0],C.wsyn(b,0)[0]) for o in map(json.loads,open(DEV)) if o['type']=='relate' for a,b in [(o['a'],o['b'])] if C.wsyn(a,0) and C.wsyn(b,0)]
    q=lambda v:dict(n=len(v),p10=round(float(np.percentile(v,10)),2),p50=round(float(np.median(v)),2),p90=round(float(np.percentile(v,90)),2))
    return dict(siblings=q(sib),dev_relation_pairs=q(rel),random_nouns=q(rnd),D0=C.D0)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['overlap','run','sample','creative','calib']);ap.add_argument('--out',default=os.path.join(ROOT,'data','adam_t','eval','cognition','report_dev.json'));ap.add_argument('--n',type=int,default=30);ap.add_argument('--data',default=DEV);ap.add_argument('seeds',nargs='*');a=ap.parse_args()
    if a.cmd=='overlap':print(json.dumps(overlap(a.data),indent=1))
    elif a.cmd=='run':print(json.dumps(run(a.data,a.out)['summary'],indent=1))
    elif a.cmd=='sample':sample(a.out,a.n)
    elif a.cmd=='creative':C.warm();print(json.dumps(creative(a.seeds or ['time','ocean','loneliness','city','fire','music','memory','heart','moon','tree','love','bridge','winter','light','money']),indent=1))
    else:C.warm();print(json.dumps(calib(),indent=1))
