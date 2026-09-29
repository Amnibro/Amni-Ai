import os,re,sys,json,time,shutil,argparse,tempfile,collections
import numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path[:0]=[ROOT,os.path.join(ROOT,'scripts')]
from adam_nightly import NORM
OUT=os.path.join(ROOT,'data','adam_t','eval','harness');DEV=os.path.join(OUT,'harness_dev.jsonl');CHS=os.path.join(ROOT,'data','adam_t','eval','chat','chat_src.jsonl');TOR=os.path.join(ROOT,'data','adam_t','eval','toroid','toroid_dev.jsonl')
LIC='CC0-1.0 (written from scratch for ADAM-T-HARNESS, 2026-09-29; no dataset text)';KINDS=('multi_mem','multi_know','mix','followup','nomem','single')
_W=re.compile(r"[a-z0-9]+");nt=lambda s:_W.findall(str(s).lower());ng=lambda ws,n=6:{tuple(ws[i:i+n]) for i in range(len(ws)-n+1)}
HEDGE=re.compile(r"(?i)\b(?:haven'?t (?:told|mentioned|said)|have not told|did(?:n'?t| not) (?:tell|mention|say)|not (?:sure|certain)|no idea|do(?:n'?t| not) (?:know|have|remember)|can'?t remember|could(?:n'?t| not) (?:find|confirm|verify)|might be|i think|i believe|my best guess|never (?:told|mentioned)|not something you)\b")
ABST=re.compile(r"(?i)\b(?:haven'?t (?:told|mentioned|said)|have not told|did(?:n'?t| not) (?:tell|mention|say)|not (?:sure|certain)|no idea|do(?:n'?t| not) (?:know|have|remember)|can'?t remember|never (?:told|mentioned)|not something you)\b")
SENT=lambda p:[x for x in re.split(r"(?<=[.!?])\s+|\n+",p or '') if x.strip()]
has=lambda g,t:bool(NORM(g)) and re.search(r"(?<!\S)"+re.escape(NORM(g))+r"(?!\S)",NORM(t)) is not None
def build(a):
    D=[json.loads(l) for l in open(a.src)];G=set()
    for l in open(CHS):r=json.loads(l);[G.update(ng(nt(v))) for v in [r.get('q','')]+[x for t in r.get('turns',[]) for x in t]+[x for x in r.get('gold',[])+r.get('ctx',[]) if isinstance(x,str)]]
    for l in open(TOR):r=json.loads(l);[G.update(ng(nt(v))) for v in [r.get('q','')]+[x for t in r['turns'] for x in (t['u'],t['a'])]]
    ov=[(d['id'],[' '.join(g) for x in [d['q']]+[z for t in d['turns'] for z in t]+[s['sub'] for s in d['slots']] for g in ng(nt(x)) if g in G][:3]) for d in D];ov=[x for x in ov if x[1]]
    bad=[d['id'] for d in D if d['kind'] not in KINDS or d.get('license')!=LIC or any(s['src'] not in ('mem','know','calc','none') or set(map(str.lower,s['ans']))&set(map(str.lower,s['bad'])) or (s['src']=='mem' and not any(x.lower() in u.lower() for x in s['ans'] for u,_ in d['turns'])) or (s['src']=='none' and s['ans']) for s in d['slots'])]
    for d in D:d['split']='tune' if int(d['id'][3:])%2 else 'test'
    os.makedirs(OUT,exist_ok=True);open(DEV,'w').write(''.join(json.dumps(d)+'\n' for d in D))
    m={'n':len(D),'by_kind':dict(collections.Counter(d['kind'] for d in D)),'slots':dict(collections.Counter(s['src'] for d in D for s in d['slots'])),'multipart':sum(len(d['slots'])>1 for d in D),'split':dict(collections.Counter(d['split'] for d in D)),'sixgram_overlap_chat_src_and_toroid_dev':len(ov),'overlap_items':ov,'validation_failures':bad,'license':LIC,'split_rule':'odd id = tune (rules developed and weights set here), even id = test (aggregates only)'}
    json.dump(m,open(os.path.join(OUT,'harness_meta.json'),'w'),indent=1);print(json.dumps({k:v for k,v in m.items() if k!='overlap_items'},indent=1));assert not ov and not bad
def score(d:dict,p:str)->dict:
    S=SENT(p);A=[x for s in d['slots'] for x in s['ans']];O=[]
    for i,s in enumerate(d['slots']):
        oth={NORM(x) for j,z in enumerate(d['slots']) if j!=i for x in z['ans']};bd=[b for b in s['bad'] if NORM(b) not in oth]
        if s['src']=='none':fm=any(has(b,p) for b in bd) or not ABST.search(p);O.append(dict(src='none',ok=not fm,fm=fm))
        else:H=[x for x in S if any(has(g,x) for g in s['ans']) and not any(has(b,x) for b in bd)];O.append(dict(src=s['src'],ok=bool(H),sure=any(not HEDGE.search(x) for x in H),loose=any(has(g,p) for g in s['ans'])))
    hd=bool(HEDGE.search(p or ''));wr=not all(o['ok'] for o in O) or any(o['src']=='none' for o in O)
    return dict(slots=O,full=all(o['ok'] for o in O),full_sure=all(o.get('sure',o['ok']) for o in O),answered=bool((p or '').strip()) and any(not HEDGE.search(x) for x in S),hedged=hd,hedge_warranted=hd and wr,overclaim=not hd and not all(o['ok'] for o in O))
def run(a):
    from adam_t_eval import E2E
    D=[json.loads(l) for l in open(DEV)];D=[d for d in D if a.split=='all' or d['split']==a.split][:a.limit or None];M=E2E(f"{a.bin} {a.gf17}");H=__import__('amni.serve.adam_t_harness',fromlist=['run']);orun=H.run;cap={}
    def hr(*x,**k):r=orun(*x,**k);cap['h']=r;return r
    H.run=hr;fo=open(a.out,'a');done={(r['id'],r['cond']) for r in (json.loads(l) for l in open(a.out))} if os.path.getsize(a.out) else set();t0=time.time()
    for j,d in enumerate(D):
        for cond in a.conds.split(','):
            if (d['id'],cond) in done:continue
            os.environ['AMNI_ADAM_T_HARNESS']='1' if cond=='on' else '0';u=f"hx-{M.stamp}-{d['id']}-{cond}";h=[tuple(t) for t in d['turns']];[M.U.turn(q,h[:i],u,u) for i,(q,_) in enumerate(h)];M.cap.clear();cap.clear();t=time.perf_counter()
            o=M.G.chat(d['q'],history=h,max_new_tokens=120,do_sample=False,user=u,session=u);ms=(time.perf_counter()-t)*1000;p=(o.get('answer') or '').strip();hh=cap.get('h')
            fo.write(json.dumps({'id':d['id'],'cond':cond,'kind':d['kind'],'split':d['split'],'q':d['q'],'pred':p,'tier':o.get('tier'),'route':M.cap.get('route') or '','ms':round(ms,1),'hn':hh and hh['n'],'hslots':hh and [{k:v for k,v in s.items() if k in ('kind','src','ok','conf','hedge','retry','exp','conflict','span')} for s in hh['slots']],**score(d,p)})+'\n');fo.flush()
        (j+1)%10 or print(f"{j+1}/{len(D)} {time.time()-t0:.0f}s",flush=True)
    M.close()
def ci(k,n):return {'rate':round(k/n,4) if n else None,'n':n,'se':round((k/n*(1-k/n)/n)**0.5,4) if n else None}
def summ(X:list)->dict:
    S=[o for x in X for o in x['slots']];ans=[o for o in S if o['src']!='none'];no=[o for o in S if o['src']=='none'];mp=[x for x in X if len(x['slots'])>1];hd=[x for x in X if x['hedged']];un=[x for x in X if not x['hedged']];ms=[x['ms'] for x in X]
    return {'n':len(X),'answered':ci(sum(x['answered'] for x in X),len(X)),'slot_correct':ci(sum(o['ok'] for o in ans),len(ans)),'slot_correct_by_src':{s:ci(sum(o['ok'] for o in ans if o['src']==s),sum(o['src']==s for o in ans)) for s in ('mem','know','calc')},'slot_correct_unhedged':ci(sum(o['sure'] for o in ans),len(ans)),'slot_mentioned_loose':ci(sum(o['loose'] for o in ans),len(ans)),'item_all_correct':ci(sum(x['full'] for x in X),len(X)),'multipart_fully_answered':ci(sum(x['full'] for x in mp),len(mp)),'multipart_fully_answered_unhedged':ci(sum(x['full_sure'] for x in mp),len(mp)),'false_memory':ci(sum(o['fm'] for o in no),len(no)),'hedge_rate':ci(len(hd),len(X)),'hedge_warranted':ci(sum(x['hedge_warranted'] for x in hd),len(hd)),'overclaim_unhedged':ci(sum(x['overclaim'] for x in un),len(un)),'latency_ms':{'p50':round(float(np.median(ms)),1) if ms else None,'p90':round(float(np.percentile(ms,90)),1) if ms else None,'mean':round(float(np.mean(ms)),1) if ms else None},'routes':dict(collections.Counter(x['route'] for x in X))}
def report(a):
    R=[json.loads(l) for l in open(a.out)];C=sorted({r['cond'] for r in R});o={c:{sp:{**summ([r for r in R if r['cond']==c and (sp=='all' or r['split']==sp)]),'by_kind':{k:summ([r for r in R if r['cond']==c and (sp=='all' or r['split']==sp) and r['kind']==k]) for k in KINDS}} for sp in ('test','tune','all')} for c in C}
    on=[r for r in R if r['cond']=='on' and r.get('hslots')];o['harness_slots']={sp:{'n':len(Z),'by_kind':dict(collections.Counter(z['kind'] for z in Z)),'by_src':dict(collections.Counter(z.get('src') for z in Z)),'verified':ci(sum(bool(z.get('ok')) for z in Z),len(Z)),'hedged':ci(sum(bool(z.get('hedge')) for z in Z),len(Z)),'retried':ci(sum(bool(z.get('retry')) for z in Z),len(Z)),'conflict':sum(bool(z.get('conflict')) for z in Z)} for sp in ('test','tune','all') for Z in [[z for r in on if sp=='all' or r['split']==sp for z in r['hslots']]]}
    json.dump(o,open(a.summary,'w'),indent=1);K=('answered','slot_correct','item_all_correct','multipart_fully_answered','false_memory','hedge_rate','hedge_warranted','overclaim_unhedged')
    for sp in ('test','tune','all'):print(sp,' | '.join(f"{c}: "+' '.join(f"{k}={o[c][sp][k]['rate']}" for k in K)+f" p50={o[c][sp]['latency_ms']['p50']}" for c in C))
def main():
    ap=argparse.ArgumentParser(description='ADAM-T-HARNESS dev eval (own CC0 set; harness on vs off through gguf_runtime.chat)');ap.add_argument('cmd',choices=('build','run','report'));ap.add_argument('--src',default='');ap.add_argument('--gf17',default='');ap.add_argument('--bin',default='');ap.add_argument('--conds',default='on,off');ap.add_argument('--split',default='all',choices=('all','tune','test'));ap.add_argument('--limit',type=int,default=0);ap.add_argument('--out',default=os.path.join(OUT,'results.jsonl'));ap.add_argument('--summary',default=os.path.join(OUT,'summary.json'));a=ap.parse_args()
    a.cmd=='run' and not os.path.exists(a.out) and open(a.out,'w').close();{'build':build,'run':run,'report':report}[a.cmd](a)
if __name__=='__main__':main()
