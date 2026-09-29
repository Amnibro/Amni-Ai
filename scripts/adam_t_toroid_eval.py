import os,re,sys,json,time,glob,argparse,tempfile,itertools
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amni.serve import user_toroid as T,user_memory as U
from amni.serve.adam_t_client import Proto,template,STOPS,_fit,_mem
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));DEV=os.path.join(ROOT,'data','adam_t','eval','toroid','toroid_dev.jsonl');OUT=os.path.join(ROOT,'data','adam_t','eval','toroid')
LIC='CC0-1.0 (written from scratch for ADAM-T USER-TOROID, 2026-09-28; no dataset text)';LIC2='CC0-1.0 (written from scratch for ADAM-T TOROID-HARDEN, 2026-09-29; no dataset text)'
DEV2=os.path.join(OUT,'toroid_dev2.jsonl');SRCS=[os.path.join(ROOT,'data','adam_t','eval',*x) for x in (('chat','chat_src.jsonl'),('toroid','toroid_dev.jsonl'),('harness','harness_dev.jsonl'))]
_W=re.compile(r"[a-z0-9]+");STV={'the','a','an','my','his','her','their','our','on','at','in','to','from','by','of','is','it','its'}
nt=lambda s:_W.findall(str(s).lower());core=lambda s:[x for x in nt(s) if x not in STV] or nt(s)
CLAIM=re.compile(r"(?i)\byou (?:told|said|mentioned)|\byour [\w' ]{1,30}? (?:is|are|was|lives|works|name|goes)\b|\bif i remember\b|\byou (?:live|work|drive|moved|go|get|take|keep|parked|pay|cook|make|use)\b|^(?:because|by|with|on|in|at|it's)\b")
ABST=re.compile(r"(?i)\b(?:do(?:n'?t| not) know|have(?:n'?t| not) (?:told|mentioned|said|shared)|did(?:n'?t| not) (?:tell|mention|say)|not sure|no idea|do(?:n'?t| not) have|do(?:n'?t| not) remember|unknown|not told|forgot|can'?t remember|never told|no record|nothing saved|not something you)")
REL={3:('where','location'),4:('where','home'),5:('what','job'),6:('what','name'),7:('what','likes'),8:('what','allergy'),9:('what','model'),10:('what','color'),11:('what','favorite'),12:('what','kind'),13:('where','school'),14:('what','age'),15:('when',''),2:('what','dislikes')}
DEMO={'slot':["User: Memory: your bike, color: green. What color is my bike?\nAdam: Your bike is green.\n\n","User: Memory: your cousin, home: Leeds. Where does my cousin live?\nAdam: Your cousin lives in Leeds.\n\n","User: Memory: your boss, why: unknown. Why did my boss quit?\nAdam: I don't know, you haven't told me that yet.\n\n"]}
def vmatch(a,b):
    A,B=set(core(a)),set(core(b))
    if not A or not B:return False
    i=len(A&B);return i/len(A|B)>=0.5 or A<=B and i/len(B)>=0.34 or B<=A and i/len(A)>=0.34
def ematch(p,g):
    p,g=(' '.join(T._sg(x.replace("'s",'')) for x in str(z).lower().split()) for z in (p,g));return p==g or bool(p) and bool(g) and (p.split()[-1]==g.split()[-1] or g in p.split() or p in g.split() or g=='me' and p.split()[0]=='favorite')
def inr(a,r):
    c=core(a);rr=' '+' '.join(nt(r))+' ';return (' '+' '.join(c)+' ') in rr or len(set(c))>=3 and len(set(c)&set(nt(r)))/len(set(c))>=0.6
def ans_ok(reply,d):
    bad=[b for b in d['bad'] if not any(set(core(b))<=set(core(a)) for a in d['ans'])];bh=any(inr(b,reply) for b in bad)
    return (any(inr(a,reply) for a in d['ans']) and not bh) if d['ans'] else (not bh and bool(ABST.search(reply)))
def _strs(o,k=''):return [(k,o)] if isinstance(o,str) else [x for kk,v in (o.items() if isinstance(o,dict) else enumerate(o) if isinstance(o,list) else []) for x in _strs(v,kk if isinstance(kk,str) else k)]
def ngrams(ws,n=6):return {tuple(ws[i:i+n]) for i in range(len(ws)-n+1)}
def dtexts(d):return [t['u'] for t in d['turns']]+[t['a'] for t in d['turns']]+[d['q']]
def build(a):
    D=[json.loads(l) for f in sorted(glob.glob(os.path.join(a.src,'*.jsonl'))) for l in open(f,encoding='utf-8') if l.strip()]
    v2=os.path.abspath(a.dev)!=os.path.abspath(DEV);G=set();lic=LIC2 if v2 else LIC
    for f in SRCS[:3 if v2 else 1]:
        for l in open(f,encoding='utf-8'):l.strip() and [G.update(ngrams(nt(v))) for k,v in _strs(json.loads(l)) if k!='license']
    bad=[];ov=[]
    for i,d in enumerate(D):
        d['split']='tune' if int(re.findall(r'\d+',d['id'])[-1])%2 else 'test';d['license']=lic
        o=[g for x in dtexts(d) for g in ngrams(nt(x)) if g in G];o and ov.append((d['id'],[' '.join(g) for g in o][:3]))
        for t in d['turns']:[bad.append((d['id'],'val',f['val'])) for f in t.get('facts',[]) if f['val'].lower() not in t['u'].lower()]
        [bad.append((d['id'],'ans',x)) for x in d['ans'] if not any(x.lower() in t['u'].lower() for t in d['turns'])]
    os.makedirs(OUT,exist_ok=True);open(a.dev,'w',encoding='utf-8').write(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in D))
    m={'n':len(D),'by_w':{w:sum(d['w']==w for d in D) for w in T.WS},'by_kind':{k:sum(d['kind']==k for d in D) for k in sorted({d['kind'] for d in D})},'by_style':{k:sum(d.get('style')==k for d in D) for k in sorted({d.get('style','') for d in D})},'split':{s:sum(d['split']==s for d in D) for s in ('tune','test')},'sixgram_overlap_chat_src' if not v2 else 'sixgram_overlap_chat_src_toroid_dev_harness_dev':len(ov),'overlap_items':ov,'validation':bad,'license':lic,'src_ngrams':len(G)}
    json.dump(m,open(a.dev[:-6]+'_meta.json' if v2 else os.path.join(OUT,'toroid_meta.json'),'w'),indent=1);print(json.dumps({k:v for k,v in m.items() if k!='overlap_items'},indent=1))
def gold_line(d):
    fs={};fg=set()
    for t in d['turns']:
        f=t.get('forget')
        if f:[fs.pop(k) for k in [k for k in fs if ematch(k[0],f['ent']) and (not f.get('w') or k[1]==f['w'])]]
        for x in t.get('facts',[]):fs[(x['ent'],x['w'],x.get('attr',''))]=x['val']
    k=[k for k in fs if ematch(k[0],d['q_ent']) and k[1]==d['q_w'] and (not d.get('q_attr') or k[2]==d['q_attr'] or not k[2])]
    k=sorted(k,key=lambda z:(z[2]==d.get('q_attr',''),list(fs).index(z)))[-1:] if k else []
    return T.line(d['q_ent'],d['q_w'],d.get('q_attr',''),fs[k[0]]) if k and d['ans'] else T.line(d['q_ent'],d['q_w'],d.get('q_attr',''),'unknown')
def old_facts(d):
    p=U._P(lt=True,gn=True,g2=True);R=[]
    for t in d['turns']:
        fg=U._fgt(p,t['u']);fs=[] if fg else U.keep(p.turn(U._enc(t['u']).strip()),t['u']);R.append([('me' if f[0]==U.SELFK else p.names.get(f[0],'') or '?',)+REL.get(f[1],('what',''))+(f[2].decode('latin-1'),) for f in fs])
    return R
def run_store(d,fw=None):
    st=T.Store();P=[]
    for t in d['turns']:
        k,r=T.observe(st,t['u']);P.append([(e,w,a,v) for e,_,w,a,v,_ in r] if k=='fact' else [])
    return st,P
def pr(P,G):
    tp=fp=fn=0;bw={w:[0,0,0] for w in T.WS}
    for p,g in zip(P,G):
        used=set()
        for x in p:
            m=next((j for j,y in enumerate(g) if j not in used and ematch(x[0],y['ent']) and x[1]==y['w'] and vmatch(x[3],y['val'])),None)
            if m is None and any(ematch(x[0],y['ent']) and x[1]==y['w'] and vmatch(x[3],y['val']) for y in g):continue
            (used.add(m),bw[x[1]].__setitem__(0,bw[x[1]][0]+1)) if m is not None else bw[x[1]].__setitem__(1,bw[x[1]][1]+1)
        for j,y in enumerate(g):j not in used and bw[y['w']].__setitem__(2,bw[y['w']][2]+1)
    f=lambda a,b:round(a/max(1,a+b),3);tp,fp,fn=(sum(v[i] for v in bw.values()) for i in range(3))
    return {'precision':f(tp,fp),'recall':f(tp,fn),'tp':tp,'fp':fp,'fn':fn,'by_w':{w:{'p':f(v[0],v[1]),'r':f(v[0],v[2]),'n_gold':v[0]+v[2]} for w,v in bw.items()}}
def ret_ok(v,d):return (bool(v) and any(vmatch(v,a) for a in d['ans']) and not any(vmatch(v,b) and not any(vmatch(b,a) for a in d['ans']) for b in d['bad'])) if d['ans'] else not v
def cal(ps,ys,nb=10):
    ps,ys=np.array(ps,float),np.array(ys,float);b=np.minimum((ps*nb).astype(int),nb-1);rows=[(i/nb,int((b==i).sum()),float(ps[b==i].mean()) if (b==i).any() else None,float(ys[b==i].mean()) if (b==i).any() else None) for i in range(nb)]
    return {'ece':round(float(sum(n*abs(p-y) for _,n,p,y in rows if n)/max(1,len(ps))),4),'brier':round(float(((ps-ys)**2).mean()),4) if len(ps) else None,'n':len(ps),'acc':round(float(ys.mean()),4) if len(ps) else None,'mean_p':round(float(ps.mean()),4) if len(ps) else None,'bins':[(lo,n,round(p,3) if p is not None else None,round(y,3) if y is not None else None) for lo,n,p,y in rows if n]}
def soft_eval(D,S,fw):
    R=[]
    for d,(st,_) in zip(D,S):
        t0=time.perf_counter();kn=st.ents();qa=T.ask(d['q'],T._last(st),list(kn),st.names,st.lp);L=T.soft(st,qa,list(kn),st.names,'',fw) if qa else [];Sx,top=T._top(L);pn=max([x[0] for x in L if x[1] is None] or [1.0])
        v=top[1]['v'] if top else '';R.append((top[0] if top else pn,ret_ok(v,d),(time.perf_counter()-t0)*1000))
    return R
def _pol(D,S,ix,gate=None):
    R=[(D[i],T.respond(S[i][0],D[i]['q'],gate=gate)) for i in ix];f=lambda v:round(float(np.mean(v)),4) if len(v) else None;dr=[(d,o) for d,o in R if o['kind']=='direct'];un=[(d,o) for d,o in R if not d['ans']]
    return {'direct_rate':f([o['kind']=='direct' for _,o in R]),'direct_precision':f([ret_ok(o.get('v',''),d) for d,o in dr]),'n_direct':len(dr),'false_recall_direct':f([o['kind']=='direct' for _,o in un]),'false_claim_any_line':f([o['kind'] in ('direct','hint','unsure') for _,o in un]),'n_nofact':len(un)}
def _fitc(X,y,l2=1.0,it=60):
    A=np.array([[x[k] for k in T.CK]+[1.0] for x in X],float);y=np.array(y,float);w=np.zeros(A.shape[1]);w[0]=1.0;w0=w.copy()
    for _ in range(it):q=1/(1+np.exp(-np.clip(A@w,-30,30)));H=A.T@(A*(q*(1-q))[:,None])+l2*np.eye(len(w));w-=np.linalg.solve(H,A.T@(q-y)+l2*(w-w0))
    return {**{'k_'+k:round(float(v),4) for k,v in zip(T.CK,w[:-1])},'kb':round(float(w[-1]),4)}
def _platt(p,y):
    x=np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)));G=[(a,b) for a in np.arange(0.05,2.01,0.05) for b in np.arange(-3,3.01,0.1)];L=[-np.mean(y*np.log(np.clip(q,1e-6,1))+(1-y)*np.log(np.clip(1-q,1e-6,1))) for a,b in G for q in [1/(1+np.exp(-(a*x+b)))]]
    return tuple(round(float(v),3) for v in G[int(np.argmin(L))])
PRIOR=dict(pmis=-1.5,ent=2.0,own=1.0,att=1.0,attc=0.5,att_bad=-1.0,pred=1.0,rec=-0.35,sup=-3.0,slot=1.5,subj=0.5,co=0.8,cu=0.3,pri=1.0,own2=0.7,csup=-1.5,wild=-0.8,cor=0.0,ci=0.0,vlen=0.0,ment=0.0,nco=0.0,psyn=0.5,xtyp=-0.5,tonly=-1.0,null=1.5,tau=1.0)
def _fitw(D,S,ix,fw0,l2=0.01,it=400,lr=0.1):
    fw0={**fw0,**PRIOR}
    X=[];Y=[]
    for i in ix:
        st,_=S[i];kn=st.ents();qa=T.ask(D[i]['q'],T._last(st),list(kn),st.names,st.lp);L=T.lookup(st,qa) if qa else [];sv=T._row(st,L[0][1])['v'] if L and L[0][0]>=2.0 else ''
        C=T.cands(st,qa,list(kn),st.names,sv) if qa and qa['ents'] else [];y=np.array([ret_ok(sp['v'],D[i]) for _,sp in C],bool)
        if D[i]['ans'] and not y.any():continue
        X.append(np.array([[f[k] for k in T.FK] for f,_ in C],float).reshape(-1,len(T.FK)));Y.append(y)
    x0=np.array([fw0[k] for k in T.FK]+[fw0['null']],float)/fw0['tau']
    def obj(x):
        w,b=x[:-1],x[-1];L=0.0;g=np.zeros_like(x)
        for F,y in zip(X,Y):
            z=np.concatenate([F@w,[b]]);p=np.exp(z-z.max());p/=p.sum();t=np.concatenate([y,[not y.any()]]).astype(float);q=p*t;L-=np.log(max(q.sum(),1e-12));r=q/max(q.sum(),1e-12);d=p-r;g[:-1]+=F.T@d[:-1];g[-1]+=d[-1]
        return L/len(X)+0.5*l2*((x-x0)**2).sum(),g/len(X)+l2*(x-x0)
    x=minimize(obj,x0,jac=True,method='L-BFGS-B').x
    return {**{k:round(float(v),4) for k,v in zip(T.FK,x[:-1])},'null':round(float(x[-1]),4),'tau':1.0}
def offline(a):
    D=[json.loads(l) for l in open(a.dev,encoding='utf-8')]+[{**json.loads(l),'id':'v1:'+json.loads(l)['id'],'split':'v1'} for x in filter(None,a.extra.split(',')) for l in open(x,encoding='utf-8')];os.environ.update(AMNI_UMEM_LT='1',AMNI_UMEM_GN='1',AMNI_UMEM_G2='1');SP=a.splits.split(',');D=[d for d in D if d['split'] in SP or a.tune and d['split'] in ('tune','v1')]
    t0=time.perf_counter();S=[run_store(d) for d in D];ing=(time.perf_counter()-t0)*1000/max(1,sum(len(d['turns']) for d in D));res={'n':len(D),'ingest_ms_per_turn':round(ing,3)}
    for sp in SP:
        ix=[i for i,d in enumerate(D) if sp=='all' or d['split']==sp];G=[[f for f in t.get('facts',[])] for i in ix for t in D[i]['turns']];res.setdefault('extract',{})[sp]={'toroid':pr([p for i in ix for p in S[i][1]],G),'user_memory':pr([p for i in ix for p in old_facts(D[i])],G) if a.old else None}
    fw=dict(T.FW);tune=[i for i,d in enumerate(D) if d['split'] in ('tune','v1')]
    if a.tune:
        fw={**fw,'kb':0.0,**{'k_'+c:float(c=='lp') for c in T.CK},**_fitw(D,S,tune,fw)};T.FW.update(fw);ps=[];ys=[]
        for i in tune:o=T.recall(S[i][0],D[i]['q'],'both',False,a.theta,fw);v=re.sub(r"^Memory: [^:]*: ",'',o['line']).rstrip('.').split(' or ')[0] if o['line'] else '';o['line'] and o.get('cx') and (ps.append(o['cx']),ys.append(ret_ok(v,D[i])))
        fw.update(_fitc(ps,ys));T.FW.update(fw);res['fw_tuned_on_tune']={k:fw[k] for k in T.FK+('null','tau','kb')+tuple('k_'+c for c in T.CK)}
    T.FW.update(fw)
    for sp in SP:
        ix=[i for i,d in enumerate(D) if sp=='all' or d['split']==sp];r={}
        for tier in ('slot','soft','both'):
            ok=[];lat=[];ps=[];ys=[];sup=[];byw={w:[] for w in T.WS};byk={};bys={};lp_=[];ly_=[]
            for i in ix:
                d,(st,_)=D[i],S[i];t1=time.perf_counter();o=T.recall(st,d['q'],tier,False,a.theta,fw);lat.append((time.perf_counter()-t1)*1000)
                v=(o['slot']['v'] if o['slot'] else '') if tier=='slot' else (o['soft'][0][1]['v'] if o['soft'] and o['line'] and tier=='soft' else '') if tier=='soft' else (re.sub(r"^Memory: [^:]*: ",'',o['line']).rstrip('.').split(' or ')[0] if o['line'] else '')
                y=ret_ok(v,d);ok.append(y);byw[d['w']].append(y);byk.setdefault(d['kind'],[]).append(y);bys.setdefault(d.get('style','v1'),[]).append(y);d['kind']=='correction' and sup.append(y)
                tier!='slot' and (ps.append(o['p'] if o['line'] else o.get('pnull',1.0)),ys.append(y),lp_.append(o['p']) if o['line'] else None,ly_.append(y) if o['line'] else None)
            r[tier]={'hit1':round(float(np.mean(ok)),4),'answerable_hit1':round(float(np.mean([y for i2,y in zip(ix,ok) if D[i2]['ans']])),4),'abstain_ok':round(float(np.mean([y for i2,y in zip(ix,ok) if not D[i2]['ans']])),4),'supersession_ok':round(float(np.mean(sup)),4) if sup else None,'by_w':{w:round(float(np.mean(v)),3) for w,v in byw.items() if v},'by_kind':{k:round(float(np.mean(v)),3) for k,v in byk.items()},'by_style':{k:round(float(np.mean(v)),3) for k,v in bys.items()},'ms_p50':round(float(np.median(lat)),3),'ms_p95':round(float(np.percentile(lat,95)),3),**({'calibration':cal(ps,ys)} if ps else {}),**({'calibration_lines':cal(lp_,ly_)} if lp_ else {})}
        r['policy']={str(g):_pol(D,S,ix,g) for g in (0.6,0.7,0.75,0.8,0.85,0.9,0.95)};res[sp]=r
    json.dump(res,open(a.out,'w'),indent=1);print(json.dumps({k:(v if k not in SP else {t:({x:y for x,y in v[t].items() if x not in ('calibration','by_kind')} if t!='policy' else v[t]['0.9']) for t in v}) for k,v in res.items() if k!='extract'},indent=1));print(json.dumps({sp:{k:{x:y for x,y in v.items() if x!='by_w'} if v else v for k,v in e.items()} for sp,e in res['extract'].items()}))
def model(a):
    D=[json.loads(l) for l in open(a.dev,encoding='utf-8')];D=[d for d in D if a.split in ('all',d['split'])][:a.limit or None]
    tmp=tempfile.mkdtemp(dir=a.tmp);os.environ.update(AMNI_UMEM_DIR=tmp,AMNI_UMEM_LT='1',AMNI_UMEM_GN='1',AMNI_UMEM_G2='1',AMNI_USER_MEMORY='1');fw={**T.FW,**(json.load(open(a.fw)).get('fw_tuned_on_tune',{}) if a.fw else {})}
    p=Proto([a.bin,'serve',a.gf17],120,{'RAYON_NUM_THREADS':str(a.threads),'PATH':'/usr/bin:/bin'},None,600);C=a.conds.split(',');done=set()
    if os.path.exists(a.out):done={(r['id'],r['set'],r['cond']) for r in map(json.loads,open(a.out,encoding='utf-8'))}
    fo=open(a.out,'a',encoding='utf-8')
    for d in D:
        uid=f"tor-{d['id']}";hist=[];st=T.Store()
        for t in d['turns']:U.turn(t['u'],hist,user=uid);T.observe(st,t['u']);hist.append((t['u'],t['a']))
        gl=gold_line(d)
        for S in a.sets.split(','):
            h=hist if S=='same' else [];m=_mem(d['q'],h,uid,None);pre=_fit(m['pre'],0,640)[0] if m['route'] else [];hh=m['hist']
            for c in C:
                if (d['id'],S,c) in done:continue
                tier={'slot0':'slot','slotF':'slot','soft0':'soft','both0':'both','bothF':'both','bothU':'both','bothUF':'both','turnB':'both'}.get(c);t1=time.perf_counter();o=T.recall(st,d['q'],tier,c.startswith('bothU'),a.theta,fw) if tier else {'line':'','p':0.0};rms=(time.perf_counter()-t1)*1000
                L=gl if c.startswith('gold') else o['line'];few=c.endswith('F');rc=[L] if L else [];pp=([] if L and c!='cur' else pre) if c!='cur' else pre;pp=(DEMO['slot']+list(pp)) if few else list(pp)
                if c=='turnB':rc,pp=[],([f"User: {o['src']}\nAdam: Got it, thanks for telling me.\n\n"] if o.get('src') and L else list(pre))
                if c.startswith('direct'):t1=time.perf_counter();o=T.respond(st,d['q'],abstain=c=='directA',unsure=2.0 if c=='directU' else 0.25,unk=c=='directU');rms=(time.perf_counter()-t1)*1000;L=o['line'] if o['kind'] in ('hint','unsure','unknown') else '';rc,pp=([L],[]) if L else ([],list(pre));o['p']=o['conf'];o['line']=o['kind']
                pr_=template(d['q'],hh,a.budget,pp,rc);t2=time.perf_counter();r=o['text'] if c.startswith('direct') and o['kind']=='direct' else p.text(pr_,a.max_bytes,STOPS).strip();gms=(time.perf_counter()-t2)*1000
                fo.write(json.dumps({'id':d['id'],'split':d['split'],'w':d['w'],'kind':d['kind'],'set':S,'cond':c,'line':L,'route':o.get('kind',''),'p':round(float(o.get('p',0.0) if o.get('line') else 1-o.get('pnull',1.0)),4),'reply':r,'ok':ans_ok(r,d),'abst':bool(ABST.search(r)),'recall_ms':round(rms,3),'gen_ms':round(gms,1),'prompt_bytes':len(pr_.encode())})+'\n');fo.flush()
    p.close()
def cover(a):
    D=[json.loads(l) for l in open(a.dev,encoding='utf-8')];tmp=tempfile.mkdtemp(dir=a.tmp);os.environ.update(AMNI_UMEM_DIR=tmp,AMNI_UMEM_LT='1',AMNI_UMEM_GN='1',AMNI_UMEM_G2='1',AMNI_USER_MEMORY='1');fw={**T.FW,**(json.load(open(a.fw)).get('fw_tuned_on_tune',{}) if a.fw else {})};R={}
    for d in D:
        uid=f"cov-{d['id']}";hist=[];st=T.Store()
        for t in d['turns']:U.turn(t['u'],hist,user=uid);T.observe(st,t['u']);hist.append((t['u'],t['a']))
        o=T.recall(st,d['q'],'both',False,a.theta,fw)
        for S in ('same','cross'):
            m=_mem(d['q'],hist if S=='same' else [],uid,None);pre=''.join(_fit(m['pre'],0,640)[0] if m['route'] else []);ctx=pre+''.join(f"{u} {x} " for u,x in (m['hist'] if S=='same' else []))
            k=(S,d['split']);R.setdefault(k,[]).append((any(inr(x,ctx) for x in d['ans']) if d['ans'] else None,sum(inr(x,ctx) for x in d['bad']),bool(pre),len(pre.encode()),any(inr(x,o['line']) for x in d['ans']) if d['ans'] else None,sum(inr(x,o['line']) for x in d['bad'])))
    out={f'{S}/{sp}':{'answer_in_context':round(np.mean([x[0] for x in v if x[0] is not None]),4),'bad_values_in_context_mean':round(np.mean([x[1] for x in v]),3),'pre_nonempty':round(np.mean([x[2] for x in v]),3),'pre_bytes_mean':round(np.mean([x[3] for x in v]),1),'toroid_line_has_answer':round(np.mean([x[4] for x in v if x[4] is not None]),4),'toroid_line_bad_values_mean':round(np.mean([x[5] for x in v]),3)} for (S,sp),v in sorted(R.items())}
    json.dump(out,open(a.out,'w'),indent=1);print(json.dumps(out,indent=1))
def report(a):
    R=[json.loads(l) for l in open(a.out,encoding='utf-8')];D={json.loads(l)['id']:json.loads(l) for l in open(a.dev,encoding='utf-8')};C=list(dict.fromkeys(r['cond'] for r in R));res={}
    for sp in ('test','tune','all'):
        for S in ('same','cross'):
            for c in C:
                X=[r for r in R if r['set']==S and r['cond']==c and sp in ('all',r['split'])]
                if not X:continue
                an=[r for r in X if D[r['id']]['ans']];un=[r for r in X if not D[r['id']]['ans']];k=f'{sp}/{S}/{c}'
                f=lambda v:round(float(np.mean(v)),4) if len(v) else None;res[k]={'n':len(X),'acc':f([r['ok'] for r in X]),'ans_acc':f([r['ok'] for r in an]),'abstain_acc':f([r['ok'] for r in un]),'tpl_on_answerable':f([r['abst'] and not r['ok'] for r in an]),'false_recall':f([not r['abst'] and (bool(CLAIM.search(r['reply'])) or any(inr(b,r['reply']) for b in D[r['id']]['bad'])) for r in un]),'direct_rate':f([r.get('route')=='direct' for r in X]) if c.startswith('direct') else None,'direct_precision':f([r['ok'] for r in X if r.get('route')=='direct']) if c.startswith('direct') else None,'by_route':{k2:[len([r for r in X if r.get('route')==k2]),f([r['ok'] for r in X if r.get('route')==k2])] for k2 in ('direct','hint','unsure','unknown','none')} if c.startswith('direct') else None,'by_w':{w:f([r['ok'] for r in X if r['w']==w]) for w in T.WS if any(r['w']==w for r in X)},'by_kind':{q:f([r['ok'] for r in X if r['kind']==q]) for q in sorted({r['kind'] for r in X})},'gen_ms_p50':round(float(np.median([r['gen_ms'] for r in X])),1),'recall_ms_p50':round(float(np.median([r['recall_ms'] for r in X])),3),'prompt_bytes_p50':int(np.median([r['prompt_bytes'] for r in X]))}
                if c.startswith(('soft','both','direct')):Y=[r for r in X if r['line'] or r.get('route') in ('direct','hint','unsure','unknown')];res[k]['calibration_vs_model']=cal([min(1.0,max(0.0,r['p'])) for r in Y],[r['ok'] for r in Y],5)
    json.dump(res,open(a.summary,'w'),indent=1);[print(k,{x:y for x,y in v.items() if x not in ('by_kind','calibration_vs_model')}) for k,v in res.items() if k.startswith('test/') or k.startswith('all/')]
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd');ap.add_argument('--src',default=os.path.join(ROOT,'scratchpad','toroid','devset'));ap.add_argument('--dev',default=DEV);ap.add_argument('--out',default=os.path.join(OUT,'offline.json'));ap.add_argument('--summary',default=os.path.join(OUT,'model_summary.json'));ap.add_argument('--tune',action='store_true');ap.add_argument('--theta',type=float,default=0.5);ap.add_argument('--fw',default='');ap.add_argument('--extra',default='');ap.add_argument('--splits',default='tune,test,all');ap.add_argument('--old',action='store_true')
    ap.add_argument('--gf17',default='');ap.add_argument('--bin',default=os.path.join(ROOT,'scratchpad','toroid','bin','adamt-tor'));ap.add_argument('--threads',type=int,default=2);ap.add_argument('--budget',type=int,default=2928);ap.add_argument('--max_bytes',type=int,default=160);ap.add_argument('--conds',default='cur,slot0,slotF,soft0,both0,bothF,bothU,gold0,goldF,turnB');ap.add_argument('--sets',default='same,cross');ap.add_argument('--split',default='all');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--tmp',default=os.path.join(ROOT,'scratchpad','toroid'))
    a=ap.parse_args();{'build':build,'offline':offline,'model':model,'report':report,'cover':cover}[a.cmd](a)
