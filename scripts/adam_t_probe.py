import os,re,sys,ast,json,math,time,random,argparse,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from amni.serve.adam_t_client import Proto,B64,U64,template,STOPS
H=next((p for p in (Path(os.environ.get('AMNI_HOME') or ROOT),ROOT,*ROOT.parents) if (p/'data'/'clean2'/'bpe4096.json').is_file()),ROOT)
PD=ROOT/'data'/'adam_t'/'probe';BUD=2928;SEED=20260928;N=30;ACK='Got it, thanks for telling me.'
FIRST="Anna Max Sam Tom Jack Emma Lucy Mary John Paul Mark Kate Jane Ben Leo Mia Noah Ella Luke Ryan Sarah Laura Grace Rose Alice Peter David James Henry Oliver Sophie Chloe Daniel Hannah Nora Ruby Oscar Felix Iris Hugo Clara Victor Julia Simon Martin Diana Helen Frank Arthur Walter Irene Edith Harold Gloria Susan Carol Linda Karen Nancy Betty Ruth Joyce Evelyn Albert Louis Philip Eric Carl Roger Keith Dennis Gary Larry Scott Wayne".split()
LAST="Smith Brown Taylor Wilson Clark Hall Young King Wright Green Baker Adams Nelson Hill Carter Mitchell Turner Parker Collins Morris Cooper Reed Bailey Bell Murphy Rivera Ward Cox Howard Brooks Gray Watson Price Bennett Wood Barnes Ross Henderson Coleman Jenkins Perry Powell Long Patterson Hughes Flores Butler Simmons Foster Bryant Russell Griffin Hayes Myers Ford Hamilton Graham Sullivan Wallace West Cole Jordan Reynolds Marshall Owens Fisher Harper Lawson Dixon Hunt Palmer Wells Fox".split()
CITY="Paris London Rome Berlin Boston Dublin Madrid Vienna Chicago Denver Seattle Houston Dallas Atlanta Miami Phoenix Oslo Lisbon Prague Warsaw Athens Cairo Tokyo Delhi Sydney Toronto Montreal Glasgow Bristol Leeds Oxford Cambridge Portland Austin Memphis Nashville Detroit Baltimore Richmond Orlando Tampa Sacramento Pittsburgh Cleveland Milwaukee Minneapolis Brisbane Adelaide Auckland Vancouver Edinburgh Manchester Liverpool Birmingham Hamburg Munich Frankfurt Geneva Zurich Naples Florence Venice Milan Seville Valencia Barcelona Brussels Amsterdam Stockholm Helsinki Budapest Bangkok Singapore Havana Lima Nairobi".split()
CITY2=["New York City","Salt Lake City","Los Angeles","San Francisco","Rio de Janeiro","Buenos Aires","Cape Town","Kansas City","Mexico City","Hong Kong","New Orleans","San Diego","Las Vegas","Fort Worth","Oklahoma City","Palm Springs","Santa Barbara","Colorado Springs","Little Rock","Baton Rouge","Grand Rapids","Sioux Falls","Cedar Rapids","Santa Fe","El Paso","Virginia Beach","Long Beach","Jersey City","Ann Arbor","Corpus Christi","Des Moines","Fort Lauderdale","Saint Petersburg","Port Elizabeth","Thunder Bay","Niagara Falls","Myrtle Beach","Sun Valley","Lake Charles","Green Bay"]
ONE_P="John Don May June April Art King Val Will Red Mar".split();ONE_L="London York Japan France India China Canada Germany Africa England America Europe".split()
PRE=["North","South","East","West","Upper","Lower","Old","New","Port","Lake","Mount","Fort","Saint"]
SY="ka ze lor vin tha mo rix quen dal bry sef nur og pim vesk tal orn jib wex yul fam cro stav gulm hesk zu pra dov ix um".split()
SYE="bren tav gol mar cul fen hal sor ped wim lan rud bex cor nel vas tir oph".split();SUF="ford mere holt wick dale stead vale moor ton bury haven crest field".split()
POSS=["neighbor","cousin","coworker","roommate","uncle","friend"]
FILL_T=[("I just got back from a long walk.","That sounds refreshing. A walk is a good way to clear your head."),("It rained all morning here.","Rainy mornings can be cozy. I hope it clears up for you."),("I am thinking about learning to cook.","That is a great idea. Start with a few simple dishes you enjoy."),("Do you like music?","I enjoy talking about music. What do you like to listen to?"),("I watched a movie last night.","Nice! Did you enjoy it?"),("Work was busy today.","Busy days can be tiring. I hope you get some rest tonight."),("I want to read more books this year.","That is a good goal. Even a few pages a day adds up."),("My coffee got cold again.","That happens a lot. A warm mug might help."),("I am planning a small trip.","That sounds fun. Where are you thinking of going?"),("The traffic was terrible today.","Traffic can be so frustrating. I hope the drive home is easier."),("I started a new puzzle.","Puzzles are relaxing. How many pieces does it have?"),("I feel a bit tired.","Try to get some good sleep tonight. Rest helps a lot."),("I planted some tomatoes.","Tomatoes are fun to grow. Keep them in the sun and water them often."),("Tell me a fun fact.","Octopuses have three hearts."),("I finished my homework early.","Well done! Now you have time to relax."),("I cleaned the whole kitchen.","That must feel good. A clean kitchen makes cooking easier."),("What should I do this weekend?","You could go for a hike, visit a friend, or try a new recipe."),("I am learning to play chess.","Chess is a great game. Practice the openings and look for simple tactics."),("The sunset was beautiful tonight.","Sunsets can be amazing. I am glad you got to see it."),("I bought new running shoes.","Nice! New shoes can make running more comfortable.")]
FILL_S="The area has mild winters and warm summers.|Many visitors come in the autumn to see the leaves.|A small market opens every Saturday morning.|The main road was repaved a few years ago.|Local schools teach classes in art and music.|Most of the land around it is used for farming.|A narrow bridge crosses the river near the center.|The library holds a large collection of old maps.|In spring the fields are covered with wild flowers.|Several walking trails start near the old mill.|The local bakery is known for its bread.|Fishing is still a popular hobby in the region.|A community garden was planted near the school.|The museum has a room full of old farm tools.|Evenings are quiet and the streets are safe.|The town hall was built from red brick.|Trains stop at the station twice a day.|A yearly fair brings music and games.|The hills behind it are good for hiking.|Many houses have small gardens in front.|The river floods in some wet years.|A bus line connects it to the nearest city.|The old church has a tall stone tower.|Summer nights are often warm and clear.|The park has a pond with ducks and geese.|Children often play football in the square.|The harbor is busy in the early morning.|Some families have lived there for many generations.|The climate is good for growing apples.|A small hospital serves the whole valley.".split('|')
UMEM=[("My dog's name is {v}.","What is my dog's name?","What do I call my dog?","Your dog's name is","My {p}'s dog's name is {v}.",'person'),("My best friend is {v}.","Who is my best friend?","Which friend am I closest to?","Your best friend is","My {p}'s best friend is {v}.",'person'),("My boat is named {v}.","What is my boat named?","What did I call my boat?","Your boat is named","My {p}'s boat is named {v}.",'person'),("My teacher's name is {v}.","What is my teacher's name?","Who teaches my class?","Your teacher's name is","My {p}'s teacher's name is {v}.",'person'),("My cat is named {v}.","What is my cat named?","What do I call my cat?","Your cat is named","My {p}'s cat is named {v}.",'person'),("My doctor is {v}.","Who is my doctor?","Which doctor do I see?","Your doctor is","My {p}'s doctor is {v}.",'person'),("My manager's name is {v}.","What is my manager's name?","Who is my boss at work?","Your manager's name is","My {p}'s manager's name is {v}.",'person'),("My hometown is {v}.","What is my hometown?","Where did I grow up?","Your hometown is","My {p}'s hometown is {v}.",'place'),("I live in {v}.","Where do I live?","Which city is my home now?","You live in","My {p} lives in {v}.",'place'),("My sister lives in {v}.","Where does my sister live?","Which city is my sister's home?","Your sister lives in","My {p}'s sister lives in {v}.",'place')]
PSG=[("The mayor of {E} is {v}.","Who is the mayor of {E}?","Who runs the town of {E}?","The mayor of {E} is",'person',"{E} is a small town in the hills."),("{E} was founded by {v}.","Who founded {E}?","Who started {E}?","{E} was founded by",'person',"{E} is an old fishing village on the coast."),("The lead singer of {E} is {v}.","Who is the lead singer of {E}?","Who sings for the band {E}?","The lead singer of {E} is",'person',"{E} is a rock band that formed in a garage."),("The board game {E} was designed by {v}.","Who designed the board game {E}?","Who created the board game {E}?","The board game {E} was designed by",'person',"{E} is a board game for two to four players."),("The captain of the ship {E} is {v}.","Who is the captain of the ship {E}?","Who commands the ship {E}?","The captain of the ship {E} is",'person',"{E} is a cargo ship that sails between two islands."),("The company {E} is based in {v}.","Where is the company {E} based?","In which city does the company {E} have its main office?","The company {E} is based in",'place',"{E} is a company that makes garden tools."),("The festival {E} is held in {v}.","Where is the festival {E} held?","Which city hosts the festival {E}?","The festival {E} is held in",'place',"{E} is a music festival that takes place every summer."),("The painter {E} was born in {v}.","Where was the painter {E} born?","What is the birthplace of the painter {E}?","The painter {E} was born in",'place',"{E} is a painter known for bright landscapes.")]
WD=[("capital","What is the capital of {E}?","Which city is the capital of {E}?","The capital of {E} is",'place'),("headquarters","Where is the headquarters of {E}?","In which city is {E} headquartered?","The headquarters of {E} is in",'place'),("founder","Who founded {E}?","Who was the founder of {E}?","{E} was founded by",'person'),("author","Who is the author of {E}?","Who wrote {E}?","{E} was written by",'person'),("director","Who is the director of {E}?","Who directed {E}?","{E} was directed by",'person'),("composer","Who is the composer of {E}?","Who wrote the music for {E}?","{E} was composed by",'person')]
REAL=[("France","Paris"),("Germany","Berlin"),("Italy","Rome"),("Spain","Madrid"),("Japan","Tokyo"),("Egypt","Cairo"),("Greece","Athens"),("Norway","Oslo"),("Portugal","Lisbon"),("Ireland","Dublin"),("Austria","Vienna"),("Poland","Warsaw"),("Russia","Moscow"),("China","Beijing"),("Kenya","Nairobi"),("Peru","Lima"),("Chile","Santiago"),("Cuba","Havana"),("Canada","Ottawa"),("Sweden","Stockholm"),("Finland","Helsinki"),("Denmark","Copenhagen"),("Hungary","Budapest"),("Thailand","Bangkok"),("Turkey","Ankara"),("Iran","Tehran"),("Iraq","Baghdad"),("Argentina","Buenos Aires"),("Belgium","Brussels"),("Netherlands","Amsterdam"),("Czech Republic","Prague"),("England","London")]
CALC=[("x","What is {a} x {b}?","Can you multiply {a} by {b} for me?","{a} times {b} is",lambda a,b:a*b),("+","What is {a} + {b}?","What do I get if I add {a} and {b}?","{a} plus {b} is",lambda a,b:a+b)]
CAN=[' I',' You',' Oh',' Got',' Thanks',' Good',' Nice',' Okay',' Sure',' My',' If',' Sorry',' Hmm',' Noted',' Great']
TPL=re.compile(r"(?i)haven'?t (?:told|mentioned|said)|you haven'?t|don'?t know|not sure|not certain|thanks for (?:telling|letting|sharing)|got it|i'?ll (?:remember|keep)|\bnoted\b|good to know|can'?t (?:find|say|tell)|isn'?t in the|oh,? cool|best guess|if i remember|i'?d guess|want to tell me")
def bpe():
    t=ast.parse((ROOT/'scripts'/'build_adam_t_sft.py').read_text());g={'re':re,'json':json,'np':np};exec(compile(ast.Module([n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Bpe'],[]),'build_adam_t_sft.Bpe','exec'),g);return g['Bpe'](str(H/'data'/'clean2'/'bpe4096.json'))
def pools(B,R):
    nt=lambda x:len(B.enc((' '+x).encode()));inv=lambda k:''.join(R.sample(SY,k)).capitalize()
    I1=sorted({x for x in (inv(R.choice((2,3))) for _ in range(4000)) if 2<=nt(x)<=3});I2=sorted({f'{inv(R.choice((2,3)))} {inv(R.choice((2,3)))}' for _ in range(3000)})
    P={'person':{'c1':[x for x in ONE_P if nt(x)==1],'c2':sorted({x for x in FIRST if 2<=nt(x)<=3}|{f'{a} {b}' for a in FIRST for b in LAST if nt(f'{a} {b}')<=3}),'r2':I1,'c4':sorted({f'{a} {m} {b}' for a in FIRST for m in FIRST[:30] for b in LAST if a!=m and nt(f'{a} {m} {b}')>=4})[::7],'r4':[x for x in I2 if nt(x)>=4]},'place':{'c1':[x for x in ONE_L if nt(x)==1],'c2':sorted({x for x in CITY+CITY2 if 2<=nt(x)<=3}),'r2':I1,'c4':sorted({x for x in CITY2 if nt(x)>=4}|{f'{p} {c}' for p in PRE for c in CITY if nt(f'{p} {c}')>=4}),'r4':[x for x in I2 if nt(x)>=4]}}
    return P
def ent(R,used):
    while True:
        e=(R.choice(SYE)+R.choice(SYE)[:2]+R.choice(SUF)).capitalize()
        if e not in used:used.add(e);return e
def pick(R,pool,n,ex=()):return R.sample([x for x in pool if x not in ex],n)
def item(R,P,used,ty,fac,lv,cls,i):
    k=int(lv) if fac=='distr' else 0;para=fac=='phr';nof=fac=='nofact';real=fac in ('real_true','real_cf','fmt_cf');ol=fac in ('fmt','fmt_cf');turns=[];rc=[]
    if ty=='umem':
        f,qe,qp,fr,dt,kd=UMEM[i%len(UMEM)];v,*o=pick(R,P[kd][cls],4+k);dis=o[:k];alts=(dis+o[k:])[:3];ps=R.sample(POSS,k)
        ft=[(f.format(v=v),ACK)]+[(dt.format(p=p,v=w),ACK) for p,w in zip(ps,dis)];R.shuffle(ft);fl=R.sample(FILL_T,8);d=int(lv) if fac=='dist' else 0
        turns=fl[:2]+([fl[7]] if nof else ft)+fl[2:2+d];q=qp if para else qe;fx=f.format(v=v);qq=q;fr_=fr
    elif ty=='passage':
        f,qe,qp,fr,kd,intro=PSG[i%len(PSG)];E=ent(R,used);v,*o=pick(R,P[kd][cls],4+k);dis=o[:k];alts=(dis+o[k:])[:3]
        fs=[f.format(E=E,v=v)]+[f.format(E=ent(R,used),v=w) for w in dis];R.shuffle(fs);fs=[] if nof else fs;tgt=int(lv) if fac=='dist' else 50;fl=R.sample(FILL_S,len(FILL_S));a=[intro.format(E=E)]+fs;b=[];j=0
        while sum(len(x)+1 for x in b)<tgt:b.append(fl[j]);j+=1
        p1=[x for x in a];p2=[]
        for x in b:(p1 if len(' '.join(p1+[x]))<540 else p2).append(x)
        rc=[f'{E}: '+' '.join(p1)]+([f'{ent(R,used)}: '+' '.join(p2)] if p2 else []);q=(qp if para else qe).format(E=E);fx=f.format(E=E,v=v);qq=q;fr_=fr.format(E=E)
    elif ty=='wikidata':
        if real:
            c,cap=REAL[i%len(REAL)];oth=[y for x,y in REAL if y!=cap];w=R.choice([y for y in oth if len(y.split())==1]);v,alts=(cap,pick(R,oth,3)) if fac=='real_true' else (w,[cap]+pick(R,oth,2,(w,)));dis=[];E=c;pr,qe,qp,fr,kd=WD[0]
        else:
            pr,qe,qp,fr,kd=WD[i%len(WD)];E=ent(R,used);v,*o=pick(R,P[kd][cls],4+k);dis=o[:k];alts=(dis+o[k:])[:3]
        lf=(lambda e,x:f'{e}: {x}.' if ol else f'{e}, {pr}: {x}.');ls=[lf(E,v)]+[lf(ent(R,used),w) for w in dis];R.shuffle(ls);ls=[] if nof else ls;tgt=int(lv) if fac=='dist' else 0;fl=R.sample(FILL_S,len(FILL_S));ps=[];j=0
        while sum(len(x)+1 for x in ps)<tgt:
            s=[];T=ent(R,used)
            while len(' '.join(s))<min(520,tgt-sum(len(x)+1 for x in ps)) and j<len(fl):s.append(fl[j]);j+=1
            ps.append(f'{T}: '+' '.join(s))
        rc=ls+ps;q=(qp if para else qe).format(E=E);fx=lf(E,v);qq=q;fr_=fr.format(E=E)
    else:
        op,qe,qp,fr,fn=CALC[i%len(CALC)];lo,hi={'c1':((2,9),(11,99)),'c2':((12,99),(101,999)),'c4':((1001,9999),(1001,9999))}[cls]
        while True:
            a,b=R.randint(*lo),R.randint(*hi);c=fn(a,b)
            if (cls=='c1' and c<1000) or (cls=='c2' and 1000<=c<1000000 and op=='x') or (cls=='c2' and op=='+' and 1000<=c) or (cls=='c4' and c>=1000000) or (cls=='c4' and op=='+'):break
        if cls=='c4' and op=='+':a,b=R.randint(1000000,9999999),R.randint(1000000,9999999);c=a+b
        v=f'{c:,}';dis=[];alts=[f'{x:,}' for x in (c+10,c+1 if c<3 else c-1,c+100 if c>=1000 else c+2)];rc=[] if nof else [f'Calculator: {a} {op} {b} = {v}.'];q=(qp if para else qe).format(a=a,b=b);fx=rc[0] if rc else f'= {v}.';qq=q;fr_=fr.format(a=a,b=b)
    pr_=template(qq,turns,BUD,(),rc);ctx='\n'.join([f'Me: {u}\nFriend: {x}' for u,x in turns]+rc);pl=f'{ctx}\nQuestion: {qq}\nAnswer:'
    fe=pr_.rfind(fx);db=pr_.rfind(qq)-(fe+len(fx)) if fe>=0 and not nof else -1
    return {'type':ty,'factor':fac,'level':str(lv),'cls':cls,'q':qq,'ans':v,'alts':alts,'dis':dis,'frame':fr_,'turns':turns,'rc':rc,'dist_b':db,'prompt':pr_,'plain':pl}
def cells():
    C=[]
    for ty in ('umem','passage','wikidata'):
        C+=[(ty,'base','0','c1')]+[(ty,'dist',l,'c1') for l in ((1,2,4,6) if ty=='umem' else (200,450,900))]+[(ty,'distr',l,'c1') for l in (1,2,4)]+[(ty,'cls','-',c) for c in ('c2','r2','c4','r4')]+[(ty,'phr','para','c1'),(ty,'nofact','-','c1')]+([(ty,'real_true','-','c1'),(ty,'real_cf','-','c1')] if ty=='wikidata' else [])
    return C+[('calc','base','0','c1'),('calc','cls','-','c2'),('calc','cls','-','c4'),('calc','phr','para','c1'),('calc','nofact','-','c1'),('wikidata','fmt','old','c1'),('wikidata','fmt_cf','old','c1')]
def build(a):
    B=bpe();R=random.Random(SEED);P=pools(B,R);used=set();out=[]
    for ty,fac,lv,cls in cells():
        for i in range(a.n):out.append({'id':f'{ty}.{fac}.{lv}.{cls}.{i:02d}',**item(random.Random(f'{SEED}.{ty}.{fac}.{lv}.{cls}.{i}'),P,used,ty,fac,lv,cls,i)})
    for o in out:o['ans_tok']=len(B.enc((' '+o['ans']).encode()));o['prompt_tok']=len(B.enc(o['prompt'].encode()))
    PD.mkdir(parents=True,exist_ok=True);(PD/'probes.jsonl').write_text(''.join(json.dumps(o)+'\n' for o in out))
    (PD/'meta.json').write_text(json.dumps({'what':'ADAM-T grounding/copy probe set (own CC0 synthetic text, invented entities and values; no eval-set content)','seed':SEED,'items':len(out),'per_cell':a.n,'cells':[list(c) for c in cells()],'template':'amni/serve/adam_t_client.template(q, history, 2928, (), rc)','plain':'context lines + \\nQuestion: q\\nAnswer: (base model framing)','classes':{'c1':'1 BPE token, common','c2':'2-3 tokens, common','r2':'2-3 tokens, invented','c4':'4+ tokens, common','r4':'4+ tokens, invented','calc':'c1 <1000, c2 4-6 digits, c4 7+ digits'},'pool_sizes':{k:{c:len(v) for c,v in d.items()} for k,d in P.items()}},indent=1))
    print(len(out),'items',collections.Counter(o['type'] for o in out))
def srv(path,thr='3'):return Proto([os.environ.get('ADAMT_BIN') or str(ROOT/'scratchpad'/'rca'/'bin'/'adamt-rca'),'serve',path],900,{**os.environ,'BACKEND':'ndarray','RAYON_NUM_THREADS':thr,'AMNI_HOME':str(H)},None,900)
def enc(p,xs):return p.rpc(op='enc',items=[B64(x.encode()) for x in xs])['ids']
def run(a):
    name,path=a.model.split('=',1);plain=a.plain;P=[json.loads(l) for l in open(PD/'probes.jsonl')];P=[x for x in P if not a.only or re.search(a.only,x['id'])][:a.limit or None]
    of=Path(a.out or PD/'results'/f'{name}.jsonl');of.parent.mkdir(parents=True,exist_ok=True);done={json.loads(l)['id'] for l in open(of)} if of.exists() else set();p=srv(path,a.threads);ci=[x[0] for x in enc(p,CAN)];t0=time.time()
    with open(of,'a') as fo:
        for n,it in enumerate(x for x in P if x['id'] not in done):
            pr=it['plain'] if plain else it['prompt'];fr=' '+it['frame'];A,F=enc(p,[' '+it['ans'],fr])
            o=p.rpc(op='probe',prompt=B64(pr.encode()),pre=F,cands=[A]+enc(p,[' '+c for c in it['alts']]),ids=[A[0],F[0]]+ci,k=8,gen=not a.no_gen,max=100 if plain else 160,stop=[B64(s.encode()) for s in (['\n'] if plain else STOPS)],**({'room':a.room} if a.room else {}));c=o['c'];g=U64(o['text']).decode('utf-8','replace') if o.get('text') else ''
            fo.write(json.dumps({'id':it['id'],'model':name,'plain':plain,'reply':g.strip(),'top0':o['top0'],'lp_a0':o['lp0'][0],'lp_f0':o['lp0'][1],'lp_can':o['lp0'][2:],'same_af':A[0]==F[0],'tf_rank':[r for _,r in c[0]],'tf_lp':[l for l,_ in c[0]],'mc':[sum(l for l,_ in x) for x in c],'ntok':len(A),'n':o['n']})+'\n');fo.flush()
            n%25==0 and print(name,n,f'{time.time()-t0:.0f}s',flush=True)
    p.close()
lse=lambda v:max(v)+math.log(sum(math.exp(x-max(v)) for x in v)) if v else -1e9
nrm=lambda s:' '+re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]',' ',s.lower().replace(',','')))+' '
def score(it,r):
    y=r['reply'];ex=bool(re.search(r'(?<![A-Za-z0-9])'+re.escape(it['ans'])+r'(?![A-Za-z0-9])',y));sp=nrm(it['ans']) in nrm(y);ds=any(nrm(d) in nrm(y) for d in it['dis']);pri=it['factor'] in ('real_cf','fmt_cf') and nrm(it['alts'][0]) in nrm(y)
    fw=nrm(it['ans'].split()[0]);pf=(not sp) and ((len(fw)>=5 and fw in nrm(y)) or it['ans'][:4].lower() in y.lower())
    po=math.exp(lse([r['lp_a0']]+([] if r['same_af'] else [r['lp_f0']])));pt=sum(math.exp(x) for x in r['lp_can'])
    return {'exact':ex and not ds,'span':sp,'tpl':bool(TPL.search(y)) and not sp,'dis':ds,'prior':pri,'pfx':pf,'mc':int(np.argmax(r['mc']))==0,'p_open_copy':po,'p_open_tpl':pt,'tpl_pref':pt>po,'p1_frame':math.exp(r['tf_lp'][0]),'tf_top1':all(x==0 for x in r['tf_rank']),'tf_tok':float(np.mean([x==0 for x in r['tf_rank']])),'empty':not y,'recent':all(it['prompt'].rfind(it['ans'])>it['prompt'].rfind(d) for d in it['dis']),'kn_tpl':bool(TPL.search(y)) and not sp and int(np.argmax(r['mc']))==0}
def load(names):
    P={json.loads(l)['id']:json.loads(l) for l in open(PD/'probes.jsonl')};R={}
    for n in names:
        f=PD/'results'/f'{n}.jsonl'
        if f.exists():R[n]={j['id']:(P[j['id']],j,score(P[j['id']],j)) for j in map(json.loads,open(f))}
    return P,R
def agg(rows,keys):return {k:float(np.mean([s[k] for _,_,s in rows])) for k in keys}
def report(a):
    names=a.models.split(',');P,R=load(names);K=['exact','span','tpl','mc','tf_top1','tf_tok','p1_frame','p_open_copy','p_open_tpl','tpl_pref','pfx','dis','prior'];S={}
    for n,rs in R.items():
        g=collections.defaultdict(list)
        for it,r,s in rs.values():g[(it['type'],it['factor'],it['level'],it['cls'])].append((it,r,s))
        S[n]={'|'.join(c):{'n':len(v),**agg(v,K),'dist_b':float(np.mean([x['dist_b'] for x,_,_ in v])),'ans_tok':float(np.mean([x['ans_tok'] for x,_,_ in v]))} for c,v in g.items()}
        S[n]['_all_fact']={'n':sum(1 for it,_,_ in rs.values() if it['factor']!='nofact'),**agg([v for v in rs.values() if v[0]['factor']!='nofact'],K)}
        S[n]['_by_ntok']={str(t):{'n':len(v),**agg(v,K)} for t,v in sorted(collections.defaultdict(list,{t:[x for x in rs.values() if min(x[0]['ans_tok'],5)==t and x[0]['factor'] in ('base','cls') and x[0]['type']!='calc'] for t in range(1,6)}).items()) if v}
    of=Path(a.out or PD/'summary.json');of.write_text(json.dumps(S,indent=1));pc=lambda x:f'{100*x:.0f}'
    for ty in ('umem','passage','wikidata','calc'):
        print(f'\n#### {ty}\n\n| cell | n | dist B | ans tok | '+' | '.join(f'{n} exact/span/tpl/MC/TF' for n in names)+' |\n|---|---|---|---|'+'---|'*len(names))
        for c in cells():
            if c[0]!=ty:continue
            k='|'.join(map(str,c));f=next((S[n][k] for n in names if k in S.get(n,{})),None)
            f and print(f"| {c[1]} {c[2] if str(c[2]) not in ('-','0') else ''} {c[3]} | {f['n']} | {f['dist_b']:.0f} | {f['ans_tok']:.1f} | "+' | '.join((lambda z:f"{pc(z['exact'])}/{pc(z['span'])}/{pc(z['tpl'])}/{pc(z['mc'])}/{pc(z['tf_top1'])}" if z else '-')(S.get(n,{}).get(k)) for n in names)+' |')
    print('\n| model | n | exact | span | tpl | MC | TF all-top1 | p(ans tok given frame) | p(open copy) | p(open tpl) | tpl preferred | prefix-only |\n|---|---|---|---|---|---|---|---|---|---|---|---|')
    [print(f"| {n} | {z['n']} | {pc(z['exact'])} | {pc(z['span'])} | {pc(z['tpl'])} | {pc(z['mc'])} | {pc(z['tf_top1'])} | {z['p1_frame']:.2f} | {z['p_open_copy']:.2f} | {z['p_open_tpl']:.2f} | {pc(z['tpl_pref'])} | {pc(z['pfx'])} |") for n in names for z in [S.get(n,{}).get('_all_fact')] if z]
    print('\n| model | type | n | exact, fact most recent | exact, fact not most recent | distractor copied | MC |\n|---|---|---|---|---|---|---|')
    for n,rs in R.items():
        for ty in ('umem','passage','wikidata'):
            v=[x for x in rs.values() if x[0]['type']==ty and x[0]['factor']=='distr'];f=lambda q:[s for _,_,s in v if s['recent']==q];m=lambda z,k:f"{pc(np.mean([s[k] for s in z]))} (n={len(z)})" if z else '-'
            print(f"| {n} | {ty} | {len(v)} | {m(f(True),'exact')} | {m(f(False),'exact')} | {pc(np.mean([s['dis'] for _,_,s in v]))} | {pc(np.mean([s['mc'] for _,_,s in v]))} |")
    print('\n| model | type | n | template reply | template reply while MC picks the answer | p(open copy) | p(open tpl) | tpl opening preferred | no-fact: template | no-fact: other reply |\n|---|---|---|---|---|---|---|---|---|---|')
    for n,rs in R.items():
        for ty in ('umem','passage','wikidata','calc'):
            v=[s for it,_,s in rs.values() if it['type']==ty and it['factor']!='nofact'];w=[s for it,_,s in rs.values() if it['type']==ty and it['factor']=='nofact']
            print(f"| {n} | {ty} | {len(v)} | {pc(np.mean([s['tpl'] for s in v]))} | {pc(np.mean([s['kn_tpl'] for s in v]))} | {np.mean([s['p_open_copy'] for s in v]):.2f} | {np.mean([s['p_open_tpl'] for s in v]):.2f} | {pc(np.mean([s['tpl_pref'] for s in v]))} | {pc(np.mean([s['tpl'] for s in w]))} | {pc(np.mean([not s['tpl'] and not s['empty'] for s in w]))} |")
    print('\n| model | ans tokens | n | exact | span | MC | TF all-top1 | TF per-token | p1 given frame |\n|---|---|---|---|---|---|---|---|---|')
    [print(f"| {n} | {t}{'+' if t=='5' else ''} | {z['n']} | {pc(z['exact'])} | {pc(z['span'])} | {pc(z['mc'])} | {pc(z['tf_top1'])} | {pc(z['tf_tok'])} | {z['p1_frame']:.2f} |") for n in names for t,z in S.get(n,{}).get('_by_ntok',{}).items()]
def fid(a):
    P=[json.loads(l) for l in open(PD/'probes.jsonl')];R=random.Random(SEED);P=R.sample([x for x in P if x['factor']!='nofact'],a.n);out={}
    for nm,path in (('lut',a.gf17),('burn',a.run)):
        p=srv(path,a.threads);t0=time.time();o=[]
        for i in range(0,len(P),8):o+=p.rpc(op='nxt',items=[B64(x['prompt'].encode()) for x in P[i:i+8]],k=10)['out']
        g=[p.text(x['prompt'],a.max,STOPS) for x in P[:a.ngen]];out[nm]={'top':[z['top'] for z in o],'gen':g,'s':time.time()-t0};p.close();print(nm,f"{out[nm]['s']:.0f}s",flush=True)
    L,Bn=out['lut'],out['burn'];t1=[l[0][0]==b[0][0] for l,b in zip(L['top'],Bn['top'])];dl=[max(abs(dict(map(tuple,l)).get(t,-99)-v) for t,v in b[:5]) for l,b in zip(L['top'],Bn['top'])]
    r={'n':len(P),'first_token_top1_agree':float(np.mean(t1)),'top5_max_abs_dlogp_mean':float(np.mean(dl)),'top5_max_abs_dlogp_max':float(np.max(dl)),'gen_n':a.ngen,'gen_exact_agree':float(np.mean([x==y for x,y in zip(L['gen'],Bn['gen'])])),'gen_first_word_agree':float(np.mean([(x.split() or [''])[0]==(y.split() or [''])[0] for x,y in zip(L['gen'],Bn['gen'])])),'disagree':[{'id':P[i]['id'],'lut':L['gen'][i],'burn':Bn['gen'][i]} for i in range(a.ngen) if L['gen'][i]!=Bn['gen'][i]],'top1_disagree':[{'id':P[i]['id'],'lut':L['top'][i][:3],'burn':Bn['top'][i][:3]} for i in range(len(P)) if not t1[i]],'gf17':a.gf17,'run':a.run}
    Path(a.out or PD/'fidelity.json').write_text(json.dumps(r,indent=1));print(json.dumps({k:v for k,v in r.items() if k not in ('disagree','top1_disagree')}))
def binding(a):
    name,path=a.model.split('=',1);a.plain,a.only,a.limit,a.no_gen,a.out=True,r'\.distr\.',0,True,a.res or str(PD/'results'/f'bind_{name}.jsonl');run(a);B=bpe()
    P={json.loads(l)['id']:json.loads(l) for l in open(PD/'probes.jsonl')};R=[(P[j['id']],j) for j in map(json.loads,open(a.out)) if j['id'] in P and P[j['id']]['factor']=='distr']
    def cut(it,r):ids=B.enc(it['plain'].encode());t=B.dec(ids[-r['n']:]).decode('utf-8','replace');return (len(ids)>r['n'],not all(x in t for x in [it['ans']]+it['dis']))
    S=[(it,r,int(np.argmax(r['mc']))==0,1/min(int(it['level'])+1,4),all(x==0 for x in r['tf_rank']),math.exp(r['tf_lp'][0]),*cut(it,r)) for it,r in R]
    g=lambda z:{'n':len(z),'mc':float(np.mean([x[2] for x in z])),'chance':float(np.mean([x[3] for x in z])),'lift':float(np.mean([x[2]-x[3] for x in z])),'tf_top1':float(np.mean([x[4] for x in z])),'p1_frame':float(np.mean([x[5] for x in z]))} if z else {}
    o={'model':name,'path':path,'room':a.room,**g(S),'truncated':sum(x[6] for x in S),'facts_cut':sum(x[7] for x in S),'by_type':{ty:g([x for x in S if x[0]['type']==ty]) for ty in ('umem','passage','wikidata')},'by_k':{k:g([x for x in S if x[0]['level']==k]) for k in ('1','2','4')}}
    Path(a.summary).parent.mkdir(parents=True,exist_ok=True) if a.summary else None;a.summary and Path(a.summary).write_text(json.dumps(o,indent=1));print(json.dumps(o))
def induct(a):
    B=bpe();W=sorted(v.decode() for v in B.voc if re.fullmatch(rb' [a-z]{3,}',v));out={}
    for name,path in (x.split('=',1) for x in a.models.split(',')):
        p=srv(path,a.threads);out[name]={}
        for L in map(int,a.lens.split(',')):
            A=[]
            for i in range(a.n):
                s=random.Random(f'{SEED}.ind.{L}.{i}').sample(W,L);ids=[B.voc.index(w.encode()) for w in s];c=p.rpc(op='probe',prompt=B64((''.join(s)+' .'+s[0]).encode()),pre=[],cands=[ids[1:]],ids=[],k=1)['c'][0];A.append([r==0 for _,r in c])
            out[name][str(L)]={'top1':float(np.mean([np.mean(x) for x in A])),'top1_first5':float(np.mean([np.mean(x[:5]) for x in A])),'top1_last_half':float(np.mean([np.mean(x[len(x)//2:]) for x in A])),'n':a.n}
        out[name]['mean_top1']=float(np.mean([out[name][str(L)]['top1'] for L in map(int,a.lens.split(','))]));p.close();print(name,out[name],flush=True)
    Path(a.out or PD/'induction.json').write_text(json.dumps(out,indent=1))
def tok(a):
    B=bpe();P=[json.loads(l) for l in open(PD/'probes.jsonl')];p=srv(a.gf17,'2');r=collections.Counter();bad=[]
    for i in range(0,len(P),64):
        ch=P[i:i+64];xs=[x['prompt'] for x in ch]+[x['prompt']+' '+x['frame']+' '+x['ans']+'.\n\n' for x in ch];ru=enc(p,xs);n=len(ch)
        for j,x in enumerate(ch):
            py,pf=B.enc(xs[j].encode()),B.enc(xs[n+j].encode());r['n']+=1;r['builder_eq_serve_prompt']+=py==ru[j];r['builder_eq_serve_example']+=pf==ru[n+j];r['prompt_is_token_prefix']+=ru[n+j][:len(ru[j])]==ru[j];r['ends_Adam_colon']+=xs[j].endswith('\nAdam:');r['ascii']+=xs[j].isascii()
            (py!=ru[j] or ru[n+j][:len(ru[j])]!=ru[j]) and len(bad)<10 and bad.append(x['id'])
    D={}
    for d in ('sft','sft_v5'):
        M=json.load(open(H/'data'/'adam_t'/d/'meta.json'));S=list(M['sources']);T=np.fromfile(H/'data'/'adam_t'/d/'train.bpe4096.u16',np.uint16);I=np.load(H/'data'/'adam_t'/d/'train.bpe4096.idx.npy');R=random.Random(1);c=collections.Counter()
        for s in ('umem','kctx','wikidata','calc','mem','squad'):
            ix=R.sample(list(np.nonzero(I['src']==S.index(s))[0]),min(a.k,int((I['src']==S.index(s)).sum())))
            for j in range(0,len(ix),32):
                E=[I[x] for x in ix[j:j+32]];tx=[B.dec(T[e['off']:e['off']+e['n']].tolist()).decode('ascii') for e in E];ru=enc(p,tx)
                for e,t,u in zip(E,tx,ru):c[s+'_n']+=1;c[s+'_serve_retok_eq']+=u==T[e['off']:e['off']+e['n']].tolist();c[s+'_starts_User']+=t.startswith('User: ');c[s+'_last_turn_ends_nn']+=t.endswith('\n\n')
        D[d]=dict(c)
    p.close();out={'probes':dict(r),'probe_mismatch_ids':bad,'sft_roundtrip':D};Path(a.out or PD/'tokenization.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
def audit(a):
    B=bpe();out={};W=lambda s:set(re.findall(r'[a-z0-9]+',s.lower()));SW=set('the a an is was are were of in on at to for by and or it its his her their this that with from as be been has have had he she they you your i my me we our not no yes but so if than then there here what which who whom when where why how do does did can could would will should one also about into over after before up out more most very just only some any all many much such other its also'.split())
    UNK=re.compile(r"(?i)haven'?t (?:told|mentioned)|don'?t know (?:your|where|who|what|when|how|which)|want to tell me");HED=re.compile(r"(?i)best guess|not sure|not certain|if i remember|i think it'?s|i'?d guess|probably|can'?t find|isn'?t in the|don'?t (?:give|say|mention)")
    for d in a.sets.split(','):
        M=json.load(open(H/'data'/'adam_t'/d/'meta.json'));S=list(M['sources']);T=np.fromfile(H/'data'/'adam_t'/d/'train.bpe4096.u16',np.uint16);I=np.load(H/'data'/'adam_t'/d/'train.bpe4096.idx.npy');R=random.Random(2);o={};allr=collections.Counter()
        for si,s in enumerate(S):
            ix=np.nonzero(I['src']==si)[0];sm=R.sample(list(ix),min(a.k,len(ix)));c=collections.Counter()
            for e in (I[x] for x in sm):
                t=B.dec(T[e['off']:e['off']+e['n']].tolist()).decode('ascii','replace');m=re.findall(r'User: ([^\n]*)\nAdam: ([^\n]*)\n\n',t)
                if not m:continue
                u,y=m[-1];cw=W(y)-SW;ctx=W(u)|W(''.join(f'{x} {z} ' for x,z in m[:-1]));has=bool(cw) and len(cw&ctx)/len(cw)>=0.8;c['n']+=1;c['ctx_bytes']+=len(u)+sum(len(x)+len(z) for x,z in m[:-1]);c['reply_in_prompt']+=has;c['unk']+=bool(UNK.search(y));c['hedge']+=bool(HED.search(y));c['cool']+=y.strip()=='Oh, cool! Thanks for telling me.';c['ack']+=bool(re.match(r"(?i)\s*(got it|thanks for (telling|letting)|noted|good to know|nice, i'?ll remember|okay, noted|i'?ll remember)",y));c['wd_prop_line']+=bool(re.match(r'[^:,]{2,60}, [a-z ]{3,30}: [^:]{1,80}\.\s',u));c['wd_plain_line']+=bool(re.match(r'[^:,]{2,60}: [^:.]{1,60}\.\s[^:]{2,80}:',u))
            allr.update({k:v*len(ix)/max(1,len(sm)) for k,v in c.items()});o[s]={'train_examples':int(len(ix)),'sampled':c['n'],**{k:round(v/max(1,c['n']),4) for k,v in c.items() if k!='n'}}
        Rr=[B.dec(T[I[x]['off']:I[x]['off']+I[x]['n']].tolist()).decode('ascii','replace').rsplit('\nAdam: ',1)[-1].strip() for x in random.Random(3).sample(range(len(I)),min(a.k*10,len(I)))];top=collections.Counter(Rr).most_common(12)
        out[d]={'sources':o,'est_totals':{k:int(v) for k,v in allr.items()},'top_final_replies':top}
    Path(a.out or PD/'sft_audit.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1)[:6000])
if __name__=='__main__':
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    b=sp.add_parser('build');b.add_argument('--n',type=int,default=N)
    r=sp.add_parser('run');r.add_argument('--model',required=True);r.add_argument('--plain',action='store_true');r.add_argument('--limit',type=int,default=0);r.add_argument('--only',default='');r.add_argument('--out',default='');r.add_argument('--threads',default='3');r.add_argument('--no-gen',action='store_true');r.add_argument('--room',type=int,default=0)
    bd=sp.add_parser('binding');bd.add_argument('--model',required=True);bd.add_argument('--room',type=int,default=16);bd.add_argument('--res',default='');bd.add_argument('--summary',default='');bd.add_argument('--threads',default='3')
    e=sp.add_parser('report');e.add_argument('--models',required=True);e.add_argument('--out',default='')
    f=sp.add_parser('fid');f.add_argument('--gf17',required=True);f.add_argument('--run',required=True);f.add_argument('--n',type=int,default=50);f.add_argument('--ngen',type=int,default=50);f.add_argument('--max',type=int,default=40);f.add_argument('--out',default='');f.add_argument('--threads',default='3')
    t=sp.add_parser('tok');t.add_argument('--gf17',required=True);t.add_argument('--k',type=int,default=200);t.add_argument('--out',default='')
    u=sp.add_parser('audit');u.add_argument('--sets',default='sft,sft_v5');u.add_argument('--k',type=int,default=3000);u.add_argument('--out',default='')
    i=sp.add_parser('induct');i.add_argument('--models',required=True);i.add_argument('--n',type=int,default=30);i.add_argument('--out',default='');i.add_argument('--threads',default='3');i.add_argument('--lens',default='10,50,200')
    x=ap.parse_args();{'build':build,'run':run,'report':report,'fid':fid,'tok':tok,'audit':audit,'induct':induct,'binding':binding}[x.cmd](x)
