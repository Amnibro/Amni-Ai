import argparse,datetime as dt,json,math,random,re,sys,time
from fractions import Fraction as Fr
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT)) if str(ROOT) not in sys.path else None
from amni.compute import eval_lock
from amni.math.answer import solve_math
D=ROOT/'data'/'math';TRAIN=D/'gsm8k'/'train.jsonl';DEV_N=1000;LOCK=D/'m08cto_eval_lock.json';OUT=D/'m08cto_reports';TODAY=dt.date(2026,9,25)
CODE=[ROOT/'amni'/'math'/x for x in ('answer.py','word.py','relate.py','sympy_tool.py','protocol.py','harness.py')]+[ROOT/'amni'/'serve'/'code_safety.py',ROOT/'scripts'/'m08cto_eval.py']
_LEN={'mm':Fr(1,1000),'cm':Fr(1,100),'m':Fr(1),'km':Fr(1000),'in':Fr(254,10000),'ft':Fr(254*12,10000),'yd':Fr(254*36,10000),'mi':Fr(254*63360,10000)}
_MASS={'g':Fr(1,1000),'kg':Fr(1),'mg':Fr(1,10**6),'lb':Fr(45359237,10**8),'oz':Fr(45359237,16*10**8)}
_VOL={'ml':Fr(1,1000),'l':Fr(1),'gal':Fr(231)*Fr(254,10000)**3*1000,'qt':Fr(231,4)*Fr(254,10000)**3*1000,'cup':Fr(231,16)*Fr(254,10000)**3*1000}
_TIME={'s':Fr(1),'min':Fr(60),'h':Fr(3600),'day':Fr(86400),'week':Fr(604800)}
_NAMES={'mm':('millimeters','mm'),'cm':('centimeters','cm'),'m':('meters','m'),'km':('kilometers','km'),'in':('inches','inches'),'ft':('feet','ft'),'yd':('yards','yards'),'mi':('miles','miles'),'g':('grams','g'),'kg':('kilograms','kg'),'mg':('milligrams','mg'),'lb':('pounds','lbs'),'oz':('ounces','oz'),'ml':('milliliters','ml'),'l':('liters','liters'),'gal':('gallons','gallons'),'qt':('quarts','quarts'),'cup':('cups','cups'),'s':('seconds','seconds'),'min':('minutes','minutes'),'h':('hours','hours'),'day':('days','days'),'week':('weeks','weeks')}
def gen_suite(n=500,seed=20260925):
 R=random.Random(seed);items=[]
 def arith():
  a,b,c=R.randint(2,999),R.randint(2,99),R.randint(2,50);k=R.randrange(9)
  return [(f"What is {a} + {b}?",Fr(a+b)),(f"what's {a} minus {b}",Fr(a-b)),(f"Calculate {a} times {b}",Fr(a*b)),(f"{a} divided by {b}",Fr(a,b)),(f"What is ({a} + {b}) * {c}?",Fr((a+b)*c)),(f"compute {b} to the power of {R.randint(2,6)}",None),(f"what is {a} - {b} * {c}",Fr(a-b*c)),(f"square root of {b*b}",Fr(b)),(f"{c}! ",None)][k]
 def pct():
  p,b=R.choice([5,10,12.5,15,20,25,30,40,50,75,8,3]),R.randint(2,400)*R.choice([1,10]);P=Fr(str(p));k=R.randrange(6)
  return [(f"What is {p}% of {b}?",P/100*b),(f"what percent of {b} is {fmtn(P/100*b)}?",P),(f"increase {b} by {p}%",b*(1+P/100)),(f"decrease {b} by {p}%",b*(1-P/100)),(f"{p}% off ${b}",b*(1-P/100)),(f"{fmtn(P/100*b)} is {p}% of what?",Fr(b))][k]
 def units():
  T=R.choice([_LEN,_MASS,_VOL,_TIME,'temp'])
  if T=='temp':
   v=R.randint(-40,212);return (f"convert {v} fahrenheit to celsius",(Fr(v)-32)*Fr(5,9)) if R.random()<0.5 else (f"{v} degrees celsius in fahrenheit",Fr(v)*Fr(9,5)+32)
  a,b=R.sample(sorted(T),2);v=R.randint(1,500);fa,fb=R.choice(_NAMES[a]),R.choice(_NAMES[b])
  return (f"convert {v} {fa} to {fb}",v*T[a]/T[b]) if R.random()<0.6 else (f"how many {_NAMES[b][0]} are in {v} {_NAMES[a][0]}?",v*T[a]/T[b])
 def dates():
  d=dt.date(2000,1,1)+dt.timedelta(days=R.randint(0,12000));e=d+dt.timedelta(days=R.randint(1,900));k=R.randrange(3);n=R.randint(1,400)
  return [(f"how many days between {d:%B} {d.day}, {d.year} and {e:%B} {e.day}, {e.year}?",Fr((e-d).days)),(f"what day of the week is {d.isoformat()}?",Fr(d.weekday())),(f"what date is {n} days after {d:%B} {d.day}, {d.year}?",Fr((d+dt.timedelta(days=n)).toordinal()))][k]
 def alg():
  x,a,b=R.randint(-20,30),R.randint(2,12),R.randint(-50,50);c=a*x+b;v=R.choice('xyn');k=R.randrange(3)
  return [(f"solve {a}{v} + {b} = {c}".replace('+ -','- '),Fr(x)),(f"Solve for {v}: {a}{v} {'+' if b>=0 else '-'} {abs(b)} = {c}",Fr(x)),(f"{c} = {a}{v} + {b}, what is {v}?".replace('+ -','- '),Fr(x))][k]
 for kind,f,m in (('arith',arith,150),('percent',pct,100),('units',units,100),('date',dates,75),('algebra',alg,75)):
  while sum(1 for i in items if i['kind']==kind)<m:
   q,g=f()
   mm=re.search(r"(\d+) to the power of (\d+)",q) if g is None else None;g=g if g is not None else Fr(int(mm.group(1))**int(mm.group(2))) if mm else Fr(math.factorial(int(q.strip()[:-1])))
   items.append({'kind':kind,'q':q,'gold':str(g)})
 return items[:n]
def split():
 rows=[json.loads(l) for l in TRAIN.read_text(encoding='utf-8').splitlines() if l.strip()];return rows[:-DEV_N],rows[-DEV_N:]
def gold(r)->Fr:return Fr(r['answer'].split('####')[-1].strip().replace(',',''))
def fmtn(v:Fr)->str:return str(v.numerator) if v.denominator==1 else str(float(v))
def match(pred,g:Fr)->bool:
 if pred is None:return False
 try:return (Fr(pred)==g) if isinstance(pred,(int,Fr)) else abs(float(pred)-float(g))<=1e-9*max(1,abs(float(g)))
 except (TypeError,ValueError):return False
def run(rows,q,g,conf_hedge=0.6,limit=None):
 rows=rows[:limit] if limit else rows;st={'n':len(rows),'answered':0,'correct':0,'wrong':0,'confident':0,'confident_correct':0,'hedged':0,'hedged_correct':0};t=time.time();per=[]
 for r in rows:
  o=solve_math(q(r),TODAY);gv=g(r);ok=o is not None and match(o.get('value'),gv);ans=o is not None and o.get('value') is not None
  st['answered']+=ans;st['correct']+=ok;st['wrong']+=ans and not ok;c=ans and o['conf']>=conf_hedge;st['confident']+=c;st['confident_correct']+=c and ok;st['hedged']+=ans and not c;st['hedged_correct']+=ans and not c and ok
  per.append({'q':q(r)[:160],'gold':str(gv),'pred':None if not ans else str(o['value']),'conf':None if not ans else round(o['conf'],3),'intent':None if o is None else o.get('intent'),'ok':ok})
 n=max(1,st['n']);st.update({'answer_rate':round(st['answered']/n,4),'accuracy':round(st['correct']/n,4),'precision':round(st['correct']/max(1,st['answered']),4),'wrong_rate':round(st['wrong']/n,4),'confident_precision':round(st['confident_correct']/max(1,st['confident']),4),'secs':round(time.time()-t,1)});return st,per
def svamp_rows():
 d=json.loads((D/'svamp'/'SVAMP.json').read_text(encoding='utf-8'));return [{'question':(x['Body'].strip().rstrip('.')+'. '+x['Question'].strip()).replace('..','.'),'gold':Fr(str(x['Answer']))} for x in d]
def test_rows():return [json.loads(l) for l in (D/'gsm8k'/'test.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
NEG=["What is the capital of France?","How many moons does Jupiter have?","Who was the 16th president of the United States?","What year did World War 2 end?","What is 1984 about?","Tell me about Apollo 11.","How are you today?","What happened on 9/11?","Who won the 2018 World Cup?","How many people live in Tokyo?","What is the population of Canada in 2020?","When was the Eiffel Tower built?","How tall is Mount Everest?","How many bones are in the human body?","What is the boiling point of water?","Can you recommend 3 good books?","Write me a poem about 2 cats.","What does HTTP 404 mean?","Who wrote Catch-22?","How many legs does a spider have?","What is the speed of light?","Tell me a joke.","Why is the sky blue?","What is Python 3?","How many continents are there?","Where is Area 51?","What is the meaning of life?","Explain photosynthesis.","Who painted the Mona Lisa?","What's the weather like?","Is Pluto a planet?","What is the tallest building in the world?","Who is the CEO of Tesla?","How many players are on a soccer team?","What is your name?","Summarize the plot of Hamlet.","What is COVID-19?","How old is the universe?","What is the chemical symbol for gold?","How many calories are in an apple?","When did the Titanic sink?","What is the Fibonacci sequence?","Translate hello into Spanish.","Give me 5 tips for sleeping better.","What is the best phone in 2024?","Who was Napoleon?","Play some music.","Remind me at 5pm to call mom.","Open the door.","What is a black hole?"]
def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['suite','dev','neg','test']);ap.add_argument('--limit',type=int);ap.add_argument('--out');ap.add_argument('--one-look',action='store_true');ap.add_argument('--reproduce',action='store_true');ap.add_argument('--seed',type=int,default=20260925);a=ap.parse_args(argv)
 OUT.mkdir(parents=True,exist_ok=True);out=Path(a.out) if a.out else OUT/f"report_{a.mode}.json";man=json.loads(LOCK.read_text())
 if a.mode=='suite':
  S=gen_suite(seed=a.seed);out.with_name(f'm08cto_suite_{a.seed}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in S));st,per=run(S,lambda r:r['q'],lambda r:Fr(r['gold']),limit=a.limit);byk={k:run([x for x in S if x['kind']==k],lambda r:r['q'],lambda r:Fr(r['gold']))[0] for k in ('arith','percent','units','date','algebra')}
  rep=eval_lock.record(out,{'mode':'suite','seed':a.seed,'overall':st,'by_kind':byk,'errors':[p for p in per if not p['ok']][:40]},code_files=CODE,splits=man,kind='prod')
 elif a.mode=='neg':
  fp=[{'q':q,'reply':(solve_math(q,TODAY) or {}).get('reply')} for q in NEG];rep=eval_lock.record(out,{'mode':'neg','n':len(NEG),'routed_to_math':sum(1 for x in fp if x['reply']),'false_positive_rate':round(sum(1 for x in fp if x['reply'])/len(NEG),4),'hits':[x for x in fp if x['reply']]},code_files=CODE,splits=man,kind='prod')
 elif a.mode=='dev':
  _,dev=split();st,per=run(dev,lambda r:r['question'],gold,limit=a.limit);rep=eval_lock.record(out,{'mode':'dev','split':'gsm8k train rows[-1000:]','overall':st,'sample':per[:60]},code_files=CODE,splits=man,kind='prod')
 else:
  if not a.one_look:raise SystemExit('REFUSE: the locked GSM8K test and SVAMP get one look; pass --one-look (or --reproduce to recompute an existing look into --out)')
  prev=OUT/'report_test.json'
  if prev.exists() and not a.reproduce:raise SystemExit(f'REFUSE: one look already taken ({prev}); use --reproduce --out <path> to recompute it without replacing it')
  if a.reproduce and out.resolve()==prev.resolve():raise SystemExit('REFUSE: --reproduce must write to a different --out')
  bad=eval_lock.verify(man)
  if bad:raise SystemExit(f'REFUSE: locked splits changed: {bad}')
  g8,sv=run(test_rows(),lambda r:r['question'],gold,limit=a.limit),run(svamp_rows(),lambda r:r['question'],lambda r:r['gold'],limit=a.limit)
  rep=eval_lock.record(out,{'mode':'test','one_look':not a.reproduce,'gsm8k_test':g8[0],'svamp':sv[0],'gsm8k_wrong_sample':[p for p in g8[1] if p['pred'] and not p['ok']][:0],'svamp_by_conf':None},code_files=CODE,splits=man,kind='prod')
 print(json.dumps({k:v for k,v in rep.items() if k not in ('eval_lock','errors','sample','hits')},indent=1)[:4000]);print('->',out)
if __name__=='__main__':main()
