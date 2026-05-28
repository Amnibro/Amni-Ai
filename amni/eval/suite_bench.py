"""suite_bench — run Adam on the leaderboard benchmarks people actually chart: ARC-Challenge, MMLU, GSM8K,
HellaSwag, Winogrande. Loads each via `datasets`, formats a prompt, calls a model-agnostic generate_fn, and scores
objectively (single-letter match for MCQ, '#### N' / last-number for GSM8K). Produces a leaderboard-style table
(+ optional PNG) with published reference scores so the result is a chart people understand. Offline-testable via a
bundled synthetic sample; real suites load on demand."""
import re,json,time
from pathlib import Path
from typing import Dict,Any,List,Optional,Callable
_LETTERS='ABCDEFGH'
def _sample_path()->Path:return Path(__file__).resolve().parent/'suite_sample.json'
def _norm_mcq(question:str,choices:List[str],answer_idx:int,task_id:str)->Dict[str,Any]:
    return {'task_id':task_id,'kind':'mcq','question':question,'choices':list(choices),'answer':_LETTERS[answer_idx]}
def load_arc(limit=None)->List[Dict[str,Any]]:
    out=[]
    try:
        from datasets import load_dataset
        ds=load_dataset('allenai/ai2_arc','ARC-Challenge',split='test')
        for i,r in enumerate(ds):
            labels=r['choices']['label'];texts=r['choices']['text'];key=r['answerKey']
            if key not in labels:continue
            out.append(_norm_mcq(r['question'],texts,labels.index(key),f'arc/{i}'))
            if limit and len(out)>=limit:break
    except Exception as e:print(f'[suite_bench] ARC unavailable ({e})',flush=True)
    return out
def load_mmlu(limit=None)->List[Dict[str,Any]]:
    out=[]
    try:
        from datasets import load_dataset
        ds=load_dataset('cais/mmlu','all',split='test')
        for i,r in enumerate(ds):
            out.append(_norm_mcq(r['question'],r['choices'],int(r['answer']),f"mmlu/{r.get('subject','')}/{i}"))
            if limit and len(out)>=limit:break
    except Exception as e:print(f'[suite_bench] MMLU unavailable ({e})',flush=True)
    return out
def load_hellaswag(limit=None)->List[Dict[str,Any]]:
    out=[]
    try:
        from datasets import load_dataset
        ds=load_dataset('Rowan/hellaswag',split='validation')
        for i,r in enumerate(ds):
            if r.get('label','')=='':continue
            out.append(_norm_mcq(r['ctx'],r['endings'],int(r['label']),f'hellaswag/{i}'))
            if limit and len(out)>=limit:break
    except Exception as e:print(f'[suite_bench] HellaSwag unavailable ({e})',flush=True)
    return out
def load_winogrande(limit=None)->List[Dict[str,Any]]:
    out=[]
    try:
        from datasets import load_dataset
        ds=load_dataset('winogrande','winogrande_xl',split='validation')
        for i,r in enumerate(ds):
            ans=r.get('answer','')
            if ans not in ('1','2'):continue
            out.append(_norm_mcq(r['sentence'],[r['option1'],r['option2']],int(ans)-1,f'winogrande/{i}'))
            if limit and len(out)>=limit:break
    except Exception as e:print(f'[suite_bench] Winogrande unavailable ({e})',flush=True)
    return out
def load_gsm8k(limit=None)->List[Dict[str,Any]]:
    out=[]
    try:
        from datasets import load_dataset
        ds=load_dataset('gsm8k','main',split='test')
        for i,r in enumerate(ds):
            m=re.search(r'####\s*([\-\d,\.]+)',r['answer'])
            if not m:continue
            out.append({'task_id':f'gsm8k/{i}','kind':'numeric','question':r['question'],'answer':m.group(1).replace(',','').strip()})
            if limit and len(out)>=limit:break
    except Exception as e:print(f'[suite_bench] GSM8K unavailable ({e})',flush=True)
    return out
def load_sample(limit=None)->List[Dict[str,Any]]:
    p=_sample_path()
    if not p.exists():return []
    items=json.loads(p.read_text(encoding='utf-8'))
    return items[:limit] if limit else items
_SUITES={'arc':load_arc,'mmlu':load_mmlu,'hellaswag':load_hellaswag,'winogrande':load_winogrande,'gsm8k':load_gsm8k,'sample':load_sample}
def format_prompt(item:Dict[str,Any])->str:
    if item['kind']=='numeric':
        return f"Solve this problem step by step. End with the final numeric answer prefixed by '####'.\n\nProblem: {item['question']}\n"
    opts='\n'.join(f'{_LETTERS[i]}. {c}' for i,c in enumerate(item['choices']))
    return f"Answer the multiple-choice question with ONLY the single letter of the correct option.\n\n{item['question']}\n{opts}\nAnswer:"
def extract_answer(item:Dict[str,Any],output:str)->str:
    out=output or ''
    if item['kind']=='numeric':
        m=re.search(r'####\s*([\-\d,\.]+)',out)
        if m:return m.group(1).replace(',','').strip().rstrip('.')
        nums=re.findall(r'-?\d[\d,]*\.?\d*',out.replace(',',''))
        return nums[-1].rstrip('.') if nums else ''
    n=len(item['choices']);valid=set(_LETTERS[:n])
    m=re.search(r'(?:answer\s*(?:is|:)?\s*)\(?([A-H])\)?',out,re.IGNORECASE)
    if m and m.group(1).upper() in valid:return m.group(1).upper()
    for ch in out:
        if ch.upper() in valid:return ch.upper()
    return ''
def _num_eq(a:str,b:str)->bool:
    try:return abs(float(a)-float(b))<1e-6
    except Exception:return str(a).strip()==str(b).strip()
def score_item(item:Dict[str,Any],output:str)->bool:
    got=extract_answer(item,output)
    return _num_eq(got,item['answer']) if item['kind']=='numeric' else (got==item['answer'])
def run_suite(name:str,generate_fn:Callable,limit:Optional[int]=None,items:Optional[List]=None)->Dict[str,Any]:
    items=items if items is not None else _SUITES.get(name,lambda limit=None:[])(limit=limit)
    if not items:return {'suite':name,'n':0,'accuracy':0.0,'results':[],'reason':'no items loaded'}
    res=[];correct=0
    for it in items:
        try:o=generate_fn(format_prompt(it))
        except Exception as e:o=f'(gen error: {e})'
        ok=score_item(it,o);correct+=1 if ok else 0
        res.append({'task_id':it['task_id'],'correct':ok,'got':extract_answer(it,o),'want':it['answer']})
    n=len(items)
    return {'suite':name,'n':n,'accuracy':round(100.0*correct/n,1),'correct':correct,'results':res}
_REF={'arc':{'gpt-4o':96.7,'llama-3.1-8b':83.4,'gemma-2-2b':55.7,'phi-3-mini':84.9},
      'mmlu':{'gpt-4o':88.7,'llama-3.1-8b':68.0,'gemma-2-2b':51.3,'phi-3-mini':68.8},
      'gsm8k':{'gpt-4o':92.0,'llama-3.1-8b':84.5,'gemma-2-2b':62.6,'phi-3-mini':82.5},
      'hellaswag':{'gpt-4o':95.3,'llama-3.1-8b':78.5,'gemma-2-2b':73.0,'phi-3-mini':78.9},
      'winogrande':{'gpt-4o':87.5,'llama-3.1-8b':77.4,'gemma-2-2b':68.3,'phi-3-mini':74.6}}
def leaderboard(adam_scores:Dict[str,float],adam_label:str='Adam (this run)')->str:
    suites=[s for s in ('arc','mmlu','gsm8k','hellaswag','winogrande') if s in _REF]
    rows=[adam_label]+sorted({m for s in suites for m in _REF[s]})
    hdr='| Model | '+' | '.join(s.upper() for s in suites)+' |'
    sep='|'+'---|'*(len(suites)+1)
    lines=['Leaderboard (accuracy %, published references — verify against current cards):',hdr,sep]
    def cell(model,s):
        if model==adam_label:v=adam_scores.get(s);return f'{v}' if v is not None else 'n/a'
        return f"{_REF[s].get(model,'n/a')}"
    for m in rows:lines.append('| '+m+' | '+' | '.join(cell(m,s) for s in suites)+' |')
    return '\n'.join(lines)
def render_chart(adam_scores:Dict[str,float],out_path:str,adam_label:str='Adam')->Dict[str,Any]:
    try:import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    except Exception as e:return {'rendered':False,'reason':f'matplotlib unavailable: {e}'}
    suites=[s for s in ('arc','mmlu','gsm8k','hellaswag','winogrande') if s in adam_scores or s in _REF]
    models=[adam_label,'gemma-2-2b','llama-3.1-8b','gpt-4o']
    import numpy as np
    x=np.arange(len(suites));w=0.2
    fig,ax=plt.subplots(figsize=(11,5))
    for i,m in enumerate(models):
        vals=[(adam_scores.get(s,0) if m==adam_label else _REF.get(s,{}).get(m,0)) for s in suites]
        ax.bar(x+(i-1.5)*w,vals,w,label=m)
    ax.set_xticks(x);ax.set_xticklabels([s.upper() for s in suites]);ax.set_ylabel('accuracy %');ax.set_ylim(0,100)
    ax.set_title('Adam vs reference models — leaderboard benchmarks');ax.legend();fig.tight_layout()
    fig.savefig(out_path,dpi=120);plt.close(fig)
    return {'rendered':True,'path':out_path}
