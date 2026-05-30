"""modern_bench — 2026-relevant benchmark loaders/scorers. Legacy ARC/HellaSwag/Winogrande saturate at frontier so
vendors stopped reporting them; this module covers what the current leaderboards actually read:
  - MMLU-Pro          (TIGER-Lab/MMLU-Pro)        : harder MCQ, 10-way, reasoning-heavy
  - MATH-500          (HuggingFaceH4/MATH-500)    : competition math, boxed-answer numeric
  - HumanEval+        (evalplus/humanevalplus)    : HumanEval with stricter/extra tests
  - MBPP+             (evalplus/mbppplus)         : same upgrade for MBPP
  - GPQA-Diamond      (Idavidrein/gpqa)           : PhD-level MCQ (gated; load on demand)
  - SWE-Bench-Verified (princeton-nlp/SWE-bench_Verified) : real PR fixes — needs a sandbox
Offline-testable via a bundled synthetic sample; real suites load on demand."""
import re,json
from pathlib import Path
from typing import Dict,Any,List,Optional,Callable
_LETTERS='ABCDEFGHIJ'
def _sample_path()->Path:return Path(__file__).resolve().parent/'modern_sample.json'
def _norm_mcq(question:str,choices:List[str],answer_idx:int,task_id:str)->Dict[str,Any]:
    return {'task_id':task_id,'kind':'mcq','question':question,'choices':list(choices),'answer':_LETTERS[answer_idx]}
def load_mmlu_pro(limit=None,subjects=None)->List[Dict[str,Any]]:
    out=[]
    try:
        from datasets import load_dataset
        ds=load_dataset('TIGER-Lab/MMLU-Pro',split='test')
        for i,r in enumerate(ds):
            if subjects and r.get('category','') not in subjects:continue
            opts=r.get('options') or r.get('choices') or []
            ai=int(r.get('answer_index',r.get('answer',-1)))
            if ai<0 or ai>=len(opts):continue
            out.append(_norm_mcq(r['question'],opts,ai,f"mmlu_pro/{r.get('category','')}/{i}"))
            if limit and len(out)>=limit:break
    except Exception as e:print(f'[modern_bench] MMLU-Pro unavailable ({e})',flush=True)
    return out
def load_math500(limit=None)->List[Dict[str,Any]]:
    out=[]
    try:
        from datasets import load_dataset
        ds=load_dataset('HuggingFaceH4/MATH-500',split='test')
        for i,r in enumerate(ds):
            ans=str(r.get('answer','')).strip()
            if not ans:continue
            out.append({'task_id':f'math500/{r.get("subject","")}/{i}','kind':'boxed','question':r['problem'],'answer':ans})
            if limit and len(out)>=limit:break
    except Exception as e:print(f'[modern_bench] MATH-500 unavailable ({e})',flush=True)
    return out
def load_humaneval_plus(limit=None)->List[Dict[str,Any]]:
    out=[]
    try:
        from datasets import load_dataset
        ds=load_dataset('evalplus/humanevalplus',split='test')
        for i,r in enumerate(ds):
            out.append({'task_id':r.get('task_id',f'HEp/{i}'),'kind':'code','prompt':r['prompt'],'entry_point':r['entry_point'],'test':r.get('test',''),'canonical':r.get('canonical_solution','')})
            if limit and len(out)>=limit:break
    except Exception as e:print(f'[modern_bench] HumanEval+ unavailable ({e})',flush=True)
    return out
def load_mbpp_plus(limit=None)->List[Dict[str,Any]]:
    out=[]
    try:
        from datasets import load_dataset
        ds=load_dataset('evalplus/mbppplus',split='test')
        for i,r in enumerate(ds):
            out.append({'task_id':r.get('task_id',f'MBPPp/{i}'),'kind':'code','prompt':r['prompt'],'entry_point':r.get('entry_point',''),'test':r.get('test',''),'canonical':r.get('canonical_solution','')})
            if limit and len(out)>=limit:break
    except Exception as e:print(f'[modern_bench] MBPP+ unavailable ({e})',flush=True)
    return out
def load_gpqa_diamond(limit=None)->List[Dict[str,Any]]:
    out=[]
    try:
        from datasets import load_dataset
        ds=load_dataset('Idavidrein/gpqa','gpqa_diamond',split='train')
        for i,r in enumerate(ds):
            correct=r.get('Correct Answer','');opts=[correct]+[r.get(f'Incorrect Answer {k}','') for k in (1,2,3)]
            opts=[o for o in opts if o]
            if len(opts)<2:continue
            out.append(_norm_mcq(r.get('Question',''),opts,0,f'gpqa/{i}'))
            if limit and len(out)>=limit:break
    except Exception as e:print(f'[modern_bench] GPQA-Diamond unavailable ({e}; likely gated — accept on HF)',flush=True)
    return out
def load_modern_sample(limit=None)->List[Dict[str,Any]]:
    p=_sample_path()
    if not p.exists():return []
    items=json.loads(p.read_text(encoding='utf-8'))
    return items[:limit] if limit else items
_SUITES={'mmlu_pro':load_mmlu_pro,'math500':load_math500,'humanevalplus':load_humaneval_plus,'mbppplus':load_mbpp_plus,'gpqa_diamond':load_gpqa_diamond,'modern_sample':load_modern_sample}
def format_prompt(item:Dict[str,Any])->str:
    k=item['kind']
    if k=='mcq':
        opts='\n'.join(f'{_LETTERS[i]}. {c}' for i,c in enumerate(item['choices']))
        return f"Answer the multiple-choice question. Think step by step, then end with a line: 'Answer: X' where X is the single letter.\n\n{item['question']}\n{opts}\nReasoning:"
    if k=='boxed':
        return f"Problem: {item['question']}\nSolve step by step. Put your final answer inside \\boxed{{ ... }} on its own line.\nSolution:"
    if k=='code':
        return item['prompt']
    return item.get('question','')
def _extract_letter(out:str,n_choices:int)->str:
    valid=set(_LETTERS[:n_choices])
    m=re.search(r'(?:answer\s*(?:is|:)?\s*)\(?([A-J])\)?',out,re.IGNORECASE)
    if m and m.group(1).upper() in valid:return m.group(1).upper()
    mb=re.search(r'(?:^|[^A-Za-z])([A-J])(?:[^A-Za-z]|$)',out)
    if mb and mb.group(1).upper() in valid:return mb.group(1).upper()
    for ch in out:
        if ch.upper() in valid:return ch.upper()
    return ''
def _extract_boxed(out:str)->str:
    m=re.search(r'\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',out or '')
    if m:return m.group(1).strip()
    m2=re.search(r'(?:final answer|answer)\s*[:=]\s*([^\n]+)',out or '',re.IGNORECASE)
    if m2:return m2.group(1).strip().rstrip('.')
    nums=re.findall(r'-?\d[\d,]*\.?\d*',(out or '').replace(',',''))
    return nums[-1].rstrip('.') if nums else ''
def extract_answer(item:Dict[str,Any],output:str)->str:
    return _extract_letter(output or '',len(item['choices'])) if item['kind']=='mcq' else (_extract_boxed(output or '') if item['kind']=='boxed' else (output or ''))
def _math_eq(a:str,b:str)->bool:
    try:return abs(float(a)-float(b))<1e-6
    except Exception:pass
    sa,sb=re.sub(r'\s+','',str(a)),re.sub(r'\s+','',str(b))
    return sa==sb or sa.replace('+','').replace('-','-')==sb
def score_item(item:Dict[str,Any],output:str,exec_fn:Optional[Callable]=None)->bool:
    k=item['kind']
    if k=='mcq':return extract_answer(item,output)==item['answer']
    if k=='boxed':return _math_eq(extract_answer(item,output),item['answer'])
    if k=='code' and exec_fn is not None:return bool(exec_fn(item,output))
    return False
def run_suite(name:str,generate_fn:Callable,limit:Optional[int]=None,items:Optional[List]=None,exec_fn:Optional[Callable]=None)->Dict[str,Any]:
    items=items if items is not None else _SUITES.get(name,lambda limit=None:[])(limit=limit)
    if not items:return {'suite':name,'n':0,'accuracy':0.0,'results':[],'reason':'no items loaded'}
    res=[];correct=0
    for it in items:
        try:o=generate_fn(format_prompt(it))
        except Exception as e:o=f'(gen error: {e})'
        ok=score_item(it,o,exec_fn=exec_fn);correct+=1 if ok else 0
        res.append({'task_id':it['task_id'],'correct':ok,'got':extract_answer(it,o)[:160],'want':it.get('answer','')})
    n=len(items)
    return {'suite':name,'n':n,'accuracy':round(100.0*correct/n,1),'correct':correct,'results':res}
def self_consistency(generate_fn:Callable,item:Dict[str,Any],k:int=5)->str:
    from collections import Counter
    answers=[];raw_outputs=[]
    for _ in range(max(1,k)):
        try:o=generate_fn(format_prompt(item))
        except Exception:o=''
        raw_outputs.append(o);a=extract_answer(item,o)
        if a:answers.append(a)
    if not answers:return raw_outputs[-1] if raw_outputs else ''
    winner,_=Counter(answers).most_common(1)[0]
    return f"Answer: {winner}" if item['kind']=='mcq' else f"\\boxed{{{winner}}}"
_REF_PUBLISHED_STALE={'mmlu_pro':{'gemma-3-4b-it':43.6,'qwen3-4b':56.9,'phi-4-mini-3.8b':52.8},
             'math500':{'gemma-3-4b-it':75.6,'qwen3-4b':84.0,'phi-4-mini-3.8b':71.7},
             'humanevalplus':{'gemma-3-4b-it':71.3,'qwen3-4b':77.4,'phi-4-mini-3.8b':74.4},
             'gpqa_diamond':{'gemma-3-4b-it':30.8,'qwen3-4b':41.7,'phi-4-mini-3.8b':36.9}}
_STALE_WARNING='WARNING: _REF_PUBLISHED_STALE are PREVIOUS-GEN (gemma-3 not 4, qwen3 not 3.6/3.7, older phi) AND run under DIFFERENT harnesses (their prompts/n-shot/CoT/extraction). NOT directly comparable to Adam-this-harness. The ONLY valid comparison is a same-harness control run (see leaderboard_controlled).'
def leaderboard_controlled(adam_scores,control_scores=None,adam_label='Adam (GF17, this harness)',control_label='Gemma-4-E2B parent (bf16, this harness)')->str:
    suites=[s for s in ('mmlu_pro','math500','humanevalplus','gpqa_diamond') if s in adam_scores or (control_scores and s in control_scores)]
    if not suites:return '(no scores yet)'
    hdr='| Model | '+' | '.join(s.upper() for s in suites)+' |';sep='|'+'---|'*(len(suites)+1)
    lines=['Same-harness controlled comparison (apples-to-apples; the valid one):',hdr,sep]
    def row(label,sc):return '| '+label+' | '+' | '.join((f'{sc.get(s)}' if sc.get(s) is not None else 'n/a') for s in suites)+' |'
    lines.append(row(adam_label,adam_scores))
    if control_scores:
        lines.append(row(control_label,control_scores))
        lines.append('| DELTA (Adam - parent) | '+' | '.join((f'{adam_scores[s]-control_scores[s]:+.1f}' if (s in adam_scores and s in control_scores) else 'n/a') for s in suites)+' |')
        lines.append('')
        lines.append('Interpretation: delta ~0 = GF17 bake is faithful/lossless. delta strongly negative = GF17 costs accuracy. delta positive (beyond noise) = unexpected, investigate.')
    else:
        lines.append('')
        lines.append('No parent control run yet — run scripts/bench_parent.py on gemma-4-E2B-it through this same harness to populate the comparison.')
    return '\n'.join(lines)
def leaderboard_modern(adam_scores:Dict[str,float],adam_label:str='Adam (this run)')->str:
    suites=[s for s in ('mmlu_pro','math500','humanevalplus','gpqa_diamond') if s in _REF_PUBLISHED_STALE]
    rows=[adam_label]+sorted({m for s in suites for m in _REF_PUBLISHED_STALE[s]})
    hdr='| Model | '+' | '.join(s.upper() for s in suites)+' |'
    sep='|'+'---|'*(len(suites)+1)
    lines=[_STALE_WARNING,'',hdr,sep]
    def cell(model,s):
        if model==adam_label:v=adam_scores.get(s);return f'{v}' if v is not None else 'n/a'
        return f"{_REF_PUBLISHED_STALE[s].get(model,'n/a')}"
    for m in rows:lines.append('| '+m+' | '+' | '.join(cell(m,s) for s in suites)+' |')
    return '\n'.join(lines)
