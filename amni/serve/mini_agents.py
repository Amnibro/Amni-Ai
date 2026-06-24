"""Mini multi-agent helpers — a tiny Workflow for Adam. All take an injected chat_fn(prompt, system=...) so they run on the real Adam or a mock. fan_out runs independent subtasks; panel gets N independent answers + a synthesis (self-consistency by reasoning, not voting); review looks through several lenses; debate argues both sides then judges; decompose splits a task, fans out, synthesizes. Sequential by default (one model, one GPU) — cheap, deterministic, no orchestration runtime needed."""
import re
from typing import Callable,List,Dict,Any,Sequence
def _lines(text:str,limit:int=8)->List[str]:
    out=[]
    for ln in (text or '').splitlines():
        s=re.sub(r'^\s*(?:[-*\d.)]+)\s*','',ln).strip()
        if s and len(s)>3:out.append(s)
        if len(out)>=limit:break
    return out
def fan_out(chat_fn:Callable,subtasks:Sequence[str],system:str=None)->List[Dict[str,Any]]:
    return [{'task':t,'result':chat_fn(t,system=system)} for t in subtasks]
def panel(chat_fn:Callable,question:str,n:int=3,system:str=None)->Dict[str,Any]:
    angles=['Answer directly and rigorously.','Answer by first listing what could make this wrong, then conclude.','Answer from first principles, ignoring your first instinct.']
    answers=[chat_fn(f'{question}\n\n({angles[i%len(angles)]})',system=system) for i in range(max(1,n))]
    joined='\n\n'.join(f'[Attempt {i+1}]\n{a}' for i,a in enumerate(answers))
    synth=chat_fn(f'You produced {len(answers)} independent attempts at a question. Reconcile them into one best, correct answer; where they disagree, reason about which is right.\n\nQuestion: {question}\n\n{joined}',system='You synthesize multiple reasoning attempts into the single best answer.')
    return {'question':question,'attempts':answers,'synthesis':synth}
def review(chat_fn:Callable,target:str,lenses:Sequence[str]=('correctness','edge cases','security','clarity'))->Dict[str,Any]:
    findings=[{'lens':L,'review':chat_fn(f'Review the following ONLY for {L}. List concrete, specific issues (each as "- <where>: <issue> -> <fix>"); if none, say "none".\n\n{target}',system='You are a precise, terse reviewer. No praise, only actionable findings.')} for L in lenses]
    return {'lenses':list(lenses),'findings':findings}
def debate(chat_fn:Callable,claim:str)->Dict[str,Any]:
    pro=chat_fn(f'Argue strongly FOR this claim with concrete evidence:\n{claim}',system='You argue the affirmative.')
    con=chat_fn(f'Argue strongly AGAINST this claim with concrete evidence:\n{claim}',system='You argue the negative.')
    verdict=chat_fn(f'Claim: {claim}\n\nFOR:\n{pro}\n\nAGAINST:\n{con}\n\nWeigh both sides and give a calibrated verdict (true / false / uncertain) with the deciding reason.',system='You are an impartial judge.')
    return {'claim':claim,'for':pro,'against':con,'verdict':verdict}
def decompose(chat_fn:Callable,task:str,max_sub:int=4,system:str=None)->Dict[str,Any]:
    plan=chat_fn(f'Break this task into {max_sub} or fewer independent subtasks, one per line, no prose:\n{task}',system='You decompose tasks into a short list of concrete, independent subtasks.')
    subtasks=_lines(plan,max_sub)
    results=fan_out(chat_fn,subtasks,system=system) if subtasks else []
    body='\n\n'.join(f'[{r["task"]}]\n{r["result"]}' for r in results)
    synth=chat_fn(f'Original task: {task}\n\nSubtask results:\n{body}\n\nSynthesize a single complete answer to the original task.',system='You integrate subtask results into one coherent deliverable.')
    return {'task':task,'subtasks':subtasks,'results':results,'synthesis':synth}
def mount(app,agent=None):
    from fastapi import Request
    def _cf():
        from amni.serve.guardian_service import _model
        return (lambda p,system=None:_model(agent,p,system=system)) if agent is not None else (lambda p,system=None:'(no agent)')
    @app.post('/agents/panel')
    async def _panel(req:Request):
        b=await req.json();return panel(_cf(),b.get('question',''),int(b.get('n',3)))
    @app.post('/agents/review')
    async def _review(req:Request):
        b=await req.json();return review(_cf(),b.get('target',''),b.get('lenses') or ('correctness','edge cases','security','clarity'))
    @app.post('/agents/debate')
    async def _debate(req:Request):
        b=await req.json();return debate(_cf(),b.get('claim',''))
    @app.post('/agents/decompose')
    async def _decompose(req:Request):
        b=await req.json();return decompose(_cf(),b.get('task',''),int(b.get('max_sub',4)))
    return app
