"""ReAct-style agentic loop — give Adam a goal, it plans tool steps, executes, observes, iterates.
Bounded by max_steps. Returns trace + final answer. Adam's mini-Qwen tier-3 svc generates the plan; tools are the existing skill registry."""
import re,json,time
from typing import Dict,List,Any,Optional
_PLAN_PROMPT='''You are an agent that achieves a goal by calling tools step by step.
Available tools: {tools}.
Each step output ONLY a single JSON line: {{"tool":"<name>","args":{{...}}}} OR {{"final":"<answer>"}}.
Rules: pick ONE tool per step. After I show you the tool result, decide the next step. When the goal is met, output {{"final":"..."}}.

Goal: {goal}
{trace}
Next step (single JSON line):'''
def _parse_step(text:str)->Optional[Dict]:
    m=re.search(r'\{[^{}]*"(?:tool|final)"[^{}]*(?:\{[^}]*\})?[^{}]*\}',text)
    if not m:return None
    try:return json.loads(m.group(0))
    except Exception:
        try:return json.loads(m.group(0).replace("'",'"'))
        except Exception:return None
def _format_trace(steps:List[Dict])->str:
    if not steps:return ''
    lines=['','Trace so far:']
    for i,s in enumerate(steps,1):
        if 'tool' in s:lines.append(f'Step {i}: called {s["tool"]}({json.dumps(s.get("args",{}))[:80]}) -> {str(s.get("result"))[:200]}')
    return '\n'.join(lines)
def run_goal(adam,skills,goal:str,max_steps:int=5,timeout_s:float=180.0)->Dict[str,Any]:
    t0=time.time()
    tool_list=', '.join(s['name'] for s in skills.list_skills())
    steps:List[Dict[str,Any]]=[]
    final=None;reason=''
    for i in range(max_steps):
        if time.time()-t0>timeout_s:reason='timeout';break
        prompt=_PLAN_PROMPT.format(tools=tool_list,goal=goal,trace=_format_trace(steps))
        try:
            svc=getattr(getattr(adam,'adam',None),'svc',None)
            if svc is None:reason='no svc';break
            resp,n=svc.chat(prompt,system='You are a precise planning agent. Output ONLY JSON.',max_new_tokens=120,do_sample=False,kb_top_k=0)
        except Exception as e:reason=f'plan error: {e}';break
        plan=_parse_step(resp or '')
        if plan is None:
            steps.append({'plan_raw':(resp or '')[:200],'parse_error':True});reason='unparseable plan';break
        if 'final' in plan:final=plan['final'];break
        tname=plan.get('tool','');targs=plan.get('args',{}) or {}
        if not skills.has(tname):
            steps.append({'tool':tname,'args':targs,'result':f'unknown tool: {tname}'});continue
        try:r=skills.call(tname,targs,ctx={'adam':adam})
        except Exception as e:steps.append({'tool':tname,'args':targs,'result':f'error: {e}'});continue
        steps.append({'tool':tname,'args':targs,'result':r.output if r.ok else f'error: {r.error}'})
    if final is None and not reason:reason=f'max_steps ({max_steps}) reached without final'
    return {'goal':goal,'steps':steps,'final':final,'stop_reason':reason or 'final_emitted','wall_s':round(time.time()-t0,3),'n_steps':len(steps)}
def _skill_goal(args,ctx,reg):
    adam=ctx.get('adam')
    g=args.get('goal') or args.get('query') or ''
    if not g:return {'error':'missing goal'}
    if adam is None:return {'error':'goal skill needs adam in ctx'}
    return run_goal(adam,reg,g,max_steps=int(args.get('max_steps',5)),timeout_s=float(args.get('timeout_s',180)))
def register(reg):
    reg.register('goal',_skill_goal,desc='Achieve a multi-step goal: Adam plans tool sequence, executes each step, returns final answer + trace. Args: {goal, max_steps?, timeout_s?}',schema={'goal':'str','max_steps':'int?','timeout_s':'float?'})
