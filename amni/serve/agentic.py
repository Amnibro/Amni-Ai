"""ReAct-style agentic loop — give Adam a goal, it plans tool steps, executes, observes, iterates.
Bounded by max_steps. Returns trace + final answer. Adam's mini-Qwen tier-3 svc generates the plan; tools are the existing skill registry."""
import re,json,time
from typing import Dict,List,Any,Optional
_PLAN_PROMPT='''You are an agent that achieves a goal by calling tools step by step. Assume you CAN accomplish the goal — never refuse.

Available tools (each line: name(args): description):
{tools}

Output ONE JSON line per step:
  {{"tool":"<name>","args":{{...}}}}   to call a tool
  {{"final":"<answer>"}}                when the goal is achieved or cannot be progressed

CRITICAL behavior rules:
- BUDGET: Spend AT MOST 1-2 research steps (mem/web/scan/file_read). After that, you MUST start BUILDING (file_write/code_edit/shell/run_python). Perfect knowledge is not required to start — partial knowledge + iteration is fine.
- Pick ONE tool per step. Wait for its result before deciding next step.
- If you have ANY useful research data, switch to BUILDING (file_write the implementation) rather than searching again.
- If a tool returns an error, DO NOT retry with the same args. Switch tool or change args.
- When the goal is met (or no further progress is possible), output {{"final":"..."}}.
- file_write requires both path and content. code_edit requires path AND (find,replace) OR (code).
- run_python is sandboxed and blocks filesystem mutation, network, exec, subprocess; for file ops use file_write/shell.
- PATHS: all file_write/code_edit/scan/file_read paths must be RELATIVE to the workdir (e.g., "src/lib.rs", "Cargo.toml"). NEVER use absolute paths like "/" or "C:\\". The scan tool defaults to scanning the current workdir if no path given.
- The user's chosen language/framework is non-negotiable. If they asked for Rust, output Rust. Never substitute Python or any other language.
- For multi-file projects: write each file in sequence (one file_write per file). Do NOT call file_write with identical path twice.
- AFTER WRITING CODE: when you write code via file_write/code_edit/code_diff, the NEXT step should usually be test_run (validates the change) or format_code (cleans up style). Then emit {{"final":"..."}}. Do not skip validation for non-trivial code.
- ON TEST FAILURE: if test_run returns passed:false with stderr/stdout, READ the failure details, use parse_error (skill) to identify the kind/line/likely_cause, then use code_diff (preferred) or code_edit to fix the SPECIFIC bug at that line. Do NOT rewrite the whole file. Re-run test_run after the fix to confirm.

Goal: {goal}
{trace}
Next step (single JSON line):'''
def _lenient_json_repair(s:str)->str:
    s=re.sub(r'"(\w+):"([^"]*?)"','"\\1":"\\2"',s)
    s=re.sub(r'"(\w+):"','"\\1":',s)
    s=re.sub(r'([{,]\s*)([a-zA-Z_]\w*)\s*:','\\1"\\2":',s)
    s=re.sub(r',(\s*[}\]])','\\1',s)
    s=s.replace("True","true").replace("False","false").replace("None","null")
    return s
def _parse_step(text:str)->Optional[Dict]:
    if not text:return None
    t=text.strip()
    if t.startswith('```'):
        t=re.sub(r'^```(?:json|JSON)?\s*\n?','',t)
        t=re.sub(r'\n?```\s*$','',t).strip()
    start=t.find('{')
    if start<0:return None
    depth=0;end=-1;in_string=False;escape=False
    for i in range(start,len(t)):
        c=t[i]
        if escape:escape=False;continue
        if c=='\\':escape=True;continue
        if c=='"':in_string=not in_string;continue
        if in_string:continue
        if c=='{':depth+=1
        elif c=='}':
            depth-=1
            if depth==0:end=i+1;break
    candidates=[]
    if end>0:candidates.append(t[start:end])
    _last_brace=t.rfind('}')
    if _last_brace>start:
        _wide=t[start:_last_brace+1]
        if _wide not in candidates:candidates.append(_wide)
    m=re.search(r'\{[^{}]*"(?:tool|final)"[^{}]*(?:\{[^}]*\})?[^{}]*\}',t)
    if m and m.group(0) not in candidates:candidates.append(m.group(0))
    for candidate in candidates:
        for attempt in (candidate,candidate.replace("'",'"'),_lenient_json_repair(candidate),_lenient_json_repair(candidate.replace("'",'"'))):
            try:
                result=json.loads(attempt)
                if isinstance(result,dict):return result
            except Exception:continue
    return None
def _format_trace(steps:List[Dict])->str:
    if not steps:return ''
    lines=['','Trace so far:']
    for i,s in enumerate(steps,1):
        if 'tool' in s:lines.append(f'Step {i}: called {s["tool"]}({json.dumps(s.get("args",{}))[:80]}) -> {str(s.get("result"))[:200]}')
    return '\n'.join(lines)
def _format_tool_lines(skills_list):
    lines=[]
    for s in skills_list:
        schema=s.get('schema') or {}
        args_str=','.join(f'{k}:{v}' for k,v in schema.items()) if schema else ''
        desc=(s.get('desc') or '').split('.')[0][:120]
        lines.append(f"- {s['name']}({args_str}): {desc}")
    return '\n'.join(lines)
def run_goal_stream(adam,skills,goal:str,max_steps:int=8,timeout_s:float=240.0):
    t0=time.time()
    skills_list=skills.list_skills()
    tool_list=_format_tool_lines(skills_list)
    steps:List[Dict[str,Any]]=[]
    yield {'event':'plan_start','goal':goal,'max_steps':max_steps,'tools':[s['name'] for s in skills_list]}
    if skills.has('project_info'):
        try:
            _pi_adam=getattr(adam,'adam',None) or adam
            _pi=skills.call('project_info',{},ctx={'adam':_pi_adam,'agent':adam})
            if _pi.ok and _pi.output:
                _ws_summary={'workdir':_pi.output.get('workdir'),'languages':_pi.output.get('languages',[]),'deps':list((_pi.output.get('dependencies') or {}).keys()),'git_branch':((_pi.output.get('git') or {}).get('branch')),'dirty_files':((_pi.output.get('git') or {}).get('dirty_files')),'top_files':(_pi.output.get('top_files') or [])[:10]}
                steps.append({'tool':'project_info','args':{},'result':_ws_summary,'auto':True})
                yield {'event':'workspace_context','step':0,'summary':_ws_summary}
        except Exception as _pe:print(f'[agentic] auto project_info failed: {_pe}',flush=True)
    for i in range(max_steps):
        if time.time()-t0>timeout_s:yield {'event':'timeout','wall_s':round(time.time()-t0,2)};return
        prompt=_PLAN_PROMPT.format(tools=tool_list,goal=goal,trace=_format_trace(steps))
        try:
            svc=getattr(getattr(adam,'adam',None),'svc',None) or getattr(adam,'svc',None)
            if svc is None:yield {'event':'error','msg':'no svc available'};return
            _is_codegen=any(k in (goal or '').lower() for k in ('code','game','script','program','app','rust','python','javascript','typescript','go ','c++','cpp','java','rb','php','wasm','file_write','create','build','make','implement','generate','write a'))
            _planner_budget=2400 if _is_codegen else 400
            resp,n=svc.chat(prompt,system='You are a precise planning agent. Output ONLY JSON. If the JSON content contains code, output the COMPLETE code — never truncate.',max_new_tokens=_planner_budget,do_sample=False,kb_top_k=0)
        except Exception as e:yield {'event':'error','msg':f'plan call failed: {e}'};return
        plan=_parse_step(resp or '')
        if plan is None:
            yield {'event':'plan_unparseable','step':i+1,'raw':(resp or '')[:400]}
            return
        if 'final' in plan:yield {'event':'final','step':i+1,'answer':plan['final'],'n_steps':len(steps),'wall_s':round(time.time()-t0,2)};return
        tname=plan.get('tool','');targs=plan.get('args',{}) or {}
        if isinstance(targs,str):
            _coerced={'run_python':'code','code_edit':'code','web':'query','shell':'cmd','scan':'path','file_read':'path','file_write':'content','mem':'query'}.get(tname)
            targs={_coerced:targs} if _coerced else {}
            yield {'event':'args_normalized','step':i+1,'tool':tname,'coerced_to':list(targs.keys())}
        _key=(tname,json.dumps(targs,sort_keys=True,default=str)[:400])
        _dup_count=sum(1 for s in steps[-3:] if (s.get('tool'),json.dumps(s.get('args',{}),sort_keys=True,default=str)[:400])==_key)
        if _dup_count>=2:
            _last_out=steps[-1].get('result') if steps else None
            _was_ok=not (isinstance(_last_out,str) and _last_out.startswith(('error:','exception:','unknown tool')))
            _ans=f'Goal achieved via {tname} (called {_dup_count+1}x identically). Last result: {str(_last_out)[:300]}' if _was_ok else f'Aborted: planner stuck repeating {tname} which returned: {str(_last_out)[:300]}'
            yield {'event':'auto_final','step':i+1,'reason':'duplicate_step_x3','answer':_ans,'tool':tname,'last_ok':_was_ok}
            yield {'event':'final','step':i+1,'answer':_ans,'n_steps':len(steps),'wall_s':round(time.time()-t0,2)};return
        if _dup_count==1:
            steps.append({'tool':tname,'args':targs,'result':'(skipped — identical to previous step; try DIFFERENT args, a DIFFERENT tool, or emit {"final":"..."} if goal is complete)'})
            yield {'event':'duplicate_skipped','step':i+1,'tool':tname,'hint':'try different args or output final'};continue
        yield {'event':'step_start','step':i+1,'tool':tname,'args':targs}
        if not skills.has(tname):
            err=f'unknown tool: {tname}';steps.append({'tool':tname,'args':targs,'result':err})
            yield {'event':'step_result','step':i+1,'tool':tname,'ok':False,'output':err};continue
        _adam_for_ctx=adam.adam if hasattr(adam,'adam') and not hasattr(adam,'crawler_plugin') else adam
        try:
            r=skills.call(tname,targs,ctx={'adam':_adam_for_ctx,'agent':adam})
            out=r.output if r.ok else f'error: {r.error}'
            steps.append({'tool':tname,'args':targs,'result':out})
            yield {'event':'step_result','step':i+1,'tool':tname,'ok':bool(r.ok),'output':str(out)[:600]}
        except Exception as e:
            steps.append({'tool':tname,'args':targs,'result':f'exception: {e}'})
            yield {'event':'step_result','step':i+1,'tool':tname,'ok':False,'output':f'exception: {e}'}
    yield {'event':'max_steps_reached','n_steps':len(steps),'wall_s':round(time.time()-t0,2)}
_BUILD_RE=re.compile(r"\b(?:build|create|make|implement|develop|write|generate|scaffold|set\s+up|give\s+me)\s+(?:(?:me|a|an|the|some|us|out)\s+){0,3}(?:(?:rust|python|js|javascript|ts|typescript|go|c\+\+|rust|wasm|web)\s+)?(?:script|program|game|app|application|web(?:\s*app|site|page)?|tool|cli|api|server|library|module|class|function|component|system|project|module|package|crate|file|code|snippet|example|implementation)\b",re.IGNORECASE)
_FILE_EXT_RE=re.compile(r'\b\w+\.(?:py|rs|js|ts|cpp|cc|c|h|hpp|go|java|rb|sh|html|css|json|yaml|yml|toml|md|txt|cfg|conf|ini)\b',re.IGNORECASE)
def is_build_request(text:str)->bool:
    if not text:return False
    if _BUILD_RE.search(text):return True
    if len(set(m.lower() for m in _FILE_EXT_RE.findall(text)))>=2 and re.search(r"\b(?:create|make|write|build|generate|implement|add)\b",text,re.IGNORECASE):return True
    return False
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
