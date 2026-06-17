"""ReAct-style agentic loop — give Adam a goal, it plans tool steps, executes, observes, iterates.
Bounded by max_steps. Returns trace + final answer. Adam's mini-Qwen tier-3 svc generates the plan; tools are the existing skill registry."""
import os,re,json,time
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
- file_write / code_edit: put ONLY the path (plus find/replace for code_edit) in the JSON args — do NOT put the file body inside the JSON. Immediately AFTER the JSON line, write the COMPLETE file content in ONE fenced code block. This avoids all JSON-escaping errors. Example:
  {{"tool":"file_write","args":{{"path":"foo.py"}}}}
  ```python
  def foo():
      return 1
  ```
  Write real newlines and quotes normally inside the fence — never escape them, never truncate.
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
    s=re.sub(r'\\x([0-9a-fA-F]{2})',r'\\u00\1',s)
    s=re.sub(r'\\(?![\\/"bfnrtu]|u[0-9a-fA-F]{4})',r'\\\\',s)
    s=re.sub(r'"(\w+):"([^"]*?)"','"\\1":"\\2"',s)
    s=re.sub(r'"(\w+):"','"\\1":',s)
    s=re.sub(r'([{,]\s*)([a-zA-Z_]\w*)\s*:','\\1"\\2":',s)
    s=re.sub(r',(\s*[}\]])','\\1',s)
    s=s.replace("True","true").replace("False","false").replace("None","null")
    return s
def _escape_ctrl_in_strings(s:str)->str:
    out=[];in_str=False;esc=False
    for c in s:
        if esc:out.append(c);esc=False;continue
        if c=='\\':out.append(c);esc=True;continue
        if c=='"':in_str=not in_str;out.append(c);continue
        if in_str:
            o=ord(c)
            out.append('\\n' if c=='\n' else '\\r' if c=='\r' else '\\t' if c=='\t' else '' if o<0x20 else c)
        else:out.append(c)
    return ''.join(out)
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
        _ec=_escape_ctrl_in_strings(candidate)
        for attempt in (candidate,_ec,candidate.replace("'",'"'),_lenient_json_repair(candidate),_lenient_json_repair(_ec),_lenient_json_repair(candidate.replace("'",'"')),_escape_ctrl_in_strings(_lenient_json_repair(candidate))):
            try:
                result=json.loads(attempt)
                if isinstance(result,dict):return result
            except Exception:continue
    return None
def _format_trace(steps:List[Dict])->str:
    if not steps:return ''
    lines=['','Trace so far:'];n=len(steps);_rc=int(os.environ.get('AMNI_TRACE_RECENT_CHARS','1800'))
    for i,s in enumerate(steps,1):
        if 'tool' not in s:continue
        cap=_rc if i>n-2 else 220
        lines.append(f'Step {i}: called {s["tool"]}({json.dumps(s.get("args",{}),default=str)[:120]}) -> {str(s.get("result"))[:cap]}')
    return '\n'.join(lines)
def _trace_for_prompt(steps,compact,since)->str:
    recent=steps[since:];parts=[]
    if compact:parts.append('\nProgress so far (compacted working memory):\n'+compact)
    if recent:parts.append(_format_trace(recent) if compact else _format_trace(steps))
    return '\n'.join(p for p in parts if p) if (compact or recent) else ''
def _extract_loc(result):
    try:
        if isinstance(result,dict):
            for k in ('hits','symbols'):
                arr=result.get(k)
                if arr and isinstance(arr,list) and isinstance(arr[0],dict) and arr[0].get('path'):return arr[0].get('path'),arr[0].get('line')
    except Exception:pass
    return None,None
def _next_step_nudge(steps):
    loc_path=loc_line=None;loc_idx=-1
    for j,s in enumerate(steps):
        p,l=_extract_loc(s.get('result'))
        if p:loc_path,loc_line,loc_idx=p,l,j
    if loc_idx>=0:
        read_after=any(s.get('tool')=='file_read' for s in steps[loc_idx+1:])
        if not read_after and loc_line is not None:return f'(You ALREADY located it: path="{loc_path}", line={loc_line}. Do NOT locate again. Your NEXT step MUST be file_read with args {{"path":"{loc_path}","line_offset":{loc_line},"line_limit":40}}.)'
    return '(Do not repeat the same call. ADVANCE — choose one: (1) take the next concrete action the goal asks for (file_write {"path":...,"content":...} / code_edit / run verification); (2) if you are stuck or lack info, ask the user: {"ask_user":"<what you found, what you don\'t understand, what specific info would help>"}; (3) make your BEST ATTEMPT from what you know so we can observe the result. If already done, emit {"final":"..."}.)'
def _format_tool_lines(skills_list):
    lines=[]
    for s in skills_list:
        schema=s.get('schema') or {}
        args_str=','.join(f'{k}:{v}' for k,v in schema.items()) if schema else ''
        desc=(s.get('desc') or '').split('.')[0][:120]
        lines.append(f"- {s['name']}({args_str}): {desc}")
    return '\n'.join(lines)
_EMPTY_RESEARCH_SIGNS=('do not contain','does not contain','no useful','could not find',"couldn't find",'no information','not address','do not directly','no relevant','did not find','unable to find','no sources','no results','not found any')
def _is_empty_research(out)->bool:
    try:t=json.dumps(out,default=str).lower() if isinstance(out,dict) else str(out).lower();return any(s in t for s in _EMPTY_RESEARCH_SIGNS)
    except Exception:return False
_EMPTY_RESEARCH_HINT='(That search found nothing useful on this topic. Do NOT repeat the same search. Choose ONE: (a) ask the user — emit {"ask_user":"<state that you found nothing, what you do/don\'t understand, and the specific info that would help>"}; or (b) make your BEST ATTEMPT now from what you already know — write the file / make the change so we can run it and learn from the result.)'
def _debug_steer(tname,out,targs):
    o=json.dumps(out,default=str).lower() if isinstance(out,dict) else str(out).lower();p=str((targs or {}).get('path') or '')
    if any(s in o for s in ('no such file','filenotfound','does not exist','file not found','file does not exist')):
        return f'(DEBUG: the file "{p}" does not exist yet. To CREATE a new file you MUST use file_write {{"path":"{p}","content":"<the full file text>"}} — code_edit/code_diff only MODIFY a file that already exists. Use file_write now.)' if tname in ('code_edit','code_diff') else f'(DEBUG: path "{p}" not found — check it is correct and relative to the workdir.)'
    if 'find string not present' in o:return '(DEBUG: your code_edit "find" text is not in the file. file_read the exact region again, then copy a SHORT verbatim fragment that actually appears there as your "find".)'
    if 'unknown tool' in o or 'unknown skill' in o:return f'(DEBUG: there is no tool named "{tname}". To WRITE a new file use file_write {{"path":...,"content":...}}; to MODIFY an existing file use code_edit {{"path":...,"find":...,"replace":...}}.)'
    if 'mid-identifier' in o:return '(DEBUG: extend your find to end/start at a clean word boundary, then retry the code_edit.)'
    if 'syntax error' in o:return '(DEBUG: fix the reported syntax error with code_edit, then it will be re-checked.)'
    return f'(DEBUG: the last call failed — {str(out)[:160]}. Do something DIFFERENT to fix it; do not repeat the same call.)'
def _stalled(steps,window=4)->bool:
    real=[s for s in steps if 'tool' in s and not s.get('auto') and not str(s.get('tool','') or '').startswith('(')]
    if len(real)<3:return False
    import difflib
    from collections import Counter
    last=real[-window:]
    tools=[s.get('tool') for s in last]
    if Counter(tools).most_common(1)[0][1]>=3:return True
    res=[str(s.get('result'))[:400] for s in last]
    if sum(1 for r in res if any(sg in r.lower() for sg in _EMPTY_RESEARCH_SIGNS))>=2:return True
    wp=[json.dumps(s.get('args',{}),default=str)[:200] for s in last if s.get('tool') in ('file_write','code_edit','code_diff')]
    if len(wp)>=2 and len(set(wp))<len(wp):return True
    sims=[difflib.SequenceMatcher(None,a,b).ratio() for a,b in zip(res,res[1:])]
    return bool(sims and max(sims)>=0.85)
def _parse_nmod(text):
    t=(text or '').lower();_m=17;_n=4
    mm=re.search(r'gf\s*\(?\s*(\d+)|mod(?:ulo)?\s+(\d+)',t)
    if mm:_m=int(mm.group(1) or mm.group(2))
    nm=re.search(r'(\d+)\s*[- ]?point|length\s*[- ]?\s*(\d+)|groups? of (\d+)',t)
    if nm:_n=int(nm.group(1) or nm.group(2) or nm.group(3))
    return max(2,min(_n,64)),max(2,min(_m,257))
def _neutralize_main(src):
    return (src or '').replace('__name__ == "__main__"','False').replace("__name__ == '__main__'",'False')
def _func_names(src):
    import ast as _a
    try:t=_a.parse(src or '')
    except Exception:return []
    return [n.name for n in t.body if isinstance(n,_a.FunctionDef) and len(n.args.args)==1 and not n.args.vararg and not n.args.kwonlyargs]
def _synth_roundtrip(names,n,mod,samples=48):
    arr='['+','.join(names)+']'
    return ("\nimport numpy as _np\n_fns=%s\n_n,_m=%d,%d\n_ok=False\nfor _f in _fns:\n for _g in _fns:\n  if _f is _g:continue\n  _good=True\n  for _t in range(%d):\n   _x=[int(_v) for _v in _np.random.randint(0,_m,_n)]\n   try:\n    _y=_g(_f(_x));_rr=_y.tolist() if hasattr(_y,'tolist') else list(_y);_rr=[int(_v)%%_m for _v in _rr]\n   except Exception:\n    _good=False;break\n   if _rr!=[_v%%_m for _v in _x]:_good=False;break\n  if _good:_ok=True;break\n if _ok:break\nassert _ok,'no forward/inverse pair round-trips over GF(%d) length-%d'\nprint('ROUNDTRIP_OK')\n")%(arr,n,mod,samples,mod,n)
def _find_existing_test(skills,art_rel):
    import os
    try:p=str(skills._abs(art_rel) if hasattr(skills,'_abs') else art_rel)
    except Exception:return ''
    d=os.path.dirname(p);stem=os.path.splitext(os.path.basename(p))[0]
    if stem.startswith('test_') or stem.endswith('_test'):return ''
    for c in (os.path.join(d,'test_'+stem+'.py'),os.path.join(d,stem+'_test.py'),os.path.join(d,'tests','test_'+stem+'.py'),os.path.join(os.path.dirname(d),'tests','test_'+stem+'.py')):
        if os.path.isfile(c):return c
    try:
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.py') and (fn.startswith('test_') or fn.endswith('_test.py')):
                try:
                    if re.search(r'\b'+re.escape(stem)+r'\b',open(os.path.join(d,fn),encoding='utf-8',errors='ignore').read()):return os.path.join(d,fn)
                except Exception:pass
    except Exception:pass
    return ''
def _strip_self_import(src,stem):
    return re.sub(r'(?m)^[ \t]*(?:from[ \t]+%s[ \t]+import[ \t].*|import[ \t]+%s(?:[ \t].*|))$'%(re.escape(stem),re.escape(stem)),'',src or '')
def run_goal_stream(adam,skills,goal:str,max_steps:int=8,timeout_s:float=240.0,plan_prompt=None,ask_cb=None):
    t0=time.time();_PT=plan_prompt or _PLAN_PROMPT
    _gfm=re.findall(r'\b[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|pyw|js|mjs|cjs|ts|tsx|jsx|rs|go|cpp|cc|c|java|rb|md|json|txt|html|css)\b',goal or '')
    _goalfile=_gfm[0] if _gfm else ''
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
    if os.environ.get('AMNI_AGENTIC_AUTOINDEX','1')=='1' and skills.has('code_index'):
        try:
            _ix_adam=getattr(adam,'adam',None) or adam
            _ix=skills.call('code_index',{'action':'build','root':str(getattr(skills,'workdir',''))},ctx={'adam':_ix_adam,'agent':adam})
            if _ix.ok and _ix.output:
                _ixsum={'languages':_ix.output.get('languages'),'files_indexed':_ix.output.get('files_indexed') or _ix.output.get('n_files'),'symbols':_ix.output.get('symbols') or _ix.output.get('n_symbols')}
                steps.append({'tool':'code_index','args':{'action':'build'},'result':_ixsum,'auto':True})
                yield {'event':'dir_learned','step':0,'summary':_ixsum}
                try:
                    _ta=getattr(adam,'adam',None) or adam
                    if hasattr(_ta,'teach'):_ta.teach(f'codebase index for goal: {goal[:80]}',f'Indexed workspace: {json.dumps(_ixsum,default=str)[:300]}')
                except Exception:pass
        except Exception as _ie:print(f'[agentic] auto code_index failed: {_ie}',flush=True)
    _edited=False;_verified=False;_vnudge=False
    _VERIFY_HINT='(NOTE: you changed code but have NOT verified it. Before final, run a verification step — shell {"cmd":"node --check <file>"} for JavaScript, or the test_run tool — to confirm the fix parses/works.)'
    _compact='';_since=0;_cevery=int(os.environ.get('AMNI_COMPACT_EVERY','4'));_runtag=str(int(t0));_last_reflect=-99
    _progress_at=0;_read_paths=set();_HARD=int(os.environ.get('AMNI_MAX_NOPROGRESS','7'));_unparse=0
    _last_artifact='';_crit_rounds=0;_CRIT=int(os.environ.get('AMNI_CRITIQUE','1'));_CRIT_MAX=int(os.environ.get('AMNI_CRITIQUE_ROUNDS','2'));_pinned=[];_forced_write=False;_created_code=set();_edit_miss={}
    _cdir=__import__('pathlib').Path(os.environ.get('AMNI_COMPACT_DIR','experiences/agentic_compact'))
    try:_cdir.mkdir(parents=True,exist_ok=True)
    except Exception:pass
    def _critique_final(answer):
        _s=getattr(getattr(adam,'adam',None),'svc',None) or getattr(adam,'svc',None)
        if _s is None:return None
        _cax=adam.adam if hasattr(adam,'adam') and not hasattr(adam,'crawler_plugin') else adam
        _art=''
        if _last_artifact:
            try:_art=open(str(skills._abs(_last_artifact) if hasattr(skills,'_abs') else _last_artifact),encoding='utf-8',errors='ignore').read()[:3000]
            except Exception:_art=''
        _ispy=str(_last_artifact).lower().endswith(('.py','.pyw'))
        _invkw=any(k in (goal or '').lower() for k in ('inverse','reversible','roundtrip','round-trip','round trip','bit-exact','bit exact','recovers','lossless'))
        yield {'event':'critique_start','step':i+1,'artifact':_last_artifact or '(answer only)'}
        _testreq=('\n\nALSO provide an executable "test" (assume the code above is ALREADY defined in scope; do NOT re-import or redefine it). Make the test ROBUST so a WRONG test cannot mislead us:\n- Do NOT assert hardcoded magic numbers you worked out in your head — that is the #1 source of wrong tests. Instead assert STRUCTURAL/RELATIONAL properties, or compare against an INDEPENDENT reference you compute INSIDE the test from the requirement.'+(' Specifically: for many random length-4 vectors x of values 0..16, assert that applying the forward then the inverse returns x exactly.' if _invkw else ' Example: for average() assert `average([k]*n)==k` (structural) or `average(xs)==sum(xs)/len(xs)` (reference) rather than `average([2,4,6])==4` (hand-computed, easy to get wrong).')+'\n- Cover SEVERAL inputs including an edge case; raise AssertionError on mismatch.') if (_art and _ispy) else ''
        _cp='You are reviewing your OWN finished work with a SKEPTICAL, adversarial eye. Do NOT be agreeable — your job is to FIND faults, not praise. You are a 3B model: do NOT trust your own hand-arithmetic, trust the executable test.\n\nORIGINAL GOAL:\n'+goal+(('\n\nPINNED FACTS the user gave you — the code MUST match these EXACT constants/values. If it uses anything different, that ALONE makes it FAULTY:\n- '+'\n- '.join(_pinned)) if _pinned else '')+'\n\nYOUR PROPOSED FINAL ANSWER:\n'+str(answer)[:800]+(('\n\nThe actual file you produced ('+_last_artifact+'):\n```\n'+_art+'\n```') if _art else '')+'\n\nWork log:\n'+_trace_for_prompt(steps,_compact,_since)[:1000]+'\n\nCritically examine the work against the goal: wrong logic/constants, missing pieces the goal demanded, unhandled cases, or claims not backed by a real run.'+_testreq+'\nOutput ONE JSON line:\n{"acceptable": true or false, "fault":"<single most important concrete defect, empty if none>", "fix":"<the ONE specific next action to fix it, empty if none>", "test":"<executable python asserting the key property, empty if not applicable>"}'
        try:_cr,_=_s.chat(_cp,system='You are a strict adversarial reviewer of your own work. Default to skepticism. Prefer an executable test over hand-tracing. Output ONLY the JSON line.',max_new_tokens=520,do_sample=False,kb_top_k=0)
        except Exception:_cr=None
        _cv=_parse_step(_cr or '')
        if _cv is None:return None
        _acc=_cv.get('acceptable');_acc=(_acc is True) or (str(_acc).strip().lower() in ('true','yes','1','ok','acceptable','accept'))
        _fault=str(_cv.get('fault') or '')[:400];_fix=str(_cv.get('fix') or '')[:400]
        _test=str(_cv.get('test') or '').strip();_tres='none';_failmsg='';_exec='none'
        if _art and _ispy and skills.has('run_python'):
            _cands=[];_fnames=_func_names(_art);_authoritative=False
            _exist=_find_existing_test(skills,_last_artifact)
            if _exist:
                try:
                    import os as _os2;_astem=_os2.path.splitext(_os2.path.basename(str(_last_artifact)))[0]
                    _exsrc=_strip_self_import(open(_exist,encoding='utf-8',errors='ignore').read(),_astem)
                    if _exsrc.strip():_cands.append((_neutralize_main(_art)+'\n'+_exsrc,True))
                    yield {'event':'critique_existing_test','step':i+1,'test_file':_os2.path.basename(_exist)}
                except Exception:pass
            if _invkw and len(_fnames)>=2:
                _n,_m=_parse_nmod(goal+' '+' '.join(_pinned));_cands.append((_neutralize_main(_art)+_synth_roundtrip(_fnames,_n,_m),True))
            if _test:_cands.append((_art+'\n'+_test,False))
            for _code,_auth in _cands:
                try:
                    _tr=skills.call('run_python',{'code':_code,'timeout':int(os.environ.get('AMNI_PYRUN_TIMEOUT','25'))},ctx={'adam':_cax,'agent':adam})
                    _to=_tr.output if _tr.ok else {'error':str(_tr.error)}
                    if isinstance(_to,dict):
                        _trc=_to.get('returncode');_r=('timeout' if (_to.get('timed_out') or _to.get('killed')) else ('blocked' if _to.get('error') else ('pass' if _trc==0 else 'fail')))
                        if _r in ('pass','fail'):_tres=_r;_authoritative=_auth;_failmsg=(str(_to.get('stderr') or 'assertion failed').strip().splitlines() or ['fail'])[-1][:240];break
                except Exception:_tres='error'
            if _tres=='pass':_exec='pass'
            elif _tres=='fail' and _authoritative:_exec='fail'
            elif _tres=='fail' and not _authoritative:
                _ap='You are auditing whether a TEST is correct. You have NOT seen the implementation — judge the TEST ALONE. Independently work out, step by step, what the expected values SHOULD be from the requirement, then decide if the test asserts the right thing.\n\nREQUIREMENT:\n'+goal+(('\n\nKnown facts (authoritative):\n- '+'\n- '.join(_pinned)) if _pinned else '')+'\n\nThe test below FAILED ('+(_failmsg or '')[:160]+'):\n```python\n'+_test[:1400]+'\n```\n\nIs the TEST itself correct (right assertions + right expected values + no bug IN THE TEST)? Output ONE JSON line:\n{"test_valid": true or false, "reason":"<one sentence>"}'
                _tv=None
                try:
                    _ar,_=_s.chat(_ap,system='You audit a test for correctness IN ISOLATION, recomputing expected values yourself. Output ONLY the JSON line.',max_new_tokens=240,do_sample=False,kb_top_k=0)
                    _av=_parse_step(_ar or '')
                    if _av is not None:
                        _tvr=_av.get('test_valid');_tv=((_tvr is True) or (str(_tvr).strip().lower() in ('true','yes','1','valid'))) if _tvr is not None else None
                except Exception:_tv=None
                yield {'event':'critique_test_audit','step':i+1,'test_valid':_tv}
                if _tv is True:_exec='fail'
        if _exec=='pass':_acc=True;_fault=''
        elif _exec=='fail':_acc=False;_fault=_fault or ('executable check FAILED — '+(_failmsg or 'assertion failed'));_fix=_fix or 'fix the code so the check passes'
        else:_acc=True
        _v={'acceptable':_acc,'fault':_fault,'fix':_fix,'test':_tres,'exec':_exec}
        yield {'event':'critique','step':i+1,'acceptable':_acc,'fault':_fault,'fix':_fix,'test':_tres,'exec':_exec}
        return _v
    for i in range(max_steps):
        if time.time()-t0>timeout_s:yield {'event':'timeout','wall_s':round(time.time()-t0,2)};return
        if i-_progress_at>=_HARD:
            _ha=f'Stopping after {i-_progress_at} steps with no progress. Accomplished: {(_compact or _format_trace(steps[-3:]))[:500]}. I was unable to complete the goal — likely blocked on: {str(steps[-1].get("result"))[:200] if steps else "unknown"}.'
            yield {'event':'no_progress_stop','step':i+1,'no_progress_steps':i-_progress_at}
            yield {'event':'final','step':i+1,'answer':_ha,'n_steps':len(steps),'wall_s':round(time.time()-t0,2)};return
        if i>1 and (i-_last_reflect)>=3 and _stalled(steps):
            _last_reflect=i;_svc=getattr(getattr(adam,'adam',None),'svc',None) or getattr(adam,'svc',None)
            if _svc is not None:
                yield {'event':'self_reflect_start','step':i+1,'reason':'repeated/similar steps detected — re-grounding on the goal'}
                _rp='You appear to be REPEATING similar steps without progress. Pause and re-ground.\n\nORIGINAL GOAL:\n'+goal+'\n\nWhat you have done so far:\n'+_trace_for_prompt(steps,_compact,_since)+'\n\nRe-read the goal and judge HONESTLY whether it is already accomplished. Output ONE JSON line:\n{"done": true or false, "reason":"<one sentence: is the goal achieved? if not, what is missing?>", "next_action":"<if not done, the ONE concrete next step that is DIFFERENT from what you keep repeating; else empty>"}'
                try:_rr,_=_svc.chat(_rp,system='You are a strict self-reviewer. Re-read the goal, judge completion honestly, output ONLY the JSON line.',max_new_tokens=220,do_sample=False,kb_top_k=0);_ref=_parse_step(_rr or '')
                except Exception:_ref=None
                if _ref is not None:
                    _rdone=_ref.get('done');_rdone=(_rdone is True) or (str(_rdone).strip().lower() in ('true','yes','1','done','complete'))
                    _rreason=str(_ref.get('reason') or '')[:300];_rnext=str(_ref.get('next_action') or '')[:300]
                    yield {'event':'self_reflect','step':i+1,'done':_rdone,'reason':_rreason,'next_action':_rnext}
                    if _rdone:
                        if _edited and not _verified and not _vnudge:
                            _vnudge=True;steps.append({'tool':'(verify-reminder)','args':{},'result':_VERIFY_HINT});yield {'event':'verify_required','step':i+1,'hint':_VERIFY_HINT};continue
                        _cv=(yield from _critique_final('Goal complete (self-review): '+_rreason)) if (_CRIT and _edited and _crit_rounds<_CRIT_MAX) else None
                        if _cv and not _cv['acceptable']:
                            _crit_rounds+=1;_verified=False;_vnudge=False
                            steps.append({'tool':'(critique)','args':{},'result':'(CRITICAL SELF-REVIEW: NOT done — '+(_cv['fault'] or 'a defect remains')+'. Next: '+(_cv['fix'] or 'fix the defect and re-verify')+'.)'})
                            yield {'event':'critique_reject','step':i+1,'round':_crit_rounds,'fault':_cv['fault']};continue
                        yield {'event':'final','step':i+1,'answer':f'Goal complete (self-review): {_rreason}','n_steps':len(steps),'wall_s':round(time.time()-t0,2)};return
                    steps.append({'tool':'(self-review)','args':{},'result':f'(Self-review: the goal is NOT done — {_rreason}. STOP repeating prior steps. Your next step MUST be: {_rnext})'})
                    continue
        _pin=('PINNED FACTS — the user gave you these EXACT values; use them verbatim and never contradict or substitute them:\n- '+'\n- '.join(_pinned)+'\n\n') if _pinned else ''
        prompt=_PT.format(tools=tool_list,goal=goal,trace=_pin+_trace_for_prompt(steps,_compact,_since))
        try:
            svc=getattr(getattr(adam,'adam',None),'svc',None) or getattr(adam,'svc',None)
            if svc is None:yield {'event':'error','msg':'no svc available'};return
            _is_codegen=any(k in (goal or '').lower() for k in ('code','game','script','program','app','rust','python','javascript','typescript','go ','c++','cpp','java','rb','php','wasm','file_write','create','build','make','implement','generate','write a'))
            _planner_budget=int(os.environ.get('AMNI_PLANNER_BUDGET') or (2400 if _is_codegen else 400))
            resp,n=svc.chat(prompt,system='You are a precise planning agent. Output ONE JSON tool-call line. For file_write/code_edit, put ONLY the path in the JSON, then the COMPLETE file body in a fenced code block right after — write code normally in the fence, never escape or truncate it.',max_new_tokens=_planner_budget,do_sample=False,kb_top_k=0)
        except Exception as e:yield {'event':'error','msg':f'plan call failed: {e}'};return
        plan=_parse_step(resp or '')
        if plan is None:
            _unparse+=1
            yield {'event':'plan_unparseable','step':i+1,'raw':(resp or '')[:400],'retry':_unparse}
            if _unparse>=3:
                yield {'event':'final','step':i+1,'answer':f'Stopping: could not produce valid JSON after {_unparse} attempts. Progress: {(_compact or "")[:300]}','n_steps':len(steps),'wall_s':round(time.time()-t0,2)};return
            steps.append({'tool':'(format-error)','args':{},'result':'(Your last output was NOT valid JSON, so it could not be executed. Emit EXACTLY ONE JSON object and nothing else — e.g. {"tool":"file_write","args":{"path":"x.py","content":"line1\\nline2"}} or {"final":"..."}. If writing a code file, escape every newline as \\n and every double-quote as \\", OR write a shorter/simpler file. Try again now.)'})
            continue
        _unparse=0
        _thought=str(plan.get('thought') or plan.get('reasoning') or '').strip()[:280]
        if _thought:yield {'event':'thought','step':i+1,'thought':_thought}
        if 'final' in plan:
            if _edited and not _verified and not _vnudge:
                _vnudge=True;steps.append({'tool':'(verify-reminder)','args':{},'result':_VERIFY_HINT})
                yield {'event':'verify_required','step':i+1,'hint':_VERIFY_HINT};continue
            _cv=(yield from _critique_final(plan['final'])) if (_CRIT and _edited and _crit_rounds<_CRIT_MAX) else None
            if _cv and not _cv['acceptable']:
                _crit_rounds+=1;_verified=False;_vnudge=False
                steps.append({'tool':'(critique)','args':{},'result':'(CRITICAL SELF-REVIEW: your answer is NOT yet correct — '+(_cv['fault'] or 'a defect remains')+'. Do NOT finalize. Next: '+(_cv['fix'] or 'fix the defect and re-verify')+'. After fixing, the file is re-run; trust the RUN result not your assumption.)'})
                yield {'event':'critique_reject','step':i+1,'round':_crit_rounds,'fault':_cv['fault']};continue
            yield {'event':'final','step':i+1,'answer':plan['final'],'n_steps':len(steps),'wall_s':round(time.time()-t0,2)};return
        tname=plan.get('tool','');targs=plan.get('args',{}) or {}
        if str(tname).lower() in ('file_write','code_edit','code_diff') and isinstance(targs,dict) and not str(targs.get('content') or targs.get('code') or '').strip():
            _fb=re.search(r'```[A-Za-z0-9_+.\-]*\r?\n(.*?)```',resp or '',re.DOTALL)
            if _fb:
                _code=_fb.group(1);_code=_code[:-1] if _code.endswith('\n') else _code
                targs['code' if str(tname).lower() in ('code_edit','code_diff') else 'content']=_code
                plan['args']=targs;yield {'event':'fence_extracted','step':i+1,'tool':tname,'chars':len(_code)}
        if str(tname).lower() in ('final','finish','done','complete','submit'):
            _fa=(targs.get('final') or targs.get('answer') or plan.get('final') or (next((str(v) for v in targs.values() if v),'') if isinstance(targs,dict) else str(targs)) or 'done')
            if _edited and not _verified and not _vnudge:
                _vnudge=True;steps.append({'tool':'(verify-reminder)','args':{},'result':_VERIFY_HINT})
                yield {'event':'verify_required','step':i+1,'hint':_VERIFY_HINT};continue
            _cv=(yield from _critique_final(_fa)) if (_CRIT and _edited and _crit_rounds<_CRIT_MAX) else None
            if _cv and not _cv['acceptable']:
                _crit_rounds+=1;_verified=False;_vnudge=False
                steps.append({'tool':'(critique)','args':{},'result':'(CRITICAL SELF-REVIEW: your answer is NOT yet correct — '+(_cv['fault'] or 'a defect remains')+'. Do NOT finalize. Next: '+(_cv['fix'] or 'fix the defect and re-verify')+'. After fixing, the file is re-run; trust the RUN result not your assumption.)'})
                yield {'event':'critique_reject','step':i+1,'round':_crit_rounds,'fault':_cv['fault']};continue
            yield {'event':'final','step':i+1,'answer':str(_fa),'n_steps':len(steps),'wall_s':round(time.time()-t0,2)};return
        if ('ask_user' in plan) or str(tname).lower() in ('ask_user','ask','clarify','help','request_info','need_info','ask_human'):
            _q=str(plan.get('ask_user') or (isinstance(targs,dict) and (targs.get('question') or targs.get('query') or targs.get('ask') or targs.get('text')) or '') or plan.get('question') or '').strip()[:600]
            yield {'event':'help_request','step':i+1,'question':_q}
            _resp=None
            try:_resp=ask_cb(_q) if ask_cb else None
            except Exception:_resp=None
            if _resp:
                _pinned.append(str(_resp)[:600]);steps.append({'tool':'(user-answered)','args':{'question':_q[:120]},'result':f'The user answered: {_resp}. You MUST use these EXACT values/constants verbatim in your code — do NOT substitute your own (e.g. if they named a specific root/constant, use THAT one).'})
                _progress_at=i+1;yield {'event':'help_answer','step':i+1,'answer':str(_resp)[:400]};continue
            steps.append({'tool':'(user-unavailable)','args':{'question':_q[:120]},'result':'(No user answer available right now. Do NOT ask again or search again. Make your BEST ATTEMPT now using what you already know — write the file / make the change — so we can observe and learn from the result.)'})
            yield {'event':'help_unanswered','step':i+1,'question':_q};continue
        if isinstance(targs,str):
            _coerced={'run_python':'code','code_edit':'code','web':'query','shell':'cmd','scan':'path','file_read':'path','file_write':'content','mem':'query'}.get(tname)
            targs={_coerced:targs} if _coerced else {}
            yield {'event':'args_normalized','step':i+1,'tool':tname,'coerced_to':list(targs.keys())}
        _key=(tname,json.dumps(targs,sort_keys=True,default=str)[:400])
        _dup_count=sum(1 for s in steps[-3:] if (s.get('tool'),json.dumps(s.get('args',{}),sort_keys=True,default=str)[:400])==_key)
        if _dup_count>=3:
            _readish=str(tname).lower() in ('web','mem','find','code_index','scan','file_read','project_info','git','parse_error')
            if _readish and not _edited and not _forced_write:
                _forced_write=True
                _pf=('\n\nUse these EXACT values the user already gave you:\n- '+'\n- '.join(_pinned)) if _pinned else ''
                _ff=(' Use the EXACT filename the goal names: file_write {"path":"'+_goalfile+'","content":"..."}.') if _goalfile else ''
                steps.append({'tool':'(force-write)','args':{},'result':'(STOP. You have repeated '+str(tname)+' '+str(_dup_count+1)+' times and already have what you need.'+_pf+'\nDo NOT search, read, or index again. Your VERY NEXT output MUST be a file_write that CREATES the deliverable the goal asks for, with the COMPLETE code in "content".'+_ff+' Write it now.)'})
                yield {'event':'force_write','step':i+1,'tool':tname,'after_repeats':_dup_count+1};continue
            _last_out=steps[-1].get('result') if steps else None
            _was_ok=not (isinstance(_last_out,str) and _last_out.startswith(('error:','exception:','unknown tool')))
            _ans=f'Stopped: planner repeated {tname} {_dup_count+1}x despite next-step guidance. Last useful result: {str(_last_out)[:300]}'
            yield {'event':'auto_final','step':i+1,'reason':'duplicate_step_stuck','answer':_ans,'tool':tname,'last_ok':_was_ok}
            yield {'event':'final','step':i+1,'answer':_ans,'n_steps':len(steps),'wall_s':round(time.time()-t0,2)};return
        if _dup_count>=1:
            _nudge=_next_step_nudge(steps)
            steps.append({'tool':tname,'args':targs,'result':_nudge})
            yield {'event':'duplicate_skipped','step':i+1,'tool':tname,'hint':_nudge[:200]};continue
        yield {'event':'step_start','step':i+1,'tool':tname,'args':targs}
        if not skills.has(tname):
            err=f'unknown tool: {tname}';steps.append({'tool':tname,'args':targs,'result':err})
            yield {'event':'step_result','step':i+1,'tool':tname,'ok':False,'output':err}
            _ds=_debug_steer(tname,err,targs)
            if _ds:steps.append({'tool':'(debug-hint)','args':{},'result':_ds});yield {'event':'debug_hint','step':i+1,'hint':_ds[:200]}
            continue
        _adam_for_ctx=adam.adam if hasattr(adam,'adam') and not hasattr(adam,'crawler_plugin') else adam
        try:
            r=skills.call(tname,targs,ctx={'adam':_adam_for_ctx,'agent':adam})
            out=r.output if r.ok else f'error: {r.error}'
            _had_err=isinstance(out,dict) and bool(out.get('error'))
            steps.append({'tool':tname,'args':targs,'result':out})
            if tname in ('web','mem') and r.ok and _is_empty_research(out):
                steps.append({'tool':'(research-note)','args':{},'result':_EMPTY_RESEARCH_HINT})
                yield {'event':'research_empty','step':i+1,'tool':tname,'hint':'found nothing — ask the user or attempt with what you know'}
            if tname=='test_run' or (tname=='shell' and '--check' in str(targs.get('cmd',''))):_verified=True
            _applied=r.ok and not _had_err and tname in ('code_edit','file_write','code_diff') and isinstance(out,dict) and (int(out.get('replacements') or 0)>0 or out.get('applied') or tname=='file_write')
            yield {'event':'step_result','step':i+1,'tool':tname,'ok':bool(r.ok) and not _had_err,'output':str(out)[:600]}
            if (not r.ok or _had_err) and tname not in ('web','mem'):
                _ds=_debug_steer(tname,out,targs)
                if _ds:steps.append({'tool':'(debug-hint)','args':{},'result':_ds});yield {'event':'debug_hint','step':i+1,'hint':_ds[:200]}
            if tname in ('code_edit','code_diff') and _had_err and ('not present' in str(out).lower() or 'not found' in str(out).lower()):
                _mp=str(targs.get('path') or '');_edit_miss[_mp]=_edit_miss.get(_mp,0)+1
                if _edit_miss[_mp]>=2:
                    _haveart=bool(_created_code) and (not _goalfile or any(_goalfile in c for c in _created_code))
                    _esc='(You have tried '+str(_edit_miss[_mp])+' times to code_edit "'+_mp+'" with a "find" fragment that is NOT in the file — you are guessing text that does not exist (do NOT invent TODO comments or lines). STOP guessing. '+('Your deliverable ('+(', '.join(sorted(_created_code)))+') is already written and verified — emit {"final":"<summary>"} NOW; the integration the goal mentions only needs to be described in a comment, not actually edited into another file.' if _haveart else 'file_read "'+_mp+'" first and copy an EXACT verbatim line from its output as your find — or emit final if the goal deliverable is already done.')+')'
                    steps.append({'tool':'(edit-miss-escalation)','args':{},'result':_esc});yield {'event':'edit_miss_escalation','step':i+1,'path':_mp,'misses':_edit_miss[_mp]}
            if _applied:
                _edited=True;_verified=False;_ep=str(targs.get('path') or '');_last_artifact=_ep or _last_artifact
                if _ep.lower().endswith(('.py','.pyw','.js','.mjs','.cjs','.ts','.jsx','.tsx','.rs','.go','.c','.cpp','.cc','.java','.rb')) and tname=='file_write' and out.get('created'):
                    if _created_code and _ep not in _created_code:
                        _canon=_goalfile or sorted(_created_code)[0]
                        steps.append({'tool':'(file-proliferation)','args':{},'result':'(STOP creating NEW files — this is a loop. You already created: '+', '.join(sorted(_created_code))+'. Creating another new filename wastes the prior work'+((' AND the goal explicitly asks for the file "'+_goalfile+'"') if _goalfile else '')+'. Pick ONE file ('+_canon+') and FIX it IN PLACE with code_edit {"path":"'+_canon+'","find":"<short verbatim fragment from it>","replace":"<the fix>"}; or if it is already correct, emit {"final":"..."}. Do NOT write a different new filename again.)'})
                        yield {'event':'file_proliferation','step':i+1,'created':sorted(_created_code),'newest':_ep,'canonical':_canon}
                    _created_code.add(_ep)
                if _ep.lower().endswith(('.js','.mjs','.cjs','.ts','.jsx','.tsx')) and skills.has('shell'):
                    try:
                        _vr=skills.call('shell',{'cmd':f'node --check {_ep}'},ctx={'adam':_adam_for_ctx,'agent':adam})
                        _vo=_vr.output if _vr.ok else f'error: {_vr.error}'
                        _vok=bool(_vr.ok and isinstance(_vo,dict) and _vo.get('returncode')==0);_verified=_vok
                        _steer=f'EDIT APPLIED to {_ep} (replacements={out.get("replacements")}) and node --check {"PASSED" if _vok else "FAILED"}. '+('Your ONLY remaining step is to emit {"final":"..."}. Do NOT read, locate, or edit again.' if _vok else 'Fix the syntax error with code_edit, then re-check.')
                        steps.append({'tool':'shell','args':{'cmd':f'node --check {_ep}'},'result':{'auto_verify':_vo,'directive':_steer},'auto':True})
                        yield {'event':'auto_verify','step':i+1,'ok':_vok,'cmd':f'node --check {_ep}','directive':_steer}
                    except Exception:pass
                elif _ep.lower().endswith(('.py','.pyw')):
                    try:
                        import ast as _ast
                        _ap=skills._abs(_ep) if hasattr(skills,'_abs') else _ep
                        _txt=open(str(_ap),encoding='utf-8',errors='ignore').read()
                        try:_ast.parse(_txt);_pok=True;_perr='';_pkind='syntax'
                        except SyntaxError as _se:_pok=False;_perr=f'{_se.msg} at line {_se.lineno}';_pkind='syntax'
                        if _pok and skills.has('run_python'):
                            _runc=_txt.replace('__name__ == "__main__"','True').replace("__name__ == '__main__'",'True')
                            _rr=skills.call('run_python',{'code':_runc,'timeout':int(os.environ.get('AMNI_PYRUN_TIMEOUT','25'))},ctx={'adam':_adam_for_ctx,'agent':adam})
                            _ro=_rr.output if _rr.ok else {'error':str(_rr.error)}
                            if isinstance(_ro,dict):
                                _rc=_ro.get('returncode')
                                if _ro.get('timed_out') or _ro.get('killed'):_pok=False;_perr='self-test did NOT finish — it timed out or was killed (too slow, hung, or an infinite loop). This is NOT a pass. Make the self-test fast, finite, and deterministic, then it will be re-run.';_pkind='timeout'
                                elif _ro.get('error'):_pok=False;_perr=str(_ro.get('error'))[:240];_pkind='runtime'
                                elif _rc!=0:_pok=False;_perr=(str(_ro.get('stderr') or ('nonzero exit' if _rc else 'no exit code returned — result indeterminate, treated as NOT verified')).strip().splitlines() or ['error'])[-1][:240];_pkind=('runtime' if _rc else 'indeterminate')
                        _verified=_pok
                        _steer=(f'WROTE+RAN {_ep}: it RUNS CLEAN. Your ONLY remaining step is to emit {{"final":"..."}}. Do NOT write or search again.' if _pok else f'{_ep} has a {_pkind.upper()} ERROR: {_perr}. Fix it with code_edit (read the file, correct that exact spot), then it will be re-run. Do NOT rewrite the whole file or search the web.')
                        steps.append({'tool':'(py-run-check)','args':{'path':_ep},'result':{'ok':_pok,'kind':_pkind,'error':_perr,'directive':_steer},'auto':True})
                        yield {'event':'auto_verify','step':i+1,'ok':_pok,'cmd':f'py-run {_ep}','directive':_steer}
                    except Exception:pass
        except Exception as e:
            steps.append({'tool':tname,'args':targs,'result':f'exception: {e}'})
            yield {'event':'step_result','step':i+1,'tool':tname,'ok':False,'output':f'exception: {e}'}
        try:
            _rok=('r' in dir()) and getattr(r,'ok',False) and not _had_err
            if _applied:_progress_at=i+1
            elif tname=='file_read' and _rok and str(targs.get('path') or '') not in _read_paths:_read_paths.add(str(targs.get('path') or ''));_progress_at=i+1
            elif tname in ('find','code_index','scan','project_info') and _rok:_progress_at=i+1
        except Exception:pass
        if _cevery>0 and (i+1-_since)>=_cevery and (i+1)<max_steps:
            try:
                _cp='Compress this agent run into working memory so far. Output EXACTLY two lines:\nDONE: <succinct comma-separated list of what is already accomplished, including any file edits and verifications, with key paths/line numbers>\nTODO: <the SINGLE next concrete tool action toward the goal, or the word finish if the goal is fully done and verified>\n\nGoal: '+goal+'\n'+_trace_for_prompt(steps,_compact,_since)
                _csum,_=svc.chat(_cp,system='You compress agent work logs into a DONE/TODO working memory. Output ONLY the two lines starting "DONE:" and "TODO:".',max_new_tokens=190,do_sample=False,kb_top_k=0)
                _csum=(_csum or '').strip()
                if 'DONE' in _csum.upper() and len(_csum)>12:
                    _compact=_csum[:1400];_since=i+1
                    try:(_cdir/f'{_runtag}.atex.md').write_text(_compact,encoding='utf-8')
                    except Exception:pass
                    try:
                        _ta=getattr(adam,'adam',None) or adam
                        if hasattr(_ta,'teach'):_ta.teach(f'agentic progress @step{i+1} for: {goal[:70]}',_compact)
                    except Exception:pass
                    yield {'event':'compacted','step':i+1,'since':_since,'compact':_compact[:500]}
            except Exception:pass
    yield {'event':'max_steps_reached','n_steps':len(steps),'wall_s':round(time.time()-t0,2)}
_BUILD_RE=re.compile(r"\b(?:build|create|make|implement|develop|write|generate|scaffold|set\s+up|give\s+me)\s+(?:(?:me|a|an|the|some|us|out)\s+){0,3}(?:(?:rust|python|js|javascript|ts|typescript|go|c\+\+|rust|wasm|web)\s+)?(?:script|program|game|app|application|web(?:\s*app|site|page)?|tool|cli|api|server|library|module|class|function|component|system|project|module|package|crate|file|code|snippet|example|implementation)\b",re.IGNORECASE)
_FILE_EXT_RE=re.compile(r'\b\w+\.(?:py|rs|js|ts|cpp|cc|c|h|hpp|go|java|rb|sh|html|css|json|yaml|yml|toml|md|txt|cfg|conf|ini)\b',re.IGNORECASE)
_FIX_VERB_RE=re.compile(r"\b(?:fix|debug|diagnose|repair|solve|resolve|troubleshoot|patch|refactor|reproduce|trace|review|audit|inspect|investigate|check|look\s+(?:at|into|through)|go\s+through|figure\s+out|work\s+out|clean\s+up|optimi[sz]e|profile)\b",re.IGNORECASE)
_CODE_OBJ_RE=re.compile(r"\b(?:bugs?|issues?|errors?|crash(?:es|ing|ed)?|exceptions?|tracebacks?|stack\s*traces?|failing|failures?|broken|regressions?|tests?|folders?|director(?:y|ies)|repos?(?:itor(?:y|ies))?|codebases?|code\s*base|code|projects?|modules?|packages?|scripts?|programs?|apps?|applications?|functions?|methods?|classes|files?|imports?|dependenc(?:y|ies)|builds?|compil(?:e|es|ing|ation)|lint(?:er|ing)?|runtime|typos?|warnings?)\b",re.IGNORECASE)
def is_build_request(text:str)->bool:
    if not text:return False
    if _BUILD_RE.search(text):return True
    if len(set(m.lower() for m in _FILE_EXT_RE.findall(text)))>=2 and re.search(r"\b(?:create|make|write|build|generate|implement|add)\b",text,re.IGNORECASE):return True
    if _FIX_VERB_RE.search(text) and (_CODE_OBJ_RE.search(text) or _FILE_EXT_RE.search(text)):return True
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
def _extract_goal_path(goal):
    m=re.search(r'\b([\w\-]+(?:[/\\][\w\-]+)*\.(?:py|js|ts|rs|go|java|rb|cpp|cc|c|sh|html|css|json|txt|md))\b',goal or '')
    return m.group(1).replace('\\','/') if m else 'solution.py'
def run_codegen(adam,skills,goal,max_attempts=4):
    t0=time.time();svc=getattr(getattr(adam,'adam',None),'svc',None) or getattr(adam,'svc',None)
    if svc is None:return {'error':'no svc'}
    path=_extract_goal_path(goal);ctx={'adam':adam,'agent':adam};prep={}
    try:
        from amni.serve import coding_runner as _cr
        prep=_cr.prepare(goal,agent=adam,max_attempts=max_attempts)
    except Exception:prep={}
    run_id=prep.get('run_id');prior=str(prep.get('context','') or '');steps=[];err='';code='';success=False
    for att in range(1,max_attempts+1):
        gp=(f'You are writing the COMPLETE contents of the file {path}.\nTASK: {goal}\n'+(prior+'\n' if att==1 else '')+(f'Your previous version of {path} FAILED when run. Here is the exact error:\n{err[:700]}\nFix THAT specific error and return the FULL corrected file.\n' if err else '')+'Keep it MINIMAL and correct: implement exactly what the task asks, no extra/unrequested methods. Use 4-space indentation only, never tabs. Use plain ASCII identifiers. Do NOT use open()/file I/O, network, subprocess, exec or eval — pure in-memory logic plus the asserts only. Output ONLY ONE fenced code block containing the entire file — real newlines and quotes, never escaped, never truncated, no prose outside the fence.')
        try:resp,_=svc.chat(gp,system='You write complete, minimal, runnable code files with no syntax errors. Use full, consistent identifier names everywhere. Output ONLY a single fenced code block.',max_new_tokens=1800,do_sample=(att>1),kb_top_k=0)
        except Exception as e:return {'error':f'gen failed: {e}','file':path,'final':f'Code generation failed: {e}'}
        fb=re.search(r'```[A-Za-z0-9_+.\-]*\r?\n(.*?)```',resp or '',re.DOTALL)
        code=(fb.group(1) if fb else (resp or '')).rstrip('\n').replace(' ',' ').replace('﻿','').expandtabs(4)
        code=''.join(c for c in code if c in '\n\t' or (ord(c)>=32 and not (0x200b<=ord(c)<=0x200f) and not (0x202a<=ord(c)<=0x202e) and ord(c)!=0x2060))
        if not code.strip():steps.append({'attempt':att,'error':'empty code'});err='model produced no code';continue
        w=skills.call('file_write',{'path':path,'content':code},ctx=ctx)
        if not getattr(w,'ok',False):steps.append({'attempt':att,'error':f'write failed: {getattr(w,"error","?")}'});err=str(getattr(w,'error',''));continue
        if path.endswith('.py'):
            import ast as _ast
            try:_ast.parse(code)
            except SyntaxError as _se:
                steps.append({'attempt':att,'wrote':path,'ok':False,'stderr':f'SyntaxError line {_se.lineno}: {_se.msg}'})
                err=f'SyntaxError at line {_se.lineno}: {_se.msg}. The bad line is: {(_se.text or "").strip()!r}. Fix ONLY this syntax error and return the full corrected file.';continue
        def _rp(c):
            _r=skills.call('run_python',{'code':c,'timeout':int(os.environ.get('AMNI_PYRUN_TIMEOUT','25'))},ctx=ctx)
            _o=_r.output if isinstance(getattr(_r,'output',None),dict) else {}
            return getattr(_r,'ok',False),str(_o.get('stdout','') or ''),str(_o.get('stderr','') or _o.get('error','') or getattr(_r,'error','') or '')
        rok,so,se=_rp(code)
        for _fx in range(3):
            _nm=re.search(r"name '(\w+)' is not defined\. Did you mean: '(\w+)'",se)
            if not _nm:break
            _nc=re.sub(r'\b'+re.escape(_nm.group(1))+r'\b',_nm.group(2),code)
            if _nc==code:break
            code=_nc;skills.call('file_write',{'path':path,'content':code},ctx=ctx)
            steps.append({'attempt':att,'auto_fix':f'{_nm.group(1)}->{_nm.group(2)}'});rok,so,se=_rp(code)
        _bad=any(k in (se+so) for k in ('Traceback','Error','error:','Exception','assert','exited'))
        ok=rok and not se.strip() and not _bad
        steps.append({'attempt':att,'wrote':path,'ok':ok,'stderr':(se or so)[:300]})
        if ok:success=True;err='';break
        err=(se or so or 'run produced no output and did not confirm the asserts passed')[:700]
    try:
        if run_id:
            from amni.serve import coding_runner as _cr
            _cr.complete(run_id,success=success,outcome=('ran clean'if success else err[:200]),errors=([err[:200]]if err else None),lesson=(f'{path}: '+('verified working'if success else 'fix: '+err[:120])),approach='fenced-codegen deterministic loop',files=[path],agent=adam)
    except Exception:pass
    ans=(f'Done — wrote {path} and it runs clean after {len(steps)} attempt(s); the asserts pass.' if success else f'Wrote {path} but it still fails after {len(steps)} attempt(s). Last error: {err[:240]}')
    return {'goal':goal,'file':path,'success':success,'attempts':len(steps),'steps':steps,'final':ans,'wall_s':round(time.time()-t0,2)}
def _skill_codegen(args,ctx,reg):
    adam=ctx.get('adam');g=args.get('goal') or args.get('task') or ''
    if not g:return {'error':'missing goal'}
    if adam is None:return {'error':'codegen needs adam in ctx'}
    return run_codegen(adam,reg,g,max_attempts=int(args.get('max_attempts',4)))
def register(reg):
    reg.register('goal',_skill_goal,desc='Achieve a multi-step goal: Adam plans tool sequence, executes each step, returns final answer + trace. Args: {goal, max_steps?, timeout_s?}',schema={'goal':'str','max_steps':'int?','timeout_s':'float?'})
    reg.register('codegen',_skill_codegen,desc='Autonomously write a code FILE and make it run: generate complete file -> write -> run -> on failure feed the error back and rewrite, up to max_attempts -> bank the lesson. Args: {goal, max_attempts?}',schema={'goal':'str','max_attempts':'int?'})
