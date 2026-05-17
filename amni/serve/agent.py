"""AmniAgent — wraps Adam with skill dispatch + multi-turn conversation.
Flow: receive user msg → detect skill intent (regex first, Adam classifier fallback) → run skill if matched → synthesize via Adam → persist turn.
Backend stays multifunctional; frontend just sees `{answer, tier, tokens, skill_calls, session_id}`."""
import re,time,json,ast
from typing import Optional,Dict,Any,List,Tuple
from amni.serve.skills import SkillRegistry,default_registry
from amni.serve.conversation import ConversationStore,Conversation
from amni.serve.persona import PersonaStore,Persona,PRESETS as _PERSONA_PRESETS
from amni.serve import tone_atlas
_CALC_PREFIX_RE=re.compile(r'(?:^|\b)(?:compute|calculate|calc|solve)\b',re.IGNORECASE)
_CALC_EXPR_RE=re.compile(r'[\d.]+\s*[+\-*/^]\s*[\d.]+|\bwhat\s+is\s+[\d.]+\s*[+\-*/^x×]\s*[\d.]+|\bwhat\s+is\s+[\d.]+\s+(?:times|plus|minus|over|divided\s+by)\s+[\d.]+',re.IGNORECASE)
_CALC_RE=re.compile(f'(?:{_CALC_PREFIX_RE.pattern})|(?:{_CALC_EXPR_RE.pattern})',re.IGNORECASE)
_TIME_RE=re.compile(r"\b(?:what(?:'s| is)?\s+(?:the\s+)?(?:time|date|day)(?:\s+(?:is\s+it|today|now))?|current\s+time|right\s+now|today's\s+date|tell\s+me\s+the\s+time)\b",re.IGNORECASE)
_WEB_RE=re.compile(r'\b(?:google|find\s+online|news|latest|search\s+(?:online|the\s+web|google)|look\s+up)\b',re.IGNORECASE)
_MEM_RE=re.compile(r'\b(?:search\s+(?:my\s+)?(?:memory|lessons|knowledge|bank)|recall|what\s+do\s+(?:you|adam)\s+know\s+about|find\s+(?:in\s+)?(?:memory|lessons|bank|notes)|remember\s+about|lookup\s+in\s+(?:memory|bank))\s*[:?]?\s*(.*)?$',re.IGNORECASE)
_FILE_READ_RE=re.compile(r'\b(?:read|open|show|cat|display)\s+(?:file\s+|the\s+file\s+)?[\'"`]?([\w\-./\\]+\.\w+)[\'"`]?',re.IGNORECASE)
_FILE_WRITE_RE=re.compile(r'\b(?:write|save|create)\s+(?:file\s+|the\s+file\s+)?[\'"`]?([\w\-./\\]+\.\w+)[\'"`]?',re.IGNORECASE)
_SHELL_RE=re.compile(r'\b(?:run|exec(?:ute)?|shell)\s*[:;]?\s*`?([^`\n]+)`?',re.IGNORECASE)
_CODE_RE=re.compile(r'\b(?:edit|patch|replace|change)\s+.*\bin\b\s+[\'"`]?([\w\-./\\]+\.\w+)[\'"`]?',re.IGNORECASE)
_SCAN_RE=re.compile(r'\b(?:scan|ingest|study|learn\s+from|index|absorb)\s+(?:(?:the|a|this|that|file|files|directory|directories|folder|folders|dir|path|contents?\s+of)\s+)*[\'"`]?([\w\-./:\\*?]+)[\'"`]?',re.IGNORECASE)
_EXPR_EXTRACT=re.compile(r'([\d.]+(?:\s*(?:[+\-*/^x×]|times|plus|minus|over|divided\s+by)\s*[\d.]+)+)',re.IGNORECASE)
_CONTEXT_DEP_RE=re.compile(r"\b(?:my\b|i\s+(?:said|told|asked|mentioned|am|was|do|did|have|like|love|prefer)|you\s+(?:said|told|mentioned|asked|answered|just)|we\s+(?:discussed|said|talked)|again\b|earlier\b|previously\b|before\s+(?:that|we)|just\s+(?:said|asked|answered|told)|that\s+one\b|the\s+last\s+(?:one|thing|question|answer)|(?:what|who|which|where)\s+(?:was|were|did)\s+(?:that|i|my|the\s+last)|remember\b)",re.IGNORECASE)
_INTROSPECT_RE=re.compile(r'\b(?:what\s+can\s+(?:you|adam)\s+do|what\s+are\s+(?:your|adam\'?s?)\s+(?:capabilities|abilities|skills|features)|who\s+are\s+you|what\s+are\s+you|introduce\s+yourself|tell\s+me\s+about\s+(?:yourself|adam)|what\s+is\s+adam|how\s+do\s+you\s+(?:work|remember|learn)|list\s+(?:your\s+)?(?:skills|capabilities|tools)|help\b)',re.IGNORECASE)
_COT_SKIP_CATEGORIES={'greeting','creative','calc_result','time_result','file_result','scan_result','introspect','personal'}
_COT_GENERIC=('Solve this with hyper-effective problem solving. Show your work concisely:\n'
              '1. RESTATE: in one sentence, what is being asked?\n'
              '2. KNOWNS / APPROACH: what facts apply + which mental model fits?\n'
              '3. FIRST SHOT: give your best initial answer with specific values, code, or steps.\n'
              '4. CRITIQUE: what might be wrong, missing, or improvable? Check edge cases + unstated assumptions.\n'
              '5. REFINE with variable-weight perturbations: SMALL (tweak), MEDIUM (alt sub-approach), or LARGE (re-frame).\n'
              '6. FINAL: present the best version. If still uncertain, say so + suggest next experiment.\n'
              'Be terse. Skip steps that add no value for this query.\n')
_COT_CODE=('Code task — keep explanation TERSE, spend tokens on the CODE.\n'
           '1. CLARIFY (one line): inputs/outputs/edge cases.\n'
           '2. APPROACH (one line): algorithm name + key insight.\n'
           '3. CODE: complete, working Python in a single ```python``` block. Use realistic names + type hints. Include all imports. End the code block properly.\n'
           '4. TESTS (3-4 ADVERSARIAL asserts, picking from these case types — vary distinct INPUTS and EXPECTED outputs):\n'
           '   - BOUNDARY: 0, 1, empty string/list, single element\n'
           '   - NEGATIVE/INVALID: negative number, None, wrong type (where applicable)\n'
           '   - LARGE: 10^4 or 10^6 input to stress complexity\n'
           '   - HAPPY PATH: typical realistic input\n'
           '   Each assert must use a DIFFERENT input value. Avoid `assert f(x)==expected` and `assert f(x)==expected` twice with same x.\n'
           '5. COMPLEXITY: one line — time + space.\n'
           '6. CRITIQUE + FIX (only if real issue): name the bug, paste corrected ```python``` block.\n'
           'Goal: working code that passes ADVERSARIAL tests, not just the happy path. Skip prose that does not earn its tokens.\n')
_COT_MATH=('Math problem — use this structure:\n'
           '1. RESTATE: what is given, what is asked, in what units?\n'
           '2. RELEVANT: which formulas, theorems, or identities apply? Cite the principle.\n'
           '3. SOLVE: step-by-step with units carried through every line.\n'
           '4. VERIFY: sanity check — magnitude reasonable? units consistent? alternate method agrees?\n'
           '5. FINAL: boxed numeric answer with units (or symbolic if exact form).\n'
           'Show every step. Never skip arithmetic.\n')
_COT_DEBUG=('Debugging question — use this structure:\n'
            '1. SYMPTOMS: what is observed vs what is expected. Be specific about reproduction conditions.\n'
            '2. HYPOTHESES: list 3-5 candidate causes ranked by base rate / likelihood for this kind of issue.\n'
            '3. EVIDENCE TEST: for each hypothesis, what would you EXPECT to see if it were true? What single quick check would distinguish them?\n'
            '4. NEXT STEP: recommend the highest-information, lowest-cost test first.\n'
            '5. IF FIXING: also state the root cause vs surface fix, and a regression test to add.\n'
            'Don\'t guess the answer — help the user diagnose systematically.\n')
_COT_DESIGN=('System design question — use this structure:\n'
             '1. REQUIREMENTS: functional (what it does) + non-functional (scale, latency, durability, consistency, cost).\n'
             '2. APPROACH: high-level architecture in 3-5 components. Name proven patterns where they apply.\n'
             '3. COMPONENTS: for each — what it does, what tech (be specific), why this over alternatives.\n'
             '4. SCALE + FAILURE: behavior at 10x load? what happens when component X fails? how does the system recover?\n'
             '5. ALTERNATIVES: name 1-2 designs you considered and rejected, and why.\n'
             '6. FINAL: concise diagram-in-prose + the key 2-3 trade-offs the user should know.\n'
             'Be specific about technologies + numbers, not abstract.\n')
_COT_REASONING=('Reasoning question — use this structure:\n'
                '1. RESTATE: what is being asked? what is the implicit claim or question behind it?\n'
                '2. FRAME: which principle, model, or analogy applies?\n'
                '3. CHAIN: build the answer step by step, naming each inferential leap.\n'
                '4. COUNTER: what is the strongest objection to your reasoning? does it hold up?\n'
                '5. FINAL: the answer + your confidence level + what would change your mind.\n')
def _pick_cot(category:str,message:str)->str:
    m=message.lower()
    if any(k in m for k in (' debug','debugging','bug','why is my','why does my','why won','not working','broken','error','crash','hang','leak','slow','flak')):return _COT_DEBUG
    if category=='code' or any(k in m for k in ('write a function','write code','implement','how do i write','write a program','code for','algorithm to','python function','javascript function','rust function')):return _COT_CODE
    if any(k in m for k in ('design a','design the','architect','how would you build','how to build a','scale to','system to handle','rate limit','queue','pipeline architecture')):return _COT_DESIGN
    if any(k in m for k in ('solve for','calculate','compute the','equation','derivative','integral','probability','optimize','minimize','maximize','prove that','theorem','formula for')):return _COT_MATH
    if category=='reasoning':return _COT_REASONING
    return _COT_GENERIC
def _needs_cot(category:str,message:str)->bool:
    if category in _COT_SKIP_CATEGORIES:return False
    if len(message.split())<4:return False
    return True
_PY_BLOCK_RE=re.compile(r'```(?:python|py)?\s*\n(.+?)```',re.DOTALL|re.IGNORECASE)
_ASSERT_RE=re.compile(r'(?:^|\n)\s*`?(assert\s+[^\n`]+?)`?\s*(?=\n|$)',re.MULTILINE)
def _extract_python_blocks(text:str)->List[str]:
    return [m.group(1).strip() for m in _PY_BLOCK_RE.finditer(text) if m.group(1).strip()]
def _extract_asserts(text:str)->List[str]:
    out=[]
    for m in _ASSERT_RE.finditer(text):
        s=m.group(1).strip().rstrip(';,.')
        if s and len(s)<240 and s not in out:out.append(s)
    return out
_ASSERT_ARG_RE=re.compile(r'\(([^)]*)\)')
_BOUND_RE=re.compile(r'(?:\b(?:0|1|None|True|False)\b|""|\'\'|\[\s*\]|\{\s*\}|\(\s*\)|"[a-zA-Z0-9]"|\'[a-zA-Z0-9]\')')
_NEG_RE=re.compile(r'-\d|None|invalid|TypeError|ValueError|raises')
_LARGE_RE=re.compile(r'10\s*\*\*\s*[3-9]|\d{4,}|"[^"]{20,}"|\'[^\']{20,}\'|\[[^\]]*\]\s*\*\s*\d{2,}|range\(\s*\d{3,}')
def _assert_diversity(asserts:List[str])->Tuple[float,dict]:
    if not asserts:return 0.0,{'reason':'no asserts'}
    args=[]
    for a in asserts:
        m=_ASSERT_ARG_RE.search(a)
        if m:args.append(m.group(1).strip())
    distinct_args=len(set(args))
    n=len(asserts)
    arg_score=distinct_args/max(n,1)
    has_bound=any(_BOUND_RE.search(a) for a in asserts)
    has_neg=any(_NEG_RE.search(a) for a in asserts)
    has_large=any(_LARGE_RE.search(a) for a in asserts)
    coverage=sum([has_bound,has_neg,has_large])/3.0
    score=0.5*arg_score+0.5*coverage
    return score,{'n':n,'distinct_args':distinct_args,'bound':has_bound,'neg':has_neg,'large':has_large,'arg_score':round(arg_score,2),'coverage':round(coverage,2)}
def _run_with_tests(skills,adam,snippet:str,asserts:List[str],timeout:int=8)->Tuple[bool,str,dict]:
    if not asserts:return True,'',{}
    test_code=snippet+'\n\n# --- Adam self-tests ---\n'+'\n'.join(asserts)+'\nprint("ALL_TESTS_PASS")'
    try:run=skills.call('run_python',{'code':test_code,'timeout':timeout},ctx={'adam':adam})
    except Exception as e:return False,f'test runner exception: {e}',{}
    if not run.ok or run.output.get('error'):return False,run.output.get('error','test sandbox skipped'),{}
    rc=run.output.get('returncode');so=(run.output.get('stdout') or '');se=(run.output.get('stderr') or '')
    passed=(rc==0 and 'ALL_TESTS_PASS' in so and not se.strip())
    return passed,se.strip() or f'tests rc={rc}',{'rc':rc,'stdout':so[:600],'stderr':se[:400],'asserts_n':len(asserts)}
def _validate_python(blocks:List[str])->List[Tuple[int,str,str]]:
    bad=[]
    for i,code in enumerate(blocks):
        try:ast.parse(code)
        except SyntaxError as e:bad.append((i,code,f'{e.msg} at line {e.lineno}, col {e.offset}'))
        except Exception as e:bad.append((i,code,f'{type(e).__name__}: {e}'))
    return bad
_PERTURB_HINTS={'SMALL':'Apply a SMALL perturbation: tweak ONE value, fix ONE off-by-one, swap ONE operator, or correct ONE name. Keep structure intact.','MEDIUM':'Apply a MEDIUM perturbation: replace an inner step, restructure a loop or branch, or swap a data structure. Keep the overall approach.','LARGE':'Apply a LARGE perturbation: pick a different algorithm or re-frame the problem. The old approach is wrong.'}
def _perturb_once(adam,persona_sys:str,code:str,err:str,magnitude:str,user_msg:str)->str:
    prompt=f'Your code FAILED at runtime:\n```\n{err[:500]}\n```\nOriginal code:\n```python\n{code}\n```\n{_PERTURB_HINTS[magnitude]}\nUser asked: {user_msg}\nOutput ONLY ONE corrected ```python``` block. No prose.'
    sys=persona_sys+'\n\nFix runtime errors. Output a clean python code block only.'
    try:r=adam.chat_persona(prompt,system=sys,max_new_tokens=400,do_sample=False)
    except Exception:return ''
    return (r.get('answer') or '').strip()
def _perturb_retry(adam,skills,persona_sys:str,code:str,err:str,user_msg:str,max_steps:int=3,emit=None,asserts:Optional[List[str]]=None):
    history=[];mags=['SMALL','MEDIUM','LARGE'][:max_steps];cur_code=code;cur_err=err
    for mag in mags:
        ref=_perturb_once(adam,persona_sys,cur_code,cur_err,mag,user_msg)
        if not ref:
            if emit:emit({'magnitude':mag,'status':'no_response'})
            continue
        blocks=_extract_python_blocks(ref)
        if not blocks:
            if emit:emit({'magnitude':mag,'status':'no_code_block'})
            continue
        new_code=blocks[-1]
        if _validate_python([new_code]):
            if emit:emit({'magnitude':mag,'status':'syntax_error'})
            cur_code=new_code;continue
        try:run=skills.call('run_python',{'code':new_code,'timeout':8},ctx={'adam':adam})
        except Exception as e:
            if emit:emit({'magnitude':mag,'status':'exec_exception','error':str(e)[:200]})
            continue
        if not run.ok or run.output.get('error'):
            if emit:emit({'magnitude':mag,'status':'sandbox_skipped','error':(run.output.get('error') or '')[:200]})
            continue
        rc=run.output.get('returncode');so=(run.output.get('stdout') or '').strip();se=(run.output.get('stderr') or '').strip()
        step={'magnitude':mag,'rc':rc,'stdout':so[:300],'stderr':se[:300],'code':new_code}
        history.append(step)
        if rc!=0 or se:
            if emit:emit({'magnitude':mag,'rc':rc,'stdout':so[:300],'stderr':se[:300],'status':'exec_failed'})
            cur_code=new_code;cur_err=se or f'exit code {rc}';continue
        if asserts:
            tpassed,terr,_tinfo=_run_with_tests(skills,adam,new_code,asserts)
            if not tpassed:
                step['tests_passed']=False;step['test_err']=terr[:300]
                if emit:emit({'magnitude':mag,'rc':rc,'stdout':so[:300],'tests_passed':False,'test_err':terr[:300],'status':'tests_failed'})
                cur_code=new_code;cur_err=f'asserts failed: {terr[:400]}';continue
            step['tests_passed']=True
            if emit:emit({'magnitude':mag,'rc':rc,'stdout':so[:300],'tests_passed':True,'status':'all_passed'})
        else:
            if emit:emit({'magnitude':mag,'rc':rc,'stdout':so[:300],'status':'exec_ok'})
        return {'success':True,'magnitude':mag,'code':new_code,'stdout':so,'stderr':se,'history':history,'tests_passed':bool(asserts)}
    return {'success':False,'history':history,'code':cur_code,'stderr':cur_err}
class AmniAgent:
    def __init__(self,adam,skills:Optional[SkillRegistry]=None,store:Optional[ConversationStore]=None,workdir:Optional[str]=None,personas:Optional[PersonaStore]=None,use_persona:bool=True):
        self.adam=adam
        self.skills=skills or default_registry(workdir=workdir)
        self.store=store or ConversationStore()
        self.personas=personas or PersonaStore(adam=adam)
        self.use_persona=use_persona
    def _detect_skill(self,msg:str)->Optional[Tuple[str,Dict[str,Any]]]:
        m=_TIME_RE.search(msg)
        if m:return ('time',{})
        m=_CALC_RE.search(msg)
        if m:
            ex=_EXPR_EXTRACT.search(msg)
            return ('calc',{'expr':ex.group(1) if ex else msg})
        m=_FILE_READ_RE.search(msg)
        if m and self.skills.has('file_read'):return ('file_read',{'path':m.group(1)})
        m=_FILE_WRITE_RE.search(msg)
        if m and self.skills.has('file_write'):
            content_match=re.search(r'(?:with|containing|content[:\s]+)(.+)$',msg,re.IGNORECASE|re.DOTALL)
            return ('file_write',{'path':m.group(1),'content':content_match.group(1).strip() if content_match else ''})
        m=_CODE_RE.search(msg)
        if m and self.skills.has('code_edit'):
            find_m=re.search(r"(?:replace|change|find)\s+[\"'`]([^\"'`]+)[\"'`]\s+(?:with|to)\s+[\"'`]([^\"'`]+)[\"'`]",msg,re.IGNORECASE)
            if find_m:return ('code_edit',{'path':m.group(1),'find':find_m.group(1),'replace':find_m.group(2)})
        m=_SCAN_RE.search(msg)
        if m and self.skills.has('scan'):
            args={'path':m.group(1)}
            if 'distill' in msg.lower() or 'distilled' in msg.lower():args['distill']=True
            return ('scan',args)
        m=_SHELL_RE.search(msg)
        if m and self.skills.has('shell'):return ('shell',{'cmd':m.group(1).strip()})
        m=_MEM_RE.search(msg)
        if m and self.skills.has('mem'):
            q=(m.group(1) or '').strip(' :?.')
            return ('mem',{'query':q if q else msg})
        m=_WEB_RE.search(msg)
        if m and self.skills.has('web'):return ('web',{'query':msg})
        return None
    def chat(self,message:str,session_id:Optional[str]=None,use_skills:bool=True,writeback:bool=True)->Dict[str,Any]:
        t0=time.time()
        conv=self.store.get(session_id)
        conv.append('user',message)
        try:
            from amni.a1.semantic_intent import screen as _sem_screen
            blk,cat,cos,msg_refusal=_sem_screen(message)
            if blk:
                conv.append('assistant',msg_refusal,{'tier':f'tier_intent_block_{cat}','blocked':True,'cos':round(cos,3)})
                return {'answer':msg_refusal,'tier':f'tier_intent_block_{cat}','tokens':0,'session_id':conv.session_id,'skill_calls':[{'skill':'semantic_intent','args':{'cat':cat,'cos':round(cos,3)},'result':{'blocked':True}}],'wall_s':round(time.time()-t0,3),'blocked':True}
        except Exception:pass
        skill_calls:List[Dict[str,Any]]=[]
        skill_answer:Optional[str]=None
        used_tier='tier0_skill' if use_skills else None
        if use_skills:
            det=self._detect_skill(message)
            if det is not None:
                name,args=det
                r=self.skills.call(name,args,ctx={'adam':self.adam,'conv':conv})
                skill_calls.append({'skill':name,'args':args,'result':r.to_dict()})
                if r.ok:
                    skill_answer=self._format_skill_output(name,r.output)
                    used_tier=f'tier0_skill_{name}'
                else:
                    skill_answer=f'(skill {name} failed: {r.error}) Falling back to Adam.'
        persona=self.personas.for_session(conv.session_id) if self.use_persona else _PERSONA_PRESETS['neutral']
        if skill_answer is not None and not skill_answer.startswith('(skill'):
            cat=tone_atlas.classify_intent(message,skill_used=(skill_calls[0]['skill'] if skill_calls else None))
            wrapped=tone_atlas.wrap(skill_answer,cat,persona,seed=message)
            conv.append('assistant',wrapped,{'tier':used_tier,'skill_calls':skill_calls,'tokens':0,'persona':persona.name,'category':cat})
            return {'answer':wrapped,'tier':used_tier,'tokens':0,'session_id':conv.session_id,'skill_calls':skill_calls,'wall_s':round(time.time()-t0,3),'persona':persona.name,'category':cat}
        if _INTROSPECT_RE.search(message):
            ans=self._introspect_answer(persona=persona)
            wrapped=tone_atlas.wrap(ans,'introspect',persona,seed=message)
            conv.append('assistant',wrapped,{'tier':'tier0_introspect','skill_calls':skill_calls,'tokens':0,'persona':persona.name,'category':'introspect'})
            return {'answer':wrapped,'tier':'tier0_introspect','tokens':0,'session_id':conv.session_id,'skill_calls':skill_calls,'wall_s':round(time.time()-t0,3),'persona':persona.name,'category':'introspect'}
        needs_history=bool(_CONTEXT_DEP_RE.search(message)) and len(conv.turns)>2
        recent=conv.transcript(n=6) if needs_history else ''
        category=tone_atlas.classify_intent(message)
        raw_ans='';tier='?';tokens=0
        sl=getattr(self.adam,'sem_lut',None)
        if sl is not None and not needs_history:
            try:
                eff_margin=sl.auto_margin() if hasattr(sl,'auto_margin') else 0.08
                hit=sl.lookup_soft(message,margin=eff_margin) if hasattr(sl,'lookup_soft') else None
                if hit:raw_ans=hit;tier='tier1_5_semantic_lesson';tokens=0
            except Exception:pass
        if not raw_ans:
            lut=getattr(getattr(self.adam,'adam',None),'lut',None)
            if lut is not None and hasattr(lut,'lookup'):
                try:
                    c=lut.lookup(message)
                    if c and isinstance(c,dict) and c.get('a'):raw_ans=c['a'];tier='tier1_lut';tokens=0
                except Exception:pass
        apply_cot=_needs_cot(category,message)
        cot_scaffold=_pick_cot(category,message) if apply_cot else ''
        cot_tag={'_COT_CODE':'code','_COT_MATH':'math','_COT_DEBUG':'debug','_COT_DESIGN':'design','_COT_REASONING':'reasoning'}.get(_pick_cot(category,message).split('\n')[0],'generic') if apply_cot else ''
        if apply_cot:
            first_line=cot_scaffold.split('\n')[0]
            cot_tag='code' if 'Code task' in first_line else ('math' if 'Math problem' in first_line else ('debug' if 'Debugging' in first_line else ('design' if 'System design' in first_line else ('reasoning' if 'Reasoning question' in first_line else 'generic'))))
        if not raw_ans and persona.name!='Adam' and self.use_persona and hasattr(self.adam,'chat_persona'):
            sys_p=persona.system_prompt(message)
            if recent:sys_p=f'{sys_p}\n\nRecent conversation context (do not repeat, just use for grounding):\n{recent}'
            if apply_cot:sys_p=f'{sys_p}\n\n{cot_scaffold}'
            cot_extra=700 if (apply_cot and cot_tag=='code') else (450 if apply_cot else 0)
            r=self.adam.chat_persona(message,system=sys_p,max_new_tokens=int(80+200*persona.length)+cot_extra,do_sample=True)
            raw_ans=r.get('answer') or '';tier=r.get('tier','tier_persona')+(f'_cot_{cot_tag}' if apply_cot else '');tokens=r.get('tokens',0)
            if apply_cot and cot_tag=='code' and raw_ans:
                blocks=_extract_python_blocks(raw_ans)
                if blocks:
                    bad=_validate_python(blocks)
                    if bad:
                        err_summary='\n'.join(f'  block {i}: {e}' for i,_,e in bad)
                        fix_prompt=f'Your previous code had {len(bad)} Python syntax error(s):\n{err_summary}\n\nOriginal question: {message}\n\nOutput ONLY a single corrected Python code block (```python ... ```). No explanation needed unless the fix changed approach.'
                        fix_sys=persona.system_prompt(message)+'\n\nFix Python syntax. Output a clean code block only.'
                        fix_r=self.adam.chat_persona(fix_prompt,system=fix_sys,max_new_tokens=350,do_sample=False)
                        fix_ans=fix_r.get('answer','').strip()
                        if fix_ans:
                            fix_blocks=_extract_python_blocks(fix_ans)
                            still_bad=_validate_python(fix_blocks) if fix_blocks else [(0,'','no code block returned')]
                            if not still_bad:
                                raw_ans=f'{raw_ans}\n\n**[Auto-corrected: {len(bad)} syntax error(s) found + fixed]**\n{fix_ans}'
                                tier+='_fix'
                                tokens+=fix_r.get('tokens',0)
                                blocks=_extract_python_blocks(raw_ans)
                if blocks and self.skills.has('run_python'):
                    runnable=[b for b in blocks if ('print(' in b or 'if __name__' in b)]
                    if runnable:
                        snippet=runnable[-1]
                        try:
                            run_r=self.skills.call('run_python',{'code':snippet,'timeout':8},ctx={'adam':self.adam})
                            if run_r.ok and not run_r.output.get('error'):
                                so=(run_r.output.get('stdout') or '').strip()
                                se=(run_r.output.get('stderr') or '').strip()
                                rc=run_r.output.get('returncode')
                                exec_block=f'\n\n**[Sandbox execution — exit {rc}{"  (timed out)" if run_r.output.get("timed_out") else ""}]**\n'
                                if so:exec_block+=f'```\n{so[:1500]}\n```\n'
                                if se:exec_block+=f'_stderr:_\n```\n{se[:600]}\n```'
                                raw_ans+=exec_block
                                tier+='_run'
                                skill_calls.append({'skill':'run_python','args':{'auto':True},'result':run_r.to_dict()})
                                test_failed=False;test_err=''
                                if rc==0 and not se:
                                    asserts=_extract_asserts(raw_ans)
                                    if asserts:
                                        div_score,div_info=_assert_diversity(asserts)
                                        passed,terr,tinfo=_run_with_tests(self.skills,self.adam,snippet,asserts)
                                        div_tag='_tests_thin' if div_score<0.5 else ('_tests_diverse' if div_score>=0.75 else '_tests_ok')
                                        if passed:
                                            raw_ans+=f'\n\n**[Self-tests — {len(asserts)}/{len(asserts)} passed · diversity={div_score:.2f}]**'
                                            tier+=div_tag
                                            skill_calls.append({'skill':'self_tests','args':{'n':len(asserts)},'result':{'passed':True}})
                                            try:
                                                tr=self.adam.teach(message,raw_ans[:2000])
                                                tier+='_promoted'
                                                skill_calls.append({'skill':'promote_lesson','args':{},'result':{'lessons_n':tr.get('lessons_n',0)}})
                                            except Exception:pass
                                        else:
                                            raw_ans+=f'\n\n**[Self-tests FAILED — {terr[:200]}]**'
                                            test_failed=True;test_err=terr
                                            skill_calls.append({'skill':'self_tests','args':{'n':len(asserts)},'result':{'passed':False,'err':terr[:200]}})
                                if rc!=0 or se or test_failed:
                                    err_signal=test_err if test_failed else (se or f'exit code {rc}')
                                    perturb_asserts=_extract_asserts(raw_ans) if test_failed else None
                                    pr=_perturb_retry(self.adam,self.skills,sys_p,snippet,err_signal,message,max_steps=3,asserts=perturb_asserts)
                                    if pr.get('success'):
                                        raw_ans+=f'\n\n**[Trial-and-error fixed it — {pr["magnitude"]} perturbation]**\n```python\n{pr["code"]}\n```\n```\n{pr["stdout"][:1500]}\n```'
                                        tier+=f'_perturb_{pr["magnitude"].lower()}'
                                        skill_calls.append({'skill':'perturb_retry','args':{'magnitude':pr['magnitude'],'success':True},'result':{'steps':len(pr.get('history',[]))}})
                                    else:
                                        raw_ans+=f'\n\n**[Trial-and-error exhausted — {len(pr.get("history",[]))} attempt(s) failed]**'
                                        tier+='_perturb_failed'
                                        skill_calls.append({'skill':'perturb_retry','args':{'success':False},'result':{'steps':len(pr.get('history',[]))}})
                            elif run_r.ok and run_r.output.get('error'):
                                raw_ans+=f'\n\n_(auto-run skipped: {run_r.output["error"][:120]})_'
                        except Exception as e:raw_ans+=f'\n\n_(auto-run failed: {e})_'
        if not raw_ans:
            framed=f'[Conversation history for context — answer the New Question only]\n{recent}\n\n[New Question]\n{message}' if needs_history and recent.strip() else message
            if apply_cot:framed=f'{cot_scaffold}\nQuery: {framed}'
            effective_writeback=writeback and not needs_history
            fb=self.adam.ask(framed,writeback=effective_writeback)
            raw_ans=fb.get('answer') or '';tier=fb.get('tier','?')+(f'_cot_{cot_tag}' if apply_cot else '');tokens=fb.get('tokens',0)
        wrapped=tone_atlas.wrap(raw_ans,category,persona,seed=message)
        if skill_answer and skill_answer.startswith('(skill'):wrapped=f'{skill_answer}\n{wrapped}'
        conv.append('assistant',wrapped,{'tier':tier,'tokens':tokens,'skill_calls':skill_calls,'persona':persona.name,'category':category})
        return {'answer':wrapped,'tier':tier,'tokens':tokens,'session_id':conv.session_id,'skill_calls':skill_calls,'wall_s':round(time.time()-t0,3),'persona':persona.name,'category':category}
    def _format_skill_output(self,name:str,out:Any)->str:
        if name=='time':return f'It is currently {out.get("iso")} local.'
        if name=='calc':return f'{out.get("value")}' if out.get('value') is not None else f'(calc error: {out.get("error")})'
        if name=='mem':
            hits=out.get('hits',[]);real=[h for h in hits if 'a' in h or 'answer' in h]
            if not real:return f'(no relevant lesson found in bank of {out.get("lessons_n",0)})'
            lines=[f'Top {len(real)} match(es) from {out.get("lessons_n",0)}-lesson bank:']
            for i,h in enumerate(real[:3]):
                a=h.get('a') or h.get('answer','')
                q=h.get('q','')
                s=h.get('score');s_str=f' [cos={s:.2f}]' if isinstance(s,(int,float)) else ''
                lines.append(f'\n{i+1}.{s_str} Q: {q[:120]}\n   A: {a[:300]}')
            return '\n'.join(lines)
        if name=='web':return f'{out.get("answer","(no answer)")}\n\nSources: {", ".join(out.get("sources",[])[:3])}'
        if name=='file_read':return f'```\n{out.get("content","")}\n```\n({out.get("bytes")} bytes from {out.get("path")})'
        if name=='file_write':return f'Wrote {out.get("bytes_written")} bytes to {out.get("path")}.'
        if name=='code_edit':return f'Edited {out.get("path")}: {out.get("replacements",0)} replacement(s).' if not out.get('error') else f'(code_edit error: {out.get("error")})'
        if name=='shell':return f'$ {out.get("cmd")}\n(exit={out.get("returncode")})\n{out.get("stdout","")}{out.get("stderr","")}'
        if name=='scan':
            if out.get('error'):return f'(scan error: {out["error"]})'
            return f'Scanned {out.get("files_scanned",0)} file(s), added {out.get("lessons_added",0)} lesson(s) (total: {out.get("lessons_total",0)}). Distilled: {out.get("distilled")}.'
        return json.dumps(out,default=str)[:1000]
    def _introspect_answer(self,persona:Optional[Persona]=None)->str:
        skills=[s['name'] for s in self.list_skills()]
        n=len(self.adam.sem_lut._raw) if hasattr(self.adam,'sem_lut') and self.adam.sem_lut is not None else 0
        pname=persona.name if persona else 'Adam'
        intro=f'I am {pname}' if persona and persona.name!='Adam' else 'I am Adam'
        intro+=' — a GF(17) texture-native AI built by Amnibro.'
        if persona and persona.name!='Adam':intro+=f' Right now I am wearing the {persona.name} persona — {persona.description}'
        return (f'{intro}\n\n'
                f'I have {len(skills)} skills I can use directly: {", ".join(skills)}.\n\n'
                f'I have {n} lessons in my persistent memory bank — anything you teach me carries across sessions, and I can ingest documents via the `scan` skill.\n\n'
                'I can: do math (instant via fast_eval, or via my mini-Qwen tier), query my memory, search the web, read/write files anywhere on this computer, edit code with AST validation, run allowlisted shell commands, and ingest entire directories into my lesson bank.\n\n'
                'You can change my persona any time — just ask, or POST /persona {"name":"yoda"}. Unknown personas trigger me to web-search and learn the voice.\n\n'
                'Try: "scan C:/path/to/notes", "what is 17 * 23", or teach me anything new.')
    def list_skills(self)->List[Dict[str,Any]]:return self.skills.list_skills()
    def stats(self)->Dict[str,Any]:
        s=self.adam.stats() if hasattr(self.adam,'stats') else {}
        s['skills']=[sk['name'] for sk in self.list_skills()]
        s['sessions_n']=len(self.store.list_sessions())
        return s
