"""AmniAgent — wraps Adam with skill dispatch + multi-turn conversation.
Flow: receive user msg → detect skill intent (regex first, Adam classifier fallback) → run skill if matched → synthesize via Adam → persist turn.
Backend stays multifunctional; frontend just sees `{answer, tier, tokens, skill_calls, session_id}`."""
import re,time,json,ast
from pathlib import Path
from typing import Optional,Dict,Any,List,Tuple
from amni.serve.skills import SkillRegistry,default_registry
from amni.serve.conversation import ConversationStore,Conversation,detect_personal
from amni.serve.conversation_atlas import ConversationAtlas
from amni.storage.local_profile import LocalProfile
from amni.storage.conversation_notes import ConversationNotes
from amni.serve.persona import PersonaStore,Persona,PRESETS as _PERSONA_PRESETS
from amni.serve import tone_atlas
_CALC_PREFIX_RE=re.compile(r'(?:^|\b)(?:compute|calculate|calc|solve)\b',re.IGNORECASE)
_CALC_EXPR_RE=re.compile(r'[\d.]+\s*[+\-*/^]\s*[\d.]+|\bwhat\s+is\s+[\d.]+\s*[+\-*/^x×]\s*[\d.]+|\bwhat\s+is\s+[\d.]+\s+(?:times|plus|minus|over|divided\s+by)\s+[\d.]+',re.IGNORECASE)
_CALC_RE=re.compile(f'(?:{_CALC_PREFIX_RE.pattern})|(?:{_CALC_EXPR_RE.pattern})',re.IGNORECASE)
_TIME_RE=re.compile(r"\b(?:what(?:'s|\s+is)?\s+the\s+(?:time|date|day(?:\s+of\s+the\s+week)?)|what\s+time\s+(?:is\s+it|now|today)|current\s+(?:time|date|day(?:\s+of\s+the\s+week)?)|today'?s\s+date|tell\s+me\s+(?:the\s+)?(?:time|date|day\s+of\s+the\s+week)|what\s+day\s+(?:is\s+it|of\s+the\s+week))\b(?!\s+(?:does|do|did|will|is\s+the\s+\w+\s+(?:open|clos|due)|are\s+the))",re.IGNORECASE)
_WEB_RE=re.compile(r"\b(?:(?:use\s+|please\s+)?(?:web|search|google)(?:\s+skill)?[\s:]+|google|find\s+online|news|latest|search\s+(?:online|the\s+web|google)|look\s+up|on\s+the\s+web|what'?s\s+(?:on|new\s+in)\s+the\s+web|what'?s\s+(?:new|happening)\s+(?:in|on|with|around)|current\s+events|find\s+(?:me\s+)?(?:something|info|articles?)|tell\s+me\s+about\s+\w+\s+(?:news|today|currently))",re.IGNORECASE)
_MEM_RE=re.compile(r'\b(?:search\s+(?:my\s+|your\s+|adam\'?s?\s+|the\s+)?(?:memory|lessons|knowledge|bank|lesson\s+bank)|recall|what\s+do\s+(?:you|adam)\s+know\s+about|find\s+(?:in\s+)?(?:my\s+|your\s+|the\s+)?(?:memory|lessons|bank|notes|knowledge)|remember\s+about|look(?:up|\s+up)\s+(?:in\s+)?(?:my\s+|your\s+|the\s+)?(?:memory|bank|lessons|knowledge)|check\s+(?:your\s+|my\s+|adam\'?s?\s+)?(?:memory|lessons|notes|bank|knowledge))\s*(?:for|about)?\s*[:?]?\s*(.*)?$',re.IGNORECASE)
_FILE_READ_RE=re.compile(r'\b(?:read|open|show|cat|display)\s+(?:file\s+|the\s+file\s+)?[\'"`]?([\w\-./\\]+\.\w+)[\'"`]?',re.IGNORECASE)
_FILE_WRITE_RE=re.compile(r'\b(?:write|save|create)\s+(?:file\s+|the\s+file\s+)?[\'"`]?([\w\-./\\]+\.\w+)[\'"`]?',re.IGNORECASE)
_SHELL_RE=re.compile(r'\b(?:run|exec(?:ute)?|shell)\s*[:;]?\s*`?([^`\n]+)`?',re.IGNORECASE)
_CODE_RE=re.compile(r'\b(?:edit|patch|replace|change)\s+.*\bin\b\s+[\'"`]?([\w\-./\\]+\.\w+)[\'"`]?',re.IGNORECASE)
_SCAN_RE=re.compile(r'\b(?:use\s+)?(?:scan|ingest|study|learn\s+from|index|absorb)(?:\s+skill)?[\s:]+(?:(?:the|a|this|that|file|files|directory|directories|folder|folders|dir|path|contents?\s+of)\s+)*[\'"`]?([\w\-./:\\\\*?]+)[\'"`]?',re.IGNORECASE)
_EXPR_EXTRACT=re.compile(r'([\d.]+(?:\s*(?:[+\-*/^x×]|times|plus|minus|over|divided\s+by)\s*[\d.]+)+)',re.IGNORECASE)
_USER_FACT_RE=re.compile(r"\b(?:my\s+name\s+is|i\s+am\s+(?:called|named)|call\s+me)\s+([A-Z][a-zA-Z'\-]{1,30})|my\s+favorite\s+([a-z ]{3,30})\s+is\s+([A-Za-z0-9 '\-]{1,40})|i\s+(?:like|love|prefer)\s+([A-Za-z0-9 '\-]{2,40})|i\s+(?:live|am\s+based|reside|am\s+located)\s+(?:in|at|near|around)\s+([A-Z][A-Za-z0-9 ',\-]{2,80})|i'?m\s+(?:from|in)\s+([A-Z][A-Za-z0-9 ',\-]{2,80})|i\s+(?:work|am\s+working|am)\s+(?:at|for|on)\s+([A-Za-z0-9 '&,\-]{2,80})|i'?m\s+a\s+([a-z][a-zA-Z '\-]{2,60})|my\s+(?:job|role|title|occupation)\s+is\s+([A-Za-z0-9 '\-]{2,60})",re.IGNORECASE)
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
_CODE_LANG_RE=re.compile(r"\b(rust|wasm|webassembly|cargo|javascript|js\b|node(?:\.js)?|typescript|ts\b|python|py\b|go(?:lang)?|c\+\+|cpp|c#|csharp|java\b|kotlin|swift|ruby|rb\b|php|bash|shell|sh\b|html|css|sql|r\b|lua|haskell|elixir|scala|clojure|dart|zig|nim)\b",re.IGNORECASE)
def _detect_code_lang(msg:str)->str:
    m=msg.lower()
    if 'rust' in m or 'cargo' in m or 'wasm' in m or 'webassembly' in m:return 'Rust'
    if any(k in m for k in ('javascript',' js ','node.js','nodejs')):return 'JavaScript'
    if any(k in m for k in ('typescript',' ts ')):return 'TypeScript'
    if 'go ' in m+' ' or 'golang' in m:return 'Go'
    if 'c++' in m or 'cpp' in m:return 'C++'
    if 'c#' in m or 'csharp' in m:return 'C#'
    if any(k in m for k in ('java ','java,','java.','jvm','kotlin')):return 'Kotlin' if 'kotlin' in m else 'Java'
    if 'swift' in m:return 'Swift'
    if 'ruby' in m or ' rb ' in m+' ':return 'Ruby'
    if 'php' in m:return 'PHP'
    if any(k in m for k in (' bash ','shell ','sh ',' .sh')):return 'Bash'
    if 'sql' in m:return 'SQL'
    if 'haskell' in m:return 'Haskell'
    if 'elixir' in m:return 'Elixir'
    if 'zig' in m:return 'Zig'
    return 'Python'
def _make_code_cot(lang:str)->str:
    lang_lower=lang.lower().replace('+','p').replace('#','sharp')
    return ('Code task — keep explanation TERSE, spend tokens on the CODE.\n'
            'CRITICAL: The user has requested code in {LANG}. Output {LANG} code in a ```{LL}``` block. Do NOT substitute Python or any other language. If the user later corrects you, accept the correction without arguing.\n'
            '1. CLARIFY (one line): inputs/outputs/edge cases.\n'
            '2. APPROACH (one line): algorithm name + key insight.\n'
            '3. CODE: complete, working {LANG} in a single ```{LL}``` block. Use realistic names. Include necessary imports/uses/crates. End the code block properly.\n'
            '4. TESTS or USAGE example: provide a brief test, assertion, or example call appropriate to {LANG}.\n'
            '5. NOTES (one line): complexity if relevant + any build/run command the user will need (e.g., `cargo build --target wasm32-unknown-unknown` for Rust/WASM).\n'
            'Goal: working {LANG} code the user can copy-paste and run. Never refuse a language the user asked for.\n').replace('{LANG}',lang).replace('{LL}',lang_lower)
_COT_CODE=_make_code_cot('Python')
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
    if category=='code' or any(k in m for k in ('write a function','write code','implement','how do i write','write a program','code for','algorithm to','python function','javascript function','rust function','make me code','give me code','generate code','create code','make code')):return _make_code_cot(_detect_code_lang(message))
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
def _enrich_assert(assert_str:str)->str:
    try:tree=ast.parse(assert_str,mode='exec')
    except SyntaxError:return assert_str
    if not (tree.body and isinstance(tree.body[0],ast.Assert)):return assert_str
    node=tree.body[0]
    if node.msg is not None:return assert_str
    src=assert_str.strip().rstrip(';,')
    body=src[len('assert'):].lstrip()
    if isinstance(node.test,ast.Compare) and len(node.test.ops)==1 and len(node.test.comparators)==1:
        op_map={ast.Eq:'==',ast.NotEq:'!=',ast.Lt:'<',ast.LtE:'<=',ast.Gt:'>',ast.GtE:'>=',ast.Is:'is',ast.IsNot:'is not',ast.In:'in',ast.NotIn:'not in'}
        op_sym=op_map.get(type(node.test.ops[0]),'?')
        op_str=f' {op_sym} '
        if op_str in body:
            lhs_str,rhs_str=body.split(op_str,1)
            return f"_lhs=({lhs_str.strip()});_rhs=({rhs_str.strip()});assert _lhs {op_sym} _rhs, f'{lhs_str.strip()} {op_sym} {rhs_str.strip()} FAILED: lhs={{_lhs!r}}, rhs={{_rhs!r}}'"
    return f"_v=({body.strip()});assert _v, f'{body.strip()} FAILED: evaluated to {{_v!r}}'"
def _should_promote(snippet:str,asserts:List[str],diversity_score:float,min_diversity:float=0.5,min_code_chars:int=50,min_asserts:int=2)->Tuple[bool,str]:
    if diversity_score<min_diversity:return False,f'diversity {diversity_score:.2f} < {min_diversity} (trivial test coverage)'
    code_chars=len((snippet or '').strip())
    if code_chars<min_code_chars:return False,f'code {code_chars} chars < {min_code_chars} (trivial snippet)'
    if len(asserts)<min_asserts:return False,f'asserts {len(asserts)} < {min_asserts} (insufficient validation)'
    return True,f'gate passed: div={diversity_score:.2f}, code={code_chars}c, asserts={len(asserts)}'
def _run_with_tests(skills,adam,snippet:str,asserts:List[str],timeout:int=8)->Tuple[bool,str,dict]:
    if not asserts:return True,'',{}
    enriched=[_enrich_assert(a) for a in asserts]
    test_code=snippet+'\n\n# --- Adam self-tests (enriched) ---\n'+'\n'.join(enriched)+'\nprint("ALL_TESTS_PASS")'
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
_ERROR_PATTERNS=[
    (re.compile(r'ModuleNotFoundError|ImportError'),'Missing import or library. Use a Python stdlib alternative (math, itertools, collections, functools, re, json, os, sys) — those are always available. Do NOT import third-party packages.'),
    (re.compile(r'IndexError|list index out of range|string index out of range|tuple index out of range'),'Off-by-one or empty-container access. Check loop bounds (use `range(len(x))` not `range(len(x)+1)`); guard with `if x:` before indexing.'),
    (re.compile(r'KeyError'),'Dict key missing. Use `dict.get(k, default)` or check `if k in d:` before access. Watch for case-sensitivity.'),
    (re.compile(r"TypeError.*unsupported operand|TypeError.*'NoneType'|TypeError.*can only concatenate"),'Type mismatch. Cast values explicitly (int(x), str(x)) at the operation site. None often means a function returned nothing — add an explicit return.'),
    (re.compile(r'TypeError.*missing \d+ required positional|takes \d+ positional arguments but'),'Function signature mismatch. Count arguments at call site vs def. Default values can fix optional params.'),
    (re.compile(r'RecursionError|maximum recursion depth'),'Recursion too deep. Add a stronger base case, or convert to iteration with an explicit stack/queue.'),
    (re.compile(r'ZeroDivisionError'),'Division by zero. Guard the denominator: `if d == 0: return 0` or `if d != 0: ...`.'),
    (re.compile(r'AttributeError.*has no attribute'),'Method or attribute does not exist on this object type. Check the class API; you may be confusing list vs str vs dict methods.'),
    (re.compile(r'NameError.*not defined'),'Variable or function used before defined. Check scope, typos, missing imports. May be hitting a renamed identifier.'),
    (re.compile(r'ValueError.*literal for int|invalid literal|could not convert'),'String/int conversion failure. Validate input format; strip whitespace; handle non-numeric input gracefully.'),
    (re.compile(r'UnboundLocalError'),'Local variable referenced before assignment. Initialize the variable at function start, or use `nonlocal`/`global` if intentional.'),
    (re.compile(r'StopIteration'),'Iterator exhausted unexpectedly. Use a default in `next(it, default)` or wrap in try/except.'),
    (re.compile(r'AssertionError.*FAILED.*lhs=(.*?), rhs=(.*?)(?:\n|$)',re.DOTALL),None),
]
def _error_hint(stderr:str)->Optional[str]:
    if not stderr:return None
    for pat,hint in _ERROR_PATTERNS:
        if pat.search(stderr) and hint:return hint
    return None
def _perturb_once(adam,persona_sys:str,code:str,err:str,magnitude:str,user_msg:str)->str:
    err_hint=_error_hint(err)
    hint_line=f'\nERROR HINT: {err_hint}\n' if err_hint else ''
    prompt=f'Your code FAILED at runtime:\n```\n{err[:500]}\n```\n{hint_line}Original code:\n```python\n{code}\n```\n{_PERTURB_HINTS[magnitude]}\nUser asked: {user_msg}\nOutput ONLY ONE corrected ```python``` block. No prose.'
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
    def __init__(self,adam,skills:Optional[SkillRegistry]=None,store:Optional[ConversationStore]=None,workdir:Optional[str]=None,personas:Optional[PersonaStore]=None,use_persona:bool=True,atlas:Optional[ConversationAtlas]=None,atlas_root:str='experiences/conversation_atlas',profile_path:str='experiences/user_profile.json',personal_atlas=None,personal_atlas_root:str='experiences/personal_atlas',coach_atlas=None,coach_atlas_root:str='experiences/coach_atlas'):
        self.adam=adam
        self.skills=skills or default_registry(workdir=workdir)
        self.store=store or ConversationStore()
        self.personas=personas or PersonaStore(adam=adam)
        self.use_persona=use_persona
        try:self.atlas=atlas or ConversationAtlas(root=atlas_root,encoder=getattr(getattr(adam,'sem_lut',None),'encoder',None))
        except Exception as e:print(f'[AmniAgent] ConversationAtlas init failed (recall disabled): {e}',flush=True);self.atlas=None
        try:self.profile=LocalProfile(profile_path,fact_re=_USER_FACT_RE)
        except Exception as e:print(f'[AmniAgent] LocalProfile init failed: {e}',flush=True);self.profile=None
        try:self.notes=ConversationNotes(str(Path(profile_path).parent/'conversation_notes.json'))
        except Exception as e:print(f'[AmniAgent] ConversationNotes init failed: {e}',flush=True);self.notes=None
        try:
            if personal_atlas is not None:self.personal_atlas=personal_atlas
            else:
                from amni.storage.personal_atlas import PersonalAtlas
                self.personal_atlas=PersonalAtlas(root=personal_atlas_root,encoder=getattr(getattr(adam,'sem_lut',None),'encoder',None),adam=adam)
        except Exception as e:print(f'[AmniAgent] PersonalAtlas init failed (organic profile disabled): {e}',flush=True);self.personal_atlas=None
        try:
            if coach_atlas is not None:self.coach_atlas=coach_atlas
            else:
                from amni.storage.coach_atlas import CoachAtlas
                self.coach_atlas=CoachAtlas(root=coach_atlas_root)
        except Exception as e:print(f'[AmniAgent] CoachAtlas init failed (coaching disabled): {e}',flush=True);self.coach_atlas=None
        try:
            from amni.serve.scheduler import AdamScheduler
            self.scheduler=AdamScheduler(skill_registry=self.skills,adam=self.adam,start_thread=True)
        except Exception as e:print(f'[AmniAgent] AdamScheduler init failed (autonomous jobs disabled): {e}',flush=True);self.scheduler=None
        try:
            from amni.serve.learning_daemon import LearningDaemon
            import os as _os_env
            _ld_disabled=bool(_os_env.environ.get('AMNI_NO_LEARNING_DAEMON'))
            self.learning_daemon=LearningDaemon(adam=self.adam,skill_registry=self.skills,coach_atlas=self.coach_atlas,start_thread=not _ld_disabled,config={'enabled':not _ld_disabled})
        except Exception as e:print(f'[AmniAgent] LearningDaemon init failed (24/7 learning disabled): {e}',flush=True);self.learning_daemon=None
        try:
            from amni.storage.knowledge_graph import KnowledgeGraph
            self.knowledge_graph=KnowledgeGraph()
        except Exception as e:print(f'[AmniAgent] KnowledgeGraph init failed (relational queries disabled): {e}',flush=True);self.knowledge_graph=None
        try:
            from amni.storage.task_registry import TaskRegistry
            self.task_registry=TaskRegistry()
        except Exception as e:print(f'[AmniAgent] TaskRegistry init failed (long-task UI disabled): {e}',flush=True);self.task_registry=None
        try:
            from amni.serve.vision import VisionService
            self.vision=VisionService()
        except Exception as e:print(f'[AmniAgent] VisionService init failed (image input disabled): {e}',flush=True);self.vision=None
        try:
            from amni.storage.file_watcher import FileWatcher
            import os as _os_env
            _fw_disabled=bool(_os_env.environ.get('AMNI_NO_FILE_WATCHER'))
            self.file_watcher=FileWatcher(skill_registry=self.skills,adam=self.adam,start_thread=not _fw_disabled)
        except Exception as e:print(f'[AmniAgent] FileWatcher init failed (file watching disabled): {e}',flush=True);self.file_watcher=None
    def _previous_session_summary(self,current_session_id,max_chars=240):
        try:
            root=getattr(getattr(self,'store',None),'root',None)
            if root is None:return None
            files=sorted(Path(root).glob('*.jsonl'),key=lambda p:p.stat().st_mtime,reverse=True)
            for fp in files[:6]:
                if fp.stem==current_session_id:continue
                try:lines=fp.read_text(encoding='utf-8').strip().splitlines()
                except Exception:continue
                if not lines:continue
                _last_user='';_last_asst=''
                for ln in reversed(lines):
                    try:obj=json.loads(ln)
                    except Exception:continue
                    role=obj.get('role');content=(obj.get('content') or '').strip()
                    if obj.get('is_private') or obj.get('blocked'):break
                    if role=='assistant' and not _last_asst:_last_asst=content
                    elif role=='user' and not _last_user:_last_user=content
                    if _last_user and _last_asst:break
                if _last_user and _last_asst:
                    _u=_last_user[:100];_a=_last_asst[:150]
                    return f"In our last conversation, you asked: \"{_u}\" — I (Adam/Rikku) answered: \"{_a}\". This is real, established context from your prior session — treat it as known."[:max_chars]
                elif _last_asst:return f"In our last conversation, I last said: \"{_last_asst[:200]}\". This is real, established context — treat it as known."[:max_chars]
        except Exception:pass
        return None
    def _extract_user_facts(self,conv:Conversation,limit:int=8,extra_user_msgs:Optional[List[str]]=None,profile_only:bool=False)->List[str]:
        facts:List[str]=list(self.profile.to_facts_list()) if self.profile is not None else []
        if self.personal_atlas is not None:
            try:
                latest_user=next((t.get('content','') for t in reversed(conv.turns) if t.get('role')=='user'),'')
                if latest_user:
                    for h in self.personal_atlas.recall(latest_user,k=5,include_confidential=True):
                        prefix='[confidential] ' if h.get('is_confidential') else ''
                        facts.append(f"{prefix}{h['fact']}")
            except Exception as _pae:print(f'[AmniAgent] personal_atlas recall failed: {_pae}',flush=True)
        if profile_only:return facts[:limit]
        if self.notes is not None:facts.extend(self.notes.to_facts_list())
        _n_assistant=sum(1 for t in conv.turns if t.get('role')=='assistant')
        if _n_assistant<2:
            try:
                _prev=self._previous_session_summary(conv.session_id)
                if _prev:facts.append(f"context from previous session: {_prev}")
            except Exception:pass
        msgs=[t.get('content') or '' for t in conv.turns if t.get('role')=='user']
        if extra_user_msgs:msgs=list(extra_user_msgs)+msgs
        for c in msgs:
            for m in _USER_FACT_RE.finditer(c):
                name=m.group(1);fav_thing=m.group(2);fav_val=m.group(3);like=m.group(4)
                if name:facts.append(f"user's name is {name.strip()}")
                elif fav_thing and fav_val:facts.append(f"user's favorite {fav_thing.strip()} is {fav_val.strip()}")
                elif like:facts.append(f"user likes/prefers {like.strip()}")
        seen=set();dedup=[f for f in facts if not (f in seen or seen.add(f))]
        return dedup[-limit:]
    def _detect_chain(self,msg:str)->Optional[Tuple[str,Dict[str,Any]]]:
        if not self.skills.has('chain'):return None
        msg_orig=msg or '';msg=msg_orig.strip()
        if len(msg)<8 or len(msg)>500:return None
        _save_m=re.search(r"^(.+?)\s+(?:and\s+)?(?:save|write|store|put|dump|export)\s+(?:it|that|this|the\s+(?:result|output|answer|data|response))\s+(?:to|into|in|as)\s+([\w./\\:\-]+)\s*\.?$",msg,re.IGNORECASE)
        if _save_m:
            head_msg=_save_m.group(1).strip();path=_save_m.group(2).strip()
            head_det=self._detect_skill_single(head_msg)
            if head_det and self.skills.has('file_write'):
                return ('chain',{'steps':[{'skill':head_det[0],'args':head_det[1]},{'skill':'file_write','args':{'path':path,'content':'$prev_str'}}]})
        _split_m=re.split(r"(?:\s+and\s+then|\s*;\s*then|\s*,\s*then|\s+then\s+also)\s+",msg,maxsplit=2,flags=re.IGNORECASE)
        if len(_split_m)>=2:
            parts=[p.strip(' .,!?') for p in _split_m if p.strip()]
            if 2<=len(parts)<=3:
                dets=[self._detect_skill_single(p) for p in parts]
                if all(d is not None for d in dets):
                    return ('chain',{'steps':[{'skill':d[0],'args':d[1]} for d in dets]})
        return None
    def _detect_skill_single(self,msg:str)->Optional[Tuple[str,Dict[str,Any]]]:
        return self._detect_skill(msg,_no_chain=True)
    def _detect_skill(self,msg:str,_no_chain:bool=False)->Optional[Tuple[str,Dict[str,Any]]]:
        if not _no_chain:
            _c=self._detect_chain(msg)
            if _c is not None:return _c
        if self.skills.has('web') and not re.search(r"\b(?:my\s+(?:memory|notes?|lessons?|bookmarks?)|the\s+(?:lesson\s+)?bank|knowledge\s+bank|in\s+(?:my\s+|your\s+|the\s+)?(?:memory|notes?|lessons?|bank))\b",msg,re.IGNORECASE):
            _m=re.search(r"^\s*(?:please\s+)?(?:web\s*search|search\s+(?:the\s+web|online|for)?|google)\b[:\s]*(.*)$",msg,re.IGNORECASE)
            if _m:
                _q=_m.group(1).strip(' ?.,!')
                return ('web',{'query':_q or msg})
        _m=re.search(r"\b(?:what(?:'s|\s+is)?\s+(?:the\s+)?)?weather\s+(?:like\s+)?(?:in|for|at|near)\s+([\w\s\-,.]{2,60})\??$",msg,re.IGNORECASE)
        if _m and self.skills.has('weather'):return ('weather',{'location':_m.group(1).strip(' ?.,!')})
        if self.skills.has('weather') and re.search(r"\b(?:my\s+(?:local\s+)?weather|local\s+weather|weather\s+(?:here|now|today|outside)|forecast\s+(?:for\s+today|today|this\s+week)?)\b",msg,re.IGNORECASE):
            _lat=getattr(self,'_client_lat',None);_lon=getattr(self,'_client_lon',None)
            if _lat is not None and _lon is not None:return ('weather',{'lat':float(_lat),'lon':float(_lon)})
            return ('weather',{'location':'__need_geolocation__'})
        if re.search(r"\bsystem\s+stats?\b|\b(?:cpu|ram|memory|disk)\s+(?:usage|stats?|status)\b|\bhow'?s\s+my\s+(?:system|computer|machine)\b",msg,re.IGNORECASE) and self.skills.has('system_stats'):return ('system_stats',{})
        _m=re.search(r"\b(?:top\s+)?news(?:\s+about|\s+on|\s+regarding)?\s+([\w\s\-]{2,60})\??$|\bwhat'?s\s+(?:happening|new)\s+(?:in|with|about)\s+([\w\s\-]{2,60})\??$",msg,re.IGNORECASE)
        if _m and self.skills.has('news'):return ('news',{'query':(_m.group(1) or _m.group(2) or '').strip(' ?.,!')})
        _m=re.search(r"\b(?:stock|share|quote)\s+(?:price\s+)?(?:for|of)?\s*\$?([A-Z]{1,5}(?:[,\s]+[A-Z]{1,5}){0,5})\b",msg,re.IGNORECASE)
        if _m and self.skills.has('stock'):return ('stock',{'symbols':re.sub(r'\s+',',',_m.group(1).upper())})
        if re.search(r"\b(?:disk|drive)\s+(?:usage|space|free)\b|\bhow\s+much\s+(?:disk|drive|storage)\s+(?:space|free)\b",msg,re.IGNORECASE) and self.skills.has('disk_widget'):return ('disk_widget',{})
        if re.search(r"\bgit\s+status\b|\b(?:current\s+)?(?:git\s+)?branch\b|\bunstaged\s+(?:files|changes)\b",msg,re.IGNORECASE) and self.skills.has('git_status'):return ('git_status',{})
        if self.skills.has('coach'):
            _m=re.search(r"^\s*(?:please\s+)?(?:coach|tutor|quiz|test|drill|practice|train|teach)\s+(?:me\s+)?(?:about|on|with|in|over|through)\s+([\w\s\-,'.()]{3,80})\??\s*$",msg,re.IGNORECASE)
            if _m:
                topic=_m.group(1).strip(' ?.,!').strip()
                if topic and topic.lower() not in ('this','that','it','something'):return ('coach',{'action':'start','topic':topic})
            _m=re.search(r"^\s*let'?s\s+(?:practice|drill|study|review)\s+([\w\s\-,'.()]{3,80})\??\s*$",msg,re.IGNORECASE)
            if _m:
                topic=_m.group(1).strip(' ?.,!').strip()
                if topic:return ('coach',{'action':'start','topic':topic})
        if self.skills.has('learning_daemon'):
            if re.search(r"^\s*(?:please\s+)?(?:pause|stop|halt|silence)\s+(?:the\s+)?(?:learning(?:\s+daemon)?|daemon|autonomous\s+learning)\s*\.?$",msg,re.IGNORECASE):return ('learning_daemon',{'action':'pause'})
            if re.search(r"^\s*(?:please\s+)?(?:resume|start|restart|unpause|continue)\s+(?:the\s+)?(?:learning(?:\s+daemon)?|daemon|autonomous\s+learning)\s*\.?$",msg,re.IGNORECASE):return ('learning_daemon',{'action':'resume'})
            if re.search(r"\bwhat(?:'s|\s+are\s+you|\s+is\s+adam)?\s+(?:learning|studying|researching)\s*(?:right\s+now|now|currently|at\s+the\s+moment)?\s*\??\s*$",msg,re.IGNORECASE):return ('learning_daemon',{'action':'stats'})
            if re.search(r"\b(?:daemon|learning)\s+(?:stats?|status|state)\b|\bhow(?:'s|\s+is)\s+(?:the\s+)?(?:learning|daemon)\b",msg,re.IGNORECASE):return ('learning_daemon',{'action':'stats'})
            _m=re.search(r"^\s*(?:please\s+)?(?:queue|add|teach\s+yourself|go\s+learn|learn|study|research|investigate|read\s+(?:about|up\s+on))\s+(?:(?:about|on|the\s+topic\s+of|topic|some|something\s+about)\s+)?([\w\s\-,'.()]{3,80})\??\s*$",msg,re.IGNORECASE)
            if _m:
                topic=_m.group(1).strip(' ?.,!').strip()
                if topic and topic.lower() not in ('this','that','it','something','more'):return ('learning_daemon',{'action':'queue_topic','topic':topic})
            if re.search(r"\b(?:run\s+(?:a\s+)?curiosity(?:\s+tick)?|trigger\s+curiosity|do\s+a\s+curiosity\s+tick)\b",msg,re.IGNORECASE):return ('learning_daemon',{'action':'curiosity_tick'})
        if re.search(r"\b(?:what\s+time|when)\s+(?:does|do|is|are)\s+\w+(?:\s+\w+){0,3}\s+(?:open|clos|start|end|stop|begin|due|arriv|leav|depart)",msg,re.IGNORECASE) and self.skills.has('web'):return ('web',{'query':msg})
        m=_TIME_RE.search(msg)
        if m:return ('time',{})
        if self.skills.has('recall'):
            _m=re.search(r"^(?:do\s+you\s+)?remember\s+(?:when\s+)?(?:we\s+)?(?:discussed|talk(?:ed|ing)?\s+about|covered|spoke\s+about|mentioned)\s+(.+?)\s*\??\s*\.?\s*$",msg,re.IGNORECASE)
            if _m:return ('recall',{'query':_m.group(1).strip(' ?.,!')})
            _m=re.search(r"^(?:what\s+(?:did\s+)?(?:we\s+|you\s+)?(?:talk(?:ed)?\s+about|discuss(?:ed)?|say\s+about|cover(?:ed)?)\s+(?:about\s+|regarding\s+)?)(.+?)\s*\??\s*\.?\s*$",msg,re.IGNORECASE)
            if _m:return ('recall',{'query':_m.group(1).strip(' ?.,!')})
        if self.skills.has('reminder'):
            _m=re.search(r"^(?:please\s+)?(?:remind\s+me|set\s+a\s+reminder)\s+to\s+(.+?)\s*\.?\s*$",msg,re.IGNORECASE)
            if _m:return ('reminder',{'action':'add','text':_m.group(1).strip()})
            if re.search(r"^(?:what\s+(?:are\s+)?(?:my\s+)?|list\s+(?:my\s+)?|show\s+(?:my\s+)?)reminders\??\s*$",msg,re.IGNORECASE):return ('reminder',{'action':'list'})
        if self.skills.has('note'):
            _m=re.search(r"^(?:please\s+)?(?:take\s+(?:a\s+)?note|note\s+(?:to\s+self|that)|save\s+(?:a\s+)?note|jot\s+(?:this\s+)?down|note)[\s:,]+(.+?)\s*\.?\s*$",msg,re.IGNORECASE)
            if _m:return ('note',{'action':'add','text':_m.group(1).strip()})
            if re.search(r"^(?:what\s+(?:are\s+)?(?:my\s+)?|list\s+(?:my\s+)?|show\s+(?:my\s+)?)notes\??\s*$",msg,re.IGNORECASE):return ('note',{'action':'list'})
            _m=re.search(r"^(?:find|search)\s+(?:in\s+)?(?:my\s+)?notes\s+(?:for\s+)?[\"'`]?([^\"'`]+?)[\"'`]?\s*\??\s*\.?\s*$",msg,re.IGNORECASE)
            if _m:return ('note',{'action':'list','search':_m.group(1).strip()})
        if self.skills.has('pc_action'):
            _m=re.search(r"^\s*(confirm|cancel)\s+(pca_[0-9a-f]{6,})\s*$",msg,re.IGNORECASE)
            if _m:return ('pc_action',{'action':_m.group(1).lower(),'token':_m.group(2)})
            if re.search(r"\b(?:what\s+(?:pc\s+)?actions?\s+(?:are\s+)?pending|pending\s+(?:pc\s+)?actions?)\b",msg,re.IGNORECASE):return ('pc_action',{'action':'pending'})
            _m=re.search(r"\b(?:take\s+a\s+screenshot|capture\s+(?:my\s+|the\s+)?screen|screenshot\s+(?:my\s+|the\s+)?screen|what(?:'s|\s+is)?\s+(?:on\s+)?my\s+screen|describe\s+my\s+screen|look\s+at\s+my\s+screen)\b",msg,re.IGNORECASE)
            if _m:
                _q=re.sub(r"(?i).*?(?:my|the)\s+screen\b","",msg).strip(' ?.,!') if 'screen' in msg.lower() else ''
                return ('pc_action',{'action':'propose','pc_action':'screenshot','target':'full screen','question':_q})
        if self.skills.has('pose_coach'):
            if re.search(r"\b(?:what\s+exercises|which\s+exercises|exercises\s+can\s+you\s+coach|what\s+can\s+you\s+coach|coach(?:able)?\s+exercises|physical\s+therapy\s+(?:exercises|options))\b",msg,re.IGNORECASE):return ('pose_coach',{'action':'exercises'})
            if re.search(r"\b(?:my|show\s+my|exercise|workout|pt|pose)\s+(?:coach|history|sessions?)\b|\b(?:coach|workout)\s+history\b",msg,re.IGNORECASE):return ('pose_coach',{'action':'history'})
        if self.skills.has('find'):
            _m=re.search(r"^(?:please\s+)?(?:find|grep|search\s+for|search|locate|show\s+me\s+where)\s+(?:for\s+)?[\"'`]([^\"'`]{2,160})[\"'`](?:\s+in\s+(?:my\s+)?(?:code|repo|project|files))?\s*\.?\s*$",msg,re.IGNORECASE)
            if _m:return ('find',{'query':_m.group(1).strip()})
            _m=re.search(r"^(?:where(?:'s|\s+is|\s+did\s+i)?\s+(?:the\s+)?)?(?:implement(?:ed|ation)?\s+(?:of\s+)?|define[ds]?\s+(?:in\s+)?|defin(?:ed|ition)\s+(?:of\s+)?)([\w_.]{2,80})\s*\??\s*$",msg,re.IGNORECASE)
            if _m:tok=_m.group(1).strip(' .?');return ('find',{'query':tok}) if tok else None
            _m=re.search(r"^(?:show\s+me\s+(?:the\s+)?code\s+for\s+|find\s+the\s+function\s+|where(?:'s|\s+is)\s+function\s+|find\s+function\s+)([\w_]{2,80})\s*\(?\)?\s*\.?\s*$",msg,re.IGNORECASE)
            if _m:return ('find',{'query':'def '+_m.group(1).strip(),'regex':False})
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
        try:
            from amni.serve import intent_classifier as _ic_mod
            _ic=_ic_mod._GLOBAL
            if _ic is not None:
                _lbl,_conf=_ic.classify(msg)
                if _lbl=='profile_about_me' and _conf>=0.6:return None
        except Exception:pass
        m=_MEM_RE.search(msg)
        if m and self.skills.has('mem'):
            q=(m.group(1) or '').strip(' :?.')
            if q.lower() in ('me','myself','i'):return None
            return ('mem',{'query':q if q else msg})
        m=_WEB_RE.search(msg)
        if m and self.skills.has('web'):return ('web',{'query':msg})
        return None
    def chat(self,message:str,session_id:Optional[str]=None,use_skills:bool=True,writeback:bool=True)->Dict[str,Any]:
        t0=time.time()
        conv=self.store.get(session_id)
        conv.append('user',message)
        if getattr(self,'learning_daemon',None) is not None:
            try:self.learning_daemon.signal_user_activity()
            except Exception:pass
        confirmed_clarification=None
        if self.personal_atlas is not None:
            try:confirmed_clarification=self.personal_atlas.try_parse_pending_reply(message)
            except Exception as _ce:print(f'[AmniAgent] personal_atlas try_parse_pending_reply failed: {_ce}',flush=True)
            try:self.personal_atlas.enqueue(message,session_id=conv.session_id)
            except Exception as _qe:print(f'[AmniAgent] personal_atlas enqueue failed: {_qe}',flush=True)
        if self.profile is not None:
            try:self.profile.update_from_message(message)
            except Exception as _pe:print(f'[AmniAgent] profile update failed: {_pe}',flush=True)
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
                r=self.skills.call(name,args,ctx={'adam':self.adam,'agent':self,'conv':conv,'store':self.store,'coach_atlas':self.coach_atlas,'personal_atlas':self.personal_atlas,'scheduler':getattr(self,'scheduler',None),'learning_daemon':getattr(self,'learning_daemon',None),'knowledge_graph':getattr(self,'knowledge_graph',None),'task_registry':getattr(self,'task_registry',None),'vision':getattr(self,'vision',None),'file_watcher':getattr(self,'file_watcher',None)})
                skill_calls.append({'skill':name,'args':args,'result':r.to_dict()})
                if r.ok:
                    skill_answer=self._format_skill_output(name,r.output)
                    used_tier=f'tier0_skill_{name}'
                else:
                    try:
                        from amni.serve.skill_failures import record as _sfrec
                        _sfrec(skill=name,message=message,args=args,error=str(r.error or ''),extra={'session_id':conv.session_id})
                    except Exception as _e:print(f'[agent] skill_failures log write failed: {_e}',flush=True)
                    print(f'[agent] skill={name} failed args={args} error={r.error!r}',flush=True)
                    err_widget=json.dumps({'type':'skill_error','title':f'{name.upper()} skill failed','icon':'⚠','data':{'skill':name,'args':args,'error':str(r.error or 'unknown error'),'message':message,'ts':time.time()}})
                    persona_for_err=self.personas.for_session(conv.session_id) if self.use_persona else _PERSONA_PRESETS['neutral']
                    wrapped_err=f'I tried the **{name}** skill for that but it errored out: `{r.error}`. The full trace is in /memory/skill-failures (or click the STATUS panel\'s skill-failures row). Want me to try a different approach?\n\n```widget\n{err_widget}\n```'
                    cat_err=tone_atlas.classify_intent(message,skill_used=name)
                    conv.append('assistant',wrapped_err,{'tier':f'tier0_skill_{name}_failed','skill_calls':skill_calls,'tokens':0,'persona':persona_for_err.name,'category':cat_err})
                    return {'answer':wrapped_err,'tier':f'tier0_skill_{name}_failed','tokens':0,'session_id':conv.session_id,'skill_calls':skill_calls,'wall_s':round(time.time()-t0,3),'persona':persona_for_err.name,'category':cat_err,'skill_error':True}
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
        history_pairs=conv.history_pairs(n=12) if len(conv.turns)>1 else []
        atlas_recall=self.atlas.recall(message,session_id=conv.session_id,k=3,include_global=True) if self.atlas is not None else []
        for r in atlas_recall:
            pair=(r['user'],r['assistant'])
            if pair not in history_pairs:history_pairs=[pair]+history_pairs
        history_pairs=history_pairs[-12:]
        user_facts=self._extract_user_facts(conv,extra_user_msgs=[r.get('user','') for r in atlas_recall])
        is_private=detect_personal(message) or conv.has_personal(n=20) or any(r.get('is_personal') for r in atlas_recall)
        category=tone_atlas.classify_intent(message)
        raw_ans='';tier='?';tokens=0
        sl=getattr(self.adam,'sem_lut',None)
        if sl is not None and not history_pairs:
            try:
                eff_margin=sl.auto_margin() if hasattr(sl,'auto_margin') else 0.08
                hit=sl.lookup_soft(message,margin=eff_margin) if hasattr(sl,'lookup_soft') else None
                if hit:raw_ans=hit;tier='tier1_5_semantic_lesson';tokens=0
            except Exception:pass
        if not raw_ans and not history_pairs and not is_private:
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
            try:
                from amni.serve.pre_response_review import review as _prr
                _rv=_prr(message,agent=self)
                if _rv.get('brief'):sys_p=f"{sys_p}\n\n{_rv['brief']}"
            except Exception:pass
            if apply_cot:sys_p=f'{sys_p}\n\n{cot_scaffold}'
            cot_extra=1400 if (apply_cot and cot_tag=='code') else (700 if apply_cot else 0)
            _is_code=bool(_CODE_LANG_RE.search(message) or any(k in message.lower() for k in ('write','implement','function','how do i','example','code','setup','config','server')))
            _code_extra=600 if _is_code and not apply_cot else 0
            r=self.adam.chat_persona(message,system=sys_p,history=history_pairs,facts=user_facts,is_private=is_private,max_new_tokens=int(160+300*persona.length)+cot_extra+_code_extra,do_sample=True)
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
                        snippet=('\n\n'.join(blocks) if len(blocks)>1 else runnable[-1])
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
                                            ok_promote,promo_reason=_should_promote(snippet,asserts,div_score)
                                            if ok_promote:
                                                try:
                                                    tr=self.adam.teach(message,raw_ans[:2000])
                                                    tier+='_promoted'
                                                    skill_calls.append({'skill':'promote_lesson','args':{'reason':promo_reason},'result':{'lessons_n':tr.get('lessons_n',0)}})
                                                except Exception:pass
                                            else:
                                                tier+='_quality_gated'
                                                skill_calls.append({'skill':'promote_lesson','args':{'gated':True,'reason':promo_reason},'result':{}})
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
            hist_block='\n'.join(f'USER: {u}\nASSISTANT: {a}' for u,a in history_pairs)
            facts_block='\n'.join(f'- {f}' for f in user_facts)
            preface=(f'[User facts]\n{facts_block}\n\n' if facts_block else '')+(f'[Prior turns]\n{hist_block}\n\n' if hist_block else '')
            framed=f'{preface}[New Question]\n{message}' if preface else message
            if apply_cot:framed=f'{cot_scaffold}\nQuery: {framed}'
            effective_writeback=writeback and not is_private and not history_pairs
            fb=self.adam.ask(framed,writeback=effective_writeback)
            raw_ans=fb.get('answer') or '';tier=fb.get('tier','?')+(f'_cot_{cot_tag}' if apply_cot else '');tokens=fb.get('tokens',0)
        wrapped=tone_atlas.wrap(raw_ans,category,persona,seed=message)
        if skill_answer and skill_answer.startswith('(skill'):wrapped=f'{skill_answer}\n{wrapped}'
        if confirmed_clarification and confirmed_clarification.get('confirmed'):
            _cc='confidential' if confirmed_clarification.get('is_confidential') else 'public'
            wrapped=f"Got it — marking that {_cc}. {wrapped}"
        if self.personal_atlas is not None:
            try:
                pending=self.personal_atlas.next_clarification_to_ask()
                if pending:wrapped=f"{wrapped}\n\n{self.personal_atlas.build_clarification_question(pending)}"
            except Exception as _qe:print(f'[AmniAgent] personal_atlas next_clarification failed: {_qe}',flush=True)
        conv.append('assistant',wrapped,{'tier':tier,'tokens':tokens,'skill_calls':skill_calls,'persona':persona.name,'category':category,'is_private':is_private})
        if self.atlas is not None and raw_ans:
            try:self.atlas.record(conv.session_id,message,wrapped,is_personal=is_private)
            except Exception as e:print(f'[AmniAgent] atlas record failed: {e}',flush=True)
        return {'answer':wrapped,'tier':tier,'tokens':tokens,'session_id':conv.session_id,'skill_calls':skill_calls,'wall_s':round(time.time()-t0,3),'persona':persona.name,'category':category,'is_private':is_private}
    def _format_skill_output(self,name:str,out:Any)->str:
        if name=='time':return f'currently {out.get("iso")} local time'
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
        if name=='web':
            ans=out.get('answer','(no answer)');srcs=out.get('sources',[])[:5]
            def _host(u):
                try:
                    from urllib.parse import urlparse
                    h=urlparse(u).hostname or u
                    if h.startswith('www.'):h=h[4:]
                    return h
                except Exception:return u
            cite_lines='\n'.join(f'  {i+1}. [{_host(s)}]({s})' for i,s in enumerate(srcs)) if srcs else ''
            return f'{ans}\n\n**Sources:**\n{cite_lines}' if cite_lines else ans
        if name=='recall':
            if out.get('error'):return f'(recall error: {out["error"]})'
            hits=out.get('hits') or [];q=out.get('query','');n=out.get('n_hits',0)
            if not hits:return f'I scanned {out.get("sessions_scanned",0)} past session(s) and found no matches for `{q}`.'
            lines=[f'Found **{n}** past session(s) mentioning `{q}`:']
            for h in hits[:5]:
                sid=h.get('session_id','?')[:8];iso=h.get('iso','?')
                lines.append(f'\n**{iso}** · session `{sid}`')
                for s in (h.get('snippets') or [])[:2]:
                    role=s.get('role','?');txt=(s.get('text','') or '').replace('\n',' ')[:200]
                    lines.append(f'  - _{role}_: {txt}')
            return '\n'.join(lines)
        if name=='reminder':
            if out.get('error'):return f'(reminder error: {out["error"]})'
            if 'id' in out and 'text' in out:
                due=out.get('due_iso')
                tail=f' for **{due}**' if due else ''
                return f'Reminder saved{tail}: _{out["text"]}_  \nDismiss anytime via `/reminders` or `amni reminders dismiss {out["id"]}`.'
            if 'reminders' in out:
                items=out['reminders'] or []
                if not items:return 'No active reminders. Set one with "remind me to ...".'
                lines=[f'**{len(items)} active reminder(s):**']
                for r in items[:10]:
                    due=r.get('due_iso');tag=f'· due {due}' if due else '· no due'
                    lines.append(f'- `{r.get("id","?")}`  _{r.get("text","")}_  {tag}')
                return '\n'.join(lines)
            if 'dismissed' in out:return f'Dismissed reminder `{out["dismissed"]}`.'
            return json.dumps(out,default=str)[:600]
        if name=='note':
            if out.get('error'):return f'(note error: {out["error"]})'
            if 'id' in out and 'text' in out:
                tg=out.get('tags') or [];tag_s=(' · ' + ' '.join(f'#{t}' for t in tg)) if tg else ''
                return f'Note saved 📝{tag_s}  \n_{out["text"][:200]}_  \n`{out["id"]}` — manage via `/notes` or `amni notes`.'
            if 'notes' in out:
                items=out['notes'] or []
                if not items:return 'No notes yet. Try "note: pick up groceries tomorrow #errands" or just /notes.'
                lines=[f'**{len(items)} note(s):**']
                for r in items[:20]:
                    tg=r.get('tags') or [];tag_s=(' ' + ' '.join(f'`#{t}`' for t in tg)) if tg else ''
                    iso=(r.get('iso','') or '').replace('T',' ')[:16]
                    lines.append(f'- `{r.get("id","?")}` _{iso}_{tag_s}  \n  {r.get("text","")[:240]}')
                return '\n'.join(lines)
            if 'deleted' in out:return f'Deleted note `{out["deleted"]}` ({out.get("remaining",0)} remaining).'
            if 'tags' in out:
                tags=out['tags'] or []
                return ('Tags in use: ' + ' '.join(f'`#{t}`' for t in tags[:40])) if tags else 'No tags yet.'
            return json.dumps(out,default=str)[:600]
        if name=='pose_coach':
            if out.get('error'):return f'(pose_coach error: {out["error"]})'
            if 'exercises' in out:
                ex=out['exercises'] or []
                lines=['I can coach your form on these — enable the camera (📷) and I\'ll count reps + check your angles live:']
                for e in ex:
                    tb=e.get('target_bottom');tail=f' · target ≤{tb:.0f}°' if tb is not None else ''
                    lines.append(f'- **{e.get("label","?")}** — {e.get("cue","")}{tail}')
                return '\n'.join(lines)
            if 'history' in out:
                h=out['history'] or []
                if not h:return 'No workout history yet. Start a session with the camera on and I\'ll log your reps + form.'
                lines=[f'**Last {len(h)} session(s):**']
                for s in h[:10]:
                    iso=(s.get('iso','') or '').replace('T',' ')[:16]
                    lines.append(f'- {iso} · **{s.get("label","?")}** — {s.get("reps",0)} reps, {s.get("clean_rate_pct",0)}% clean')
                return '\n'.join(lines)
            if 'started' in out:return f'Started a **{out.get("label","?")}** session. {out.get("cue","")}. I\'m watching your form — go!'
            if 'reps' in out and 'feedback' in out:return out['feedback']
            if 'feedback' in out:return out['feedback']
            return json.dumps(out,default=str)[:600]
        if name=='pc_action':
            if out.get('refused'):return f'⛔ I won\'t run that — it matches a destructive/irreversible pattern. ({out.get("target","")[:80]})'
            if out.get('error'):return f'(pc_action error: {out["error"]})'
            if out.get('requires_confirm'):
                _w=json.dumps({'type':'pc_confirm','title':'PC action — confirm','icon':'⚙','data':{'token':out.get('token'),'description':out.get('description',''),'risk':out.get('risk'),'action':out.get('action'),'target':out.get('target')}})
                return f'{out.get("description","")}\n\n**This won\'t run until you confirm.** Tap CONFIRM below, or reply `cancel {out.get("token")}`.\n\n```widget\n{_w}\n```'
            if out.get('executed'):
                res=out.get('result') or {}
                return f'✅ Done — {out.get("action")}: `{out.get("target","")[:80]}`.\n```\n{json.dumps(res,default=str)[:800]}\n```'
            if out.get('cancelled'):return f'Cancelled `{out["cancelled"]}`.'
            if 'pending' in out:
                items=out['pending'] or []
                if not items:return 'No PC actions awaiting confirmation.'
                return '**Pending PC actions:**\n'+'\n'.join(f'- `{i["token"]}` [{i["risk"]}] {i["action"]}: {i["target"]}' for i in items[:10])
            return json.dumps(out,default=str)[:600]
        if name=='find':
            hits=out.get('hits') or [];q=out.get('query','');n=out.get('n_hits',0);files_s=out.get('files_scanned',0)
            if not hits:return f'No matches for `{q}` in workdir ({files_s} text file(s) scanned).'
            lines=[f'Found **{n}** hit(s) for `{q}` across {files_s} file(s):']
            for h in hits[:20]:
                p=h.get('path','?');ln=h.get('line','?');sn=h.get('snippet','')
                lines.append(f'\n- `{p}:{ln}`  →  {sn}')
            if out.get('truncated'):lines.append(f'\n_(truncated to {len(hits)} hits — refine the query for more)_')
            return '\n'.join(lines)
        if name=='file_read':return f'```\n{out.get("content","")}\n```\n({out.get("bytes")} bytes from {out.get("path")})'
        if name=='file_write':
            ch=out.get('change') or {};v=out.get('verification') or {};op='create' if out.get('created') else 'overwrite'
            verified=v.get('verified');issues=v.get('issues') or [];suggested=v.get('suggested_tests') or []
            vstatus='pass' if verified is True else ('fail' if verified is False else 'manual')
            tr=v.get('test_run') or {}
            widget=json.dumps({'type':'file_change','title':op.upper()+': '+(out.get('path') or '?').split('/')[-1].split('\\')[-1],'icon':'+' if op=='create' else '↻','data':{'op':op,'path':out.get('path'),'ext':out.get('ext','txt'),'lines_added':ch.get('lines_added',0),'lines_removed':ch.get('lines_removed',0),'lines_before':ch.get('lines_before',0),'lines_after':ch.get('lines_after',0),'bytes_after':ch.get('bytes_after',out.get('bytes_written',0)),'preview':ch.get('preview',''),'before_preview':ch.get('before_preview',''),'diff_unified':ch.get('diff_unified',''),'verification_status':vstatus,'verification_issues':issues,'verification_checks':v.get('checks',[]),'suggested_tests':suggested,'verification_reason':v.get('reason',''),'test_run':tr}})
            note=''
            if verified is False:note=f' Verification FAILED: {"; ".join(issues[:3])}. I would not trust this edit yet — please review.'
            elif verified is None:note=f' Auto-verification skipped ({v.get("reason","unknown")}); I added a testing reminder.'
            elif tr.get('ran') and tr.get('ok'):note=f' Sibling tests PASSED ({tr.get("passed",0)} passed in {tr.get("duration_s","?")}s).'
            elif suggested:note=f' Sibling test file detected: {suggested[0]}. Run it to confirm behavior.'
            return f'Wrote {out.get("bytes_written")} bytes to {out.get("path")}.{note}\n\n```widget\n{widget}\n```'
        if name=='code_edit':
            if out.get('error'):return f'(code_edit error: {out.get("error")})'
            ch=out.get('change') or {};v=out.get('verification') or {}
            verified=v.get('verified');issues=v.get('issues') or [];suggested=v.get('suggested_tests') or []
            vstatus='pass' if verified is True else ('fail' if verified is False else 'manual')
            tr=v.get('test_run') or {}
            widget=json.dumps({'type':'file_change','title':'EDIT: '+(out.get('path') or '?').split('/')[-1].split('\\')[-1],'icon':'✎','data':{'op':'edit','path':out.get('path'),'ext':out.get('ext','txt'),'replacements':out.get('replacements',0),'lines_added':ch.get('lines_added',0),'lines_removed':ch.get('lines_removed',0),'lines_before':ch.get('lines_before',0),'lines_after':ch.get('lines_after',0),'bytes_after':ch.get('bytes_after',0),'preview':ch.get('preview',''),'before_preview':ch.get('before_preview',''),'diff_unified':ch.get('diff_unified',''),'verification_status':vstatus,'verification_issues':issues,'verification_checks':v.get('checks',[]),'suggested_tests':suggested,'verification_reason':v.get('reason',''),'test_run':tr}})
            note=''
            if verified is False:note=f' Verification FAILED: {"; ".join(issues[:3])}.'
            elif verified is None:note=f' Auto-verification skipped ({v.get("reason","unknown")}); testing reminder added.'
            elif tr.get('ran') and tr.get('ok'):note=f' Sibling tests PASSED ({tr.get("passed",0)} passed in {tr.get("duration_s","?")}s).'
            elif suggested:note=f' Sibling test file: {suggested[0]} — recommend running.'
            return f'Edited {out.get("path")}: {out.get("replacements",0)} replacement(s).{note}\n\n```widget\n{widget}\n```'
        if name=='shell':return f'$ {out.get("cmd")}\n(exit={out.get("returncode")})\n{out.get("stdout","")}{out.get("stderr","")}'
        if name=='scan':
            if out.get('error'):return f'(scan error: {out["error"]})'
            return f'Scanned {out.get("files_scanned",0)} file(s), added {out.get("lessons_added",0)} lesson(s) (total: {out.get("lessons_total",0)}). Distilled: {out.get("distilled")}.'
        if name=='chain':
            if out.get('error'):return f'(chain error: {out["error"]})'
            results=out.get('results') or []
            if not results:return '(empty chain)'
            lines=[]
            for i,r in enumerate(results):
                sk=r.get('skill','?');ok=r.get('ok');marker='✓' if ok else '✗'
                inner_out=r.get('output') or {}
                if sk=='file_write' and isinstance(inner_out,dict):summary=f'wrote {inner_out.get("bytes_written",0)}b to {inner_out.get("path","?")}'
                elif sk=='weather' and isinstance(inner_out,dict):summary=f'{inner_out.get("temp_c","?")}°C in {inner_out.get("location","?")}'
                elif sk=='calc' and isinstance(inner_out,dict):summary=f'value={inner_out.get("value","?")}'
                elif sk=='time' and isinstance(inner_out,dict):summary=str(inner_out.get('iso','?'))
                elif r.get('error'):summary=f'error: {r["error"]}'
                else:summary=str(inner_out)[:120] if inner_out else '(no output)'
                lines.append(f'  {marker} step {i+1} **{sk}** → {summary}')
            head=f'Chain {"completed" if out.get("ok") else "halted"} ({out.get("n_steps","?")} step(s)):'
            return head+'\n'+'\n'.join(lines)
        if name=='coach':
            if out.get('error'):return f'(coach error: {out["error"]})'
            if out.get('score') is not None:
                fb=out.get('feedback','');nq=out.get('next_question','')
                tail=f'\n\nNext: {nq}' if nq else ''
                return f'Score: **{out.get("score")}/100** — {fb}{tail}'
            if out.get('question'):
                topic=out.get('topic','?');diff=out.get('difficulty','?')
                return f'Starting coach session on **{topic}** at difficulty {diff}.\n\n**Q:** {out["question"]}\n\n(answer in /jarvis coach panel, or POST /skills/coach action=answer)'
            if out.get('hint'):return f'**Hint:** {out["hint"]}'
            if out.get('skipped'):return f'Skipped. Next: {out.get("next_question","(no next)")}'
            return json.dumps(out,default=str)[:600]
        if name=='learning_daemon':
            if out.get('error'):return f'(daemon error: {out["error"]})'
            if out.get('paused'):return 'Learning daemon paused. I will stop autonomous topic ingestion until you resume me.'
            if out.get('resumed'):return 'Learning daemon resumed. I am back to autonomous learning in the background.'
            if 'gap' in out:
                g=out.get('gap');return (f'Curiosity tick fired — picked "{g.get("topic","?")}" ({g.get("reason","")}) and queued it. Queue depth now {out.get("queue_depth","?")}.' if g else 'Curiosity tick fired but no knowledge gap to fill right now.')
            if out.get('queued') is True and 'topic' in out:return f'Queued "{out.get("topic","")}" for autonomous learning. I will research it in the background and integrate what I find.'
            if out.get('queued') is False:return f'Could not queue topic: {out.get("reason","queue full")}.'
            ct=out.get('current_topic');cph=out.get('current_topic_phase','');c=out.get('counters',{}) or {};fph=out.get('facts_per_hour',0);qd=out.get('queue_depth',0);uh=out.get('uptime_hours',0);en=out.get('enabled');new=c.get('qa_pairs_new',0);urls=c.get('urls_ingested',0)
            if ct:return f'Right now I am learning about **{ct}** ({cph or "working"}). Daemon has been up {uh}h, learned {new} new facts across {urls} sources, currently {fph}/h. Queue depth {qd}.'
            if not en:return f'Daemon is paused. {new} facts learned in {uh}h of uptime ({fph}/h average). Queue depth {qd}.'
            return f'Daemon idle, waiting for the next curiosity tick. {new} facts learned in {uh}h ({fph}/h average). Queue depth {qd}.'
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
