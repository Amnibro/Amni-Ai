"""PlanExecuteCoder — the blazingly-fast coding pipeline. A strong PLANNER (12B Nvfp4AtexChatService) decomposes the task into small self-contained functions; a fast EXECUTOR (3B GraniteAtexChatService, GPU-resident) writes each one; the macro/PTEX accelerator scaffolds known boilerplate + arms the speculative-decode draft index; every VERIFIED block is captured into the PERSISTENT PTEX macro library (load_lib on start, save_lib after) so the next session starts pre-warmed and coverage->1. Strict-additive: verification gates capture, the library only grows, wall-clock drops the more you code. planner_svc defaults to executor_svc (single-model safe mode); pass the 12B as planner for the plan/execute split. Built on AdamCoder (generate->sanitize->verify->bank) + MacroCodeEngine (coverage/draft_index)."""
import re
from amni.inference.adam_coder import AdamCoder
from amni.inference.macro_code_engine import MacroCodeEngine
_W=re.compile(r'[A-Za-z_]\w+')
class PlanExecuteCoder:
    def __init__(s,executor_svc,planner_svc=None,engine=None,ptex_root='experiences/code_macros_ptex'):
        s.exec_svc=executor_svc;s.plan_svc=planner_svc or executor_svc;s.ptex_root=ptex_root
        s.engine=engine or MacroCodeEngine(getattr(executor_svc,'tok',None))
        s.loaded=s.engine.load_lib(ptex_root) if ptex_root else 0
        s.coder=AdamCoder(executor_svc,engine=s.engine);s._arm()
    def _arm(s):
        try:s.exec_svc.draft_index=s.engine.build_draft_index()
        except Exception:pass
    def plan(s,task,max_tokens=420):
        sysp='You are a senior software architect. Decompose the coding task into a numbered list of small, self-contained Python functions. Each line: "N. function_name(args) — one-line purpose". Output ONLY the numbered list: no code, no prose, no preamble.'
        r,_=s.plan_svc.chat(f'Task: {task}',system=sysp,max_new_tokens=max_tokens)
        steps=[re.sub(r'^\s*\d+[.)]\s*','',ln).strip() for ln in r.split('\n') if re.match(r'\s*\d+[.)]',ln)]
        return steps or [task]
    def _scaffold(s,step,k=3):
        w={m.group(0).lower() for m in _W.finditer(step)}
        scored=sorted(((len(w&{x.group(0).lower() for x in _W.finditer(b)}),b) for b in s.engine.blocks),key=lambda t:t[0],reverse=True)
        return '\n'.join(b for n,b in scored[:k] if n>0)
    def execute(s,task,plan=None,selftest=False,max_tokens=700):
        steps=plan if plan is not None else s.plan(task);out=[]
        for st in steps:
            r=s.coder.write(f'a Python function: {st}',scaffold=s._scaffold(st),max_tokens=max_tokens,selftest=selftest)
            out.append({'step':st,'verified':r['verified'],'banked':r.get('banked_macros',0),'attempts':r['attempts'],'code':r['code'],'error':r.get('error')})
        if s.ptex_root:s.engine.save_lib(s.ptex_root)
        s._arm()
        return {'task':task,'plan':steps,'results':out,'verified':sum(r['verified'] for r in out),'lib':len(s.engine.blocks)}
    def speedup(s,code):return s.engine.coverage(code)
