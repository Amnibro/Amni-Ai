import os,sys,time,json,subprocess
try:sys.stdout.reconfigure(encoding='utf-8',errors='replace');sys.stderr.reconfigure(encoding='utf-8',errors='replace')
except Exception:pass
for k,v in (('HIP_VISIBLE_DEVICES','1'),('PYTORCH_ROCM_ARCH','gfx1101'),('HSA_OVERRIDE_GFX_VERSION','11.0.0'),('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL','1'),('HIP_FORCE_DEV_KERNARG','1'),('GPU_MAX_HW_QUEUES','8'),('MIOPEN_FIND_MODE','2'),('MIOPEN_FIND_ENFORCE','NONE'),('AMNI_BLOCK_SPEC','0')):os.environ.setdefault(k,v)
_home=os.path.expanduser('~')
for k,v in (('MIOPEN_USER_DB_PATH',os.path.join(_home,'.miopen')),('MIOPEN_CUSTOM_CACHE_DIR',os.path.join(_home,'.miopen')),('TRITON_CACHE_DIR',os.path.join(_home,'.triton'))):os.environ.setdefault(k,v)
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
bake=cfg['bake'];repo_url=cfg.get('repo_url');ref=cfg.get('ref');sandbox=str(Path(cfg['sandbox']).resolve());goal=cfg['goal'].strip()
max_steps=int(cfg.get('max_steps',16));timeout_s=float(cfg.get('timeout_s',1000));evlog=cfg.get('evlog','logs/ac_events.jsonl');docout=cfg.get('doc','docs/ac_report.md')
from amni.adam import Adam
from amni.serve.skills import default_registry
from amni.serve.agentic import run_goal_stream
NAV_PROMPT='''You are an autonomous software engineer fixing ONE bug in an existing repository that is ALREADY checked out in your working directory. The codebase has already been indexed for you. Work ONE step at a time, thinking out loud.

Available tools (name(args): description):
{tools}

Output EXACTLY ONE JSON object per step — nothing else. Each step must be DIFFERENT from the last and ADVANCE the plan:
  {{"tool":"<name>","args":{{...}}}}
  {{"ask_user":"<a specific question — use when you are stuck or missing info you can't find>"}}
  {{"final":"<answer>"}}

The plan ADVANCES through these phases — do each ONCE, in order, then move on:
1. READ THE ISSUE (once) if a number is given: shell {{"cmd":"gh issue view <n> --repo <owner/name>"}}.
2. LOCATE (once): find {{"query":"function <name>"}} — returns a "path" and a "line" L. After this, NEVER locate again.
3. READ (once) the region: file_read {{"path":"<the path>","line_offset":<the line L>,"line_limit":40}}. After this you HAVE the code — do NOT read or locate again.
4. DIAGNOSE: confirm the defect is ACTUALLY present in what you read. If the code is ALREADY correct, do NOT edit — emit final stating it is already correct.
5. FIX minimally: code_edit {{"path":"<file>","find":"<SHORT unique verbatim fragment of the ONE buggy line>","replace":"<that fragment corrected>"}}.
6. VERIFY: shell {{"cmd":"node --check <file>"}} (JS) or test_run.
7. FINISH (once verified): {{"final":"<file+function, the bug, the exact change you made, and the node --check result>"}}.

RULES:
- If you are UNSURE how an algorithm or concept works, use web {{"query":"<what you need>"}} to look it up — but at most once or twice. If the web returns NOTHING useful, do NOT keep searching the same thing: EITHER ask the user with {{"ask_user":"I searched for X and found nothing; I don't understand Y; can you tell me Z?"}}, OR make your BEST ATTEMPT from what you already know (write the file) so we can run it and learn from the result.
- When you write a code file, include a SHORT self-test under `if __name__ == "__main__":` that ASSERTS the key correctness property on a sample input — e.g. for a transform, `assert inverse(forward(x)) == x`; for a fix, assert the expected output. The harness RUNS the file; if an assert fails or it errors, you'll get the message and must fix the logic (not just syntax).
- ONE tool per step. Paths RELATIVE to the workdir.
- Locate before reading; line_limit <= 40; never dump a whole big file.
- code_edit "find" MUST be a SHORT fragment copied verbatim from file_read output.
- If a tool errors or returns nothing useful, change args/tool — do NOT repeat it identically.
- NEVER delete files, git reset --hard, git clean, or force-push. Smallest change only.
- If the bug is already fixed in the code, say so in final instead of editing.

Goal: {goal}
{trace}
Next step (ONE JSON object):'''
def log(m):print(m,flush=True)
log(f'[ac] bake={bake} sandbox={sandbox}\n[ac] repo={repo_url} ref={ref}')
Path(sandbox).mkdir(parents=True,exist_ok=True)
repo_dir=sandbox
if repo_url:
    name=repo_url.rstrip('/').split('/')[-1].replace('.git','')
    repo_dir=str(Path(sandbox)/name)
    if not Path(repo_dir,'.git').exists():
        log(f'[ac] HARNESS: git clone {repo_url}');subprocess.run(['git','clone','--quiet',repo_url,repo_dir],check=True)
    if ref:
        log(f'[ac] HARNESS: git checkout {ref}');subprocess.run(['git','-C',repo_dir,'checkout','--quiet',ref],check=True)
log(f'[ac] workdir(repo)={repo_dir}')
t0=time.time();adam=Adam(bake=bake,model=bake,web_unrestricted=True);log(f'[ac] Adam loaded {time.time()-t0:.1f}s')
CODING={'shell','scan','find','code_index','file_read','code_edit','code_diff','file_write','git','test_run','parse_error','project_info','web','mem'}
skills=default_registry(workdir=repo_dir,unrestricted=False)
for nm in [s['name'] for s in skills.list_skills()]:
    if nm not in CODING:skills._skills.pop(nm,None)
log(f'[ac] tools={sorted(skills._skills.keys())}  confined={[str(r) for r in skills.roots]}')
_user_help=cfg.get('user_help');_asked=[False]
def _ask_cb(q):
    if _user_help and not _asked[0]:_asked[0]=True;log(f'   [user provides help]: {str(_user_help)[:160]}…');return _user_help
    return None
fh=open(evlog,'w',encoding='utf-8');events=[];tstart=time.time()
for ev in run_goal_stream(adam,skills,goal,max_steps=max_steps,timeout_s=timeout_s,plan_prompt=NAV_PROMPT,ask_cb=_ask_cb):
    ev['_t']=round(time.time()-tstart,1);events.append(ev);fh.write(json.dumps(ev,default=str)+'\n');fh.flush()
    e=ev.get('event')
    if e=='dir_learned':log(f"\n[+{ev['_t']}s] 📚 LEARNED DIRECTORY: {json.dumps(ev.get('summary'),default=str)[:200]}")
    elif e=='thought':log(f"[+{ev['_t']}s] 💭 {ev.get('thought')}")
    elif e=='step_start':log(f"[+{ev['_t']}s] 🔧 STEP {ev.get('step')}: {ev.get('tool')} {json.dumps(ev.get('args'),default=str)[:160]}")
    elif e=='step_result':log(f"          -> ok={ev.get('ok')} {str(ev.get('output'))[:150]}")
    elif e=='self_reflect_start':log(f"[+{ev['_t']}s] 🧭 SIDE-QUEST: repeated/similar steps detected — re-grounding on the original goal…")
    elif e=='self_reflect':log(f"[+{ev['_t']}s] 🧭 self-review: done={ev.get('done')} | {ev.get('reason')}" + (f" | next: {ev.get('next_action')}" if not ev.get('done') else ""))
    elif e=='no_progress_stop':log(f"[+{ev['_t']}s] 🛑 NO-PROGRESS GUARD: {ev.get('no_progress_steps')} steps without progress → stopping with an honest status (no pointless looping)")
    elif e=='debug_hint':log(f"[+{ev['_t']}s] 🐞 DEBUG: {ev.get('hint')}")
    elif e=='research_empty':log(f"[+{ev['_t']}s] 🔎 web found nothing — agent should ask the user or attempt with what it knows")
    elif e=='help_request':log(f"[+{ev['_t']}s] 🙋 ADAM ASKS THE USER: {ev.get('question')}")
    elif e=='help_unanswered':log(f"[+{ev['_t']}s] (no user available → steering Adam to attempt with what it knows)")
    elif e=='auto_verify':log(f"[+{ev['_t']}s] ✔️ HARNESS AUTO-VERIFY ok={ev.get('ok')} ({ev.get('cmd')})")
    elif e=='compacted':log(f"[+{ev['_t']}s] 🧠 RECAP (atex): {str(ev.get('compact')).replace(chr(10),' | ')[:260]}")
    elif e=='verify_required':log(f"[+{ev['_t']}s] ⚠️ harness requires verification before finishing")
    elif e=='force_write':log(f"[+{ev['_t']}s] ✍️ FORCE-WRITE: repeated {ev.get('tool')} {ev.get('after_repeats')}x but already has what it needs → steering hard to WRITE the deliverable now (not giving up)")
    elif e=='file_proliferation':log(f"[+{ev['_t']}s] 🗂️ FILE-PROLIFERATION GUARD: keeps creating new files {ev.get('created')} → steering to consolidate + edit ONE file in place (not spawn another)")
    elif e=='edit_miss_escalation':log(f"[+{ev['_t']}s] 🚧 EDIT-MISS ESCALATION: code_edit on {ev.get('path')} missed {ev.get('misses')}x (hallucinated find) → hard steer to read-exact-line or finalize (stop guessing)")
    elif e=='critique_start':log(f"[+{ev['_t']}s] 🧐 CRITICAL SELF-EXAM: re-reading {ev.get('artifact')} and judging its own answer with a skeptical eye…")
    elif e=='critique':log(f"[+{ev['_t']}s] 🧐 self-verdict: acceptable={ev.get('acceptable')} | invariant-test={ev.get('test')}" + (f" | fault: {ev.get('fault')}" if not ev.get('acceptable') else " (affirmed; test executed by harness)"))
    elif e=='critique_reject':log(f"[+{ev['_t']}s] ↩️ SELF-REJECTED (round {ev.get('round')}): recognized its own fault → looping back to fix instead of shipping it")
    elif e=='final':log(f"\n[+{ev['_t']}s] ✅ FINAL: {str(ev.get('answer'))[:500]}")
    elif e=='plan_unparseable':log(f"[+{ev['_t']}s] [re-format] planner emitted invalid JSON (retry {ev.get('retry')}/3) — re-prompting for valid JSON")
    elif e in ('timeout','max_steps_reached','auto_final','duplicate_skipped','args_normalized'):log(f"[+{ev['_t']}s] [{e}]")
fh.close()
def _narrate(ev):
    e=ev.get('event')
    if e=='dir_learned':s=ev.get('summary') or {};return f"Learned the codebase: indexed {s.get('files_indexed')} files / {s.get('symbols')} symbols ({s.get('languages')})."
    if e=='step_start':
        t=ev.get('tool');a=ev.get('args') or {}
        if t=='find':return f"Searched the repo for `{a.get('query')}` to locate the code."
        if t=='code_index':return f"Queried the code index for `{a.get('term') or a.get('query')}`."
        if t=='file_read':return f"Read `{a.get('path')}` around line {a.get('line_offset')} to inspect the code."
        if t in('code_edit','code_diff'):return f"Edited `{a.get('path')}` — applied the minimal fix."
        if t=='shell':return f"Ran shell: `{(a.get('cmd') or '')[:80]}`."
        if t=='file_write':return f"Wrote `{a.get('path')}`."
        return f"Used `{t}`."
    if e=='auto_verify':return f"Harness auto-verified the edit with `{ev.get('cmd')}` → {'passed' if ev.get('ok') else 'FAILED'}."
    if e=='compacted':return f"Recap (working memory → ATEX): {str(ev.get('compact')).replace(chr(10),' / ')[:240]}"
    if e=='verify_required':return "Harness required a verification step before finishing."
    if e=='critique':return (f"Critically self-examined the finished answer — affirmed it after scrutiny." if ev.get('acceptable') else f"Critically self-examined the finished answer and CAUGHT a fault: {ev.get('fault')}.")
    if e=='critique_reject':return f"Recognized its own work was wrong (round {ev.get('round')}) and looped back to fix it rather than ship it."
    return None
narr=[n for n in (_narrate(e) for e in events) if n]
diff=subprocess.run(['git','-C',repo_dir,'diff'],capture_output=True,text=True).stdout if Path(repo_dir,'.git').exists() else ''
node_ok=None
if Path(repo_dir,'.git').exists() and diff.strip():
    changed=[ln[6:] for ln in diff.splitlines() if ln.startswith('+++ b/')]
    js=[c for c in changed if c.lower().endswith(('.js','.mjs','.cjs','.ts','.jsx','.tsx'))]
    if js:
        node_ok=all(subprocess.run(['node','--check',str(Path(repo_dir)/c)],capture_output=True,text=True).returncode==0 for c in js)
        if node_ok is False:
            subprocess.run(['git','-C',repo_dir,'checkout','--','.'],capture_output=True,text=True);diff='';log('[ac] NON-DESTRUCTIVE GUARD: final node --check FAILED -> reverted all changes (left repo clean)')
finals=[e for e in events if e.get('event')=='final']
recaps=[e for e in events if e.get('event')=='compacted']
thoughts=[e for e in events if e.get('event')=='thought']
steps=[e for e in events if e.get('event')=='step_start']
md=['# Autonomous Coder Report','',f'**Goal:** {goal}','',f'**Repo:** {repo_url} @ {ref}  ·  **Model:** {bake}  ·  **Wall:** {round(time.time()-tstart,1)}s  ·  **Steps:** {len(steps)}  ·  **Final node --check:** {node_ok}','']
md+=['## Reasoning / thinking trail (what Adam did and why)','']+[f"- {n}" for n in narr]+['']
md+=['## Tool steps','']+[f"{e.get('step')}. `{e.get('tool')}` {json.dumps(e.get('args'),default=str)[:140]}" for e in steps]+['']
if recaps:md+=['## ATEX recaps','']+[f"- {str(e.get('compact')).replace(chr(10),' / ')[:400]}" for e in recaps]+['']
md+=['## Final answer','',(finals[-1].get('answer') if finals else '(no final emitted)'),'']
md+=['## Diff applied','','```diff',(diff[:4000] if diff.strip() else '(no changes — code was already correct or no edit made)'),'```','']
Path(docout).parent.mkdir(parents=True,exist_ok=True);Path(docout).write_text('\n'.join(md),encoding='utf-8')
log(f'\n[ac] report -> {docout}  ·  diff_lines={len(diff.splitlines())}')
