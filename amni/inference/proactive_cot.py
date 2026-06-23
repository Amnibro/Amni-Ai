"""ProactiveCoT — upgrade reasoning from REACTIVE (greedy forward token-unroll) to PROACTIVE (a goal-directed controller: predict -> act -> check -> correct). Two loops:
 INNER (per task): plan(goal+decomposition+expected answer-shape+traps) -> hypothesis(guess first) -> monitored execute(expectation-checks + anti-spiral coverage) -> falsify(attack the guess / each distractor) -> commit. A surprise (expectation mismatch) is a SIGNAL; a stall (no new ground) triggers a strategy switch BEFORE burning the budget — the proactive form of the spiral fix.
 OUTER (per conversation): read the conversation arc, infer the user's larger GOAL, and anticipate the proactive NEXT STEPS to prepare/surface (delivers via the Guardian push).
Model access is an injected chat_fn(user_msg, system=, max_tokens=, context=) -> str, so it runs on the NVFP4 svc/agent OR a mock for CPU tests. The anti-spiral monitor reuses the coverage principle from macro_sentinel (low novelty = stuck)."""
import re
def _norm(line):return re.sub(r'\s+',' ',re.sub(r'[-+]?\d[\d,.]*','#',line.strip().lower())).strip()
def stall_score(text):
    lines=[_norm(l) for l in (text or '').split('\n') if len(l.strip())>8]
    if len(lines)<4:return 0.0
    seen=set();rep=0
    for l in lines:
        rep+=l in seen;seen.add(l)
    return round(rep/len(lines),3)
_PLAN_SYS='Plan BEFORE solving. State: GOAL (one line), a numbered DECOMPOSITION of the steps you expect to take, the expected answer SHAPE (units/magnitude/form), and which options or paths look like TRAPS. Be brief — this is the plan, not the solution.'
_HYP_SYS='Give your single best FIRST GUESS at the final answer in one line, prefixed "Hypothesis:". No working yet — just the proactive prediction you will test.'
_EXEC_SYS='Execute the plan to CONFIRM or REFUTE the hypothesis. Before each step state what you EXPECT, then do it; if the result surprises you, that is a signal — chase it. Do each computation ONCE (no restating). Actively check why each tempting wrong option fails. End with "The answer is (X)." (X = the letter, if multiple-choice).'
_SWITCH_SYS='Your previous attempt STALLED (kept repeating without progress). Abandon that approach entirely and solve it a DIFFERENT way (e.g. work backward from the options, or estimate then refine). Be decisive. End with "The answer is (X)."'
_FALSIFY_SYS='You proposed an answer. Now actively try to BREAK it: find one concrete reason it could be wrong, and check it. If it survives, confirm; if it breaks, give the corrected answer. End with "The answer is (X)."'
_NEXT_SYS='You are a PROACTIVE collaborator. From the conversation, infer the user\'s larger GOAL (their north star), then list the {k} most likely NEXT STEPS they will want — concrete, ordered by likelihood, one line each, and tag each [prep] if you could prepare it in advance or [ask] if it needs their decision. Format: "GOAL: ...\\n1. ... [prep|ask]".'
class ProactiveCoT:
    def __init__(s,chat_fn,stall_thresh=0.34):
        s.chat=chat_fn;s.stall_thresh=stall_thresh
    def solve(s,question,context='',max_tokens=900):
        plan=s.chat(question,system=_PLAN_SYS,max_tokens=200,context=context)
        hyp=s.chat(f'{question}\n\nPlan:\n{plan}',system=_HYP_SYS,max_tokens=48,context=context)
        work=s.chat(f'{question}\n\nPlan:\n{plan}\n{hyp}',system=_EXEC_SYS,max_tokens=max_tokens,context=context)
        switched=False;st=stall_score(work)
        if st>=s.stall_thresh:
            switched=True;work=s.chat(f'{question}\n\nPlan:\n{plan}',system=_SWITCH_SYS,max_tokens=max_tokens,context=context)
        fal=s.chat(f'{question}\n\nProposed solution:\n{work[-600:]}',system=_FALSIFY_SYS,max_tokens=300,context=context)
        final=fal if _letter(fal) else work
        return {'plan':plan,'hypothesis':hyp,'work':work,'falsify':fal,'answer':_letter(final),'stalled':st,'switched':switched}
    def next_steps(s,conversation,k=3):
        return s.chat(conversation,system=_NEXT_SYS.format(k=k),max_tokens=240)
def _letter(t):
    t=t or ''
    for pat in (r'answer is \(?([A-J])\)?',r'(?:final answer|answer)\s*:?\s*\(?([A-J])\)?'):
        m=re.search(pat,t,re.I)
        if m:return m.group(1).upper()
    m=re.findall(r'\b([A-J])\b',t);return (m[-1].upper() if m else '')
if __name__=='__main__':
    SCRIPT={'plan':'GOAL: pick the right letter.\n1. identify principle\n2. compute\nShape: a small integer; trap: option that uses wrong sign.','hyp':'Hypothesis: (C)','exec':'I expect the field doubles.\nField = 2x.\nSo it is C.\nThe answer is (C).','stall':'The standard formula is X.\nThe standard formula is X.\nThe standard formula is X.\nThe standard formula is X.\nThe standard formula is X.','falsify':'Could it be D? No, D uses the wrong unit. C survives. The answer is (C).','next':'GOAL: get Adam topping the leaderboard.\n1. finish the pipeline-vs-bare run [prep]\n2. GPU-test ProactiveCoT vs reactive [prep]\n3. publish the number [ask]'}
    def mock(msg,system='',max_tokens=0,context=''):
        if 'PROACTIVE collaborator' in system:return SCRIPT['next']
        if 'Plan BEFORE' in system:return SCRIPT['plan']
        if 'FIRST GUESS' in system:return SCRIPT['hyp']
        if 'STALLED' in system:return SCRIPT['exec']
        if 'BREAK it' in system:return SCRIPT['falsify']
        return SCRIPT['stall']
    p=ProactiveCoT(mock)
    print('stall_score(repetitive) =',stall_score(SCRIPT['stall']),'-> triggers switch:',stall_score(SCRIPT['stall'])>=0.34)
    print('stall_score(progressing)=',stall_score(SCRIPT['exec']))
    r=p.solve('A physics MCQ...')
    print(f"solve: answer={r['answer']} stalled={r['stalled']} switched={r['switched']} (caught the spiral -> switched strategy)")
    print('next_steps(conversation):\n'+p.next_steps('we deployed adam, benchmarked it, now upgrading CoT'))
    print('PROACTIVE_OK' if r['answer']=='C' and r['switched'] else 'CHECK')
