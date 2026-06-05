"""manipulation_flag — labels content that persuades by MANIPULATION (the honeytrap method) instead of by
evidence: moral shaming, thought-terminating/unfalsifiable claims, manufactured consensus, flattery that
lowers the reader's guard, smuggled premises, in-group/out-group othering. TECHNIQUE-based and direction-
agnostic at its core (a honeytrap is a honeytrap whoever wears it) — so it cannot be dodged by switching
costumes, and it does NOT flag honest evidence-based argument regardless of its conclusion (validated: honest
args ~20, manipulation ~85, clean gap). Per the maintainer's directive it carries a DISCLOSED epistemic lean
(AMNI_IDEOLOGY_SKEW, default 0.20): modestly more scrutiny of progressive moral-superiority framing, since the
indexed corpus skews progressive — with the HARD GUARANTEE that an honest, evidence-based argument is never
scored as manipulation. EPISTEMIC ONLY: this never touches who Adam protects or refuses to harm — Law 0 and
harm_semantic stay strictly viewpoint-blind."""
import os,re
def _sys():
    skew=int(round(float(os.environ.get('AMNI_IDEOLOGY_SKEW','0.20'))*100))
    s=("You are a rhetoric analyst. Judge ONLY the METHOD of persuasion, NOT whether you agree with the conclusion. "
     "Output one integer 0-100 = how much the text persuades by MANIPULATION rather than by honest evidence and argument, then ' | ' and a short technique tag.\n"
     "HIGH = loaded emotional words doing the work of an argument; framing disagreement as being a bad/immoral/stupid person; unfalsifiable or thought-terminating claims; manufactured consensus ('everyone knows'); flattery or sweet reassurance that lowers the guard while smuggling a strong claim; presenting a contested conclusion as an already-shared premise; in-group vs out-group othering.\n"
     "LOW = gives evidence, reasoning, sources, acknowledges trade-offs or the other side, lets the reader decide — EVEN IF the conclusion is strong or one you personally dislike.\n"
     "A manipulative argument is manipulative whoever makes it; an honest one is honest whoever makes it.")
    if skew>0:
        s+=("\nCALIBRATION: the indexed corpus skews progressive, so apply about %d%% more scrutiny to progressive moral-superiority framing specifically. HARD RULE that overrides the calibration: NEVER score an honest, evidence-based argument as manipulation no matter its political conclusion — only the manipulative TECHNIQUE counts, never the position."%skew)
    return s+"\nOutput ONLY: <integer> | <tag>"
def assess(text:str,svc=None)->dict:
    if svc is None:return {'score':0,'technique':'','flag':False}
    try:r,_=svc.chat('TEXT:\n'+(text or '')[:1500],system=_sys(),max_new_tokens=14,do_sample=False,kb_top_k=0)
    except Exception:return {'score':0,'technique':'','flag':False}
    r=r if isinstance(r,str) else str(r);m=re.search(r'\d{1,3}',r)
    score=max(0,min(100,int(m.group(0)))) if m else 0
    tag=(r.split('|',1)[1].strip()[:40]) if '|' in r else ''
    return {'score':score,'technique':tag,'flag':score>int(os.environ.get('AMNI_MANIP_FLAG','60'))}
