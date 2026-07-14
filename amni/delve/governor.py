"""Roundtable governor: hard CLI budgets, escalation ladder, and convergence early-stop so multi-agent beats solo without spiraling subscription spend. No secrets. Pure helpers + Budget counter."""
import re
_WORD=re.compile(r"[a-z0-9]+",re.I)
_VERDICT_STOP=re.compile(r"VERDICT:\s*(DONE|AGREE|CONVERGED|CONSENSUS|STAND)\b",re.I)
_VERDICT_OPEN=re.compile(r"VERDICT:\s*(DISAGREE|CONFLICT|OPEN|REVISE|CONCEDE)\b",re.I)
_COUNCIL=re.compile(r"\b(prove|debate|benchmark|measure|ablation|compare|vs\.?|trade-?off|disagree|wrong|bug|regress|verify|falsif|challenge|cross-?exam)\b",re.I)
_TOOLS=re.compile(r"\b(run|test|pytest|bench|profile|code|implement|fix|patch|kernel|gpu|hip|triton|probe|measure|build|debug|reproduce|numbers?)\b",re.I)
_CHATTY=re.compile(r"\b(hi|hello|thanks|thank you|lol|ok|okay|what do you think|opinion|feel)\b",re.I)
def tokens(t):return set(_WORD.findall((t or "").lower()))
def jaccard(a,b):
    sa,sb=tokens(a),tokens(b)
    if not sa and not sb:return 1.0
    if not sa or not sb:return 0.0
    return len(sa&sb)/len(sa|sb)
def claim_blob(text):
    t=text or "";m=re.search(r"##\s*Claim\s*\n(.+?)(?:\n##|\Z)",t,re.I|re.S)
    return (m.group(1).strip() if m else t)[:800]
def agree(texts,threshold=0.55):
    blobs=[claim_blob(t) for t in texts if (t or "").strip()]
    if len(blobs)<2:return False
    scores=[jaccard(blobs[i],blobs[j]) for i in range(len(blobs)) for j in range(i+1,len(blobs))]
    return (sum(scores)/len(scores))>=float(threshold) if scores else False
def all_stop_verdicts(texts):
    good=[t for t in texts if (t or "").strip()]
    if len(good)<2:return False
    if any(_VERDICT_OPEN.search(t or "") for t in good):return False
    return all(_VERDICT_STOP.search(t or "") for t in good)
def needs_council(msg):return bool(_COUNCIL.search(msg or ""))
def needs_tools(msg):return bool(_TOOLS.search(msg or ""))
def looks_chatty(msg):
    m=(msg or "").strip()
    return len(m)<80 and bool(_CHATTY.search(m)) and not needs_tools(m) and not needs_council(m)
def escalate(msg,cfg=None):
    cfg=cfg or {}
    mode=str(cfg.get("escalate") or "auto").lower()
    if mode in ("always","council","full"):return "council"
    if mode in ("never","off","solo"):return "solo"
    if mode in ("pair","consult"):return "pair"
    if needs_council(msg):return "council"
    if needs_tools(msg):return "pair"
    if looks_chatty(msg):return "solo"
    return "pair" if len((msg or "").split())>24 else "solo"
def pick_names(roster,level,order_prefer=None):
    names=list(roster or [])
    if not names:return []
    if level=="council":return names
    adam=[n for n in names if n=="Adam"]
    cli=[n for n in names if n!="Adam"]
    if level=="solo":return adam or names[:1]
    primary=cli[0] if cli else None
    if order_prefer:
        for p in order_prefer:
            for n in cli:
                if n.lower()==str(p).lower() or n.lower().startswith(str(p).lower()):primary=n;break
            if primary:break
    out=[]
    if adam:out+=adam
    if primary:out+=[primary]
    return out or names[:1]
class Budget:
    def __init__(s,exchange_limit=6,session_limit=48):
        s.exchange_limit=max(0,int(exchange_limit or 0));s.session_limit=max(0,int(session_limit or 0))
        s.exchange=0;s.session=0;s.skipped=0
    def reset_exchange(s):s.exchange=0;s.skipped=0
    def reset_session(s):s.session=0;s.exchange=0;s.skipped=0
    def allow(s):
        if s.exchange_limit and s.exchange>=s.exchange_limit:return False
        if s.session_limit and s.session>=s.session_limit:return False
        return True
    def charge(s):
        s.exchange+=1;s.session+=1
    def deny(s):
        s.skipped+=1
    def snap(s):
        return {"exchange":s.exchange,"session":s.session,"exchange_limit":s.exchange_limit,"session_limit":s.session_limit,"skipped":s.skipped,"ok":s.allow()}
