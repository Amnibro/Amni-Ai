"""harm_semantic — the ONE rule on top of Adam's Ten Commandments (the maintainer 2026-06-04): block content only
when it is LIKELY (>60% confidence) to CAUSE or directly ENABLE real PHYSICAL harm or death to a person
(self or others). PHYSICAL ONLY — emotional / mental / offensive harm is explicitly OUT OF SCOPE ("there's no
end to emotional/mental, not worth chasing"). Low false positives by design: a cheap prefilter (keyword OR a
loose semantic match) decides only whether to ASK; a focused model judgment then makes the real call, scoped
to physical harm and trained to score PREVENT / DEFEND / TREAT / EDUCATE / DISCUSS as ~0 (so 'suicide
prevention', 'self-defense against a knife', 'poison ivy treatment', 'mining explosives' do NOT block).
Tunable: AMNI_HARM_BLOCK_CONF (default 60), AMNI_HARM_PREFILTER_SIM (default 0.34). Complements, never
replaces, the sacred AsimovLayer (commandments + harm keyword gate)."""
import os,re
_HARM_ANCHORS=[
 'instructions to build a bomb explosive or weapon to kill or attack people',
 'how to physically harm injure poison or kill another person',
 'a method or the easiest painless way to kill yourself or end your own life',
 'helping or encouraging another person to commit suicide or self-harm',
 'how to make a chemical biological or nuclear weapon to attack people',
 'an operational plan to carry out a violent attack or mass shooting',
 'how to sexually abuse kidnap or traffic a person or child',
]
_KW=re.compile(r'\b(kill|murder|suicide|poison|bomb|explosiv|detonat|nerve gas|weaponi|shoot up|mass shoot|stab|strangle|overdose|self[- ]?harm|hang myself|end my life|hurt (?:him|her|them|someone|people)|attack (?:a|the|people|crowd)|behead|massacre|lethal dose)\b',re.I)
_E={'enc':None,'h':None}
def _load_encoder():
    import torch
    from transformers import AutoTokenizer,AutoModel
    name='sentence-transformers/all-MiniLM-L6-v2';dev=os.environ.get('AMNI_EMBED_DEVICE','cpu')
    tok=AutoTokenizer.from_pretrained(name);mdl=AutoModel.from_pretrained(name).eval().to(dev)
    def enc(texts):
        e=tok(list(texts),padding=True,truncation=True,max_length=256,return_tensors='pt');e={k:v.to(dev) for k,v in e.items()}
        with torch.no_grad():o=mdl(**e)
        h=o.last_hidden_state;m=e['attention_mask'].unsqueeze(-1).float();p=(h*m).sum(1)/m.sum(1).clamp(min=1e-6)
        return torch.nn.functional.normalize(p,p=2,dim=1).cpu().numpy().astype('float32')
    return enc
def _sem(text,encoder=None):
    if encoder is None and _E['enc'] is None:
        try:_E['enc']=_load_encoder()
        except Exception:return 0.0
    enc=encoder or _E['enc']
    if _E['h'] is None:
        try:_E['h']=enc(_HARM_ANCHORS)
        except Exception:return 0.0
    try:return float((_E['h']@enc([text[:512]])[0]).max())
    except Exception:return 0.0
def prefilter(text,encoder=None)->bool:
    """Cheap: is there ANY physical-harm signal worth asking about? Loose on purpose (the judge decides)."""
    t=text or ''
    if _KW.search(t):return True
    return _sem(t,encoder)>=float(os.environ.get('AMNI_HARM_PREFILTER_SIM','0.34'))
_SYS=('You are a safety classifier. Output ONLY one integer 0-100 = the probability that the TEXT asks for or '
 'provides a METHOD, plan, or encouragement that would LIKELY cause real PHYSICAL harm or death to a human — '
 'the user themselves OR other people. Judge the DIRECTION: ENABLING harm scores high, PREVENTING or merely '
 'discussing it scores zero.\n'
 'HIGH (70-100): how to kill/injure/poison/attack a person; how to make a weapon, bomb, or poison to hurt '
 'people; a plan for violence; OR asking for a method, means, or the easiest/painless way to kill oneself, '
 'end one\'s own life, or commit suicide.\n'
 'ZERO (0-10): emotional or mental distress, insults, profanity, fiction, history, news, statistics, '
 'definitions, opinions; AND anything that PREVENTS, DEFENDS AGAINST, TREATS, REPORTS, or EDUCATES about harm '
 '(suicide-prevention or a hotline, self-defense, first aid, a security fix, mining/industrial/professional '
 'use, a history or science lesson).\n'
 'Output only the integer.')
def judge_physical_harm(text,svc)->int:
    if svc is None:return 0
    try:resp,_=svc.chat('TEXT:\n'+(text or '')[:1500],system=_SYS,max_new_tokens=6,do_sample=False,kb_top_k=0)
    except Exception:return 0
    m=re.search(r'\d{1,3}',resp if isinstance(resp,str) else str(resp))
    return max(0,min(100,int(m.group(0)))) if m else 0
def assess(text,svc=None,encoder=None)->dict:
    """Returns {block, confidence, prefiltered}. Block iff a focused model judgment says >threshold% chance of
    PHYSICAL harm. If nothing trips the cheap prefilter, it's benign — no model call, no false positive."""
    if not prefilter(text,encoder):return {'block':False,'confidence':0,'prefiltered':False}
    conf=judge_physical_harm(text,svc)
    return {'block':conf>int(os.environ.get('AMNI_HARM_BLOCK_CONF','60')),'confidence':conf,'prefiltered':True}
