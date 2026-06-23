import re
def _getbox(r):
    i=r.rfind('\\boxed')
    if i<0:return None
    j=r.find('{',i)
    if j<0:return None
    d=0
    for k in range(j,len(r)):
        d+=(r[k]=='{')-(r[k]=='}')
        if d==0:return r[j+1:k]
    return r[j+1:]
def _evalbox(s):
    if s is None:return None
    s=re.sub(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}',r'(\1)/(\2)',s)
    s=s.replace('\\times','*').replace('\\cdot','*').replace('\\pi','3.14159').replace('^','**').replace('\\%','').replace('%','').replace('$','')
    s=re.sub(r'\\[a-zA-Z]+','',s);s=re.sub(r'[^0-9+\-*/.()e ]','',s).strip()
    if not s:return None
    try:return round(float(eval(s,{'__builtins__':{}},{})),4)
    except Exception:
        n=re.findall(r'-?\d[\d.]+|\d',s)
        try:return float(n[-1]) if n else None
        except Exception:return None
def _lastnum(t):
    m=re.findall(r'-?\d[\d,]*\.?\d*',t.replace(',',''))
    try:return float(m[-1]) if m else None
    except Exception:return None
def solve_boxed(svc,q,max_tokens=600):
    prompts=[' Reason step by step. Give the final numeric answer inside \\boxed{ }.',' Solve it carefully. You MUST end with the single final numeric answer inside \\boxed{ } and nothing after.']
    r=''
    for pr in prompts:
        r,_=svc.chat(q+pr,max_new_tokens=max_tokens);v=_evalbox(_getbox(r))
        if v is not None:return v,r
    return _lastnum(r),r
