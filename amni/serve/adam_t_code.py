import os,re,ast,math,time,builtins,doctest,threading
from pathlib import Path
from amni.serve.self_debug import run_in_sandbox
CORP=Path(os.environ.get('AMNI_ADAM_T_CODE_CORPUS') or str(Path.home()/'ai'/'amni-code-corpus'/'Python'))
LIC='TheAlgorithms/Python (MIT License)'
DIRS=('maths','strings','sorts','searches','conversions','bit_manipulation','boolean_algebra','dynamic_programming','other','geometry','financial','physics','backtracking','divide_and_conquer','greedy_methods','data_structures','ciphers','hashes','matrix','linear_algebra')
OKMOD={'math','re','string','itertools','collections','functools','heapq','bisect','statistics','operator','fractions','decimal','typing','__future__'}
REQ=re.compile(r"(?i)^\W*(?:(?:hey|hi|ok(?:ay)?|so|please|adam)\W+)*(?:(?:can|could|would|will) you\s+)?(?:please\s+)?(?:write|implement|create|code|make|give me|show me|build|generate)\b(?:\s+(?:me|us))?(?:\s+(?:a|an|the|some))?\s*(?:(?:small|short|simple|quick|python|py)\s+)*(?:function|code|program|script|method|snippet|implementation|routine)\b|^\W*(?:a\s+)?python\s+(?:function|code|program|script)\s+(?:to|that|for|which)\b|^\W*(?:(?:can|could|would) you\s+)?implement\s+\w+")
NOT=re.compile(r"(?i)\b(?:fix|debug|bug|error|wrong|broken|doesn'?t work|what does|explain|why does)\b|\bdef\s+\w+\s*\(|\b(?:javascript|java|rust|golang|c\+\+|c#|typescript|ruby|php|sql|html|css|bash|shell)\b")
_B=set(dir(builtins))
_ST=set('a an the of to in and or is are be that this it for with as by on from at python function code program implement implementation write create returns return make given whether check checks can you please me my i we us some which will takes take using use into way method snippet script simple quick short how many much all any each there get gets does do give gives compute computes calculate calculates find finds determine determines given'.split())
_sm=lambda w:w[:-3]+'y' if w.endswith('ies') and len(w)>4 else w[:-2] if w.endswith(('ches','shes','xes','sses')) else w[:-1] if w.endswith('s') and not w.endswith(('ss','us','is')) and len(w)>3 else w
_W=lambda t:[_sm(w) for w in re.findall(r"[a-z]+",re.sub(r"([a-z])([A-Z])",r"\1 \2",t).replace('_',' ').lower()) if w not in _ST]
_IX={};_L=threading.Lock()
def want(m:str)->bool:return bool(REQ.search(m or '')) and not NOT.search(m or '')
def _lit(n)->bool:
    try:ast.literal_eval(n);return True
    except Exception:return False
def _free(fn,imp)->set:
    A={n.id for n in ast.walk(fn) if isinstance(n,ast.Name) and not isinstance(n.ctx,ast.Load)}|{a.arg for a in ast.walk(fn) if isinstance(a,ast.arg)}|{n.name for n in ast.walk(fn) if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.ExceptHandler)) and n.name}|{al.asname or al.name.split('.')[0] for n in ast.walk(fn) if isinstance(n,(ast.Import,ast.ImportFrom)) for al in n.names}
    return {n.id for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}-A-_B-set(imp)-{fn.name}
def _file(f:Path,d:str)->list:
    try:src=f.read_text(encoding='utf-8');T=ast.parse(src)
    except Exception:return []
    imp={}
    for n in T.body:
        isinstance(n,ast.Import) and [imp.__setitem__(a.asname or a.name.split('.')[0],f"import {a.name}"+(f" as {a.asname}" if a.asname else '')) for a in n.names if a.name.split('.')[0] in OKMOD]
        isinstance(n,ast.ImportFrom) and n.module and not n.level and n.module.split('.')[0] in OKMOD and [imp.__setitem__(a.asname or a.name,f"from {n.module} import {a.name}"+(f" as {a.asname}" if a.asname else '')) for a in n.names]
    R=[]
    for fn in T.body:
        if not isinstance(fn,ast.FunctionDef) or fn.decorator_list or fn.name.startswith('_') or fn.args.vararg or fn.args.kwarg:continue
        doc=ast.get_docstring(fn) or '';code=ast.get_source_segment(src,fn) or ''
        if not doc or not code.isascii() or code.count('\n')>60 or _free(fn,imp):continue
        ex=[]
        for e in doctest.DocTestParser().get_examples(doc):
            try:c=ast.parse(e.source.strip(),mode='eval').body;ok=isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id==fn.name and not c.keywords and all(_lit(a) for a in c.args) and _lit(ast.parse(e.want.strip(),mode='eval').body)
            except Exception:ok=False
            ok and len(e.source)<200 and len(e.want)<120 and ex.append((e.source.strip(),e.want.strip()))
        ln={n.id for n in ast.walk(fn) if isinstance(n,ast.Name)}|{n.value.id for n in ast.walk(fn) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name)}
        len(ex)>=2 and R.append({'name':fn.name,'params':[a.arg for a in fn.args.args],'doc':re.sub(r"\s+",' ',doc.split('\n\n')[0]).strip(),'code':code,'imports':[v for k,v in imp.items() if k in ln],'ex':ex[:8],'path':f"{d}/{f.relative_to(CORP/d)}",'words':_W(fn.name+' '+fn.name+' '+doc.split('\n\n')[0]+' '+f.stem)})
    return R
def index()->dict:
    with _L:
        if 'r' in _IX:return _IX
        R=[x for d in DIRS if (CORP/d).is_dir() for f in sorted((CORP/d).rglob('*.py')) if not f.name.startswith(('test_','__')) for x in _file(f,d)];df={}
        [df.__setitem__(w,df.get(w,0)+1) for r in R for w in set(r['words'])];N=max(1,len(R))
        _IX.update(r=R,idf={w:math.log(1+(N-c+0.5)/(c+0.5)) for w,c in df.items()},avg=sum(len(r['words']) for r in R)/N);return _IX
def _ty(m:str)->str:l=m.lower();return 'list' if re.search(r"\b(?:list|array|numbers|items|elements)\b",l) else 'str' if re.search(r"\b(?:string|strings|word|words|sentence|text|character|characters|letters?|palindrome|anagrams?)\b",l) else 'int' if re.search(r"\b(?:number|integer|int|year|n)\b",l) else ''
def _aty(r:dict)->str:
    try:a=ast.parse(r['ex'][0][0],mode='eval').body.args;v=ast.literal_eval(a[0]) if a else None
    except Exception:return ''
    return 'list' if isinstance(v,(list,tuple)) else 'str' if isinstance(v,str) else 'int' if isinstance(v,(int,float)) and not isinstance(v,bool) else ''
def find(m:str,k:int=4)->list:
    X=index();q=list(dict.fromkeys(_W(re.sub(r"(?i)\bin python\b",'',m))));ty=_ty(m);tw=2 if re.search(r"(?i)\btwo\b|\bboth\b|\bpair\b",m) else 0;S=[];mx=max(X['idf'].values(),default=1.0);qs=set(q)
    for r in X['r']:
        tf={};[tf.__setitem__(w,tf.get(w,0)+1) for w in r['words']];nw=set(_W(r['name']));wr=set(r['words']);s=sum(X['idf'].get(w,0)*tf[w]*2.2/(tf[w]+1.2*(0.25+0.75*len(r['words'])/X['avg'])) for w in q if w in tf)
        s=s+1.5*(nw<=qs)+0.5*(r['path'].rsplit('/',1)[-1]==r['name']+'.py')-float(os.environ.get('AMNI_ADAM_T_CODE_XN','1.0'))*len(nw-qs)-1.0*(bool(tw) and len(r['params'])!=tw);cv=sum(X['idf'].get(w,mx) for w in q if w in wr)/max(1e-9,sum(X['idf'].get(w,mx) for w in q))
        dw=set(_W(r['doc']));s=s+3.0*cv;s>0 and len(nw&qs)>=0.5*len(nw) and (nw<=qs or len(dw&qs)>=float(os.environ.get('AMNI_ADAM_T_CODE_DP','0.3'))*len(dw)) and not (ty and _aty(r) not in ('',ty)) and cv>=float(os.environ.get('AMNI_ADAM_T_CODE_COV','0.6')) and S.append((s,r))
    S=sorted(S,key=lambda z:-z[0]);return [r for s,r in S[:k] if s>=max(float(os.environ.get('AMNI_ADAM_T_CODE_MIN','2.0')),0.5*S[0][0])] if S else []
def _sb(src:str,h:str)->str:
    r=run_in_sandbox(src,harness=h,timeout=int(float(os.environ.get('AMNI_ADAM_T_CODE_TIMEOUT','8'))));return (r.get('stdout') or '') if r.get('ran') else ''
def check(code:str,ex:list,imports=())->bool:return 'H4OK' in _sb('\n'.join(list(imports)+[code]),'\n'.join(f"assert ({s}) == ({w})" for s,w in ex)+"\nprint('H4OK')")
def _mut(v)->list:return [] if isinstance(v,bool) else [v+1,v-1,0,1,2,7,10,97,1234,-3] if isinstance(v,int) else [v*2,-v,0.5,0.0] if isinstance(v,float) else ['',v[:1] or 'a',v[::-1],v+v,v.upper(),'Hello World','abcba'] if isinstance(v,str) else [[],v[:1],list(v)[::-1],sorted(v) if all(isinstance(x,(int,float)) for x in v) else list(v),list(v)+list(v)[:2]] if isinstance(v,(list,tuple)) else []
def probes(ex:list)->list:
    P=[]
    for s,_ in ex[:2]:
        a=[ast.literal_eval(x) for x in ast.parse(s,mode='eval').body.args];P+=[tuple(a[:i]+[y]+a[i+1:]) for i,v in enumerate(a) for y in _mut(v)]
    return list(dict.fromkeys(repr(p) for p in P))[:16]
def agree(code:str,r:dict)->bool:
    t=ast.parse(r['code']);fn=t.body[0];[setattr(n,'id','ref_'+r['name']) for n in ast.walk(fn) if isinstance(n,ast.Name) and n.id==r['name']];fn.name='ref_'+r['name'];P=probes(r['ex'])
    h="def _h4(f,a):\n    try:\n        return repr(f(*a))\n    except Exception as e:\n        return 'E'+type(e).__name__\n"+f"print('H4AG',sum(_h4({r['name']},a)!=_h4(ref_{r['name']},a) for a in [{','.join(P)}]))"
    return bool(re.search(r"H4AG 0\b",_sb('\n'.join(list(r['imports'])+[code,ast.unparse(fn)]),h))) if P else False
def _task(m:str)->str:return re.sub(r"\s+",' ',re.sub(r"(?i)^\W*(?:(?:hey|hi|ok(?:ay)?|so|please|adam)\W+)*(?:(?:can|could|would|will) you\s+)?(?:please\s+)?(?:write|implement|create|code|make|give me|show me|build|generate)?\s*(?:me\s+)?(?:a|an|the|some)?\s*(?:python\s+)?(?:function|code|program|script)?\s*",'',m)).strip(' ?.!') or m
def _clean(r:dict)->str:
    fn=ast.parse(r['code']).body[0];b=fn.body[1:] if fn.body and isinstance(fn.body[0],ast.Expr) and isinstance(getattr(fn.body[0],'value',None),ast.Constant) and isinstance(fn.body[0].value.value,str) else fn.body;fn.body=[ast.Expr(ast.Constant(r['doc']))]+(b or [ast.Pass()]);return '\n'.join(list(r['imports'])+[ast.unparse(fn)])
def solve(m:str)->dict:
    t0=time.perf_counter();C=find(m);TS=__import__('amni.compute.tile_synth',fromlist=['solve']);bud=float(os.environ.get('AMNI_ADAM_T_CODE_BUDGET','4'));task=_task(m);ms=lambda:round((time.perf_counter()-t0)*1000)
    for r in C[:int(os.environ.get('AMNI_ADAM_T_CODE_K','3'))]:
        if not check(r['code'],r['ex'],r['imports']):continue
        stub=f"def {r['name']}({', '.join(r['params'])}):\n    \"\"\"{task.replace(chr(34),chr(39))}.\"\"\"\n"
        try:o=TS.solve(stub,[f"{s} == {w}" for s,w in r['ex']],budget=bud)
        except Exception:o={}
        code=f"def {r['name']}({', '.join(r['params'])}):\n{o['body'].rstrip()}" if o.get('found') and o.get('body') else ''
        if code and not re.search(r"\b_[a-z]\w*\(",code) and check(code,r['ex']) and agree(code,r):return {'ok':True,'how':'composed','code':code,'ex':r['ex'],'ref':r['path'],'probes':len(probes(r['ex'])),'prov':[f"tile_synth compose-and-verify: {len(r['ex'])} examples and {len(probes(r['ex']))} differential probes against {LIC}: {r['path']}"],'ms':ms()}
        return {'ok':True,'how':'retrieved','code':_clean(r),'ex':r['ex'],'ref':r['path'],'probes':0,'prov':[f"{LIC}: {r['path']}"],'ms':ms()}
    return {'ok':False,'how':'none','code':'','ex':[],'ref':C[0]['path'] if C else '','probes':0,'prov':[],'ms':ms()}
def reply(o:dict)->str:
    e=o['ex'][0] if o['ex'] else None
    return (f"Here's a Python function for that:\n\n```python\n{o['code']}\n```\n\nI checked it against {len(o['ex'])} examples and {o['probes']} extra inputs, for instance `{e[0]}` gives `{e[1]}`." if o['how']=='composed' else f"Here's a Python function for that, adapted from {LIC}:\n\n```python\n{o['code']}\n```\n\nIt passes its {len(o['ex'])} examples, for instance `{e[0]}` gives `{e[1]}`.") if o['ok'] else "I couldn't put together a function I could verify for that, so I'd rather not guess. If you give me the function name and a couple of example inputs and outputs, I can try again."
