import json,re
from amni.serve.skills import _sympy_jail
_SAFE=re.compile(r"^(?!.*__)[0-9A-Za-z\s+\-*/^%().,=<>!]+$")
JOB=("import json,sympy as sp\nfrom sympy.parsing.sympy_parser import parse_expr,standard_transformations as S,implicit_multiplication_application as I,convert_xor as X\n"
"j=json.loads(J);L={k:sp.Symbol(k) for k in j['vars']};P=lambda s:parse_expr(s,transformations=S+(I,X),local_dict=L);v=j['verb']\n"
"def n(x):\n try:c=complex(sp.N(x,30));return [c.real,c.imag]\n except Exception:return None\n"
"if v=='solve':\n E=[sp.Eq(P(a),P(b)) for a,b in j['eqs']];R=sp.solve(E,[L[k] for k in j['vars']],dict=True)\n print(json.dumps({'sols':[{str(k):[str(w),n(w)] for k,w in d.items()} for d in R]}))\n"
"elif v=='eval':\n e=P(j['body']);print(json.dumps({'res':str(sp.N(e,50)) if not e.free_symbols else str(sp.simplify(e)),'num':n(e) if not e.free_symbols else None,'ok':True}))\n"
"else:\n f=P(j['body']);x=L[j['vars'][0]] if j['vars'] else sp.Symbol('x');r={'diff':lambda:sp.diff(f,x),'integrate':lambda:sp.integrate(f,x),'factor':lambda:sp.factor(f),'expand':lambda:sp.expand(f),'simplify':lambda:sp.simplify(f)}[v]()\n"
" ok={'diff':lambda:sp.simplify(sp.integrate(r,x)-f).diff(x)==0,'integrate':lambda:sp.simplify(sp.diff(r,x)-f)==0,'factor':lambda:sp.expand(r-f)==0,'expand':lambda:sp.expand(r-f)==0,'simplify':lambda:sp.simplify(r-f)==0}[v]()\n"
" print(json.dumps({'res':str(r).replace('**','^'),'ok':bool(ok)}))\n")
def job(verb:str,vars_:list,body:str=None,eqs:list=None,timeout:int=8):
 parts=([body] if body is not None else [])+[y for e in (eqs or []) for y in e]
 if not parts or any(not isinstance(x,str) or len(x)>500 or not _SAFE.match(x) for x in parts) or not all(re.fullmatch(r"[A-Za-z]",v) for v in vars_):return None
 out=_sympy_jail(JOB,{'verb':verb,'vars':list(vars_),'body':body,'eqs':eqs},timeout=timeout)
 try:return json.loads(out) if out else None
 except ValueError:return None
def _norm(v)->str|None:
 s=str(v).strip().replace(',','') if v is not None else ''
 if not s or s.lower() in ('zoo','nan','oo','-oo','none'):return None
 try:f=float(s);return str(int(round(f))) if abs(f-round(f))<1e-8 else ("%.12g"%f)
 except (ValueError,OverflowError):return s if len(s)<=300 else None
def _vars(e:str)->list:return sorted(set(re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])",e))-{'E','I'})
def run_sympy(expr:str,timeout:float=8.0)->str|None:
 e=(expr or '').strip();d=job('eval',_vars(e),body=e,timeout=int(timeout)) if e else None;return _norm(d['res']) if d and d.get('ok') else None
def solve_eq(equation:str,var:str="x",timeout:float=8.0)->str|None:
 eq=(equation or '').strip();l,_,r=eq.partition('=');d=job('solve',[var],eqs=[[l.strip(),r.strip() or '0']],timeout=int(timeout)) if eq and re.fullmatch(r"[A-Za-z]",var or '') else None
 rs=[x[var][1][0] if x[var][1] and abs(x[var][1][1])<1e-12 else x[var][0] for x in (d or {}).get('sols',[]) if var in x]
 return ', '.join(_norm(x) for x in sorted(rs,key=lambda y:(isinstance(y,str),-y if not isinstance(y,str) and y<0 else 0,y if not isinstance(y,str) else 0))) if rs else None
def sympy_eval(expr:str)->str|None:return run_sympy(expr)
