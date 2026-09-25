import re
from amni.math.answer import evaluate,fmt
from amni.math.sympy_tool import run_sympy,solve_eq
CALC_RE=re.compile(r"⟦calc:\s*(.*?)⟧")
SYMPY_RE=re.compile(r"⟦sympy:\s*(.*?)⟧")
SOLVE_RE=re.compile(r"⟦solve:\s*(.*?)\s*\|\s*([A-Za-z_][A-Za-z0-9_]*)\s*⟧")
def _arith(expr:str)->str|None:
 try:v=evaluate(expr.replace("^","**"))
 except OverflowError:return ""
 except (ValueError,ZeroDivisionError,TypeError,SyntaxError,ArithmeticError):return None
 return None if isinstance(v,complex) else fmt(v,12).replace(',','').split(' ≈ ')[-1] if not isinstance(v,float) or abs(v)<1e15 else "%.12g"%v
def exec_calc(expr:str)->str|None:
 e=(expr or "").strip();a=_arith(e) if e and len(e)<=500 else None
 return a if a else (None if a=="" or not e or len(e)>500 else run_sympy(e))
def apply_protocol(text:str)->tuple[str,list]:
 s=text or "";tr=[]
 for rx,f,tag in ((SOLVE_RE,lambda m:solve_eq(m.group(1),m.group(2)),'solve'),(SYMPY_RE,lambda m:run_sympy(m.group(1)),'sympy'),(CALC_RE,lambda m:exec_calc(m.group(1)),'calc')):
  for m in rx.finditer(s):v=f(m);tr.append({"tag":tag,"expr":m.group(1),"value":v});s=s.replace(m.group(0),v if v is not None else m.group(0),1)
 return s,tr
