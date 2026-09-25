from amni.math.relate import solve_word
def solve(problem:str,min_conf:float=0.0)->tuple[str,list]|None:
 r=solve_word(problem or '') if (problem or '').strip() else None
 return None if r is None or r['conf']<min_conf else (r['value_str'] if 'value_str' in r else (str(r['value'].numerator) if r['value'].denominator==1 else "%.12g"%float(r['value'])),[{'path':r.get('path','word'),'conf':round(r['conf'],3),'hedged':r['conf']<0.6,'reply':r['reply'],'trace':r.get('trace')}])
