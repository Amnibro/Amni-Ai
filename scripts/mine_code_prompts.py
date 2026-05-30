import os,sys,ast,re,json,argparse,random
from pathlib import Path
_SKIP={'test','tests','__pycache__','.git','node_modules','vendor','dist','build','.github'}
_JS_FN=re.compile(r'(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)')
_JS_ARROW=re.compile(r'(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>')
def _skip(p):return any(s in _SKIP for s in p.parts) or 'test' in p.stem.lower()
def py_prompts(txt):
    out=[]
    try:tree=ast.parse(txt)
    except Exception:return out
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            if n.name.startswith('_'):continue
            args=', '.join(a.arg for a in n.args.args if a.arg!='self')
            doc=(ast.get_docstring(n) or '').strip().split('\n')[0][:120]
            out.append(f'Write a Python function {n.name}({args}) that {doc[0].lower()+doc[1:]}' if doc else f'Write a Python function {n.name}({args}).')
        elif isinstance(n,ast.ClassDef):
            doc=(ast.get_docstring(n) or '').strip().split('\n')[0][:120]
            out.append(f'Write a Python class {n.name}: {doc}' if doc else f'Write a Python class {n.name}.')
    return out
def js_prompts(txt):
    out=[]
    for m in _JS_FN.finditer(txt):
        if not m.group(1).startswith('_'):out.append(f'Write a JavaScript function {m.group(1)}({m.group(2).strip()}).')
    for m in _JS_ARROW.finditer(txt):
        if not m.group(1).startswith('_'):out.append(f'Write a JavaScript function {m.group(1)}({m.group(2).strip()}).')
    return out
def main():
    ap=argparse.ArgumentParser(description='Model-free: mine realistic coding prompts from a code corpus (feeds seed_block_bank.py / live server).')
    ap.add_argument('--corpus',required=True);ap.add_argument('--out',default='_code_prompts.json')
    ap.add_argument('--max',type=int,default=4000);ap.add_argument('--max-file-bytes',type=int,default=60000)
    a=ap.parse_args();sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    seen=set();prompts=[]
    files=[p for p in Path(a.corpus).rglob('*') if p.is_file() and p.suffix in ('.py','.js') and not _skip(p)]
    random.seed(17);random.shuffle(files)
    for p in files:
        if len(prompts)>=a.max:break
        try:txt=p.read_text(encoding='utf-8',errors='ignore')[:a.max_file_bytes]
        except Exception:continue
        for pr in (py_prompts(txt) if p.suffix=='.py' else js_prompts(txt)):
            k=pr.lower().strip()
            if 12<len(pr)<200 and k not in seen:seen.add(k);prompts.append(pr)
            if len(prompts)>=a.max:break
    json.dump({'corpus':prompts},open(a.out,'w',encoding='utf-8'))
    print(f'[mine] {len(prompts)} unique coding prompts from {len(files)} files -> {a.out}',flush=True)
    for pr in prompts[:8]:print('   -',pr[:100],flush=True)
if __name__=='__main__':main()
