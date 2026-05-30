import os,sys,ast,argparse,time,random
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(_ROOT))
_SKIP={'test','tests','__pycache__','.git','node_modules','vendor','dist','build','.github'}
def _skip(p):return any(s in _SKIP for s in p.parts) or 'test' in p.stem.lower()
def extract(txt):
    out=[]
    try:tree=ast.parse(txt)
    except Exception:return out
    for n in ast.walk(tree):
        if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):continue
        if n.name.startswith('_'):continue
        src=ast.get_source_segment(txt,n)
        if not src or not (30<len(src)<3500):continue
        doc=(ast.get_docstring(n) or '').strip().split('\n')[0][:140]
        kind='class' if isinstance(n,ast.ClassDef) else 'function'
        out.append((f'Python {kind} {n.name}: {doc}' if doc else f'Python {kind} {n.name}',src))
    return out
def main():
    ap=argparse.ArgumentParser(description='Model-free: write a code corpus STRAIGHT into the semantic lesson PTEX as (description->code) recall pairs.')
    ap.add_argument('--corpus',required=True);ap.add_argument('--out',default='experiences/code_lessons.npz')
    ap.add_argument('--max',type=int,default=800);ap.add_argument('--max-file-bytes',type=int,default=60000)
    a=ap.parse_args();sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    os.environ.setdefault('AMNI_EMBED_DEVICE','cpu')
    from amni.inference.semantic_ptex_lut import SemanticPTEXLUT
    lut=SemanticPTEXLUT(grid=64,pca_dim=8)
    files=[p for p in Path(a.corpus).rglob('*.py') if not _skip(p)];random.seed(17);random.shuffle(files)
    seen=set();n=0
    for p in files:
        if n>=a.max:break
        try:txt=p.read_text(encoding='utf-8',errors='ignore')[:a.max_file_bytes]
        except Exception:continue
        for desc,src in extract(txt):
            k=desc.lower()
            if k in seen:continue
            seen.add(k);lut.add(desc,src);n+=1
            if n>=a.max:break
    print(f'[lessons] added {n} (description->code) pairs; fitting embeddings on CPU...',flush=True)
    t=time.time();lut.fit();print(f'[lessons] fit done in {time.time()-t:.0f}s',flush=True)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    lut.save(str(a.out).removesuffix('.npz'))
    print(f'[lessons] {lut.stats()} -> {a.out}  (point Adam at it via lessons_path, or merge into adam_lessons.npz)',flush=True)
    q='write a function to merge two sorted lists'
    hit=lut.lookup_soft(q,k=1)
    print(f'[lessons] sample recall for {q!r}: {(hit[0][0][:80] if hit else None)!r}',flush=True)
if __name__=='__main__':main()
