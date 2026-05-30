import os,sys,argparse,time,random
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(_ROOT))
_SKIP_DIR={'test','tests','__pycache__','.git','node_modules','vendor','dist','build','docs','doc','.github'}
def gather(root,exts,max_bytes):
    out=[]
    for p in Path(root).rglob('*'):
        if not p.is_file() or p.suffix not in exts:continue
        if any(s in _SKIP_DIR for s in p.parts):continue
        if 'test' in p.stem.lower() or 'conftest' in p.stem.lower():continue
        try:
            if p.stat().st_size>max_bytes*4 or p.stat().st_size<40:continue
        except OSError:continue
        out.append(p)
    return out
def main():
    ap=argparse.ArgumentParser(description='Model-free: pack a permissive code corpus into an ADAM-SPEC block bank (tokenizer only, no model).')
    ap.add_argument('--bake',default='bakes/granite41_3b_gf17')
    ap.add_argument('--corpus',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--exts',default='.py,.js')
    ap.add_argument('--max-sigs',type=int,default=800000)
    ap.add_argument('--max-file-bytes',type=int,default=24000)
    ap.add_argument('--max-file-toks',type=int,default=3000)
    a=ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    os.environ['AMNI_BLOCK_PERSIST']='0';os.environ['AMNI_BLOCK_MAXSIGS']=str(a.max_sigs);os.environ['AMNI_BLOCK_MAXTOK']=str(a.max_sigs)
    from transformers import AutoTokenizer
    from amni.inference.block_speculator import PTEXBlockBank
    tok=AutoTokenizer.from_pretrained(a.bake)
    bank=PTEXBlockBank(a.out,tok,h_sizes=(8,6,4))
    exts=tuple(e if e.startswith('.') else '.'+e for e in a.exts.split(','))
    files=gather(a.corpus,exts,a.max_file_bytes);random.seed(17);random.shuffle(files)
    print(f'[pack] {len(files)} candidate files under {a.corpus} ({exts}); cap {a.max_sigs} sigs',flush=True)
    t0=time.time();nf=0;skip=0
    for p in files:
        if len(bank._sig2off)>=a.max_sigs:break
        try:
            txt=p.read_text(encoding='utf-8',errors='ignore')[:a.max_file_bytes]
            ids=tok(txt,add_special_tokens=False).input_ids[:a.max_file_toks]
            if len(ids)<10:skip+=1;continue
            bank.add_sequence(ids);nf+=1
        except Exception:skip+=1;continue
        if nf%150==0:print(f'  {nf} files | {len(bank._sig2off)} sigs | {len(bank._toks)} toks',flush=True)
    bank.save()
    print(f'[pack] DONE {nf} files ({skip} skipped) -> {len(bank._sig2off)} sigs, {len(bank._toks)} toks in {time.time()-t0:.0f}s',flush=True)
    print(f'[pack] bank at {a.out} (point a server at it via AMNI_BLOCK_BANK)',flush=True)
if __name__=='__main__':main()
