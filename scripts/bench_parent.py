"""bench_parent — run the UNQUANTIZED parent (gemma-4-E2B-it) through the IDENTICAL harness (same format_prompt,
same items, same greedy decoding, same extraction/scoring) as Adam, to get the only valid comparison: does the
GF17 bake match its own parent? Published model-card numbers are different-harness + prev-gen and are NOT comparable.
This loads the parent via transformers — HEAVY (bf16 ~10GB VRAM). Run in a CLEAN GPU window (Adam stopped; ideally
Azno/Chat paused). Persona-free raw continuation to match Adam's /complete path; code uses the same instruct preamble
prepended (persona-free) so both arms share one prompting path.
Usage:
  python scripts/bench_parent.py --model E:/Amni-Ai-Models/gemma-4-E2B-it --suites mmlu_pro math500 humanevalplus --limit 20 --out eval_reports/parent_control_2026-05-28.json"""
import sys,json,argparse,re,tempfile,os,subprocess,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.eval import modern_bench as mb
from amni.eval import harness_config as hc
_CODE_PREAMBLE='Complete this Python function. Output the COMPLETE function definition including the signature line and any needed imports, correctly indented. No prose, no markdown fences.\n\n'
def _load(model_path):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model_path)
    mdl=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).cuda().eval()
    return mdl,tok
def _make_gen(mdl,tok,max_tokens):
    import torch
    def g(prompt):
        ids=tok(prompt,return_tensors='pt').to('cuda')
        with torch.no_grad():out=mdl.generate(**ids,max_new_tokens=max_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids['input_ids'].shape[1]:],skip_special_tokens=True)
    return g
def _strip_code(t):t=re.sub(r'^```\w*\s*\n?','',t.strip());return re.sub(r'\n?```\s*$','',t)
def _extract_function(body,ep):
    imports=[l for l in body.split('\n') if l.startswith(('import ','from '))]
    idx=body.find(f'def {ep}')
    if idx<0:return None
    lines=body[idx:].split('\n');block=[lines[0]]
    for l in lines[1:]:
        if l.strip()=='':block.append('');continue
        if (len(l)-len(l.lstrip()))>0:block.append(l)
        else:break
    return ('\n'.join(imports)+'\n' if imports else '')+'\n'.join(block).rstrip()
def _exec_code(item,completion,timeout=10):
    body=_strip_code(completion);ep=item.get('entry_point','');test=item.get('test','')
    fn=_extract_function(body,ep)
    if fn is None:return False
    pimp='\n'.join(l for l in item['prompt'].split('\n') if l.startswith(('import ','from ')))
    src=(pimp+'\n' if pimp else '')+fn+'\n'+test+(f'\ncheck({ep})\n' if 'def check' in test else '')
    p=None
    try:
        with tempfile.NamedTemporaryFile('w',suffix='.py',delete=False,encoding='utf-8') as f:f.write(src);p=f.name
        return subprocess.run([sys.executable,'-B',p],capture_output=True,timeout=timeout,text=True).returncode==0
    except Exception:return False
    finally:
        if p:
            try:os.unlink(p)
            except Exception:pass
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='E:/Amni-Ai-Models/gemma-4-E2B-it')
    ap.add_argument('--suites',nargs='+',default=['mmlu_pro','math500','humanevalplus'])
    ap.add_argument('--limit',type=int,default=20)
    ap.add_argument('--out',default=None)
    a=ap.parse_args()
    print(f'[parent] loading {a.model} (bf16, ~10GB VRAM) ...',flush=True)
    mdl,tok=_load(a.model)
    print('[parent] loaded',flush=True)
    scores={};full={}
    for s in a.suites:
        cfg=hc.PER_BENCH.get(s,{'max_tokens':512,'kind':'mcq'})
        gen0=_make_gen(mdl,tok,cfg['max_tokens'])
        is_code=cfg['kind']=='code'
        gen=(lambda p,g=gen0:g(_CODE_PREAMBLE+p)) if is_code else gen0
        exec_fn=_exec_code if is_code else None
        print(f'[parent] {s} (limit={a.limit}, kind={cfg["kind"]}) ...',flush=True)
        t0=time.time();r=mb.run_suite(s,gen,limit=a.limit,exec_fn=exec_fn);full[s]=r
        if r['n']>0:scores[s]=r['accuracy'];print(f'[parent] {s}: {r["accuracy"]}% ({r["correct"]}/{r["n"]}) in {time.time()-t0:.0f}s',flush=True)
    print();print(mb.leaderboard_controlled({},scores,control_label='Gemma-4-E2B parent (bf16, this harness)'));print()
    if a.out:
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        Path(a.out).write_text(json.dumps({'scores':scores,'full':full,'model':a.model,'harness':hc.stamp()},indent=2,default=str),encoding='utf-8')
        print(f'[parent] wrote {a.out}',flush=True)
if __name__=='__main__':main()
