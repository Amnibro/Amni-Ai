import os,sys,json,time,argparse
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(_ROOT))
def main():
    ap=argparse.ArgumentParser(description='Bake a warm ADAM-SPEC block bank from a corpus of common coding prompts (pure-greedy continuations).')
    ap.add_argument('--bake',default='bakes/granite41_3b_gf17')
    ap.add_argument('--corpus',required=True,help='json file: {"corpus":[...]} or a bare [...] of prompts')
    ap.add_argument('--out',default=None,help='bank dir (default: configured block_bank)')
    ap.add_argument('--max-new',type=int,default=140)
    ap.add_argument('--budget-mb',type=int,default=8000)
    a=ap.parse_args()
    os.environ['AMNI_BLOCK_SPEC']='1';os.environ['AMNI_HIP_GEMV_ON']='0';os.environ['AMNI_BLOCK_PERSIST']='1'
    if a.out:os.environ['AMNI_BLOCK_BANK']=a.out
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    import torch
    from amni.inference.streaming_chat import StreamingChatService
    from amni.inference.streaming_linear import StreamingLinear
    raw=json.loads(Path(a.corpus).read_text(encoding='utf-8'))
    prompts=raw['corpus'] if isinstance(raw,dict) and 'corpus' in raw else (raw if isinstance(raw,list) else [])
    if not prompts:print('[seed] empty corpus, nothing to bake');return 1
    svc=StreamingChatService(a.bake,a.bake,budget_mb=a.budget_mb);dev=svc.device
    reg=next((m.registry for _,m in svc.model.named_modules() if isinstance(m,StreamingLinear)),None)
    if reg is not None:reg.autosize_budget(cap_bytes=int(14*1024**3));reg.pin_hot();reg.warmup()
    bank=svc._block_bank
    if bank is None:print('[seed] block bank inactive (need Granite + AMNI_BLOCK_SPEC)');return 1
    t0=time.time()
    for i,p in enumerate(prompts):
        try:
            txt=svc.tok.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True)
            ids=svc.tok(txt,return_tensors='pt',add_special_tokens=False).input_ids.to(dev)
            with torch.no_grad():o=svc.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=a.max_new,do_sample=False,pad_token_id=svc.tok.pad_token_id)
            bank.add_sequence(o[0].tolist())
        except Exception as e:print(f'  [warn] prompt {i} failed: {repr(e)[:120]}',flush=True)
        if (i+1)%20==0:print(f'  seeded {i+1}/{len(prompts)} -> {len(bank._toks)} toks, {len(bank._sig2off)} sigs',flush=True)
    bank.save()
    print(f'[seed] baked {len(prompts)} prompts -> {len(bank._toks)} toks, {len(bank._sig2off)} sigs in {time.time()-t0:.0f}s -> {a.out or bank.bank_dir}',flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
