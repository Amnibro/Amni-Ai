import os,sys,json,time,argparse
os.environ.setdefault('HIP_VISIBLE_DEVICES','1');os.environ.setdefault('PYTHONIOENCODING','utf-8')
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(_ROOT))
import torch
from amni.inference.streaming_chat import StreamingChatService
from amni.inference.streaming_linear import StreamingLinear
PROMPTS=['Write a Python function that returns whether a number is prime.','Explain what a binary search tree is, in two sentences.','List three benefits of regular exercise.','In one sentence, what is photosynthesis?']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bake',default='bakes/granite41_3b_gf17');ap.add_argument('--model-path',default='bakes/granite41_3b_gf17');ap.add_argument('--m0',type=int,default=0);ap.add_argument('--cap-gb',type=float,default=14.0);ap.add_argument('--budget-mb',type=int,default=4000);ap.add_argument('--gen',type=int,default=64);ap.add_argument('--prefill',type=int,default=4);ap.add_argument('--n',type=int,default=3);a=ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    hip=os.environ.get('AMNI_HIP_GEMV_ON','0')=='1'
    print(f'[cfg] bake={a.bake} m0={a.m0} hip={hip} budget_mb={a.budget_mb} cap_gb={a.cap_gb} gen={a.gen}',flush=True)
    t0=time.time();svc=StreamingChatService(a.bake,a.model_path,budget_mb=a.budget_mb);dev=svc.device
    reg=next((m.registry for _,m in svc.model.named_modules() if isinstance(m,StreamingLinear)),None)
    print(f'[load] {time.time()-t0:.1f}s device={dev} registry={reg is not None}',flush=True)
    if a.m0 and reg is not None:
        b=reg.autosize_budget(cap_bytes=int(a.cap_gb*1024**3));reg.pin_hot()
        tw=time.time();reg.warmup();print(f'[m0] autosize_budget={b/1e9:.2f}GB pinned warmup={time.time()-tw:.1f}s',flush=True)
    def gen(p,g):
        ids=svc.tok(svc.tok.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True),return_tensors='pt',add_special_tokens=False).input_ids.to(dev)
        torch.cuda.synchronize() if dev.startswith('cuda') else None;t=time.time()
        with torch.no_grad():out=svc.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=g,do_sample=False,pad_token_id=svc.tok.pad_token_id)
        torch.cuda.synchronize() if dev.startswith('cuda') else None;dt=time.time()-t
        return svc.tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True),out.shape[1]-ids.shape[1],dt
    gen(PROMPTS[0],a.prefill)
    st0=reg.stats() if reg is not None else {}
    print(f'[warm] resident={st0.get("resident_bytes",0)/1e9:.2f}GB evictions={st0.get("evictions",0)}',flush=True)
    tps=[]
    for p in PROMPTS[:a.n]:
        ev0=reg.stats()['evictions'] if reg is not None else 0
        tp,_,dtp=gen(p,a.prefill);txt,new,dt=gen(p,a.gen)
        ev=(reg.stats()['evictions'] if reg is not None else 0)-ev0
        dec=max(1e-6,dt-dtp);decn=max(1,new-a.prefill);r=decn/dec;tps.append(r)
        print(f'[tokps] decode={r:.2f} tok/s  (gen{new} {dt:.2f}s - prefill{a.prefill} {dtp:.2f}s) evict={ev}',flush=True)
        print('  Q: '+p,flush=True);print('  A: '+txt.replace(chr(10),' / '),flush=True)
    print(f'[summary] m0={a.m0} hip={hip} mean_decode_tok/s={sum(tps)/max(1,len(tps)):.2f} evictions={reg.stats()["evictions"] if reg is not None else 0}',flush=True)
if __name__=='__main__':main()
