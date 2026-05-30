import os,sys,json,time,argparse
os.environ.setdefault('HIP_VISIBLE_DEVICES','1')
os.environ.setdefault('PYTHONUNBUFFERED','1')
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(_ROOT))
import torch,torch.nn.functional as F
from amni.inference.streaming_chat import StreamingChatService
from amni.storage.ptex_memory import PtexMemoryAtlas
from amni.training.distill_targets import pack_topk
def load_corpus(p,limit):
    items=[]
    with open(p,encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:items.append(json.loads(line))
            if limit and len(items)>=limit:break
    return items
def vram_ok(dev,floor=0.05):
    return True if not(dev.startswith('cuda') and torch.cuda.is_available()) else (torch.cuda.mem_get_info()[0]/torch.cuda.mem_get_info()[1])>=floor
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--teacher-bake',default='bakes/granite41_8b_gf17');ap.add_argument('--model-path',default='downloaded_models/granite-4.1-8b')
    ap.add_argument('--corpus',default='data/distill_corpus_real_v2.jsonl');ap.add_argument('--out',default='experiences/distill_targets/granite8b_targets.ptex')
    ap.add_argument('--limit',type=int,default=0);ap.add_argument('--topk',type=int,default=64);ap.add_argument('--max-len',type=int,default=512);ap.add_argument('--budget-mb',type=int,default=3000)
    a=ap.parse_args();t0=time.time();print(f'[targets] teacher={a.teacher_bake} corpus={a.corpus} topk={a.topk} maxlen={a.max_len}',flush=True)
    svc=StreamingChatService(a.teacher_bake,a.model_path,budget_mb=a.budget_mb);tok=svc.tok;dev=svc.device
    atlas=PtexMemoryAtlas(a.out);start=len(atlas);items=load_corpus(a.corpus,a.limit)
    print(f'[targets] loaded {len(items)} items; resuming from {start}; device={dev}',flush=True);done=0
    for i in range(start,len(items)):
        if not vram_ok(dev):print(f'[targets] VRAM floor hit at {i}; stopping cleanly',flush=True);break
        e=items[i];sm=e.get('system','');usr=e.get('prompt') or e.get('user') or '';resp=e.get('response') or e.get('answer') or ''
        pre=([{'role':'system','content':sm}] if sm else [])+[{'role':'user','content':usr}]
        ptxt=tok.apply_chat_template(pre,tokenize=False,add_generation_prompt=True);ftxt=tok.apply_chat_template(pre+[{'role':'assistant','content':resp}],tokenize=False)
        pn=tok(ptxt,return_tensors='pt',add_special_tokens=False).input_ids.shape[1]
        ids=tok(ftxt,return_tensors='pt',add_special_tokens=False).input_ids[:,:a.max_len].to(dev);mask=torch.ones_like(ids)
        with torch.no_grad():logits=svc.model(input_ids=ids,attention_mask=mask).logits[0].float()
        vals,idx=torch.topk(F.log_softmax(logits,dim=-1),a.topk,dim=-1)
        atlas.append(pack_topk(idx.cpu().numpy(),vals.cpu().numpy()),meta={'i':i,'T':int(ids.shape[1]),'k':a.topk,'prefix_len':int(min(pn,ids.shape[1])),'cat':e.get('category','')});done+=1
        if done==1 or done%10==0:print(f'[targets] {i+1}/{len(items)} done={done} T={ids.shape[1]} rate={done/(time.time()-t0):.2f}/s vram={torch.cuda.memory_allocated()/1e9:.2f}GB',flush=True)
    print(f'[targets] DONE wrote {done} (atlas total {len(atlas)}) in {time.time()-t0:.1f}s -> {a.out}',flush=True)
if __name__=='__main__':main()
