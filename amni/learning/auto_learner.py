import json,re,time,sys,gc
from pathlib import Path
from typing import List,Optional,Iterable
import numpy as np,torch,torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModelForCausalLM,AutoTokenizer,get_linear_schedule_with_warmup
from amni.learning.gf17_writer import LearningWriter,WeightAccessError,AsimovProtectedError
class _DistillDataset(Dataset):
    def __init__(self,records,tokenizer,max_len=384):
        self.records=records;self.tok=tokenizer;self.max_len=max_len
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        sys_msg=r.get('system','')
        prompt=r.get('prompt','')
        response=r.get('response','')
        full=(sys_msg+'\n\n'+prompt+'\n'+response).strip() if sys_msg else (prompt+'\n'+response).strip()
        ids=self.tok(full,truncation=True,max_length=self.max_len,return_tensors='pt')['input_ids'][0]
        return {'input_ids':ids,'labels':ids.clone()}
def _collate(batch,pad_id):
    max_len=max(b['input_ids'].size(0) for b in batch)
    input_ids=torch.full((len(batch),max_len),pad_id,dtype=torch.long)
    labels=torch.full((len(batch),max_len),-100,dtype=torch.long)
    attn=torch.zeros((len(batch),max_len),dtype=torch.long)
    for i,b in enumerate(batch):
        l=b['input_ids'].size(0)
        input_ids[i,:l]=b['input_ids']
        labels[i,:l]=b['labels']
        attn[i,:l]=1
    return {'input_ids':input_ids,'labels':labels,'attention_mask':attn}
class ResidualSFTLearner:
    def __init__(self,bake_dir,model_path,trainable_layer_min=None,verbose=True):
        self.bake_dir=Path(bake_dir);self.model_path=Path(model_path)
        self.verbose=verbose
        self.writer=LearningWriter(bake_dir)
        self.trainable_layer_min=trainable_layer_min
        if not self.writer.list_immutable():
            protected=self.writer.auto_protect_foundational()
            if verbose:print(f'[learner] auto-protected {len(protected)} foundational tensors')
        self.immutable_names=set(self.writer.list_immutable())
        self.model=None;self.tokenizer=None;self.opt=None
    def load_model(self,device='cuda',dtype=torch.bfloat16):
        if self.verbose:print(f'[learner] loading {self.model_path} dtype={dtype}')
        self.tokenizer=AutoTokenizer.from_pretrained(str(self.model_path))
        self.model=AutoModelForCausalLM.from_pretrained(str(self.model_path),torch_dtype=dtype).to(device)
        self.model.config.pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        return self.model
    def apply_residuals_to_model(self):
        if self.model is None:raise RuntimeError('call load_model first')
        residual_tensors=self.writer.list_residual_tensors()
        if not residual_tensors:
            if self.verbose:print('[learner] no residuals to apply (clean baseline start)')
            return 0
        if self.verbose:print(f'[learner] applying {len(residual_tensors)} residuals to in-memory model')
        applied=0
        for name,param in self.model.named_parameters():
            if name not in residual_tensors:continue
            info=self.writer.tensor_info(name)
            n=int(info['n_pixels'])
            po=info['plane_offsets']
            base_path=self.writer._path(info)
            base_arr=np.fromfile(base_path,dtype=np.uint8)
            res_path=self.writer._residual_path(info,create=False)
            if res_path is None:continue
            res_arr=np.fromfile(res_path,dtype=np.uint8)
            d=[(base_arr[int(po[k]):int(po[k])+n].astype(np.uint16)+res_arr[int(po[k]):int(po[k])+n].astype(np.uint16))%17 for k in ('d0','d1','d2','d3')]
            u16=(d[0].astype(np.uint32)+d[1].astype(np.uint32)*17+d[2].astype(np.uint32)*289+d[3].astype(np.uint32)*4913).astype(np.uint16)
            sd=info.get('source_dtype','float16')
            t=torch.from_numpy(u16.view(np.float16).copy())
            if sd=='bfloat16':t=t.view(torch.bfloat16)
            t=t.reshape(info['shape']).to(param.device)
            with torch.no_grad():param.data.copy_(t)
            applied+=1
        if self.verbose:print(f'[learner] applied residuals to {applied} tensors')
        return applied
    def configure_trainable(self):
        if self.model is None:raise RuntimeError('call load_model first')
        n_frozen=0;n_trainable=0
        layer_re=re.compile(r'model\.layers\.(\d+)\.')
        for name,param in self.model.named_parameters():
            if name in self.immutable_names:param.requires_grad=False;n_frozen+=1;continue
            if self.trainable_layer_min is not None:
                m=layer_re.search(name)
                if m is not None and int(m.group(1))<self.trainable_layer_min:
                    param.requires_grad=False;n_frozen+=1;continue
            param.requires_grad=True;n_trainable+=1
        if self.verbose:print(f'[learner] frozen={n_frozen} trainable={n_trainable}')
        return n_trainable
    def train_on_corpus(self,corpus_records,epochs=1,batch_size=1,grad_accum=8,lr=2e-6,max_len=384,warmup_steps=10,nan_skip=True,log_every=20):
        if self.model is None:raise RuntimeError('call load_model first')
        self.configure_trainable()
        ds=_DistillDataset(corpus_records,self.tokenizer,max_len=max_len)
        pad_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        loader=DataLoader(ds,batch_size=batch_size,shuffle=True,collate_fn=lambda b:_collate(b,pad_id))
        total_steps=max(1,(len(loader)//grad_accum)*epochs)
        opt=torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],lr=lr,betas=(0.9,0.95),weight_decay=0.01)
        sched=get_linear_schedule_with_warmup(opt,warmup_steps,total_steps)
        if self.verbose:print(f'[learner] dataset n={len(ds)} total_steps={total_steps} lr={lr}')
        self.model.train()
        t0=time.time();step=0;running_loss=0.0;ns=0;skipped=0
        for ep in range(epochs):
            ep_loss=0.0;ep_n=0;opt.zero_grad()
            for bi,batch in enumerate(loader):
                batch={k:v.to('cuda') for k,v in batch.items()}
                out=self.model(**batch)
                loss=out.loss/grad_accum
                lv=float(loss.item())*grad_accum
                if nan_skip and not (lv==lv and abs(lv)<1e6):
                    opt.zero_grad();skipped+=1
                    if self.verbose and (bi+1)%log_every==0:print(f'[learner][ep {ep+1}/{epochs}][batch {bi+1}/{len(loader)}] SKIP nan/inf')
                    continue
                loss.backward()
                ep_loss+=lv;ep_n+=1;running_loss+=lv;ns+=1
                if (bi+1)%grad_accum==0:
                    torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.0)
                    opt.step();sched.step();opt.zero_grad();step+=1
                if self.verbose and (bi+1)%log_every==0:
                    vram=torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
                    print(f'[learner][ep {ep+1}/{epochs}][batch {bi+1}/{len(loader)}] loss={lv:.4f} avg={running_loss/max(1,ns):.4f} vram={vram:.2f}GB step={step}/{total_steps} skipped={skipped}')
            if self.verbose:print(f'[learner][ep {ep+1}/{epochs}] DONE avg_loss={ep_loss/max(1,ep_n):.4f} wall={time.time()-t0:.1f}s')
        self.opt=opt
        return {'wall':time.time()-t0,'final_avg_loss':ep_loss/max(1,ep_n),'skipped':skipped,'steps':step}
    def encode_trained_as_residuals(self,additive=False,subject='global'):
        if self.model is None:raise RuntimeError('call train_on_corpus first')
        self.model.eval()
        manifest=self.writer.manifest['tensors']
        n_encoded=0;n_skipped=0;n_immutable=0
        for name,param in self.model.named_parameters():
            if name in self.immutable_names:n_immutable+=1;continue
            if name not in manifest:n_skipped+=1;continue
            if not param.requires_grad:n_skipped+=1;continue
            target=param.detach().cpu().to(torch.float32).numpy()
            try:
                self.writer.encode_target_array_as_residuals(name,target,additive=additive,subject=subject)
                n_encoded+=1
            except (WeightAccessError,AsimovProtectedError) as e:
                n_skipped+=1
                if self.verbose:print(f'[learner] skip {name}: {e}')
        if self.verbose:print(f'[learner] encoded {n_encoded} tensors as residuals (skipped {n_skipped} non-trainable, {n_immutable} immutable)')
        return n_encoded
    def train_from_atlas(self,atlas,outcomes_filter=(1,),subject=None,**kwargs):
        records=atlas.to_records_list(outcomes_filter=set(outcomes_filter) if outcomes_filter else None)
        if not records:raise RuntimeError(f'atlas {atlas.subject} has no matching records (outcomes_filter={outcomes_filter})')
        if self.verbose:print(f'[learner] distilling {len(records)} experiences from atlas (subject={atlas.subject}, outcomes={outcomes_filter})')
        stats=self.train_on_corpus(records,**kwargs)
        target_subject=subject if subject else atlas.subject
        if self.verbose:print(f'[learner] encoding residuals under subject={target_subject!r} (must match bench --subjects to be visible at inference)')
        n_encoded=self.encode_trained_as_residuals(subject=target_subject)
        return stats,n_encoded
    def shutdown(self):
        if self.model is not None:
            del self.model;self.model=None
        if self.opt is not None:
            del self.opt;self.opt=None
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
def read_jsonl(path,limit=None):
    out=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:continue
            out.append(json.loads(line))
            if limit and len(out)>=limit:break
    return out
