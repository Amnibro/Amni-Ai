import os,json,numpy as np,torch
from pathlib import Path
from transformers.generation.candidate_generator import PromptLookupCandidateGenerator
from amni.compute.reffelt4 import encode_ids_to_rgba2,decode_rgba2_to_ids
_FNV_OFF=0xcbf29ce484222325
_FNV_PRIME=0x100000001b3
_M64=0xFFFFFFFFFFFFFFFF
def fnv1a64(ids):
    h=_FNV_OFF
    for x in ids:
        x&=0xFFFFFFFF
        for _ in range(4):
            h=((h^(x&0xFF))*_FNV_PRIME)&_M64;x>>=8
    return h
class PTEXBlockBank:
    def __init__(self,bank_dir,tok,h_sizes=(12,8,6,4,3),k_max=None,min_h=3):
        self.bank_dir=bank_dir;self.tok=tok;self.h_sizes=tuple(sorted(h_sizes,reverse=True));self.min_h=min_h
        self.k_max=int(os.environ.get('AMNI_BLOCK_K','12')) if k_max is None else k_max
        self._rgba=[];self._toks=[];self._sig2off={};self._sig2ctx={};self._stats={}
        self.proposed_steps=0;self.accepted_tokens=0;self.max_toks=int(os.environ.get('AMNI_BLOCK_MAXTOK','2000000'))
        self._dirty=0;self._flush_every=int(os.environ.get('AMNI_BLOCK_FLUSH','256'));self.persist=os.environ.get('AMNI_BLOCK_PERSIST','1')=='1' and bool(bank_dir)
        self._min_tries=int(os.environ.get('AMNI_BLOCK_MINTRIES','20'));self._min_ratio=float(os.environ.get('AMNI_BLOCK_MINRATIO','0.10'))
        if self.persist:
            try:self.load()
            except Exception as e:print(f'[block-spec] bank load skipped: {e}',flush=True)
    def _store(self,block):
        if not block:return None
        arr=np.asarray(block,dtype=np.int64)
        rgba=encode_ids_to_rgba2(arr.astype(np.uint64))
        dec=decode_rgba2_to_ids(rgba,arr.size)
        off=len(self._toks)
        self._rgba.append(rgba);self._toks.extend(int(v) for v in dec.tolist())
        return off,arr.size
    def add_sequence(self,tokens):
        if len(self._toks)>=self.max_toks:return 0
        n=len(tokens);added=0
        for i in range(self.min_h,n):
            block=tokens[i:i+self.k_max]
            if len(block)<1:continue
            stored=None
            for H in self.h_sizes:
                if H>i or H<self.min_h:continue
                ctx=tuple(tokens[i-H:i]);sig=fnv1a64(ctx)
                if sig in self._sig2off and self._sig2ctx.get(sig)==ctx:continue
                stored=stored or self._store(block)
                if stored is None:break
                self._sig2off[sig]=stored;self._sig2ctx[sig]=ctx;self._stats.setdefault(sig,[0,0]);added+=1
        self._dirty+=added
        return added
    def save(self):
        if not self.bank_dir:return False
        self.prune()
        d=Path(self.bank_dir);d.mkdir(parents=True,exist_ok=True)
        rgba=np.concatenate(self._rgba,axis=0) if self._rgba else np.zeros((0,4),dtype=np.uint8)
        n_px=int(rgba.shape[0]);pw=4096;h=max((n_px+pw-1)//pw,1)
        page=np.zeros((h,pw,4),dtype=np.uint8);page.reshape(-1,4)[:n_px]=rgba
        np.save(str(d/'blocks.ptex.tmp'),page);os.replace(str(d/'blocks.ptex.tmp.npy'),str(d/'blocks.ptex.npy'))
        meta={'scheme':'blocktex_gf17_2px','n_pixels':n_px,'n_toks':len(self._toks),'page_w':pw,'sigs':[[int(s),int(o),int(l),list(self._sig2ctx[s]),int(self._stats.get(s,[0,0])[0]),int(self._stats.get(s,[0,0])[1])] for s,(o,l) in self._sig2off.items()]}
        (d/'block_bank.json.tmp').write_text(json.dumps(meta),encoding='utf-8');os.replace(str(d/'block_bank.json.tmp'),str(d/'block_bank.json'))
        self._dirty=0;return True
    def flush(self,force=False):
        return self.save() if self.bank_dir and (force or self._dirty>=self._flush_every) else False
    def load(self):
        d=Path(self.bank_dir);f=d/'block_bank.json';p=d/'blocks.ptex.npy'
        if not (f.exists() and p.exists()):return False
        meta=json.loads(f.read_text(encoding='utf-8'));n_px=int(meta['n_pixels'])
        rgba=np.ascontiguousarray(np.load(str(p),mmap_mode='r').reshape(-1,4)[:n_px],dtype=np.uint8)
        self._toks=[int(v) for v in decode_rgba2_to_ids(rgba,n_px//2).tolist()];self._rgba=[rgba] if n_px else []
        self._sig2off={};self._sig2ctx={};self._stats={}
        for row in meta.get('sigs',[]):
            s,o,l,ctx,pr,ac=row;s=int(s);self._sig2off[s]=(int(o),int(l));self._sig2ctx[s]=tuple(int(c) for c in ctx);self._stats[s]=[int(pr),int(ac)]
        return True
    def expected_gain_ok(self,sig):
        pr,ac=self._stats.get(sig,(0,0))
        return pr<self._min_tries or (ac/pr if pr else 1.0)>=self._min_ratio
    def lookup(self,tail):
        L=len(tail)
        for H in self.h_sizes:
            if H>=L or H<self.min_h:continue
            ctx=tuple(tail[-H:]);sig=fnv1a64(ctx);off=self._sig2off.get(sig)
            if off is not None and self._sig2ctx.get(sig)==ctx and self.expected_gain_ok(sig):
                o,ln=off;return sig,self._toks[o:o+ln]
        return None
    def prune(self):
        drop=[s for s,(pr,ac) in list(self._stats.items()) if pr>=self._min_tries and (ac/pr if pr else 1.0)<self._min_ratio and s in self._sig2off]
        for s in drop:self._sig2off.pop(s,None);self._sig2ctx.pop(s,None);self._stats.pop(s,None)
        return len(drop)
    def record_propose(self,sig,n):
        self.proposed_steps+=1;st=self._stats.setdefault(sig,[0,0]);st[0]+=int(n)
    def record_accept(self,sig,n):
        self.accepted_tokens+=int(n);st=self._stats.setdefault(sig,[0,0]);st[1]+=int(n)
class PTEXBlockCandidateGenerator(PromptLookupCandidateGenerator):
    def __init__(self,bank=None,**kw):
        super().__init__(**kw)
        self.bank=bank;self.k_max=int(self.num_output_tokens);self._ema=float(self.k_max);self._last_sig=None
    def get_candidates(self,input_ids):
        bsz,L=input_ids.shape
        if self.max_length==L+1:return input_ids,None
        self._last_sig=None
        try:
            hit=self.bank.lookup(input_ids[0].tolist()) if self.bank is not None else None
            if hit is not None:
                sig,block=hit;block=block[:self.num_output_tokens]
                if block:
                    chosen=torch.tensor(block,dtype=input_ids.dtype,device=input_ids.device)
                    if self.eos_token_id is not None:
                        nz=torch.nonzero(torch.isin(chosen,self.eos_token_id))
                        chosen=chosen[:int(nz[0].item())] if nz.numel()>0 else chosen
                    if self.logits_processor is not None and chosen.shape[0]>0:
                        seq=input_ids;fake=torch.ones((bsz,self.vocab_size),device=input_ids.device,dtype=torch.float32);keep=chosen.shape[0]
                        for ci,tk in enumerate(chosen.tolist()):
                            fl=self.logits_processor(seq,fake)[0,tk]
                            if fl in (-float('Inf'),torch.finfo(fl.dtype).min):keep=ci;break
                            seq=torch.cat((input_ids,chosen[:ci+1].unsqueeze(0)),dim=1)
                        chosen=chosen[:keep]
                    if chosen.shape[0]>0:
                        self._last_sig=sig;self.bank.record_propose(sig,int(chosen.shape[0]))
                        return torch.cat((input_ids,chosen.unsqueeze(0)),dim=1),None
        except Exception:
            self._last_sig=None
        return super().get_candidates(input_ids)
    def update_candidate_strategy(self,input_ids,scores,num_matches):
        nm=int(num_matches)
        if self._last_sig is not None:self.bank.record_accept(self._last_sig,nm)
        self._ema=0.9*self._ema+0.1*nm
        self.num_output_tokens=1 if nm==0 else max(1,min(self.k_max,round(2*self._ema)+1))
