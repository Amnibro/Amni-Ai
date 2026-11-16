"""GraniteAtexChatService — GPU-resident int4-group ATEX svc for the granite-3B bake (bakes/granite41_3b_gf17_atex). Same .chat() interface as StreamingChatService / Nvfp4AtexChatService so AdamLoop runs unchanged. The 3B served like the 12B: compact native int4-group format RESIDENT on the GPU (~1.7GB), decode-on-kernel (int4grp_gemv) — NOT bf16-materialize-and-stream. Applies granite logits_scaling (GraniteForCausalLM divides logits by it; we call lm_head directly so we must). Loads resident, GEMM prefill + GEMV decode, greedy/sample, EOS stop."""
import torch,torch.nn as nn,json,os,sys,glob
sys.path.insert(0,'.');os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF','expandable_segments:True')
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from accelerate import init_empty_weights
from safetensors import safe_open
try:from amni.inference.int4_group_gemv import int4grp_gemv,int4grp_gemm,int4grp_gemv_packed,int4grp_gemm_packed
except Exception:int4grp_gemv=int4grp_gemm=int4grp_gemv_packed=int4grp_gemm_packed=None  # triton missing -> CPU dequant path below
from amni.utils.base_models import chat_template_kwargs,strip_thinking
GS=128
def _dequant_linear(codes,scale,inf,out,gs=GS):
    """CPU fallback: int4-group codes (one code per byte, +8 offset) * per-group scale -> dense bf16 nn.Linear. Used when no GPU / no triton."""
    c=codes.reshape(out,-1)[:,:inf].to(torch.int16)-8
    w=(c.reshape(out,-1,gs).float()*scale.float().reshape(out,-1,1)).reshape(out,inf).to(torch.bfloat16)
    lin=nn.Linear(inf,out,bias=False,device='cpu',dtype=torch.bfloat16);lin.weight=nn.Parameter(w,requires_grad=False);return lin
class AtexLin(nn.Module):
    def __init__(s,codes,scale,inf,out,packed=False):
        super().__init__();s.packed=packed
        s.register_buffer('codes',codes.contiguous() if packed else codes.reshape(out,-1)[:,:inf].contiguous())
        s.register_buffer('scale',scale);s.inf=inf;s.out=out;s.register_buffer('_y',torch.empty(out,device=codes.device,dtype=torch.float16))
    def forward(s,x):
        flat=x.reshape(-1,x.shape[-1])
        if s.packed:
            if flat.shape[0]==1:
                int4grp_gemv_packed(s.codes,s.scale,flat[0].half(),y=s._y);return s._y.reshape(*x.shape[:-1],s.out).to(x.dtype)
            return int4grp_gemm_packed(s.codes,s.scale,flat.half()).reshape(*x.shape[:-1],s.out).to(x.dtype)
        if flat.shape[0]==1:
            int4grp_gemv(s.codes,s.scale,flat[0].half(),y=s._y);return s._y.reshape(*x.shape[:-1],s.out).to(x.dtype)
        return int4grp_gemm(s.codes,s.scale,flat.half()).reshape(*x.shape[:-1],s.out).to(x.dtype)
class GraniteAtexChatService:
    def __init__(s,bake=None,tok_src=None,device=None):
        if bake is None:bake=next((b for b in ('bakes/granite42_3b_gf17_atex','bakes/granite41_3b_gf17_atex') if os.path.exists(b+'/bake_manifest.json')),'bakes/granite42_3b_gf17_atex')
        man=json.load(open(bake+'/bake_manifest.json'));s.tok=AutoTokenizer.from_pretrained(tok_src or bake);cfg=AutoConfig.from_pretrained(bake)
        s.device=device or ('cuda' if (torch.cuda.is_available() and int4grp_gemv is not None) else 'cpu');dev=s.device
        if dev=='cpu' and man.get('packed',False):raise RuntimeError('packed int4 ATEX bake needs the GPU triton kernels; re-bake unpacked (scripts/bake_granite_atex.py) for CPU use')
        s.logits_scale=float(getattr(cfg,'logits_scaling',1.0) or 1.0)
        with init_empty_weights():m=AutoModelForCausalLM.from_config(cfg).eval()
        def par(k):
            p=m;ps=k.split('.')
            for q in ps[:-1]:p=getattr(p,q)
            return p,ps[-1]
        tens={}
        for fp in sorted(glob.glob(bake+'/*.safetensors')):
            f=safe_open(fp,framework='pt')
            for k in f.keys():tens[k]=f
        g=lambda k:tens[k].get_tensor(k)
        for k,info in man['tensors'].items():
            try:
                if info['q']==1:
                    p2,mn=par(k[:-len('.weight')])
                    if dev=='cpu':setattr(p2,mn,_dequant_linear(g(k+'.codes'),g(k+'.scale'),info['inf'],info['shape'][0],gs=int(man.get('gs',GS))));continue
                    codes=g(k+'.codes').cuda();scale=g(k+'.scale').cuda()
                    setattr(p2,mn,AtexLin(codes,scale,info['inf'],info['shape'][0],packed=man.get('packed',False)).cuda())
                else:
                    p2,a=par(k);tt=g(k);tt=tt.to(torch.bfloat16).to(dev) if tt.is_floating_point() else tt.to(dev)
                    if a in p2._parameters and p2._parameters[a] is not None:p2._parameters[a]=nn.Parameter(tt,requires_grad=False)
                    else:p2._buffers[a]=tt
            except (AttributeError,KeyError):pass
        if getattr(cfg,'tie_word_embeddings',False):  # granite-4.1 ties; granite-4.2 / qwen3.8 ship a separate lm_head
            try:m.tie_weights()
            except Exception:pass
        for mod in m.modules():
            for nm,buf in list(mod._buffers.items()):
                if buf is not None and (buf.is_meta or buf.device.type!=torch.device(dev).type):
                    if buf.is_meta and nm=='inv_freq' and hasattr(mod,'rope_init_fn'):
                        inv,sc=mod.rope_init_fn(mod.config,dev);mod._buffers[nm]=inv;mod.attention_scaling=sc
                    elif buf.is_meta and nm=='inv_freq' and hasattr(mod,'compute_default_rope_parameters'):
                        inv,sc=mod.compute_default_rope_parameters(mod.config,dev);mod._buffers[nm]=inv;mod.attention_scaling=sc
                    elif nm=='original_inv_freq' and 'inv_freq' in mod._buffers:mod._buffers[nm]=mod._buffers['inv_freq'].clone()
                    else:mod._buffers[nm]=buf.to(dev)
        s.m=m;s.model=m;s.lm=m.model
        gcp=bake+'/generation_config.json';e=s.tok.eos_token_id
        if os.path.exists(gcp):e=json.load(open(gcp)).get('eos_token_id',e)
        s.eos_ids=set(int(x) for x in (e if isinstance(e,list) else [e]) if x is not None)
    def _gen(s,ids,mx,do_sample,temp=0.7,no_repeat=0):
        gen=[];ng={}
        with torch.no_grad():
            out=s.lm(input_ids=ids,use_cache=True);cache=out.past_key_values
            for _ in range(mx):
                lg=(s.m.lm_head(out.last_hidden_state[:,-1:])[:,-1].float())/s.logits_scale
                if no_repeat and len(gen)>=no_repeat-1:
                    for t in ng.get(tuple(gen[-(no_repeat-1):]),()):lg[0,t]=-1e9
                nt=torch.multinomial(torch.softmax(lg/temp,-1),1) if do_sample else lg.argmax(-1,keepdim=True)
                tid=int(nt.item())
                if tid in s.eos_ids:break
                if no_repeat and len(gen)>=no_repeat-1:ng.setdefault(tuple(gen[-(no_repeat-1):]),set()).add(tid)
                gen.append(tid)
                out=s.lm(input_ids=nt,past_key_values=cache,use_cache=True);cache=out.past_key_values
        return s.tok.decode(gen,skip_special_tokens=True),len(gen)
    def chat(s,user_msg,system=None,history=None,facts=None,max_new_tokens=80,do_sample=False,subject=None,kb_top_k=3,**kw):
        c=user_msg
        if facts:c=str(facts).strip()+'\n\n'+c
        if system:c=system.strip()+'\n\n'+c
        msgs=[]
        if history:
            for h in history:
                if isinstance(h,(list,tuple)) and len(h)==2:msgs+=[{'role':'user','content':str(h[0])},{'role':'assistant','content':str(h[1])}]
        msgs.append({'role':'user','content':c})
        prompt=s.tok.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False,**chat_template_kwargs(s.tok))
        ids=s.tok(prompt,return_tensors='pt',add_special_tokens=False).input_ids.to(s.device)
        txt,n=s._gen(ids,max_new_tokens,do_sample,temp=float(kw.get('temperature',0.7)),no_repeat=kw.get('no_repeat',0))
        return strip_thinking(txt),n
    def chat_stream(s,*a,**k):
        t,n=s.chat(*a,**k);yield t
if __name__=='__main__':
    import time;t=time.time();svc=GraniteAtexChatService();print(f'loaded {time.time()-t:.0f}s on {svc.device}'+(f' resv{torch.cuda.memory_reserved()/1e9:.2f}G' if svc.device=='cuda' else ''),flush=True)
    for q in ['A car accelerates from rest at 3 m/s^2 for 4 s. Final velocity in m/s? Only the number.','What is 17*23?']:
        r,n=svc.chat(q,max_new_tokens=40);print(f'Q:{q[:40]} -> [{n}tok] {r[:70]!r}',flush=True)
    print('GRANITE_ATEX_SVC_OK',flush=True)
