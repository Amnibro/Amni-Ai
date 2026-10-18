"""QwenAtexChatService — one serve path for Qwen3 (target id Qwen/Qwen3-8B).

Load paths (no new compression — existing int4-group GS=128 / Gf17Atex contract):
  1. Gf17Atex bake dir: bake_manifest.json + *.safetensors with q=1 `.codes` + `.scale`
     (same contract as Gf17AtexChatService / GraniteAtexChatService).
  2. Plain HF checkpoint dir (or hub id): pack Linear weights at load with the
     granite bake_granite_atex recipe (int4-group GS=128). Embed + lm_head stay bf16.

.chat() / .chat_stream() match GraniteAtex so AdamLoop/eval swap unchanged.
Device: .cuda() when CUDA/HIP is available (ROCm exposes the CUDA API); CPU
construction is allowed for CI. Triton gemv/gemm is skipped when there is no GPU.

Serve (after Kimahri writes the bake):
  python scripts/amni_serve.py --bake bakes/qwen3_8b_gf17_atex_probe
  python scripts/amni_serve.py --bake bakes/qwen3_8b_gf17_atex_probe --model bakes/qwen3_8b_gf17_atex_probe
"""
import torch,torch.nn as nn,torch.nn.functional as F,json,os,sys,glob
from pathlib import Path
sys.path.insert(0,'.');os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF','expandable_segments:True')
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from accelerate import init_empty_weights
from safetensors import safe_open
from amni.inference.atex_select import is_gf17_atex_bake
from amni.inference.streaming_chat import _GDN_ARCHS
GS=128
_HF_SKIP=('lm_head','embed')
_DEFAULT_BAKE='bakes/qwen3_8b_gf17_atex_probe'
_DEFAULT_HF='Qwen/Qwen3-8B'
try:
    from amni.inference.int4_group_gemv import int4grp_gemv,int4grp_gemm,int4grp_gemv_packed,int4grp_gemm_packed
except Exception:
    int4grp_gemv=int4grp_gemm=int4grp_gemv_packed=int4grp_gemm_packed=None

def _cuda_ok():
    try:return bool(torch.cuda.is_available())
    except Exception:return False

def _to_dev(t):
    return t.cuda() if _cuda_ok() else t

class AtexLin(nn.Module):
    """Same int4-group Linear as Granite/Gf17Atex (codes 0-15 offset +8, GS=128)."""
    def __init__(s,codes,scale,inf,out,packed=False):
        super().__init__();s.packed=packed
        s.register_buffer('codes',codes.contiguous() if packed else codes.reshape(out,-1)[:,:inf].contiguous())
        s.register_buffer('scale',scale);s.inf=inf;s.out=out
        s.register_buffer('_y',torch.empty(out,device=codes.device,dtype=torch.float16))
    def dequant(s):
        """CPU-safe decode of the Gf17Atex codes/scale contract (not a new scheme)."""
        c=s.codes
        if s.packed:
            q=torch.empty(s.out,c.shape[1]*2,dtype=torch.int16,device=c.device)
            q[:,0::2]=(c&0xF).to(torch.int16)-8;q[:,1::2]=((c>>4)&0xF).to(torch.int16)-8
            inn=q.shape[1];ng=s.scale.shape[-1];gs=max(1,inn//ng)
            W=(q.reshape(s.out,-1,gs).float()*s.scale.unsqueeze(-1).float()).reshape(s.out,-1)
            return W[:,:s.inf]
        inn=c.shape[1];ng=int(s.scale.shape[-1]);gs=max(1,(inn+ng-1)//ng)
        pad=(gs*ng)-inn
        cf=F.pad(c.to(torch.int16),(0,pad)) if pad else c.to(torch.int16)
        W=((cf.reshape(s.out,ng,gs).float()-8.0)*s.scale.unsqueeze(-1).float()).reshape(s.out,-1)
        return W[:,:s.inf]
    def forward(s,x):
        flat=x.reshape(-1,x.shape[-1])
        use_k=_cuda_ok() and s.codes.device.type=='cuda' and int4grp_gemv is not None
        if not use_k:
            y=F.linear(flat.float(),s.dequant().to(dtype=torch.float32),None).to(x.dtype)
            return y.reshape(*x.shape[:-1],s.out)
        if s.packed:
            if flat.shape[0]==1:
                int4grp_gemv_packed(s.codes,s.scale,flat[0].half(),y=s._y);return s._y.reshape(*x.shape[:-1],s.out).to(x.dtype)
            return int4grp_gemm_packed(s.codes,s.scale,flat.half()).reshape(*x.shape[:-1],s.out).to(x.dtype)
        if flat.shape[0]==1:
            int4grp_gemv(s.codes,s.scale,flat[0].half(),y=s._y);return s._y.reshape(*x.shape[:-1],s.out).to(x.dtype)
        return int4grp_gemm(s.codes,s.scale,flat.half()).reshape(*x.shape[:-1],s.out).to(x.dtype)

def pack_int4_group(W,gs=GS):
    """Existing granite/Gf17Atex int4-group pack (do not invent a new scheme)."""
    out,inf=W.shape
    pad=(gs-inf%gs)%gs
    Wp=F.pad(W.float(),(0,pad)) if pad else W.float()
    Wg=Wp.reshape(out,-1,gs)
    sc=Wg.abs().amax(-1,keepdim=True).clamp_min(1e-8)/7.0
    codes=(torch.clamp(torch.round(Wg/sc),-8,7).to(torch.int32)+8).reshape(out,Wp.shape[1])
    return codes.to(torch.uint8).contiguous(),sc.squeeze(-1).float().contiguous(),inf

class _DummyTok:
    """Minimal tokenizer so a synthetic bake fixture can construct the svc in CI."""
    eos_token_id=1;pad_token_id=0;chat_template=None
    def apply_chat_template(s,msgs,add_generation_prompt=True,tokenize=False,**k):
        parts=[]
        for m in msgs:parts.append(str((m or {}).get('content','')))
        return '\n'.join(parts)
    def __call__(s,text,return_tensors=None,add_special_tokens=False,**k):
        ids=[1,2,3]
        if return_tensors=='pt':
            class _R:pass
            r=_R();r.input_ids=torch.tensor([ids],dtype=torch.long);return r
        return {'input_ids':ids}
    def decode(s,ids,skip_special_tokens=True,**k):
        try:return ' '.join(str(int(x)) for x in ids)
        except Exception:return str(ids)
    def encode(s,text,add_special_tokens=False,**k):
        return [1]

def _par(m,k):
    p=m;ps=k.split('.')
    for q in ps[:-1]:p=getattr(p,q)
    return p,ps[-1]

def _assign_dense(m,k,tt):
    p2,a=_par(m,k)
    if a in p2._parameters and p2._parameters[a] is not None:p2._parameters[a]=nn.Parameter(tt,requires_grad=False)
    else:p2._buffers[a]=tt

def _open_st_index(root):
    tens={}
    for fp in sorted(glob.glob(str(Path(root)/'*.safetensors'))):
        f=safe_open(fp,framework='pt')
        for k in f.keys():tens[k]=f
    return tens

def _fix_meta_and_device(m):
    dev=torch.device('cuda' if _cuda_ok() else 'cpu')
    for name,buf in list(m.named_buffers()):
        if buf is None:continue
        pp,_,leaf=name.rpartition('.')
        parent=m.get_submodule(pp) if pp else m
        if buf.is_meta:
            parent._buffers[leaf]=torch.zeros(buf.shape,dtype=buf.dtype,device=dev)
        elif buf.device!=dev:
            parent._buffers[leaf]=buf.to(dev)
    for name,p in list(m.named_parameters()):
        if p is None:continue
        if p.is_meta:
            pp,_,leaf=name.rpartition('.')
            parent=m.get_submodule(pp) if pp else m
            parent._parameters[leaf]=nn.Parameter(torch.zeros(p.shape,dtype=p.dtype if p.dtype.is_floating_point else torch.float32,device=dev),requires_grad=False)

def _empty_causal(cfg,path_hint=None):
    archs=tuple(getattr(cfg,'architectures',None) or [])
    is_gdn=any(a in _GDN_ARCHS for a in archs)
    if is_gdn:
        try:
            from amni.inference import triton_gdn_patch
            triton_gdn_patch.apply()
        except Exception:pass
    _ai=os.environ.get('AMNI_ATTN','eager')
    try:cfg._attn_implementation=_ai
    except Exception:pass
    with init_empty_weights():
        m=AutoModelForCausalLM.from_config(cfg).eval()
    if is_gdn:
        try:
            from amni.inference import triton_gdn_patch
            triton_gdn_patch.reattach_to_model(m)
        except Exception:pass
    return m,is_gdn

def _has_tokenizer_files(src):
    p=Path(src) if src else None
    if p is None or not p.is_dir():return False
    return any((p/n).exists() for n in ('tokenizer.json','tokenizer_config.json','vocab.json','spiece.model','tokenizer.model','chat_template.jinja'))

def _load_tokenizer(path,tok_src=None):
    for src in (tok_src,path):
        if not src:continue
        try:
            if Path(str(src)).exists() and not _has_tokenizer_files(src):
                continue
            kw={}
            if Path(str(src)).exists():kw['local_files_only']=True
            tok=AutoTokenizer.from_pretrained(src,**kw)
            if getattr(tok,'pad_token',None) is None and getattr(tok,'eos_token',None) is not None:tok.pad_token=tok.eos_token
            return tok
        except Exception:
            continue
    return _DummyTok()

def _eos_ids(tok,root):
    e=getattr(tok,'eos_token_id',None)
    gcp=Path(root)/'generation_config.json'
    if gcp.exists():
        try:e=json.loads(gcp.read_text(encoding='utf-8')).get('eos_token_id',e)
        except Exception:pass
    return set(int(x) for x in (e if isinstance(e,list) else [e]) if x is not None)

class QwenAtexChatService:
    def __init__(s,path=None,tok_src=None,bake=None,device=None):
        path=path or bake or _DEFAULT_BAKE
        s.path=str(path);s.device=device or ('cuda' if _cuda_ok() else 'cpu')
        s.gpu=_cuda_ok()
        s.tok=_load_tokenizer(path,tok_src)
        if is_gf17_atex_bake(path):s._load_atex_bake(path)
        else:s._load_hf(path)
        s.eos_ids=_eos_ids(s.tok,path)
        if not s.eos_ids:s.eos_ids={int(getattr(s.tok,'eos_token_id',1) or 1)}
        s.source='gf17_atex_bake' if is_gf17_atex_bake(path) else 'hf_int4_pack'
        print(f'[QwenAtex] loaded {s.source} from {path} device={s.device}',flush=True)

    def _load_atex_bake(s,bake):
        with open(os.path.join(bake,'bake_manifest.json'),encoding='utf-8') as _mf:man=json.load(_mf)
        cfg=AutoConfig.from_pretrained(bake)
        m,s.is_gdn=_empty_causal(cfg,bake)
        tens=_open_st_index(bake)
        if not tens:raise FileNotFoundError(f'Gf17Atex bake {bake!r} has bake_manifest.json but no *.safetensors')
        g=lambda k:tens[k].get_tensor(k)
        packed=bool(man.get('packed',False))
        n_q=0
        for k,info in (man.get('tensors') or {}).items():
            try:
                if (info or {}).get('q')==1:
                    codes=_to_dev(g(k+'.codes'));scale=_to_dev(g(k+'.scale'))
                    p2,mn=_par(m,k[:-len('.weight')])
                    inf=int(info.get('inf',info.get('shape',[0,0])[1]))
                    out=int(info.get('shape',[0,0])[0])
                    setattr(p2,mn,AtexLin(codes,scale,inf,out,packed=packed).to(codes.device))
                    n_q+=1
                else:
                    tt=g(k);tt=_to_dev(tt.to(torch.bfloat16) if tt.is_floating_point() else tt)
                    _assign_dense(m,k,tt)
            except (AttributeError,KeyError):
                continue
        try:m.tie_weights()
        except Exception:pass
        _fix_meta_and_device(m)
        s.m=m;s.model=m;s.lm=m.model;s.manifest=man;s.n_quant=n_q

    def _load_hf(s,hf_path):
        cfg=AutoConfig.from_pretrained(hf_path)
        m,s.is_gdn=_empty_causal(cfg,hf_path)
        linw={n+'.weight' for n,mod in m.named_modules() if isinstance(mod,nn.Linear) and not any(k in n for k in _HF_SKIP)}
        local=Path(hf_path).is_dir()
        n_q=0
        if local and list(Path(hf_path).glob('*.safetensors')):
            tens=_open_st_index(hf_path)
            g=lambda k:tens[k].get_tensor(k)
            keys=list(tens.keys())
            man={'gs':GS,'tensors':{},'source':str(hf_path),'packed':False}
            for k in keys:
                try:
                    w=g(k)
                    if k in linw and getattr(w,'dim',lambda:0)()==2:
                        codes,sc,inf=pack_int4_group(w,GS)
                        p2,mn=_par(m,k[:-len('.weight')])
                        setattr(p2,mn,AtexLin(_to_dev(codes),_to_dev(sc),inf,int(w.shape[0]),packed=False).to('cuda' if _cuda_ok() else 'cpu'))
                        man['tensors'][k]={'q':1,'shape':list(w.shape),'inf':inf};n_q+=1
                    else:
                        tt=_to_dev(w.to(torch.bfloat16) if w.is_floating_point() else w)
                        _assign_dense(m,k,tt)
                        man['tensors'][k]={'q':0,'shape':list(w.shape)}
                except (AttributeError,KeyError):
                    continue
        else:
            # Hub id or dir without shards: materialize then pack (real Qwen/Qwen3-8B path).
            full=AutoModelForCausalLM.from_pretrained(hf_path,torch_dtype=torch.bfloat16).eval()
            man={'gs':GS,'tensors':{},'source':str(hf_path),'packed':False}
            for name,mod in list(full.named_modules()):
                if not isinstance(mod,nn.Linear) or any(k in name for k in _HF_SKIP):continue
                if mod.weight.dim()!=2:continue
                codes,sc,inf=pack_int4_group(mod.weight.detach().cpu(),GS)
                p2,mn=_par(m,name)
                setattr(p2,mn,AtexLin(_to_dev(codes),_to_dev(sc),inf,int(mod.weight.shape[0]),packed=False).to('cuda' if _cuda_ok() else 'cpu'))
                man['tensors'][name+'.weight']={'q':1,'shape':list(mod.weight.shape),'inf':inf};n_q+=1
            for name,p in full.named_parameters():
                if name in man['tensors'] or name[:-7] in {k[:-7] for k in man['tensors'] if k.endswith('.weight') and man['tensors'][k].get('q')==1}:
                    continue
                try:
                    tt=_to_dev(p.detach().to(torch.bfloat16) if p.is_floating_point() else p.detach())
                    _assign_dense(m,name,tt)
                    man['tensors'][name]={'q':0,'shape':list(p.shape)}
                except (AttributeError,KeyError):
                    continue
            del full
        try:m.tie_weights()
        except Exception:pass
        _fix_meta_and_device(m)
        s.m=m;s.model=m;s.lm=m.model;s.manifest=man;s.n_quant=n_q

    def _place_ids(s,ids):
        return ids.cuda() if s.gpu else ids

    def _gen(s,ids,mx,do_sample,temp=0.7,no_repeat=0):
        if not s.gpu:
            raise RuntimeError('QwenAtexChatService._gen needs CUDA/HIP (skipped in CPU CI)')
        gen=[];ng={}
        with torch.no_grad():
            out=s.lm(input_ids=ids,use_cache=True);cache=out.past_key_values
            for _ in range(mx):
                lg=s.m.lm_head(out.last_hidden_state[:,-1:])[:,-1].float()
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
        try:
            prompt=s.tok.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False)
        except Exception:
            prompt='\n'.join(str(m.get('content','')) for m in msgs)
        ids=s.tok(prompt,return_tensors='pt',add_special_tokens=False).input_ids
        return s._gen(s._place_ids(ids),max_new_tokens,do_sample,temp=float(kw.get('temperature',0.7)),no_repeat=kw.get('no_repeat',0))

    def chat_stream(s,*a,**k):
        t,n=s.chat(*a,**k);yield t

    def chat_mcq(s,prompt,max_new_tokens=256,n_opts=10):
        """Eval helper: reason, then continue 'The answer is (' and read letter-argmax."""
        reasoning,_=s.chat(prompt,max_new_tokens=max_new_tokens,do_sample=False,kb_top_k=0)
        tail='\nThe answer is ('
        base=s.tok.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,tokenize=False)
        ids=s.tok(base+(reasoning or '')+tail,return_tensors='pt',add_special_tokens=False).input_ids
        with torch.no_grad():
            out=s.lm(input_ids=s._place_ids(ids),use_cache=False)
            lg=s.m.lm_head(out.last_hidden_state[:,-1:])[:,-1].float()
            tid=int(lg.argmax(-1).item())
        letter=s.tok.decode([tid],skip_special_tokens=True).strip()[:1].upper()
        if letter not in [chr(65+i) for i in range(n_opts)]:letter='?'
        return letter,reasoning

if __name__=='__main__':
    import tempfile,time
    from amni.inference.atex_select import select_svc_kind
    print('select probe',select_svc_kind('bakes/qwen3_8b_gf17_atex_probe'))
    if _cuda_ok():
        t=time.time();svc=QwenAtexChatService();print(f'loaded {time.time()-t:.0f}s',flush=True)
        r,n=svc.chat('What is 17*23?',max_new_tokens=16);print(f'[{n}tok] {r[:80]!r}',flush=True)
        print('QWEN_ATEX_SVC_OK',flush=True)
    else:
        print('no CUDA/HIP — construction smoke is tests/test_qwen_atex_bake.py',flush=True)
