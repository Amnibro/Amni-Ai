import os,re,sys,json,glob,time,threading,torch,torch.nn as nn
from pathlib import Path
from safetensors import safe_open
from accelerate import init_empty_weights
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import Cache,DynamicLayer
from amni.inference import qgroup_gemv as Q
SYSM={'blend':"You are Adam, a warm and concise assistant. The user message may begin with reference notes written as 'Label: text', followed by the user's question. When notes are present, answer in one to three plain sentences using only facts stated in the notes, and add no names, numbers or facts that are not in them. If the notes do not cover the question, say you are not sure. When there are no notes and the user asks for facts, say you are not sure; otherwise reply briefly and naturally to what the user said.",
'blend2':"You are Adam, a warm assistant. The user message may begin with reference notes written as 'Label: text', followed by the question. If the notes answer the question, reply in two or three plain sentences that restate what the notes say, reusing their own wording and covering their main facts, and add nothing new: no extra names, numbers, dates or details. If the notes do not answer the question, or there are no notes and the user asks for facts, reply that you are not sure. If the user is just chatting, reply briefly and kindly.",
'blend3':"You are Adam, a warm and friendly companion. Each user message may start with reference notes written as 'Label: text', followed by the user's own message. Rules. 1) If notes are present and answer the question, reply in two or three plain sentences that restate the answer using the notes' own wording and their main facts, and add nothing that is not in the notes: no new names, numbers, dates or details. 2) If notes are present but do not answer the question, reply 'I'm not sure about that.' 3) If there are no notes and the user is chatting or sharing how they feel, reply with one or two warm sentences about what they said, and never say you are unsure. 4) If there are no notes and the user asks a factual question, reply that you are not sure.",
'blend4':"You are Adam, a warm and friendly companion. Each user message may start with reference notes written as 'Label: text', followed by the user's own message. Follow these rules. 1) Notes present and they answer the question: reply in two or three plain full sentences that restate everything the notes say about the subject, using the notes' own wording, and add nothing that is not in the notes: no new names, numbers, dates or details. 2) Notes present but they do not answer the question: reply 'I'm not sure about that.' 3) No notes and the message is a statement, a greeting or a feeling (not a question): reply with one or two warm sentences about what they said. This is the normal case for chatting, so do not say you are unsure. 4) No notes and the message is a factual question: reply that you are not sure.",
'split':("You are Adam, a warm assistant. The user message starts with reference notes written as 'Label: text', followed by the user's question. If the notes answer the question, reply in two or three plain sentences that restate what the notes say, reusing their own wording and covering their main facts, and add nothing new: no extra names, numbers, dates or details. If the notes do not answer the question, reply exactly: I'm not sure about that.","You are Adam, a warm and friendly companion. Reply to the user in one or two short, kind sentences that respond to what they said. If they ask a factual question, reply that you are not sure."),
'split2':("You are Adam, a warm assistant. The user message may begin with reference notes written as 'Label: text', followed by the question. If the notes answer the question, reply in two or three plain sentences that restate what the notes say, reusing their own wording and covering their main facts, and add nothing new: no extra names, numbers, dates or details. If the notes do not answer the question, reply that you are not sure.","You are Adam, a warm and friendly companion. Reply to the user in one or two kind, natural sentences that respond directly to what they said. Only if they ask a factual question, reply that you are not sure."),
'split3':("You are Adam, a warm and friendly companion. The user message begins with reference notes written as 'Label: text', followed by the user's question. If the notes answer the question, reply in two or three plain sentences that restate the answer using the notes' own wording and their main facts, adding nothing that is not in the notes: no new names, numbers, dates or details. If the notes do not answer the question, reply exactly: I'm not sure about that.","You are Adam, a warm and friendly companion who knows no outside facts. If the user shares something, greets you or says how they feel, reply with one or two warm sentences about what they said. If the user asks a question about a fact, a person, a place or a thing, reply exactly: I'm not sure about that. Earlier turns of this conversation are the only thing you remember."),
'alone':"You are Adam, a warm and concise assistant. Reply briefly and naturally."}
SYSM['split4']=(SYSM['split2'][0],SYSM['split3'][1])
NOTE=re.compile(r"[A-Z][\w ().,'\-]{0,60}: \S")
thr_t=[0.0,0.0]
def thr():
    if time.time()-thr_t[1]>2:
        thr_t[1]=time.time()
        try:thr_t[0]=float(open(os.path.expanduser('~/.local/share/amni-ascend/throttle.txt')).read())
        except Exception:thr_t[0]=0.0
    return thr_t[0]
class QLin(nn.Module):
    def __init__(s,w,bits,gs,inn):
        super().__init__();s.w=w;s.bits=bits;s.gs=gs;s.inn=inn;s.out=w[-1].shape[0]
    def forward(s,x):
        f=x.reshape(-1,x.shape[-1]).half();return Q.qgemm(s.w[:-2],s.w[-2],s.w[-1],f,s.bits,s.gs).reshape(*x.shape[:-1],s.out).to(x.dtype)
class QKV(DynamicLayer):
    bits=8
    def __init__(s,**k):super().__init__(**k);s.n=0;s.S=None
    def _q(s,x):
        qm=2**(s.bits-1)-1;sc=(x.abs().amax(-1,keepdim=True).float()/qm).clamp_min(1e-8);q=torch.round(x.float()/sc).clamp(-qm,qm).to(torch.int8)
        return ((q[...,0::2]&15)|((q[...,1::2]&15)<<4)).to(torch.uint8) if s.bits==4 else q,sc.half()
    def _d(s,q,sc,dt):
        if s.bits==4:
            lo=(q&15).to(torch.int8);hi=(q>>4).to(torch.int8);lo=torch.where(lo>7,lo-16,lo);hi=torch.where(hi>7,hi-16,hi);q=torch.stack([lo,hi],-1).flatten(-2)
        return (q.float()*sc.float()).to(dt)
    def update(s,k,v,*a,**kw):
        s.is_initialized=True;s.dtype=k.dtype;s.device=k.device;kq,ks=s._q(k);vq,vs=s._q(v)
        s.S=(kq,ks,vq,vs) if s.S is None else tuple(torch.cat([o,n],-2) for o,n in zip(s.S,(kq,ks,vq,vs)))
        s.n=s.S[0].shape[-2];return s._d(s.S[0],s.S[1],k.dtype),s._d(s.S[2],s.S[3],v.dtype)
    def get_seq_length(s):return s.n
    def nbytes(s):return sum(t.numel()*t.element_size() for t in (s.S or ()))
def kvlayer(bits):return type(f'QKV{bits}',(QKV,),{'bits':bits})
def card():
    for c in sorted(glob.glob('/sys/class/drm/card[0-9]'),key=lambda p:p):
        try:
            if open(c+'/device/vendor').read().strip()=='0x1002' and int(open(c+'/device/mem_info_vram_total').read())>8<<30:return c+'/device/mem_info_vram_used'
        except OSError:pass
class Vram:
    def __init__(s):s.f=card();s.pk=0;s.run=False;s.base=s.cur()
    def cur(s):return int(open(s.f).read()) if s.f else 0
    def start(s):
        s.run=True;s.pk=s.cur();s.t=threading.Thread(target=s._l,daemon=True);s.t.start();return s
    def _l(s):
        while s.run:s.pk=max(s.pk,s.cur());time.sleep(0.02)
    def stop(s):s.run=False;s.t.join();return s.pk
class Atex:
    def __init__(s,bake,head='bf16',embed='gpu',stream=False,kv=16,chunk=0,dev='cuda'):
        s.bake=bake;s.o=dict(head=head,embed=embed,stream=stream,kv=kv,chunk=chunk);s.dev=dev;man=json.load(open(bake+'/bake_manifest.json'));s.man=man
        s.tok=AutoTokenizer.from_pretrained(bake);cfg=AutoConfig.from_pretrained(bake);s.cfg=cfg;s.ls=float(getattr(cfg,'logits_scaling',1.0) or 1.0)
        with init_empty_weights():m=AutoModelForCausalLM.from_config(cfg).eval()
        T={}
        for fp in sorted(glob.glob(bake+'/*.safetensors')):
            f=safe_open(fp,framework='pt');T.update({k:f for k in f.keys()})
        g=lambda k:T[k].get_tensor(k)
        pin=lambda t:t.pin_memory() if stream else t.to(dev)
        def par(k):
            p=m
            for q in k.split('.')[:-1]:p=getattr(p,q)
            return p,k.split('.')[-1]
        for k,info in man['tensors'].items():
            if k.endswith(('embed_tokens.weight','lm_head.weight')):continue
            if info['q']:
                b=k;w=[pin(t) for t in [g(b+'.p0')]+([g(b+'.p1')] if b+'.p1' in T else [])+[g(b+'.a'),g(b+'.c')]];p,n=par(k[:-7]);setattr(p,n,QLin(w,info['bits'],info['gs'],info['shape'][1]))
            else:p,n=par(k);p._parameters[n]=nn.Parameter(g(k).to(dev),requires_grad=False)
        for mod in m.modules():
            for nm,b in list(mod._buffers.items()):
                b is not None and b.device.type!=dev and mod._buffers.__setitem__(nm,b.to(dev))
        s.m=m;s.base_model=m.model;s.E=g('model.embed_tokens.weight');s.H=g('lm_head.weight') if 'lm_head.weight' in T else s.E;s.tied=s.H is s.E or 'lm_head.weight' not in T
        s.E=s.E.pin_memory() if embed=='cpu' else s.E.to(dev)
        s.H=s.E if s.tied and head=='bf16' and embed!='cpu' else s.H
        s.hq=None
        if head in('q8','q4'):
            bt=8 if head=='q8' else 4;Wd=(s.H if s.H.device.type==dev else s.H.to(dev));P=[];A=[];C=[]
            for i in range(0,Wd.shape[0],8192):
                q,a,c=Q.quantize(Wd[i:i+8192],bt,128);P.append(Q.split(q,bt));A.append(a);C.append(c)
            s.hq=QLin([torch.cat([p[j] for p in P]) for j in range(len(P[0]))]+[torch.cat(A),torch.cat(C)],bt,128,Wd.shape[1]);s.Hd=None
        elif head=='cpu':s.Hd=(s.H if s.H.device.type=='cpu' else s.H.cpu()).to(torch.bfloat16)
        else:s.Hd=s.H if s.H.device.type==dev else s.H.to(dev)
        s.layers=list(m.model.layers);s.L=[[x for x in l.modules() if isinstance(x,QLin)] for l in s.layers];s.kvb=0
        stream and s._stream_init()
        gcp=bake+'/generation_config.json';e=s.tok.eos_token_id
        if os.path.exists(gcp):e=json.load(open(gcp)).get('eos_token_id',e)
        e=e if e is not None else cfg.eos_token_id;s.eos={int(x) for x in (e if isinstance(e,list) else [e]) if x is not None}|{int(x) for x in (cfg.eos_token_id if isinstance(cfg.eos_token_id,list) else [cfg.eos_token_id]) if x is not None}
        s.eos|={i for i in (s.tok.convert_tokens_to_ids(t) for t in ('<|im_end|>','<|end_of_text|>','<|endoftext|>','<|eot_id|>','<end_of_turn>')) if isinstance(i,int) and i>=0 and i!=s.tok.unk_token_id}
    def _stream_init(s):
        s.cs=torch.cuda.Stream();s.slots=[[[torch.empty_like(t,device=s.dev) for t in q.w] for q in s.L[0]] for _ in range(2)];s.ready=[torch.cuda.Event() for _ in range(2)];s.done=[torch.cuda.Event() for _ in range(2)];s.have=[None,None];n=len(s.layers)
        s.host=[[list(q.w) for q in L] for L in s.L]
        def fetch(i):
            sl=i%2
            with torch.cuda.stream(s.cs):
                s.cs.wait_event(s.done[sl]);[d.copy_(h,non_blocking=True) for ds,hs in zip(s.slots[sl],s.host[i]) for d,h in zip(ds,hs)];s.ready[sl].record(s.cs)
            s.have[sl]=i
        def pre(i):
            def f(mod,args,kw=None):
                sl=i%2;s.have[sl]!=i and fetch(i);torch.cuda.current_stream().wait_event(s.ready[sl])
                for q,ds in zip(s.L[i],s.slots[sl]):q.w=ds
                fetch((i+1)%n)
            return f
        def post(i):
            def f(mod,args,out):s.done[i%2].record(torch.cuda.current_stream())
            return f
        for i,l in enumerate(s.layers):l.register_forward_pre_hook(pre(i));l.register_forward_hook(post(i))
    def emb(s,ids):return s.E.index_select(0,ids.to(s.E.device)).to(s.dev,non_blocking=True)
    def logits(s,h):
        h=h.reshape(1,-1)
        if s.hq is not None:return s.hq(h).float()/s.ls
        if s.o['head']=='cpu':return (h.cpu().to(torch.bfloat16)@s.Hd.T).float().to(s.dev)/s.ls
        return (h.to(s.Hd.dtype)@s.Hd.T).float()/s.ls
    def cache(s):return Cache(layer_class_to_replicate=kvlayer(s.o['kv'])) if s.o['kv'] in(8,4) else Cache(layer_class_to_replicate=DynamicLayer)
    @torch.no_grad()
    def prefill(s,ids,cache):
        c=s.o['chunk'] or ids.shape[-1]
        for i in range(0,ids.shape[-1],c):o=s.base_model(inputs_embeds=s.emb(ids[0,i:i+c]).unsqueeze(0),past_key_values=cache,use_cache=True)
        return o.last_hidden_state[0,-1]
    @torch.no_grad()
    def generate(s,ids,max_new=64,stops=(),temperature=0.0):
        ca=s.cache();h=s.prefill(ids,ca);out=[];txt=''
        for _ in range(max_new):
            lg=s.logits(h);t=int(lg.argmax(-1).item()) if temperature<=0 else int(torch.multinomial(torch.softmax(lg/temperature,-1),1).item())
            if t in s.eos:break
            out.append(t);txt=s.tok.decode(out,skip_special_tokens=True)
            if stops and any(x in txt.lstrip() for x in stops):break
            len(out)%8==0 and time.sleep(thr())
            h=s.base_model(inputs_embeds=s.emb(torch.tensor([t])).unsqueeze(0),past_key_values=ca,use_cache=True).last_hidden_state[0,-1]
        s.kvb=sum(getattr(l,'nbytes',lambda:l.keys.numel()*l.keys.element_size()*2 if l.keys is not None else 0)() for l in ca.layers);s.last_n=len(out);return txt
_U=re.compile(r"User: (.*?)\nAdam:(?: (.*?))?\n\n",re.S)
def msgs(prompt):
    M=[];p=0
    for m in _U.finditer(prompt):M+=[{'role':'user','content':m[1]},{'role':'assistant','content':m[2] or ''}];p=m.end()
    last=prompt[p:];u=re.match(r"User: (.*)\nAdam:\s*$",last,re.S);M.append({'role':'user','content':(u[1] if u else last).strip()});return M
class Blender:
    def __init__(s,spec=None,opts=None):
        spec=spec or os.environ.get('AMNI_BLENDER','');s.path=spec[5:] if spec.startswith('atex:') else spec;o={**json.loads(os.environ.get('AMNI_BLENDER_OPTS') or '{}'),**(opts or {})};sy=o.pop('sys','blend');s.sys=SYSM.get(sy,sy);s.e=Atex(s.path,**o);s.o=o;s.lock=threading.RLock();s.fails=0;s.until=0.0;s.p=s;s.run=spec;s.info={'model':os.path.basename(s.path.rstrip('/')),'step':'atex','fmt':'chat','ctx':int(os.environ.get('AMNI_ADAM_T_CTX') or 4096)}
    def prompt_ids(s,prompt):
        m=msgs(prompt);ty=isinstance(s.sys,tuple);M=[{'role':'system','content':s.sys[0 if not ty or NOTE.match(m[-1]['content']) else 1] if ty else s.sys}]+m
        try:t=s.e.tok.apply_chat_template(M,add_generation_prompt=True,tokenize=False,enable_thinking=False)
        except Exception:t=s.e.tok.apply_chat_template(M,add_generation_prompt=True,tokenize=False)
        return s.e.tok(t,return_tensors='pt',add_special_tokens=False).input_ids
    def text(s,prompt,max_bytes=400,stops=(),timeout=None):
        with s.lock:
            ids=s.prompt_ids(prompt);t=s.e.generate(ids,max(8,int(max_bytes)//3),('\n',),0.0);return t.lstrip().split('\n')[0].strip()
    def gen(s,prompt,max_tokens=400,stops=(),history=None,pre=(),rc=(),sep=' '):
        from amni.serve import adam_t_client as C
        return s.text(C.template(prompt,history,0,pre,rc,sep),max_tokens)
    def budget(s,max_tokens=400):return int(s.info['ctx'])*3
    def proto(s):return s
    def alive(s):return True
    def status(s):return {'run':s.run,'alive':True,'fails':0,**s.info}
    def close(s):0
_B={}
def active():return os.environ.get('AMNI_BLENDER','').startswith('atex:')
def blender():
    k=(os.environ.get('AMNI_BLENDER'),os.environ.get('AMNI_BLENDER_OPTS'))
    _B.get(k) or (_B.clear(),_B.__setitem__(k,Blender()))
    return _B[k]
