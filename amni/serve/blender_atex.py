import os,re,json,glob,time,threading,torch,torch.nn as nn
from safetensors import safe_open
from accelerate import init_empty_weights
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import Cache,DynamicLayer,DynamicCache
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
SYSM['copyA']=("You are Adam, a warm assistant. The user message begins with reference notes written as 'Label: text', followed by the user's question. If the notes answer the question, reply by restating the notes almost word for word: reuse their own sentences and wording, keep all of their facts, change only what is needed to make the reply flow as an answer, and add nothing that is not in the notes. If the notes clearly do not answer the question, reply that you are not sure.",SYSM['split3'][1])
SYSM['copyB']=("You are Adam, a warm assistant. The user message begins with reference notes written as 'Label: text', followed by the user's question. If the notes answer the question, reply with one or two short plain sentences that copy the notes' wording, and then repeat any remaining sentences from the notes unchanged. Never swap in synonyms and never add a fact, name, number or date that is not in the notes. If the notes clearly do not answer the question, reply that you are not sure.",SYSM['split3'][1])
SYSM['copyC']=("You are Adam, a warm assistant. The user message begins with reference notes written as 'Label: text', followed by the user's question. If the notes answer the question, reply with a short paragraph that states every fact in the notes, sentence by sentence, in the notes' own words, starting with the part that answers the question. Do not leave any sentence of the notes out, do not swap in synonyms, and add no name, number, date or detail that is not in the notes. If the notes clearly do not answer the question, reply that you are not sure.",SYSM['split3'][1])
NOTE=re.compile(r"[A-Z][\w ().,'\-]{0,60}: \S")
thr_t=[0.0,0.0]
def thr():
    if os.environ.get('AMNI_BLENDER_NOTHR'):return 0.0
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
    bits=8;blk=256
    def __init__(s,**k):super().__init__(**k);s.n=0;s.S=None
    def _q(s,x):
        qm=2**(s.bits-1)-1;sc=(x.abs().amax(-1,keepdim=True).float()/qm).clamp_min(1e-8);q=torch.round(x.float()/sc).clamp(-qm,qm).to(torch.int8)
        return ((q[...,0::2]&15)|((q[...,1::2]&15)<<4)).to(torch.uint8) if s.bits==4 else q,sc.half()
    def _d(s,q,sc,dt):
        if s.bits==4:
            lo=(q&15).to(torch.int8);hi=(q>>4).to(torch.int8);lo=torch.where(lo>7,lo-16,lo);hi=torch.where(hi>7,hi-16,hi);q=torch.stack([lo,hi],-1).flatten(-2)
        return (q.float()*sc.float()).to(dt)
    def update(s,k,v,*a,**kw):
        s.is_initialized=True;s.dtype=k.dtype;s.device=k.device;N=(*s._q(k),*s._q(v));e=s.n+k.shape[-2];c=-(-e//s.blk)*s.blk
        s.S=s.S if s.S is not None and e<=s.S[0].shape[-2] else tuple((lambda z:(o is not None and z[...,:s.n,:].copy_(o[...,:s.n,:]),z)[1])(x.new_empty(*x.shape[:-2],c,x.shape[-1])) for o,x in zip(s.S or (None,)*4,N))
        [o[...,s.n:e,:].copy_(x) for o,x in zip(s.S,N)];s.n=e;return s._d(s.S[0][...,:e,:],s.S[1][...,:e,:],k.dtype),s._d(s.S[2][...,:e,:],s.S[3][...,:e,:],v.dtype)
    def get_seq_length(s):return s.n
    def nbytes(s):return sum(t.numel()*t.element_size() for t in (s.S or ()))
def noexp():torch.version.hip and getattr(torch._C,'_accelerator_setAllocatorSettings',torch.cuda.memory._set_allocator_settings)('expandable_segments:False')
def lbytes(l):return sum(x.numel()*x.element_size() for a in ('keys','values','conv_states','recurrent_states','S') for x in (lambda v:list(v.values()) if isinstance(v,dict) else v if isinstance(v,(list,tuple)) else [v])(getattr(l,a,None)) if torch.is_tensor(x))
def kvlayer(bits):return type(f'QKV{bits}',(QKV,),{'bits':bits})
def card():
    for c in sorted(glob.glob('/sys/class/drm/card[0-9]')):
        try:
            if open(c+'/device/vendor').read().strip()=='0x1002' and int(open(c+'/device/mem_info_vram_total').read())>8<<30:return c+'/device/mem_info_vram_used'
        except OSError:pass
def proc_vram(pid='self'):
    d={}
    for f in glob.glob(f'/proc/{pid}/fdinfo/*'):
        try:t=open(f).read()
        except OSError:continue
        m=re.search(r'drm-client-id:\s*(\d+)',t);v=re.search(r'drm-memory-vram:\s*(\d+) KiB',t)
        m and v and d.__setitem__(m[1],int(v[1])*1024)
    return sum(d.values())
class Vram:
    def __init__(s,dev=False):s.f=card() if dev else None;s.pk=0;s.run=False;s.base=s.cur()
    def cur(s):return int(open(s.f).read()) if s.f else proc_vram()
    def start(s):
        s.run=True;s.pk=s.cur();s.t=threading.Thread(target=s._l,daemon=True);s.t.start();return s
    def _l(s):
        while s.run:s.pk=max(s.pk,s.cur());time.sleep(0.02)
    def stop(s):s.run=False;s.t.join();return s.pk
class Atex:
    def __init__(s,bake,head='bf16',embed='gpu',stream=False,kv=16,chunk=0,dev='cuda'):
        noexp();s.bake=bake;s.o=dict(head=head,embed=embed,stream=stream,kv=kv,chunk=chunk);s.dev=dev;man=json.load(open(bake+'/bake_manifest.json'));s.man=man
        s.tok=AutoTokenizer.from_pretrained(bake);cfg=AutoConfig.from_pretrained(bake);s.cfg=cfg;s.tc=cfg.get_text_config() if hasattr(cfg,'get_text_config') else cfg;s.host_bytes=0;s.ls=float(getattr(cfg,'logits_scaling',1.0) or 1.0)
        with init_empty_weights():m=AutoModelForCausalLM.from_config(cfg).eval()
        T={}
        for fp in sorted(glob.glob(bake+'/*.safetensors')):
            f=safe_open(fp,framework='pt');T.update({k:f for k in f.keys()})
        g=lambda k:T[k].get_tensor(k)
        pin=lambda t:t if stream else t.to(dev)
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
        s.m=m;s.base_model=m.model;E=g('model.embed_tokens.weight');Hh=g('lm_head.weight') if 'lm_head.weight' in T else E;s.tied=Hh is E
        bt=lambda x:int(x[1:]) if x[0]=='q' else 0
        def qt(t,b):
            P=[];A=[];C=[]
            for i in range(0,t.shape[0],1024):q,a,c=Q.quantize(t[i:i+1024].to(dev),b,128);P.append(Q.split(q,b));A.append(a);C.append(c)
            return QLin([torch.cat([p[j] for p in P]) for j in range(len(P[0]))]+[torch.cat(A),torch.cat(C)],b,128,t.shape[1])
        s.QE=s.QH=s.E=s.Hd=None
        if bt(embed):s.QE=qt(E,bt(embed))
        elif embed=='cpu':s.E=E.pin_memory()
        else:s.E=E.to(dev)
        if bt(head):s.QH=s.QE if s.tied and bt(embed)==bt(head) else qt(Hh,bt(head))
        elif head=='cpu':s.Hd=(s.E if s.tied and s.E is not None and s.E.device.type=='cpu' else Hh).to(torch.bfloat16)
        else:s.Hd=s.E if s.tied and s.E is not None and s.E.device.type==dev else Hh.to(dev)
        s.layers=list(m.model.layers);s.L=[[x for x in l.modules() if isinstance(x,QLin)] for l in s.layers];s.kvb=0
        stream and s._stream_init()
        gcp=bake+'/generation_config.json';e=s.tok.eos_token_id
        if os.path.exists(gcp):e=json.load(open(gcp)).get('eos_token_id',e)
        e=e if e is not None else cfg.eos_token_id;s.eos={int(x) for x in (e if isinstance(e,list) else [e]) if x is not None}|{int(x) for x in (cfg.eos_token_id if isinstance(cfg.eos_token_id,list) else [cfg.eos_token_id]) if x is not None}
        s.eos|={i for i in (s.tok.convert_tokens_to_ids(t) for t in ('<|im_end|>','<|end_of_text|>','<|endoftext|>','<|eot_id|>','<end_of_turn>')) if isinstance(i,int) and i>=0 and i!=s.tok.unk_token_id}
    def _stream_init(s):
        n=len(s.layers);al=lambda b:(b+255)//256*256;M=[[(t,t.numel()*t.element_size()) for q in L for t in q.w] for L in s.L];O=[[sum(al(b) for _,b in m[:j]) for j in range(len(m)+1)] for m in M];sz=max(o[-1] for o in O);s.cs=torch.cuda.Stream()
        s.host=[torch.empty(max(1,o[-1]),dtype=torch.uint8).pin_memory() for o in O];[s.host[i][o:o+b].copy_(t.contiguous().reshape(-1).view(torch.uint8)) for i,(m,oo) in enumerate(zip(M,O)) for (t,b),o in zip(m,oo)];s.host_bytes=sum(h.numel() for h in s.host)
        s.slots=[torch.empty(sz,dtype=torch.uint8,device=s.dev) for _ in range(2)];s.ready=[torch.cuda.Event() for _ in range(2)];s.done=[torch.cuda.Event() for _ in range(2)];s.have=[None,None]
        it=lambda i:iter([s.slots[i%2][o:o+b].view(t.dtype).view(t.shape) for (t,b),o in zip(M[i],O[i])]);s.V=[(lambda g:[[next(g) for _ in q.w] for q in L])(it(i)) for i,L in enumerate(s.L)];[setattr(q,'w',w) for L,V in zip(s.L,s.V) for q,w in zip(L,V)];s.hsz=[o[-1] for o in O]
        def fetch(i):
            sl=i%2
            with torch.cuda.stream(s.cs):
                s.cs.wait_event(s.done[sl]);s.slots[sl][:s.hsz[i]].copy_(s.host[i][:s.hsz[i]],non_blocking=True);s.ready[sl].record(s.cs)
            s.have[sl]=i
        def pre(i):
            def f(mod,args,kw=None):
                sl=i%2;s.have[sl]!=i and fetch(i);torch.cuda.current_stream().wait_event(s.ready[sl])
                for q,w in zip(s.L[i],s.V[i]):q.w=w
                (i+1<n or n%2==0) and fetch((i+1)%n)
            return f
        def post(i):
            def f(mod,args,out):s.done[i%2].record(torch.cuda.current_stream())
            return f
        for i,l in enumerate(s.layers):l.register_forward_pre_hook(pre(i));l.register_forward_hook(post(i))
    def emb(s,ids):
        if s.QE is not None:
            w=s.QE.w;return Q.dequant([p.index_select(0,ids.to(p.device)) for p in w[:-2]],w[-2].index_select(0,ids.to(w[-2].device)),w[-1].index_select(0,ids.to(w[-1].device)),s.QE.bits,s.QE.gs,s.QE.inn).to(torch.bfloat16)
        return s.E.index_select(0,ids.to(s.E.device)).to(s.dev)
    def logits(s,h):
        h=h.reshape(-1,h.shape[-1])
        if s.QH is not None:return s.QH(h).float()/s.ls
        if s.o['head']=='cpu':return (h.cpu().to(torch.bfloat16)@s.Hd.T).float().to(s.dev)/s.ls
        return (h.to(s.Hd.dtype)@s.Hd.T).float()/s.ls
    def cache(s):
        K=kvlayer(s.o['kv']) if s.o['kv'] in(8,4) else DynamicLayer;lt=list(getattr(s.tc,'layer_types',None) or [])
        return Cache(layer_class_to_replicate=K) if 'linear_attention' not in lt else (lambda c:[c.layers.__setitem__(i,K()) for i,t in enumerate(lt) if t=='full_attention'] and c)(DynamicCache(config=s.tc))
    @torch.no_grad()
    def prefill(s,ids,cache,allh=False):
        c=s.o['chunk'] or ids.shape[-1];H=[]
        for i in range(0,ids.shape[-1],c):o=s.base_model(inputs_embeds=s.emb(ids[0,i:i+c]).unsqueeze(0),past_key_values=cache,use_cache=True);allh and H.append(o.last_hidden_state[0]);c<ids.shape[-1] and torch.cuda.empty_cache()
        return torch.cat(H) if allh else o.last_hidden_state[0,-1]
    @torch.no_grad()
    def generate(s,ids,max_new=64,stops=(),temperature=0.0,ign=False,tm=None,top_p=1.0,seed=None,lg_out=None):
        t0=time.time();ca=s.cache();h=s.prefill(ids,ca);torch.cuda.synchronize();t1=time.time();out=[];txt='';g=torch.Generator(device=s.dev).manual_seed(int(seed)) if seed is not None and temperature>0 else None
        def nuc(p):
            v,i=torch.sort(p,-1,descending=True);k=(v.cumsum(-1)-v)<top_p;return torch.zeros_like(p).scatter(-1,i,v*k)
        for _ in range(max_new):
            lg=s.logits(h);lg_out is not None and lg_out.append(lg[0].float().cpu());t=int(lg.argmax(-1).item()) if temperature<=0 else int(torch.multinomial(nuc(torch.softmax(lg/temperature,-1)) if top_p<1 else torch.softmax(lg/temperature,-1),1,generator=g).item())
            if t in s.eos and not ign:break
            out.append(t)
            if stops:
                txt=s.tok.decode(out,skip_special_tokens=True)
                if any(x in txt.lstrip() for x in stops):break
            len(out)%8==0 and time.sleep(thr());len(out)%QKV.blk==0 and torch.cuda.empty_cache()
            h=s.base_model(inputs_embeds=s.emb(torch.tensor([t])).unsqueeze(0),past_key_values=ca,use_cache=True).last_hidden_state[0,-1]
        torch.cuda.synchronize();t2=time.time();tm is not None and tm.update(prefill_s=t1-t0,decode_s=t2-t1,n_prompt=int(ids.shape[-1]),n_new=len(out))
        s.kvb=sum(lbytes(l) for l in ca.layers);s.last_n=len(out);s.last_ids=out;return s.tok.decode(out,skip_special_tokens=True)
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
    def text(s,prompt,max_bytes=400,stops=(),timeout=None,dec=None):
        with s.lock:
            d=dec or {};ids=s.prompt_ids(prompt);t=s.e.generate(ids,max(8,int(max_bytes)//3),('\n',),float(d.get('temperature',0.0)),top_p=float(d.get('top_p',1.0)),seed=d.get('seed'));from amni.serve import adam_t_client as C;return C.asc(t.lstrip().split('\n')[0].strip())
    def gen(s,prompt,max_tokens=400,stops=(),history=None,pre=(),rc=(),sep=' ',dec=None):
        from amni.serve import adam_t_client as C
        nm=os.environ.get('AMNI_BLENDER_NOMAT')=='1';return s.text(C.template(prompt,history,0,() if nm else pre,() if nm else rc,sep),max_tokens,dec=dec)
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
