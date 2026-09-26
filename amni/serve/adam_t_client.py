import os,re,json,time,shlex,base64,queue,ctypes,logging,threading,subprocess,unicodedata
from pathlib import Path
from typing import Iterable,List,Optional,Sequence,Tuple
B64=lambda b:base64.b64encode(b).decode();U64=lambda s:base64.b64decode(s)
LOG=logging.getLogger('amni.serve.adam_t')
ROOT=Path(__file__).resolve().parents[2]
STOPS=('\n\n','\nUser:','\nAdam:')
ACK='Got it, thanks for telling me.'
try:_PRCTL=ctypes.CDLL(None,use_errno=True).prctl if os.name=='posix' else None
except Exception:_PRCTL=None
_pdeath=lambda pp:lambda:(_PRCTL and _PRCTL(1,9),os.getppid()!=pp and os._exit(1))
_SQ=queue.Queue();_ST=[None];_SL=threading.Lock()
def _spawner():
    while True:
        a,k,r=_SQ.get()
        try:r.put((1,subprocess.Popen(*a,**k)))
        except BaseException as e:r.put((0,e))
def spawn(*a,**k)->subprocess.Popen:
    with _SL:_ST[0] and _ST[0].is_alive() or (_ST.__setitem__(0,threading.Thread(target=_spawner,daemon=True,name='adam-t-spawn')),_ST[0].start())
    r=queue.Queue(1);_SQ.put((a,{**k,**({'preexec_fn':_pdeath(os.getpid())} if os.name=='posix' else {})},r));ok,v=r.get()
    if not ok:raise v
    return v
class ProtoError(RuntimeError):pass
class Proto:
    def __init__(s,cmd,timeout:Optional[float]=None,env=None,stderr=None,load_timeout:Optional[float]=None):
        s.cmd=cmd;s.timeout=timeout;s.lock=threading.RLock();s.q=queue.Queue()
        s.p=spawn(shlex.split(cmd) if isinstance(cmd,str) else list(cmd),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8',bufsize=1,env=env,start_new_session=True)
        threading.Thread(target=s._pump,args=(s.p,s.q),daemon=True).start()
        try:s.info={'cmd':cmd if isinstance(cmd,str) else shlex.join(cmd),**s.rpc(timeout=load_timeout or timeout,op='hello')}
        except BaseException:s.kill();raise
    @staticmethod
    def _pump(p,q):
        try:[q.put(l) for l in iter(p.stdout.readline,'')]
        except Exception:pass
        q.put(None)
    def alive(s)->bool:return s.p.poll() is None
    def kill(s)->None:
        try:s.p.kill();s.p.wait(5)
        except Exception:pass
    def rpc(s,timeout:Optional[float]=None,**k)->dict:
        with s.lock:
            t=timeout or s.timeout;end=time.monotonic()+t if t else None
            if not s.alive():raise ProtoError(f'server exited rc={s.p.poll()}')
            try:s.p.stdin.write(json.dumps(k)+'\n');s.p.stdin.flush()
            except (OSError,ValueError) as e:s.kill();raise ProtoError(f'write failed: {e}')
            while True:
                try:l=s.q.get(timeout=max(0.0,end-time.monotonic()) if end else None)
                except queue.Empty:s.kill();raise ProtoError(f"timeout after {t}s on {k.get('op')}")
                if l is None:s.kill();raise ProtoError(f'server exited rc={s.p.poll()}')
                try:r=json.loads(l)
                except ValueError:continue
                if not isinstance(r,dict):continue
                if 'error' in r:raise ProtoError(str(r['error']))
                return r
    def text(s,prompt:str,max_bytes:int,stops:Iterable[str]=(),timeout:Optional[float]=None)->str:
        return U64(s.rpc(timeout=timeout,op='gen',prompt=B64(prompt.encode('utf-8')),max=int(max_bytes),stop=[B64(x.encode('utf-8')) for x in stops])['text']).decode('utf-8','replace')
    def lp(s,P:Sequence[Tuple[bytes,bytes]],chunk:int=64)->List[float]:
        return [float(x) for i in range(0,len(P),chunk) for x in s.rpc(op='lp',items=[[B64(a),B64(b)] for a,b in P[i:i+chunk]])['lp']]
    def close(s)->None:
        try:s.alive() and (s.rpc(timeout=10,op='bye'),s.p.wait(30))
        except Exception:s.kill()
_nl=lambda x:re.sub(r'\n\s*\n+','\n',str(x or '').strip())
_SYM=str.maketrans({'\u00a3':'GBP ','\u20ac':'EUR ','\u00a5':'JPY ','\u00b0':' degrees','\u2044':'/','\u2212':'-','\u00d7':'x','\u00f7':'/','\u221a':'sqrt','\u00bd':' 1/2','\u00bc':' 1/4','\u00be':' 3/4','\u2018':"'",'\u2019':"'",'\u201c':'"','\u201d':'"','\u2013':'-','\u2014':'-','\u2026':'...'})
asc=lambda t:re.sub(r'\s+',' ',re.sub(r'\[(?:\d+|[a-z]|citation needed|note \d+)\]','',unicodedata.normalize('NFKD',str(t or '').translate(_SYM)).encode('ascii','ignore').decode())).strip()
clip=lambda t,n:t if len(t.encode('utf-8'))<=n else (lambda c:c.rsplit(' ',1)[0] if ' ' in c else c)(t.encode('utf-8')[:max(0,n)].decode('utf-8','ignore'))
def _fit(T:list,n:int,budget:int)->Tuple[list,int]:
    k=len(T)
    while k and budget>0 and n+len(T[k-1].encode('utf-8'))<=budget:n+=len(T[k-1].encode('utf-8'));k-=1
    return (T[k:] if budget>0 else T),n
def template(message:str,history=None,budget:int=0,pre:Sequence[str]=(),rc:Sequence[str]=())->str:
    R=[x for x in rc if x];cur=lambda R:f"User: {' '.join(R+[_nl(message)])}\nAdam:"
    while R and budget>0 and len(cur(R).encode('utf-8'))>budget:R.pop()
    T=[f'User: {_nl(u)}\nAdam: {_nl(a)}\n\n' for u,a in [tuple(h)[:2] for h in (history or []) if h and len(h)>=2]];P,n=_fit(list(pre),len(cur(R).encode('utf-8')),budget);H,_=_fit(T,n,budget)
    return ''.join(P)+''.join(H)+cur(R)
def _f(k:str,d:float)->float:
    try:return float(os.environ.get(k) or d)
    except ValueError:return d
def _mem(message:str,history,user,session)->dict:
    from amni.serve.user_memory import turn
    m=turn(message,history,user,session);return {'reply':m['reply'],'hist':m['hist'],'route':bool(m['route']),'pre':[t.replace('\nAdam: Got it.\n\n',f'\nAdam: {ACK}\n\n') for t in re.findall(r'User: [^\n]*\nAdam: [^\n]*\n\n',m['pre'] or '')]}
def _math(message:str)->str:
    r=__import__('amni.math.answer',fromlist=['answer']).answer(message) if os.environ.get('AMNI_MATH','1')=='1' else None;r=asc(r[0]) if r and r[1]>=_f('AMNI_MATH_MIN_CONF',0.6) else '';return r+('' if not r or r[-1] in '.!?' else '.')
def _trim(S:list,sc:list,cap:int)->str:
    i=max(range(len(S)),key=lambda j:(sc[j],-j));k=set();n=-1
    for j in sorted(range(len(S)),key=lambda j:(abs(j-i),j<i)):n+len(S[j])+1<=cap and (k.add(j),n:=n+len(S[j])+1)
    return ' '.join(S[j] for j in sorted(k)) or clip(S[i],cap)
def _know(message:str,n:int,cap:int)->list:
    from amni.compute import rknow as R
    if not R.factq(message,True):return []
    I=R.ix();W={w:I.widf(w) for w in dict.fromkeys(R.toks(message)) if w not in R.QSTOP};tot=sum(W.values()) or 1.0;out=[];seen=set();cov=_f('AMNI_ADAM_T_KNOW_COV',0.5)
    wf=__import__('amni.compute.wikifact',fromlist=['ask']) if os.environ.get('AMNI_RKNOW_WF','1')=='1' else None;w=wf and wf.ask(message);w=w if w and w.get('answer') and float(w.get('conf') or 0)>=wf.THR and (wf.TEMPORAL or not w.get('temporal')) else None
    w and out.append(clip(f"{asc(wf.store().label(w['e'])) or 'Wikidata'}: {asc(w['answer'])}.",cap))
    for _,pid in I.search(message,max(8,3*n)):
        t,x,g=I.text(pid);t,x=asc(t),asc(x);sw=set(R.toks(t+' '+x));S=[z for z in re.split(r'(?<=[.!?])\s+',x) if z]
        if g in seen or not S or sum(v for q,v in W.items() if q in sw)/tot<cov:continue
        seen.add(g);out.append(f'{t}: '+_trim(S,[sum(v for q,v in W.items() if q in set(R.toks(z))) for z in S],cap-len(t)-2))
        if len(out)>=n+bool(w):break
    return out
def context(message:str,history=None,user=None,session=None,budget:int=0)->dict:
    t0=time.perf_counter();c={'reply':None,'hist':list(history or []),'pre':[],'rc':[],'n':{}}
    if os.environ.get('AMNI_ADAM_T_CONTEXT','1')!='1':return {**c,'ms':0.0}
    mb,ab,kb,np_,pb=(int(_f(k,d)) for k,d in (('AMNI_ADAM_T_MEM_BYTES',640),('AMNI_ADAM_T_MATH_BYTES',240),('AMNI_ADAM_T_KNOW_BYTES',1100),('AMNI_ADAM_T_PASSAGES',2),('AMNI_ADAM_T_PASSAGE_BYTES',560)));m={'route':False};K=[]
    B=lambda:budget-len(template(message,None,0,(),c['rc']).encode('utf-8')) if budget>0 else 1<<30
    try:m=_mem(message,history,user,session);c.update(reply=m['reply'],hist=m['hist'])
    except Exception as e:c['n']['mem_err']=type(e).__name__
    try:ma=(lambda r,n:clip('Calculator: '+r,n) if r and n>=32 else '')(_math(message),min(ab,B()-1)) if ab>0 and not c['reply'] else '';c['rc']=[ma] if ma else []
    except Exception as e:c['n']['math_err']=type(e).__name__
    kq=min(kb,B()-np_)
    try:K=_fit(_know(message,np_,min(pb,kq//max(1,np_)))[::-1],0,kq)[0][::-1] if kq>=64 and np_>0 and not (c['reply'] or m['route'] or c['rc']) else []
    except Exception as e:c['n']['know_err']=type(e).__name__
    c['rc']+=K;q=min(mb,B());c['pre']=_fit(m['pre'],0,q)[0] if m['route'] and q>0 else []
    c['n'].update(mem=len(c['pre']),know=len(K),math=int(len(c['rc'])>len(K)));c['ms']=(time.perf_counter()-t0)*1000;return c
class AdamT:
    def __init__(s,run:str='',bin:str='',timeout:float=0,load_timeout:float=0):
        s.run=run or os.environ.get('AMNI_ADAM_T','').strip();s.bin=bin or os.environ.get('AMNI_ADAM_T_BIN','').strip() or str(ROOT/'crates'/'target'/'release'/'adam-t')
        s.timeout=timeout or _f('AMNI_ADAM_T_TIMEOUT',30);s.load_timeout=load_timeout or _f('AMNI_ADAM_T_LOAD_TIMEOUT',180);s.lock=threading.RLock();s.p=None;s.fails=0;s.until=0.0
    def target(s)->str:
        g=os.path.join(s.run,'model.gf17');return s.run if s.run.endswith('.gf17') or not os.path.isfile(g) else g
    def env(s)->dict:
        return {**os.environ,'BACKEND':os.environ.get('AMNI_ADAM_T_BACKEND') or 'ndarray','RAYON_NUM_THREADS':os.environ.get('AMNI_ADAM_T_THREADS') or '2'}
    def proto(s)->Proto:
        if s.p and s.p.alive():return s.p
        if time.monotonic()<s.until:raise ProtoError(f'restart backoff {s.until-time.monotonic():.1f}s')
        s.p=None;s.p=Proto([s.bin,'serve',s.target()],s.timeout,s.env(),subprocess.DEVNULL,s.load_timeout);return s.p
    def _fail(s)->None:s.fails+=1;s.until=time.monotonic()+min(300.0,2.0**s.fails);s.p and not s.p.alive() and setattr(s,'p',None)
    def _budget(s,max_tokens:int)->int:return max(64,int(_f('AMNI_ADAM_T_CTX',s.proto().info.get('ctx') or 1024))-int(max_tokens))
    def budget(s,max_tokens:int=400)->int:
        with s.lock:
            try:return s._budget(max_tokens)
            except BaseException:s._fail();raise
    def gen(s,prompt:str,max_tokens:int=400,stops:Iterable[str]=STOPS,history=None,pre:Sequence[str]=(),rc:Sequence[str]=())->str:
        with s.lock:
            try:b=s._budget(max_tokens);t=s.p.text(template(prompt,history,b,pre,rc),int(max_tokens),stops).strip();s.fails=0;return t
            except BaseException:s._fail();raise
    def status(s)->dict:
        p=s.p;return {'run':s.run,'alive':bool(p and p.alive()),'fails':s.fails,**({k:p.info.get(k) for k in ('model','step','fmt','ctx')} if p else {})}
    def close(s)->None:
        with s.lock:s.p and s.p.close();s.p=None
_C={}
_L=threading.Lock()
def on()->bool:return bool(os.environ.get('AMNI_ADAM_T','').strip())
def warm()->None:
    try:_know('who wrote the novel moby dick?',1,64)
    except Exception:pass
def client()->AdamT:
    r=os.environ.get('AMNI_ADAM_T','').strip()
    with _L:
        c=_C.get(r)
        if c is None:[x.close() for x in _C.values()];_C.clear();c=_C[r]=AdamT(r);os.environ.get('AMNI_ADAM_T_CONTEXT','1')=='1' and os.environ.get('AMNI_ADAM_T_WARM','1')=='1' and threading.Thread(target=warm,daemon=True,name='adam-t-warm').start()
        return c
def answer(message:str,history=None,max_new_tokens:int=512,user=None,session=None)->Optional[str]:
    if not on():return None
    t0=time.monotonic();c={'ms':0.0,'n':{}}
    try:
        cl=client()
        if not (cl.p and cl.p.alive()) and time.monotonic()<cl.until:raise ProtoError('restart backoff')
        mx=min(int(_f('AMNI_ADAM_T_MAX_BYTES',400)),max(64,4*int(max_new_tokens or 0)));c=context(message,history,user,session,cl.budget(mx));t=c['reply'] or cl.gen(message,mx,STOPS,c['hist'],c['pre'],c['rc']);why='' if t else 'empty'
    except Exception as e:t,why='',type(e).__name__
    LOG.debug('route=%s ms=%d ctx_ms=%.1f ctx=%s',why and f'fallback:{why}' or ('memory_action' if c.get('reply') else 'adam_t'),int((time.monotonic()-t0)*1000),c.get('ms',0.0),','.join(f'{k}:{v}' for k,v in sorted(c.get('n',{}).items())));return t or None
