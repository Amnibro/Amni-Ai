import os,re,json,time,shlex,base64,queue,logging,threading,subprocess
from pathlib import Path
from typing import Iterable,List,Optional,Sequence,Tuple
B64=lambda b:base64.b64encode(b).decode();U64=lambda s:base64.b64decode(s)
LOG=logging.getLogger('amni.serve.adam_t')
ROOT=Path(__file__).resolve().parents[2]
STOPS=('\n\n','\nUser:','\nAdam:')
class ProtoError(RuntimeError):pass
class Proto:
    def __init__(s,cmd,timeout:Optional[float]=None,env=None,stderr=None,load_timeout:Optional[float]=None):
        s.cmd=cmd;s.timeout=timeout;s.lock=threading.RLock();s.q=queue.Queue()
        s.p=subprocess.Popen(shlex.split(cmd) if isinstance(cmd,str) else list(cmd),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8',bufsize=1,env=env,start_new_session=True)
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
def template(message:str,history=None,budget:int=0)->str:
    cur=f'User: {_nl(message)}\nAdam:';T=[f'User: {_nl(u)}\nAdam: {_nl(a)}\n\n' for u,a in [tuple(h)[:2] for h in (history or []) if h and len(h)>=2]];n=len(cur.encode('utf-8'));k=len(T)
    while k and budget>0 and n+len(T[k-1].encode('utf-8'))<=budget:n+=len(T[k-1].encode('utf-8'));k-=1
    return ''.join(T[k:] if budget>0 else T)+cur
def _f(k:str,d:float)->float:
    try:return float(os.environ.get(k) or d)
    except ValueError:return d
class AdamT:
    def __init__(s,run:str='',bin:str='',timeout:float=0,load_timeout:float=0):
        s.run=run or os.environ.get('AMNI_ADAM_T','').strip();s.bin=bin or os.environ.get('AMNI_ADAM_T_BIN','').strip() or str(ROOT/'crates'/'target'/'release'/'adam-t')
        s.timeout=timeout or _f('AMNI_ADAM_T_TIMEOUT',30);s.load_timeout=load_timeout or _f('AMNI_ADAM_T_LOAD_TIMEOUT',180);s.lock=threading.Lock();s.p=None;s.fails=0;s.until=0.0
    def target(s)->str:
        g=os.path.join(s.run,'model.gf17');return s.run if s.run.endswith('.gf17') or not os.path.isfile(g) else g
    def env(s)->dict:
        return {**os.environ,'BACKEND':os.environ.get('AMNI_ADAM_T_BACKEND') or 'ndarray','RAYON_NUM_THREADS':os.environ.get('AMNI_ADAM_T_THREADS') or '2'}
    def proto(s)->Proto:
        if s.p and s.p.alive():return s.p
        if time.monotonic()<s.until:raise ProtoError(f'restart backoff {s.until-time.monotonic():.1f}s')
        s.p=None;s.p=Proto([s.bin,'serve',s.target()],s.timeout,s.env(),subprocess.DEVNULL,s.load_timeout);return s.p
    def gen(s,prompt:str,max_tokens:int=400,stops:Iterable[str]=STOPS,history=None)->str:
        with s.lock:
            try:
                p=s.proto();ctx=int(_f('AMNI_ADAM_T_CTX',p.info.get('ctx') or 1024));t=p.text(template(prompt,history,max(64,ctx-int(max_tokens))),int(max_tokens),stops).strip();s.fails=0;return t
            except BaseException:
                s.fails+=1;s.until=time.monotonic()+min(300.0,2.0**s.fails);s.p and not s.p.alive() and setattr(s,'p',None);raise
    def status(s)->dict:
        p=s.p;return {'run':s.run,'alive':bool(p and p.alive()),'fails':s.fails,**({k:p.info.get(k) for k in ('model','step','fmt','ctx')} if p else {})}
    def close(s)->None:
        with s.lock:s.p and s.p.close();s.p=None
_C={}
_L=threading.Lock()
def on()->bool:return bool(os.environ.get('AMNI_ADAM_T','').strip())
def client()->AdamT:
    r=os.environ.get('AMNI_ADAM_T','').strip()
    with _L:
        c=_C.get(r)
        if c is None:[x.close() for x in _C.values()];_C.clear();c=_C[r]=AdamT(r)
        return c
def answer(message:str,history=None,max_new_tokens:int=512)->Optional[str]:
    if not on():return None
    t0=time.monotonic()
    try:t=client().gen(message,min(int(_f('AMNI_ADAM_T_MAX_BYTES',400)),max(64,4*int(max_new_tokens or 0))),STOPS,history);why='' if t else 'empty'
    except Exception as e:t,why='',type(e).__name__
    LOG.debug('route=%s ms=%d',why and f'fallback:{why}' or 'adam_t',int((time.monotonic()-t0)*1000));return t or None
