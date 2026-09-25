"""rate_limit — in-process per-key sliding-window rate limiter to blunt flooding/abuse on public endpoints
(a runaway client or loop can't hammer /teach or /chat). Thread-safe; self-GCs idle keys. Tunable via env."""
import time,threading,os,ipaddress
_FWD=('x-forwarded-for','forwarded','x-real-ip','cf-connecting-ip')
class RateLimiter:
    def __init__(self,max_calls,window_s,name=''):
        self.max=max(1,int(max_calls));self.window=float(window_s);self.name=name
        self._hits={};self._lock=threading.Lock()
    def allow(self,key):
        now=time.time();cutoff=now-self.window
        with self._lock:
            q=self._hits.get(key)
            if q is None:q=[];self._hits[key]=q
            while q and q[0]<cutoff:q.pop(0)
            if len(q)>=self.max:
                retry=round(max(0.0,self.window-(now-q[0])),1)
                return (False,{'limited':True,'limit':self.max,'window_s':self.window,'retry_after_s':retry,'name':self.name})
            q.append(now)
            if len(self._hits)>5000:self._gc(cutoff)
            return (True,{'limited':False,'remaining':self.max-len(q)})
    def _gc(self,cutoff):
        for k in list(self._hits.keys()):
            qq=self._hits[k]
            while qq and qq[0]<cutoff:qq.pop(0)
            if not qq:
                try:del self._hits[k]
                except Exception:pass
def from_env(name,default_max,default_window=60):
    try:mx=int(os.environ.get(f'AMNI_RATE_{name.upper()}',default_max))
    except Exception:mx=default_max
    return RateLimiter(mx,default_window,name=name)
def _loop(h):
    try:return ipaddress.ip_address(str(h).strip('[]')).is_loopback
    except Exception:return str(h).lower()=='localhost'
def _peer(request):
    c=getattr(request,'client',None);return (c.host if c else '') or 'unknown'
def _hdrs(request):
    try:return getattr(request,'headers',None) or {}
    except Exception:return {}
def client_key(request):
    try:
        h=_peer(request);xf=str(_hdrs(request).get('x-forwarded-for') or '')
        return (xf.split(',')[-1].strip() or h) if (xf and _loop(h)) else h
    except Exception:return 'unknown'
def is_local(request):
    try:return _loop(_peer(request)) and not any(_hdrs(request).get(k) for k in _FWD)
    except Exception:return False
class TokenBucket:
    def __init__(self,rate_per_min,burst,name=''):
        self.rate=max(0.001,float(rate_per_min))/60.0;self.cap=max(1.0,float(burst));self.name=name;self._b={};self._lock=threading.Lock()
    def allow(self,key):
        now=time.monotonic()
        with self._lock:
            tok,ts=self._b.get(key,(self.cap,now));tok=min(self.cap,tok+(now-ts)*self.rate);ok=tok>=1.0
            self._b[key]=(tok-1.0 if ok else tok,now)
            if len(self._b)>5000:self._b={k:v for k,v in self._b.items() if v[0]+(now-v[1])*self.rate<self.cap}
            return (ok,{'limited':not ok,'limit':int(self.cap),'rate_per_min':round(self.rate*60,3),'retry_after_s':0.0 if ok else round((1.0-tok)/self.rate,1),'name':self.name,'remaining':int(tok-1.0) if ok else 0})
def bucket_from_env(name,rpm,burst):
    g=lambda k,d:float(os.environ.get(f'AMNI_RL_{name.upper()}_{k}',d) or d)
    try:return TokenBucket(g('RPM',rpm),g('BURST',burst),name=name)
    except Exception:return TokenBucket(rpm,burst,name=name)
