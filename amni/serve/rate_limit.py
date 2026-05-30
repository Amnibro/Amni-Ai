"""rate_limit — in-process per-key sliding-window rate limiter to blunt flooding/abuse on public endpoints
(a runaway client or loop can't hammer /teach or /chat). Thread-safe; self-GCs idle keys. Tunable via env."""
import time,threading,os
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
def client_key(request):
    try:
        c=getattr(request,'client',None)
        return (c.host if c else 'unknown') or 'unknown'
    except Exception:return 'unknown'
