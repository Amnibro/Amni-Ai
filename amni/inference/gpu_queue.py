import threading,queue
class _GpuQueue:
    def __init__(self):
        self._q=queue.Queue()
        self._worker=threading.Thread(target=self._run,name='amni-gpu',daemon=True)
        self._worker.start()
    def _run(self):
        while True:
            fn,a,k,res=self._q.get()
            try:res['v']=fn(*a,**k)
            except BaseException as e:res['e']=e
            finally:res['done'].set()
    def submit(self,fn,*a,**k):
        if threading.current_thread() is self._worker:return fn(*a,**k)
        res={'done':threading.Event()}
        self._q.put((fn,a,k,res))
        res['done'].wait()
        if 'e' in res:raise res['e']
        return res.get('v')
    def submit_async(self,fn,*a,**k):
        res={'done':threading.Event()}
        if threading.current_thread() is self._worker:
            try:fn(*a,**k)
            finally:res['done'].set()
            return res['done']
        self._q.put((fn,a,k,res))
        return res['done']
GPU_QUEUE=_GpuQueue()
def run_on_gpu(fn,*a,**k):return GPU_QUEUE.submit(fn,*a,**k)
