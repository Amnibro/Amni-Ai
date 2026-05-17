"""GrowthDaemon — background process that consumes low-confidence questions from a JSONL queue, crawls web sources, distills via Gemma, records lessons to PTEX.
Architecture: Adam.answer() returns immediately with best-guess + queues background work to a JSONL file. Daemon process tails the queue, processes items at its own pace, updates the shared lesson PTEX KB. Adam's next call benefits from accumulated lessons.
Communication via filesystem: zero shared-memory, robust to crashes, daemon can be restarted.
"""
import json,os,time,signal,sys
from pathlib import Path
from typing import Optional
class GrowthQueue:
    def __init__(self,queue_path:str):
        self.path=Path(queue_path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.touch(exist_ok=True)
        self._cursor_path=self.path.with_suffix('.cursor')
    def enqueue(self,question:str,subject:Optional[str]=None,confidence:float=0.0,meta:Optional[dict]=None):
        rec={'ts':time.time(),'question':question,'subject':subject,'confidence':confidence,'meta':meta or {}}
        with open(self.path,'a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    def cursor(self)->int:
        try:return int(self._cursor_path.read_text())
        except Exception:return 0
    def set_cursor(self,n:int):
        tmp=self._cursor_path.with_suffix('.cursor.tmp');tmp.write_text(str(n))
        os.replace(str(tmp),str(self._cursor_path))
    def pending(self):
        cur=self.cursor();out=[];i=0
        with open(self.path,'r',encoding='utf-8') as f:
            for line in f:
                if i>=cur:
                    try:out.append(json.loads(line))
                    except Exception:pass
                i+=1
        return out,i
    def stats(self):
        cur=self.cursor()
        with open(self.path,'r',encoding='utf-8') as f:total=sum(1 for _ in f)
        return {'total':total,'processed':cur,'pending':total-cur}
class GrowthDaemon:
    def __init__(self,adam_loop,queue_path:str,poll_sec:float=5.0,max_per_cycle:int=5):
        self.adam=adam_loop
        self.q=GrowthQueue(queue_path)
        self.poll=poll_sec;self.max_per_cycle=max_per_cycle
        self._running=False
    def run_one_cycle(self):
        items,new_cursor=self.q.pending()
        if not items:return 0
        items=items[:self.max_per_cycle]
        n=0
        for it in items:
            q=it.get('question','')
            if not q:continue
            try:
                if self.adam.crawler is not None:
                    ans,sources,_=self.adam.crawler.crawl_and_distill(q,subject=it.get('subject'),letter_only=False)
                    if ans:
                        self.adam.record_lesson(q,ans,subject=it.get('subject'),reasoning=f'Crawled sources:\n'+'\n'.join(sources[:3]),auto_concept=True)
                        n+=1
                else:
                    ans,tier,_=self.adam.answer(q,writeback=True);n+=1
            except Exception as e:print(f'[daemon] item failed: {e}',flush=True)
        self.q.set_cursor(self.q.cursor()+len(items))
        return n
    def run_forever(self):
        self._running=True
        def _sig(*a):self._running=False
        try:signal.signal(signal.SIGINT,_sig);signal.signal(signal.SIGTERM,_sig)
        except Exception:pass
        print(f'[daemon] starting, poll={self.poll}s max_per_cycle={self.max_per_cycle}',flush=True)
        while self._running:
            try:
                n=self.run_one_cycle()
                if n>0:print(f'[daemon] processed {n} items, queue={self.q.stats()}',flush=True)
            except Exception as e:print(f'[daemon] cycle error: {e}',flush=True)
            time.sleep(self.poll)
        print(f'[daemon] shut down cleanly, final stats={self.q.stats()}',flush=True)
