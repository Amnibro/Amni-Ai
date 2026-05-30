"""v6.10.136 — single-thread GPU serialization queue: all GPU work runs on ONE worker thread (no concurrent kernel launches), preserving FIFO order, re-entrant-safe, propagating results/exceptions."""
import sys,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.inference.gpu_queue import GPU_QUEUE,run_on_gpu
def test_all_jobs_run_on_one_thread():
    seen=set()
    def job():seen.add(threading.current_thread().ident);return 1
    threads=[threading.Thread(target=lambda:run_on_gpu(job)) for _ in range(8)]
    [t.start() for t in threads];[t.join() for t in threads]
    assert len(seen)==1,f'all GPU jobs must execute on a single worker thread, saw {len(seen)}'
def test_no_overlap_under_concurrency():
    active=[0];max_seen=[0];lk=threading.Lock()
    def job():
        with lk:active[0]+=1;max_seen[0]=max(max_seen[0],active[0])
        time.sleep(0.01)
        with lk:active[0]-=1
    threads=[threading.Thread(target=lambda:run_on_gpu(job)) for _ in range(12)]
    [t.start() for t in threads];[t.join() for t in threads]
    assert max_seen[0]==1,f'GPU jobs must never overlap, peak concurrency was {max_seen[0]}'
def test_result_and_exception_propagation():
    assert run_on_gpu(lambda:7*6)==42
    try:run_on_gpu(lambda:1/0);assert False,'exception must propagate to caller'
    except ZeroDivisionError:pass
def test_reentrant_no_deadlock():
    def inner():return 'inner'
    def outer():return run_on_gpu(inner)+'+outer'
    assert run_on_gpu(outer)=='inner+outer','a GPU job calling run_on_gpu must run inline, not deadlock'
def test_submit_async_signals_completion():
    box={}
    done=GPU_QUEUE.submit_async(lambda:box.setdefault('ran',True))
    assert done.wait(timeout=2.0),'async job must signal completion'
    assert box.get('ran') is True
def test_fifo_order():
    order=[]
    def mk(i):
        return lambda:order.append(i)
    threads=[]
    for i in range(6):
        run_on_gpu(mk(i))
    assert order==[0,1,2,3,4,5],f'serial submits must run in order, got {order}'
def test_streaming_chat_uses_queue():
    src=(Path(__file__).resolve().parents[1]/'amni/inference/streaming_chat.py').read_text(encoding='utf-8')
    assert 'from amni.inference.gpu_queue import' in src
    assert 'GPU_QUEUE.submit_async' in src,'streaming generation must run on the GPU worker, not its own thread'
    assert 'run_on_gpu' in src,'blocking generate paths must route through the queue'
def test_embedder_uses_queue():
    src=(Path(__file__).resolve().parents[1]/'amni/inference/semantic_ptex_lut.py').read_text(encoding='utf-8')
    assert 'run_on_gpu' in src,'the background embedder must serialize through the same GPU queue'
