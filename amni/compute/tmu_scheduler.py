import numpy as np,threading
from concurrent.futures import ThreadPoolExecutor,Future
from typing import Callable,List,Tuple,Optional
class TMUALUScheduler:
    __slots__=('_tmu_pool','_alu_pool','_pending','_lock')
    def __init__(self,tmu_workers:int=1,alu_workers:int=1):
        self._tmu_pool=ThreadPoolExecutor(max_workers=tmu_workers,thread_name_prefix="tmu")
        self._alu_pool=ThreadPoolExecutor(max_workers=alu_workers,thread_name_prefix="alu")
        self._pending:List[Future]=[]
        self._lock=threading.Lock()
    def submit_tmu(self,fn:Callable,*args)->Future:
        f=self._tmu_pool.submit(fn,*args)
        with self._lock:self._pending.append(f)
        return f
    def submit_alu(self,fn:Callable,*args)->Future:
        f=self._alu_pool.submit(fn,*args)
        with self._lock:self._pending.append(f)
        return f
    def barrier(self)->List:
        with self._lock:futs=list(self._pending);self._pending.clear()
        return[f.result()for f in futs]
    def close(self):
        self._tmu_pool.shutdown(wait=False)
        self._alu_pool.shutdown(wait=False)
_sched=None
def get_scheduler()->TMUALUScheduler:
    global _sched
    if _sched is None:_sched=TMUALUScheduler()
    return _sched
def pipeline_block_forward(blk,x:np.ndarray)->np.ndarray:
    from amni.compute.gf17_ops import gf17_rms_norm,gf17_add,gf17_fused_mlp,gf17_matmul_t,gf17_norm_matmul_t
    sc=get_scheduler()
    xn=gf17_rms_norm(x)
    q_fut=sc.submit_tmu(gf17_matmul_t,xn,blk.attn.q_proj.w)
    k_fut=sc.submit_tmu(gf17_matmul_t,xn,blk.attn.k_proj.w)
    v_fut=sc.submit_tmu(gf17_matmul_t,xn,blk.attn.v_proj.w)
    q,k,v=q_fut.result(),k_fut.result(),v_fut.result()
    B=1;S=x.shape[0]if x.ndim==2 else x.shape[1];H=blk.attn.n_heads;Hkv=blk.attn.n_kv_heads;Hd=blk.attn.head_dim
    xin=x[np.newaxis]if x.ndim==2 else x
    q=q.reshape(B,S,H,Hd);k=k.reshape(B,S,Hkv,Hd);v=v.reshape(B,S,Hkv,Hd)
    reps=blk.attn._reps
    if reps>1:k=np.repeat(k,reps,axis=2);v=np.repeat(v,reps,axis=2)
    attn_out=blk.attn._dsp_score_negacyclic(q,k,v)
    o=gf17_matmul_t(attn_out.reshape(B,S,blk.attn.hidden),blk.attn.o_proj.w)
    o=o[0]if x.ndim==2 else o
    x=gf17_add(x,o)
    mlp_out=gf17_fused_mlp(gf17_rms_norm(x),blk.mlp.gate.w,blk.mlp.up.w,blk.mlp.down.w)
    return gf17_add(x,mlp_out)
def pipeline_forward(model,token_ids:np.ndarray)->np.ndarray:
    from amni.compute.gf17_ops import gf17_rms_norm,gf17_matmul_t
    x=model.embed[token_ids]
    for blk in model.blocks:x=pipeline_block_forward(blk,x)
    x=gf17_rms_norm(x)
    last=x[:,-1,:]if x.ndim==3 else x[-1:]
    return gf17_matmul_t(last,model.head)
