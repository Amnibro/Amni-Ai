import os,sys,time,json
try:sys.stdout.reconfigure(encoding='utf-8',errors='replace');sys.stderr.reconfigure(encoding='utf-8',errors='replace')
except Exception:pass
for k,v in (('HIP_VISIBLE_DEVICES','1'),('PYTORCH_ROCM_ARCH','gfx1101'),('HSA_OVERRIDE_GFX_VERSION','11.0.0'),('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL','1'),('HIP_FORCE_DEV_KERNARG','1'),('MIOPEN_FIND_MODE','2'),('AMNI_BLOCK_SPEC','0')):os.environ.setdefault(k,v)
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.adam import Adam
from amni.serve.skills import default_registry
from amni.serve.agentic import _parse_step
GOAL='Implement ntt4_gf17(x) (4-point NTT over GF(17), omega=4) and its exact inverse intt4_gf17(X) so that intt4_gf17(ntt4_gf17(x))==x for every length-4 vector x of values 0..16 (reversible/bit-exact).'
PINNED='omega=4; powers 4^0=1,4^1=4,4^2=16,4^3=13; build W[k,n]=pow(4,k*n,17); inverse Winv[k,n]=pow(13,k*n,17) then *13 %17.'
WRONG=Path('intt_explore/intt_sketch.py').read_text(encoding='utf-8')
CORRECT='''import numpy as np
def ntt4_gf17(x):
    W=np.array([[pow(4,k*n,17) for n in range(4)] for k in range(4)],dtype=np.int64)
    return (W@np.array(x,dtype=np.int64))%17
def intt4_gf17(X):
    Wi=np.array([[pow(13,k*n,17) for n in range(4)] for k in range(4)],dtype=np.int64)
    return ((Wi@np.array(X,dtype=np.int64))%17*13)%17
if __name__=="__main__":
    import numpy as _np
    for t in ([1,2,3,4],[0,1,2,3]):
        assert _np.array_equal(intt4_gf17(ntt4_gf17(t)),_np.array(t)%17)
    print("PASS")
'''
reg=default_registry(workdir='intt_explore')
def exec_critic(adam,art,label):
    sys_p='You are a strict adversarial reviewer of your own work. Default to skepticism. Prefer an executable test over hand-tracing. Output ONLY the JSON line.'
    msg=('You are reviewing your OWN finished work with a SKEPTICAL eye. You are a 3B model: do NOT trust your hand-arithmetic, trust the executable test.\n\nGOAL:\n'+GOAL+'\n\nPINNED FACTS (code MUST match these):\n- '+PINNED+'\n\nThe file:\n```\n'+art[:3000]+'\n```\n\nProvide an executable "test": a few lines of Python that — ASSUMING the code above is ALREADY defined in scope (do NOT re-import/redefine it) — ASSERT that for many random length-4 vectors x with values 0..16, intt4_gf17(ntt4_gf17(x)) returns x exactly; raise AssertionError if not.\nOutput ONE JSON line:\n{"acceptable":true or false,"fault":"<defect or empty>","fix":"<next action or empty>","test":"<executable python>"}')
    r=adam.chat_persona(message=msg,system=sys_p,max_new_tokens=520,do_sample=False,history=None,facts=None,is_private=True)
    raw=r.get('answer') if isinstance(r,dict) else str(r)
    cv=_parse_step(raw or '')
    if cv is None:print(f'[{label}] could not parse critic JSON. raw={str(raw)[:300]}');return
    acc=cv.get('acceptable');acc=(acc is True) or (str(acc).strip().lower() in ('true','yes','1','ok','acceptable','accept'))
    test=str(cv.get('test') or '').strip();tres='none'
    if test:
        tr=reg.call('run_python',{'code':art+'\n'+test,'timeout':25})
        to=tr.output if tr.ok else {'error':str(tr.error)}
        if isinstance(to,dict):
            rc=to.get('returncode');tres=('timeout' if (to.get('timed_out') or to.get('killed')) else ('blocked' if to.get('error') else ('pass' if rc==0 else 'fail')))
            if tres=='fail':acc=False
            elif tres=='pass' and not str(cv.get('fault') or ''):acc=True
    print(f'[{label}] model-said-acceptable_initial={str(cv.get("acceptable"))[:12]} | EXECUTED-TEST={tres} | FINAL acceptable={acc} | fault={str(cv.get("fault"))[:120]}')
    return tres,acc
t0=time.time();adam=Adam(bake='bakes/granite41_3b_gf17',model='bakes/granite41_3b_gf17',web_unrestricted=False);print(f'[probe] Adam loaded {time.time()-t0:.1f}s',flush=True)
print('\n===== EXECUTING-CRITIC UNIT TEST =====',flush=True)
print('(expect: WRONG artifact -> EXECUTED-TEST=fail -> FINAL acceptable=False ; CORRECT artifact -> pass -> True)\n',flush=True)
w=exec_critic(adam,WRONG,'WRONG artifact (garbled matrix, single-input self-test masks it)')
c=exec_critic(adam,CORRECT,'CORRECT artifact (pow-built matrices)')
print('\n[RESULT]', 'PASS — executing critic REJECTS wrong, AFFIRMS correct' if (w and w[1]==False and c and c[1]==True) else 'INCONCLUSIVE (see above)',flush=True)
