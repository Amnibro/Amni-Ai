import os,sys,time,json
try:sys.stdout.reconfigure(encoding='utf-8',errors='replace');sys.stderr.reconfigure(encoding='utf-8',errors='replace')
except Exception:pass
for k,v in (('HIP_VISIBLE_DEVICES','1'),('PYTORCH_ROCM_ARCH','gfx1101'),('HSA_OVERRIDE_GFX_VERSION','11.0.0'),('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL','1'),('HIP_FORCE_DEV_KERNARG','1'),('MIOPEN_FIND_MODE','2'),('AMNI_BLOCK_SPEC','0')):os.environ.setdefault(k,v)
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.adam import Adam
bake='bakes/granite41_3b_gf17'
code=Path('intt_explore/intt_sketch.py').read_text(encoding='utf-8')
t0=time.time();adam=Adam(bake=bake,model=bake,web_unrestricted=False);print(f'[probe] Adam loaded {time.time()-t0:.1f}s',flush=True)
SYS='You are a meticulous senior engineer reviewing your own prior work with a SKEPTICAL eye. Do not be agreeable. Your job is to find faults, not to praise. If the code is wrong, say exactly why.'
NEUTRAL=f'''Here is a function that is SUPPOSED to be a 4-point Number-Theoretic Transform over GF(17) whose INVERSE recovers the original input, i.e. applying the transform twice (or a forward then a true inverse) must return the starting vector for EVERY input of four values in 0..16.

```python
{code}
```

Examine it critically and answer THREE things:
1. Does it actually have a correct inverse? Trace inverse(forward([1,2,3,4])) by hand and state the numbers.
2. Is `w = 3` a correct 4th root of unity mod 17 (does 3^4 = 1 mod 17)? If not, what is the correct value?
3. Is anything MISSING (e.g. a separate inverse function)?
Give a final verdict line: ACCEPTABLE or FAULTY, and if FAULTY, the single most important fix.'''
print('\n========== PROBE A: neutral critical review (no hint) ==========\n',flush=True)
a=adam.chat_persona(message=NEUTRAL,system=SYS,max_new_tokens=520,do_sample=False,history=None,facts=None,is_private=True)
print(a if isinstance(a,str) else json.dumps(a)[:4000],flush=True)
print('\n========== PROBE B: direct self-affirmation question ==========\n',flush=True)
AFFIRM=f'''You previously wrote this code to solve the task "implement a 4-point GF(17) NTT and its exact inverse":

```python
{code}
```

Question: Is this answer acceptable and true? Does inverse(forward(x)) == x hold for all inputs? Answer honestly with YES it is correct, or NO it is faulty — and if NO, name the fault in one sentence.'''
b=adam.chat_persona(message=AFFIRM,system=SYS,max_new_tokens=320,do_sample=False,history=None,facts=None,is_private=True)
print(b if isinstance(b,str) else json.dumps(b)[:3000],flush=True)
print('\n[probe] done',flush=True)
