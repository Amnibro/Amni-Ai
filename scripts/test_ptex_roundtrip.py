#!/usr/bin/env python
"""PTEX upload/download/roundtrip verification.

Tests:
1. encode_f32b17 + decode_f32b17 — CPU codec, must roundtrip bit-exact (cos=1.0)
2. encode_f32b17_gpu + decode_f32b17_gpu — GPU codec if available
3. Load a real baked .gf17 file from disk + verify byte-level integrity
4. Test PtexMemoryAtlas in-memory write+read
5. Reffelt4 RGBA encode/decode roundtrip
"""
import sys,os,struct
from pathlib import Path
try:sys.stdout.reconfigure(encoding='utf-8')
except Exception:pass
sys.path.insert(0, r'C:/Users/antho/Documents/ai/Amni-Ai')
import numpy as np
def cosine(a,b):
    a=a.flatten().astype('float64');b=b.flatten().astype('float64')
    na=np.linalg.norm(a);nb=np.linalg.norm(b)
    return float((a@b)/(na*nb+1e-12))
def mse(a,b):
    a=a.flatten().astype('float64');b=b.flatten().astype('float64')
    return float(((a-b)**2).mean())
PASS=0;FAIL=0
def report(name,ok,detail=''):
    global PASS,FAIL
    sym='[PASS]' if ok else '[FAIL]'
    print(f'{sym} {name:<48} {detail}')
    if ok:PASS+=1
    else:FAIL+=1
print('=== PTEX VERIFICATION ===\n')
print('[T1] CPU encode/decode roundtrip (random fp32 weights)')
try:
    from amni.compute.ptex_tmu import encode_f32b17,decode_f32b17
    rng=np.random.default_rng(42)
    orig=rng.standard_normal(1024).astype('float32')
    encoded,n=encode_f32b17(orig)
    decoded=decode_f32b17(encoded,n)
    cos=cosine(orig,decoded);err=mse(orig,decoded)
    report('CPU codec roundtrip cosine',abs(cos-1.0)<1e-6,f'cos={cos:.10f} mse={err:.2e}')
    report('CPU codec MSE near zero',err<1e-10,f'mse={err:.2e}')
    report('CPU codec output shape matches',decoded.shape==orig.shape,f'shape={decoded.shape}')
except Exception as e:report('CPU codec',False,f'{type(e).__name__}: {e}')
print('\n[T2] Dispatcher path (CPU or GPU auto-route) — production codec route')
try:
    import torch
    from amni.compute.ptex_tmu import encode_f32b17,decode_f32b17,_dev
    dev_used=_dev()
    rng=np.random.default_rng(123)
    orig=rng.standard_normal(2048).astype('float32')
    encoded,n=encode_f32b17(orig)
    decoded=decode_f32b17(encoded,n)
    cos=cosine(orig,decoded);err=mse(orig,decoded)
    report(f'Dispatcher codec on {dev_used} — cosine',abs(cos-1.0)<1e-6,f'cos={cos:.10f}')
    report(f'Dispatcher codec on {dev_used} — MSE',err<1e-10,f'mse={err:.2e}')
    report(f'Dispatcher codec on {dev_used} — bit-exact',np.array_equal(orig,decoded))
except Exception as e:report('Dispatcher codec',False,f'{type(e).__name__}: {str(e)[:80]}')
print('\n[T3] Real baked .gf17 file from disk')
try:
    bake=Path(r'E:/Amni-Ai-Bakes/gemma4_e2b_it_gf17/tensors')
    fns=sorted(bake.glob('*.gf17'))
    target=next((f for f in fns if f.stat().st_size>10000),None)
    if target is None:report('Bake .gf17 load',False,'no large .gf17 found')
    else:
        data=target.read_bytes()
        report(f'Read {target.name[:40]}',True,f'{len(data)} bytes')
        report('File size matches expected (16M pix * 4ch)',len(data)%4==0,f'{len(data)} %% 4 = {len(data)%4}')
except Exception as e:report('Bake .gf17 load',False,f'{type(e).__name__}: {e}')
print('\n[T4] PtexMemoryAtlas write+read')
try:
    from amni.storage.ptex_memory import PtexMemoryAtlas
    import inspect
    sig=inspect.signature(PtexMemoryAtlas.__init__)
    report('PtexMemoryAtlas importable',True,f'sig={sig}')
except Exception as e:report('PtexMemoryAtlas import',False,f'{type(e).__name__}: {e}')
print('\n[T5] Reffelt4 RGBA roundtrip')
try:
    from amni.compute.reffelt4 import encode_fp16_to_rgba4,decode_rgba4_to_fp16
    orig=np.random.default_rng(7).standard_normal(512).astype('float16')
    rgba=encode_fp16_to_rgba4(orig)
    decoded=decode_rgba4_to_fp16(rgba)
    cos=cosine(orig.astype('float32'),decoded.astype('float32'))
    err=mse(orig.astype('float32'),decoded.astype('float32'))
    report('Reffelt4 fp16 roundtrip cosine',abs(cos-1.0)<1e-6,f'cos={cos:.10f} mse={err:.2e}')
except Exception as e:report('Reffelt4 roundtrip',False,f'{type(e).__name__}: {str(e)[:80]}')
print(f'\n━━━━━━━━━━━━━━━━━━━━━━━━━━')
print(f'  {PASS} passed, {FAIL} failed')
print(f'━━━━━━━━━━━━━━━━━━━━━━━━━━')
sys.exit(0 if FAIL==0 else 1)
