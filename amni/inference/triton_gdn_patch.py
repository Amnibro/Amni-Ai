import sys
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(_ROOT))
from amni.compute.triton_gdn import causal_conv1d_fn,causal_conv1d_update
_PATCH_TARGETS=('transformers.models.qwen3_5.modeling_qwen3_5','transformers.models.qwen3_5_moe.modeling_qwen3_5_moe')
def apply(verbose=False):
    import importlib
    patched=[]
    for mod_name in _PATCH_TARGETS:
        try:m=importlib.import_module(mod_name)
        except Exception as e:
            if verbose:print(f'  [skip] {mod_name}: {e}')
            continue
        if hasattr(m,'causal_conv1d_fn'):
            m.causal_conv1d_fn=causal_conv1d_fn
            m.causal_conv1d_update=causal_conv1d_update
            patched.append(mod_name)
            if verbose:print(f'  [patched] {mod_name}.causal_conv1d_fn / causal_conv1d_update')
    return patched
def reattach_to_model(model,verbose=False):
    n=0
    for name,mod in model.named_modules():
        if hasattr(mod,'causal_conv1d_fn') and hasattr(mod,'causal_conv1d_update'):
            mod.causal_conv1d_fn=causal_conv1d_fn
            mod.causal_conv1d_update=causal_conv1d_update
            n+=1
            if verbose:print(f'  [reattach] {name}')
    return n
