"""Detect Qwen3.5 / 3.8 Gated DeltaNet (Mamba-class) configs without loading weights."""
_GDN_ARCHS=('Qwen3_5ForCausalLM','Qwen3_5MoeForCausalLM','Qwen3_5ForConditionalGeneration','Qwen3_8ForCausalLM','Qwen3_8ForConditionalGeneration','MiniMaxText01ForCausalLM','Qwen3CoderNextForCausalLM')
def is_gdn_config(cfg)->bool:
    if cfg is None:return False
    if isinstance(cfg,dict):
        archs=tuple(cfg.get('architectures') or ())
        lt=cfg.get('layer_types')
        text=cfg.get('text_config') or {}
        if not archs:archs=tuple(text.get('architectures') or ())
        if lt is None:lt=text.get('layer_types')
    else:
        archs=tuple(getattr(cfg,'architectures',None) or ())
        lt=getattr(cfg,'layer_types',None)
        text=getattr(cfg,'text_config',None)
        if not archs and text is not None:archs=tuple(getattr(text,'architectures',None) or ())
        if lt is None and text is not None:lt=getattr(text,'layer_types',None)
    if any(a in _GDN_ARCHS for a in archs):return True
    if lt:
        seq=lt if isinstance(lt,(list,tuple)) else [lt]
        return any(('linear_attention' in str(x).lower()) or ('gated_delta' in str(x).lower()) or ('mamba' in str(x).lower()) for x in seq)
    return False
