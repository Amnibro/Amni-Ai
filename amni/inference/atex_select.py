"""Bake/svc picker for Adam(bake=...). Torch-free so CI can unit-test routing.

Adam historically treated every Gf17Atex bake_manifest (gs + q=1 codes/scale) as
GraniteAtexChatService. A Qwen3 bake (name like qwen3_8b_gf17_atex_probe, or
config.json model_type/architectures Qwen3*) must instead boot QwenAtexChatService
so Kimahri's Gf17Atex-shaped directory and a plain HF Qwen3 checkpoint share one path.
"""
import json
from pathlib import Path

_QWEN3_NAME_MARKERS=('qwen3','qwen-3')
_QWEN3_MODEL_TYPES=('qwen3','qwen3_5','qwen3_5_moe')
_QWEN3_ARCH_PREFIXES=('qwen3',)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return None


def load_bake_manifest(bake):
    if not bake:return {}
    p=Path(bake)/'bake_manifest.json'
    return _read_json(p) or {}


def load_model_config(path):
    if not path:return {}
    p=Path(path)
    cfg=_read_json(p/'config.json') if p.is_dir() else None
    if cfg is None and p.is_file() and p.name=='config.json':cfg=_read_json(p)
    return cfg or {}


def is_nvfp4_manifest(man):
    return str((man or {}).get('format') or '').startswith('nvfp4')


def is_gf17_atex_manifest(man):
    """Same contract Gf17Atex/Granite use: tensors with q=1 mean .codes + .scale (int4-group)."""
    tens=(man or {}).get('tensors') or {}
    return any((t or {}).get('q')==1 for t in tens.values())


def is_gf17_atex_bake(path):
    man=load_bake_manifest(path)
    return bool(man) and is_gf17_atex_manifest(man)


def is_streaming_bake(path):
    """Palette/tilepack/Reffelt streaming bake (manifest.json), not Gf17Atex codes/scale."""
    p=Path(path) if path else None
    if p is None or not p.is_dir():return False
    if is_gf17_atex_bake(p):return False
    m=_read_json(p/'manifest.json') or {}
    if not m.get('tensors'):return False
    scheme=str(m.get('scheme') or m.get('reffelt_scheme') or '')
    return bool(scheme) or True


def is_hf_weight_dir(path):
    p=Path(path) if path else None
    if p is None or not p.is_dir():return False
    if not (p/'config.json').exists():return False
    if is_gf17_atex_bake(p):return False
    return any(p.glob('*.safetensors'))


def looks_like_qwen3(path=None,man=None,cfg=None):
    """True when bake name, bake_manifest, or HF config.json indicates Qwen3 / Qwen3.5."""
    if path:
        name=Path(path).name.lower().replace('-','_')
        blob=str(path).lower().replace('-','_')
        if any(m.replace('-','_') in name or m.replace('-','_') in blob for m in _QWEN3_NAME_MARKERS):
            return True
    man=man or {}
    for key in ('arch','architectures','model','model_type','source','base_model','family'):
        v=man.get(key)
        if v is None:continue
        s=json.dumps(v).lower() if not isinstance(v,str) else v.lower()
        if 'qwen3' in s.replace('-','').replace('_',''):return True
        if 'qwen3' in s:return True
    cfg=cfg or {}
    mt=str(cfg.get('model_type') or '').lower()
    if mt in _QWEN3_MODEL_TYPES or mt.startswith('qwen3'):return True
    archs=cfg.get('architectures') or []
    if isinstance(archs,str):archs=[archs]
    for a in archs:
        al=str(a).lower()
        if any(al.startswith(p) for p in _QWEN3_ARCH_PREFIXES):return True
        if 'qwen3' in al:return True
    return False


def select_svc_kind(bake,model=None):
    """Return nvfp4 | qwen_atex | granite_atex | streaming.

    Verified Adam hook (amni/adam.py):
      1. NVFP4 format in bake_manifest -> Nvfp4AtexChatService
      2. Qwen3 (name / config / manifest) -> QwenAtexChatService
         (Gf17Atex bake dir OR plain HF safetensors; the svc picks the load path)
      3. Other Gf17Atex (gs + q=1) -> GraniteAtexChatService
      4. Else StreamingChatService (palette/tilepack/CPU stream)
    """
    bake=bake or ''
    model=model or bake
    man=load_bake_manifest(bake)
    cfg=load_model_config(bake) or load_model_config(model)
    if is_nvfp4_manifest(man):return 'nvfp4'
    qwen=looks_like_qwen3(bake,man,cfg) or looks_like_qwen3(model,None,cfg)
    if qwen:return 'qwen_atex'
    if is_gf17_atex_manifest(man):return 'granite_atex'
    return 'streaming'
