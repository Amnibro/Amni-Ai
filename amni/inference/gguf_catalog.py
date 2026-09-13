"""Scan local GGUFs and pick one that fits the card.

Adam's daily driver is still the Granite GF17 bake. A GGUF here is an optional
heavier brain — Qwen3.8-27B Q3_K_P is the 16GB-card pick from the HauhauCS repo.
"""
import os,re
from pathlib import Path
from typing import Optional,Dict,List
_ROOT=Path(__file__).resolve().parents[2]
_MODELS=_ROOT/'models'
_HAUHAU_STEM='Qwen3.8-27B-Uncensored-HauhauCS-Aggressive'
_SKIP_NAME=re.compile(r'(mmproj|fastmtp|draft)',re.I)
def _scan_roots()->List[Path]:
    extra=[]
    raw=os.environ.get('AMNI_GGUF_PATHS') or ''
    for p in raw.replace(';',os.pathsep).split(os.pathsep):
        if p.strip():extra.append(Path(p.strip()))
    if os.environ.get('AMNI_GGUF_ONLY'):return extra or [_MODELS]
    return [_MODELS]+extra
def list_ggufs()->List[Dict]:
    out=[]
    for root in _scan_roots():
        if not root.exists():continue
        for p in root.rglob('*.gguf'):
            if not p.is_file():continue
            if _SKIP_NAME.search(p.name):continue
            try:sz=p.stat().st_size
            except Exception:continue
            if sz<50_000_000:continue
            q=_quant_of(p.name)
            out.append({
                'id':_id_for(p),
                'name':p.stem,
                'path':str(p),
                'quant':q,
                'size_gb':round(sz/1e9,2),
                'family':_family_of(p),
                'fits_16gb':sz<=14.2e9,
                'sidecar_fastmtp':str(_sidecar(p,'FastMTP')) if _sidecar(p,'FastMTP') else None,
            })
    out.sort(key=lambda m:(0 if 'hauhau' in m['family'] else 1,m['size_gb']))
    return out
def _id_for(p:Path)->str:
    return re.sub(r'[^a-z0-9]+','-',p.stem.lower()).strip('-')[:48]
def _quant_of(name:str)->str:
    m=re.search(r'(IQ\d+_[A-Z]+|Q\d+_K(?:_[A-Z]+)?|Q\d+_0|Q\d+_1)',name,re.I)
    return m.group(1).upper() if m else 'unknown'
def _family_of(p:Path)->str:
    n=p.name.lower()
    if 'hauhau' in n or 'qwen3.8-27b' in n:return 'hauhau-qwen38-27b'
    if 'qwen3.5-9b' in n:return 'qwen35-9b'
    if 'qwen3.5-2b' in n:return 'qwen35-2b'
    return 'gguf'
def _sidecar(p:Path,kind:str)->Optional[Path]:
    for c in p.parent.glob(f'*{kind}*.gguf'):
        if c.is_file():return c
    return None
def hauhau_q3_path()->Optional[Path]:
    named=_MODELS/'HauhauCS'/'Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF'/f'{_HAUHAU_STEM}-Q3_K_P.gguf'
    if named.exists() and named.stat().st_size>1_000_000_000:return named
    for m in list_ggufs():
        if m['family']=='hauhau-qwen38-27b' and 'Q3_K' in m['quant']:
            return Path(m['path'])
    return None
def recommended_gguf(vram_gb:float=16.0)->Optional[Dict]:
    cands=list_ggufs()
    if not cands:return None
    hau=[m for m in cands if m['family']=='hauhau-qwen38-27b']
    pool=hau or cands
    if vram_gb<18:
        fit=[m for m in pool if m['fits_16gb']]
        return (fit or pool)[0]
    pool.sort(key=lambda m:-m['size_gb'])
    return pool[0]
def status(vram_gb:float=0.0)->Dict:
    items=list_ggufs()
    rec=recommended_gguf(vram_gb or 16.0)
    return {
        'n':len(items),
        'models':items,
        'recommended':rec,
        'hauhau_q3':str(hauhau_q3_path() or ''),
        'note':'Q3_K_P (~13.4GB) is the 16GB-card pick. Q4_K_P (~18GB) needs offload or a bigger GPU.',
    }
