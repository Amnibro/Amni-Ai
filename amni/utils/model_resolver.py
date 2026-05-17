import os,json
from pathlib import Path
from typing import Optional,Dict,List,Tuple
_PROJECT_ROOT=Path(__file__).resolve().parents[2]
_MODELS_DIR=_PROJECT_ROOT/"models"
_QWEN35_9B_Q8_REL=Path("Qwen3.5-9B")/"Qwen3.5-9B-Q8_0.gguf"
_QWEN35_9B_Q4_REL=Path("Qwen3.5-9B")/"Qwen3.5-9B-Q4_K_M.gguf"
_JACKRONG_9B_REL=Path("Jackrong")/"Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"/"Qwen3.5-9B.Q8_0.gguf"
_LUFFY_9B_Q8_REL=Path("LuffyTheFox")/"Qwen3.5-9B-Claude-4.6-Opus-Uncensored-Distilled-GGUF"/"Qwen3.5-9B.Q8_0.gguf"
_LUFFY_9B_Q4_REL=Path("LuffyTheFox")/"Qwen3.5-9B-Claude-4.6-Opus-Uncensored-Distilled-GGUF"/"Qwen3.5-9B.Q4_K_M.gguf"
_2B_QUANTS=["Qwen3.5-2B.Q3_K_L.gguf","Qwen3.5-2B.Q3_K_M.gguf","Qwen3.5-2B.Q3_K_S.gguf","Qwen3.5-2B.Q2_K.gguf"]
_DELIB_PORT=int(os.environ.get("AMNI_DELIB_PORT","8787"))
_SCOUT_PORT=int(os.environ.get("AMNI_SCOUT_PORT","8788"))
_DELIB_URL=os.environ.get("AMNI_DELIB_URL",f"http://127.0.0.1:{_DELIB_PORT}")
_SCOUT_URL=os.environ.get("AMNI_SCOUT_URL",f"http://127.0.0.1:{_SCOUT_PORT}")
def _adam_weight_path(mid:str)->Optional[Path]:
    weights_dir=_PROJECT_ROOT/"textures"/"models"
    candidates=[
        _MODELS_DIR/f"{mid}.safetensors",
        _MODELS_DIR/"adam"/f"{mid}.safetensors",
        weights_dir/mid/f"{mid}.safetensors",
        weights_dir/mid/"meta.json",
        weights_dir/f"{mid}-b17"/"meta.json",
        _PROJECT_ROOT/"checkpoints"/"continuous"/mid/f"{mid}_c1_final.npz",
    ]
    return next((p for p in candidates if p.exists()),None)
# --- PTEX model discovery ---
def ptex_models()->List[Dict]:
    """Scan models dir for PTEX manifest.json files and return model descriptors."""
    found=[]
    scan_dirs=[_MODELS_DIR]
    for scan in scan_dirs:
        if not scan.is_dir():continue
        for mdir in scan.iterdir():
            if not mdir.is_dir():continue
            mf=mdir/"manifest.json"
            if not mf.exists():continue
            try:
                d=json.loads(mf.read_text(encoding="utf-8"))
                if d.get("format")!="ptex":continue
                mid=mdir.name
                base_model=d.get("model",mid)
                n_layers=d.get("n_layers",d.get("n_layers",len(d.get("layers",{}))))
                hidden=d.get("hidden",0)
                mode_name=d.get("mode_name","PTEX")
                # Count actual ptex files (not .ptex.json metadata)
                sz_bytes=sum(f.stat().st_size for f in mdir.glob("*.ptex") if f.is_file())
                found.append({
                    "id":mid,"name":f"{base_model} [PTEX {mode_name}/{mid}]",
                    "path":str(mdir),"role":"ptex","engine":"prismtex",
                    "n_layers":n_layers,"hidden":hidden,"mode_name":mode_name,
                    "size_mb":round(sz_bytes/1048576,1),"available":True,
                    "model":base_model
                })
            except Exception:continue
    found.sort(key=lambda x:x["id"])
    return found
# --- Gemma native safetensors discovery ---
def gemma_native_models()->List[Dict]:
    """Find native safetensors Gemma models under models/google/."""
    found=[]
    gdir=_MODELS_DIR/"google"
    if not gdir.exists():return found
    for mdir in gdir.iterdir():
        if not mdir.is_dir():continue
        cfg_f=mdir/"config.json"
        if not cfg_f.exists():continue
        try:
            cfg=json.loads(cfg_f.read_text(encoding="utf-8"))
            arch=cfg.get("model_type","")
            if "gemma" not in arch.lower():continue
            shards=[f for f in mdir.glob("*.safetensors") if f.is_file()]
            if not shards:continue
            sz=sum(f.stat().st_size for f in shards)
            mid=mdir.name
            found.append({
                "id":mid,"name":f"{mid} [native safetensors]",
                "path":str(mdir),"role":"native","engine":"hf",
                "size_mb":round(sz/1048576,1),"available":True,
                "model":mid
            })
        except Exception:continue
    found.sort(key=lambda x:x["id"])
    return found
# --- Adam model discovery ---
def adam_models()->List[Dict]:
    """Return Adam model configs that have saved weights available."""
    try:
        from amni.model.adam import ADAM_CONFIGS
    except Exception:
        return []
    found=[]
    for mid,cfg in ADAM_CONFIGS.items():
        path=_adam_weight_path(mid)
        available=path is not None
        sz=path.stat().st_size if path else 0
        hidden=cfg.get("hidden",0)
        n_blocks=cfg.get("n_blocks",0)
        found.append({
            "id":mid,"name":f"{mid} [Adam GF17, h={hidden}, b={n_blocks}]",
            "path":str(path) if path else "","role":"adam","engine":"gf17",
            "size_mb":round(sz/1048576,1),"available":available,
            "model":mid,"hidden":hidden,"n_blocks":n_blocks
        })
    found.sort(key=lambda x:x["id"])
    return found
def deliberator_url()->str:
    return _DELIB_URL
def scout_url()->str:
    return _SCOUT_URL
def scout_available()->bool:
    import urllib.request
    try:
        req=urllib.request.Request(f"{_SCOUT_URL}/health")
        with urllib.request.urlopen(req,timeout=2) as r:
            import json
            return json.loads(r.read().decode()).get("status")=="ok"
    except Exception:
        return False
def project_root()->Path:
    return _PROJECT_ROOT
def models_dir()->Path:
    return _MODELS_DIR
def qwen35_9b_q8_path()->Optional[Path]:
    p=_MODELS_DIR/_QWEN35_9B_Q8_REL
    return p if p.exists() and p.stat().st_size>500_000_000 else None
def qwen35_9b_q4_path()->Optional[Path]:
    p=_MODELS_DIR/_QWEN35_9B_Q4_REL
    return p if p.exists() and p.stat().st_size>500_000_000 else None
def jackrong_9b_path()->Optional[Path]:
    p=_MODELS_DIR/_JACKRONG_9B_REL
    return p if p.exists() else None
def luffy_9b_q8_path()->Optional[Path]:
    p=_MODELS_DIR/_LUFFY_9B_Q8_REL
    return p if p.exists() else None
def luffy_9b_q4_path()->Optional[Path]:
    p=_MODELS_DIR/_LUFFY_9B_Q4_REL
    return p if p.exists() else None
def best_2b_path()->Optional[Path]:
    for name in _2B_QUANTS:
        p=_MODELS_DIR/name
        if p.exists() and p.stat().st_size>100_000_000:
            return p
    return None
def all_2b_paths()->List[Path]:
    return [_MODELS_DIR/n for n in _2B_QUANTS if (_MODELS_DIR/n).exists() and (_MODELS_DIR/n).stat().st_size>100_000_000]
def get_active_gguf()->Optional[Path]:
    return qwen35_9b_q8_path() or qwen35_9b_q4_path() or jackrong_9b_path() or luffy_9b_q8_path() or luffy_9b_q4_path()
def is_local_ready()->bool:
    return get_active_gguf() is not None
def active_model_name()->str:
    active=os.environ.get("AMNI_ACTIVE_MODEL","").strip()
    if active:return active
    if qwen35_9b_q8_path():return "Qwen3.5-9B (local Q8_0, WSL ROCm Railgun)"
    if qwen35_9b_q4_path():return "Qwen3.5-9B (local Q4_K_M, WSL ROCm Railgun)"
    if jackrong_9b_path():return "Qwen3.5-9B-Claude-4.6-Opus-Reasoning (Jackrong Q8)"
    if luffy_9b_q4_path():return "Qwen3.5-9B-Claude-4.6-Opus-Uncensored (LuffyTheFox Q4_K_M)"
    if best_2b_path():return "Qwen3.5-2B ({})".format(best_2b_path().stem)
    return "none"
def model_roster()->List[Dict]:
    roster=[]
    checks=[
        ("qwen35-9b-q8","Qwen3.5-9B local Q8_0 (WSL ROCm Railgun)",qwen35_9b_q8_path,"proposer",9500),
        ("qwen35-9b-q4","Qwen3.5-9B local Q4_K_M (WSL ROCm Railgun)",qwen35_9b_q4_path,"proposer",5500),
        ("jackrong-9b-q8","Jackrong 9B Q8 (Reasoning)",jackrong_9b_path,"proposer",9086),
        ("luffy-9b-q4","LuffyTheFox 9B Q4 (Uncensored)",luffy_9b_q4_path,"auditor",5366),
        ("luffy-9b-q8","LuffyTheFox 9B Q8 (Uncensored)",luffy_9b_q8_path,"heavy",9086),
    ]
    for mid,name,fn,role,mb in checks:
        p=fn()
        if p:roster.append({"id":mid,"name":name,"path":str(p),"role":role,"size_mb":mb,"available":True})
    for bp in all_2b_paths():
        roster.append({"id":"2b-{}".format(bp.stem.lower()),"name":"Qwen3.5-2B ({})".format(bp.stem),"path":str(bp),"role":"lightweight","size_mb":int(bp.stat().st_size/1048576),"available":True})
    # PTEX models (custom GF17 texture format — runs via Prismtex engine)
    for m in ptex_models():
        roster.append(m)
    # Gemma native safetensors models
    for m in gemma_native_models():
        roster.append(m)
    # Adam GF17 native models (show all configs, available=True only when weights exist)
    for m in adam_models():
        roster.append(m)
    return roster
def server_urls()->Dict:
    return {"deliberator":_DELIB_URL,"scout":_SCOUT_URL,"scout_port":_SCOUT_PORT,"delib_port":_DELIB_PORT}
def status()->Dict:
    _ptex=ptex_models()
    _gemma=gemma_native_models()
    _adam=adam_models()
    return {
        "active_model":active_model_name(),
        "active_gguf":str(get_active_gguf() or ""),
        "qwen35_9b_q8_available":qwen35_9b_q8_path() is not None,
        "qwen35_9b_q4_available":qwen35_9b_q4_path() is not None,
        "jackrong_9b_available":jackrong_9b_path() is not None,
        "luffy_9b_q8_available":luffy_9b_q8_path() is not None,
        "luffy_9b_q4_available":luffy_9b_q4_path() is not None,
        "best_2b":str(best_2b_path() or ""),
        "models_2b_count":len(all_2b_paths()),
        "local_ready":is_local_ready(),
        "ptex_models":_ptex,
        "gemma_models":_gemma,
        "adam_models":_adam,
        "roster":model_roster(),
        "servers":server_urls(),
        "scout_available":scout_available(),
    }