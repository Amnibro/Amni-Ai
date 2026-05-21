"""Amni-Ai bootstrap: config dir, model auto-download, sane defaults.
Config lives at ~/.amni/config.json. Model bake auto-detected from common paths or downloaded from HF."""
import os,json,sys,platform
from pathlib import Path
from typing import Dict,Any,Optional
HOME=Path.home()
INSTALL_POINTER=HOME/'.amni-ai'/'last_install_home.txt'
def _resolve_amni_home()->str:
    e=os.environ.get('AMNI_HOME')
    if e:return e
    if INSTALL_POINTER.exists():
        try:
            p=INSTALL_POINTER.read_text(encoding='utf-8').strip()
            if p and Path(p).exists():return p
        except Exception:pass
    return str(HOME/'.amni-ai')
CONFIG_DIR=Path(_resolve_amni_home())
CONFIG_FILE=CONFIG_DIR/'config.json'
DEFAULT_HF_REPO='amnibro/gemma-4-E2B-it-gf17'
DEFAULT_BASE_REPO='google/gemma-2-2b-it'
DEFAULT_PORT=7700
DEFAULT_HOST='127.0.0.1'
_DEFAULTS={'bake':None,'model':None,'lessons':None,'lut_root':None,'conv_root':None,'persona_bank':None,'audit_log':None,'workdir':None,'default_persona':'rikku','port':DEFAULT_PORT,'host':DEFAULT_HOST,'unrestricted_files':False,'cors':True,'open_browser':True,'first_run_done':False,'hf_bake_repo':DEFAULT_HF_REPO,'hf_base_repo':DEFAULT_BASE_REPO,'budget_mb':8000}
def _extra_candidates(var:str):
    raw=os.environ.get(var) or ''
    return [Path(p) for p in raw.replace(';',os.pathsep).split(os.pathsep) if p.strip()]
def _candidate_bake_paths():
    return _extra_candidates('AMNI_BAKE_PATHS')+[CONFIG_DIR/'bakes'/'gemma4_e2b_it_gf17',Path('./bakes/gemma4_e2b_it_gf17'),Path.home()/'amni-bakes'/'gemma4_e2b_it_gf17',Path.home()/'.amni-ai'/'bakes'/'gemma4_e2b_it_gf17']
def _candidate_model_paths():
    return _extra_candidates('AMNI_MODEL_PATHS')+[CONFIG_DIR/'models'/'gemma-4-E2B-it',Path('./models/gemma-4-E2B-it'),Path.home()/'amni-models'/'gemma-4-E2B-it',Path.home()/'.amni-ai'/'models'/'gemma-4-E2B-it']
def detect_bake()->Optional[Path]:
    for p in _candidate_bake_paths():
        if p.exists() and (p/'manifest.json').exists() and (p/'config.json').exists() and (p/'tokenizer.json').exists():return p
    return None
def detect_model()->Optional[Path]:
    for p in _candidate_model_paths():
        if p.exists() and (p/'config.json').exists() and (p/'tokenizer.json').exists():return p
    return None
def bake_has_runtime_metadata(bake_dir)->bool:
    p=Path(bake_dir) if bake_dir else None
    return bool(p and p.exists() and (p/'config.json').exists() and (p/'tokenizer.json').exists())
def load_config()->Dict[str,Any]:
    cfg=dict(_DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            saved=json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
            saved={k:v for k,v in saved.items() if k in _DEFAULTS}
            cfg.update(saved)
        except Exception as e:print(f'[bootstrap] config parse failed: {e}',flush=True)
    if cfg.get('bake') and not (Path(cfg['bake']).exists() and (Path(cfg['bake'])/'manifest.json').exists()):cfg['bake']=None
    if cfg.get('model') and not (Path(cfg['model']).exists() and (Path(cfg['model'])/'config.json').exists()):cfg['model']=None
    if not cfg.get('bake'):
        b=detect_bake();cfg['bake']=str(b) if b else None
    if not cfg.get('model'):
        m=detect_model();cfg['model']=str(m) if m else None
    cdir=CONFIG_DIR;cwd=Path.cwd()
    def _pick(local_rel:str,global_path:str,prefer_local_if_exists:bool=True)->str:
        local=cwd/local_rel
        if prefer_local_if_exists and local.exists():return str(local)
        return global_path
    if not cfg.get('lessons'):cfg['lessons']=_pick('experiences/adam_lessons.npz',str(cdir/'experiences'/'adam_lessons.npz'))
    if not cfg.get('lut_root'):cfg['lut_root']=_pick('experiences/adam_lut',str(cdir/'experiences'/'adam_lut'))
    if not cfg.get('conv_root'):cfg['conv_root']=_pick('experiences/conversations',str(cdir/'experiences'/'conversations'))
    if not cfg.get('persona_bank'):cfg['persona_bank']=_pick('experiences/personas.json',str(cdir/'experiences'/'personas.json'))
    if not cfg.get('audit_log'):cfg['audit_log']=_pick('logs/agent_skill_calls.jsonl',str(cdir/'logs'/'agent_skill_calls.jsonl'))
    if not cfg.get('workdir'):cfg['workdir']=str(cwd)
    return cfg
def save_config(cfg:Dict[str,Any]):
    CONFIG_DIR.mkdir(parents=True,exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    default_home=HOME/'.amni-ai'
    if CONFIG_DIR.resolve()!=default_home.resolve():
        try:
            default_home.mkdir(parents=True,exist_ok=True)
            INSTALL_POINTER.write_text(str(CONFIG_DIR),encoding='utf-8')
        except Exception as _e:print(f'[bootstrap] could not write install pointer {INSTALL_POINTER}: {_e}',flush=True)
def ensure_dirs(cfg:Dict[str,Any]):
    for k in ('lessons','lut_root','conv_root','persona_bank','audit_log'):
        v=cfg.get(k)
        if not v:continue
        Path(v).parent.mkdir(parents=True,exist_ok=True)
def download_bake(cfg:Dict[str,Any],force:bool=False)->Optional[Path]:
    target=Path(cfg.get('bake') or (CONFIG_DIR/'bakes'/'gemma4_e2b_it_gf17'))
    if target.exists() and (target/'manifest.json').exists() and not force:
        print(f'[bootstrap] bake already at {target}',flush=True);return target
    try:from huggingface_hub import snapshot_download
    except ImportError:print('[bootstrap] huggingface_hub not installed. pip install huggingface_hub',flush=True);return None
    repo=cfg.get('hf_bake_repo',DEFAULT_HF_REPO)
    print(f'[bootstrap] downloading HF bake "{repo}" -> {target} (~20 GB, self-contained GF(17) artifact, one-time)',flush=True)
    target.mkdir(parents=True,exist_ok=True)
    try:
        snapshot_download(repo_id=repo,local_dir=str(target))
        print(f'[bootstrap] bake ready at {target}',flush=True);return target
    except Exception as e:
        print(f'[bootstrap] HF bake download failed ({type(e).__name__}: {str(e)[:160]})',flush=True)
        print(f'[bootstrap] Adam ships as a self-contained GF(17) bake — there is no fallback to upstream Gemma 4 weights for public installs.',flush=True)
        print(f'[bootstrap] Retry: check network, then re-run `amni init`. If the bake repo "{repo}" is unreachable, file an issue at https://github.com/Amnibro/Amni-Ai/issues',flush=True)
        return None
def generate_bake_local(cfg:Dict[str,Any],target:Path,force:bool=False)->Optional[Path]:
    """DEV-ONLY: re-encode upstream Gemma 4 E2B IT into a GF(17) bake locally.
    Public install never calls this — the prebuilt HF bake is the only delivery channel.
    Requires gated HF access to the upstream base model + the encoder chain (amni.compute.reffelt4).
    First-run cost: ~5GB base download + 1-5 min GF(17) re-encode on GPU."""
    import subprocess,sys
    base_repo=cfg.get('hf_base_repo',DEFAULT_BASE_REPO)
    bake_script=Path(__file__).resolve().parents[1]/'scripts'/'adam1_bake.py'
    if not bake_script.exists():
        print(f'[bootstrap] cannot generate bake locally: missing {bake_script}',flush=True);return None
    print(f'[bootstrap] generating bake locally from {base_repo} -> {target}',flush=True)
    print(f'[bootstrap] first-run cost: ~5GB base download + 1-5 min GF(17) re-encode on GPU',flush=True)
    target.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(bake_script),'--hf-id',base_repo,'--out',str(target)]
    try:r=subprocess.run(cmd,check=False)
    except Exception as e:print(f'[bootstrap] local bake generation failed: {e}',flush=True);return None
    if r.returncode!=0:print(f'[bootstrap] local bake generation exited rc={r.returncode}',flush=True);return None
    if not (target/'manifest.json').exists():print(f'[bootstrap] local bake completed but manifest.json missing at {target}',flush=True);return None
    print(f'[bootstrap] local bake ready at {target}',flush=True);return target
def download_base_model(cfg:Dict[str,Any],force:bool=False)->Optional[Path]:
    """DEV-ONLY: pull upstream Gemma 4 E2B IT from HF for local re-baking.
    Public install never calls this — Adam ships as a self-contained GF(17) bake.
    Only invoke directly when rebuilding the bake from source on a HF-authenticated dev machine."""
    bake=cfg.get('bake')
    if bake and bake_has_runtime_metadata(bake) and not force:
        print(f'[bootstrap] base-model download skipped — prebuilt bake at {bake} already ships tokenizer.json + config.json (sufficient for streaming runtime)',flush=True)
        return Path(bake)
    target=Path(cfg.get('model') or (CONFIG_DIR/'models'/'gemma-4-E2B-it'))
    if target.exists() and (target/'config.json').exists() and not force:
        print(f'[bootstrap] base model already at {target}',flush=True);return target
    try:from huggingface_hub import snapshot_download
    except ImportError:print('[bootstrap] huggingface_hub not installed.',flush=True);return None
    repo=cfg.get('hf_base_repo',DEFAULT_BASE_REPO)
    print(f'[bootstrap] downloading base model from HF "{repo}" -> {target}',flush=True)
    target.mkdir(parents=True,exist_ok=True)
    try:snapshot_download(repo_id=repo,local_dir=str(target),allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*'])
    except Exception as e:
        msg=str(e)
        if '401' in msg or 'gated' in msg.lower() or 'restricted' in msg.lower():
            print(f'[bootstrap] base model "{repo}" is a GATED HF repo and your machine is not authenticated.',flush=True)
            print(f'[bootstrap] You usually do NOT need it — the prebuilt bake "{cfg.get("hf_bake_repo",DEFAULT_HF_REPO)}" includes tokenizer + config and is enough to chat.',flush=True)
            print(f'[bootstrap] If you really want the base weights:  1) accept the license at https://huggingface.co/{repo}   2) `pip install -U huggingface_hub && huggingface-cli login`   3) re-run `amni init`',flush=True)
            print(f'[bootstrap] Or skip entirely: `python install.py --skip-model` (the bake already supports inference).',flush=True)
        else:
            print(f'[bootstrap] base model download failed: {e}',flush=True)
        return None
    print(f'[bootstrap] base model ready at {target}',flush=True);return target
def is_first_run()->bool:
    cfg=load_config()
    return not cfg.get('first_run_done',False)
def mark_first_run_done():
    cfg=load_config();cfg['first_run_done']=True;save_config(cfg)
