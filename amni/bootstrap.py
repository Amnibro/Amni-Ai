"""Amni-Ai bootstrap: config dir, model auto-download, sane defaults.
Config lives at ~/.amni/config.json. Model bake auto-detected from common paths or downloaded from HF."""
import os,json,sys,platform
from pathlib import Path
from typing import Dict,Any,Optional
HOME=Path.home()
CONFIG_DIR=Path(os.environ.get('AMNI_HOME',str(HOME/'.amni-ai')))
CONFIG_FILE=CONFIG_DIR/'config.json'
DEFAULT_HF_REPO='Amnibro/gemma-4-E2B-it-gf17'
DEFAULT_BASE_REPO='google/gemma-2-2b-it'
_DEFAULTS={'bake':None,'model':None,'lessons':None,'lut_root':None,'conv_root':None,'persona_bank':None,'audit_log':None,'workdir':None,'default_persona':'rikku','port':8002,'host':'127.0.0.1','unrestricted_files':False,'cors':True,'open_browser':True,'first_run_done':False,'hf_bake_repo':DEFAULT_HF_REPO,'hf_base_repo':DEFAULT_BASE_REPO,'budget_mb':8000}
def _candidate_bake_paths():
    return [Path('E:/Amni-Ai-Bakes/gemma4_e2b_it_gf17'),CONFIG_DIR/'bakes'/'gemma4_e2b_it_gf17',Path('./bakes/gemma4_e2b_it_gf17'),Path.home()/'amni-bakes'/'gemma4_e2b_it_gf17']
def _candidate_model_paths():
    return [Path('E:/Amni-Ai-Models/gemma-4-E2B-it'),CONFIG_DIR/'models'/'gemma-4-E2B-it',Path('./models/gemma-4-E2B-it'),Path.home()/'amni-models'/'gemma-4-E2B-it']
def detect_bake()->Optional[Path]:
    for p in _candidate_bake_paths():
        if p.exists() and (p/'manifest.json').exists():return p
    return None
def detect_model()->Optional[Path]:
    for p in _candidate_model_paths():
        if p.exists() and (p/'config.json').exists():return p
    return None
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
    print(f'[bootstrap] downloading bake from HF "{repo}" -> {target} (this may take a few minutes; ~5 GB)',flush=True)
    target.mkdir(parents=True,exist_ok=True)
    try:snapshot_download(repo_id=repo,local_dir=str(target),local_dir_use_symlinks=False)
    except Exception as e:
        print(f'[bootstrap] bake download failed: {e}',flush=True)
        print(f'[bootstrap] you can manually place the bake at {target} or set AMNI_BAKE env var',flush=True)
        return None
    print(f'[bootstrap] bake ready at {target}',flush=True);return target
def download_base_model(cfg:Dict[str,Any],force:bool=False)->Optional[Path]:
    target=Path(cfg.get('model') or (CONFIG_DIR/'models'/'gemma-4-E2B-it'))
    if target.exists() and (target/'config.json').exists() and not force:
        print(f'[bootstrap] base model already at {target}',flush=True);return target
    try:from huggingface_hub import snapshot_download
    except ImportError:print('[bootstrap] huggingface_hub not installed.',flush=True);return None
    repo=cfg.get('hf_base_repo',DEFAULT_BASE_REPO)
    print(f'[bootstrap] downloading base model from HF "{repo}" -> {target}',flush=True)
    target.mkdir(parents=True,exist_ok=True)
    try:snapshot_download(repo_id=repo,local_dir=str(target),local_dir_use_symlinks=False,allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*'])
    except Exception as e:print(f'[bootstrap] base model download failed: {e}',flush=True);return None
    print(f'[bootstrap] base model ready at {target}',flush=True);return target
def is_first_run()->bool:
    cfg=load_config()
    return not cfg.get('first_run_done',False)
def mark_first_run_done():
    cfg=load_config();cfg['first_run_done']=True;save_config(cfg)
