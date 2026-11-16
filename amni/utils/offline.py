"""Offline mode: one switch for "no network at runtime".
Turn on with `AMNI_OFFLINE=1` or `--offline` (amni serve / scripts/amni_serve.py). It also sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE /
HF_DATASETS_OFFLINE so transformers + huggingface_hub never try the Hub at load time.
Without the switch, is_offline() auto-detects: one bounded TCP probe (default 1.5 s, run in a thread so a stuck DNS lookup cannot hang startup),
cached for AMNI_NET_RECHECK_S seconds (default 300). AMNI_OFFLINE=0 forces online and skips the probe.
Network features (web crawler, web-search / API skills, federation pulls, update checks) call net_allowed() and fail soft when it is False."""
import os,socket,threading,time
from urllib.parse import urlparse
_TRUE=('1','true','yes','on')
_cache={'t':0.0,'online':None}
_HF_VARS=('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE')
def forced()->bool|None:
    v=os.environ.get('AMNI_OFFLINE','').strip().lower()
    if v in _TRUE:return True
    if v in ('0','false','no','off'):return False
    return None
def apply_offline_env(reason:str='')->None:
    """Flip the process into offline mode (idempotent). Safe to call before transformers/huggingface_hub are imported; also patches the already-imported hub constant."""
    os.environ['AMNI_OFFLINE']='1'
    for k in _HF_VARS:os.environ[k]='1'
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1');os.environ.setdefault('DO_NOT_TRACK','1')
    try:
        import sys
        c=sys.modules.get('huggingface_hub.constants')
        if c is not None:c.HF_HUB_OFFLINE=True
    except Exception:pass
    _cache.update(t=time.time(),online=False)
    if reason:print(f'[offline] offline mode ON ({reason}): web crawler, web-search skills, federation pulls and update checks are skipped; HF hub is not contacted.',flush=True)
def _probe_target():
    px=os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    if px:
        u=urlparse(px if '://' in px else 'http://'+px);return u.hostname or '127.0.0.1',u.port or 8080
    return os.environ.get('AMNI_NET_PROBE_HOST','huggingface.co'),int(os.environ.get('AMNI_NET_PROBE_PORT','443'))
def probe(timeout:float=None)->bool:
    """True if a TCP connection to the probe host (or the configured HTTPS proxy) opens within `timeout` seconds."""
    timeout=float(timeout if timeout is not None else os.environ.get('AMNI_NET_PROBE_TIMEOUT','1.5'))
    host,port=_probe_target();res={'ok':False}
    def _run():
        try:
            with socket.create_connection((host,port),timeout=timeout):res['ok']=True
        except Exception:res['ok']=False
    th=threading.Thread(target=_run,daemon=True);th.start();th.join(timeout+0.5)
    return bool(res['ok'])
def is_offline(refresh:bool=False)->bool:
    f=forced()
    if f is not None:return f
    ttl=float(os.environ.get('AMNI_NET_RECHECK_S','300'))
    if refresh or _cache['online'] is None or time.time()-_cache['t']>ttl:
        _cache.update(online=probe(),t=time.time())
    return not _cache['online']
def net_allowed(feature:str='')->bool:
    return not is_offline()
def offline_reply(feature:str)->str:
    return f'{feature} needs the internet, and Adam is running offline. Everything else (chat, files, memory, local skills) still works.'
def status()->dict:
    f=forced();off=is_offline()
    return {'offline':off,'forced':f,'source':'AMNI_OFFLINE' if f is not None else 'probe','hf_hub_offline':os.environ.get('HF_HUB_OFFLINE')=='1','probe_target':'%s:%s'%_probe_target()}
def init_from_args(flag:bool=False)->bool:
    """Call once at process start. Applies offline env when --offline / AMNI_OFFLINE=1, or when auto-detect finds no network. Returns the offline state."""
    if flag:apply_offline_env('--offline');return True
    f=forced()
    if f is True:apply_offline_env('AMNI_OFFLINE=1');return True
    if f is False:return False
    if is_offline(refresh=True):apply_offline_env('no network detected');return True
    return False
