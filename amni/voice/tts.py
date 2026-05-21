"""TTS — natural neural voices via Piper, with pyttsx3 (Windows SAPI / espeak) fallback.
Piper voices live in ~/.amni/voices/ as .onnx + .onnx.json pairs.
Download: python scripts/download_voices.py [--voice amy|ryan|lessac|alan|jenny|libritts]
"""
import io,os,tempfile,wave
from pathlib import Path
from typing import Optional,List,Dict,Any
_VOICE_CACHE={}
_VOICE_ALIASES={'amy':'en_US-amy-medium','ryan':'en_US-ryan-high','lessac':'en_US-lessac-medium','libritts':'en_US-libritts_r-medium','alan':'en_GB-alan-medium','jenny':'en_GB-jenny_dioco-medium'}
def _voices_dir()->Path:
    p=Path.home()/'.amni'/'voices'
    p.mkdir(parents=True,exist_ok=True);return p
def _resolve_voice_path(voice:Optional[str])->Optional[Path]:
    root=_voices_dir()
    if voice:
        cand=Path(voice)
        if cand.exists() and cand.suffix=='.onnx':return cand
        full_name=_VOICE_ALIASES.get(voice.lower(),voice)
        p=root/f'{full_name}.onnx'
        if p.exists():return p
        for f in root.glob('*.onnx'):
            if voice.lower() in f.stem.lower():return f
    onnx_files=sorted(root.glob('*.onnx'))
    return onnx_files[0] if onnx_files else None
def _try_piper(text:str,voice:Optional[str]=None)->Optional[bytes]:
    try:
        from piper import PiperVoice
    except ImportError:return None
    vp=_resolve_voice_path(voice)
    if vp is None:return None
    try:
        if str(vp) not in _VOICE_CACHE:
            _VOICE_CACHE[str(vp)]=PiperVoice.load(str(vp))
        pv=_VOICE_CACHE[str(vp)]
        buf=io.BytesIO()
        with wave.open(buf,'wb') as wf:
            pv.synthesize_wav(text,wf)
        return buf.getvalue()
    except Exception as e:print(f'[tts.piper] {e}',flush=True);return None
def _try_pyttsx3(text:str)->Optional[bytes]:
    try:import pyttsx3
    except ImportError:return None
    try:
        eng=pyttsx3.init()
        tmp=tempfile.NamedTemporaryFile(suffix='.wav',delete=False);tmp.close()
        eng.save_to_file(text,tmp.name)
        eng.runAndWait()
        data=Path(tmp.name).read_bytes()
        try:os.unlink(tmp.name)
        except Exception:pass
        return data
    except Exception as e:print(f'[tts.pyttsx3] {e}',flush=True);return None
def available_backend()->str:
    try:
        from piper import PiperVoice
        if list(_voices_dir().glob('*.onnx')):return 'piper'
    except ImportError:pass
    try:import pyttsx3;return 'pyttsx3'
    except ImportError:pass
    return 'none'
def list_voices()->List[Dict[str,Any]]:
    out=[]
    root=_voices_dir()
    try:
        from piper import PiperVoice
        for f in sorted(root.glob('*.onnx')):
            alias=next((a for a,n in _VOICE_ALIASES.items() if n==f.stem),None)
            label='Female (US)' if 'amy' in f.stem.lower() else 'Male (US)' if any(k in f.stem.lower() for k in ('ryan','lessac')) else 'Male (UK)' if 'alan' in f.stem.lower() else 'Female (UK)' if 'jenny' in f.stem.lower() else 'Multi-speaker' if 'libritts' in f.stem.lower() else 'Neural'
            out.append({'id':str(f),'alias':alias,'name':f.stem,'backend':'piper','label':label,'sample_rate':22050})
    except ImportError:pass
    try:
        import pyttsx3
        eng=pyttsx3.init()
        for v in eng.getProperty('voices'):
            out.append({'id':v.id,'name':getattr(v,'name','?'),'backend':'pyttsx3','label':'OS robotic (fallback)'})
    except Exception:pass
    return out
def speak(text:str,backend:Optional[str]=None,voice:Optional[str]=None)->Optional[bytes]:
    if not text or not text.strip():return None
    chosen=backend or available_backend()
    if chosen=='piper':
        out=_try_piper(text,voice=voice)
        if out:return out
        chosen='pyttsx3'
    if chosen=='pyttsx3':return _try_pyttsx3(text)
    return None
