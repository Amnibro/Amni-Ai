"""TTS — natural neural voices via Piper, with pyttsx3 (Windows SAPI / espeak) fallback.
Piper voices live in ~/.amni/voices/ as .onnx + .onnx.json pairs.
Download: python scripts/download_voices.py [--voice amy|ryan|lessac|alan|jenny|libritts]
Persona-specific prosody is applied via SynthesisConfig: length_scale (rate)
and noise_scale (variation). Defaults are slightly expressive (noise=0.85,
length=0.95) so speech feels less robotic than Piper's flat defaults.
"""
import io,os,tempfile,wave
from pathlib import Path
from typing import Optional,List,Dict,Any
_VOICE_CACHE={}
_VOICE_ALIASES={'amy':'en_US-amy-medium','ryan':'en_US-ryan-high','lessac':'en_US-lessac-medium','libritts':'en_US-libritts_r-medium','alan':'en_GB-alan-medium','jenny':'en_GB-jenny_dioco-medium'}
_PROSODY_PRESETS={
    'rikku':{'length_scale':0.88,'noise_scale':1.0,'noise_w_scale':0.95},
    'amy':{'length_scale':0.92,'noise_scale':0.95,'noise_w_scale':0.9},
    'yoda':{'length_scale':1.18,'noise_scale':0.9,'noise_w_scale':0.85},
    'alan':{'length_scale':1.05,'noise_scale':0.85,'noise_w_scale':0.85},
    'mentor':{'length_scale':1.0,'noise_scale':0.85,'noise_w_scale':0.85},
    'ryan':{'length_scale':1.0,'noise_scale':0.85,'noise_w_scale':0.85},
    'pirate':{'length_scale':1.08,'noise_scale':1.0,'noise_w_scale':0.95},
    'scientist':{'length_scale':1.02,'noise_scale':0.8,'noise_w_scale':0.85},
    'jobs':{'length_scale':1.0,'noise_scale':0.85,'noise_w_scale':0.85},
    'haiku':{'length_scale':1.25,'noise_scale':0.9,'noise_w_scale':0.85},
    'jenny':{'length_scale':0.98,'noise_scale':0.9,'noise_w_scale':0.85},
    'sherlock':{'length_scale':1.0,'noise_scale':0.9,'noise_w_scale':0.85},
    'jarvis':{'length_scale':1.0,'noise_scale':0.82,'noise_w_scale':0.85},
    'lessac':{'length_scale':1.0,'noise_scale':0.85,'noise_w_scale':0.85},
    'libritts':{'length_scale':1.0,'noise_scale':0.85,'noise_w_scale':0.85},
    '_default':{'length_scale':0.95,'noise_scale':0.85,'noise_w_scale':0.85},
}
def _get_prosody(voice_alias:Optional[str])->Dict[str,float]:
    if not voice_alias:return _PROSODY_PRESETS['_default']
    return _PROSODY_PRESETS.get(voice_alias.lower(),_PROSODY_PRESETS['_default'])
def _add_expressive_punctuation(text:str)->str:
    """Tiny preprocessing to coax better prosody — pause hints + run-on splitting."""
    import re as _re
    t=_re.sub(r'([.!?])\s*([A-Z])',r'\1 \2',text)
    t=_re.sub(r',\s*',', ',t)
    t=_re.sub(r'!{2,}','!',t)
    t=_re.sub(r'\.{3,}','…',t)
    return t.strip()
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
def _try_piper(text:str,voice:Optional[str]=None,persona:Optional[str]=None)->Optional[bytes]:
    try:
        from piper import PiperVoice
        from piper.config import SynthesisConfig
    except ImportError:return None
    vp=_resolve_voice_path(voice)
    if vp is None:return None
    try:
        if str(vp) not in _VOICE_CACHE:
            _VOICE_CACHE[str(vp)]=PiperVoice.load(str(vp))
        pv=_VOICE_CACHE[str(vp)]
        _key=(persona or voice or vp.stem).lower()
        _prosody=_PROSODY_PRESETS.get(_key,_PROSODY_PRESETS.get(voice and voice.lower() or '','')) or _PROSODY_PRESETS['_default']
        cfg=SynthesisConfig(length_scale=_prosody.get('length_scale'),noise_scale=_prosody.get('noise_scale'),noise_w_scale=_prosody.get('noise_w_scale'))
        prepped=_add_expressive_punctuation(_decase(text))
        buf=io.BytesIO()
        with wave.open(buf,'wb') as wf:
            pv.synthesize_wav(prepped,wf,syn_config=cfg)
        return buf.getvalue()
    except Exception as e:print(f'[tts.piper] {e}',flush=True);return None
def _decase(text:str)->str:
    import re
    a=[c for c in text if c.isalpha()]
    if not a or sum(c.isupper() for c in a)/len(a)<=0.6:return text
    t=text.lower()
    t=re.sub(r'(^|[.!?]\s+)([a-z])',lambda m:m.group(1)+m.group(2).upper(),t)
    return re.sub(r'\bi\b','I',t)
_KOKORO_PIPE={}
_KOKORO_VOICE={'amy':'af_bella','rikku':'af_bella','ryan':'am_michael','jarvis':'am_michael','neutral':'am_michael','mentor':'am_michael','alan':'bm_george','alfred':'bm_george','yoda':'am_onyx','sherlock':'bm_george','jenny':'bf_emma','lessac':'am_adam','libritts':'af_heart','scientist':'am_adam','jobs':'am_michael','pirate':'am_onyx','haiku':'af_nicole'}
def _kokoro_voice(voice:Optional[str],persona:Optional[str])->str:
    for key in ((persona or '').lower(),(voice or '').lower()):
        if key in _KOKORO_VOICE:return _KOKORO_VOICE[key]
    if voice and (voice.startswith(('af_','am_','bf_','bm_'))):return voice
    return 'af_heart'
def _try_kokoro(text:str,voice:Optional[str]=None,persona:Optional[str]=None)->Optional[bytes]:
    try:
        from kokoro import KPipeline
        import numpy as np
    except ImportError:return None
    try:
        kv=_kokoro_voice(voice,persona);lang='b' if kv.startswith(('bf_','bm_')) else 'a'
        if lang not in _KOKORO_PIPE:_KOKORO_PIPE[lang]=KPipeline(lang_code=lang,device='cpu')
        pros=_get_prosody(persona or voice);speed=max(0.8,min(1.2,2.0-float(pros.get('length_scale',1.0))))
        chunks=[a for _,_,a in _KOKORO_PIPE[lang](_add_expressive_punctuation(_decase(text)),voice=kv,speed=speed) if a is not None]
        if not chunks:return None
        audio=np.concatenate([np.asarray(c,dtype=np.float32) for c in chunks])
        pcm=(np.clip(audio,-1.0,1.0)*32767.0).astype('<i2')
        buf=io.BytesIO()
        with wave.open(buf,'wb') as wf:
            wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(24000);wf.writeframes(pcm.tobytes())
        return buf.getvalue()
    except Exception as e:print(f'[tts.kokoro] {e}',flush=True);return None
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
        import importlib.util
        if importlib.util.find_spec('kokoro') is not None:return 'kokoro'
    except Exception:pass
    try:
        from piper import PiperVoice
        if list(_voices_dir().glob('*.onnx')):return 'piper'
    except ImportError:pass
    try:import pyttsx3;return 'pyttsx3'
    except ImportError:pass
    return 'none'
def list_voices()->List[Dict[str,Any]]:
    out=[]
    try:
        import importlib.util
        if importlib.util.find_spec('kokoro') is not None:
            for vid,label in (('af_bella','Female (US) — bubbly'),('af_heart','Female (US) — warm'),('af_nicole','Female (US) — soft'),('am_michael','Male (US) — smooth'),('am_adam','Male (US)'),('am_onyx','Male (US) — deep'),('bf_emma','Female (UK)'),('bm_george','Male (UK) — butler')):
                out.append({'id':vid,'alias':vid,'name':vid,'backend':'kokoro','label':label,'sample_rate':24000})
    except Exception:pass
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
def speak(text:str,backend:Optional[str]=None,voice:Optional[str]=None,persona:Optional[str]=None)->Optional[bytes]:
    if not text or not text.strip():return None
    chosen=backend or available_backend()
    if chosen=='kokoro':
        out=_try_kokoro(text,voice=voice,persona=persona)
        if out:return out
        chosen='piper'
    if chosen=='piper':
        out=_try_piper(text,voice=voice,persona=persona)
        if out:return out
        chosen='pyttsx3'
    if chosen=='pyttsx3':return _try_pyttsx3(text)
    return None
