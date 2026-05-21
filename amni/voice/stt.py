"""STT — try faster-whisper first, fall back to vosk or return install hint."""
import io,os,tempfile,wave
from pathlib import Path
from typing import Optional,Dict,Any
_whisper_model=None
def _detect_audio_suffix(audio_bytes:bytes)->str:
    h=audio_bytes[:16] if audio_bytes else b''
    if h.startswith(b'RIFF') and b'WAVE' in h[:16]:return '.wav'
    if h.startswith(b'\x1A\x45\xDF\xA3'):return '.webm'
    if h.startswith(b'OggS'):return '.ogg'
    if h.startswith(b'ID3') or h[:2]==b'\xff\xfb' or h[:2]==b'\xff\xf3':return '.mp3'
    if h.startswith(b'fLaC'):return '.flac'
    if h[4:8]==b'ftyp':return '.m4a'
    return '.bin'
def _try_faster_whisper(audio_bytes:bytes,model_size:str='base')->Optional[Dict[str,Any]]:
    try:from faster_whisper import WhisperModel
    except ImportError:return None
    global _whisper_model
    try:
        if _whisper_model is None:
            cache=Path.home()/'.amni'/'whisper_models'
            cache.mkdir(parents=True,exist_ok=True)
            _whisper_model=WhisperModel(model_size,device='auto',compute_type='auto',download_root=str(cache))
        suffix=_detect_audio_suffix(audio_bytes)
        tmp=tempfile.NamedTemporaryFile(suffix=suffix,delete=False);tmp.write(audio_bytes);tmp.close()
        segs,info=_whisper_model.transcribe(tmp.name,beam_size=1,language=None)
        text=' '.join(s.text.strip() for s in segs)
        try:os.unlink(tmp.name)
        except Exception:pass
        return {'text':text.strip(),'language':info.language,'language_prob':round(info.language_probability,2),'backend':'faster-whisper','model':model_size,'detected_format':suffix.lstrip('.')}
    except Exception as e:print(f'[stt.faster_whisper] {e}',flush=True);return None
def _try_vosk(audio_bytes:bytes)->Optional[Dict[str,Any]]:
    try:
        import vosk,json as _j
    except ImportError:return None
    try:
        cache=Path.home()/'.amni'/'vosk_models'
        models=list(cache.glob('vosk-model-*')) if cache.exists() else []
        if not models:return None
        model=vosk.Model(str(models[0]))
        tmp=tempfile.NamedTemporaryFile(suffix='.wav',delete=False);tmp.write(audio_bytes);tmp.close()
        with wave.open(tmp.name,'rb') as wf:
            rec=vosk.KaldiRecognizer(model,wf.getframerate())
            text_chunks=[]
            while True:
                data=wf.readframes(4000)
                if len(data)==0:break
                if rec.AcceptWaveform(data):
                    res=_j.loads(rec.Result());text_chunks.append(res.get('text',''))
            final=_j.loads(rec.FinalResult());text_chunks.append(final.get('text',''))
        try:os.unlink(tmp.name)
        except Exception:pass
        return {'text':' '.join(t for t in text_chunks if t).strip(),'backend':'vosk'}
    except Exception as e:print(f'[stt.vosk] {e}',flush=True);return None
def available_backend()->str:
    try:import faster_whisper;return 'faster_whisper'
    except ImportError:pass
    try:
        import vosk
        cache=Path.home()/'.amni'/'vosk_models'
        if cache.exists() and list(cache.glob('vosk-model-*')):return 'vosk'
    except ImportError:pass
    return 'none'
def transcribe(audio_bytes:bytes,backend:Optional[str]=None,model_size:str='base')->Dict[str,Any]:
    if not audio_bytes:return {'error':'empty audio'}
    chosen=backend or available_backend()
    if chosen=='faster_whisper':
        out=_try_faster_whisper(audio_bytes,model_size=model_size)
        if out:return out
    if chosen in ('vosk','faster_whisper'):
        out=_try_vosk(audio_bytes)
        if out:return out
    return {'error':'no STT backend installed','install_hint':'pip install faster-whisper  (~120MB base model on first use, GPU-accelerated)'}
