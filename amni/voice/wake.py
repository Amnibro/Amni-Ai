"""Wake-word detection via openwakeword. Pre-built models include 'alexa', 'hey_mycroft', 'hey_jarvis', 'hey_rhasspy', 'timer'. Custom 'hey adam' requires training a custom model (see openwakeword docs).
Usage:
  from amni.voice.wake import listen_for_wake
  while True:
      if listen_for_wake('hey_jarvis', timeout_s=30):
          # do voice chat here
          ...
"""
import time
from typing import Optional,List
_model_cache={}
def available_wake_words()->List[str]:
    try:
        from openwakeword.model import Model
        m=_get_or_create_model();return list(m.models.keys()) if m else []
    except Exception:return []
def _get_or_create_model(wake_word:str='hey_jarvis'):
    try:
        from openwakeword.model import Model
        if wake_word in _model_cache:return _model_cache[wake_word]
        try:m=Model(wakeword_models=[wake_word])
        except Exception:m=Model()
        _model_cache[wake_word]=m
        return m
    except ImportError:return None
def listen_for_wake(wake_word:str='hey_jarvis',timeout_s:float=10.0,threshold:float=0.5,samplerate:int=16000,chunk_ms:int=80)->bool:
    """Block until wake word detected or timeout. Returns True if detected."""
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:return False
    m=_get_or_create_model(wake_word)
    if m is None:return False
    chunk_samples=int(samplerate*chunk_ms/1000)
    deadline=time.time()+timeout_s
    try:
        with sd.InputStream(samplerate=samplerate,channels=1,dtype='int16',blocksize=chunk_samples) as stream:
            while time.time()<deadline:
                data,_=stream.read(chunk_samples)
                arr=data.flatten()
                preds=m.predict(arr)
                for kw,score in preds.items():
                    if score>=threshold:return True
        return False
    except Exception as e:print(f'[wake.listen] {e}',flush=True);return False
def detect_in_buffer(audio_int16,wake_word:str='hey_jarvis',threshold:float=0.5)->dict:
    """One-shot: run a numpy int16 audio array through the wake-word model. Returns {detected:bool, scores:dict}."""
    m=_get_or_create_model(wake_word)
    if m is None:return {'detected':False,'error':'openwakeword not installed or model load failed'}
    try:
        import numpy as np
        if not isinstance(audio_int16,np.ndarray):
            try:audio_int16=np.frombuffer(audio_int16,dtype=np.int16)
            except Exception:return {'detected':False,'error':'audio must be int16 array or bytes'}
        preds=m.predict(audio_int16.flatten())
        return {'detected':any(s>=threshold for s in preds.values()),'scores':{k:float(v) for k,v in preds.items()},'threshold':threshold}
    except Exception as e:return {'detected':False,'error':str(e)}
