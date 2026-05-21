"""Voice subsystem — text↔audio for Adam.
Backends are plug-and-play; system uses the first available in priority order:
  TTS:  piper > xtts > pyttsx3 (Windows SAPI / espeak / NSSpeechSynthesizer)
  STT:  faster_whisper > vosk > none-installed-error
Install upgrades:
  pip install piper-tts                # higher-quality offline TTS
  pip install faster-whisper           # local STT, ROCm/CUDA-accelerated
  pip install TTS                      # XTTS v2 voice-cloning
"""
from .tts import speak, list_voices, available_backend as tts_backend
from .stt import transcribe, available_backend as stt_backend
from .wake import listen_for_wake, detect_in_buffer, available_wake_words
__all__=['speak','list_voices','transcribe','tts_backend','stt_backend','listen_for_wake','detect_in_buffer','available_wake_words']
