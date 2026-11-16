"""Voice HTTP endpoints — local Piper TTS + faster-whisper STT via the existing tts/stt skills.
POST /voice/speak       {text, voice?, backend?}        → {audio_base64, content_type, backend, voice}
POST /voice/transcribe  {audio_base64, model_size?}     → {text, language?, backend}
GET  /voice/status                                       → {tts:{available, backend, voices?}, stt:{available, backend}}
The browser falls back to its built-in Web Speech if any endpoint errors. Each call wraps in TaskRegistry so it shows in /jarvis tray for long Whisper transcriptions."""
import base64,time
_TTS_NEED='no local TTS engine installed - pip install kokoro (or piper-tts + a voice in the voices dir, or pyttsx3). The browser uses its built-in speech meanwhile.'
_STT_NEED='no local speech-to-text engine installed - pip install faster-whisper (first use downloads a ~150 MB Whisper model) or vosk + a model under ~/.amni/vosk_models. The browser uses its built-in recognition meanwhile.'
def mount(app,agent):
    from fastapi import Request,HTTPException
    skills=getattr(agent,'skills',None)
    adam=getattr(agent,'adam',None)
    task_reg=getattr(agent,'task_registry',None)
    def _has(name):return skills is not None and skills.has(name)
    def _probe_tts():
        if not _has('tts'):return {'available':False,'reason':'tts skill not registered'}
        try:
            r=skills.call('tts',{'list_voices':True},ctx={'adam':adam})
            if r.ok and (r.output or {}).get('backend','none')=='none':return {'available':False,'backend':'none','reason':_TTS_NEED}
            if r.ok:return {'available':True,'backend':r.output.get('backend','?'),'voices':(r.output.get('voices') or [])[:20]}
            return {'available':False,'reason':r.error or 'list_voices failed'}
        except Exception as e:return {'available':False,'reason':f'{type(e).__name__}: {e}'}
    def _probe_stt():
        if not _has('stt'):return {'available':False,'reason':'stt skill not registered'}
        try:
            from amni.voice.stt import available_backend as _sb;b=_sb()
        except Exception as e:return {'available':False,'reason':f'voice module import failed: {e}'}
        if b=='none':return {'available':False,'backend':'none','reason':_STT_NEED}
        return {'available':True,'backend':b}
    @app.get('/voice/status')
    def status():return {'tts':_probe_tts(),'stt':_probe_stt()}
    @app.post('/voice/speak')
    async def speak(req:Request):
        if not _has('tts'):raise HTTPException(503,'tts skill not available')
        body=await req.json()
        text=(body.get('text') or '').strip()
        if not text:raise HTTPException(400,'need text')
        if len(text)>100000:raise HTTPException(413,'text too large')
        voice=body.get('voice')
        backend=body.get('backend')
        tid=task_reg.register('voice_speak',label=f'TTS ({len(text)} chars)') if task_reg else None
        try:
            args={'text':text[:2000]}
            if voice:args['voice']=voice
            if backend:args['backend']=backend
            r=skills.call('tts',args,ctx={'adam':adam})
            if not r.ok:
                if task_reg and tid:task_reg.fail(tid,r.error or 'tts failed')
                raise HTTPException(500,f'tts failed: {r.error}')
            out=r.output or {}
            if out.get('error'):
                if task_reg and tid:task_reg.fail(tid,out['error'])
                raise HTTPException(503,f"tts unavailable: {out['error']}"+(f' - {_TTS_NEED}' if out.get('backend','none')=='none' else ''))
            audio_b64=out.get('audio_base64') or out.get('audio_b64')
            if not audio_b64:
                if task_reg and tid:task_reg.fail(tid,'tts returned no audio')
                raise HTTPException(500,'tts returned no audio (out_path mode?)')
            res={'audio_base64':audio_b64,'content_type':out.get('content_type','audio/wav'),'backend':out.get('backend','?'),'voice':out.get('voice'),'duration_s':out.get('duration_s'),'bytes':out.get('bytes')}
            if task_reg and tid:task_reg.complete(tid,outcome={'backend':res['backend'],'bytes':res.get('bytes')})
            return res
        except HTTPException:raise
        except Exception as e:
            if task_reg and tid:task_reg.fail(tid,str(e)[:200])
            raise HTTPException(500,f'tts exception: {e}')
    @app.post('/voice/transcribe')
    async def transcribe(req:Request):
        if not _has('stt'):raise HTTPException(503,'stt skill not available')
        body=await req.json()
        b64=body.get('audio_base64') or body.get('base64')
        if not b64:raise HTTPException(400,'need audio_base64')
        from amni.serve.code_safety import b64_within_limit
        if not b64_within_limit(b64):raise HTTPException(413,'audio too large')
        try:base64.b64decode(b64.split(',',1)[-1] if ',' in b64 else b64,validate=True)
        except Exception as e:raise HTTPException(400,f'base64 decode: {e}')
        b64_clean=b64.split(',',1)[-1] if ',' in b64 else b64
        tid=task_reg.register('voice_stt',label=f'STT ({len(b64_clean)//1000}kB audio)') if task_reg else None
        try:
            args={'audio_base64':b64_clean}
            if body.get('model_size'):args['model_size']=body.get('model_size')
            if body.get('backend'):args['backend']=body.get('backend')
            r=skills.call('stt',args,ctx={'adam':adam})
            if not r.ok:
                if task_reg and tid:task_reg.fail(tid,r.error or 'stt failed')
                raise HTTPException(500,f'stt failed: {r.error}')
            out=r.output or {}
            if out.get('error') or out.get('backend') in (None,'none'):
                msg=out.get('error') or 'no STT backend'
                if task_reg and tid:task_reg.fail(tid,msg)
                raise HTTPException(503,f'stt unavailable: {msg}'+(f' - {_STT_NEED}' if out.get('backend') in (None,'none') else ''))
            res={'text':out.get('text',''),'language':out.get('language'),'backend':out.get('backend','?'),'duration_s':out.get('duration_s'),'wall_s':out.get('wall_s')}
            if task_reg and tid:task_reg.complete(tid,outcome={'backend':res['backend'],'chars':len(res['text'])})
            return res
        except HTTPException:raise
        except Exception as e:
            if task_reg and tid:task_reg.fail(tid,str(e)[:200])
            raise HTTPException(500,f'stt exception: {e}')
