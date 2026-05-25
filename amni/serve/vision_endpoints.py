"""Vision HTTP endpoints — describe / ask about images.
POST /vision/describe   {image_base64}                  → {caption, dims, wall_s}
POST /vision/ask        {image_base64, question}        → {answer, question, wall_s}
POST /vision/upload     multipart (image=<file>)        → caption
GET  /vision/status                                     → {available, init_error?, models}
Wraps inference in a TaskRegistry entry so it shows up in the /jarvis task tray with a progress bar + cancel."""
import base64,time
def mount(app,agent):
    from fastapi import Request,HTTPException
    from fastapi.responses import JSONResponse
    vision=getattr(agent,'vision',None)
    task_reg=getattr(agent,'task_registry',None)
    @app.get('/vision/status')
    def status():
        if vision is None:return {'available':False,'error':'VisionService not initialized on agent'}
        return {'available':vision.is_available(),'init_error':vision.init_error,'caption_model':vision.caption_model_name,'vqa_model':vision.vqa_model_name,'device':vision.device}
    def _run_describe(image_bytes,question=''):
        tid=task_reg.register('vision','describing image' if not question else f'vqa: {question[:40]}') if task_reg else None
        try:
            if question:r=vision.caption_with_question(image_bytes,question)
            else:r=vision.describe(image_bytes)
            if task_reg and tid:
                if 'error' in r:task_reg.fail(tid,r['error'])
                else:task_reg.complete(tid,outcome={k:r[k] for k in r if k in ('caption','answer','width','height')})
            return r
        except Exception as e:
            if task_reg and tid:task_reg.fail(tid,str(e)[:200])
            raise
    @app.post('/vision/describe')
    async def describe(req:Request):
        if vision is None:raise HTTPException(503,'vision not initialized')
        body=await req.json()
        b64=body.get('image_base64') or body.get('base64')
        if not b64:raise HTTPException(400,'need image_base64')
        try:image_bytes=base64.b64decode(b64.split(',',1)[-1] if ',' in b64 else b64)
        except Exception as e:raise HTTPException(400,f'base64 decode: {e}')
        return _run_describe(image_bytes,question='')
    @app.post('/vision/ask')
    async def ask(req:Request):
        if vision is None:raise HTTPException(503,'vision not initialized')
        body=await req.json()
        b64=body.get('image_base64') or body.get('base64')
        question=(body.get('question') or '').strip()
        if not b64:raise HTTPException(400,'need image_base64')
        if not question:raise HTTPException(400,'need question')
        try:image_bytes=base64.b64decode(b64.split(',',1)[-1] if ',' in b64 else b64)
        except Exception as e:raise HTTPException(400,f'base64 decode: {e}')
        return _run_describe(image_bytes,question=question)
    try:
        import multipart as _mp_check
        _has_multipart=True
    except ImportError:_has_multipart=False
    if _has_multipart:
        from fastapi import UploadFile,File
        @app.post('/vision/upload')
        async def upload(image:UploadFile=File(...),question:str=''):
            if vision is None:raise HTTPException(503,'vision not initialized')
            try:image_bytes=await image.read()
            except Exception as e:raise HTTPException(400,f'upload read: {e}')
            return _run_describe(image_bytes,question=question.strip())
