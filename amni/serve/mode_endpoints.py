"""Quant-mode selector: GET/POST /api/mode — Default(lossless palette) / Quick(native 8-bit) / Turbo(nested 4-bit). Default<->others = model reload (restart Adam); Quick<->Turbo on a loaded nested map = live decode-switch (no reload). v6.12.5."""
from amni.bootstrap import load_config,save_config
try:from amni.bootstrap import QUANT_VARIANTS,quant_variant
except ImportError:
    QUANT_VARIANTS=[{'key':'palette','repo':'amnibro/granite41-3b-palette','dir':'granite41_3b_palette','label':'Palette (default, RAM-light, lossless to fp16)','bpw':1.6,'scorecard':'bit-exact cossim=1.0 to fp16','warn':None,'ready':True},{'key':'native8','repo':'amnibro/granite41-3b-native8','dir':'granite41_3b_native8','label':'Native GF 8-bit (cleanest native QAT, cos=1-to-trained)','bpw':8.25,'scorecard':'vs fp16: short factual 100%, multi-step reasoning 97%, long-form coherent (looping ~6%)','warn':None,'ready':False},{'key':'nested4','repo':'amnibro/granite41-3b-nested4','dir':'granite41_3b_nested4','label':'Native GF 4-bit nested (smallest+fast: 4-bit fast path + 8-bit full read from ONE map)','bpw':4.16,'scorecard':'4-bit fast path vs fp16: short 100%, reasoning 97%, long-coverage 91%; full 8-bit read 100%/100%','warn':'DEGRADATION: the 4-bit fast path falls into repetition loops on long-form answers ~16-18% of the time (vs ~6% at 8-bit), and short factual is ~94-100% run-to-run. Short answers + reasoning hold up; for long-form or critical work prefer the native8 variant or read the full 8-bit tier of this map.','ready':False}]
    def quant_variant(key):return next((v for v in QUANT_VARIANTS if v['key']==key),None)
_MODE2KEY={'default':'palette','quick':'native8','turbo':'nested4'}
_KEY2MODE={v:k for k,v in _MODE2KEY.items()}
def _svc(adam):return getattr(adam,'svc',None)
def mount(app,adam):
    from fastapi import Request
    @app.get('/api/mode')
    def get_mode():
        cfg=load_config();cur=cfg.get('quant_mode','default')
        modes=[{'mode':_KEY2MODE.get(v['key'],v['key']),'key':v['key'],'label':v['label'],'scorecard':v['scorecard'],'warn':v.get('warn'),'ready':bool(v.get('ready')),'bpw':v['bpw']} for v in QUANT_VARIANTS]
        return {'current':cur,'modes':modes,'loaded':bool(_svc(adam))}
    @app.post('/api/mode')
    async def set_mode(req:Request):
        try:body=await req.json()
        except Exception:body={}
        mode=str(body.get('mode') or 'default').lower();key=_MODE2KEY.get(mode,mode);v=quant_variant(key)
        if v is None:return {'ok':False,'error':f'unknown mode "{mode}"'}
        cfg=load_config();svc=_svc(adam);nested=bool(svc and getattr(svc,'_nested_map',False))
        if mode in ('quick','turbo') and nested:
            try:setattr(svc,'_slice_bits',4 if mode=='turbo' else 0)
            except Exception:pass
            cfg['quant_mode']=mode;save_config(cfg)
            return {'ok':True,'mode':mode,'applied':'live','reload':False,'warn':v.get('warn')}
        cfg['quant_mode']=mode;cfg['quant_variant']=key;save_config(cfg)
        if not v.get('ready'):return {'ok':True,'mode':mode,'applied':'pending','reload':True,'ready':False,'note':f"the '{key}' bake isn't installed yet — export + upload it, then restart Adam.",'warn':v.get('warn')}
        return {'ok':True,'mode':mode,'applied':'pending','reload':True,'note':'restart Adam (amni serve) to load this model.','warn':v.get('warn')}
    @app.post('/rawgen')
    async def rawgen(req:Request):
        try:body=await req.json()
        except Exception:body={}
        svc=_svc(adam)
        if svc is None:return {'error':'no inference svc'}
        pr=str(body.get('prompt',''));sy=body.get('system') or 'You are a helpful assistant. Answer the question directly and completely.';mt=int(body.get('max_tokens',64))
        try:resp,n=svc.chat(pr,system=sy,max_new_tokens=mt,do_sample=False,kb_top_k=0,cache_writeback=False)
        except TypeError:resp,n=svc.chat(pr,system=sy,max_new_tokens=mt,do_sample=False,kb_top_k=0)
        return {'text':resp,'tokens':n}
