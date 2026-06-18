"""Quant-mode selector: GET/POST /api/mode — Default(lossless palette) / Quick(native 8-bit) / Turbo(nested 4-bit). Default<->others = model reload (restart Adam); Quick<->Turbo on a loaded nested map = live decode-switch (no reload). v6.12.5."""
from amni.bootstrap import QUANT_VARIANTS,load_config,save_config,quant_variant
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
