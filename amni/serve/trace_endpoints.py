"""trace_endpoints — admin HTTP endpoints for the per-layer residual-state trace pass. Mounts:
  POST /admin/trace/attach   — register read-only forward hooks on the underlying transformer
  POST /admin/trace/start    — enable accumulation
  POST /admin/trace/stop     — disable accumulation, return snapshot
  POST /admin/trace/reset    — clear counters
  GET  /admin/trace/status   — current counter sizes
  GET  /admin/trace/snapshot — full snapshot of distinct-hash counts per layer
  POST /admin/trace/detach   — remove hooks
Wraps amni.serve.trace_hooks. Read-only on the model; lossless guarantee unaffected (no bake/weight/activation mutation)."""
from amni.serve import trace_hooks as th
def _resolve_inner_model(agent):
    candidates=[]
    for path in ('adam.svc.model','adam.svc.adam.model','adam.model','adam.runtime.model','adam.svc.runtime.model'):
        cur=agent
        ok=True
        for p in path.split('.'):
            if cur is None:ok=False;break
            cur=getattr(cur,p,None)
        if ok and cur is not None:candidates.append((path,cur))
    return candidates
def mount(app,agent):
    from fastapi import Request
    @app.post('/admin/trace/attach')
    def attach():
        cands=_resolve_inner_model(agent)
        if not cands:return {'status':'error','reason':'could not resolve inner transformer model from agent','tried':['adam.svc.model','adam.svc.adam.model','adam.model','adam.runtime.model','adam.svc.runtime.model']}
        path,model=cands[0]
        r=th.attach(model)
        r['resolved_path']=path
        return r
    @app.post('/admin/trace/detach')
    def detach():return th.detach()
    @app.post('/admin/trace/start')
    def start():return th.start()
    @app.post('/admin/trace/stop')
    def stop():return th.stop()
    @app.post('/admin/trace/reset')
    def reset():return th.reset()
    @app.get('/admin/trace/status')
    def status():return th.status()
    @app.get('/admin/trace/snapshot')
    def snapshot():return th.snapshot()
    @app.post('/admin/trace/configure_raws')
    async def configure_raws(req:Request):
        body=await req.json()
        return th.configure_raws(save=body.get('save',True),every=body.get('every',1),max_per_layer=body.get('max_per_layer',1000))
    @app.post('/admin/trace/dump_raws')
    async def dump_raws(req:Request):
        body=await req.json()
        return th.dump_raws(path=body.get('path','eval_reports/trace_raws.npz'))
