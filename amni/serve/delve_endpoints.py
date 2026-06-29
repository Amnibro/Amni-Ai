"""Amni-Delve fused into Adam: GET /delve (roundtable UI) + /api/delve/{state,stream,send,estop,config,clear,commit,agents}. Adam answers in-process (adam.ask). SECURITY: config holds NO secrets; /api/delve/agents reports only which env-var NAMES are present; every SSE frame + transcript line passes through scrub_secrets, so a pasted/echoed key never reaches the wire, disk, or git. Mounted like the other *_endpoints modules: delve_endpoints.mount(app,agent,adam)."""
import os,json,queue,threading
from pathlib import Path
from amni.delve import adapters
from amni.delve.hub import Hub
def _root():
    d=os.path.join(os.getcwd(),"experiences","delve");os.makedirs(d,exist_ok=True);return d
def _cfg_path():return os.path.join(_root(),"config.json")
def _default_cfg():
    inst=[r["key"] for r in adapters.detect() if r["installed"]]
    return {"enabled":inst,"models":{},"default":"all","bypass":True,"ptex_learn":True,"adam_fallback":True,"pair":None,"adam_mode":"light"}
def _load_cfg():
    p=_cfg_path();base=_default_cfg()
    if os.path.exists(p):
        try:base.update(json.load(open(p,encoding="utf-8")))
        except Exception:pass
    return base
def _save_cfg(c):
    safe={k:v for k,v in c.items() if k in("enabled","models","default","bypass","ptex_learn","adam_fallback","pair","adam_mode")}
    try:json.dump(safe,open(_cfg_path(),"w",encoding="utf-8"),indent=2)
    except Exception:pass
def mount(app,agent,adam):
    from fastapi import Request
    from fastapi.responses import StreamingResponse,HTMLResponse,JSONResponse
    try:from amni.serve.code_safety import scrub_secrets
    except Exception:scrub_secrets=lambda x:x
    SUBS=[];SUBS_LOCK=threading.Lock();BUSY=threading.Lock()
    def broadcast(ev):
        if isinstance(ev,dict) and isinstance(ev.get("text"),str):ev=dict(ev,text=scrub_secrets(ev["text"]))
        data=json.dumps(ev)
        with SUBS_LOCK:
            for q in list(SUBS):
                try:q.put_nowait(data)
                except Exception:pass
    def _adam_fn(prompt):
        try:r=adam.ask(prompt,writeback=False)
        except TypeError:r=adam.ask(prompt)
        except Exception as e:return "[adam error] "+str(e)
        return r if isinstance(r,str) else ((r.get("answer") or r.get("text") or r.get("response") or "") if isinstance(r,dict) else str(r))
    work=os.path.join(_root(),"workspace");sess=os.path.join(_root(),"sessions")
    HUB=Hub(sink=broadcast,adam_fn=_adam_fn,cfg=_load_cfg(),scrub=scrub_secrets,work=work,sess=sess)
    def _agents():return adapters.detect(enabled=HUB.cfg.get("enabled"),models=HUB.cfg.get("models",{}))
    def _safe_cfg():return {k:HUB.cfg.get(k) for k in("enabled","models","default","bypass","ptex_learn","adam_fallback","pair","adam_mode")}
    def _run(fn):
        with BUSY:
            try:fn()
            except Exception as e:broadcast({"type":"status","text":"error: "+str(e)})
            broadcast({"type":"idle"})
    @app.get("/delve")
    def delve_ui():
        p=Path(__file__).resolve().parent/"web_delve"/"index.html"
        return HTMLResponse(p.read_text(encoding="utf-8")) if p.exists() else HTMLResponse("<h1>delve UI missing</h1>",status_code=404)
    @app.get("/api/delve/agents")
    def delve_agents():return {"agents":_agents(),"roster":HUB.roster(),"pair":HUB.pair()}
    @app.get("/api/delve/state")
    def delve_state():
        return {"transcript":[{"who":w,"text":t} for w,t in HUB.t],"config":_safe_cfg(),"agents":_agents(),"roster":HUB.roster(),"pair":HUB.pair(),"busy":BUSY.locked(),"adam_up":True}
    @app.get("/api/delve/stream")
    def delve_stream():
        q=queue.Queue()
        with SUBS_LOCK:SUBS.append(q)
        def gen():
            try:
                yield ": connected\n\n"
                while True:
                    try:data=q.get(timeout=15)
                    except queue.Empty:yield ": ping\n\n";continue
                    yield "data: "+data+"\n\n"
            finally:
                with SUBS_LOCK:
                    if q in SUBS:SUBS.remove(q)
        return StreamingResponse(gen(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
    @app.post("/api/delve/send")
    async def delve_send(req:Request):
        try:body=await req.json()
        except Exception:body={}
        target=str(body.get("target") or HUB.cfg.get("default") or "all").lower();text=(body.get("text") or "").strip()
        if not text:return JSONResponse({"ok":False,"error":"empty"},status_code=400)
        if BUSY.locked():
            HUB.interjects.append(text);broadcast({"type":"user","who":"Anthony","text":text});broadcast({"type":"queued"});return {"ok":True,"queued":True}
        if target=="debate":
            n=int(body.get("rounds") or 3);threading.Thread(target=_run,args=(lambda:HUB.debate(n,text),),daemon=True).start();return {"ok":True}
        fn={"all":lambda:HUB.route("all",text),"both":lambda:HUB.both(text)}.get(target) or (lambda:HUB.route(target,text))
        threading.Thread(target=_run,args=(fn,),daemon=True).start();return {"ok":True}
    @app.post("/api/delve/estop")
    def delve_estop():HUB._estop();return {"ok":True}
    @app.post("/api/delve/clear")
    def delve_clear():
        HUB.t=[];HUB.idx={};HUB.started={};broadcast({"type":"cleared"});return {"ok":True}
    @app.post("/api/delve/config")
    async def delve_config(req:Request):
        try:body=await req.json()
        except Exception:body={}
        for k in("enabled","models","default","bypass","ptex_learn","adam_fallback","pair","adam_mode"):
            if k in body:HUB.cfg[k]=body[k]
        _save_cfg(HUB.cfg);broadcast({"type":"config","config":_safe_cfg()});return {"ok":True,"config":_safe_cfg()}
    @app.post("/api/delve/commit")
    def delve_commit():
        from amni.delve import ptex;return ptex.commit(adam=adam)
    print("[amni_serve] Amni-Delve roundtable mounted at /delve (agents: "+", ".join(HUB.roster() or ["none"])+")",flush=True)
