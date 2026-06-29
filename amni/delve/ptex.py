"""In-process PTEX capture: every multi-agent exchange feeds Adam's learning. Each AI's first take + final (post-review) answer become a coding_ledger record (recall-able, PTEX-baked on boot) plus a raw pair into lessons/delve_debates_ptex.json. Runs inside Adam's own process now — no cross-repo bridge."""
import os,json,threading
_LOCK=threading.Lock()
def _ledger():
    try:
        from amni.serve import coding_ledger as cl;return cl
    except Exception:return None
def _pairs_path():
    d=os.path.join(os.getcwd(),"lessons");os.makedirs(d,exist_ok=True);return os.path.join(d,"delve_debates_ptex.json")
def _append_pairs(new):
    p=_pairs_path()
    with _LOCK:
        data={"grid":48,"pca_dim":6,"pairs":[]}
        if os.path.exists(p):
            try:data=json.load(open(p,encoding="utf-8"))
            except Exception:data={"grid":48,"pca_dim":6,"pairs":[]}
        data.setdefault("pairs",[]).extend(new);json.dump(data,open(p,"w",encoding="utf-8"),ensure_ascii=False)
    return p
def feed(question,items,session_id,kind="debate"):
    question=(question or "").strip() or ("[Amni-Delve "+kind+"]");by={}
    for who,text in items:
        if who!="Anthony" and (text or "").strip():by.setdefault(who,[]).append(text.strip())
    if len(by)<2:return {"ok":False,"reason":"need >=2 agents"}
    pairs=[];recorded=0;cl=_ledger()
    for ai,msgs in by.items():
        first=msgs[0];last=msgs[-1];pairs.append([question,last])
        if first!=last:pairs.append(["["+ai+" first take] "+question,first])
        if cl is not None:
            try:cl.record(task=question,approach="Amni-Delve "+kind+" — "+ai,outcome=first[:600],lesson=last[:500],success=None,session_id="delve_"+str(session_id));recorded+=1
            except Exception:pass
    path=_append_pairs(pairs);return {"ok":True,"pairs":len(pairs),"ledger_records":recorded,"file":os.path.basename(path)}
def commit(adam=None):
    cl=_ledger()
    if cl is None:return {"ok":False,"reason":"coding_ledger not found"}
    try:return {"ok":True,**cl.maybe_commit_to_ptex(adam=adam,min_new=1)}
    except Exception as e:return {"ok":False,"error":str(e)}
