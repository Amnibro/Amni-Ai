"""Delve cold memory: infinite addressable transcript archive on disk at ~0 VRAM.
Each turn is a cell keyed by token-bag address (coordinate-style, no model weights).
Hot window stays tiny; recall pulls top-k cells into the prompt. Brief = compacted digest cell.
Optional ConversationAtlas hook when encoder is available (still disk-backed)."""
import os,json,time,re,math,threading,hashlib
from amni.delve.ptex import cell_coords,_canon_stems,_doc_tokens,_query_tokens,_tok_hit
_WORD=re.compile(r"[a-z0-9]{2,}",re.I)
def _tok(t):return _doc_tokens(t)
def _jac(a,b):
    if not a and not b:return 1.0
    if not a or not b:return 0.0
    return len(a&b)/len(a|b)
def _addr(text):
    toks=_canon_stems(text)[:48]
    h=hashlib.sha1(" ".join(toks).encode("utf-8","replace")).hexdigest()[:16]
    return h
def extractive_brief(turns,max_bullets=24,max_chars=2400):
    lines=[]
    for who,text in turns:
        t=(text or "").strip().replace("\n"," ")
        if not t:continue
        m=re.search(r"##\s*Claim\s*\n(.+?)(?:\n##|\Z)",text or "",re.I|re.S)
        body=(m.group(1).strip() if m else t)[:220]
        lines.append("- **"+who+":** "+body)
        if len(lines)>=max_bullets:break
    out="## Session brief (compacted)\n"+("\n".join(lines) if lines else "(empty)")
    return out[:max_chars]
class DelveMemory:
    def __init__(s,root=None,session_key="table"):
        s.root=root or os.path.join(os.getcwd(),"experiences","delve","memory")
        os.makedirs(s.root,exist_ok=True)
        s.session_key=session_key or "table"
        s.path=os.path.join(s.root,s.session_key+".jsonl")
        s.brief_path=os.path.join(s.root,s.session_key+".brief.md")
        s.lock=threading.Lock();s.entries=[];s.brief="";s._load()
    def _load(s):
        s.entries=[]
        if os.path.exists(s.path):
            try:
                for line in open(s.path,encoding="utf-8"):
                    line=line.strip()
                    if not line:continue
                    try:s.entries.append(json.loads(line))
                    except Exception:pass
            except Exception:pass
        if os.path.exists(s.brief_path):
            try:s.brief=open(s.brief_path,encoding="utf-8").read()
            except Exception:s.brief=""
    def _append(s,row):
        with s.lock:
            s.entries.append(row)
            try:open(s.path,"a",encoding="utf-8").write(json.dumps(row,ensure_ascii=False)+"\n")
            except Exception:pass
    def write(s,who,text,query=""):
        text=(text or "").strip()
        if not text:return None
        q=(query or text)[:500];body=who+": "+text[:2000]
        cell,cx,cy=cell_coords(q,text,48)
        row={"ts":time.time(),"who":who,"q":q,"a":body,"addr":_addr(q+" "+text),"toks":list(_tok(q+" "+text))[:64],"cell":cell,"x":cx,"y":cy}
        s._append(row);return row.get("addr")
    def set_brief(s,text):
        s.brief=(text or "").strip()
        try:open(s.brief_path,"w",encoding="utf-8").write(s.brief)
        except Exception:pass
    def recall(s,query,k=5,exclude_recent=None):
        qtoks=_query_tokens(query or "")
        q_set=_tok(query or "");out=[]
        if s.brief:
            out.append({"who":"Brief","text":s.brief[:1200],"score":1.0,"kind":"brief"})
        recent=set(exclude_recent or [])
        q_cell,_,_=cell_coords(query or "","",48)
        cands=[]
        for e in s.entries:
            body=e.get("a") or ""
            if body in recent:continue
            et=set(e.get("toks") or [])|_tok((e.get("q") or "")+" "+body)
            if not et:continue
            cands.append((e,et))
        N=max(1,len(cands))
        dfs={tp:sum(1 for _,et in cands if _tok_hit(tp,et)) for tp in qtoks}
        scored=[]
        for pos,(e,et) in enumerate(cands):
            hits=0;bm=0.0
            for tp in qtoks:
                if _tok_hit(tp,et):
                    hits+=1
                    df=dfs.get(tp,1)
                    bm+=math.log(1.0+(N-df+0.5)/(df+0.5))+0.5
            jac=_jac(q_set,et)
            if hits==0 and jac<=0:continue
            ecell=e.get("cell")
            if not (isinstance(ecell,list) and len(ecell)==6):
                ecell,_,_=cell_coords(e.get("q") or "",e.get("a") or "",48)
            l1=sum(abs(a-b) for a,b in zip(q_cell,ecell))
            spatial=0.25*max(0.0,1.0-l1/120.0)
            sc=0.55*jac+0.35*(bm/max(1.0,len(qtoks)))+spatial+0.05*(pos/N)
            scored.append((sc,e))
        scored.sort(key=lambda x:(-x[0],-float(x[1].get("ts") or 0)))
        for sc,e in scored[:max(0,int(k))]:
            out.append({"who":e.get("who","?"),"text":(e.get("a") or "")[:800],"score":round(sc,3),"kind":"cell","addr":e.get("addr"),"cell":e.get("cell")})
        return out
    def pack_for_prompt(s,query,k=5,exclude_recent=None):
        hits=s.recall(query,k=k,exclude_recent=exclude_recent)
        if not hits:return ""
        lines=["Relevant cold memory (addressable archive — not full transcript):"]
        for h in hits:
            tag=h.get("kind","cell");sc=h.get("score")
            prefix="[brief]" if tag=="brief" else ("[cell "+str(h.get("addr","")[:8])+(" s="+str(sc) if sc is not None else "")+"]")
            lines.append(prefix+" "+(h.get("text") or "")[:600])
        return "\n".join(lines)
    def ingest_turns(s,turns,query=""):
        n=0
        for who,text in turns or []:
            if s.write(who,text,query=query or text):n+=1
        return n
    def clear_cold(s):
        with s.lock:
            s.entries=[];s.brief=""
            for p in (s.path,s.brief_path):
                try:
                    if os.path.exists(p):os.unlink(p)
                except Exception:pass
        return True
    def stats(s):
        return {"entries":len(s.entries),"brief_chars":len(s.brief or ""),"path":s.path,"has_brief":bool(s.brief)}
