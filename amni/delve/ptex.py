"""In-process PTEX capture & recall: every multi-agent exchange feeds Adam's learning.
Each AI's first take + final (post-review) answer become a coding_ledger record (recall-able,
PTEX-baked on boot) plus a raw pair into lessons/delve_debates_ptex.json with 6D SimHash
spatial coordinates and hybrid BM25 + 6D L1 recall."""
import os,re,json,math,time,hashlib,threading
_LOCK=threading.Lock()
_CACHE={"path":"","mtime":-1.0,"size":-1,"data":None,"indexed":None}
REJ_MARK="[REJECTED by peers]"
_FIRST_TAKE_RX=re.compile(r'^\[(\w+)\s+first\s+take\]\s*',re.I)
_TECH_TOK_RX=re.compile(r'[._/-]|[0-9]')
_PTEX_STOP=set("the a an is are was were be been being do does did done what which who whom whose when where why how this that these those there here and or but if then else for with from into over under about not no yes can could will would shall should may might must have has had having".split())
def _ledger():
    try:
        from amni.serve import coding_ledger as cl;return cl
    except Exception:return None
def _pairs_path():
    p=(os.environ.get("AMNI_DELVE_PAIRS") or "").strip()
    if p:return p
    d=os.path.join(os.getcwd(),"lessons");os.makedirs(d,exist_ok=True);return os.path.join(d,"delve_debates_ptex.json")
def _stem(w):
    if len(w)>5 and w.endswith("ing"):return w[:-3]
    if len(w)>4 and w.endswith("ed"):return w[:-2]
    if len(w)>4 and w.endswith("es"):return w[:-2]
    if len(w)>3 and w.endswith("s") and not w.endswith("ss"):return w[:-1]
    return w
def _ordered_stems(text,cap=32):
    raw=re.findall(r'[a-z0-9_./-]+',str(text or "").lower())
    out=[]
    for t in raw:
        t=t.strip("._/-")
        if len(t)>=2 and t not in _PTEX_STOP:
            st=_stem(t)
            out.append(st if len(st)>=3 else t)
            if len(out)>=cap:break
    return out
def _doc_bigrams(text,cap=28):
    stems=_ordered_stems(text,cap=cap+1)
    if len(stems)<2:return set()
    return {stems[i]+"_"+stems[i+1] for i in range(len(stems)-1)}
def _doc_tokens(text):
    raw=re.findall(r'[a-z0-9_./-]+',str(text or "").lower())
    out=set()
    for t in raw:
        t=t.strip("._/-")
        if len(t)>=2 and t not in _PTEX_STOP:
            out.add(t)
            st=_stem(t)
            if len(st)>=3:out.add(st)
    return out
def _canon_stems(text):
    raw=re.findall(r'[a-z0-9_./-]+',str(text or "").lower())
    out=set()
    for t in raw:
        t=t.strip("._/-")
        if len(t)>=2 and t not in _PTEX_STOP:
            st=_stem(t)
            out.add(st if len(st)>=3 else t)
    return sorted(out)
def _query_tokens(query,cap=14):
    raw=re.findall(r'[a-z0-9_./-]{3,}',str(query or "").lower())
    out=[];seen=set()
    for t in raw:
        t=t.strip("._/-")
        if len(t)<3 or t in _PTEX_STOP:continue
        st=_stem(t)
        key=st if len(st)>=3 else t
        if key not in seen:
            seen.add(key);out.append((t,key))
        if len(out)>=cap:break
    return out
def _tok_hit(t_pair,doc_toks):
    t,st=t_pair
    if t in doc_toks or st in doc_toks:return True
    if len(st)>=4:
        for dt in doc_toks:
            if len(dt)>=4 and (dt.startswith(st) or st.startswith(dt)):
                return True
    return False
def cell_coords(q,a="",grid=48):
    """Deterministic 6D weighted SimHash / Gaussian-CDF projection onto [0, grid-1]^6."""
    grid=max(8,int(grid or 48))
    q_clean=_FIRST_TAKE_RX.sub("",str(q or ""))
    q_toks=_canon_stems(q_clean)
    a_toks=_canon_stems(a)
    bg=sorted(_doc_bigrams(q_clean+" "+str(a or ""),cap=16))
    weighted=[];seen=set();roots=set()
    for t in q_toks[:20]:
        w=1.45*(1.55 if _TECH_TOK_RX.search(t) else 1.0)*(1.0+0.06*min(len(t),10))
        weighted.append((t,w));seen.add(t)
        if len(t)>=5:roots.add(t[:5])
    for t in a_toks[:28]:
        if t in seen:continue
        w=1.0*(1.55 if _TECH_TOK_RX.search(t) else 1.0)*(1.0+0.06*min(len(t),10))
        weighted.append((t,w));seen.add(t)
        if len(t)>=5:roots.add(t[:5])
    for r5 in sorted(roots)[:16]:
        weighted.append(("r5:"+r5,0.45))
    for b in bg[:6]:
        weighted.append(("bg:"+b,0.30))
    if not weighted:
        weighted=[(str(q or a or "empty"),1.0)]
    acc=[0.0]*6;w2_sum=0.0
    for tok,w in weighted:
        h=hashlib.blake2b(tok.encode("utf-8","replace"),digest_size=6).digest()
        w2_sum+=w*w
        for d in range(6):
            z=(float(h[d])-127.5)/73.900834
            acc[d]+=w*z
    norm=math.sqrt(max(w2_sum,1e-9))
    cell=[]
    for d in range(6):
        u=acc[d]/norm
        p=0.5*(1.0+math.erf(u/1.41421356237))
        c=int(round(p*(grid-1)))
        cell.append(max(0,min(grid-1,c)))
    return cell,cell[0],cell[1]
def _build_indexed(pairs,grid=48,base_ts=0):
    if not base_ts:base_ts=int(time.time())
    out=[];n_tot=len(pairs)
    for idx,pr in enumerate(pairs):
        if not isinstance(pr,(list,tuple)) or len(pr)<2:continue
        q,a=str(pr[0] or ""),str(pr[1] or "")
        if not (q.strip() or a.strip()):continue
        rk=str(pr[2]) if (len(pr)>2 and pr[2]) else ""
        ts=int(pr[3]) if (len(pr)>3 and isinstance(pr[3],(int,float)) and pr[3]>0) else (base_ts-max(0,n_tot-1-idx))
        sl=str(pr[4]) if (len(pr)>4 and pr[4]) else ""
        m=_FIRST_TAKE_RX.match(q)
        who=m.group(1) if m else ""
        if not sl:
            sl=("delve_"+who) if who else (rk if rk else "__global__")
        is_rej=(REJ_MARK in a or REJ_MARK.lower() in a.lower())
        rej_why=a.split(REJ_MARK,1)[1].strip() if REJ_MARK in a else ""
        qt_set=_doc_tokens(_FIRST_TAKE_RX.sub("",q))
        at_set=_doc_tokens(rej_why if is_rej else a)
        all_set=qt_set|at_set
        bigrams=_doc_bigrams(q+" "+(rej_why if is_rej else a))
        cell,cx,cy=cell_coords(q,a,grid)
        out.append({
            "idx":idx,"q":q,"a":a,"rk":rk,"ts":ts,"sl":sl,"who":who,
            "is_first":bool(m),"is_rej":is_rej,"rej_why":rej_why,
            "qt_set":qt_set,"at_set":at_set,"all_set":all_set,
            "bigrams":bigrams,"cell":cell,"x":cx,"y":cy,
        })
    return out
def _load_pairs_cached():
    p=_pairs_path()
    try:
        st=os.stat(p)
        with _LOCK:
            if _CACHE["path"]==p and _CACHE["mtime"]==st.st_mtime and _CACHE["size"]==st.st_size and _CACHE["data"] is not None:
                return list(_CACHE["data"].get("pairs") or [])
            with open(p,encoding="utf-8") as f:
                d=json.load(f) or {}
            pairs=list(d.get("pairs") or [])
            grid=int(d.get("grid") or 48)
            _CACHE.update(path=p,mtime=st.st_mtime,size=st.st_size,data=d,indexed=_build_indexed(pairs,grid,int(st.st_mtime)))
            return pairs
    except Exception:
        return []
def _load_indexed_cached():
    _load_pairs_cached()
    with _LOCK:
        return list(_CACHE.get("indexed") or []),int((_CACHE.get("data") or {}).get("grid") or 48)
def recall_local(query,k=3,room=""):
    qtoks=_query_tokens(query)
    if not qtoks:return []
    indexed,grid=_load_indexed_cached()
    if not indexed:return []
    need=2 if len(qtoks)>=3 else 1
    rk=str(room or "").strip()
    q_bigrams=_doc_bigrams(query)
    q_cell,_,_=cell_coords(query,"",grid)
    cands=[];rej_topics=[]
    for row in indexed[-4000:]:
        if rk and row["rk"] and row["rk"]!=rk:continue
        if row["is_rej"]:
            if row["all_set"]:
                rej_topics.append((row["q"].strip().lower(),row["qt_set"],row["at_set"]))
            continue
        if not row["a"].strip() or not row["all_set"]:continue
        cands.append(row)
    if not cands:return []
    N=len(cands)
    dfs={tp:sum(1 for r in cands if _tok_hit(tp,r["all_set"])) for tp in qtoks}
    scored=[]
    for pos,r in enumerate(cands):
        hits=0;bm=0.0;all_set=r["all_set"];qt_set=r["qt_set"]
        for tp in qtoks:
            if _tok_hit(tp,all_set):
                hits+=1
                df=dfs.get(tp,1)
                idf=math.log(1.0+(N-df+0.5)/(df+0.5))+0.5
                q_boost=1.38 if _tok_hit(tp,qt_set) else 1.0
                tech_boost=1.22 if _TECH_TOK_RX.search(tp[0]) else 1.0
                bm+=idf*q_boost*tech_boost
        if hits>=need:
            bg_hits=len(q_bigrams & r["bigrams"]) if q_bigrams else 0
            bg_bonus=0.60*min(bg_hits,3)
            l1=sum(abs(a-b) for a,b in zip(q_cell,r["cell"]))
            spatial_bonus=0.28*max(0.0,1.0-l1/max(1.0,2.5*grid))
            recency=0.08*(pos/max(1,N))
            first_pen=-0.25 if r["is_first"] else 0.0
            superseded_pen=0.0
            if rej_topics:
                rq_clean=_FIRST_TAKE_RX.sub("",r["q"]).strip().lower()
                for rej_q,rej_qt,rej_at in rej_topics:
                    if rq_clean and rq_clean==rej_q and r["is_first"]:
                        superseded_pen=-1.60;break
                    if r["at_set"] and rej_at:
                        ov=len(r["at_set"] & rej_at)/float(max(1,min(len(r["at_set"]),len(rej_at))))
                        if ov>=0.65:
                            superseded_pen=min(superseded_pen,-1.10)
            scored.append((hits,bm+bg_bonus+spatial_bonus+recency+first_pen+superseded_pen,re.sub(r'\s+',' ',r["a"]).strip()[:300]))
    scored.sort(key=lambda x:(-x[0],-x[1]))
    seen=set();out=[]
    for _,sc,a in scored:
        if sc<-0.5 and len(scored)>k:continue
        if a and a not in seen:seen.add(a);out.append(a)
        if len(out)>=k:break
    return out
def recall_cells_local(slot,query="",k=5,room=""):
    indexed,grid=_load_indexed_cached()
    if not indexed:return [],[]
    qtoks=_query_tokens(query) if query else []
    q_cell,_,_=cell_coords(query,"",grid) if query else ([grid//2]*6,grid//2,grid//2)
    rk=str(room or "").strip()
    want_sl=str(slot or "").strip()
    own_scored=[];shared_scored=[]
    for pos,r in enumerate(indexed[-4000:]):
        if r["is_rej"] or not r["a"].strip():continue
        if rk and r["rk"] and r["rk"]!=rk:continue
        hits=sum(1 for tp in qtoks if _tok_hit(tp,r["all_set"])) if qtoks else 1
        if qtoks and hits<1:continue
        l1=sum(abs(a-b) for a,b in zip(q_cell,r["cell"]))
        sc=hits*1.5+0.4*max(0.0,1.0-l1/max(1.0,2.5*grid))+0.05*(pos/max(1,len(indexed)))
        txt=re.sub(r'\s+',' ',r["a"]).strip()[:300]
        if want_sl and (r["sl"]==want_sl or r["rk"]==want_sl):
            own_scored.append((sc,txt))
        else:
            shared_scored.append((sc,txt))
    own_scored.sort(key=lambda x:-x[0]);shared_scored.sort(key=lambda x:-x[0])
    own=[];shared=[]
    for _,t in own_scored:
        if t not in own:own.append(t)
        if len(own)>=k:break
    for _,t in shared_scored:
        if t not in own and t not in shared:shared.append(t)
        if len(shared)>=k:break
    return own,shared
def recall_rejections(query="",k=4,room=""):
    indexed,_=_load_indexed_cached()
    if not indexed:return []
    qtoks=_query_tokens(query) if query else []
    rk=str(room or "").strip()
    out=[]
    rejs=[r for r in reversed(indexed[-2000:]) if r["is_rej"]]
    for idx,r in enumerate(rejs):
        if rk and r["rk"] and r["rk"]!=rk:continue
        why=r["rej_why"]
        if not why:continue
        if qtoks:
            hits=sum(1 for tp in qtoks if _tok_hit(tp,r["all_set"]))
            if hits<1:continue
            sc=hits-idx*1e-4
        else:
            sc=-idx*1e-4
        who=r["who"]
        clean=re.sub(r'\s+',' ',(who+": "+why) if who and not why.lower().startswith(who.lower()) else why).strip()[:280]
        out.append((sc,clean))
    out.sort(key=lambda x:-x[0])
    seen=set();res=[]
    for _,txt in out:
        if txt and txt not in seen:seen.add(txt);res.append(txt)
        if len(res)>=k:break
    return res
def _append_pairs(new,replace_q=False):
    p=_pairs_path()
    with _LOCK:
        data={"grid":48,"pca_dim":6,"pairs":[]}
        if os.path.exists(p):
            try:
                with open(p,encoding="utf-8") as f:data=json.load(f) or data
            except Exception:data={"grid":48,"pca_dim":6,"pairs":[]}
        existing=data.setdefault("pairs",[])
        index_by_key={};index_by_qslot={}
        for i,row in enumerate(existing):
            if isinstance(row,(list,tuple)) and len(row)>=2:
                rk=str(row[2]) if len(row)>2 and row[2] else ""
                sl=str(row[4]) if len(row)>4 and row[4] else ""
                index_by_key[(str(row[0]),str(row[1]),rk)]=i
                if REJ_MARK not in str(row[1]):
                    index_by_qslot[(str(row[0]),sl or rk)]=i
        for item in (new or []):
            if not isinstance(item,(list,tuple)) or len(item)<2:continue
            item_list=list(item)
            rk=str(item_list[2]) if len(item_list)>2 and item_list[2] else ""
            sl=str(item_list[4]) if len(item_list)>4 and item_list[4] else ""
            key=(str(item_list[0]),str(item_list[1]),rk)
            qslot=(str(item_list[0]),sl or rk)
            if replace_q and qslot in index_by_qslot and index_by_qslot[qslot]<len(existing):
                existing[index_by_qslot[qslot]]=item_list
            elif key in index_by_key and index_by_key[key]<len(existing):
                existing[index_by_key[key]]=item_list
            else:
                idx_new=len(existing)
                index_by_key[key]=idx_new;index_by_qslot[qslot]=idx_new
                existing.append(item_list)
        tmp=p+".tmp."+str(os.getpid())
        with open(tmp,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False)
        os.replace(tmp,p)
        try:
            st=os.stat(p)
            grid=int(data.get("grid") or 48)
            _CACHE.update(path=p,mtime=st.st_mtime,size=st.st_size,data=data,indexed=_build_indexed(existing,grid,int(st.st_mtime)))
        except Exception:
            _CACHE["data"]=None;_CACHE["indexed"]=None
    return p
def record_local(slot,question,answer,room="",replace_q=False):
    q=re.sub(r'\s+',' ',str(question or "")).strip()[:600]
    a=re.sub(r'\s+',' ',str(answer or "")).strip()[:1200]
    if not q or not a:return {"ok":False,"error":"empty question or answer"}
    rk=str(room or "").strip()
    sl=str(slot or "__global__").strip()
    if not rk and "_" in sl and not sl.startswith("delve_") and not sl.startswith("__"):
        rk=sl.rsplit("_",1)[0]
    p=_append_pairs([[q,a,rk,int(time.time()),sl]],replace_q=replace_q)
    return {"ok":True,"recorded":True,"slot":sl,"file":os.path.basename(p)}
def feed(question,items,session_id,kind="debate",corrections=None,room=""):
    question=(question or "").strip() or ("[Amni-Delve "+kind+"]")
    corr=corrections if isinstance(corrections,dict) else {}
    by={}
    for who,text in items:
        if who!="Anthony" and (text or "").strip():by.setdefault(who,[]).append(text.strip())
    if not by:return {"ok":False,"reason":"no usable messages"}
    pairs=[];recorded=0;corrected=0;cl=_ledger();now_ts=int(time.time())
    rk=str(room or session_id or "")
    for ai,msgs in by.items():
        first=msgs[0];last=msgs[-1];c=(corr.get(ai) or "").strip()
        corrected+=1 if c else 0
        sl="delve_"+str(ai)
        pairs.append([question,(("[REJECTED by peers] "+c) if c else last),rk,now_ts,sl])
        if first!=last:pairs.append(["["+ai+" first take] "+question,first,rk,now_ts,sl])
        if cl is not None:
            try:
                cl.record(
                    task=question,
                    approach="Amni-Delve "+kind+" — "+ai,
                    outcome=last[:600],
                    lesson=(("PEER CORRECTION — do not repeat this: "+c)[:500] if c else last[:500]),
                    success=(False if c else None),
                    session_id="delve_"+str(session_id),
                )
                recorded+=1
            except Exception:pass
    path=_append_pairs(pairs)
    return {"ok":True,"pairs":len(pairs),"ledger_records":recorded,"corrected":corrected,"file":os.path.basename(path)}
def commit(adam=None):
    cl=_ledger()
    if cl is None:return {"ok":False,"reason":"coding_ledger not found"}
    try:return {"ok":True,**cl.maybe_commit_to_ptex(adam=adam,min_new=1)}
    except Exception as e:return {"ok":False,"error":str(e)}
