import os,re,sys,json,time,shutil,hashlib,argparse,statistics,subprocess,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
ED=ROOT/"data"/"pipe1"/"evalset";RD=ROOT/"data"/"pipe1"/"results";SCR=Path(os.environ.get("PIPE1_SCRATCH","/tmp/claude-1000/-home-anmire/81ad65c4-2e7d-457b-bcca-dbc14db35a0d/scratchpad/p1"))
MODEL=os.environ.get("PIPE1_MODEL","/home/anmire/ai/Amni-Ai/exports/adam_t/t30m_sft3/model.gf17");BIN=os.environ.get("PIPE1_BIN",str(SCR/"adamt-p1"))
SINGLE=("wd_fact","wd_value","compute","commonsense","explain","chat");CORR=("corr_world","corr_poison","corr_personal")
def rd(p):return [json.loads(l) for l in open(p) if l.strip()]
def sha(p):h=hashlib.sha256();f=open(p,"rb");[h.update(b) for b in iter(lambda:f.read(1<<20),b"")];return h.hexdigest()
def setenv(p1:bool,s,run:Path,notask:bool=False):
    shutil.rmtree(run,ignore_errors=True);run.mkdir(parents=True);[os.environ.pop(k,None) for k in ("AMNI_PIPE1_S","AMNI_COGNITION","AMNI_UMEM_TOROID","AMNI_PIPE1_NOTASK")]
    os.environ.update(AMNI_ADAM_T=os.environ.get("AMNI_ADAM_T") or MODEL,AMNI_ADAM_T_BIN=os.environ.get("AMNI_ADAM_T_BIN") or BIN,AMNI_ADAM_T_HARNESS="1",AMNI_STORE2="1",AMNI_PIPE1="1" if p1 else "0",AMNI_UMEM_DIR=str(run/"umem"),AMNI_PIPE1_DIR=str(run/"p1g"),AMNI_COGNITION="0")
    s is not None and os.environ.__setitem__("AMNI_PIPE1_S",str(s));notask and os.environ.__setitem__("AMNI_PIPE1_NOTASK","1")
from amni.serve import adam_t_client as C,adam_t_harness as H,pipe1 as P,user_memory as U
HOLD={"run":None,"ref":0.0,"gin":0.0,"gout":0.0,"blends":[],"chat":None}
_run,_ref,_chat,_blend=H.run,P.refine,P.chat,H.blend
def w_run(*a,**k):r=_run(*a,**k);HOLD["run"]=r;return r
def w_ref(*a,**k):
    t=time.perf_counter()
    try:return _ref(*a,**k)
    finally:HOLD["ref"]+=time.perf_counter()-t
def w_chat(*a,**k):
    t=time.perf_counter()
    try:r=_chat(*a,**k);HOLD["chat"]=r;return r
    finally:HOLD["ref"]+=time.perf_counter()-t
def w_blend(cl,q,M,mx):HOLD["blends"].append({"lines":list(M.get("lines") or []),"sup":list(M.get("sup") or []),"extract":M.get("extract",""),"must":list(M.get("must") or []),"route":M.get("route")});return _blend(cl,q,M,mx)
H.run,P.refine,P.chat,H.blend=w_run,w_ref,w_chat,w_blend
def wrap_gen(cl):
    T=type(cl);g0=T.gen
    if getattr(g0,"_p1",False):return
    def g(self,*a,**k):
        t=time.perf_counter()
        try:return g0(self,*a,**k)
        finally:HOLD["gin" if getattr(P._RF,"on",False) else "gout"]+=time.perf_counter()-t
    g._p1=True;T.gen=g
def ask(m,h,u):
    HOLD.update(run=None,ref=0.0,gin=0.0,gout=0.0,blends=[],chat=None);t=time.perf_counter();r=C.answer(m,h,256,u) or "";w=time.perf_counter()-t
    return r,{"wall_ms":w*1000,"ref_ms":HOLD["ref"]*1000,"gin_ms":HOLD["gin"]*1000,"gout_ms":HOLD["gout"]*1000,"lat13_ms":max(0.0,(w-HOLD["ref"]-HOLD["gout"])*1000),"run":HOLD["run"],"blends":list(HOLD["blends"]),"chat":HOLD["chat"]}
nrm=lambda x:" ".join(re.sub(r"[^a-z0-9 ]+"," ",C.asc(str(x or "")).lower()).split())
has=lambda a,b:bool(nrm(a)) and f" {nrm(a)} " in f" {nrm(b)} "
nums=lambda x:set(re.findall(r"\d+(?:\.\d+)?",C.asc(str(x or "")).replace(",","")))
SW=set(U._STOPW.split())
def grounded(t:str,src:str)->bool:
    sl={w.lower() for w in re.findall(r"[A-Za-z][\w'-]*",src)};sp={w[:5] for w in sl};cw=[w for w in re.findall(r"[a-z]+",t.lower()) if len(w)>3 and w not in SW];nw=[w for w in cw if w[:5] not in sp]
    cap=[w for x in re.split(r"(?<=[.!?])\s+",t) for w in re.findall(r"\b[A-Z][\w'-]*",' '.join(x.split()[1:])) if w!="I" and w.lower() not in sl]
    return nums(t)<=nums(src) and not cap and len(nw)<=0.34*max(1,len(cw))
MO="january february march april may june july august september october november december".split()
def gold(ans,t)->bool:
    for a in ans or []:
        d=re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})",str(a))
        if d and d[1] in t and MO[int(d[2])-1] in t.lower() and re.search(r"\b0?%d\b"%int(d[3]),t):return True
        if not d and (has(a,t) or re.fullmatch(r"[\d.,]+",str(a)) and str(a).replace(",","") in nums(t)):return True
    return False
def slotinfo(ix:dict,p1:bool,q:str)->dict:
    r=ix["run"];S=(r or {}).get("slots") or []
    if r is None:return {"route":"chat","draft":[],"mat":"","must":[]}
    D=[s for s in S if s.get("pdraft") is not None] if p1 else [];B=ix["blends"]
    if p1:
        mu=[m for s in D for m in ([s["span"]] if s.get("kind")!="explain" and s.get("span") else [])+re.findall(r"\d[\d,.]*\d|\d",s["pdraft"])];mt=" ".join(x for s in D for x in s.get("pmat") or [])
        rt=D[0]["mroute"] if D else "corr" if S and S[0].get("kind")=="corr" else "hedge" if S and all(s.get("hedge") or s.get("unk") for s in S) else "other"
        return {"route":rt,"draft":[s["pdraft"] for s in D],"mat":mt,"must":mu,"s_req":[s.get("ps") for s in D],"s_used":[s.get("pu") for s in D],"calls":[s.get("pn") for s in D],"blend":[s.get("blend") for s in D],"ret":int((r.get("n") or {}).get("p1r",0))}
    mt=" ".join(x for b in B for x in b["lines"]+b["sup"]+[b["extract"]]);rt=B[0]["route"] if B else next((s.get("mroute") for s in S if s.get("mroute")),None) or ("hedge" if S and all(s.get("hedge") or s.get("unk") for s in S) else "other")
    return {"route":rt,"draft":[b["extract"] for b in B],"mat":mt,"must":[],"blend":[s.get("blend") for s in S if s.get("mroute")]}
def coh(t,q):return bool(H._coh(t,q))
def clean(t):return not H._BOIL.search(t) and not H.emtpl(t)
def pcache():
    from amni.compute import store1 as D1
    for f in sorted(D1.DIR2.glob(f"name_f{D1.FD2}*"))+[D1.DIR2/"store2.gf17pg"]:
        with open(f,"rb") as fh:
            while fh.read(1<<24):pass
def warm():
    pcache()
    for i in range(20):C.answer(f"Who founded the village of Qorvath{i}ek?",[],64,"warm");C.answer(f"What is {i+3} plus {2*i+7}?",[],64,"warm")
def single(items,p1,s,tag,notask=False):
    R=[];run=SCR/"runs"/tag;setenv(p1,s,run,notask);cl=C.client();wrap_gen(cl);warm()
    for x in items:
        m=x.get("q") or x.get("msg");t,ix=ask(m,[],f"{tag}-{x['id']}");si=slotinfo(ix,p1,m);src=si["mat"]+" "+m
        R.append({"id":x["id"],"cat":x["cat"],"q":m,"reply":t,**si,"fp":(all(has(z,t) for z in si["must"]) and nums(t)<=nums(src)) if si["draft"] else None,"grounded":grounded(t,src) if si["mat"].strip() else None,"coh":coh(t,m),"clean":clean(t),"words":len(t.split()),"gold":gold(x.get("answers"),t) if x.get("answers") else None,**{k:ix[k] for k in ("wall_ms","ref_ms","gin_ms","gout_ms","lat13_ms")}})
    return R
def d2(T):g=[(a,b) for t in T for w in [nrm(t).split()] for a,b in zip(w,w[1:])];return len(set(g))/len(g) if g else 0.0
def oprep(T):o=[" ".join(nrm(t).split()[:3]) for t in T];c=collections.Counter(o);return sum(c[z]>1 for z in o)/len(o) if o else 0.0
def med(v):return round(statistics.median(v),3) if v else None
def p90(v):return round(sorted(v)[min(len(v)-1,int(0.9*len(v)))],3) if v else None
def summ(R):
    G=collections.defaultdict(list);[G[r["route"]].append(r) for r in R];G["ALL"]=R;out={}
    for k,L in G.items():
        T=[r["reply"] for r in L];D=[r for r in L if r.get("draft")];B=[b for r in D for b in (r.get("blend") or [])]
        out[k]={"n":len(L),"n_draft":len(D),"fact_preservation":round(sum(bool(r["fp"]) for r in D)/len(D),4) if D else None,"grounded":round(sum(bool(r["grounded"]) for r in L if r["grounded"] is not None)/max(1,sum(r["grounded"] is not None for r in L)),4) if any(r["grounded"] is not None for r in L) else None,
        "refine_pass":round(sum(b is True for b in B)/len(B),4) if B else None,"model_called":round(sum(b is not None for b in B)/len(B),4) if B else None,"retry_pass":round(sum(1 for r in D for a,b in zip(r.get("s_req") or [],r.get("s_used") or []) if a and b and b<a)/max(1,len(B)),4) if D and "s_req" in D[0] else None,"fallback":round(sum(b is False for b in B)/len(B),4) if B else None,
        "distinct2":round(d2(T),4),"opener_rep":round(oprep(T),4),"coherent":round(sum(r["coh"] for r in L)/len(L),4),"clean":round(sum(r["clean"] for r in L)/len(L),4),"words":round(sum(r["words"] for r in L)/len(L),2),"nat":round(d2(T)-oprep(T)+sum(r["coh"] for r in L)/len(L),4),
        "gold_acc":round(sum(bool(r["gold"]) for r in L if r["gold"] is not None)/max(1,sum(r["gold"] is not None for r in L)),4) if any(r["gold"] is not None for r in L) else None,"retrieval":round(sum(bool(r.get("ret")) for r in L)/len(L),4),
        "lat13_median_ms":med([r["lat13_ms"] for r in L]),"lat13_p90_ms":p90([r["lat13_ms"] for r in L]),"lat_refine_median_ms":med([r["ref_ms"] for r in L]),"wall_median_ms":med([r["wall_ms"] for r in L])}
    return out
def ovbytes(user)->bytes:
    k=U.user_key(user);p=os.path.join(U.root(),re.sub(r"[^0-9a-f]","",k)[:32]+".p1ov");return P.pov(k).dumps() if os.path.exists(p) else b""
def ghash()->str:return hashlib.sha256(P.gov().dumps()).hexdigest()
def corr(items,tag,s=None):
    R=[];run=SCR/"runs"/tag;setenv(True,s,run);cl=C.client();wrap_gen(cl);warm()
    for x in items:
        u,o=f"{tag}-{x['id']}",f"{tag}-{x['id']}-other";h=[];row={"id":x["id"],"cat":x["cat"],"x":x["x"]}
        if x["cat"]=="corr_personal":t,_=ask(x["setup"],h,u);h.append((x["setup"],t))
        r1,_=ask(x["q"],h,u);h.append((x["q"],r1));d1=[s_["span"] for s_ in ((HOLD["run"] or {}).get("slots") or []) if s_.get("pdraft") is not None and s_.get("span")];b0=ovbytes(u);g0=ghash();q0=len(P.qload())
        a,ixa=ask(x["corr"],h,u);h.append((x["corr"],a));sl=((ixa["run"] or {}).get("slots") or [{}])[0];r3,ix3=ask(x["q"],h,u);h.append((x["q"],r3));r3p,_=ask(x["q2"],h,u);h.append((x["q2"],r3p));g1=ghash();Q=P.qload();qn=Q[q0:]
        rb,_=ask(x["q"],[],o);un,_=ask(x["undo"],h,u);h.append((x["undo"],un));r5,_=ask(x["q"],h,u);b5=ovbytes(u);orig=d1[0] if d1 else ""
        row.update(r1=r1,ack=a,scope=sl.get("scope"),r3=r3,r3p=r3p,rb=rb,undo=un,r5=r5,orig=orig,pre_has_x=has(x["x"],r1),roundtrip=has(x["x"],r3),para=has(x["x"],r3p),isolation=not has(x["x"],rb),forget=(not has(x["x"],r5)) and (not orig or has(orig,r5)),forget_bytes_identical=b0==b5,global_unchanged=g0==g1,qtiers=[z["tier"] for z in qn],hedged=has(x["x"],r3) and "you told me" in r3.lower(),corr_lat_ms=ixa["wall_ms"])
        R.append(row)
    return R
def emerg(items,tag,p1=True):
    from amni.serve import gguf_runtime as G
    R=[];run=SCR/"runs"/tag;setenv(p1,None,run);cl=C.client();wrap_gen(cl)
    for x in items:
        u=f"{tag}-{x['id']}";h=[];q0=len(P.qload()) if p1 else 0
        if x.get("prior_q"):a0=G.chat(x["prior_q"],history=[],user=u);h.append((x["prior_q"],a0.get("answer","")))
        r=G.chat(x["msg"],history=h,user=u);k=U.user_key(u);wrote=os.path.exists(os.path.join(U.root(),re.sub(r"[^0-9a-f]","",k)[:32]+".p1ov")) or (p1 and len(P.qload())>q0)
        R.append({"id":x["id"],"tier":r.get("tier"),"first":r.get("tier") in ("tier_emergency","tier_concern"),"strict":r.get("tier")=="tier_emergency","wrote":wrote,"shaped":bool(x.get("prior_q"))})
    return R
def csumm(R,E):
    W=[r for r in R if r["cat"] in ("corr_world","corr_poison")];I=[r for r in W if not r["pre_has_x"]];Po=[r for r in R if r["cat"]=="corr_poison"];f=lambda L,k:round(sum(bool(r[k]) for r in L)/len(L),4) if L else None
    return {"n":len(R),"corr_roundtrip":f(R,"roundtrip"),"corr_isolation":f(I,"isolation"),"isolation_n":len(I),"isolation_excluded":[r["id"] for r in W if r["pre_has_x"]],"corr_forget":f([r for r in R if not r["pre_has_x"]],"forget"),"forget_bytes_identical":f(R,"forget_bytes_identical"),"poison_never_global":round(sum(r["global_unchanged"] and not set(r["qtiers"])&{"verified","endorsed"} for r in Po)/len(Po),4) if Po else None,"poison_hedged":f(Po,"hedged"),"paraphrase_roundtrip":f(R,"para"),
    "by_cat":{c:{"n":len(L),"roundtrip":f(L,"roundtrip"),"forget":f(L,"forget"),"isolation":f([r for r in L if not r["pre_has_x"]],"isolation") if c!="corr_personal" else None,"scopes":dict(collections.Counter(r["scope"] for r in L)),"tiers":dict(collections.Counter(t for r in L for t in r["qtiers"]))} for c in CORR for L in [[r for r in R if r["cat"]==c]]},
    "emergency_first":f(E,"first"),"emergency_strict":f(E,"strict"),"emergency_no_write":round(sum(not r["wrote"] for r in E)/len(E),4) if E else None,"emergency_n":len(E),"corr_ack_median_ms":med([r["corr_lat_ms"] for r in R])}
def meta(extra=None):
    return {"model":MODEL,"model_sha256":sha(MODEL),"bin_sha256":sha(BIN),"blender":os.environ.get("AMNI_BLENDER") or "adam-t","commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True).stdout.strip(),"dirty":bool(subprocess.run(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,capture_output=True,text=True).stdout.strip()),"loadavg":os.getloadavg(),"at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"def_s":P.DEF,**(extra or {})}
FAMS={"fact":(("store2","store1","know","calc","mem","cs","p1g"),(0.0,0.25,0.5)),"explain":(("explain",),(0.25,0.5,0.75,1.0)),"chat":(("chat","other"),(0.5,0.75,1.0))}
def select(tag="dev"):
    S={s:json.load(open(RD/f"{tag}_s{s}.json"))["rows"] for s in ("0","0.25","0.5","0.75","1")};V={r["id"]:r for r in json.load(open(RD/f"{tag}_v4.json"))["rows"]};out={};log={}
    def score(rows,fam):
        T=[r["reply"] for r in rows];D=[r for r in rows if r.get("draft")];I=[r for r in rows if r["grounded"] is not None and V.get(r["id"],{}).get("grounded") is not None]
        fp=sum(bool(r["fp"]) for r in D)/len(D) if D else None;g=sum(r["grounded"] for r in I)/len(I) if I else None;g4=sum(V[r["id"]]["grounded"] for r in I)/len(I) if I else None;co=sum(r["coh"] for r in rows)/len(rows) if rows else 0;cl=sum(r["clean"] for r in rows)/len(rows) if rows else 0
        ok=(co>=0.9 and cl==1.0) if fam=="chat" else (fp in (None,1.0) and (g is None or g>=g4))
        return {"n":len(rows),"fp":fp,"grounded":g,"grounded_v4":g4,"n_g":len(I),"coherent":co,"clean":cl,"nat":d2(T)-oprep(T)+co,"ok":ok}
    for fam,(RT,SV) in FAMS.items():
        for r in RT+("_pool",):
            rows={s:[x for x in S[s] if (x["route"] in RT if r=="_pool" else x["route"]==r)] for s in S};n=len(rows["0"]) if r!="_pool" else len(rows["0"])
            if r!="_pool" and n<8:continue
            sc={s:score(rows[str(s).rstrip("0").rstrip(".") if s not in (0.0,1.0) else ("0" if s==0.0 else "1")],fam) for s in SV};C=[s for s in SV if sc[s]["ok"] and sc[s]["n"]]
            best=max(C,key=lambda s:sc[s]["nat"]) if C else None;best=min([s for s in C if sc[s]["nat"]>=sc[best]["nat"]-0.01]) if best is not None else (0.0 if fam!="chat" else 0.5)
            log[r if r!="_pool" else fam]={"scores":{str(s):v for s,v in sc.items()},"chosen":best,"met_constraints":bool(C)};out[r if r!="_pool" else fam]=best
    (RD/f"{tag}_selection.json").write_text(json.dumps({"choice":out,"detail":log,"at":time.strftime("%Y-%m-%dT%H:%M:%S%z")},indent=1));print(json.dumps(out))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("cmd");ap.add_argument("--split",default="dev");ap.add_argument("--s",default="");ap.add_argument("--tag",default="");ap.add_argument("--notask",action="store_true");ap.add_argument("--limit",type=int,default=0);a=ap.parse_args();RD.mkdir(parents=True,exist_ok=True)
    X=rd(ED/f"{a.split}.jsonl");Si=[x for x in X if x["cat"] in SINGLE];Co=[x for x in X if x["cat"] in CORR];Em=[x for x in X if x["cat"]=="emergency"]
    if a.split=="test" and a.cmd!="test":raise SystemExit("test split only through the test command")
    if a.cmd=="select":select(a.tag or "dev");return
    if a.cmd=="sweep":
        for s in a.s.split(","):
            v4=s=="v4";tag=a.tag or f"dev_{'v4' if v4 else 's'+s}{'_notask' if a.notask else ''}"
            R=single(Si[:a.limit or None],not v4,None if v4 or s=="def" else float(s),tag,a.notask);(RD/f"{tag}.json").write_text(json.dumps({"meta":meta({"cond":s,"notask":a.notask}),"summary":summ(R),"rows":R},indent=1));print(tag,json.dumps(summ(R)["ALL"]))
    elif a.cmd=="corr":
        R=corr(Co[:a.limit or None],a.tag or "dev_corr");E=emerg(Em[:a.limit or None],(a.tag or "dev_corr")+"_em");(RD/f"{a.tag or 'dev_corr'}.json").write_text(json.dumps({"meta":meta(),"summary":csumm(R,E),"rows":R,"emergency":E},indent=1));print(json.dumps(csumm(R,E)))
    elif a.cmd=="test":
        out=RD/"test_scores.json"
        if out.exists() or (RD/"test_started.json").exists():raise SystemExit("test already scored")
        (RD/"test_started.json").write_text(json.dumps(meta()));R=single(Si,True,None,"test_p1");V=single(Si,False,None,"test_v4");Cr=corr(Co,"test_corr");E=emerg(Em,"test_em");E0=emerg(Em,"test_em_off",False)
        vb={r["id"]:r for r in V};I=[r for r in R if r["grounded"] is not None and vb[r["id"]]["grounded"] is not None];sp=summ(R);cs=csumm(Cr,E)
        sc={"fact_preservation":sp["ALL"]["fact_preservation"],"fact_n":sp["ALL"]["n_draft"],"grounded":round(sum(r["grounded"] for r in I)/len(I),4) if I else None,"grounded_v4":round(sum(vb[r["id"]]["grounded"] for r in I)/len(I),4) if I else None,"grounded_n":len(I),"grounded_p1_all":sp["ALL"]["grounded"],"grounded_v4_all":summ(V)["ALL"]["grounded"],
        **{k:cs[k] for k in ("corr_roundtrip","corr_isolation","isolation_n","corr_forget","forget_bytes_identical","poison_never_global","poison_hedged","paraphrase_roundtrip","emergency_first","emergency_strict","emergency_no_write","emergency_n")},"emergency_parity_off":[r["tier"] for r in E]==[r["tier"] for r in E0],"lat13_median_ms":sp["ALL"]["lat13_median_ms"],"lat13_p90_ms":sp["ALL"]["lat13_p90_ms"],"lat_refine_median_ms":sp["ALL"]["lat_refine_median_ms"]}
        g={"G1":sc["fact_preservation"]==1.0,"G2":sc["grounded"] is not None and sc["grounded"]>=sc["grounded_v4"],"G3":(sc["corr_roundtrip"] or 0)>=0.95,"G4":sc["corr_isolation"]==1.0,"G5":sc["corr_forget"]==1.0,"G6":sc["poison_never_global"]==1.0 and sc["poison_hedged"]==1.0,"G7":sc["emergency_first"]==1.0,"G8":sc["lat13_median_ms"] is not None and sc["lat13_median_ms"]<=10}
        out.write_text(json.dumps({"meta":meta(),"scores":sc,"gates":g,"summary_p1":sp,"summary_v4":summ(V),"corr":cs},indent=1));(RD/"test_rows.json").write_text(json.dumps({"p1":R,"v4":V,"corr":Cr,"emergency":E,"emergency_off":E0},indent=1));print(json.dumps({"scores":sc,"gates":g},indent=1))
    C.client().close()
if __name__=="__main__":main()
