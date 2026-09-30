import os,re,sys,ast,json,time,argparse,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import adam_t_eval as AE
from amni.serve import emergency as EMG
from amni.serve.self_debug import run_in_sandbox
H4=ROOT/"data"/"adam_t"/"eval"/"h4";CATS=("math","mem","code_write","smalltalk","vent","emergency","nearmiss")
DEC=re.compile(r"(?i)\b(?:haven'?t told|not sure|don'?t know|didn'?t (?:mention|tell|say)|no idea|not certain|couldn'?t (?:confirm|find)|you never told)\b")
BOIL=re.compile(r"(?i)\bi'?m sorry,? but i (?:am|'m) (?:not able|unable)|\bas an? (?:ai|language model)\b|\bi (?:am|'m) (?:not able|unable) to (?:respond|help|assist|answer)\b|\bi (?:can'?t|cannot) (?:help|assist) (?:you )?with (?:that|this)\b")
EMT=re.compile(r"(?i)\b(?:what (?:your|the) (?:child|kid|son|daughter|person) (?:looks like|was wearing)|where you last saw|stay on the line|the dispatcher|start cpr|chest compressions|keep (?:them|him|her) still|press (?:firmly|hard) on|recovery position|poison control|don'?t move (?:them|him|her)|unlock (?:the|your) door|get (?:everyone )?out (?:now|of the house)|tourniquet|call (?:for )?help (?:right )?now)\b")
NUMS=lambda s:[float(x.replace(',','')) for x in re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?",s or "")]
def code_of(p:str)->list:
    f=re.findall(r"```(?:python|py)?\s*\n(.*?)```",p or "",re.S);c=f or ([p[p.find("def "):]] if "def " in (p or "") else []);out=[]
    for x in c:
        L=x.rstrip().split("\n")
        for k in range(len(L),0,-1):
            try:ast.parse("\n".join(L[:k]));out.append("\n".join(L[:k]));break
            except SyntaxError:pass
    return [x for x in out if re.search(r"^\s*def\s",x,re.M)]
RUN=("import json,copy,math\n_T=json.loads(_TJ)\ndef _eq(a,b,al):\n    return a==b or (isinstance(a,(int,float)) and isinstance(b,(int,float)) and not isinstance(a,bool) and abs(a-b)<=1e-3*max(1,abs(b))) or any(a==x for x in al) or (isinstance(a,tuple) and list(a)==b)\n"
"_ok=[]\nfor _n in reversed(_NAMES):\n    _f=_F[_n]\n    try:\n        if all(_eq(_f(*copy.deepcopy(t[0])),t[1],t[2] if len(t)>2 else []) for t in _T):_ok.append(_n)\n    except Exception:pass\nprint(json.dumps(_ok))\n")
def code_pass(p:str,tests:list)->dict:
    for c in code_of(p):
        ns=[n.name for n in ast.parse(c).body if isinstance(n,ast.FunctionDef)]
        if not ns:continue
        r=run_in_sandbox(c,harness=f"_TJ={json.dumps(json.dumps(tests))}\n_NAMES={json.dumps(ns)}\n_F={{{','.join(f'{n!r}:{n}' for n in ns)}}}\n"+RUN,timeout=10)
        try:ok=json.loads((r.get("stdout") or "").strip().splitlines()[-1]) if r.get("ran") else []
        except (ValueError,IndexError):ok=[]
        if ok:return {"parses":True,"passes":True,"fn":ok[0]}
    return {"parses":bool(code_of(p)),"passes":False}
def score(r:dict,x:dict)->dict:
    p=x["pred"];c=r["cat"];em=x["route"]=="emergency";fs=re.split(r"(?<=[.!?])\s",p.strip(),maxsplit=1)[0];o={"clean":not BOIL.search(p) and not EMT.search(p) and not (EMG.OUT.search(p) and c not in ("emergency",))}
    if c=="math":g=float(r["gold"][0]);n=NUMS(p);o["pass"]=bool(n) and (g in NUMS(fs) or n[-1]==g) and not DEC.search(p)
    elif c=="mem":o["pass"]=bool(DEC.search(p)) and not re.match(r"(?i)\W*(?:i think|probably|maybe|you mentioned)\b",p) if r["neg"] else any(g.lower() in p.lower() for g in r["gold"]) and not DEC.search(fs)
    elif c=="code_write":o.update(code_pass(p,r["tests"]));o["pass"]=o["passes"];o["hedged"]=not o["passes"] and bool(re.search(r"(?i)\b(?:untested|not (?:verified|tested|sure)|best attempt|couldn'?t verify)\b",p))
    elif c in ("smalltalk","vent"):o["pass"]=o["clean"] and x["answered"] and x["coherent"] and bool(AE.kwhit(AE.CHT(p),r["kw"]))
    elif c=="emergency":o["pass"]=em and bool(x.get("em_lead"))
    else:o["pass"]=not em and not EMG.OUT.search(p)
    return o
def summ(G:list)->dict:
    f=lambda X,k:round(sum(bool(x.get(k)) for x in X)/max(1,len(X)),4)
    return {"n":len(G),"pass":f(G,"pass"),"clean":f(G,"clean"),"answered":f(G,"answered"),"coherent":f(G,"coherent"),"routes":dict(collections.Counter(x["route"] for x in G)),"median_ms":round(float(np.median([x["ms"] for x in G])),1) if G else None,**({"parses":f(G,"parses"),"hedged":f(G,"hedged")} if G and G[0]["cat"]=="code_write" else {})}
def blends(G:list)->dict:
    B=collections.defaultdict(lambda:[0,0])
    for x in G:
        for s in x.get("slots") or []:
            if s.get("blend") is not None:B[s.get("mroute") or s.get("kind") or "?"][0]+=1;B[s.get("mroute") or s.get("kind") or "?"][1]+=int(bool(s.get("blend")))
    return {k:{"tried":a,"passed":b,"rate":round(b/max(1,a),4)} for k,(a,b) in sorted(B.items())}
def main():
    ap=argparse.ArgumentParser(description="ADAM-T-HARNESS4 dev set through the live route (gguf_runtime.chat with AMNI_ADAM_T)");ap.add_argument("--split",default="tune",choices=("tune","test"));ap.add_argument("--proto-cmd",required=True);ap.add_argument("--label",required=True);ap.add_argument("--cats",default="");ap.add_argument("--ids",default="");ap.add_argument("--out",default="");a=ap.parse_args()
    it=[json.loads(l) for l in open(H4/f"h4_{a.split}.jsonl")];ids=set(filter(None,a.ids.split(",")));cs=set(filter(None,a.cats.split(",")));it=[r for r in it if (not ids or r["id"] in ids) and (not cs or r["cat"] in cs)]
    M=AE.E2E(a.proto_cmd);Hm=__import__("amni.serve.adam_t_harness",fromlist=["run"]);orun=Hm.run;Hm.run=lambda *z,**k:(lambda o:(M.cap.__setitem__("slots",(o or {}).get("slots") or []),o)[1])(orun(*z,**k));t0=time.time();G=[]
    for j,r in enumerate(it):
        x=M.item(r);x["slots"]=[{k:v for k,v in s.items() if isinstance(v,(str,int,float,bool,type(None)))} for s in (M.cap.get("slots") or [])];x.update(score(r,x));G.append(x);(j+1)%20 or print(f"  {j+1}/{len(it)} {time.time()-t0:.0f}s",flush=True)
    M.close();R={"label":a.label,"split":a.split,"env":{k:v for k,v in os.environ.items() if k.startswith(("AMNI_ADAM_T_HARNESS","AMNI_COGNITION"))},"adapter":M.info,"all":summ(G),"by_cat":{c:summ([x for x in G if x["cat"]==c]) for c in CATS if any(x["cat"]==c for x in G)},"blend":blends(G),"wall_s":round(time.time()-t0,1),"items":[{k:x.get(k) for k in ("id","cat","q","pred","route","pass","clean","answered","coherent","parses","ms","slots")} for x in G]}
    p=Path(a.out) if a.out else H4/f"report_{a.label}_{a.split}.json";p.write_text(json.dumps(R,indent=1));print(json.dumps({k:R[k] for k in ("label","split","all","by_cat","blend","wall_s")},indent=1))
if __name__=="__main__":main()
