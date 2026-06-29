"""Amni-Delve engine, generalized. The roster is whatever's enabled+installed in the adapter registry (not hardcoded claude/grok). `all` fans to the whole roster; `both`/`debate` use a configurable pair. Adam is driven in-process via adam_fn (no HTTP). One shared transcript, delta-routing per agent (idx[]), E-STOP, interjection, PTEX capture. sink emits {type:...} events; the FastAPI layer fans them over SSE. scrub() runs on every stored/emitted line so a pasted key never lands on disk or the wire."""
import os,threading,datetime
from amni.delve import adapters
class Hub:
    def __init__(s,sink=None,adam_fn=None,cfg=None,scrub=None,work=None,sess=None):
        s.sink=sink or (lambda ev:None);s.adam_fn=adam_fn;s.cfg=cfg or {};s.scrub=scrub or (lambda x:x)
        s.work=work or os.getcwd();s.sess=sess or os.path.join(os.getcwd(),"experiences","delve","sessions")
        os.makedirs(s.work,exist_ok=True);os.makedirs(s.sess,exist_ok=True)
        s.t=[];s.idx={};s.started={};s.stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        s.file=os.path.join(s.sess,"session_"+s.stamp+".md")
        s.abort=threading.Event();s.proc=None;s.interjects=[]
    def available(s):
        en=s.cfg.get("enabled");rows=adapters.detect(enabled=en,models=s.cfg.get("models",{}))
        out=[]
        for r in rows:
            if not r["installed"]:continue
            if r["kind"]=="inproc" and s.adam_fn is None:continue
            if en is not None and r["key"] not in en:continue
            out.append(r)
        return out
    def roster(s):return [r["name"] for r in s.available()]
    def pair(s):
        p=s.cfg.get("pair");names=s.roster()
        if p:sel=[adapters.get(k)["name"] for k in p if adapters.get(k) and adapters.get(k)["name"] in names]
        else:sel=[n for n in names if n!="Adam"][:2] or names[:2]
        return sel[:2]
    def emit(s,ev):
        if "text" in ev and isinstance(ev["text"],str):ev=dict(ev,text=s.scrub(ev["text"]))
        try:s.sink(ev)
        except Exception:pass
    def add(s,who,text):
        text=s.scrub(text or "");s.t.append((who,text));s.flush();return text
    def flush(s):
        try:open(s.file,"w",encoding="utf-8").write("# Amni-Delve session "+s.stamp+"\n\n"+"\n\n".join("**"+w+":** "+x for w,x in s.t)+"\n")
        except Exception:pass
    def render(s,start,name,extra=""):
        head="Conversation so far:" if start==0 else "New messages since your last turn:"
        body="\n".join(w+": "+x for w,x in s.t[start:]) or "(nothing new)"
        parts=", ".join([n for n in s.roster() if n!=name] or ["the others"])
        sysmsg=("You are "+name+" in a live group chat. Participants: Anthony (the human), "+parts+", and you. Read the messages below and reply ONLY as "+name+", in first person, conversationally and concisely. Do not prefix your reply with your own name. You may use your tools in the working directory only when a message asks for real work.")
        tail=("\n\n"+extra) if extra else ""
        return sysmsg+"\n\n"+head+"\n"+body+tail+"\n\nReply now as "+name+":"
    def _call(s,name,prompt,cont):
        spec=adapters.spec_for(name)
        if spec is None:return "[unknown agent "+str(name)+"]"
        if spec["kind"]=="inproc":
            try:return (s.adam_fn(prompt) or "").strip() or "[adam empty]"
            except Exception as e:return "[adam error] "+str(e)
        key=spec["key"];model=(s.cfg.get("models",{}) or {}).get(key,"") or ""
        out,_rc=adapters.run(spec,prompt,cont=cont,model=model,bypass=s.cfg.get("bypass",True),cwd=s.work,abort=s.abort,proc_sink=lambda p:setattr(s,"proc",p))
        s.proc=None;return out
    def _is_fail(s,r):return (not r) or (r.startswith("[") and ("error" in r.lower() or "not installed" in r.lower() or "timeout" in r.lower() or "stopped" in r.lower()))
    def _adam_ok(s):return s.adam_fn is not None
    def turn(s,name,start=None,extra="",record=True):
        inproc=adapters.spec_for(name) and adapters.spec_for(name)["kind"]=="inproc"
        st=0 if inproc else (s.idx.get(name,0) if start is None else start)
        s.emit({"type":"thinking","who":name})
        reply=s._call(name,s.render(st,name,extra),s.started.get(name,False));speaker=name
        if (not inproc) and s._is_fail(reply) and not s.abort.is_set() and s.cfg.get("adam_fallback",True) and s._adam_ok():
            s.emit({"type":"status","text":name+" unavailable — Adam stepping in"})
            a=s._call("Adam",s.render(0,"Adam"),False)
            if a and not a.startswith("[adam"):speaker="Adam";reply=a
        s.started[speaker]=True
        if record:reply=s.add(speaker,reply);s.idx[speaker]=len(s.t)
        s.emit({"type":"msg","who":speaker,"text":reply});return speaker,reply
    def _estop(s):
        s.abort.set();p=s.proc
        if p is not None:
            try:p.kill()
            except Exception:pass
        s.emit({"type":"estop"})
    def _drain(s):
        msgs=s.interjects;s.interjects=[]
        for m in msgs:s.add("Anthony",m);s.emit({"type":"user","who":"Anthony","text":m})
        return bool(msgs)
    def maybe_learn(s,question,items,kind):
        if not s.cfg.get("ptex_learn",True):return
        good=[(w,t) for w,t in items if w!="Adam" and t and not(t.startswith("[") and any(k in t.lower() for k in("stopped","error","skipped","not installed","timeout")))]
        if len({w for w,_ in good})<2:return
        try:
            from amni.delve import ptex as P;r=P.feed(question,good,s.stamp,kind)
        except Exception as e:r={"ok":False,"error":str(e)}
        s.emit({"type":"ptex","data":r})
    def route(s,target,msg):
        s.add("Anthony",msg);s.emit({"type":"user","who":"Anthony","text":msg})
        names=s.roster() if target=="all" else [adapters.spec_for(target)["name"]] if adapters.spec_for(target) else []
        s.abort.clear();s.interjects=[];items=[]
        for n in names:
            if s.abort.is_set():break
            sp,rp=s.turn(n);items.append((sp,rp));s._drain()
        if target=="all":s.maybe_learn(msg,items,"exchange")
    def both(s,msg):
        s.add("Anthony",msg);s.emit({"type":"user","who":"Anthony","text":msg});pr=s.pair()
        if len(pr)<2:s.emit({"type":"status","text":"need 2 agents for both-mode"});return s.route("all",msg) if False else None
        a,b=pr[0],pr[1];sa=s.idx.get(a,0);sb=s.idx.get(b,0);s.abort.clear();s.interjects=[];items=[]
        s.emit({"type":"status","text":"independent answers"})
        wa,ra=s.turn(a,start=sa,record=False);ra=s.add(wa,ra);items.append((wa,ra))
        if not s.abort.is_set():
            wb,rb=s.turn(b,start=sb,record=False);rb=s.add(wb,rb);items.append((wb,rb))
        s._drain()
        if not s.abort.is_set():
            s.emit({"type":"status","text":"peer review"})
            rv=lambda other:other+" answered the SAME question independently (shown above as "+other+"). Compare "+other+"'s approach with yours — call out strengths, gaps, or mistakes on either side — then refine or defend your own answer."
            wa2,ra2=s.turn(a,start=sa,extra=rv(b),record=False);ra2=s.add(wa2,ra2);items.append((wa2,ra2))
            if not s.abort.is_set():
                wb2,rb2=s.turn(b,start=sb,extra=rv(a),record=False);rb2=s.add(wb2,rb2);items.append((wb2,rb2))
        s.idx[a]=len(s.t);s.idx[b]=len(s.t);s.maybe_learn(msg,items,"both")
    def debate(s,rounds,topic):
        if topic:s.add("Anthony",topic);s.emit({"type":"user","who":"Anthony","text":topic})
        q=topic or next((x for w,x in reversed(s.t) if w=="Anthony"),"[Amni-Delve debate]")
        names=[n for n in s.roster() if n!="Adam"] or s.roster();s.abort.clear();s.interjects=[];items=[]
        for i in range(rounds):
            if s.abort.is_set():break
            s.emit({"type":"status","text":"round "+str(i+1)+"/"+str(rounds)})
            for n in names:
                if s.abort.is_set():break
                sp,rp=s.turn(n);items.append((sp,rp));s._drain()
        s.maybe_learn(q,items,"debate")
