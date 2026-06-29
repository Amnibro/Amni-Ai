"""Amni-Delve adapter registry: declarative SPECS that drive any common agentic CLI headless (or Adam in-process). SECURITY (Anthony's rule): API keys are NEVER read into config or returned over the wire. Child CLIs inherit os.environ directly — their own subscription login or a *_API_KEY already in Anthony's env reaches the tool without Delve ever storing it. auth_status reports only which env-var NAMES are present, never their values. Add a platform = one dict; an absent CLI is installed:false, never an error (strictly additive)."""
import os,shutil,subprocess,tempfile
SPECS=[
{"key":"claude","name":"Claude","color":"#a78bfa","kind":"cli","exec":"claude","fallbacks":["~/.local/bin/claude.exe"],"prompt_via":"stdin","base":["-p","--output-format","text"],"cont":["--continue"],"model_flag":"--model","bypass":["--permission-mode","bypassPermissions"],"auth":"subscription","env_keys":["ANTHROPIC_API_KEY"]},
{"key":"grok","name":"Grok","color":"#fbbf24","kind":"cli","exec":"grok","fallbacks":["~/.grok/bin/grok.exe"],"prompt_via":"file","file_flag":"--prompt-file","base":["--output-format","plain"],"cont":["-c"],"model_flag":"--model","bypass":["--permission-mode","bypassPermissions"],"auth":"subscription","env_keys":["XAI_API_KEY","GROK_API_KEY"]},
{"key":"gemini","name":"Gemini","color":"#4f8cf7","kind":"cli","exec":"gemini","fallbacks":["~/AppData/Roaming/npm/gemini.cmd"],"prompt_via":"arg","prompt_flag":"-p","base":[],"cont":[],"model_flag":"-m","bypass":["-y"],"auth":"subscription","env_keys":["GEMINI_API_KEY","GOOGLE_API_KEY"]},
{"key":"codex","name":"Codex","color":"#10a37f","kind":"cli","exec":"codex","fallbacks":[],"prompt_via":"arg","subcmd":["exec"],"base":["--skip-git-repo-check"],"cont":[],"model_flag":"-m","bypass":["--dangerously-bypass-approvals-and-sandbox"],"auth":"subscription","env_keys":["OPENAI_API_KEY"]},
{"key":"aider","name":"Aider","color":"#e879a6","kind":"cli","exec":"aider","fallbacks":[],"prompt_via":"arg","prompt_flag":"--message","base":["--yes-always","--no-pretty","--no-auto-commits","--no-stream"],"cont":[],"model_flag":"--model","bypass":[],"auth":"apikey","env_keys":["OPENAI_API_KEY","ANTHROPIC_API_KEY","GEMINI_API_KEY","DEEPSEEK_API_KEY"]},
{"key":"cursor","name":"Cursor","color":"#9aa3b2","kind":"cli","exec":"cursor-agent","fallbacks":[],"prompt_via":"arg","prompt_flag":"-p","base":[],"cont":[],"model_flag":"-m","bypass":["--force"],"auth":"subscription","env_keys":["CURSOR_API_KEY"]},
{"key":"ollama","name":"Ollama","color":"#c0c4cc","kind":"cli","exec":"ollama","fallbacks":["~/AppData/Local/Programs/Ollama/ollama.exe"],"prompt_via":"stdin","subcmd":["run"],"needs_model":True,"default_model":"llama3.2","base":[],"cont":[],"auth":"local","env_keys":[]},
{"key":"adam","name":"Adam","color":"#2dd4bf","kind":"inproc","auth":"in-process","env_keys":[]},
]
BY_KEY={s["key"]:s for s in SPECS}
BY_NAME={s["name"]:s for s in SPECS}
def get(key):return BY_KEY.get(key)
def spec_for(name):return BY_NAME.get(name) or BY_KEY.get(str(name).lower())
def resolve(spec):
    p=shutil.which(spec.get("exec","")) if spec.get("kind")=="cli" else None
    if p:return p
    for fb in spec.get("fallbacks",[]):
        fb=os.path.expanduser(fb)
        if os.path.exists(fb):return fb
    return None
def auth_status(spec):
    present=[k for k in spec.get("env_keys",[]) if os.environ.get(k)]
    sub=spec.get("auth") in ("subscription","local","in-process")
    return {"mode":spec.get("auth"),"keys_present":present,"ready":bool(sub or present)}
def detect(enabled=None,models=None):
    models=models or {};out=[]
    for spec in SPECS:
        inproc=spec.get("kind")=="inproc";exe=None if inproc else resolve(spec);installed=inproc or bool(exe)
        out.append({"key":spec["key"],"name":spec["name"],"color":spec["color"],"kind":spec["kind"],"installed":installed,"exec":"in-process" if inproc else (exe or spec.get("exec","")),"needs_model":bool(spec.get("needs_model")),"default_model":spec.get("default_model",""),"model":models.get(spec["key"],""),"auth":auth_status(spec),"enabled":(spec["key"] in enabled) if enabled is not None else installed})
    return out
def build_cmd(spec,exe,prompt,cont=False,model="",bypass=True):
    args=list(spec.get("subcmd",[]))
    if spec.get("needs_model"):args+=[model or spec.get("default_model") or "llama3.2"]
    args+=list(spec.get("base",[]))
    if bypass:args+=list(spec.get("bypass",[]))
    if model and spec.get("model_flag") and not spec.get("needs_model"):args+=[spec["model_flag"],model]
    if cont:args+=list(spec.get("cont",[]))
    inp=None;tmp=None;via=spec.get("prompt_via")
    if via=="stdin":inp=prompt
    elif via=="file":
        f=tempfile.NamedTemporaryFile("w",suffix=".txt",delete=False,encoding="utf-8");f.write(prompt);f.close();tmp=f.name;args+=[spec["file_flag"],tmp]
    else:args+=([spec["prompt_flag"]] if spec.get("prompt_flag") else [])+[prompt]
    return [exe]+args,inp,tmp
def run(spec,prompt,cont=False,model="",bypass=True,cwd=None,timeout=600,abort=None,proc_sink=None):
    if spec.get("kind")=="inproc":return "[%s is in-process — call via adam_fn]"%spec["name"],-1
    exe=resolve(spec)
    if not exe:return "[%s not installed]"%spec["name"],-1
    cmd,inp,tmp=build_cmd(spec,exe,prompt,cont,model,bypass)
    out=err="";rc=-1
    try:
        p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=cwd,text=True,encoding="utf-8",errors="replace")
        if proc_sink:proc_sink(p)
        out,err=p.communicate(input=inp,timeout=timeout);rc=p.returncode
    except subprocess.TimeoutExpired:
        try:p.kill()
        except Exception:pass
        out,err,rc="","[%s timeout after %ss]"%(spec["key"],timeout),-1
    except Exception as e:out,err,rc="","[exec error] "+str(e),-1
    finally:
        if tmp:
            try:os.unlink(tmp)
            except Exception:pass
    aborted=abort is not None and abort.is_set()
    return (out or "").strip() or ("[%s stopped]"%spec["key"] if aborted else "[%s error] %s"%(spec["key"],(err or "no output").strip())),rc
