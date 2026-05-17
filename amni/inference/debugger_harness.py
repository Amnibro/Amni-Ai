import os,sys,re,json,time,shutil,tempfile,hashlib,subprocess,threading,queue
from pathlib import Path
from amni.storage.ptex_memory import PtexMemoryAtlas
_PROMPT='(Pdb) '
_DBG_RE=re.compile(r'<dbg>([\s\S]*?)</dbg>')
_PATCH_RE=re.compile(r'<patch[^>]*>([\s\S]*?)</patch>')
_FILE_RE=re.compile(r'<file>([^<]+)</file>')
_FIND_RE=re.compile(r'<find>([\s\S]*?)</find>')
_REPLACE_RE=re.compile(r'<replace>([\s\S]*?)</replace>')
_GIVEUP_RE=re.compile(r'<give-up\s*/?>')
_ALLOWED={'step','s','next','n','continue','c','cont','break','b','clear','cl','print','p','pp','where','w','bt','list','l','quit','q','exit','args','a','up','u','down','d','return','r','tbreak','condition','until','unt','disable','enable','run','restart','jump','j','retval','rv','longlist','ll','source','display','undisplay','whatis'}
_HARNESS_VERSION='v5.2.0'
_MAX_STDOUT=262144
class _Reader:
    def __init__(self,stream):
        self.stream=stream;self.q=queue.Queue();self.alive=True
        self.t=threading.Thread(target=self._run,daemon=True);self.t.start()
    def _run(self):
        try:
            while self.alive:
                ch=self.stream.read(1)
                if not ch:break
                self.q.put(ch)
        except Exception:pass
    def read_until(self,marker,timeout):
        buf='';deadline=time.time()+timeout
        while time.time()<deadline:
            try:ch=self.q.get(timeout=0.05)
            except queue.Empty:
                if buf.endswith(marker):return buf[:-len(marker)],True
                continue
            buf+=ch
            if buf.endswith(marker):return buf[:-len(marker)],True
            if len(buf)>_MAX_STDOUT:return buf[-_MAX_STDOUT:]+'<truncated>',False
        return buf,False
    def stop(self):self.alive=False
class DebuggerSession:
    def __init__(self,script_path,cwd=None,timeout_per_cmd=5.0,boot_timeout=10.0):
        env=dict(os.environ);env['PYTHONUNBUFFERED']='1';env['PYTHONIOENCODING']='utf-8'
        self.proc=subprocess.Popen([sys.executable,'-u','-m','pdb',str(script_path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0,cwd=str(cwd) if cwd else None,env=env,text=True,encoding='utf-8',errors='replace')
        self.timeout=timeout_per_cmd
        self.reader=_Reader(self.proc.stdout)
        intro,ok=self.reader.read_until(_PROMPT,timeout=boot_timeout)
        self.intro=intro;self.alive=ok
    def cmd(self,line):
        if not self.alive or self.proc.poll() is not None:return '<session-dead>',False
        verb=(line.strip().split(' ',1)[0] or '').lower()
        if verb not in _ALLOWED:return f'unknown command: {verb}',False
        try:
            self.proc.stdin.write(line+'\n');self.proc.stdin.flush()
        except Exception as e:
            self.alive=False;return f'<write-failed:{e}>',False
        out,ok=self.reader.read_until(_PROMPT,timeout=self.timeout)
        if self.proc.poll() is not None:
            self.alive=False
            tail,_=self.reader.read_until(_PROMPT,timeout=0.2)
            return (out+tail+'\n<session-ended>'),False
        return out,ok
    def close(self):
        if self.alive and self.proc.poll() is None:
            try:
                self.proc.stdin.write('quit\n');self.proc.stdin.flush()
            except Exception:pass
        try:self.proc.wait(timeout=2)
        except Exception:
            try:self.proc.terminate();self.proc.wait(timeout=1)
            except Exception:
                try:self.proc.kill()
                except Exception:pass
        self.reader.stop();self.alive=False
class PatchApplier:
    @staticmethod
    def apply(sandbox_dir,patch_block_text):
        f=_FILE_RE.search(patch_block_text);fi=_FIND_RE.search(patch_block_text);re_=_REPLACE_RE.search(patch_block_text)
        if not (f and fi and re_):return False,'malformed patch (need <file>, <find>, <replace>)'
        rel=f.group(1).strip();find=fi.group(1);rep=re_.group(1)
        target=Path(sandbox_dir)/rel
        if not target.exists():return False,f'file not in sandbox: {rel}'
        body=target.read_text(encoding='utf-8')
        if find not in body:return False,f'find block not found in {rel} (chars={len(find)})'
        new=body.replace(find,rep,1)
        target.write_text(new,encoding='utf-8')
        return True,f'applied to {rel} ({len(find)}->{len(rep)} chars)'
class TrajectoryRecorder:
    def __init__(self,atlas_path):self.atlas=PtexMemoryAtlas(atlas_path)
    def record(self,bug_id,bug_sha,model_id,condition,solved,turns,wall_seconds,turn_log,final_patch_text=None,extra_meta=None):
        body=json.dumps({'bug_id':bug_id,'turns':turns,'turn_log':turn_log,'final_patch':final_patch_text},separators=(',',':'),ensure_ascii=False)
        patch_sha=hashlib.sha256((final_patch_text or '').encode('utf-8')).hexdigest()[:16] if final_patch_text else None
        tool_counts={}
        for t in turn_log:
            for c in t.get('dbg_cmds',[]):
                v=(c.split(' ',1)[0] or '').lower()
                if v:tool_counts[v]=tool_counts.get(v,0)+1
        meta={'subject':'debugger_run','bug_id':bug_id,'bug_sha':bug_sha,'model_id':model_id,'condition':condition,'solved':bool(solved),'turns':int(turns),'wall_seconds':float(wall_seconds),'tool_call_counts':tool_counts,'final_patch_sha':patch_sha,'harness_version':_HARNESS_VERSION}
        if extra_meta:meta.update(extra_meta)
        return self.atlas.append(body,meta=meta)
    def stats(self):return self.atlas.stats()
class DebuggerLoop:
    SYS_WITH_DBG=("You are a coding assistant fixing a Python bug. You have a pdb debugger.\n"
                  "Issue zero or more commands per turn as <dbg>cmd</dbg>. Allowed: step, next, continue, break <loc>, "
                  "clear <loc>, print <expr>, pp <expr>, where, list, args, up, down, return, quit.\n"
                  "When confident, emit a single patch:\n"
                  "<patch><file>RELATIVE.py</file><find>EXACT BUGGY CODE</find><replace>FIXED CODE</replace></patch>\n"
                  "<find> must match the file byte-for-byte; only first match is replaced. After a patch the test is re-run; "
                  "if it still fails you continue. Emit <give-up/> if stuck. Be terse.")
    SYS_NO_DBG=("You are a coding assistant fixing a Python bug. You see source and failing output. No debugger.\n"
                "Emit a patch when ready:\n"
                "<patch><file>RELATIVE.py</file><find>EXACT BUGGY CODE</find><replace>FIXED CODE</replace></patch>\n"
                "Emit <give-up/> if stuck. Be terse.")
    def __init__(self,chat_callable,recorder,bug_id,bug_sha,model_id,condition,sandbox_dir,fixture_file,repro_cmd,max_turns=30,wall_budget=120.0,session_timeout_per_cmd=5.0):
        assert condition in ('no_debugger','with_debugger'),f'bad condition: {condition}'
        self.chat=chat_callable;self.recorder=recorder;self.bug_id=bug_id;self.bug_sha=bug_sha;self.model_id=model_id;self.condition=condition
        self.sandbox=Path(sandbox_dir);self.fixture=fixture_file;self.repro_cmd=repro_cmd
        self.max_turns=max_turns;self.wall_budget=wall_budget;self.cmd_timeout=session_timeout_per_cmd
    def _run_repro(self):
        try:
            r=subprocess.run(self.repro_cmd,cwd=str(self.sandbox),capture_output=True,text=True,timeout=15)
            return r.returncode==0,(r.stdout or '')+(r.stderr or '')
        except subprocess.TimeoutExpired:return False,'<repro-timeout>'
        except Exception as e:return False,f'<repro-error:{e}>'
    def _sys_prompt(self):return self.SYS_WITH_DBG if self.condition=='with_debugger' else self.SYS_NO_DBG
    def _format_observation(self,passed,output):
        src=(self.sandbox/self.fixture).read_text(encoding='utf-8')[:4000]
        return f"Test {'PASSED' if passed else 'FAILED'}.\nOutput:\n{output[-2000:]}\n\nFixture file: {self.fixture}\nFile contents:\n{src}"
    def run(self):
        t0=time.time();turn_log=[];final_patch=None;solved=False
        passed,obs=self._run_repro()
        if passed:
            tid=self.recorder.record(self.bug_id,self.bug_sha,self.model_id,self.condition,True,0,0.0,[])
            return {'solved':True,'turns':0,'turn_log':[],'trajectory_id':tid,'wall_seconds':0.0}
        observation=self._format_observation(passed,obs)
        sess=None
        if self.condition=='with_debugger':
            try:sess=DebuggerSession(self.sandbox/self.fixture,cwd=self.sandbox,timeout_per_cmd=self.cmd_timeout)
            except Exception as e:observation+=f'\n<pdb-spawn-failed:{e}>'
        try:
            for turn in range(1,self.max_turns+1):
                if time.time()-t0>self.wall_budget:break
                user=f"Turn {turn}/{self.max_turns}.\n\n{observation}"
                try:reply=self.chat(self._sys_prompt(),user)
                except Exception as e:
                    turn_log.append({'turn':turn,'reply':f'<chat-failed:{e}>','dbg_cmds':[],'patch_attempted':False})
                    break
                dbg_cmds=[m.strip() for m in _DBG_RE.findall(reply)]
                patch_m=_PATCH_RE.search(reply)
                gave_up=bool(_GIVEUP_RE.search(reply))
                outs=[]
                if sess is not None and sess.alive:
                    for c in dbg_cmds:
                        out,ok=sess.cmd(c)
                        outs.append({'cmd':c,'ok':ok,'out':out[-2048:]})
                turn_entry={'turn':turn,'reply':reply[-2048:],'dbg_cmds':dbg_cmds,'dbg_outs':outs,'patch_attempted':bool(patch_m)}
                patch_applied=False
                if patch_m:
                    ok,msg=PatchApplier.apply(self.sandbox,patch_m.group(0))
                    turn_entry['patch_apply_ok']=ok;turn_entry['patch_apply_msg']=msg
                    if ok:
                        patch_applied=True
                        if sess is not None:
                            try:sess.close()
                            except Exception:pass
                            sess=None
                        passed,r_out=self._run_repro()
                        observation=self._format_observation(passed,r_out)
                        turn_entry['post_patch_passed']=passed
                        if passed:
                            final_patch=patch_m.group(0);solved=True;turn_log.append(turn_entry);break
                        if self.condition=='with_debugger':
                            try:sess=DebuggerSession(self.sandbox/self.fixture,cwd=self.sandbox,timeout_per_cmd=self.cmd_timeout)
                            except Exception as e:observation+=f'\n<pdb-respawn-failed:{e}>'
                    else:
                        observation=observation[:6000]+f'\nPatch rejected: {msg}'
                if not patch_applied and outs:
                    obs_tail='\n'.join(f'<dbg-out cmd={json.dumps(o["cmd"])}>{o["out"][-1024:]}</dbg-out>' for o in outs)
                    observation=observation[:6000]+'\n'+obs_tail
                turn_log.append(turn_entry)
                if gave_up:break
        finally:
            if sess is not None:
                try:sess.close()
                except Exception:pass
        wall=time.time()-t0
        tid=self.recorder.record(self.bug_id,self.bug_sha,self.model_id,self.condition,solved,len(turn_log),wall,turn_log,final_patch_text=final_patch)
        return {'solved':solved,'turns':len(turn_log),'turn_log':turn_log,'trajectory_id':tid,'wall_seconds':wall,'final_patch':final_patch}
def make_sandbox(fixture_dir):
    tmp=tempfile.mkdtemp(prefix='dbg_sandbox_')
    src=Path(fixture_dir);dst=Path(tmp)
    for p in src.iterdir():
        if p.is_file():shutil.copy2(p,dst/p.name)
    return Path(tmp)
def cleanup_sandbox(sandbox):
    try:shutil.rmtree(sandbox,ignore_errors=True)
    except Exception:pass
def file_sha(path):
    h=hashlib.sha256();h.update(Path(path).read_bytes());return h.hexdigest()[:16]
def run_with_chat_service(svc,**kw):
    def chat_callable(system,user):
        out,_=svc.chat(user,system=system,max_new_tokens=400,do_sample=False)
        return out
    return DebuggerLoop(chat_callable=chat_callable,**kw)
