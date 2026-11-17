import os,json,threading,subprocess,tempfile,shutil
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import pytest
from amni.serve import code_harness as C
class _Reg:
    def __init__(s,root):s.root=Path(root)
    def _in_allowed_roots(s,p):return Path(p).resolve().is_relative_to(s.root.resolve())
    def _abs(s,p):return Path(p)
@pytest.fixture()
def repo(monkeypatch,tmp_path):
    monkeypatch.setenv('AMNI_CODE_HARNESS_HOME',str(tmp_path/'harness'))
    d=Path(tempfile.mkdtemp(prefix='code-harness-'));(d/'a.py').write_text('x = 1\n');(d/'old.txt').write_text('bye\n');(d/'.gitignore').write_text('build/\n')
    subprocess.run(['git','init','-q'],cwd=d);subprocess.run(['git','-c','user.name=t','-c','user.email=t@t','add','.'],cwd=d);subprocess.run(['git','-c','user.name=t','-c','user.email=t@t','commit','-qm','i'],cwd=d)
    yield d;shutil.rmtree(d,ignore_errors=True)
def test_modes_and_status():
    assert C.mode_of('edit')=='agent' and C.mode_of('PLAN')=='plan' and C.mode_of('whatever')=='plan' and C.mode_of(None)=='agent'
    s=C.status();assert set(s)>={'ok','binary','sandbox','provider','base_url','instances','docread_mcp'}
def test_gate(repo):
    r=_Reg(repo)
    assert C._gate({'working_dir':str(repo)},{},r)=='missing task'
    assert C._gate({'task':'x','working_dir':'/etc'},{},r).startswith('working_dir outside allowed roots')
    assert C._gate({'task':'x','working_dir':str(repo/'a.py')},{},r).startswith('not a directory')
def test_env_drops_secrets(monkeypatch,tmp_path):
    monkeypatch.setenv('FAKE_API_KEY','s3cret');monkeypatch.setenv('AMNI_CODE_HARNESS_TOKEN','t0k');monkeypatch.setenv('PATH','/usr/bin')
    e=C._env(tmp_path,4321);assert 'FAKE_API_KEY' not in e and 'AMNI_CODE_HARNESS_TOKEN' not in e and e['PATH']=='/usr/bin' and e['HOME']==str(tmp_path) and e['AMNI_CODE_PORT']=='4321' and e['GIT_CEILING_DIRECTORIES']==str(tmp_path/'runs')
def test_sandbox_command(monkeypatch,repo,tmp_path):
    monkeypatch.setattr(C,'sandbox',lambda:True);c=C._cmd(str(repo),tmp_path,{})
    assert c[0]=='bwrap' and ['--bind',str(repo),str(repo)]==c[c.index(str(repo))-1:c.index(str(repo))+2] and '--die-with-parent' in c and c[-1]=='--serve'
    assert C._cmd(str(repo),tmp_path,{},['sh','-c','true'])[-3:]==['sh','-c','true']
    monkeypatch.setattr(C,'sandbox',lambda:False);assert C._cmd(str(repo),tmp_path,{})[-1]=='--serve'
def test_seed_config_lists_docread(monkeypatch,tmp_path):
    (tmp_path/'.amni').mkdir();monkeypatch.setattr(C,'_docread',lambda:'/opt/docread');monkeypatch.setenv('AMNI_CODE_HARNESS_PROVIDER','ollama');monkeypatch.setenv('AMNI_CODE_HARNESS_MODEL','m1')
    c=json.loads((C._seed(tmp_path,'/w') and tmp_path/'.amni'/'config.json').read_text())
    assert c['mcp_servers']==[{'name':'docread','command':'/opt/docread','args':['mcp']}] and c['model']=='m1' and c['api_key']=='' and c['mode']=='plan' and not c['auto_approve']
    monkeypatch.setattr(C,'_docread',lambda:None);assert C._seed(tmp_path,'/w')['mcp_servers']==[]
def test_copy_and_changes(repo,tmp_path):
    man=C._manifest(repo);assert set(man)=={'a.py','old.txt','.gitignore'}
    s=tmp_path/'runs'/'r1';s.parent.mkdir();assert C._copy(repo,s) in ('reflink','copy') and (s/'.git').is_dir() and (s/'a.py').read_text()=='x = 1\n'
    (s/'a.py').write_text('x = 2\n');(s/'old.txt').unlink();(s/'new'/'b.py').parent.mkdir();(s/'new'/'b.py').write_text('y = 1');(s/'build').mkdir();(s/'build'/'out.bin').write_bytes(b'\0'*9);(s/'__pycache__').mkdir();(s/'__pycache__'/'a.cpython-314.pyc').write_bytes(b'\0');(s/'node_modules'/'x').mkdir(parents=True);(s/'node_modules'/'x'/'i.js').write_text('1')
    ch,gone=C._changed(repo,s,man);assert ch==['a.py','new/b.py'] and gone==['old.txt']
    p=Path(C._patch(tmp_path,'r1',repo,s,ch,gone)).read_text();assert '-x = 1\n+x = 2\n' in p and '+++ b/new/b.py' in p and '\\ No newline at end of file' in p and '+++ /dev/null' in p
    chk=subprocess.run(['git','apply','--check','-'],cwd=repo,input=p.encode(),capture_output=True);assert chk.returncode==0,chk.stderr
    r=C._apply(repo,s,man,ch,gone);assert r=={'applied':['a.py','new/b.py'],'deleted':['old.txt'],'conflicts':[]} and (repo/'a.py').read_text()=='x = 2\n' and not (repo/'old.txt').exists() and not (repo/'build').exists()
def test_apply_refuses_concurrent_edit(repo,tmp_path):
    man=C._manifest(repo);s=tmp_path/'r2';C._copy(repo,s);(s/'a.py').write_text('x = 2\n');(repo/'a.py').write_text('x = 9\n')
    r=C._apply(repo,s,man,['a.py'],[]);assert r['conflicts']==['a.py'] and (repo/'a.py').read_text()=='x = 9\n'
def test_validator_detects_projects(tmp_path):
    assert C.validator(tmp_path) is None
    (tmp_path/'test_x.py').write_text('');assert 'python3 -m' in C.validator(tmp_path)
    (tmp_path/'Cargo.toml').write_text('');assert C.validator(tmp_path)=='cargo test --quiet'
class _Fake(BaseHTTPRequestHandler):
    seen=[];body='x = 2\n'
    def log_message(s,*a):pass
    def do_POST(s):
        b=json.loads(s.rfile.read(int(s.headers.get('Content-Length') or 0)) or b'{}');_Fake.seen.append((s.path,b))
        if s.path=='/api/chat':
            Path(b['working_dir'],'a.py').write_text(_Fake.body);s.send_response(200);s.send_header('Content-Type','text/event-stream');s.end_headers()
            for ev,d in (('session',{'session_id':b['session_id']}),('tool_start',{'tool':'write_file','input':{'path':'a.py'}}),('tool_result',{'tool':'write_file','input':{'path':'a.py'},'output':'ok','status':'ok'}),('message',{'message':'Changed x to 2.'}),('done',{'session_id':b['session_id']})):
                s.wfile.write(f'event: {ev}\ndata: {json.dumps(d)}\n\n'.encode());s.wfile.flush()
            return
        s.send_response(200);s.send_header('Content-Type','application/json');s.end_headers();s.wfile.write(b'{"ok":true}')
@pytest.fixture()
def fake(monkeypatch):
    srv=ThreadingHTTPServer(('127.0.0.1',0),_Fake);threading.Thread(target=srv.serve_forever,daemon=True).start();_Fake.seen.clear()
    i={'proc':None,'port':srv.server_port,'url':f'http://127.0.0.1:{srv.server_port}','sessions':set(),'sandbox':False,'t':0.0};monkeypatch.setattr(C,'instance',lambda wd,h=None:i)
    yield i;srv.shutdown()
def test_agent_run_works_in_copy_then_places_and_deletes(repo,fake):
    evs=list(C.run('make x two',str(repo),'s1','agent'))
    assert [e['type'] for e in evs]==['sandbox','start','session','tool_start','tool_result','message','done','apply','summary']
    sb=Path(evs[0]['path']);assert not sb.exists() and sb!=repo and ('/api/config',{'mode':'agent','working_dir':str(sb)}) in _Fake.seen
    s=evs[-1];assert s['applied']==['a.py'] and s['sandbox']=='deleted' and s['changed']==['a.py'] and s['final']=='Changed x to 2.' and s['tools']==1 and Path(s['patch']).is_file()
    assert (repo/'a.py').read_text()=='x = 2\n'
def test_plan_run_never_touches_target(repo,fake):
    evs=list(C.run('look',str(repo),'s2','plan'));a=[e for e in evs if e['type']=='apply'][0]
    assert a['applied']==[] and a['skipped']=='plan mode is read-only' and (repo/'a.py').read_text()=='x = 1\n' and not Path(evs[0]['path']).exists()
def test_failed_validation_keeps_target(repo,fake):
    evs=list(C.run('break it',str(repo),'s3','agent',validate='exit 3'));v=[e for e in evs if e['type']=='validate']
    assert [x['status'] for x in v]==['running','failed'] and v[1]['exit']==3 and (repo/'a.py').read_text()=='x = 1\n'
    s=evs[-1];assert s['applied']==[] and 'validation failed' in s['skipped'] and Path(s['patch']).is_file() and not Path(evs[0]['path']).exists()
    o=C.run_task('fix',str(repo),'s4','agent',validate='true');assert o['ok'] and o['applied']==['a.py'] and o['validation']['ok'] and o['tools'][0]['tool']=='write_file' and (repo/'a.py').read_text()=='x = 2\n'
def test_concurrent_edit_blocks_apply(repo,fake,monkeypatch):
    real=C._validate;monkeypatch.setattr(C,'_validate',lambda s,h,c:((repo/'a.py').write_text('x = 7\n'),real(s,h,c))[1])
    s=list(C.run('race',str(repo),'s5','agent',validate='true'))[-1];assert s['conflicts']==['a.py'] and s['applied']==[] and (repo/'a.py').read_text()=='x = 7\n'
def test_skill_forces_plan_for_model_calls(monkeypatch,repo):
    got={};monkeypatch.setattr(C,'run_task',lambda task,wd,sid,mode,doc,t=0.0,*a:got.update(mode=mode) or {'ok':True})
    C._skill({'task':'x','working_dir':str(repo),'mode':'agent'},{},_Reg(repo));assert got['mode']=='plan'
    C._skill({'task':'x','working_dir':str(repo),'mode':'agent'},{'owner':True},_Reg(repo));assert got['mode']=='agent'
def test_sandbox_never_inside_target(monkeypatch,tmp_path):
    t=tmp_path/'proj';(t/'sub').mkdir(parents=True);monkeypatch.setenv('AMNI_CODE_HARNESS_HOME',str(t/'sub'));monkeypatch.setenv('XDG_CACHE_HOME',str(tmp_path/'c'))
    h=C._home(str(t));r=C._runs(h,t.resolve());assert not r.resolve().is_relative_to(t.resolve()) and r==tmp_path/'c'/'amni-code-runs'/h.name
def test_routes_stream_a_sandboxed_run(repo,fake,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    class Sk(_Reg):
        def __init__(s,root):super().__init__(root);s.reg={}
        def register(s,n,f,**k):s.reg[n]=(f,k)
    app=FastAPI();sk=Sk(repo);monkeypatch.setattr(C,'binary',lambda:'/bin/true');C.mount(app,None,sk);c=TestClient(app)
    assert 'code_agent' in sk.reg and c.get('/code/status').json()['ok'] in (True,False)
    assert c.post('/code/run',json={'working_dir':str(repo)}).status_code==400 and c.post('/code/run',json={'task':'x','working_dir':'/etc'}).status_code==403
    r=c.post('/code/run',json={'task':'make x two','working_dir':str(repo),'session_id':'w1','validate':'true'});evs=[json.loads(l[6:]) for l in r.text.splitlines() if l.startswith('data: ')]
    assert r.status_code==200 and [e['type'] for e in evs][-3:]==['validate','apply','summary'] and evs[-1]['applied']==['a.py'] and evs[-1]['sandbox']=='deleted' and (repo/'a.py').read_text()=='x = 2\n'
