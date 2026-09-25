import os,sys,time,json,signal,socket,subprocess,http.client
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from amni.serve import serve_policy as sp,retention
from amni.serve.conversation import detect_personal,PERSONAL
from amni.serve.rate_limit import TokenBucket,client_key,is_local
TOKEN='tok-serve-policy-'+os.urandom(8).hex()
def _port():
    s=socket.socket();s.bind(('127.0.0.1',0));p=s.getsockname()[1];s.close();return p
def _req(port,method,path,headers=None,body=None):
    c=http.client.HTTPConnection('127.0.0.1',port,timeout=10);c.request(method,path,body=body,headers=headers or {});r=c.getresponse();d=r.read();c.close();return r.status,{k.lower():v for k,v in r.getheaders()},d
@pytest.fixture(scope='module')
def server(tmp_path_factory):
    port=_port();log=tmp_path_factory.mktemp('serve_policy')/'server.log'
    env={k:v for k,v in os.environ.items() if not k.startswith(('AMNI_','FORWARDED_ALLOW_IPS'))}
    env.update({'AMNI_AUTH_TOKEN':TOKEN,'AMNI_CORS_ORIGINS':'https://allowed.example','AMNI_RL_CHAT_RPM':'6','AMNI_RL_CHAT_BURST':'3','AMNI_RL_AUTHFAIL_RPM':'600','AMNI_RL_AUTHFAIL_BURST':'200','OMP_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'})
    fh=open(log,'wb');proc=subprocess.Popen([sys.executable,str(ROOT/'tests/security/serve_policy_harness.py'),str(port)],stdout=fh,stderr=subprocess.STDOUT,env=env,cwd=str(ROOT));pid=proc.pid
    try:
        for _ in range(150):
            try:
                if _req(port,'GET','/healthz')[0]==200:break
            except Exception:time.sleep(0.1)
        else:pytest.fail('harness did not start: '+log.read_text(errors='replace')[-800:])
        yield {'port':port,'log':log,'pid':pid}
    finally:
        try:os.kill(pid,signal.SIGTERM);proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.kill(pid,signal.SIGKILL);proc.wait(timeout=5)
        except ProcessLookupError:pass
        fh.close()
def _auth(extra=None):return {'Authorization':'Bearer '+TOKEN,**(extra or {})}
@pytest.mark.parametrize('origin,allowed',[('http://localhost:5173',True),('http://127.0.0.1:7700',True),('https://allowed.example',True),('https://evil.example',False),('null',False)])
def test_cors_preflight_allowed_vs_foreign(server,origin,allowed):
    st,h,_=_req(server['port'],'OPTIONS','/chat',{'Origin':origin,'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'authorization,content-type'})
    assert (h.get('access-control-allow-origin')==origin)==allowed
    assert h.get('access-control-allow-origin')!='*'
    assert (h.get('access-control-allow-credentials')=='true') if allowed else st>=400
def test_cors_simple_request_foreign_origin_gets_no_acao(server):
    _,h,_=_req(server['port'],'GET','/api/ping',_auth({'Origin':'https://evil.example'}))
    assert 'access-control-allow-origin' not in h
    _,h,_=_req(server['port'],'GET','/api/ping',_auth({'Origin':'http://localhost:3000'}))
    assert h.get('access-control-allow-origin')=='http://localhost:3000'
def test_cross_origin_post_blocked_even_with_token(server):
    assert _req(server['port'],'POST','/skills/shell',_auth({'Origin':'https://evil.example','Content-Type':'text/plain'}),b'{}')[0]==403
    assert _req(server['port'],'POST','/skills/shell',_auth({'Origin':'http://localhost:5173'}),b'{}')[0]==200
    assert _req(server['port'],'POST','/skills/shell',_auth(),b'{}')[0]==200
def test_header_and_cookie_tokens_accepted(server):
    p=server['port']
    assert _req(p,'GET','/api/ping',_auth())[0]==200
    assert _req(p,'GET','/api/ping',{'X-Amni-Token':TOKEN})[0]==200
    assert _req(p,'GET','/api/ping',{'Cookie':'amni_token='+TOKEN})[0]==200
    assert _req(p,'GET','/api/ping')[0]==401
    assert _req(p,'GET','/api/ping',{'Authorization':'Bearer wrong'})[0]==401
    assert _req(p,'GET','/healthz')[0]==200
def test_query_token_rejected_by_default_and_not_logged(server):
    st,_,d=_req(server['port'],'GET','/api/ping?token='+TOKEN)
    assert st==401 and 'query-string tokens are disabled' in json.loads(d)['hint']
    _req(server['port'],'GET','/jarvis?token='+TOKEN+'&x=1')
    time.sleep(0.3);log=server['log'].read_text(errors='replace')
    assert TOKEN not in log and '[REDACTED]' in log
def test_rate_limit_trips_for_forwarded_client_only(server):
    p=server['port'];fx={'X-Forwarded-For':'203.0.113.7'}
    codes=[_req(p,'POST','/chat',_auth(fx),b'{}')[0] for _ in range(6)]
    assert codes[:3]==[200,200,200] and 429 in codes[3:]
    st,h,_=_req(p,'POST','/chat',_auth(fx),b'{}')
    assert st==429 and int(h['retry-after'])>=1
    assert _req(p,'POST','/chat',_auth({'X-Forwarded-For':'203.0.113.8'}),b'{}')[0]==200
    assert all(_req(p,'POST','/chat',_auth(),b'{}')[0]==200 for _ in range(12))
    assert _req(p,'POST','/chat/stream',_auth(fx),b'{}')[0]==429
def test_forwarded_header_cannot_claim_local_exemption(server):
    p=server['port'];fx={'X-Forwarded-For':'127.0.0.1'}
    assert 429 in [_req(p,'POST','/chat',_auth(fx),b'{}')[0] for _ in range(6)]
def test_client_key_and_local_rules():
    class R:
        def __init__(s,host,headers):s.client=type('c',(),{'host':host});s.headers=headers
    assert client_key(R('10.0.0.5',{'x-forwarded-for':'1.2.3.4'}))=='10.0.0.5'
    assert client_key(R('127.0.0.1',{'x-forwarded-for':'127.0.0.1, 198.51.100.4'}))=='198.51.100.4'
    assert is_local(R('127.0.0.1',{})) and is_local(R('::1',{}))
    assert not is_local(R('127.0.0.1',{'x-forwarded-for':'127.0.0.1'})) and not is_local(R('192.168.1.9',{}))
def test_token_bucket_refills():
    b=TokenBucket(600,2);assert b.allow('k')[0] and b.allow('k')[0] and not b.allow('k')[0]
    time.sleep(0.15);assert b.allow('k')[0]
def test_origin_allowlist_rules(monkeypatch):
    monkeypatch.setenv('AMNI_CORS_ORIGINS','https://a.example, https://b.example/')
    assert sp.origin_allowed('https://b.example') and sp.origin_allowed('http://[::1]:8080') and not sp.origin_allowed('http://localhost.evil.example')
    assert sp.origin_allowed('https://adam.example','adam.example') and not sp.origin_allowed('https://evil.example','adam.example')
def test_scrub_redacts_tokens():
    s=sp.scrub('GET /x?token=abc123&y=2 HTTP/1.1 Authorization: Bearer zzz.yyy')
    assert 'abc123' not in s and 'zzz.yyy' not in s and 'y=2' in s
def test_unrestricted_roots_need_flag_and_auth(monkeypatch,tmp_path):
    monkeypatch.delenv('AMNI_UNRESTRICTED_FILES',raising=False);monkeypatch.delenv('AMNI_ROOTS',raising=False)
    r,ok,why=sp.resolve_roots([],True,'tok');assert not ok and 'refused' in why
    monkeypatch.setenv('AMNI_UNRESTRICTED_FILES','1')
    assert not sp.resolve_roots([],True,'')[1] and sp.resolve_roots([],True,'tok')[1]
    monkeypatch.setenv('AMNI_ROOTS',str(tmp_path))
    r,ok,_=sp.resolve_roots([],False,'',workdir='/proj');assert r==['/proj',str(tmp_path)] and not ok
    assert sp.resolve_roots([],False,'')[0][0]==os.getcwd()
@pytest.mark.parametrize('text,pii,egress,learn',[
    ('what is the capital of France',False,False,False),
    ('how do I reverse a list in python',False,False,True),
    ('my doctor gave me a diagnosis, what diet helps',False,True,True),
    ('my email is someone@example.com',True,True,True),
])
def test_single_personal_classifier_scopes(text,pii,egress,learn):
    assert detect_personal(text) is pii and detect_personal(text,'egress') is egress and detect_personal(text,'learn') is learn
def test_ray_learn_uses_shared_classifier():
    from amni.compute import ray_learn
    assert ray_learn.PERSONAL is PERSONAL
    assert ray_learn.learn('contact someone@example.com about the capital of France',None) is None
def test_retention_purge_on_temp_dir(tmp_path):
    now=time.time();old=now-200*86400;conv=tmp_path/'experiences'/'conversations';logs=tmp_path/'logs';mem=tmp_path/'experiences'/'user_memory'
    [d.mkdir(parents=True) for d in (conv,logs,mem)]
    (conv/'old.jsonl').write_text('{}\n');(conv/'new.jsonl').write_text('{}\n');(logs/'amni_serve_out_20250101_000000.log').write_text('x')
    (mem/'facts.json').write_text('{}');[os.utime(p,(old,old)) for p in (conv/'old.jsonl',logs/'amni_serve_out_20250101_000000.log',mem/'facts.json')]
    (logs/'agent_skill_calls.jsonl').write_text(json.dumps({'ts':now-400*86400})+'\n'+json.dumps({'ts':now})+'\n')
    dry=retention.sweep(root=tmp_path,days=90,dry_run=True)
    assert len(dry['deleted'])==2 and (conv/'old.jsonl').exists()
    r=retention.sweep(root=tmp_path,days=90,audit_days=365)
    assert not (conv/'old.jsonl').exists() and not (logs/'amni_serve_out_20250101_000000.log').exists() and (conv/'new.jsonl').exists()
    assert (mem/'facts.json').exists() and len((logs/'agent_skill_calls.jsonl').read_text().splitlines())==1
    assert oct((conv/'new.jsonl').stat().st_mode&0o777)=='0o600'
    out=subprocess.run([sys.executable,'-m','amni.serve.retention','--root',str(tmp_path),'--days','0'],cwd=str(ROOT),capture_output=True,text=True,timeout=60)
    assert out.returncode==0 and not (conv/'new.jsonl').exists() and (mem/'facts.json').exists()
def test_retention_size_cap(tmp_path):
    conv=tmp_path/'experiences'/'conversations';conv.mkdir(parents=True);now=time.time()
    for i in range(4):p=conv/f's{i}.jsonl';p.write_bytes(b'x'*400000);os.utime(p,(now-i*60,now-i*60))
    r=retention.sweep(root=tmp_path,days=90,max_mb=1)
    assert sorted(Path(x).name for x in r['deleted'])==['s2.jsonl','s3.jsonl']
def test_auto_retention_disabled_by_env(monkeypatch):
    monkeypatch.setenv('AMNI_RETENTION_AUTO','0');assert retention.start_background() is None
    monkeypatch.setenv('AMNI_RETENTION_AUTO','1');monkeypatch.setenv('AMNI_RETENTION_DAYS','0');assert retention.start_background() is None
def test_server_wiring():
    src=(ROOT/'scripts/amni_serve.py').read_text(encoding='utf-8');jw=(ROOT/'amni/serve/jarvis_web.py').read_text(encoding='utf-8')
    assert '_sp.install(app,cors=args.cors,token=_AUTH_TOKEN)' in src and "allow_origins=['*'],allow_credentials=True" not in src
    assert "request.query_params.get('token'" not in src and '_ret.start_background(' in src and '_sp.resolve_roots(' in src
    assert "_dp(req.message,'egress')" in src and "'?token='" not in jw and 'amni_token=' in jw
