import os,sys,stat
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from amni.serve import user_toroid as T,user_memory as U
def _st(msgs):
    st=T.Store();[T.observe(st,m) for m in msgs];return st
def _ans(st,q,tier='both'):
    r=T.recall(st,q,tier);return r['line']
@pytest.fixture
def env(tmp_path,monkeypatch):
    monkeypatch.setenv('AMNI_UMEM_DIR',str(tmp_path));monkeypatch.setenv('AMNI_UMEM_TOROID','1');[monkeypatch.setenv(f'AMNI_UMEM_{k}','1') for k in ('LT','GN','G2')];T._cache.clear();return tmp_path
def test_flag_default_off(monkeypatch):
    monkeypatch.delenv('AMNI_UMEM_TOROID',raising=False);assert not T.enabled() and T.turn('My car is red.')['facts']==0
@pytest.mark.parametrize('msg,want',[("My car is a blue Honda Civic.",[('car','what','blue Honda Civic'),('car','what','blue')]),("My sister Priya lives in Denver.",[('sister','what','Priya'),('sister','where','Denver')]),("I moved to Boston in 2019 because the rent was cheaper.",[('me','where','Boston'),('me','when','2019'),('me','why','because the rent was cheaper')]),("I get to work by train.",[('me','how','by train')]),("My dentist appointment is next Tuesday at 3 pm.",[('dentist appointment','when','next Tuesday at 3 pm')]),("I keep my passport in the top drawer of my desk.",[('passport','where','the top drawer of my desk')]),("I make my coffee with a French press.",[('coffee','how','with a French press')])])
def test_extract_5w(msg,want):
    F=T.extract(msg)[0];got={(e,w,v) for e,_,w,_,v,_,_ in F};assert all(x in got for x in want),(F,want)
def test_questions_and_requests_not_stored():
    st=_st(["What's a good pasta recipe?","Can you write a poem about my dog?"]);assert len(st.s)==0 and len(st.u)==0
def test_supersession_and_ellipsis():
    st=_st(["My car is a blue Honda Civic.","Actually it's red."]);assert 'red' in _ans(st,'What color is my car?') and 'blue' not in _ans(st,'What color is my car?')
    st=_st(["My flight is at 7 am tomorrow.","Oh wait, I misread the ticket, it's 9:40 am."]);assert '9:40' in _ans(st,"What time's my flight?")
    st=_st(["I live in Leeds.","I moved last month, I'm in Sheffield now."]);assert 'Sheffield' in _ans(st,'Where do I live?')
def test_multi_entity_binding():
    st=_st(["My sister lives in Denver and my brother lives in Tucson.","My mom lives in Leeds."])
    assert 'Tucson' in _ans(st,'Where does my brother live?') and 'Denver' in _ans(st,'Where does my sister live?') and 'Leeds' in _ans(st,'Remind me where my mom lives?')
@pytest.mark.parametrize('tier',['slot','soft','both'])
def test_unknown_slot_abstains(tier):
    st=_st(["My flight is on Friday."]);assert _ans(st,'Why am I flying?',tier)=='' and _ans(st,'Where does my cousin live?',tier)==''
def test_forget_entity_w_and_all(env):
    k=U.user_key('t');[T.remember(k,m) for m in ("My brother lives in Austin and works at a bakery.","My sister lives in Denver.")]
    assert T.forget(k,'Can you delete where my brother lives?')[1]>0;st=T.load(k)
    assert _ans(st,'Where does my brother live?')=='' and 'Denver' in _ans(st,'Where does my sister live?') and not any('Austin' in st.utext(i) for i in range(len(st.uidx()[0])))
    assert T.forget(k,'Please forget everything about my brother.')[0]=='brother' and _ans(T.load(k),'Where does my brother work?')==''
    assert T.forget(k,'Forget everything you know about me.')[0]=='all' and not os.path.exists(T.path(k))
@pytest.mark.parametrize('msg',["I forgot my keys.","Don't forget to call mom.","I always forget where I parked.","Delete the file on my desktop.","Clear my schedule for Friday.","Never mind."])
def test_forget_vetoes_keep_memory(msg):
    st=_st(["My brother lives in Austin.","I parked on level 2."]);n=len(st.s);k,_=T.observe(st,msg);assert len(st.s)>=n and 'Austin' in _ans(st,'Where does my brother live?')
def test_pages_gf17_roundtrip_and_crc():
    st=_st(["My car is a blue Honda Civic.","I moved to Boston in 2019 because the rent was cheaper.","My dog is called Biscuit."]);raw=T.encode(st);d=np.frombuffer(raw[T.HDR.size+8:],np.uint8)
    assert d.max()<17 and len(d)%4==0;st2=T.decode(raw);assert np.array_equal(st.s,st2.s) and np.array_equal(st.u,st2.u) and st2.head==st.head
    assert [st2.utext(i) for i in range(len(st2.uidx()[0]))][-1]=='My dog is called Biscuit.'
    bad=bytearray(raw);bad[-1]=(bad[-1]+1)%17
    with pytest.raises(ValueError):T.decode(bytes(bad))
def test_torus_clock_wraps_and_rebases():
    st=T.Store(head=T.TM-2);[T.observe(st,m) for m in ("My car is blue.","Actually it's red.","Actually it's green.","Sorry, it's yellow.")];assert st.head>T.TM and 'yellow' in _ans(st,'What color is my car?')
    st.head+=T.TM//2+5;raw=T.encode(st);st2=T.decode(raw);assert st2.age(st2.s[:,4]).max()<T.TM//2 and 'yellow' in _ans(st2,'What color is my car?')
def test_turn_persists_private_file(env,monkeypatch):
    [T.turn(m,user='u1') for m in ("My dog is called Biscuit.","My sister lives in Denver.")];r=T.turn('What is my dog called?',user='u1');p=T.path(U.user_key('u1'))
    assert r['direct']=="Your dog's name is Biscuit." and r['lines']==[] and stat.S_IMODE(os.stat(p).st_mode)==0o600 and b'Biscuit' not in open(p,'rb').read() and b'Denver' not in open(p,'rb').read()
    assert T.turn('What is my dog called?',user='u2')['lines']==[] and T.turn('What is my dog called?',user='u2')['direct']==''
    monkeypatch.setenv('AMNI_UMEM_TOROID_DIRECT','0');assert T.turn('What is my dog called?',user='u1')['lines']==['Memory: your dog, name: Biscuit.']
def test_client_context_injects_memory_line(env,monkeypatch):
    [monkeypatch.setenv(k,v) for k,v in (('AMNI_MATH','0'),('AMNI_ADAM_T_KNOW_BYTES','0'),('AMNI_ADAM_T_CONTEXT','1'))];from amni.serve import adam_t_client as C
    h=[];[(C.context(m,h,'u3'),h.append((m,'Got it.'))) for m in ("My car is a blue Honda Civic.","Actually it's red.")];assert C.context('What color is my car?',h,'u3')['reply']=='Your car is red.'
    monkeypatch.setenv('AMNI_UMEM_TOROID_DIRECT','0');c=C.context('What color is my car?',h,'u3')
    assert c['reply'] is None and c['rc'][-1]=='Memory: your car, color: red.' and c['pre']==[] and C.template('What color is my car?',c['hist'],2928,c['pre'],c['rc']).endswith('User: Memory: your car, color: red. What color is my car?\nAdam:')
    monkeypatch.setenv('AMNI_UMEM_TOROID','0');assert C.context('What color is my car?',h,'u3')['n'].get('tor',0)==0
def test_bus_spine_verify_supersede_and_private(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);st=T.Store();B=T.bus(st);seen=[]
    orig=B.record_learning;B.record_learning=lambda *a,**k:(seen.append((a,k)),orig(*a,**k))[1]
    [T.ingest(st,m,B) for m in ("My car is blue.","Actually it's red.","Sorry, it's blue after all.")]
    assert all(k['kind']=='user_fact' and k['exactness']=='exact' for _,k in seen) and [k['supersedes'] for _,k in seen]==[None,'blue','red'] and not os.path.exists(tmp_path/'data'/'corrections.jsonl')
    assert T.respond(st,'What color is my car?')['text']=='Your car is blue.' and B.recall('What color is my car?')[1]=='tier0_atex_override'
    B.answer_lut.lookup=lambda q:None;n=len(st.s);assert not orig('slot::car\x1f\x1fwhat\x1fcolor\x1f0.9','green',kind='user_fact',provenance='user:toroid',exactness='exact')['stored'] and len(st.s)==n
    from amni.serve.memory_bus import MemoryBus
    assert MemoryBus.__init__.__defaults__[-1]!=os.devnull and B.ledger.as_posix()==os.devnull and B.adam is None and B.kb is None and B.la is None
@pytest.mark.parametrize('q,want',[('Where does my brother live?','Your brother lives in Austin.'),('Why did I move to Boston?','Because the rent was cheaper.'),('How do I get to work?','By train.'),('When is my flight?',"It's Friday at 6 pm."),('What does my sister do?','Your sister works as a nurse.')])
def test_direct_answer_templates(q,want):
    st=_st(["My brother lives in Austin.","I moved to Boston because the rent was cheaper.","I get to work by train.","My flight is on Friday at 6 pm.","My sister is a nurse."]);r=T.respond(st,q,gate=0.5);assert r["kind"]=="direct" and r["text"]==want,r["text"]
def test_no_fact_no_false_recall():
    st=_st(["My flight is on Friday.","My sister lives in Denver."])
    assert all(T.respond(st,q)['kind']!='direct' for q in ('Why am I flying?','What is my sister\'s name?','Where does my cousin live?'))
    assert T.respond(st,'Why am I flying?',abstain=True)['text'] in ("You haven't told me that yet.",'')
def _put(k,*ms):[T.remember(k,m) for m in ms];return open(T.path(k),'rb').read()
def test_crypto_roundtrip_ciphertext_and_key_file(env):
    k=U.user_key('c1');raw=_put(k,"My dog is called Biscuit.","I moved to Boston because the rent was cheaper.");kp=T.kpath(k)
    assert raw[:4]==T.MAGIC and int.from_bytes(raw[4:6],'little')==T.EVER and not any(x in raw for x in (b'Biscuit',b'Boston',b'rent'))
    assert stat.S_IMODE(os.stat(kp).st_mode)==0o600 and stat.S_IMODE(os.stat(T.kdir()).st_mode)==0o700 and len(open(kp,'rb').read())==32 and not os.path.exists(kp+'.new')
    T._cache.clear();st=T.load(k);assert 'Biscuit' in _ans(st,'What is my dog called?') and 'rent' in _ans(st,'Why did I move to Boston?')
    k2=U.user_key('c2');_put(k2,"My cat is called Mochi.");assert open(T.kpath(k2),'rb').read()!=open(kp,'rb').read() and T._kid(T._uk(open(kp,'rb').read(),k))!=T._kid(T._uk(open(kp,'rb').read(),k2))
@pytest.mark.parametrize('off',[2,6,9,20,40,-1,-17])
def test_crypto_tamper_detected(env,off):
    k=U.user_key('c3');raw=bytearray(_put(k,"My car is a blue Honda Civic."));raw[off]^=0x01;open(T.path(k),'wb').write(bytes(raw));T._cache.clear()
    with pytest.raises(ValueError):T.load(k)
    r=T.turn('My sister lives in Denver.',user='c3');assert r.get('err')=='ValueError' and bytes(raw)==open(T.path(k),'rb').read()
def test_crypto_wrong_key_and_swapped_file_fail(env):
    k=U.user_key('c4');_put(k,"My car is red.");kp=T.kpath(k);open(kp,'wb').write(os.urandom(32));T._cache.clear()
    with pytest.raises(ValueError):T.load(k)
    k5,k6=U.user_key('c5'),U.user_key('c6');_put(k5,"My car is red.");_put(k6,"My car is blue.");open(T.path(k6),'wb').write(open(T.path(k5),'rb').read());T._cache.clear()
    with pytest.raises(ValueError):T.load(k6)
    ku=U.user_key('c7');_put(ku,"My car is red.");os.remove(T.kpath(ku));T._cache.clear()
    with pytest.raises(ValueError):T.load(ku)
    with pytest.raises(ValueError):T.unseal(T.seal(b'x'*40,os.urandom(32)),os.urandom(32))
def test_crypto_strict_key_permissions(env):
    k=U.user_key('c8');_put(k,"My car is red.");os.chmod(T.kpath(k),0o644);T._cache.clear()
    with pytest.raises(PermissionError):T.load(k)
    os.chmod(T.kpath(k),0o600);os.chmod(T.kdir(),0o755);T._cache.clear()
    with pytest.raises(PermissionError):T.load(k)
    os.chmod(T.kdir(),0o700);T._cache.clear();assert 'red' in _ans(T.load(k),'What color is my car?')
def test_crypto_migrates_plaintext_pages(env):
    k=U.user_key('c9');st=_st(["My dog is called Biscuit.","My sister lives in Denver."]);p=T.path(k);os.makedirs(os.path.dirname(p),exist_ok=True);open(p,'wb').write(T.encode(st));os.chmod(p,0o600)
    assert b'Denver' not in open(p,'rb').read() and open(p,'rb').read()[4:6]==b'\x01\x00' and not os.path.exists(T.kpath(k))
    s2=T.load(k);raw=open(p,'rb').read();assert raw[4:6]==b'\x02\x00' and os.path.exists(T.kpath(k)) and 'Denver' in _ans(s2,'Where does my sister live?')
    T._cache.clear();assert np.array_equal(T.load(k).s,st.s) and T.decode(T.unseal(raw,T._uk(open(T.kpath(k),'rb').read(),k))).u.shape==st.u.shape
def test_crypto_forget_rotates_key_and_shreds(env):
    k=U.user_key('c10');old=_put(k,"My brother lives in Austin.","My sister lives in Denver.");ok=open(T.kpath(k),'rb').read()
    assert T.forget(k,'Please forget everything about my brother.')[0]=='brother';new=open(T.path(k),'rb').read();nk=open(T.kpath(k),'rb').read()
    assert nk!=ok and new[8:16]!=old[8:16] and not os.path.exists(T.kpath(k)+'.new');st=T.load(k)
    assert _ans(st,'Where does my brother live?')=='' and 'Denver' in _ans(st,'Where does my sister live?') and not any('Austin' in st.utext(i) for i in range(len(st.uidx()[0])))
    with pytest.raises(ValueError):T.unseal(old,T._uk(nk,k))
    r=T.turn('Forget everything you know about me.',user='c10x');assert r['forgot']==('all',0)
    kk=U.user_key('c10x');[T.turn(m,user='c10x') for m in ("My car is red.",)];assert os.path.exists(T.path(kk)) and os.path.exists(T.kpath(kk))
    r=T.turn('Forget everything you know about me.',user='c10x');assert r['forgot']==('all',1) and not os.path.exists(T.path(kk)) and not os.path.exists(T.kpath(kk))
def test_crypto_forget_all_works_when_undecryptable(env):
    k=U.user_key('c11');_put(k,"My car is red.");open(T.kpath(k),'wb').write(os.urandom(32));T._cache.clear()
    assert T.turn('Forget everything you know about me.',user='c11')['forgot']==('all',1) and not os.path.exists(T.path(k)) and not os.path.exists(T.kpath(k))
def test_crypto_interrupted_rotation_recovers(env):
    k=U.user_key('c12');_put(k,"My car is red.");kp=T.kpath(k);st=T.load(k);kf=os.urandom(32);T._wfile(kp+'.new',kf);T._wfile(T.path(k),T.seal(T.encode(st),T._uk(kf,k)));T._cache.clear()
    assert 'red' in _ans(T.load(k),'What color is my car?') and open(kp,'rb').read()==kf and not os.path.exists(kp+'.new')
    T._wfile(kp+'.new',os.urandom(32));T._cache.clear();assert 'red' in _ans(T.load(k),'What color is my car?') and open(kp,'rb').read()==kf and not os.path.exists(kp+'.new')
def test_crypto_no_plaintext_or_key_in_bus_ledger_or_logs(env,tmp_path,monkeypatch,caplog):
    monkeypatch.chdir(tmp_path);k=U.user_key('c13');_put(k,"My passport is in the top drawer of my desk.");kf=open(T.kpath(k),'rb').read()
    assert not caplog.records and not [f for f in os.listdir(tmp_path) if f not in ('keys',) and not f.endswith(('.torus','.salt'))]
    assert all(kf not in open(os.path.join(dp,f),'rb').read() for dp,_,fs in os.walk(tmp_path) for f in fs if not f.endswith('.key'))
@pytest.mark.skipif(T._cog() is None,reason='concept data unavailable')
@pytest.mark.parametrize('raw,want',[("i moved here in febuary bc the rent was cheeper","february because"),("my freind oskar skipped the reunoin","friend reunion"),("whats my sis's bday","what 's sister's birthday"),("the dentist is thurs at 8:40am","thursday 8:40 am")])
def test_normalisation_repairs_casual_and_typos(raw,want):
    assert all(w in ' '.join(t[0].lower() for t in T.toks(raw)[1]) for w in want.split())
@pytest.mark.parametrize('msg,q,want',[("quit sugar last spring bc my dentist nagged me","why did i give up sugar",'dentist nagged'),("The reason my uncle sold the boat was that the mooring fees doubled.","Why did my uncle sell the boat?",'mooring fees'),("my knees were killing me, so I stopped running","why did I stop running?",'knees'),("You know how my aunt Greta lives in Tromso now?","where does my aunt live",'Tromso'),("moved all my recipes into a shared notebook last night, happy?","what am i using for my recipes now",'shared notebook')])
def test_general_phrasings(msg,q,want):
    st=_st([msg]);assert want.lower() in _ans(st,q).lower(),(_ans(st,q),T.ask(q))
@pytest.mark.skipif(T._cog() is None,reason='concept data unavailable')
@pytest.mark.parametrize('msg,target',[("can u delete that thing about my gym schedule, dont need it saved","gym schedule"),("actualy dont rember where my cousin works ok","cousin"),("Let's not keep the bit about my landlord on file.","landlord"),("Please wipe the details of my dentist appointment from your memory.","dentist appointment")])
def test_extended_forget_phrasings(msg,target):
    f=T._fg2(msg) or U._fg(msg);assert f and not f[1] and target.split()[-1] in f[0].lower(),f
def test_forget_redacts_only_the_forgotten_clause(env):
    k=U.user_key('r1');[T.remember(k,m) for m in ("My brother lives in Austin and works at a bakery.",)];T.forget(k,'Can you delete where my brother lives?');st=T.load(k)
    assert _ans(st,'Where does my brother live?')=='' and 'bakery' in _ans(st,'Where does my brother work?') and not any('Austin' in st.utext(i) for i in range(len(st.uidx()[0])))
def test_forget_clears_span_cache(env):
    k=U.user_key('r2');T.remember(k,"My neighbour Ingrid keeps bees in her garden.");T.recall(T.load(k),'Where does my neighbour keep bees?');assert T._spc.cache_info().currsize>0
    T.forget(k,'Please forget everything about my neighbour.');assert T._spc.cache_info().currsize==0
def test_policy_defaults_gate_and_no_unsure(monkeypatch):
    st=_st(["My brother lives in Austin.","My sister lives in Denver."]);r=T.respond(st,'Where does my brother live?');assert r['kind']=='direct' and r['conf']>=0.9 and r['text']=='Your brother lives in Austin.'
    assert T.respond(st,'Where does my cousin live?')['kind']=='none'
