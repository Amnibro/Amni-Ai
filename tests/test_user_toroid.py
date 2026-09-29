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
    st=_st(["My brother lives in Austin.","I moved to Boston because the rent was cheaper.","I get to work by train.","My flight is on Friday at 6 pm.","My sister is a nurse."]);r=T.respond(st,q);assert r['kind']=='direct' and r['text']==want,r['text']
def test_no_fact_no_false_recall():
    st=_st(["My flight is on Friday.","My sister lives in Denver."])
    assert all(T.respond(st,q)['kind']!='direct' for q in ('Why am I flying?','What is my sister\'s name?','Where does my cousin live?'))
    assert T.respond(st,'Why am I flying?',abstain=True)['text'] in ("You haven't told me that yet.",'')
