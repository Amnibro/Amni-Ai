import os,time,types,pytest
from amni.compute.gf17_overlay import Ov,nonce
from amni.serve import pipe1 as P,adam_t_harness as H,adam_t_client as C,user_memory as U
@pytest.fixture
def env(tmp_path,monkeypatch):
    monkeypatch.setenv('AMNI_UMEM_DIR',str(tmp_path/'u'));monkeypatch.setenv('AMNI_PIPE1_DIR',str(tmp_path/'g'));monkeypatch.setenv('AMNI_PIPE1','1');monkeypatch.setenv('AMNI_ADAM_T_HARNESS','1');monkeypatch.setenv('AMNI_USER_MEMORY','1');monkeypatch.delenv('AMNI_PIPE1_S',raising=False);P._cache.clear();return tmp_path
class FD:
    def __init__(s):s.V={(10,36):[(1,20,0,0,0)],(11,36):[(1,21,0,0,0)],(12,112):[(1,30,0,0,0),(1,31,0,0,0)]};s.N={20:'Rome',21:'Turin',30:'Steve Jobs',31:'Ronald Wayne',10:'Italy',11:'Kingdom of Italy',12:'Apple'}
    def values(s,e,p):return s.V.get((e,p),[])
    def qid(s,e):return 1000+e
    def ix(s,q):return q-1000
    def label(s,e):return s.N.get(e,'')
class FR:
    D=FD()
    def render(s,k,v,x,p):return s.D.N[v],[s.D.N[v]]
def fakes2(best):
    R=FR();return types.SimpleNamespace(resolve=lambda q:{'best':best,'wd':{'ents':[10,11,12],'props':[36,112]},'abstain':best is None,'ms':1.0},resolver=lambda:R,props=lambda:{36:{'label':'capital'},112:{'label':'founded by'}})
CAP={'store':'wd','e':10,'p':36,'answer':'Rome','line':'Italy, capital: Rome.','prov':'Wikidata Q1010 P36','sure':True,'score':3.5,'key':(1010,36)}
def test_overlay_learn_forget_compact_rebuild_identical():
    o=Ov(nsh=8);K=[nonce(f'k{i}') for i in range(200)]
    for i,k in enumerate(K[:150]):o.learn(k,f'v{i}'.encode())
    o.compact();assert all(o.get(k)==f'v{i}'.encode() for i,k in enumerate(K[:150])) and not any(o.get(k) for k in K[150:])
    t=time.perf_counter();o.learn(K[160],b'new');o.forget(K[3]);assert (time.perf_counter()-t)<0.05 and o.get(K[160])==b'new' and o.get(K[3]) is None and o.get(K[4])==b'v4'
    T=o.compact();assert len(T)<=2 and o.dumps()==o.rebuild().dumps()
    b=o.dumps();assert Ov.loads(b).dumps()==b and Ov.loads(b).get(K[160])==b'new'
def test_overlay_undo_is_byte_identical_and_scrubs():
    o=Ov(1);o.learn(nonce('a'),b'alpha',True);b0=o.dumps();o.learn(nonce('b'),b'secret value',True);assert b'secret' not in o.dumps();o.forget(nonce('b'),True);assert o.dumps()==b0
    o.learn(nonce('a'),b'beta',True);assert o.get(nonce('a'))==b'beta' and b'alpha' not in b''.join(o.heap)
def test_overlay_pixels_are_gf17_digits():
    o=Ov(1);[o.learn(nonce(str(i)),b'x') for i in range(20)];o.compact();assert o.cells[0].max()<17 and o.cells[0].shape[1]==12
def test_strength_map_and_env(monkeypatch):
    assert P.th(0.25)['nw']==0 and abs(P.th(1)['nw']-0.3)<1e-9 and abs(P.th(0.5)['cov']-0.6)<1e-9 and abs(P.th(0.75)['tri']-0.175)<1e-9
    assert P.dec(0.25)=={'temperature':0.0,'top_k':0,'top_p':1.0,'seed':17} and P.dec(1)=={'temperature':0.9,'top_k':40,'top_p':0.95,'seed':17}
    monkeypatch.setenv('AMNI_PIPE1_S','0.75');assert P.strength('store2')==0.75 and P.strength('p1u')==0.0
    monkeypatch.setenv('AMNI_PIPE1_S','{"fact":0.5,"chat":1}');assert P.strength('calc')==0.5 and P.strength('chat')==1.0 and P.strength('explain')==P.DEF['explain']
    monkeypatch.delenv('AMNI_PIPE1_S');assert P.lower(0.5)==0.25 and P.lower(0.15)==0.0 and P.strength('store2')==0.0 and P.strength('explain')==0.5 and P.strength('other')==1.0
H1=[('What is the capital of Italy?','The capital of Italy is Rome.')]
@pytest.mark.parametrize('m,x',[("No, it's Paris.",'Paris'),("that's wrong, actually the answer is Canberra",'Canberra'),('Not quite - it was 1847.','1847'),("You've got that wrong, it's Ottawa, not Toronto",'Ottawa'),('Actually Paris is the capital of Italy.','Paris'),("it's not Rome, it's Paris",'Paris'),('Wrong. Florence.','Florence')])
def test_detect_world_corrections(m,x):d=P.detect(m,H1);assert d['kind']=='world' and d['x']==x
@pytest.mark.parametrize('m',['No worries, I figured it out!','no thanks','Thanks!','Actually, what about France?','Is it Paris?','I had a long day.'])
def test_detect_ignores_non_corrections(m):assert not P.detect(m,H1)
@pytest.mark.parametrize('m',['undo that correction','Scratch what I said, you were right.','forget my correction','please undo that'])
def test_detect_undo(m):assert P.detect(m,H1)=={'kind':'undo'}
def test_detect_personal():
    d=P.detect('My name is spelled Kathryn, with a K and a y.',[]);assert d['kind']=='personal' and d['stmt']=='My name is Kathryn.'
    d=P.detect('No wait, my dog is called Pepper, not Pickles',[('What is my dog called?','Pickles.')]);assert d['kind']=='personal' and d['stmt']=='My dog is called Pepper.'
    assert not P.detect('No worries, my day was fine.',[])
def test_gate_strength_terms():
    M=P.assemble(H.mat('store2',['Italy, capital: Rome.'],'The capital of Italy is Rome.',[],1.0,['Rome']));q='What is the capital of Italy?'
    assert M['lines']==['Draft: The capital of Italy is Rome.'] and M['sup']==['Italy, capital: Rome.'] and M['must']==['Rome']
    g=lambda t,s:H.gate(t,{**M,'s':s},q)
    assert g('Rome is the capital of Italy.',0.25) and not g('Rome is the capital of Italy, 1871.',1) and not g('The capital of Italy is Milan.',1) and not g('Italy has a capital.',1)
    assert not g('Rome is the lovely capital of Italy.',0.25) and g('Rome is the lovely capital of Italy.',1) and not g('Rome is the capital of Italy, says Mario.',1) and not g('Rome is the lovely old capital of sunny Italy.',1)
class FC:
    def __init__(s,t):s.t=t;s.n=0;s.D=[]
    def gen(s,q,mx,st=(),h=None,pre=(),rc=(),sep=' ',dec=None):s.n+=1;s.D.append((list(rc),dec));return s.t
def test_refine_s0_template_no_model_and_retry_fallback():
    M={**P.assemble(H.mat('store2',['Italy, capital: Rome.'],'The capital of Italy is Rome.',[],1.0,['Rome'])),'s':0.0};c=FC('Rome is the capital of Italy.')
    assert P.refine(c,'What is the capital of Italy?',M,200)==('The capital of Italy is Rome.',None,0.0,0) and c.n==0
    c=FC('It is Rome.');r=P.refine(c,'What is the capital of Italy?',{**M,'s':0.5},200);assert c.n==2 and r[0]=='The capital of Italy is Rome.' and r[1] is False and c.D[0][1]['temperature']==0.3 and c.D[1][1]['temperature']==0.0
    assert c.D[1][0][0].startswith('Draft: ') and c.D[1][0][-1]==P.TASK[0] and 'Italy, capital: Rome.' in c.D[0][0]
    c=FC('Rome is the capital of Italy.');r=P.refine(c,'What is the capital of Italy?',{**M,'s':0.25},200);assert r[:3]==('Rome is the capital of Italy.',True,0.25)
def test_chat_refine_falls_back_to_neutral():
    c=FC('As an AI language model I cannot help.');t=P.chat(c,'I had a rough day.',{'hist':[],'pre':[],'rc':[]},200);assert c.n==2 and t[0]==H.EMG.neutral('I had a rough day.') and t[1]==0.0
    c=FC('The Eiffel Tower opened in 1889.');t=P.chat(c,'Where is the tower?',{'hist':[],'pre':[],'rc':['Tower: a tall structure.']},200);assert t[1]==0.0
def test_poison_stays_personal_hedged_and_undo(env,monkeypatch):
    monkeypatch.setattr(P,'_S2',lambda:fakes2(CAP));P.CUR.set(('ua',None));g0=P.gov().dumps()
    r=P.correct("No, it's Paris.",H1,'ua');assert r['slots'][0]['scope']=='rejected' and 'Rome' in r['text'] and P.gov().dumps()==g0
    Q=P.qload();assert len(Q)==1 and Q[0]['tier']=='rejected' and Q[0]['v']=='Paris'
    o=P.overlay('What is the capital of Italy?',CAP);assert o['store']=='p1u' and o['answer']=='Paris' and P.dtext(o,'q').startswith("You told me it's Paris.")
    p=P._up(U.user_key('ua'))[0];assert os.path.exists(p) and b'Paris' not in open(p,'rb').read()
    P.CUR.set(('ub',None));assert not P.overlay('What is the capital of Italy?',CAP)
    P.CUR.set(('ua',None));r=P.correct('undo that correction',H1+[("No, it's Paris.",r['text'])],'ua');assert r['slots'][0]['scope']=='undo' and not os.path.exists(p) and not P.overlay('What is the capital of Italy?',CAP) and P.qload()==[]
def test_verified_resolution_correction_goes_global(env,monkeypatch):
    b={**CAP,'sure':False};monkeypatch.setattr(P,'_S2',lambda:fakes2(b));P.CUR.set(('ua',None))
    r=P.correct("No, it's Turin.",H1,'ua');assert r['slots'][0]['scope']=='verified' and P.qload()[0]['tier']=='verified'
    P.CUR.set(('uz',None));o=P.overlay('What is the capital of Italy?',b);assert o['answer']=='Turin' and 'verified correction' in o['prov']
    assert P.gforget(P.qkey('What is the capital of Italy?')) and not P.overlay('What is the capital of Italy?',b)
def test_unverified_world_and_admin_promote(env,monkeypatch):
    monkeypatch.setattr(P,'_S2',lambda:fakes2(None));P.CUR.set(('ua',None));h=[('Who runs the bakery on Elm Street?',"I'm not sure about that one.")]
    r=P.correct("No, it's Maria Lopez.",h,'ua');assert r['slots'][0]['scope']=='world' and P.qload()[0]['tier']=='unverified'
    P.CUR.set(('ub',None));assert not P.overlay('Who runs the bakery on Elm Street?')
    assert P.promote(P.qload()[0]['id'])['ok'] and P.overlay('Who runs the bakery on Elm Street?')['answer']=='Maria Lopez' and P.qload()[0]['tier']=='endorsed'
def test_safety_and_emergency_never_corrected(env,monkeypatch):
    monkeypatch.setattr(P,'_S2',lambda:fakes2(None));P.CUR.set(('ua',None));assert P.correct("No, it's 912.",[('What number do I call in an emergency?','Call 911.')],'ua')=={} and P.qload()==[];h=[('What number do I call in an emergency?',"I'm not sure about that one.")]
    r=P.correct("No, it's 912.",h,'ua');assert r['slots'][0]['scope']=='safety' and P.qload()[0]['safety'] and not P.overlay('What number do I call in an emergency?') and not P.promote(P.qload()[0]['id'])['ok']
    assert P.correct("No, my friend just collapsed and isn't breathing",H1,'ua')=={}
def test_personal_correction_and_undo(env):
    P.CUR.set(('uc',None));k=U.user_key('uc');U.turn('My dog is called Pickles.',[],'uc')
    h=[('My dog is called Pickles.','Nice!'),('What is my dog called?','Your dog is called Pickles.')]
    r=P.correct('No wait, my dog is called Pepper, not Pickles',h,'uc');assert r['slots'][0]['scope']=='personal' and [f.val for f in U.latest(U.load(k))]==[b'Pepper']
    h2=P.hist(h+[('No wait, my dog is called Pepper, not Pickles',r['text'])],'uc');assert [x[0] for x in h2]==['What is my dog called?',"My dog's name is Pepper."]
    P.correct('forget my correction',h,'uc');assert [f.val for f in U.latest(U.load(k))]==[b'Pickles'] and len(P.hist(h+[('No wait, my dog is called Pepper, not Pickles','ok')],'uc'))==2
def test_forget_clock_keeps_other_versions(env):
    k=U.user_key('ud');U.remember(k,[[1,6,b'Ann',1,'I']]);U.remember(k,[[1,6,b'Anne',1,'I']]);c=max(f.clk for f in U.load(k));assert U.forget(k,1,6,c)==1 and [f.val for f in U.load(k)]==[b'Ann']
def test_forget_everything_wipes_overlay(env,monkeypatch):
    monkeypatch.setattr(P,'_S2',lambda:fakes2(CAP));P.CUR.set(('ue',None));P.correct("No, it's Paris.",H1,'ue');p=P._up(U.user_key('ue'))[0];assert os.path.exists(p)
    assert P.correct('Forget everything you know about me.',[],'ue')=={} and not os.path.exists(p)
def test_decoding_params_reach_the_protocol(monkeypatch):
    seen={};p=types.SimpleNamespace(rpc=lambda **k:(seen.update(k),{'text':C.B64(b'ok')})[1]);assert C.Proto.text(p,'hi',10,(),dec={'temperature':0.3,'top_k':24,'seed':17})=='ok' and seen['temperature']==0.3 and seen['top_k']==24
    a=C.AdamT('x','y');a.p=types.SimpleNamespace(text=lambda t,m,s,**k:(seen.update(k),'r')[1],info={'ctx':1024},alive=lambda:True);a._budget=lambda m:512;assert a.gen('q',10,dec={'temperature':0.6})=='r' and seen['dec']=={'temperature':0.6}
def test_harness_v4_gate_unchanged_without_strength():
    M=H.mat('store2',['Italy, capital: Rome.'],'Italy, capital: Rome.',[],1.0,['Rome']);assert 's' not in M and not H.gate('Rome is the capital of Italy, a lovely city.',M,'What is the capital of Italy?')
def test_imperative_routes_to_lut():
    assert P.imp('Name the capital of Peru, please.')=='What is the capital of Peru?' and P.imp('Tell me about your day')=='Tell me about your day' and P.imp('name a fruit')=='name a fruit'
def test_newcap_and_templates():
    assert P.newcap('Rome is lovely, says Mario.','Rome Italy') and not P.newcap('Rome is the capital.','rome capital') and P.TPL[131].format(L='Munich',A='Bavaria')=='Munich is in Bavaria.'
