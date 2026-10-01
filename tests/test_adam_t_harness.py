import os,re,sys,types
sys.path[:0]=[os.path.dirname(os.path.dirname(os.path.abspath(__file__))),os.path.dirname(os.path.abspath(__file__))]
import pytest
import amni.compute
from amni.serve import adam_t_client as C,adam_t_harness as H,gguf_runtime as G,user_memory as U
from test_adam_t_route import fake,rules,prompts,starts,EM,_fakes
class FC:
    def __init__(s,t=''):s.t,s.P=t,[]
    def gen(s,q,mx,st=(),h=None,pre=(),rc=(),sep=' '):s.P.append((q,list(rc)));return s.t
    def budget(s,mx):return 2000
@pytest.fixture
def on(tmp_path,monkeypatch):
    monkeypatch.setenv('AMNI_UMEM_DIR',str(tmp_path/'umem'));monkeypatch.setenv('AMNI_ADAM_T_HARNESS','1');monkeypatch.delenv('AMNI_UMEM_TOROID',raising=False);[monkeypatch.setenv(f'AMNI_UMEM_{k}','1') for k in ('LT','GN','G2')];U._EID.clear();return tmp_path
FR=types.SimpleNamespace(istitle=lambda x:x.lower() in ('pride and prejudice','romeo and juliet'))
HIST=[("My sister Priya lives in Denver.","Nice!"),("My brother Tomas lives in Tucson.","Got it."),("My dog is called Biscuit.","Cute."),("I'm allergic to penicillin.","Noted.")]
def test_flag_default_off(monkeypatch):
    monkeypatch.delenv('AMNI_ADAM_T_HARNESS',raising=False);assert not H.on() and H.run("What's my sister's name?",HIST,'u','u',FC(),160) is None
    c={'n':{}};assert C._har("What's my sister's name?",HIST,'u','u',FC(),160,c) is None and c=={'n':{}}
@pytest.mark.parametrize('q,want',[("What's my sister's name and where does she live?",["What's my sister's name","where does she live"]),("whats my dog called, and an octopus has how many hearts",["what's my dog called","an octopus has how many hearts"]),("when was he born, and where?",["when was he born","where was he born"]),("Quick, what's the smallest country, and what's the largest?",["what's the smallest country","what's the largest"]),("I love pizza. It is great.",[])])
def test_split(q,want):assert [p.rstrip('?') for p in H.split(q)]==want
@pytest.mark.parametrize('p,want',[("What are the capitals of Norway, Sweden and Denmark?",["What is the capital of Norway","What is the capital of Sweden","What is the capital of Denmark"]),("Who wrote Pride and Prejudice?",["Who wrote Pride and Prejudice?"]),("Where does my sister live and work?",["Where does my sister live","Where does my sister work"]),("What is my sister's name and age?",["What is my sister's name","What is my sister's age"]),("Iron's chemical symbol, and its atomic number?",["Iron's chemical symbol","what is its atomic number"]),("A spider has how many legs, and an insect?",["A spider has how many legs","an insect has how many legs"])])
def test_coord(on,p,want):assert H.coord(p,H.store([],'c'),FR)==want
@pytest.mark.parametrize('q,want',[("Where does my brother live",'where your brother lives'),("When did I move",'when you moved'),("How many legs does a spider have",'how many legs a spider has'),("which unit am i in",'which unit you are in'),("when was Mary Shelley born",'when Mary Shelley was born'),("Remind me of my street address",'about your street address'),("What did my grandma do",'what your grandma did')])
def test_indirect_question(q,want):assert H.ind(q)==want
@pytest.mark.parametrize('e,v,ok',[('time','next Tuesday at 3 pm',1),('time','Denver',0),('num','94',1),('num','Priya',0),('name','Priya',1),('name','Mr',0),('place','because the rent was cheaper',0),('place','Denver',1),('color','dark green',1),('job','Lisbon',0),('job','nurse',1)])
def test_type_fit(e,v,ok):assert H.fits(e,v)==bool(ok)
def test_multi_slot_memory_with_pronoun_and_binding(on):
    cl=FC('model text');r=H.run("What's my sister's name and where does she live?",HIST,'u1','u1',cl,160)
    assert r['text']=="Your sister's name is Priya. Your sister lives in Denver." and [s['src'] for s in r['slots']]==['mem','mem'] and all(s['ok'] for s in r['slots']) and [x[1] for x in cl.P]==[['You told me: My sister Priya lives in Denver.']]*2 and [s['blend'] for s in r['slots']]==[False,False]
    r=H.run("Where does my brother live?",HIST,'u1','u1',cl,160);assert 'Tucson' in r['text'] and 'Denver' not in r['text']
def test_missing_memory_is_hedged_not_invented(on):
    r=H.run("What's my cat's name and how old is my sister?",HIST,'u2','u2',FC('Your cat is Biscuit.'),160)
    assert r['text']=="You haven't told me what your cat's name is yet. You haven't told me how old your sister is yet." and 'Biscuit' not in r['text'] and 'Priya' not in r['text'] and r['n']['hh']==2 and r['n']['hv']==0
def test_possessor_binding_rejects_other_entity(on):
    r=H.run("When does my flight land?",[("My brother's flight lands at 4:30.","Okay.")],'u3','u3',FC(),160);assert '4:30' not in r['text'] and "haven't told me" in r['text']
def test_non_factoid_and_chat_defer(on):
    [H.run(q,[],'u4','u4',FC(),160) is None or pytest.fail(q) for q in ("Guess what, I adopted a puppy!","How do I bake bread?","Can you help me write an email?","What do you think about cats?","I'm feeling sad today.")]
def test_forget_and_statements_defer(on):assert H.run("Forget where my sister lives",HIST,'u5','u5',FC(),160) is None and H.run("My cat is called Olive.",HIST,'u5','u5',FC(),160) is None
def test_calculator_and_memory_mix(on):
    r=H.run("What is 23 times 4 and what's my dog's name?",HIST,'u6','u6',FC(),160);assert r['text']=="23 x 4 = 92. Your dog's name is Biscuit." and [s['kind'] for s in r['slots']]==['calc','mem']
def test_memory_value_feeds_knowledge_pronoun(on,monkeypatch):
    seen=[];monkeypatch.setattr(H,'fill_know',lambda s,cl,R,rq,mx:(seen.append(rq),{**s,'rq':rq,'src':'wd','ok':True,'span':'Alexander Fleming','text':'Penicillin was discovered by Alexander Fleming.'})[1])
    r=H.run("What am I allergic to, and who discovered it?",HIST,'u7','u7',FC(),160);assert seen==['who discovered penicillin?'] and r['text']=="You're allergic to penicillin. Penicillin was discovered by Alexander Fleming."
def test_knowledge_verify_complete_and_hedge(on,monkeypatch):
    R=_fakes(monkeypatch);R.istitle=lambda x:False;W=sys.modules['amni.compute.wikifact'];W.NOUN={'author':[50],'capital':[36]};cl=FC('It was Captain Ahab.')
    monkeypatch.setattr(C,'_know',lambda m,n,cap,h=None,w=None:(["Herman Melville: Herman Melville wrote Moby Dick in 1851."],m));monkeypatch.setattr(C,'_math',lambda m,f=False:('','',None) if f else '')
    r=H.run("who wrote moby dick?",[],'k','k',cl,160);s=r['slots'][0];assert s['src']=='wd' and s['ok'] and not s['conflict'] and r['text']=="Moby-Dick was written by Herman Melville." and len(cl.P)==2 and not s['blend'] and s['mroute']=='know'
    monkeypatch.setenv('AMNI_ADAM_T_HARNESS_AGREE','0');cl.P.clear();r=H.run("who wrote moby dick?",[],'k','k',cl,160);assert r['slots'][0]['ok'] and len(cl.P)==1;monkeypatch.delenv('AMNI_ADAM_T_HARNESS_AGREE')
    W.ask=lambda q:{"answer":"Herman Melville","conf":0.2,"e":5,"p":50,"lf":1.0};r=H.run("who wrote moby dick?",[],'k','k',cl,160);assert r['slots'][0]['src']=='wd+psg' and r['slots'][0]['ok']
    W.ask=lambda q:{"answer":"Nathaniel Hawthorne","conf":0.2,"e":5,"p":50,"lf":1.0};cl.P.clear();r=H.run("who wrote moby dick?",[],'k','k',cl,160);s=r['slots'][0]
    assert s['src']=='psg' and not s['ok'] and s['hedge'] and s['guess']=='Herman Melville' and r['text']=="I think it's Herman Melville, but I couldn't confirm it." and 'Ahab' not in r['text'] and 'moby' not in r['text'].lower() and cl.P[0][1][0].startswith('Herman Melville:')
    cl.t='Herman Melville wrote it.';r=H.run("who wrote moby dick?",[],'k','k',cl,160);assert r['slots'][0]['ok'] and r['text']=="Herman Melville wrote it."
    W.ask=lambda q:{"answer":"Herman Melville","conf":0.9,"e":5,"p":50,"lf":1.0};monkeypatch.setattr(C,'_know',lambda m,n,cap,h=None,w=None:(["Moby-Dick: Moby-Dick was written by Nathaniel Hawthorne in 1851."],m));cl.t='Nathaniel Hawthorne wrote it.';R.candidates=lambda q:('person',[{'v':'Nathaniel Hawthorne','s':0.6}],[])
    r=H.run("who wrote moby dick?",[],'k','k',cl,160);s=r['slots'][0];assert s['conflict'] and s['hedge'] and r['text']=="I think Moby-Dick was written by Herman Melville, or maybe Nathaniel Hawthorne."
@pytest.mark.parametrize('z,e,want',[('1945','time','in the 1940s'),('7 November 1867','time','in the 1860s'),('44 BC','time','in the 1st century BC'),('1215','time','in the 13th century'),('May','time','in May'),('eight','num','fewer than ten'),('37','num','somewhere between ten and a hundred'),('8,849 metres','num','somewhere in the thousands'),('1.4 billion','num','somewhere in the billions'),('71 percent','num','somewhere between 70 and 80 percent'),('Jupiter','time','')])
def test_coarse_guess(z,e,want):assert H._coarse(z,e)==want
def test_hedge_wording_short_and_varied():
    assert H._hedge('Jupiter','ent')=="I think it's Jupiter, but I couldn't confirm it." and H._hedge('in the 1940s','time')=="I think it was in the 1940s, but I couldn't confirm it." and H._hedge('Mars','ent',1)=="Probably Mars, though I couldn't verify that." and H._hedge('Mars','ent',2)=="Maybe Mars, but I'm not certain."
    assert H._guess("I'm not sure the passage says.",'any')==('','') and H._guess('Paris','name')==('Paris','ent') and H._guess('1969','time')==('in the 1960s','time')
def test_multi_hedges_vary_and_no_candidate_defers(on,monkeypatch):
    monkeypatch.setattr(H,'fill_know',lambda s,cl,R,rq,mx:{**s,'rq':rq,'kind':'know','src':'psg','ok':False,'hedge':True,'span':'X','guess':'Mars' if 'planet' in rq else 'Venus','gk':'ent','text':''});R=_fakes(monkeypatch);R.istitle=lambda x:False;monkeypatch.setattr(C,'_math',lambda m,f=False:('','',None) if f else '')
    r=H.run("Which planet is red? Which star is brightest?",[],'k2','k2',FC(),160);assert r['text']=="I think it's Mars, but I couldn't confirm it. Probably Venus, though I couldn't verify that."
    monkeypatch.setattr(H,'fill_know',lambda s,cl,R,rq,mx:{**s,'rq':rq,'kind':'other'});assert H.run("Which planet is red?",[],'k2','k2',FC(),160) is None
def test_route_debug_log_has_no_user_text(fake,rules,on,monkeypatch,caplog):
    caplog.set_level('DEBUG',logger='amni.serve.adam_t');r=G.chat("What's my secret dog's name?",history=[("My secret dog is called Zorro.","Cute.")],user='ann',session='s')
    recs=[x for x in caplog.records if x.name=='amni.serve.adam_t'];assert r['tier']=='tier_adam_t' and 'Zorro' in r['answer'] and all(x['p'].startswith('User: You told me:') for x in prompts(fake)) and rules==[]
    assert len(recs)==1 and recs[0].levelname=='DEBUG' and 'route=harness' in recs[0].getMessage() and 'hs:1' in recs[0].getMessage() and not any(w in x.getMessage() for x in caplog.records for w in ('secret','Zorro','dog'))
def test_emergency_first_unchanged(fake,rules,on,monkeypatch):
    monkeypatch.setattr(H,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('harness ran')))
    r=G.chat(EM,user='ann');assert r['tier']=='tier_emergency' and '911' in r['answer'] and '911' in ''.join(G.chat_stream('my baby is choking',user='ann')) and prompts(fake)==[] and starts(fake)==[] and rules==[]
def test_harness_error_falls_back_to_old_path(fake,rules,on,monkeypatch,caplog):
    caplog.set_level('DEBUG',logger='amni.serve.adam_t');monkeypatch.setattr(H,'run',lambda *a,**k:1/0);monkeypatch.setattr(C,'_know',lambda m,n,cap,h=None,w=None:([],m))
    r=G.chat("tell me about rivers?");assert r['tier']=='tier_adam_t' and r['answer'].startswith('ADAM-T heard:') and any('har_err:ZeroDivisionError' in x.getMessage() and 'route=adam_t' in x.getMessage() for x in caplog.records)
def test_template_newline_sep_keeps_default():
    assert C.template('q?',None,0,(),['a.','b.'])=="User: a. b. q?\nAdam:" and C.template('q?',None,0,(),['a.','b.'],'\n')=="User: a.\nb.\nq?\nAdam:"
def test_memory_quotes_statement_when_parser_misses(on):
    h=[("I play the cello in a community orchestra.","Cool."),("My brother plays the drums.","Loud!")];r=H.run("What instrument do I play?",h,'s1','s1',FC(),160)
    assert r['text']=="You mentioned you play the cello in a community orchestra." and r['slots'][0]['said'] and r['n']['hsd']==1
    r=H.run("What's my cat's name?",HIST,'s2','s2',FC(),160);assert r['text']=="You haven't told me that yet." and not r['slots'][0].get('said')
    r=H.run("Where does my brother work?",HIST,'s3','s3',FC(),160);assert "Tucson" not in r['text'] and "haven't told me" in r['text']
def test_job_attribute_binds_without_work_verb(on):
    r=H.run("What does my sister do for a living?",[("My sister is a nurse at the children's hospital.","Nice.")],'s4','s4',FC(),160);assert 'nurse' in r['text'] and r['slots'][0]['ok'] and not r['slots'][0].get('said')
def test_our_is_not_user_memory(on,monkeypatch):
    seen=[];monkeypatch.setattr(H,'fill_know',lambda s,cl,R,rq,mx:(seen.append(rq),{**s,'rq':rq,'src':'wd','ok':True,'span':'Jupiter','text':'Jupiter.'})[1])
    r=H.run("What is the largest planet in our solar system?",[],'s5','s5',FC(),160);assert r['text']=="Jupiter." and seen and r['slots'][0]['kind']=='know'
def test_between_is_not_a_list(on):assert H.coord("Which ocean lies between Africa and Australia?",H.store([],'c'),FR)==["Which ocean lies between Africa and Australia?"]
def test_advice_with_my_goes_to_model(on):
    cl=FC('Sort by season.');r=H.run("What's a good way to organize my closet?",[],'s6','s6',cl,160);assert r is None
def test_cognition_route_lines_checks_and_compose(on,monkeypatch):
    G=types.SimpleNamespace(mode=lambda m:('factual' if 'capital' in m else 'creative' if 'poem' in m else 'analytical',1.0),parse=lambda m:('analogy',{}) if ' is to ' in m else ('relate',{}) if 'connected' in m or 'poem' in m else (None,{}),context=lambda m,h=None,u=None,b=0:{'mode':'analytical','op':'analogy','lines':["Paris, capital of: France.","Tokyo, capital of: Japan."],'answer':'Japan','ms':1.0} if ' is to ' in m else {'mode':'x','op':'relate','lines':["bee, type of: insect.","Pollination: Bees pollinate almond trees."],'ms':1.0})
    monkeypatch.setitem(sys.modules,'amni.serve.cognition',G);monkeypatch.setenv('AMNI_COGNITION','1');cl=FC("Japan. Tokyo is the capital of Japan.")
    r=H.run("Paris is to France as Tokyo is to what?",[],'g','g',cl,160);assert r['text']=="Japan. Tokyo is the capital of Japan." and r['n']['hg']==1 and r['n']['hgv']==1 and cl.P[0][1]==["Paris, capital of: France.","Tokyo, capital of: Japan."]
    cl.t="Korea. Seoul is nice.";r=H.run("Paris is to France as Tokyo is to what?",[],'g','g',cl,160);assert r['text']=="Japan. Paris is the capital of France. Tokyo is the capital of Japan." and r['n']['hgc']==1
    cl.t="Bees pollinate almond trees, and almond growers rent hives from beekeepers.";r=H.run("How are bees and almonds connected?",[],'g','g',cl,160);assert r['n']['hgc']==1 and r['text'].startswith("Bee is a type of insect.")
    cl.t="Moonlit bees drift over almond blossoms, humming a silver tune.";r=H.run("Write a poem about bees and almonds.",[],'g','g',cl,160);assert r['text']==cl.t and r['n']['hgk']==1 and r['n']['hgv']==1
    cl.t="bees bees bees bees bees bees bees bees";r=H.run("Write a poem about bees and almonds.",[],'g','g',cl,160);assert r is None
    n=len(cl.P);H.run("What is the capital of France?",[],'g','g',cl,160);monkeypatch.setenv('AMNI_COGNITION','0');H.run("How are bees and almonds connected?",[],'g','g',cl,160);assert all('\n' not in ''.join(x[1]) for x in cl.P[n:])
def test_cognition_template_newlines(monkeypatch):
    seen=[];monkeypatch.setattr(C,'template',lambda *a,**k:(seen.append(a),'x')[1]);p=types.SimpleNamespace(text=lambda t,m,s:'ok',alive=lambda:True,info={'ctx':1024});cl=C.AdamT('r','b');cl.p=p;cl.gen('How?',64,C.STOPS,[],(),['a.','b.'],'\n');assert seen[0][5]=='\n'
@pytest.mark.parametrize('q,ok',[("Why do volcanoes erupt?",1),("How do glaciers form?",1),("What causes acid fog?",1),("Can you explain how magnets work?",1),("How can I sleep better?",0),("How do I fix my bike?",0),("How many moons does Mars have?",0),("Why is my phone slow?",0),("What makes you happy?",0),("How are you?",0)])
def test_explain_detection(q,ok):assert bool(H._EXQ.match(q) and not H._EXADV.search(q))==bool(ok)
def test_type_gate_rejects_wrong_class(monkeypatch):
    G_=types.SimpleNamespace(qclass=lambda q:'organ',isa=lambda z,c:(z.lower() in ('liver','kidney'),''),wsyn=lambda w,p=-1:())
    monkeypatch.setitem(sys.modules,'amni.serve.cognition',G_);assert H._tgate("Which organ cleans the blood?","Like")==(False,'organ') and H._tgate("Which organ cleans the blood?","liver")==(True,'organ')
    G_.qclass=lambda q:'';assert H._tgate("What do termites eat?","Slow Wood Inc")==(False,'') and H._tgate("What does NASA stand for?","National Aeronautics")==(True,'')
    monkeypatch.setenv('AMNI_ADAM_T_HARNESS_TGATE','0');assert H._tgate("Which organ cleans the blood?","Like")==(True,'')
def test_commonsense_slot_and_hedge(on,monkeypatch):
    G_=types.SimpleNamespace(know=lambda q:{'answer':'wood','text':'Termites eat wood.','conf':0.8,'prov':['WordNet termite.n.01: ...'],'kind':'object'})
    monkeypatch.setitem(sys.modules,'amni.serve.cognition',G_);s=H._cs({'q':'x'},'What do termites eat?','any');assert s['ok'] and s['text']=='Termites eat wood.' and s['src']=='cs' and s['prov']
    G_.know=lambda q:{'answer':'wood','text':'Termites eat wood.','conf':0.5,'prov':['x'],'kind':'object'};s=H._cs({},'What do termites eat?','any');assert not s['ok'] and s['hedge'] and s['text']=='I think termites eat wood.'
    monkeypatch.setenv('AMNI_ADAM_T_HARNESS_CS','0');assert H._cs({},'What do termites eat?','any')=={}
def test_explain_paraphrase_must_stay_grounded():
    x='A glacier forms where the accumulation of snow exceeds its ablation over many years.';M=H.mat('explain',['Glacier: '+x],x,['Wikipedia: Glacier (passage 1)']);q='How do glaciers form?';b=lambda t:H.blend(FC(t),q,M,200)
    assert b('A glacier forms where snow accumulation exceeds its ablation over many years.')==('A glacier forms where snow accumulation exceeds its ablation over many years.',True)
    assert all(b(t)==(x,False) for t in ('Glaciers are made of chocolate and dragons over many years.','Snow exceeds.','Glacier: A glacier forms where the accumulation of snow exceeds its ablation.'))
HAVEIX=os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data','ks','index_rknow'))
@pytest.mark.skipif(not HAVEIX,reason='wikipedia index missing')
def test_explain_extracts_grounded_passage(on,monkeypatch):
    monkeypatch.setenv('AMNI_ADAM_T_HARNESS_EXPLAIN_PARA','0');o=H.fill_expl({'q':'How do glaciers form?'},FC(),H._rk(),'How do glaciers form?',200)
    assert o['kind']=='explain' and o['ok'] and o['grounded'] and o['prov'][0].startswith('Wikipedia: ') and 'glacier' in o['text'].lower() and len(o['text'].encode())<=380 and o['text'][-1] in '.!'
