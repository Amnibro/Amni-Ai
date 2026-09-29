import os,sys,time,random
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from amni.serve import cognition as C
HAVE=(C.CD/'wordnet_graph.npz').exists() and (C.CD/'sphere.npz').exists()
HAVEWD=HAVE and C.WF.store() is not None
need=pytest.mark.skipif(not HAVE,reason='concept data missing')
needwd=pytest.mark.skipif(not HAVEWD,reason='wikidata store missing')
@pytest.fixture(scope='module')
def warm():
    C.warm();return True
def test_gf17_rapidity_roundtrip_and_additive_boosts():
    assert all(C.rn(C.rz(n))==n for n in range(0,4096))
    rng=random.Random(5)
    for _ in range(300):
        n=rng.randrange(0,5000);ks=[rng.randrange(-40,41) for _ in range(rng.randrange(1,12))];c=C.rz(n)
        for k in ks:c=C.boost(c,k)
        assert C.rn(c)==n+sum(ks) and c==C.rz(n+sum(ks))
    assert all(pow(3,C.L17[C.rz(n)[1]],17)==C.rz(n)[1] for n in range(64))
def test_lorentz_distance_and_band():
    assert C.ldist(3,3,1.0)==pytest.approx(0.0,abs=1e-6) and C.ldist(2,5,0.3)==pytest.approx(C.ldist(5,2,0.3))
    assert C.ldist(4,4,-1.0)==pytest.approx(C.DL*8) and C.ldist(4,4,0.9)<C.ldist(4,4,0.0)<C.ldist(4,4,-0.9)
    assert C.band(C.D0)==pytest.approx(1.0) and C.band(0.0)==0.0 and C.band(0.1)<0.2 and C.band(4*C.D0)<C.band(C.D0) and C.band(None)==0.6
@pytest.mark.parametrize('q,m',[("What is the capital of Peru?",'factual'),("Who painted Guernica?",'factual'),("How many bones are in the human hand?",'factual'),("How are volcanoes and earthquakes connected?",'analytical'),("Why do birds migrate?",'analytical'),("What do sharks and rays have in common?",'analytical'),("Is a bat a kind of bird?",'analytical'),("Write a haiku about snow.",'creative'),("Give me metaphors for grief.",'creative'),("What if the oceans were made of honey?",'creative')])
def test_mode_dial(q,m):
    assert C.mode(q)[0]==m
@pytest.mark.parametrize('q,op,args',[("How are bees and almonds connected?",'relate',{'a':'bees','b':'almonds'}),("What's the relationship between inflation and wages?",'relate',{'a':'inflation','b':'wages'}),("What do a whale and a bat have in common?",'common',{'a':'whale','b':'bat'}),("What do Mars and Venus share?",'common',{'a':'Mars','b':'Venus'}),("Paris is to France as Tokyo is to what?",'analogy',{'a':'Paris','b':'France','c':'Tokyo'}),("puppy : dog :: kitten : ?",'analogy',{'a':'puppy','b':'dog','c':'kitten'}),("What kind of animal is a sparrow?",'abstract',{'a':'sparrow'}),("Is a whale a mammal?",'deduce',{'a':'whale','b':'mammal'}),("Is Paris in Europe?",'deduce_in',{'a':'Paris','b':'Europe'}),("Give me metaphors for time",'connect',{'a':'time'}),("Write a poem about the moon and the sea",'connect',{'a':'moon','b':'sea'}),("Who wrote Hamlet?",None,{})])
def test_parse(q,op,args):
    o,a=C.parse(q);assert o==op and a==args,(o,a)
@need
def test_wordnet_depth_and_hypernym_chain(warm):
    s=C.wsyn('sparrow',0)[0];b=C.wsyn('bird',0)[0];A=C.anc(s);assert b in A and C.dep(s)>C.dep(b)
    ch=C.chain(A,b);assert ch[0]==s and ch[-1]==b and len(ch)==A[b][0]+1
    assert C.wsyn('bats',0)[0]==C.wsyn('bat',0)[0] and C.hd(C.wsyn('bat',0)[0])=='bat'
@need
def test_abstract_steps_up_with_exact_boosts(warm):
    o=C.abstract('sparrow');assert o['ok'] and any('bird' in e['b'] for e in o['paths'][0]['e'])
    n=[c['n'] for c in o['chain']];assert n==sorted(n,reverse=True) and all(C.rn(c['rap'])==c['n'] for c in o['chain'])
    assert len(C.abstract('sparrow','factual')['lines'])==1
@needwd
def test_relate_paths_have_provenance_and_are_fast(warm):
    t=time.perf_counter();o=C.relate('bees','almonds');ms=(time.perf_counter()-t)*1000
    assert o['ok'] and o['lines'] and ms<200 and all(e['ref'] for p in o['paths'] for e in p['e'])
    assert any(k in ' '.join(o['lines']).lower() for k in ('pollinat','bloom','orchard'))
    o=C.relate('Marie Curie','Pierre Curie');assert 'spouse' in o['lines'][0]
@needwd
def test_factual_mode_pulls_back_to_verified_sources(warm):
    for a,b in (('bees','almonds'),('coffee','Ethiopia'),('moon','sea')):
        o=C.relate(a,b,'factual');assert len(o['lines'])<=C.NL['factual']
        assert all(e['src'] in C.STRICT or (e['src']=='tx' and e['c']>C.CF['tx']) for p in o['paths'] for e in p['e'])
@needwd
def test_common_whale_bat(warm):
    o=C.common('whale','bat');t=' '.join(o['lines']).lower();assert 'mammal' in t or 'placental' in t
@needwd
@pytest.mark.parametrize('a,b,c,want',[('Paris','France','Tokyo','japan'),('hot','cold','tall','short'),('finger','hand','toe','foot'),('dog','mammal','sparrow','bird'),('Rome','Italy','Madrid','spain')])
def test_analogy(warm,a,b,c,want):
    o=C.analogy(a,b,c);assert want in (o['answer'] or '').lower(),o['answer']
    assert len(o['paths'][0]['e'])==2 and all(e['line'] for e in o['paths'][0]['e'])
@needwd
def test_deduce_chains(warm):
    assert C.deduce('whale','mammal')['holds'] and C.deduce('Paris','Europe','in')['holds']
@need
def test_connect_scores_validity_times_band(warm):
    o=C.connect('winter',full=True);A=o['all'];assert o['lines'] and len(A)>10
    assert all(abs(x['cr']-x['v']*C.band(x['D']))<2e-3 for x in A if x['D'] is not None)
    assert A[0]['cr']>=A[-1]['cr'] and A[-1]['D'] is not None and C.band(A[-1]['D'])<0.2
@needwd
def test_user_toroid_entities_link_into_concept_map(warm):
    from amni.serve import user_toroid as T
    st=T.Store();T.observe(st,"My dog is called Biscuit.");o=C.relate('my dog','wolf',user=st);L=' '.join(o['lines'])
    assert 'your dog' in L and 'Biscuit' in L and o['ok']
def test_context_is_flag_gated(monkeypatch):
    monkeypatch.delenv('AMNI_COGNITION',raising=False);assert C.context('How are bees and almonds connected?')['lines']==[]
@needwd
def test_context_lines_fit_budget(warm,monkeypatch):
    monkeypatch.setenv('AMNI_COGNITION','1');o=C.context('How are bees and almonds connected?',budget=300)
    assert o['mode']=='analytical' and o['op']=='relate' and o['lines'] and sum(len(x)+1 for x in o['lines'])<=300 and all(x.isascii() for x in o['lines'])
