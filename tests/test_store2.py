import os,pytest
from amni.compute import store1 as D1,wikifact as WF
pytestmark=pytest.mark.skipif(not (D1.DIR2/'store2.gf17pg').exists() or not os.path.exists(D1.LIB2),reason='STORE-2 drive not built')
def test_flag_off_is_inert(monkeypatch):
    from amni.serve import adam_t_harness as H
    monkeypatch.setenv('AMNI_STORE1','0');monkeypatch.setenv('AMNI_STORE2','0');assert H.fill_s1({'q':'x'},'What is the capital of Kenya?')=={}
def test_flag_on_material_format(monkeypatch):
    from amni.serve import adam_t_harness as H
    monkeypatch.setenv('AMNI_STORE1','0');monkeypatch.setenv('AMNI_STORE2','1');o=H.fill_s1({'q':'What is the capital of Kenya?'},'What is the capital of Kenya?')
    assert o['ok'] and o['src']=='store2' and o['mat']['route']=='store2' and 'Nairobi' in o['mat']['lines'][0] and o['mat']['must']==['Nairobi'] and ': ' in o['mat']['lines'][0]
def test_resolver_routes(monkeypatch):
    monkeypatch.setenv('AMNI_STORE2','1')
    from amni.serve import store1 as S
    assert S.resolve('Is 97 prime?')['best']['answer']=='yes'
    assert S.resolve('What is 84 divided by 7?')['best']['answer']=='12'
    r=S.resolve('When was Abraham Lincoln born?');assert r['best']['store']=='wd' and '1809' in r['best']['answer']
    assert S.resolve('Who founded the village of Quellbrook Vennick Hollow?')['abstain']
def test_drive_matches_v3_sample():
    S3=WF.store();D=D1.drive2()
    if S3 is None:pytest.skip('v3 store missing')
    for e in (0,1,5,62,1000,123456,5000000,11900000):
        assert D.label(e)==S3.label(e) and D.title(e)==S3.title(e) and D.names(e)==S3.names(e) and D.qid(e)==int(S3.e_qid[e]) and D.sitelinks(e)==int(S3.e_sl[e]) and D.ix(int(S3.e_qid[e]))==e
        a,n=int(S3.e_fs[e]),int(S3.e_fn[e])
        for p in sorted({int(x) for x in S3.f_p[a:a+n]}):
            assert [(k,v) if k<4 else (k,D.string(v)) for k,v,*_ in D.values(e,p)]==[(int(S3.f_k[f]),int(S3.f_v[f])) if int(S3.f_k[f])<4 else (int(S3.f_k[f]),S3.text(int(S3.f_v[f]))) for f in S3.facts(e,[p])]
    assert D.ents('paris')==S3.ents('paris') and D.ents('abraham lincoln')==S3.ents('abraham lincoln')
def test_piece_nonces():
    D=D1.drive2();p=D.pieces('abraham lincoln')
    assert p and all(0<=x<83521 for x in p) and D.key('abraham lincoln')==D.keys_many(['x','abraham lincoln'])[1]
