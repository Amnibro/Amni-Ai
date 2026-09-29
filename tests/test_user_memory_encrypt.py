import os,sys,stat,json,hashlib
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from amni.serve import user_memory as U,user_toroid as T,at_rest as AR
from amni.storage.local_profile import LocalProfile
from amni.storage.conversation_notes import ConversationNotes
from amni.storage.personal_atlas import PersonalAtlas
from amni.inference.semantic_ptex_lut import SemanticPTEXLUT
SEED=("My sister lives in Denver.","I work as a nurse.","My brother drives a Volvo.")
@pytest.fixture
def env(tmp_path,monkeypatch):
    monkeypatch.setenv('AMNI_UMEM_DIR',str(tmp_path/'umem'));monkeypatch.delenv('AMNI_UMEM_KEY_DIR',raising=False);monkeypatch.delenv('AMNI_USER_MEMORY',raising=False);[monkeypatch.setenv(f'AMNI_UMEM_{k}','1') for k in ('LT','GN','G2')];U._cache.clear();U._EID.clear();U._rcache.clear();return tmp_path
def _seed(u):
    [U.turn(m,user=u) for m in SEED];k=U.user_key(u);U._cache.clear();return k
def _vals(k):return sorted(f.val.decode() for f in U.load(k))
def _disk(d):return [open(os.path.join(r,f),'rb').read() for r,_,fs in os.walk(d) for f in fs]
def _hdr(p,m):b=open(p,'rb').read(6);return b[:4]==m and b[4:6]==b'\x02\x00'
def test_umem_roundtrip_encrypted_private_and_no_plaintext(env):
    k=_seed('r1');p=U.path(k);kp=U.kpath(k);v=_vals(k);assert v==['Denver','Volvo','nurse'] and _hdr(p,b'AMUM')
    assert stat.S_IMODE(os.stat(kp).st_mode)==0o600 and stat.S_IMODE(os.stat(U.kdir()).st_mode)==0o700 and stat.S_IMODE(os.stat(p).st_mode)==0o600 and len(open(kp,'rb').read())==32
    leg=U.encode(U.load(k));kf=open(kp,'rb').read();assert not any(x in b for b in _disk(env) for x in (b'Denver',b'Volvo',b'nurse',leg[-128:],leg[:48]) if b!=kf) and not any(kf in b for b in _disk(env) if b!=kf)
    raw=open(p,'rb').read();U._cache.clear();U.turn("My mom lives in Leeds.",user='r1');assert open(p,'rb').read()[16:44]!=raw[16:44] and 'Leeds' in _vals(k)
def test_umem_tamper_detected_file_untouched(env):
    k=_seed('t1');p=U.path(k);raw=open(p,'rb').read()
    for o in (0,5,8,20,36,50,len(raw)//2,len(raw)-1):
        b=bytearray(raw);b[o]^=1;open(p,'wb').write(bytes(b));U._cache.clear()
        with pytest.raises(ValueError):U.load(k)
        assert open(p,'rb').read()==bytes(b)
    open(p,'wb').write(raw[:40]);U._cache.clear()
    with pytest.raises(ValueError):U.load(k)
    open(p,'wb').write(raw);U._cache.clear();assert _vals(k)==['Denver','Volvo','nurse']
def test_umem_wrong_key_missing_key_and_cross_user(env):
    k=_seed('w1');k2=_seed('w2');p=U.path(k);kp=U.kpath(k);kf=open(kp,'rb').read();raw=open(p,'rb').read()
    open(kp,'wb').write(os.urandom(32));U._cache.clear()
    with pytest.raises(ValueError):U.load(k)
    os.remove(kp);U._cache.clear()
    with pytest.raises(ValueError):U.load(k)
    open(kp,'wb').write(kf);os.chmod(kp,0o600);open(U.path(k2),'wb').write(raw);U._cache.clear()
    with pytest.raises(ValueError):U.load(k2)
    with pytest.raises(ValueError):U._V.unseal(raw,U._V.uk(kf,k2))
    with pytest.raises(ValueError):T.unseal(raw,T._uk(kf,k))
    assert _vals(k)==['Denver','Volvo','nurse']
def test_umem_strict_key_permissions(env):
    k=_seed('s1');os.chmod(U.kpath(k),0o644);U._cache.clear()
    with pytest.raises(PermissionError):U.load(k)
    os.chmod(U.kpath(k),0o600);os.chmod(U.kdir(),0o755);U._cache.clear()
    with pytest.raises(PermissionError):U.load(k)
    os.chmod(U.kdir(),0o700);U._cache.clear();assert 'Denver' in _vals(k)
def test_umem_plaintext_migrates_and_is_scrubbed(env):
    k=_seed('m1');fs=U.load(k);leg=U.encode(fs);p=U.path(k);U.wipe(k);open(p,'wb').write(leg);os.link(p,str(env/'old_inode'));U._cache.clear()
    assert not os.path.exists(U.kpath(k)) and _vals(k)==['Denver','Volvo','nurse'] and _hdr(p,b'AMUM') and os.path.exists(U.kpath(k))
    old=open(env/'old_inode','rb').read();assert len(old)==len(leg) and old!=leg;os.remove(env/'old_inode')
    assert not any(x in b for b in _disk(env) for x in (leg[-128:],leg[:48],b'Denver',b'nurse')) and U.decode(U._V.unseal(open(p,'rb').read(),U._V.uk(open(U.kpath(k),'rb').read(),k)))==fs
    U._cache.clear();assert _vals(k)==['Denver','Volvo','nurse'] and _hdr(p,b'AMUM')
def test_umem_forget_rotates_key_and_shreds_old(env):
    k=_seed('f1');p=U.path(k);kp=U.kpath(k);old=open(p,'rb').read();kf=open(kp,'rb').read();os.link(kp,str(env/'old_key'));os.link(p,str(env/'old_ct'))
    r=U.turn("Forget what I told you about my brother.",user='f1');assert 'brother' in r['reply'];U._cache.clear()
    nk=open(kp,'rb').read();assert nk!=kf and _vals(k)==['Denver','nurse'] and not os.path.exists(kp+'.new')
    assert open(env/'old_key','rb').read()!=kf and open(env/'old_ct','rb').read()!=old
    with pytest.raises(ValueError):U._V.unseal(old,U._V.uk(nk,k))
    with pytest.raises(ValueError):U._V.unseal(open(p,'rb').read(),U._V.uk(kf,k))
    k2=_seed('f2');kf2=open(U.kpath(k2),'rb').read();U.forget(k2,99999);U._cache.clear();assert open(U.kpath(k2),'rb').read()==kf2
def test_umem_forget_all_shreds_store_and_keys_even_if_undecryptable(env):
    k=_seed('a1');p=U.path(k);kp=U.kpath(k);assert U.turn("Forget everything.",user='a1')['reply'].startswith("Okay") and not os.path.exists(p) and not os.path.exists(kp)
    k=_seed('a2');p=U.path(k);open(p,'r+b').write(b'AMUM\x02\x00junk');U._cache.clear()
    with pytest.raises(ValueError):U.load(k)
    r=U.turn("Delete everything you know about me.",user='a2');assert "everything" in r['reply'] and not os.path.exists(p) and not os.path.exists(U.kpath(k)) and _vals(k)==[]
def test_interrupted_rotation_recovers(env):
    k=_seed('i1');kp=U.kpath(k);fs=U.load(k);kf=os.urandom(32);AR.wfile(kp+'.new',kf);AR.wfile(U.path(k),U._V.seal(U.encode(fs),U._V.uk(kf,k)));U._cache.clear()
    assert _vals(k)==['Denver','Volvo','nurse'] and open(kp,'rb').read()==kf and not os.path.exists(kp+'.new')
def test_shared_helper_and_separate_keys(env,monkeypatch):
    monkeypatch.setenv('AMNI_UMEM_TOROID','1');k=_seed('h1');T.turn("My car is red.",user='h1');T._cache.clear()
    assert T.kdir is U.kdir and type(T._V) is type(U._V) is AR.Vault and os.path.exists(T.kpath(k)) and T.kpath(k)!=U.kpath(k) and open(T.kpath(k),'rb').read()!=open(U.kpath(k),'rb').read()
    T.wipe(k);assert _vals(k)==['Denver','Volvo','nurse'] and not os.path.exists(T.kpath(k))
    with pytest.raises(ValueError):AR.Vault('umem',b'AMUM',1).unseal(T.seal(b'x'*40,U._V.uk(b'k'*32,k)),U._V.uk(b'k'*32,k))
def test_profile_and_notes_encrypted_roundtrip_forget_migration(env):
    pp=env/'exp'/'user_profile.json';lp=LocalProfile(str(pp),fact_re=None);lp.data={'name':'Zebulon','location':'Ottawa'};lp.save()
    assert _hdr(pp,b'AMAT') and LocalProfile(str(pp),fact_re=None).data=={'name':'Zebulon','location':'Ottawa'} and not any(b'Zebulon' in b or b'Ottawa' in b for b in _disk(env))
    kp=AR.ATLAS.fkey(str(pp))[0];assert kp==str(env/'exp'/'keys'/'user_profile.json.key') and stat.S_IMODE(os.stat(kp).st_mode)==0o600;kf=open(kp,'rb').read();old=open(pp,'rb').read()
    lp.forget();assert open(kp,'rb').read()!=kf and LocalProfile(str(pp),fact_re=None).data=={}
    with pytest.raises(ValueError):AR.ATLAS.unseal(old,AR.ATLAS.uk(open(kp,'rb').read(),'user_profile.json'))
    np_=env/'exp'/'conversation_notes.json';cn=ConversationNotes(str(np_));cn.add_note('call me Captain Quahog');assert _hdr(np_,b'AMAT') and ConversationNotes(str(np_)).data['notes'][0]['text']=='call me Captain Quahog' and not any(b'Quahog' in b for b in _disk(env))
    js=json.dumps({'corrections':[],'notes':[{'ts':1,'text':'legacy Xylophone'}],'behavior_prefs':[]}).encode();open(np_,'wb').write(js);os.link(np_,str(env/'old_notes'))
    assert ConversationNotes(str(np_)).data['notes'][0]['text']=='legacy Xylophone' and _hdr(np_,b'AMAT') and open(env/'old_notes','rb').read()!=js;os.remove(env/'old_notes')
    assert not any(b'Xylophone' in b for b in _disk(env));open(pp,'r+b').write(b'AMAT\x02\x00zz');assert LocalProfile(str(pp),fact_re=None).data=={}
def test_atlas_env_key_dir_is_namespaced(env,monkeypatch):
    monkeypatch.setenv('AMNI_UMEM_KEY_DIR',str(env/'kd'));a,b=env/'x'/'user_profile.json',env/'y'/'user_profile.json'
    la,lb=LocalProfile(str(a),None),LocalProfile(str(b),None);la.data={'n':'A'};la.save();lb.data={'n':'B'};lb.save()
    assert AR.ATLAS.fkey(str(a))[0]!=AR.ATLAS.fkey(str(b))[0] and os.path.dirname(AR.ATLAS.fkey(str(a))[0])==str(env/'kd'/'atlas') and LocalProfile(str(a),None).data=={'n':'A'} and LocalProfile(str(b),None).data=={'n':'B'}
def _enc(qs):return np.stack([np.frombuffer(hashlib.sha256(q.encode()).digest()+hashlib.sha256(q.encode()+b'1').digest(),np.uint8).astype(np.float32) for q in qs])
FACTS=("user likes Marisol brand tea","user takes lisinopril daily","user's favorite band is Wolfmother","user lives near Kelowna","user's dog is called Biscuit","user works at Hollowbrook Labs")
def test_personal_atlas_encrypted_roundtrip_forget(env):
    r=env/'pa';A=PersonalAtlas(root=str(r),encoder=_enc,enable_worker=False);[A.record_direct(f,source_excerpt=f) for f in FACTS];A._pending=[{'id':'p1','fact':"user's sister is Ingrid"}];A._save_pending();A._dead_letter({'kind':'x','message':'secret Tobermory'})
    fs=sorted(p.name for p in r.iterdir() if p.is_file());assert fs==['dead_letter.jsonl','pending_clarifications.jsonl','personal___local_user__.json','personal___local_user__.meta.json','personal___local_user__.npz'] and all(_hdr(r/f,b'AMAT') for f in fs)
    assert not any(w in b for b in _disk(env) for w in (b'Marisol',b'lisinopril',b'Ingrid',b'Tobermory',b'PK\x03\x04')) and stat.S_IMODE(os.stat(r/'keys').st_mode)==0o700
    B=PersonalAtlas(root=str(r),encoder=_enc,enable_worker=False);assert sorted(x['fact'] for x in B.list_facts())==sorted(FACTS) and B._pending[0]['fact']=="user's sister is Ingrid" and sorted(q for q,_ in B._slot.lut._raw)==sorted(FACTS) and B.recall(FACTS[0],k=6,max_radius=99)
    assert AR.ATLAS.get(str(r/'dead_letter.jsonl')).count(b'Tobermory')==1;B._dead_letter({'kind':'y'});assert AR.ATLAS.get(str(r/'dead_letter.jsonl')).count(b'\n')==2
    kp=AR.ATLAS.fkey(str(r/'personal___local_user__.meta.json'))[0];kf=open(kp,'rb').read();old=open(r/'personal___local_user__.meta.json','rb').read()
    assert B.forget('lisinopril')['forgot']==1 and open(kp,'rb').read()!=kf
    with pytest.raises(ValueError):AR.ATLAS.unseal(old,AR.ATLAS.uk(open(kp,'rb').read(),'personal___local_user__.meta.json'))
    C=PersonalAtlas(root=str(r),encoder=_enc,enable_worker=False);assert len(C.list_facts())==5 and not any('lisinopril' in q for q,_ in C._slot.lut._raw)
    kf=open(kp,'rb').read();C.forget('nomatchzz');assert open(kp,'rb').read()==kf
    C.forget(forget_all=True);D=PersonalAtlas(root=str(r),encoder=_enc,enable_worker=False)
    assert D.list_facts()==[] and D._pending==[] and D._slot.lut._raw==[] and not (r/'personal___local_user__.npz').exists() and not (r/'dead_letter.jsonl').exists() and not any(w in b for b in _disk(env) for w in (b'Marisol',b'Ingrid',b'Tobermory'))
def test_personal_atlas_migrates_plaintext(env):
    r=env/'pb';r.mkdir();lut=SemanticPTEXLUT(encoder=_enc);[lut.add(f,f) for f in FACTS];lut.fit();lut.save(str(r/'personal___local_user__'))
    assert (r/'personal___local_user__.npz').read_bytes()[:2]==b'PK' and b'Marisol' in (r/'personal___local_user__.json').read_bytes()
    (r/'personal___local_user__.meta.json').write_text(json.dumps({f'{f}::{f}':{'ts':1,'is_confidential':True,'conf':1.0,'source':f} for f in FACTS}));(r/'pending_clarifications.jsonl').write_text(json.dumps({'id':'p','fact':'Ingrid'})+'\n')
    A=PersonalAtlas(root=str(r),encoder=_enc,enable_worker=False);assert sorted(x['fact'] for x in A.list_facts())==sorted(FACTS) and A._pending==[{'id':'p','fact':'Ingrid'}] and sorted(q for q,_ in A._slot.lut._raw)==sorted(FACTS)
    assert all(_hdr(p,b'AMAT') for p in r.iterdir() if p.is_file()) and not any(w in b for b in _disk(env) for w in (b'Marisol',b'Ingrid',b'PK\x03\x04'))
def test_lut_without_vault_unchanged(tmp_path):
    lut=SemanticPTEXLUT(encoder=_enc);[lut.add(f,f) for f in FACTS];lut.fit();lut.save(str(tmp_path/'x'));assert (tmp_path/'x.npz').read_bytes()[:2]==b'PK' and json.loads((tmp_path/'x.json').read_text())['pairs'][0][0]==FACTS[0] and not (tmp_path/'keys').exists()
    assert SemanticPTEXLUT.load(str(tmp_path/'x'),encoder=_enc)._raw==[(f,f) for f in FACTS]
