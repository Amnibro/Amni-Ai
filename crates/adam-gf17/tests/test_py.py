import os,sys,time,numpy as np
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))
sys.path[:0]=[os.environ.get('ADAM_GF17_SO_DIR',ROOT),ROOT]
import adam_gf17
from amni.compute.reffelt4 import encode_fp16_to_rgba4,decode_rgba4_to_fp16
PRE=os.environ.get('ADAM_GF17_PREFIX',os.path.join(os.environ.get('AMNI_HOME',ROOT),'exports/gf17_continuum/rayhash_clean1v2_base_v18_s100k'))
OUT=os.environ.get('ADAM_GF17_OUT',os.path.join(ROOT,'crates/target/gf17/py_test.gf17'))
def main():
    if not os.path.exists(PRE+'_sid.npy'):return print('skip: no pack at',PRE)
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    t=time.time();st=adam_gf17.pack(PRE,OUT,budget_mb=256);print('pack %.2fs'%(time.time()-t),st.tables)
    ref={k:np.load('%s_%s.npy'%(PRE,k)) for k in ('sid','sw','dp','bias')}
    got=adam_gf17.unpack(st)
    for k,a in ref.items():
        b=got[k];assert b.dtype==a.dtype and b.shape==a.shape and np.array_equal(a.view(np.uint8),b.view(np.uint8)),k
        if a.dtype==np.float16:
            x,y=a.astype(np.float64).ravel(),b.astype(np.float64).ravel();m=np.isfinite(x)&np.isfinite(y);c=float(x[m]@y[m]/np.sqrt((x[m]@x[m])*(y[m]@y[m])));assert c==1.0,(k,c);print(k,'bit-exact, cos',c)
        else:print(k,'bit-exact')
    L,T,S=ref['sw'].shape;g=np.random.default_rng(0);l,c,s=g.integers(0,L,1000),g.integers(0,T,1000),g.integers(0,S,1000)
    assert all(st.fetch(int(a),int(b),int(d))==(int(ref['sid'][a,b,d]),float(ref['sw'][a,b,d])) for a,b,d in zip(l,c,s))
    ids,ws=st.fetch_many(l.tolist(),c.tolist(),s.tolist());assert np.array_equal(ids,ref['sid'][l,c,s]) and np.array_equal(ws.view(np.uint16),ref['sw'][l,c,s].view(np.uint16))
    pg=st.page('sw',0,0);n=ref['sw'][0].size;assert pg.shape[1:]==(4096,4) and np.array_equal(pg.reshape(-1,4)[:n],encode_fp16_to_rgba4(ref['sw'][0].ravel())) and np.array_equal(decode_rgba4_to_fp16(pg.reshape(-1,4)[:n]).view(np.uint16),ref['sw'][0].ravel().view(np.uint16))
    print('fetch/fetch_many exact; sw page 0 RGBA == reffelt4.encode_fp16_to_rgba4')
    co=adam_gf17.unpack(st,tiers=1);assert np.array_equal(co['sid'],ref['sid']) and np.array_equal(co['bias'].view(np.uint32),ref['bias'].view(np.uint32))
    print('d3-only: exact tables exact, sw finite %.4f, stats'%np.isfinite(co['sw']).mean(),st.stats)
    raw_pg,raw_co,rsz=pg,co,st.size;del st;os.remove(OUT)
    t=time.time();sc=adam_gf17.pack(PRE,OUT,budget_mb=256,tile=128);print('coded pack %.2fs tile %d size %d (%.3fx raw page file, %.3fx npy) streams %s'%(time.time()-t,sc.tile,sc.size,sc.size/rsz,sc.size/sum(a.nbytes for a in ref.values()),sc.streams))
    assert sc.tile==128 and sc.size<rsz and sum(sc.streams)<sc.size
    got=adam_gf17.unpack(sc)
    assert all(got[k].dtype==a.dtype and got[k].shape==a.shape and np.array_equal(a.view(np.uint8),got[k].view(np.uint8)) for k,a in ref.items())
    assert all(sc.fetch(int(a),int(b),int(d))==(int(ref['sid'][a,b,d]),float(ref['sw'][a,b,d])) for a,b,d in zip(l,c,s))
    sc.cold(0);ids,ws=sc.fetch_many(l.tolist(),c.tolist(),s.tolist());assert np.array_equal(ids,ref['sid'][l,c,s]) and np.array_equal(ws.view(np.uint16),ref['sw'][l,c,s].view(np.uint16))
    assert np.array_equal(sc.page('sw',0,0),raw_pg) and all(np.array_equal(v.view(np.uint8),raw_co[k].view(np.uint8)) for k,v in adam_gf17.unpack(sc,tiers=1).items())
    print('coded: bit-exact unpack, fetch/fetch_many exact, RGBA page == raw page, d3-only == raw d3-only, stats',sc.stats)
    del sc;b=bytearray(open(OUT,'rb').read())
    b[len(b)//2]^=0x10;open(OUT,'wb').write(b);sc=adam_gf17.Store(OUT)
    try:adam_gf17.unpack(sc);raise SystemExit('corruption not detected')
    except IOError as x:print('corruption detected:',str(x)[:120])
    del sc;os.remove(OUT)
main()
