import os,sys,json,zlib,hashlib,random,subprocess
import numpy as np,pyarrow as pa,pytest
from amni.compute.ks_codec import KSCodec,KStore,write_store,paginate,table,PAGE,TOT,CODERS
_H0=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT=("Adam stores coordinates, not numbers. "*40+"\n\n").encode()+"Ünïcödé ✓ 漢字テキスト Ελληνικά עברית 🚀🧪\n\n".encode()*20+open(os.path.join(_H0,"amni","compute","ks_codec.py"),"rb").read()[:6000]
PAGES=[b"",b"x",b"\n",TEXT[:PAGE],"—ñ€𝄞".encode()*200,open(os.path.join(_H0,"amni","compute","ray_qa.py"),"rb").read()[:3000],bytes(random.Random(7).getrandbits(8) for _ in range(1500))]+paginate(TEXT)
SNIP="import hashlib;from amni.compute.ks_codec import KSCodec;from tests.test_ks_codec import PAGES;print(hashlib.sha256(b''.join(KSCodec(ver='ks1-v16',arch='cpu').encode(PAGES))).hexdigest())"
@pytest.fixture(scope="module")
def codec():return KSCodec(ver="ks1-v16",arch="cpu")
def test_paginate_exact():
 ps=paginate(TEXT);assert b"".join(ps)==TEXT and all(0<len(p)<=PAGE for p in ps) and paginate(b"")==[b""]
def test_table_integer_deterministic():
 lg=np.random.default_rng(0).normal(0,4,(8,256));a=table(lg);b=table(lg.copy())
 assert a.dtype==np.int64 and np.array_equal(a,b) and (a>=1).all() and (a.sum(axis=1)<=TOT).all()
def test_roundtrip_varied(codec):
 cs=codec.encode(PAGES);assert codec.decode([len(p) for p in PAGES],cs)==PAGES and cs[0]==b"" and len(cs[3])<len(PAGES[3])
def test_batch_independent(codec):
 cs=codec.encode(PAGES);assert [codec.encode([p])[0] for p in PAGES[:6]]==cs[:6] and codec.encode(PAGES[::-1])[::-1]==cs
def test_deterministic_across_runs(codec):
 h=hashlib.sha256(b"".join(codec.encode(PAGES))).hexdigest();env=dict(os.environ,PYTHONPATH=_H0,AMNI_RAY_ARCH="cpu",TI_CPU_MAX_NUM_THREADS="4")
 assert subprocess.run([sys.executable,"-c",SNIP],cwd=_H0,env=env,capture_output=True,text=True,check=True).stdout.split()[-1]==h
def _store(tmp,codec):
 out=str(tmp/"ks");write_store(out,[{"src":"t","title":f"doc{i}","text":p,"tier":"endorsed","score":0.7,"evidence":{"urls":["u"]}} for i,p in enumerate([TEXT,b"short page",b""])],codec);return out
def test_store_roundtrip(tmp_path,codec):
 st=KStore(_store(tmp_path,codec),codec);assert b"".join(st.get(list(range(len(st)))))==TEXT+b"short page" and [st.title[st.doc_of(len(st)-k)] for k in (2,1)]==["doc1","doc2"] and set(st.pm["tier"].to_pylist())=={1}
@pytest.mark.parametrize("how",["payload","digit","truncate","magic"])
def test_corrupt_index_guard(tmp_path,codec,how):
 out=_store(tmp_path,codec);f=os.path.join(out,"dir.ptex");raw=bytearray(open(f,"rb").read());n=8+int.from_bytes(raw[4:8],"little")
 if how=="payload":raw[n+5]^=0x55;open(f,"wb").write(raw)
 if how=="magic":raw[0]^=1;open(f,"wb").write(raw)
 if how=="truncate":pb=os.path.join(out,"pages.bin");d=open(pb,"rb").read();open(pb,"wb").write(d[:-3])
 if how=="digit":
  hd=json.loads(raw[8:n]);d=bytearray(pa.decompress(bytes(raw[n:]),hd["raw"],codec="zstd",asbytes=True));d[2]=(d[2]+1)%17;pay=pa.compress(bytes(d),codec="zstd",asbytes=True);hd["crc32"]=zlib.crc32(pay);h=json.dumps(hd).encode();open(f,"wb").write(bytes(raw[:4])+len(h).to_bytes(4,"little")+h+pay)
 with pytest.raises(ValueError,match="corrupt"):KStore(out,codec)
def test_frozen_coder_guard(tmp_path,codec):
 out=_store(tmp_path,codec);m=json.load(open(os.path.join(out,"meta.json")));m["coders"][0]["pack_id"]="0"*32;json.dump(m,open(os.path.join(out,"meta.json"),"w"))
 with pytest.raises(ValueError,match="frozen-coder"):KStore(out,codec).get([0])
def test_coder_version_guard(tmp_path,codec):
 ks2=KSCodec(ver="ks2-clean1",arch="cpu");lx=KSCodec(pack=CODERS["ks1-v16"][0],lex="clean1",arch="cpu");a,b,c=codec.ident(),ks2.ident(),lx.ident();out=_store(tmp_path,codec)
 assert a["ver"]=="ks1-v16" and b["ver"]=="ks2-clean1" and b["lex"]=="clean1" and len({a["pack_id"],b["pack_id"],c["pack_id"]})==3 and a["lex_id"]!=b["lex_id"]==c["lex_id"] and a["coder"]!=b["coder"]
 for w in (ks2,lx):
  with pytest.raises(ValueError,match="frozen-coder"):KStore(out,w).get([0])
 st=KStore(out,[ks2,codec]);assert b"".join(st.get(list(range(len(st)))))==TEXT+b"short page"
 o2=str(tmp_path/"ks2");write_store(o2,[{"src":"t","title":"d","text":TEXT}],ks2);assert json.load(open(os.path.join(o2,"meta.json")))["coders"][0]["lex"]=="clean1" and b"".join(KStore(o2,ks2).get(list(range(len(KStore(o2,ks2))))))==TEXT
 with pytest.raises(ValueError,match="frozen-coder"):KStore(o2,codec).get([0])
def test_default_coder_is_clean(tmp_path,monkeypatch):
 import amni.compute.ks_retrieve as KR
 assert KSCodec(arch="cpu").ver=="ks2-clean1" and KSCodec(arch="cpu").lex=="clean1" and KR.STORE=="store_clean"
 o=str(tmp_path/"d");write_store(o,[{"src":"t","title":"d","text":TEXT}],KSCodec(arch="cpu"));assert b"".join(KStore(o).get(list(range(len(KStore(o))))))==TEXT
