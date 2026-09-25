import os,sys,json,time,shutil,hashlib,resource
import pyarrow.parquet as pq
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))));sys.path.insert(0,_H0)
from amni.compute.ks_codec import KSCodec,KStore,write_store,paginate
TMP=os.path.join(_H0,"crates","adam-ks","tmp");SRC=sys.argv[1] if len(sys.argv)>1 else os.path.join(TMP,"heldout1000.parquet");VER=os.environ.get("KS_VER","ks2-clean1");NPY=int(os.environ.get("KS_PY_PAGES","1000"))
sha=lambda f:hashlib.file_digest(open(f,"rb"),"sha256").hexdigest()
def timed(f,*a):t=time.perf_counter();r=f(*a);return r,time.perf_counter()-t
def main():
 os.makedirs(TMP,exist_ok=True);R=[r for b in pq.ParquetFile(SRC).iter_batches(batch_size=2048) for r in b.to_pylist()];P=[p for r in R for p in paginate(r["text"].encode())];L=[len(p) for p in P];nb=sum(L)
 rs,py=KSCodec(ver=VER,arch="cpu",ks="rust"),KSCodec(ver=VER,arch="cpu",ks="python");assert rs.rs() is not None and py.rs() is None and rs.ident()==py.ident()==json.loads(rs.rs().ident())
 rs.encode([b"warm up\n"]);py.encode([b"warm up\n"]);cr,te=timed(rs.encode,P);dr,td=timed(rs.decode,L,cr);cp,tpe=timed(py.encode,P[:NPY]);dp,tpd=timed(py.decode,L[:NPY],cr[:NPY]);xr=rs.decode(L[:NPY],cp)
 out={"pages":len(P),"src_bytes":nb,"coder":rs.ident()["coder"],"code_bpb_rust":8*sum(map(len,cr))/nb,"py_pages":NPY,"code_bpb_python":8*sum(map(len,cp))/sum(L[:NPY]),"codes_identical":sum(a==b for a,b in zip(cr,cp))/NPY,"roundtrip_rust":sum(a==b for a,b in zip(dr,P))/len(P),"cross_py_decodes_rust":sum(a==b for a,b in zip(dp,P))/NPY,"cross_rust_decodes_py":sum(a==b for a,b in zip(xr,P))/NPY,"rust_threads":rs.rs().threads,"rust_enc_Bps":nb/te,"rust_dec_Bps":nb/td,"py_enc_Bps":sum(L[:NPY])/tpe,"py_dec_Bps":sum(L[:NPY])/tpd}
 so,sp=os.path.join(TMP,"parity_rs"),os.path.join(TMP,"parity_py");[shutil.rmtree(x,ignore_errors=True) for x in (so,sp)]
 write_store(so,SRC,rs);docs=({"src":"wikipedia","title":r["title"],"text":r["text"].encode(),"tier":"endorsed","score":r["score"],"stale":r["stale"],"date":r["date"],"evidence":{"urls":[r["url"]],"agree":1,"refs":r["refs"],**({"license":r["license"]} if "license" in r else {}),"signals":{},"code":None}} for r in R);write_store(sp,docs,py)
 out["store_files_identical"]={f:sha(os.path.join(so,f))==sha(os.path.join(sp,f)) for f in ("pages.bin","dir.ptex","meta.json")}|{f:pq.read_table(os.path.join(so,f)).equals(pq.read_table(os.path.join(sp,f))) for f in ("docs.parquet","pagemeta.parquet")};out["pages_sha256"]=sha(os.path.join(so,"pages.bin"))
 n=min(NPY,len(P));out["kstore_rust_store_python_decoder"]=sum(a==b for a,b in zip(KStore(so,py).get(list(range(n))),P[:n]))/n;out["kstore_python_store_rust_decoder"]=sum(a==b for a,b in zip(KStore(sp,rs).get(list(range(len(P)))),P))/len(P);out["peak_rss_gb"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
 [shutil.rmtree(x,ignore_errors=True) for x in (so,sp)];print(json.dumps(out,indent=1));return out
if __name__=="__main__":o=main();sys.exit(0 if o["codes_identical"]==1 and o["roundtrip_rust"]==1 and o["cross_py_decodes_rust"]==1 and o["cross_rust_decodes_py"]==1 and all(o["store_files_identical"].values()) else 1)
