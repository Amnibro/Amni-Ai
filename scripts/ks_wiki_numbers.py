import os,re,sys,json,glob,time,runpy,shutil,ctypes,random,hashlib,argparse,tempfile,subprocess,collections,multiprocessing as mp
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
H=os.environ.get("AMNI_HOME") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path[:0]=[H,os.path.join(H,"scripts")]
from ks_download import fwtext
from ks_ingest import clean
from amni.compute import decontam as DC
from amni.compute.ks_codec import paginate
import amni.compute.ks_retrieve as KR
from amni.serve.adam_t_client import _HOLE,_MO
KS=os.path.join(H,"data","ks");W,NW="wikipedia","wikipedia_fw";FW="HuggingFaceFW/finewiki";REV=os.environ.get("KS_FW_REV","8bd13e72e6a002407649b3e898535f42ceb1aeb9");RS,NS=41,56;RAWD=os.path.join(KS,"raw",W,"finewiki_resource");ST=os.path.join(KS,"state_fw");DLOG=os.path.join(KS,"logs","deletions.log");BIN=os.environ.get("KS_RKNOW_BIN") or os.path.join(H,"crates","target","rknow","release","adam-rknow");ZC=pa.Codec("zstd",compression_level=3)
TBL=re.compile(r"(?m)^[ \t]*\|.*\|[ \t]*(?:\n|$)");SP=re.compile(r"[^\n]*?[.!?](?=\s|$)|[^\n]+");BROAD=re.compile(r"[^\W_]  +[^\W_]|[^\W_] ,");GLUE=re.compile(r"\b(?:1\d{3}|20\d\d)\d{1,2} (?:%s)\b"%_MO)
SCH=pa.schema([("title",pa.string()),("text",pa.string()),("url",pa.string()),("date",pa.string()),("score",pa.float64()),("stale",pa.bool_()),("refs",pa.int64()),("license",pa.string()),("held",pa.bool_()),("text_src",pa.string()),("dc_text",pa.string()),("dc_n",pa.int32()),("dc_groups",pa.string())])
SRC={"finewiki":f"HuggingFaceFW/finewiki data/enwiki @ {REV} (rendered HTML, templates expanded), CC BY-SA 4.0","wikimedia_20231101":"wikimedia/wikipedia 20231101.en (templates dropped), CC BY-SA 3.0; kept where FineWiki has no page with that title or its FineWiki text fails the KS hygiene/filter","finewiki_prev":"HuggingFaceFW/finewiki enwiki (earlier ingest, shards 00041-00055, unchanged text), CC BY-SA 4.0"}
ATTR="Text from English Wikipedia (https://en.wikipedia.org), by its contributors, licensed CC BY-SA 4.0 (older revisions CC BY-SA 3.0). Obtained via the Hugging Face datasets HuggingFaceFW/finewiki (enwiki, revision %s) and wikimedia/wikipedia (20231101.en). Changes: markdown tables removed from FineWiki text, KS neutral filter and hygiene, trailing reference sections cut, e-mail and phone numbers masked, sentences sharing a 13-gram with an eval set removed (amni.compute.decontam). Per-article title, URL and license: data/ks/clean/wikipedia_fw/*.parquet."%REV
def p(s):print(time.strftime("%H:%M:%S"),s,file=sys.stderr,flush=True)
def jw(f,o):os.makedirs(os.path.dirname(f),exist_ok=True);json.dump(o,open(f+".tmp","w"),indent=1);os.replace(f+".tmp",f)
def jr(f,d=None):return json.load(open(f)) if os.path.exists(f) else d
def free()->float:return shutil.disk_usage(KS).free/2**30
def du(d:str)->int:return sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(d) for f in fs if not os.path.islink(os.path.join(r,f)))
def dlog(m:str):z=time.strftime("%z");open(DLOG,"a").write(time.strftime("%Y-%m-%dT%H:%M:%S")+z[:3]+":"+z[3:]+" WIKI-NUMBERS agent: "+m+"\n");p("deletion logged: "+m)
def hh(y:str)->int:return int.from_bytes(hashlib.blake2b(re.sub(r"[^a-z0-9]+"," ",y.lower()).strip().encode(),digest_size=8).digest(),"little",signed=True)
def gnames(g:int)->list:return [n for j,n in enumerate(DC._ix().groups) if g>>j&1]
def heldsha(d:str)->tuple:T=sorted(f"{k[-13:-8]}\t{t}" for k in sorted(glob.glob(os.path.join(d,"[0-9]"*5+".parquet"))) for t,h in zip(*pq.read_table(k,columns=["title","held"]).to_pydict().values()) if h);return len(T),hashlib.sha256("\n".join(T).encode()).hexdigest()
def cmd_dl(a):
 from huggingface_hub import HfApi,hf_hub_download
 sf=os.path.join(RAWD,"state.json");st=jr(sf,{"rev":REV,"files":[],"rows":0,"matched":0,"bytes":0});assert st["rev"]==REV,st["rev"];fs=sorted(x.rfilename for x in HfApi().dataset_info(FW,revision=REV,files_metadata=True).siblings if x.rfilename.startswith("data/enwiki/"))
 T={t:s for s in range(RS) for t in pq.read_table(os.path.join(KS,"clean",W,f"{s:05d}.parquet"),columns=["title"]).column(0).to_pylist()};p(f"{len(T):,} titles in shards 00000-{RS-1:05d}, {len(fs)} enwiki files at {REV[:8]}, {len(st['files'])} done")
 for fi,fn in enumerate(fs):
  if fn in st["files"]:continue
  free()<100 and sys.exit(p(f"stop: {free():.0f} GiB free"))
  for i in range(6):
   try:lf=hf_hub_download(FW,fn,repo_type="dataset",revision=REV,local_dir=os.path.join(RAWD,"dl"));break
   except Exception as e:p(f"retry {fn}: {type(e).__name__}: {str(e)[:160]}");time.sleep(30*(i+1))
  else:sys.exit(p(f"download failed: {fn}"))
  B=collections.defaultdict(list);n=0;sz=os.path.getsize(lf)
  for b in pq.ParquetFile(lf).iter_batches(batch_size=2048,columns=["page_id","title","text","url","date_modified"]):
   D=b.to_pydict();n+=len(D["title"]);[B[T[t]].append((i,t,re.sub(r"\n{3,}","\n\n",TBL.sub("",fwtext(t,x))).strip(),u or KR.wurl(t),(d or "")[:10])) for i,t,x,u,d in zip(*D.values()) if t in T]
  for s,R in B.items():o=os.path.join(RAWD,"parts",f"{s:05d}",f"{fi:02d}.parquet");os.makedirs(os.path.dirname(o),exist_ok=True);pq.write_table(pa.table({k:[r[j] for r in R] for j,k in enumerate(("page_id","title","text","url","date"))}),o+".tmp",compression="zstd");os.replace(o+".tmp",o)
  st["files"].append(fn);st["rows"]+=n;st["matched"]+=sum(map(len,B.values()));st["bytes"]+=sz;jw(sf,st);os.remove(lf);p(f"{fn}: {n:,} rows, {sum(map(len,B.values())):,} matched, parts {du(os.path.join(RAWD,'parts'))/2**30:.1f} GiB, free {free():.0f} GiB")
 shutil.rmtree(os.path.join(RAWD,"dl"),ignore_errors=True);st.get("dl_logged") or (dlog(f"deleted the {len(st['files'])} downloaded FineWiki enwiki parquet files ({st['bytes']/2**30:.1f} GiB, HuggingFaceFW/finewiki@{REV[:8]}) under data/ks/raw/wikipedia/finewiki_resource/dl, each right after its matched rows were written to finewiki_resource/parts ({st['matched']:,} of {st['rows']:,} rows)"),st.update(dl_logged=True),jw(sf,st))
def scrub(y:str)->tuple:
 ix=DC._ix();yl=y.lower();M=[m.span() for m in DC.TOK.finditer(yl)];_,q,g=DC._hits([yl[a:b] for a,b in M],ix)
 if not len(q):return y,0,0
 o=None if len(yl)==len(y) else np.cumsum([0]+[len(c.lower()) for c in y]);f=(lambda i:i) if o is None else (lambda i:int(np.searchsorted(o,i,"right"))-1);iv=[(f(M[j][0]),f(M[j+ix.n-1][1]-1)+1) for j in q.tolist()];cut=[(a,b) for a,b in (m.span() for m in SP.finditer(y)) if any(a<e and s<b for s,e in iv)];z=y
 for a,b in reversed(cut):z=z[:a]+z[b:]
 return re.sub(r"\n{3,}","\n\n",re.sub(r"[ \t]+(?=\n|$)","",z)).strip(),len(cut),int(np.bitwise_or.reduce(g))
def work(a:tuple)->dict:
 t,fx,u,d,ox=a;r=clean(("wikipedia",t,fx,u,d,None,"cc-by-sa-4.0")) if fx is not None else {"why":"nofw"};ok=not r["why"];y=r["text"] if ok else ox;z,n,g=scrub(y)
 return {"why":r["why"],"text":y,"h":r["h"] if ok else hh(y),"dc":z if n else None,"dn":n,"dg":g,"terms":r["terms"] if ok and not n else KR.terms(t,paginate(z.encode("utf-8","replace")))}|({k:r[k] for k in ("url","date","score","stale","refs")} if ok else {})
def cshard(s:int,pool,seen:np.ndarray)->np.ndarray:
 k=f"{s:05d}";t0=time.time();F={};C=collections.Counter();G=collections.Counter();cur=set();TM=[];gp=0;wr=None;out=os.path.join(KS,"clean",NW,k+".parquet");fd=os.path.join(KS,"fetch",NW,k);shutil.rmtree(fd+".tmp",ignore_errors=True);os.makedirs(fd+".tmp");os.makedirs(os.path.dirname(out),exist_ok=True);bf=open(os.path.join(fd+".tmp","blob.bin"),"wb");off=[0];raw=[];pg=[0];th=[]
 for f in sorted(glob.glob(os.path.join(RAWD,"parts",k,"*.parquet"))):
  for _,t,x,u,d in zip(*pq.read_table(f).to_pydict().values()):C["fw_dup_title"]+=t in F;(t not in F or d>F[t][2]) and F.__setitem__(t,(x,u,d))
 for b in pq.ParquetFile(os.path.join(KS,"clean",W,k+".parquet")).iter_batches(batch_size=4096):
  D=b.to_pydict();R=pool.map(work,[(t,*F.get(t,(None,None,None)),x) for t,x in zip(D["title"],D["text"])],chunksize=16);rows=[]
  for j,r in enumerate(R):
   t=D["title"][j];hd=bool(D["held"][j]);src="finewiki" if not r["why"] else "finewiki_prev" if s>=RS else "wikimedia_20231101";C[src]+=1;C["old_docs"]+=1;s<RS and r["why"] and C.update(["fallback_"+r["why"]]);dup=bool(r["h"] in cur or len(seen) and seen[min(int(np.searchsorted(seen,r["h"])),len(seen)-1)]==r["h"])
   if dup and not hd:C["dup_dropped"]+=1;continue
   C["held_dup_kept"]+=dup;cur.add(r["h"]);y=r["dc"] or r["text"];e=(t+"\x00"+y).encode("utf-8","replace");c=ZC.compress(e,asbytes=True);bf.write(c);off.append(off[-1]+len(c));raw.append(len(e));n=len(paginate(y.encode("utf-8","replace")));pg.append(pg[-1]+n);th.append(KR.th64(t));q=r["terms"];TM.append((q[0],q[1]+np.uint32(gp),q[2]));gp+=n
   r["dn"] and (C.update(dc_docs=1,dc_sentences=r["dn"]),G.update(gnames(r["dg"])));C["docs"]+=1;C["held"]+=hd;C["bytes"]+=len(y.encode("utf-8","replace"))
   rows.append({"title":t,"text":r["text"],"url":r.get("url") or D["url"][j],"date":r.get("date") or D["date"][j],"score":r.get("score",D["score"][j]),"stale":r.get("stale",D["stale"][j]),"refs":r.get("refs",D["refs"][j]),"license":"cc-by-sa-4.0" if src=="finewiki" else D["license"][j],"held":hd,"text_src":src,"dc_text":r["dc"],"dc_n":r["dn"],"dc_groups":",".join(gnames(r["dg"]))})
  wr=wr or pq.ParquetWriter(out+".tmp",SCH,compression="zstd");wr.write_table(pa.Table.from_pylist(rows,schema=SCH))
 wr.close();os.replace(out+".tmp",out);bf.close();[np.save(os.path.join(fd+".tmp",n+".npy"),np.asarray(v,dt)) for n,v,dt in (("off",off,np.int64),("raw",raw,np.int64),("pg",pg,np.int64),("th",th,np.uint64))];shutil.rmtree(fd,ignore_errors=True);os.replace(fd+".tmp",fd)
 HT=np.concatenate([np.zeros(0,np.uint32)]+[x[0] for x in TM]);PG=np.concatenate([np.zeros(0,np.uint32)]+[x[1] for x in TM]);WT=np.concatenate([np.zeros(0,np.uint8)]+[x[2] for x in TM]);o=np.argsort(HT,kind="stable");u,i0=np.unique(HT[o],return_index=True);KR.save_index(os.path.join(KS,"index",NW,k),terms=u,ptr=np.r_[i0,len(HT)].astype(np.int64),pages=PG[o],w=WT[o],npages=np.array([gp],np.int64))
 jw(os.path.join(ST,k+".json"),{"shard":k,"counts":dict(C),"dc_groups":dict(G),"fw_titles":len(F),"pages":gp,"s":round(time.time()-t0,1)});p(f"shard {k}: {dict(C)} dc {dict(G)} {time.time()-t0:.0f}s free {free():.0f} GiB");return np.union1d(seen,np.fromiter(cur,np.int64,len(cur)))
def titles(fd:str):
 Z=[(np.load(os.path.join(g,"th.npy")),int(os.path.basename(g))) for g in sorted(glob.glob(os.path.join(fd,"[0-9]"*5)))];h=np.concatenate([z for z,_ in Z]);o=np.argsort(h,kind="stable")
 np.savez(os.path.join(fd,"titles.npz"),h=h[o],seg=np.concatenate([np.full(len(z),g,np.int16) for z,g in Z])[o],doc=np.concatenate([np.arange(len(z),dtype=np.int32) for z,_ in Z])[o]);p(f"titles {len(h):,} -> {fd}/titles.npz")
def cmd_clean(a):
 os.makedirs(ST,exist_ok=True);missing=[s for s in range(RS) if not glob.glob(os.path.join(RAWD,"parts",f"{s:05d}","*.parquet"))];assert not missing or a.allow_missing,missing;dl=jr(os.path.join(RAWD,"state.json"),{});assert len(dl.get("files",[]))==15 or a.allow_missing,"FineWiki download incomplete"
 ix=DC.index();KR.terms("warm",[b"warm up the idf table"]);clean(("wikipedia","Warm","Warm is a word. "*40,"","2025-01-01",None,"cc-by-sa-4.0"));sf=os.path.join(ST,"seen.npy");seen=np.load(sf) if os.path.exists(sf) else np.zeros(0,np.int64);p(f"decontam {ix!r}; {len(seen):,} hashes seen")
 with mp.get_context("fork").Pool(a.workers) as pool:
  for s in range(a.start,NS):
   k=f"{s:05d}"
   if os.path.exists(os.path.join(ST,k+".json")) and os.path.isdir(os.path.join(KS,"fetch",NW,k)):continue
   free()<85 and sys.exit(p(f"stop: {free():.0f} GiB free"));seen=cshard(s,pool,seen);np.save(sf+".tmp.npy",seen);os.replace(sf+".tmp.npy",sf)
 titles(os.path.join(KS,"fetch",NW));n0,h0=heldsha(os.path.join(KS,"clean",W));n1,h1=heldsha(os.path.join(KS,"clean",NW));ho=sorted(t for f in glob.glob(os.path.join(KS,"heldout",W,"*.parquet")) for t in pq.read_table(f,columns=["title"]).column(0).to_pylist())
 S=[jr(f) for f in sorted(glob.glob(os.path.join(ST,"[0-9]"*5+".json")))];C=sum((collections.Counter(x["counts"]) for x in S),collections.Counter());G=sum((collections.Counter(x["dc_groups"]) for x in S),collections.Counter())
 r={"heldout":{"old":{"docs":n0,"sha256":h0},"new":{"docs":n1,"sha256":h1},"identical":h0==h1 and n0==n1,"heldout_parquet_titles":len(ho),"heldout_parquet_sha256":hashlib.sha256("\n".join(ho).encode()).hexdigest()},"counts":dict(C),"dc_groups":dict(G),"decontam":{"module":"amni.compute.decontam","index_key":ix.key,"n":ix.n,"eval_files":len(ix.files),"grams":int(len(ix.H)),"policy":"13-gram hits against every registered eval/hold set; the article is kept and every sentence overlapping a hit is removed"},"finewiki":{**dl,"revision":REV}}
 jw(os.path.join(ST,"clean_summary.json"),r);jw(os.path.join(KS,"clean",NW,"license_manifest.json"),{"license":"CC BY-SA 4.0 (Wikipedia; per-doc license column, cc-by-sa-3.0 for wikimedia/wikipedia 20231101 text)","sources":SRC,"docs_by_source":{k:C[k] for k in SRC},"attribution":ATTR});open(os.path.join(KS,"clean",NW,"ATTRIBUTION.txt"),"w").write(ATTR+"\n")
 p(json.dumps(r["heldout"]));assert r["heldout"]["identical"],"held-out set changed"
def cmd_fetch(a):
 fd=os.path.join(KS,"fetch",NW+a.variant);col="text" if a.variant=="_nodc" else sys.exit("unknown variant")
 for k in sorted(glob.glob(os.path.join(KS,"clean",NW,"[0-9]"*5+".parquet"))):
  g=os.path.join(fd,os.path.basename(k)[:5]);os.makedirs(g+".tmp",exist_ok=True);bf=open(os.path.join(g+".tmp","blob.bin"),"wb");off=[0];raw=[];pg=[0];th=[]
  for b in pq.ParquetFile(k).iter_batches(batch_size=4096,columns=["title",col]):
   for t,y in zip(*b.to_pydict().values()):e=(t+"\x00"+y).encode("utf-8","replace");c=ZC.compress(e,asbytes=True);bf.write(c);off.append(off[-1]+len(c));raw.append(len(e));pg.append(pg[-1]+len(paginate(y.encode("utf-8","replace"))));th.append(KR.th64(t))
  bf.close();[np.save(os.path.join(g+".tmp",n+".npy"),np.asarray(v,dt)) for n,v,dt in (("off",off,np.int64),("raw",raw,np.int64),("pg",pg,np.int64),("th",th,np.uint64))];shutil.rmtree(g,ignore_errors=True);os.replace(g+".tmp",g);p(f"fetch {g}: {len(raw):,} docs")
 titles(fd)
def view(v:str="")->str:
 V=os.path.join(tempfile.mkdtemp(prefix="ks_view_"),"ks");os.makedirs(V);M={("fetch",W):os.path.join("fetch",NW+v),("index",W):os.path.join("index",NW),("clean",W):os.path.join("clean",NW),("index_rknow","wiki_v2"):os.path.join("index_rknow","wiki_v3"+v)}
 for e in os.listdir(KS):
  if e not in ("fetch","index","clean","index_rknow"):os.symlink(os.path.join(KS,e),os.path.join(V,e));continue
  os.makedirs(os.path.join(V,e));[os.symlink(os.path.join(KS,M.get((e,x),os.path.join(e,x))),os.path.join(V,e,x)) for x in set(os.listdir(os.path.join(KS,e)))|{b for a,b in M if a==e}]
 os.environ.pop("AMNI_RKNOW_INDEX",None);KR.KS=V;KR.FETCH=os.path.join(V,"fetch");KR.KSIndex.__init__.__defaults__=(V,None,(W,));KR.Titles.__init__.__defaults__=(KR.FETCH,(W,));p(f"view {V}: wikipedia -> {NW+v}, wiki_v2 -> wiki_v3{v}");return V
def cmd_run(a):
 a.view and view(a.variant);sys.argv=[a.script]+a.args;runpy.run_path(a.script,run_name="__main__")
def meta(od:str,v:str):
 m=jr(os.path.join(od,"meta.json"));old=jr(os.path.join(KS,"index_rknow","wiki_v2","meta.json"));cs=jr(os.path.join(ST,"clean_summary.json"),{});C=cs.get("counts",{});al=jr(os.path.join(ST,f"rknow_alias{v}.json"),{})
 m.update(source=f"data/ks/fetch/wikipedia (English Wikipedia; built as data/ks/fetch/{NW+v} from data/ks/clean/{NW}; CC BY-SA 4.0)",license="CC BY-SA 4.0 (Wikipedia text; per-article license in data/ks/clean/wikipedia_fw, cc-by-sa-3.0 for the wikimedia/wikipedia 20231101 text)",attribution=ATTR,sources=[{"text_src":k,"dataset":d,"docs":C.get(k,0)} for k,d in SRC.items()],finewiki={"repo":FW,"config":"data/enwiki","revision":REV,"dataset_license":["cc-by-sa-4.0","gfdl"],"files":len(cs.get("finewiki",{}).get("files",[])),"bytes":cs.get("finewiki",{}).get("bytes"),"tables":"markdown tables removed","match":"by exact title against shards 00000-00040"},heldout=cs.get("heldout"),decontam={**cs.get("decontam",{}),**({"applied":False,"note":"no-decontam A/B variant"} if v=="_nodc" else {"applied":True,"docs":C.get("dc_docs",0),"sentences":C.get("dc_sentences",0),"by_group":cs.get("dc_groups",{})})},counts=C,alias=al,prev={"index":"wiki_v2","n_pass":old["n_pass"],"n_doc":old["n_doc"],"source":old["source"]},builder="scripts/ks_wiki_numbers.py (WIKI-NUMBERS)",built=time.strftime("%Y-%m-%dT%H:%M:%S"))
 jw(os.path.join(od,"meta.json"),m)
def cmd_rknow(a):
 v=a.variant;fd=os.path.join(KS,"fetch",NW+v);od=os.path.join(KS,"index_rknow","wiki_v3"+v);os.path.exists(os.path.join(fd,"titles.npz")) or titles(fd);os.makedirs(ST,exist_ok=True)
 for c in ("build","alias"):p(f"adam-rknow {c} {fd} {od}");subprocess.run([BIN,c,fd,od,str(a.threads)],check=True,stdout=open(os.path.join(ST,f"rknow_{c}{v}.json"),"w"),stderr=open(os.path.join(ST,f"rknow_{c}{v}.log"),"w"))
 view(v);import amni.compute.rknow as RK;I=RK.Index(od);I.btitles();meta(od,v);p(f"{od}: n_doc {I.m['n_doc']:,} n_pass {I.N:,} d_first {len(I.first):,} bt {len(I.btitles()[0]):,}")
def holes(ixd:str,fd:str,cd:str,n:int=3000,seed:int=17)->dict:
 m=jr(os.path.join(ixd,"meta.json"));db=np.asarray(m["doc_base"],np.int64);first=np.fromfile(os.path.join(ixd,"d_first.u32"),np.uint32);po=np.memmap(os.path.join(ixd,"p_off.u32"),np.uint32,"r");pl=np.memmap(os.path.join(ixd,"p_len.u32"),np.uint32,"r");R=random.Random(seed);F={};S={};out={}
 for gn,lo,hi in (("00000-00040",0,RS),("00041-00055",RS,NS)):
  C=collections.defaultdict(collections.Counter);k=0
  while k<n:
   g=R.randrange(int(db[lo]),int(db[hi]))
   if first[g]>=first[g+1]:continue
   i=int(np.searchsorted(db,g,"right"))-1;sg=m["segments"][i];d=g-int(db[i]);f=F.get(sg) or F.setdefault(sg,KR.fstore(os.path.join(fd,sg)));b=pa.decompress(f["blob"][f["off"][d]:f["off"][d+1]].tobytes(),int(f["raw"][d]),codec="zstd",asbytes=True);j=int(first[g]);x=b[int(po[j]):int(po[j])+int(pl[j])].decode("utf-8","replace")
   cp=os.path.join(cd,sg+".parquet");ts=S.get(sg) or S.setdefault(sg,pq.read_table(cp,columns=["text_src"]).column(0).to_pylist() if "text_src" in pq.read_schema(cp).names else None);src=ts[d] if ts else "all"
   for z in {src,"all"}:C[z].update(n=1,hole=bool(_HOLE.search(x)),broad=bool(BROAD.search(x)),glued=bool(GLUE.search(x)))
   k+=1
  out[gn]={z:{"n":c["n"],**{q:round(c[q]/c["n"],4) for q in ("hole","broad","glued")}} for z,c in C.items()}
 return out
def cmd_holes(a):r={"index":a.index,"holes":holes(os.path.join(KS,"index_rknow",a.index),os.path.join(KS,"fetch",a.fetch),os.path.join(KS,"clean",a.clean),a.n)};jw(os.path.join(ST,f"holes_{a.index}.json"),r);p(json.dumps(r))
def cmd_verify(a):
 o,n=(jr(os.path.join(KS,"index_rknow",x,"meta.json")) for x in ("wiki_v2","wiki_v3"));ho,hn=(jr(os.path.join(ST,f"holes_{x}.json"))["holes"]["00000-00040"]["all"] for x in ("wiki_v2","wiki_v3"));B,A=(jr(os.path.join(x,"retr.summary.json")) for x in (a.before,a.after));cs=jr(os.path.join(ST,"clean_summary.json"))
 gk=lambda S,s,c:S[s][c]["gold_in_ctx"];g=[(s,c,gk(B,s,c),gk(A,s,c)) for s in ("all","tune","hold") for c in ("know","multi")]
 G={"readiness":{"n_pass_old":o["n_pass"],"n_pass_new":n["n_pass"],"ratio":round(n["n_pass"]/o["n_pass"],4),"pass":abs(n["n_pass"]/o["n_pass"]-1)<=0.02},"gain":{"hole_old":ho["hole"],"hole_new":hn["hole"],"pass":hn["hole"]<=0.5*ho["hole"] and hn["hole"]<0.057},"gold_in_ctx":{"rows":g,"pass":all(y>=x for *_,x,y in g)},"heldout":{**cs["heldout"],"pass":cs["heldout"]["identical"]}}
 [G[k].update({"pass_measured":G[k]["pass"],"pass":True,"override":r,"override_t":time.strftime("%Y-%m-%dT%H:%M:%S")}) for k,r in (x.split("=",1) for x in a.override)];G["pass"]=all(v["pass"] for v in G.values());jw(os.path.join(ST,"verify.json"),G);mf=os.path.join(KS,"index_rknow","wiki_v3","meta.json");m=jr(mf);m["verify"]=G;jw(mf,m);p(json.dumps(G,indent=1))
PAIRS=(("fetch",W,NW),("index",W,NW),("index_rknow","wiki_v2","wiki_v3"))
def cmd_swap(a):
 v=jr(os.path.join(ST,"verify.json"),{});assert v.get("pass") or a.force,"verification has not passed; swap refused"
 lc=ctypes.CDLL("libc.so.6",use_errno=True)
 for d,x,y in PAIRS:
  s,t=os.path.join(KS,d,x),os.path.join(KS,d,y);assert os.path.isdir(s) and os.path.isdir(t),(s,t)
  lc.renameat2(-100,s.encode(),-100,t.encode(),2)==0 or sys.exit(p(f"renameat2 exchange failed for {s}: errno {ctypes.get_errno()}"));os.rename(t,os.path.join(KS,d,x+"_pre_fw"));p(f"swapped {s} <-> {t}; old copy now {x}_pre_fw")
 jw(os.path.join(ST,"swap.json"),{"t":time.strftime("%Y-%m-%dT%H:%M:%S"),"pairs":PAIRS})
def cmd_purge(a):
 if a.what=="old":
  assert jr(os.path.join(ST,"swap.json")),"no swap recorded"
  for d,x,_ in PAIRS:o=os.path.join(KS,d,x+"_pre_fw");os.path.isdir(o) and (sz:=du(o),shutil.rmtree(o),dlog(f"deleted data/ks/{d}/{x}_pre_fw ({sz/2**30:.1f} GiB, the pre-FineWiki data/ks/{d}/{x} swapped out after the rebuilt one passed verification: passage count, hole rate, retrieval dev gold-in-context, identical held-out set)"))
 if a.what=="raw":
  o=os.path.join(RAWD,"parts");st=jr(os.path.join(RAWD,"state.json"),{});os.path.isdir(o) and (sz:=du(o),shutil.rmtree(o),dlog(f"deleted data/ks/raw/wikipedia/finewiki_resource/parts ({sz/2**30:.1f} GiB: {st.get('matched',0):,} title-matched FineWiki articles converted from HuggingFaceFW/finewiki@{REV[:8]}) after data/ks/clean/{NW} was built from them"))
if __name__=="__main__":
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True);sp.add_parser("dl");c=sp.add_parser("clean");c.add_argument("--workers",type=int,default=2);c.add_argument("--start",type=int,default=0);c.add_argument("--allow-missing",dest="allow_missing",action="store_true")
 f=sp.add_parser("fetch");f.add_argument("--variant",default="_nodc");r=sp.add_parser("rknow");r.add_argument("--variant",default="");r.add_argument("--threads",type=int,default=2);h=sp.add_parser("holes");h.add_argument("--index",default="wiki_v2");h.add_argument("--fetch",default=W);h.add_argument("--clean",default=W);h.add_argument("--n",type=int,default=3000)
 v=sp.add_parser("verify");v.add_argument("--before",required=True);v.add_argument("--after",required=True);v.add_argument("--override",action="append",default=[]);s=sp.add_parser("swap");s.add_argument("--force",action="store_true");g=sp.add_parser("purge");g.add_argument("what",choices=["old","raw"])
 u=sp.add_parser("run");u.add_argument("--view",action="store_true");u.add_argument("--variant",default="");u.add_argument("script");u.add_argument("args",nargs=argparse.REMAINDER)
 a=ap.parse_args();globals()["cmd_"+a.cmd](a)
