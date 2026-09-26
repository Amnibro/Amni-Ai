import os,re,sys,json,time,gzip,glob,mmap,shutil,hashlib,subprocess,collections,multiprocessing as mp
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
import adam_ks
from amni.compute.ks_codec import KStore,crc16
from amni.compute.ray_field import BASE_PACK
from amni.compute import decontam as DC
from scripts.build_clean1_corpus import PGX
_H0=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));E=os.environ.get
OUT=os.path.join(_H0,"data",E("ADAM_T_OUT","adam_t/stream"));KS=os.path.join(_H0,"data","ks","store_clean");KC=os.path.join(_H0,"data","ks","clean");C2=os.path.join(_H0,"data","clean2")
TOTAL=float(E("ADAM_T_GB","95"))*1e9;MIX={"fineweb":0.70,"wiki":0.20,"other":0.10};SHARD=int(float(E("ADAM_T_SHARD_GIB","1"))*2**30);HOLD=float(E("ADAM_T_HOLD","0.005"));SEED=int(E("SEED","17"));W=int(E("ADAM_T_WORKERS","4"));TH=int(E("ADAM_T_THREADS","3"));PW=int(E("ADAM_T_PLAN_WORKERS","6"));MINFREE=float(E("ADAM_T_MIN_FREE_GIB","61"))*2**30;BLK=65536;MAXREP=int(E("ADAM_T_MAX_REP","3"));NPG=int(E("ADAM_T_BATCH_PAGES","3072"));AL=4096;PLAN_ONLY=E("ADAM_T_PLAN_ONLY")=="1";MAXSH=int(E("ADAM_T_MAX_SHARDS","0"));FWLAST=E("ADAM_T_FW_LAST","00056")
SRC=("fineweb_edu","wikipedia","gutenberg","arxiv_abs","openwebmath","clean2");GRP=("fineweb","wiki","other","other","other","other");TITLED={1,2,3};C2T,C2H=65535,65534
LIC={"fineweb_edu":{"odc-by-1.0"},"wikipedia":{"cc-by-sa-3.0","cc-by-sa-4.0"},"gutenberg":{"us-pd"},"arxiv_abs":{"cc0-1.0"},"openwebmath":{"odc-by-1.0"}}
CREDIT={"fineweb_edu":"HuggingFaceFW/fineweb-edu (all dumps newest first, int_score>=3 at KS-FULL ingest; store 00000 = sample-10BT int_score>=4 via M05), ODC-By 1.0 plus Common Crawl terms of use; per-doc URL recoverable from the source KS store via the shard .idx.npy (st,d) columns","wikipedia":"English Wikipedia (wikimedia/wikipedia 20231101.en + HuggingFaceFW/finewiki enwiki via data/ks/store_clean/wikipedia), CC BY-SA 3.0/4.0; per-doc title+URL+license in attribution.jsonl.gz","gutenberg":"Project Gutenberg US public-domain texts via common-pile/project_gutenberg (M05 store); paragraphs referencing Project Gutenberg removed (build_clean1_corpus.PGX); ebook URL in attribution.jsonl.gz","arxiv_abs":"arXiv abstracts (title + abstract) via gfissore/arxiv-abstracts-2021, CC0 1.0; arxiv.org/abs URL in attribution.jsonl.gz","openwebmath":"OpenWebMath (open-web-math/open-web-math), ODC-By 1.0 plus Common Crawl terms of use; page URL in attribution.jsonl.gz","clean2":"data/clean2 train.bin/hold.bin (Cosmopedia v2 ODC-By, Wikipedia CC BY-SA, FineWeb-Edu ODC-By, Gutenberg US-PD, arXiv CC0, OpenWebMath ODC-By, own code own-ceo, codeparrot permissive); per-doc attribution in attribution_clean2.jsonl.gz (copy of data/clean2/attribution.jsonl.gz)"}
IT=np.dtype([("s","u1"),("st","u2"),("d","u4"),("n","u4")]);IX=np.dtype([("off","u8"),("s","u1"),("st","u2"),("d","u4")])
XEV=sorted(f for f in glob.glob(os.path.join(_H0,E("ADAM_T_EVAL_DIR","data/adam_t/eval"),"**","*.json*"),recursive=True) if not DC.SKIP.search(os.path.basename(f)) and "lock" not in os.path.basename(f));ST=[(k,p) for k,s in enumerate(SRC[:5]) for p in sorted(glob.glob(os.path.join(KS,s,"*"))) if os.path.exists(os.path.join(p,"meta.json")) and (k or os.path.basename(p)<=FWLAST)];G={}
def p(s):print(time.strftime("%H:%M:%S"),s,flush=True)
def hk(k:str)->tuple:d=hashlib.blake2b(k.encode(),digest_size=16).digest();return int.from_bytes(d[:8],"little"),int.from_bytes(d[8:],"little")/2**64
def rel(x:str)->str:return os.path.relpath(x,_H0)
def plan(i:int)->dict:
 pa.set_cpu_count(1);pa.set_io_thread_count(1);s,path=ST[i];src=SRC[s];k=KStore(path,check=False);f=k.first;nd=len(f);fe=np.r_[f,len(k)].astype(np.int64);ti=[t or "" for t in k.title]
 ev=[json.loads(x) for x in k.pm.column("evidence").take(pa.array(f)).to_pylist()];url=[(e.get("urls") or [""])[0] or "" for e in ev];lic=[e.get("license") or "" for e in ev]
 hp=os.path.join(KC,src,os.path.basename(path)+".parquet");hd=pq.read_table(hp,columns=["title","held"]) if s and os.path.exists(hp) else None;hm=hd is not None and hd.num_rows==nd and [t or "" for t in hd.column("title").to_pylist()]==ti
 assert s!=1 or hm,("wiki held flags not aligned",path);held=np.array(hd.column("held").to_pylist(),bool) if hm else np.zeros(nd,bool)
 key=[t if s==1 else u for t,u in zip(ti,url)];real=np.array([bool(x) for x in key],bool);H=[hk(x or f"{rel(path)}:{j}") for j,x in enumerate(key)];a=np.fromiter((h[0] for h in H),np.uint64,nd);u=np.fromiter((h[1] for h in H),np.float32,nd)
 c=np.r_[0,np.cumsum(k.nby.astype(np.int64))];enc=[t.encode() for t in ti] if s in TITLED else None;nb=c[fe[1:]]-c[fe[:-1]]+(np.array([len(x)+2 for x in enc],np.int64) if enc else 0)+2
 cd=k.meta["coders"];assert len(cd)==1 and cd[0]["ver"]=="ks2-clean1",path
 return {"i":i,"off":k.off.astype(np.int64),"nby":k.nby.astype(np.uint16),"crc":k.crc.astype(np.uint16),"first":fe.astype(np.uint32),"nb":nb.astype(np.uint32),"a":a,"real":real,"u":u,"held":held,"held_src":rel(hp) if hm else None,"licok":np.array([l in LIC[src] for l in lic],bool),"c2":np.isin(a,G["c2"])&real,"lc":collections.Counter(lic),"cid":(cd[0]["pack_id"],cd[0]["lex_id"]),"blob":b"".join(enc) if enc else None,"to":np.r_[0,np.cumsum([len(x) for x in enc])].astype(np.int64) if enc else None}
def its(s:int,st:int,d:np.ndarray,n:np.ndarray)->np.ndarray:
 o=np.zeros(len(d),IT);o["s"]=s;o["st"]=st;o["d"]=d;o["n"]=n;return o
def blocks(path:str)->tuple:
 n=os.path.getsize(path);o=[0]
 with open(path,"rb") as f:
  m=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
  while o[-1]+BLK<n:
   j=m.find(b"\n\n",o[-1]+BLK-2);j=n if j<0 else j+2
   while j<n and m[j]==10:j+=1
   o.append(j)
  m.close()
 o=o+([n] if o[-1]<n else []);o=np.array(o,np.int64);return o[:-1],np.diff(o).astype(np.int64)
def job(a:tuple)->dict:
 n,its=a;k=G.get("k");t0=time.time()
 if k is None:k=G["k"]=adam_ks.Codec(BASE_PACK,os.path.join(_H0,"data","clean1"),"ks2-clean1",TH);ci=json.loads(k.ident());assert all((ci["pack_id"],ci["lex_id"])==c for c in G["cid"]),(ci,G["cid"])
 T,B,ix,fd=G["T"],G["B"],G["ix"],G.setdefault("fd",{});fp=lambda q,path:fd.get(q) or fd.setdefault(q,os.open(path,os.O_RDONLY));st=collections.defaultdict(collections.Counter);out=os.path.join(OUT,n+".bin");f=open(out+".part","wb");h=hashlib.sha256();pos=0;X=[];j=0
 while j<len(its):
  e=j;pg=0
  while e<len(its) and pg<NPG:q=int(its[e]["st"]);d=int(its[e]["d"]);pg+=int(T[q]["first"][d+1])-int(T[q]["first"][d]) if q<C2H else 16;e+=1
  L=[];Cd=[];R=[]
  for t in its[j:e]:
   q,d=int(t["st"]),int(t["d"])
   if q>=C2H:bo,bl,bp=B[q];R.append(os.pread(fp(q,bp),int(bl[d]),int(bo[d])));continue
   P=T[q];a,b=int(P["first"][d]),int(P["first"][d+1]);o=P["off"][a:b+1]-P["off"][a];raw=os.pread(fp(q,P["pages"]),int(o[-1]),int(P["off"][a]));Cd+=[raw[o[x]:o[x+1]] for x in range(b-a)];R.append((a,b,len(L)));L+=P["nby"][a:b].tolist()
  D=k.decode(L,Cd) if L else []
  for t,r in zip(its[j:e],R):
   s,q,d=int(t["s"]),int(t["st"]),int(t["d"]);c=st[SRC[s]]
   if isinstance(r,tuple):
    a,b,o=r;P=T[q];pp=D[o:o+b-a];bad=[x for x,(y,z) in enumerate(zip(pp,P["crc"][a:b])) if crc16(y)!=int(z)]
    if bad:raise ValueError(f"KS crc mismatch {n} store {q} doc {d} pages {bad[:4]}")
    x=b"".join(pp).strip();ti=P["blob"][P["to"][d]:P["to"][d+1]] if P["blob"] is not None else b"";x=ti+b"\n\n"+x if ti and (s!=2 or not x.startswith(ti)) else x
    if s==2 and PGX.search(x.decode("utf-8","replace")):ps=re.split(r"\n\s*\n",x.decode("utf-8","replace"));kp=[y for y in ps if not PGX.search(y)];c["pgx_paragraphs_removed"]+=len(ps)-len(kp);x="\n\n".join(kp).strip().encode()
    if not x:c["empty"]+=1;continue
    x+=b"\n\n"
   else:x=r
   _,hp,hm=DC._hits(DC.words(x.decode("utf-8","replace")),ix);gb=int(np.bitwise_or.reduce(hm)) if len(hp) else 0
   if gb:c["dirty_docs"]+=1;c["dirty_bytes"]+=len(x);c.update({"dirty_group:"+g:1 for b_,g in enumerate(ix.groups) if gb>>b_&1});continue
   f.write(x);h.update(x);X.append((pos,s,q,d));pos+=len(x);c["docs"]+=1;c["bytes"]+=len(x)
  j=e
 pad=(-pos)%AL;f.write(b"\n"*pad);h.update(b"\n"*pad);f.flush();os.fsync(f.fileno());f.close();os.replace(out+".part",out);np.save(os.path.join(OUT,n+".idx.npy"),np.array(X,IX))
 r={"file":n+".bin","idx":n+".idx.npy","bytes":pos+pad,"pad":pad,"docs":len(X),"sha256":h.hexdigest(),"items":len(its),"index_key":G["ix"].key,"secs":round(time.time()-t0,1),"src":{k_:dict(v) for k_,v in sorted(st.items())}};json.dump(r,open(os.path.join(OUT,n+".json"),"w"),indent=1);return r
def concat(path:str,parts:list):
 o=os.open(path,os.O_WRONLY|os.O_CREAT,0o644)
 for x in parts:
  with open(x,"rb") as i:
   n=os.fstat(i.fileno()).st_size;off=os.fstat(o).st_size;d=0
   while d<n:d+=os.copy_file_range(i.fileno(),o,n-d,d,off+d)
 os.close(o)
def attrib(done:dict):
 S=collections.defaultdict(list);C=collections.Counter()
 for n,r in done.items():
  x=np.load(os.path.join(OUT,r["idx"]));x=x[(x["st"]<C2H)&(x["s"]>0)];[S[int(q)].append((x["d"][x["st"]==q],n)) for q in np.unique(x["st"])]
 with gzip.open(os.path.join(OUT,"attribution.jsonl.gz.part"),"wt") as g:
  for q in sorted(S):
   s,path=ST[q];dt=pq.read_table(os.path.join(path,"docs.parquet"),columns=["title","first"]);ti=dt.column("title").to_pylist();fi=dt.column("first").to_numpy();ev=pq.read_table(os.path.join(path,"pagemeta.parquet"),columns=["evidence"]).column("evidence")
   for d,n in S[q]:
    for j in np.sort(d):e_=json.loads(ev[int(fi[j])].as_py());C[SRC[s]]+=1;g.write(json.dumps({"src":SRC[s],"title":ti[j],"url":(e_.get("urls") or [""])[0],"license":e_.get("license"),"file":n,"store":rel(path),"doc":int(j)})+"\n")
 os.replace(os.path.join(OUT,"attribution.jsonl.gz.part"),os.path.join(OUT,"attribution.jsonl.gz"));shutil.copyfile(os.path.join(C2,"attribution.jsonl.gz"),os.path.join(OUT,"attribution_clean2.jsonl.gz"));return dict(C)
def mix(done:dict,c2m:dict)->dict:
 b=collections.Counter();dd=collections.Counter();db=collections.Counter()
 for n,r in done.items():
  if n.startswith("train"):[b.update({k_:v.get("bytes",0)}) or dd.update({k_:v.get("dirty_docs",0)}) or db.update({k_:v.get("dirty_bytes",0)}) for k_,v in r["src"].items()]
 tot=max(1,sum(b.values()));ws=c2m["sources"]["wiki"]["bytes"]/c2m["total_bytes"];g={"fineweb":b["fineweb_edu"],"wiki":b["wikipedia"]+b["clean2"]*ws,"other":b["gutenberg"]+b["arxiv_abs"]+b["openwebmath"]+b["clean2"]*(1-ws)}
 orig=collections.Counter({"fineweb_edu":b["fineweb_edu"],"wikipedia":b["wikipedia"],"gutenberg":b["gutenberg"],"arxiv_abs":b["arxiv_abs"],"openwebmath":b["openwebmath"]});[orig.update({"clean2:"+k_:b["clean2"]*v["bytes"]/c2m["total_bytes"]}) for k_,v in c2m["sources"].items()]
 return {"train_doc_bytes":tot,"by_stream_source":{k_:{"bytes":int(v),"share":round(v/tot,4)} for k_,v in sorted(b.items())},"by_group":{k_:{"bytes":int(v),"share":round(v/tot,4),"target":MIX[k_]} for k_,v in g.items()},"by_original_source":{k_:{"bytes":int(v),"share":round(v/tot,4)} for k_,v in sorted(orig.items())},"clean2_apportion":"clean2 bytes split by data/clean2/meta.json per-source byte shares (clean2 is included as whole 64 KiB blocks, see clean2_blocks)","decontam_dropped":{"docs":dict(dd),"bytes":dict(db)}}
def main():
 os.makedirs(OUT,exist_ok=True);t0=time.time();c2m=json.load(open(os.path.join(C2,"meta.json")));K=[]
 with gzip.open(os.path.join(C2,"attribution.jsonl.gz"),"rt") as f:
  for l in f:r=json.loads(l);x=r.get("title") if r["src"]=="wiki" else r.get("url") if r["src"] in ("fineweb","arxiv","openwebmath","gutenberg") else None;x and K.append(hk(x)[0])
 G["c2"]=np.unique(np.array(K,np.uint64));p(f"clean2 attribution keys {len(G['c2'])}; planning {len(ST)} KS stores with {PW} workers")
 with mp.get_context("fork").Pool(PW) as pool:P=pool.map(plan,range(len(ST)),chunksize=1)
 A=np.concatenate([q["a"] for q in P]);RL=np.concatenate([q["real"] for q in P]);ri=np.flatnonzero(RL);dup=np.zeros(len(A),bool);dup[ri]=True;dup[ri[np.unique(A[ri],return_index=True)[1]]]=False;cs=np.cumsum([0]+[len(q["a"]) for q in P]);del A,RL
 ex=collections.defaultdict(collections.Counter);av=collections.Counter();lic=collections.defaultdict(collections.Counter);U={}
 for q,x in zip(P,range(len(P))):
  s=ST[q["i"]][0];src=SRC[s];dp=dup[cs[x]:cs[x+1]];el=q["licok"]&~q["held"]&~q["c2"]&~dp;ho=(q["a"]%np.uint64(10**6))<np.uint64(int(HOLD*1e6));U[x]=(el,ho);lic[src].update(q["lc"]);av[GRP[s]]+=int(q["nb"][el&~ho].sum());av["hold:"+GRP[s]]+=int(q["nb"][el&ho].sum())
  ex[src].update({"docs":len(el),"ks_held":int(q["held"].sum()),"license":int((~q["licok"]).sum()),"in_clean2":int(q["c2"].sum()),"dup_key":int(dp.sum()),"eligible":int(el.sum()),"hold_candidates":int((el&ho).sum())});ex[src]["held_flags_from"]=q["held_src"] is not None
 del dup;tb=blocks(os.path.join(C2,"train.bin"));hb=blocks(os.path.join(C2,"hold.bin"));c2b=int(tb[1].sum());c2h=int(hb[1].sum());ws=c2m["sources"]["wiki"]["bytes"]/c2m["total_bytes"]
 R=int(np.clip(round(MIX["other"]*TOTAL/(c2b*(1-ws)+av["other"])),1,MAXREP));rw=min(1.0,max(0.0,MIX["wiki"]*TOTAL-R*c2b*ws)/max(1,av["wiki"]));rf=min(1.0,max(0.0,TOTAL-R*(c2b+av["other"])-rw*av["wiki"])/max(1,av["fineweb"]));rate={"fineweb":rf,"wiki":rw,"other":1.0}
 rng=np.random.default_rng(SEED);TR=[];HO=[]
 for q,x in zip(P,range(len(P))):
  s=ST[q["i"]][0];el,ho=U[x];sm=q["u"]<rate[GRP[s]];d=np.flatnonzero(el&~ho&sm);h=np.flatnonzero(el&ho&sm);mk=lambda d:its(s,x,d,q["nb"][d])
  TR+=[mk(d)]*(R if GRP[s]=="other" else 1);HO.append(mk(h))
 rh=min(1.0,R*c2b*HOLD/(1-HOLD)/max(1,c2h));hs=np.flatnonzero(rng.random(len(hb[0]))<rh);TR+=[its(5,C2T,np.arange(len(tb[0])),tb[1])]*R;HO.append(its(5,C2H,hs,hb[1][hs]))
 TR=np.concatenate(TR);TR=TR[rng.permutation(len(TR))];HO=np.concatenate(HO);HO=HO[rng.permutation(len(HO))];c=np.cumsum(TR["n"].astype(np.int64));cut=np.r_[0,np.searchsorted(c,np.arange(1,int(c[-1]//SHARD)+1)*SHARD),len(TR)];cut=np.unique(cut)
 bud={"total_target_bytes":TOTAL,"clean2_repeat":R,"sample_rate":{k_:round(v,5) for k_,v in rate.items()},"available_train_bytes":dict(av),"clean2_train_bytes":c2b,"clean2_hold_bytes":c2h,"clean2_wiki_share":round(ws,4),"clean2_hold_block_rate":round(rh,4),"planned_train_bytes":int(c[-1]),"planned_hold_bytes":int(HO["n"].astype(np.int64).sum()),"planned_train_items":len(TR),"planned_hold_items":len(HO),"planned_shards":len(cut)-1,"planned_by_src":{SRC[s]:int(TR["n"][TR["s"]==s].astype(np.int64).sum()) for s in range(6)}}
 p(json.dumps(bud));p(json.dumps({k_:dict(v) for k_,v in ex.items()}))
 if PLAN_ONLY:return
 np.save(os.path.join(OUT,"plan_train.npy"),TR);np.save(os.path.join(OUT,"plan_hold.npy"),HO);np.save(os.path.join(OUT,"clean2_train_blocks.npy"),np.stack(tb,1));np.save(os.path.join(OUT,"clean2_hold_blocks.npy"),np.stack(hb,1))
 G["T"]=[{"off":q["off"],"nby":q["nby"],"crc":q["crc"],"first":q["first"],"blob":q["blob"],"to":q["to"],"pages":os.path.join(ST[q["i"]][1],"pages.bin")} for q in P];G["cid"]={q["cid"] for q in P};G["B"]={C2T:(tb[0],tb[1],os.path.join(C2,"train.bin")),C2H:(hb[0],hb[1],os.path.join(C2,"hold.bin"))};del P,U
 ix=DC.index(files=DC.eval_sources()+XEV);G["ix"]=ix;DC._MEMO["default"]=ix;p(f"decontam index {ix!r} groups {ix.groups} extra {[rel(f) for f in XEV]}")
 jobs=[("hold",HO)]+[(f"train_{j:05d}",TR[cut[j]:cut[j+1]]) for j in range(min(MAXSH or 10**9,len(cut)-1))];done={n:json.load(open(os.path.join(OUT,n+".json"))) for n,_ in jobs if os.path.exists(os.path.join(OUT,n+".bin")) and os.path.exists(os.path.join(OUT,n+".json"))};q=[x for x in jobs if x[0] not in done];run={};nxt=0;stop=None;tbin=os.path.join(OUT,"train.bin")
 [os.path.exists(x) and os.remove(x) for x in [tbin]+glob.glob(os.path.join(OUT,"*.part"))]
 def man(status:str,extra:dict=None):
  sh=[done[f"train_{j:05d}"] for j in range(nxt)];m={"name":"ADAM-T training stream","status":status,"built":time.strftime("%Y-%m-%dT%H:%M:%S"),"elapsed_s":round(time.time()-t0,1),"cmd":" ".join(f"{k_}={E(k_)}" for k_ in sorted(os.environ) if k_.startswith("ADAM_T_") or k_=="SEED")+" python -m scripts.build_adam_t_stream",
   "format":{"encoding":"raw UTF-8 bytes (byte-level, no tokenizer)","delimiter":"each document is followed by b'\\n\\n' exactly like data/clean1 and data/clean2 (docs joined by \\n\\n); wiki/arXiv docs are 'Title\\n\\nText' like clean2","exact_boundaries":"<shard>.idx.npy: structured array off(u8) s(u1 index into sources) st(u2 KS store index into stores, 65535 = clean2 train block, 65534 = clean2 hold block) d(u4 doc id in that store, or clean2 block id)","padding":"each shard ends with 0..4095 extra b'\\n' so its size is a multiple of 4096 (btrfs reflink alignment)","shuffle":f"global document-level permutation (numpy default_rng({SEED})) of all selected documents across every source, then cut into ~{SHARD/2**30:g} GiB shards; clean2 enters as doc-aligned-where-possible 64 KiB blocks (cut at the first \\n\\n after each 64 KiB) because its per-doc boundaries are not recoverable from train.bin","train_bin":"train.bin = btrfs reflink concatenation (copy_file_range, shared extents, ~0 extra disk) of the finished train shards in order; data/adam_t/stream is directly usable as CORPUS=adam_t/stream by crates/adam-t (mmaps data/<CORPUS>/train.bin + hold.bin)","hold_bin":"hold.bin = held-out split (~0.5% of documents by blake2b(url or title) hash, disjoint from train; clean2 part from data/clean2/hold.bin)","reader":"amni/compute/adam_t_stream.py"},
   "seed":SEED,"shards_cap":MAXSH or None,"shards_cap_note":"the global permutation is cut into the planned shards and only the first shards_cap are built; any prefix of the permutation is a uniform random document sample, so the mix is preserved" if MAXSH else None,"sources":list(SRC),"stores":[rel(x[1]) for x in ST],"budget":bud,"exclusions":{k_:dict(v) for k_,v in ex.items()},"licenses":{k_:sorted(v) for k_,v in lic.items()}|{"clean2":c2m["licenses"]},"credits":CREDIT,"clean2_credits":{k_:v["credit"] for k_,v in c2m["sources"].items()},
   "decontam":{"module":"amni.compute.decontam","method":"every document (and every clean2 block) re-checked at build time with DC.dirty (shared 13-gram of lowercased alnum words vs every eval/hold set in decontam.eval_sources()); any hit drops the document","index_key":ix.key,"index_files":len(ix.files),"index_grams":int(len(ix.H)),"index_groups":ix.groups,"extra_eval_files":{rel(f):hashlib.sha256(open(f,"rb").read()).hexdigest() for f in XEV},"extra_eval_note":"data/adam_t/eval sets (NQ multiple-choice, reading-comprehension dev, D3-style dev dialogues, ARC-Easy test 500, PIQA valid 500) are not in decontam.eval_sources(); passed explicitly via DC.index(files=eval_sources()+extra); NQ dev tune/hold are already registered via data/concepts/eval/rknow/nq_dev_*.json; group adam_t in dirty_group counts"},
   "shards":sh,"train_bin":{"file":"train.bin","bytes":sum(x["bytes"] for x in sh),"shards":len(sh)},"hold":done.get("hold"),"mix":mix({x["file"][:-4]:x for x in sh},c2m),"disk":{"free_gib":round(shutil.disk_usage(OUT).free/2**30,1),"min_free_gib":MINFREE/2**30,"stopped_for_disk":stop}}|(extra or {})
  json.dump(m,open(os.path.join(OUT,"manifest.json.part"),"w"),indent=1);os.replace(os.path.join(OUT,"manifest.json.part"),os.path.join(OUT,"manifest.json"));return m
 pool=mp.get_context("fork").Pool(W);p(f"{len(jobs)} jobs, {len(done)} already done, workers {W}x{TH} threads")
 while q or run:
  while q and len(run)<W and stop is None:
   fr=shutil.disk_usage(OUT).free
   if fr-(len(run)+1)*SHARD*1.05<MINFREE:stop={"free_gib":round(fr/2**30,1),"skipped_jobs":len(q)};p(f"disk guard: stop scheduling, {stop}");break
   n,it=q.pop(0);run[n]=pool.apply_async(job,((n,it),))
  if stop is not None:q=[]
  for n in [n for n,r in run.items() if r.ready()]:done[n]=run.pop(n).get();p(f"{n}: {done[n]['bytes']/2**30:.3f} GiB, {done[n]['docs']} docs, {done[n]['secs']}s, free {shutil.disk_usage(OUT).free/2**30:.1f} GiB")
  a=nxt
  while f"train_{nxt:05d}" in done:nxt+=1
  nxt>a and (concat(tbin,[os.path.join(OUT,f"train_{j:05d}.bin") for j in range(a,nxt)]),man("building"),p(f"train.bin now {nxt} shards"))
  "hold" in done and not os.path.exists(os.path.join(OUT,"manifest.json")) and man("building")
  run and time.sleep(5)
 pool.close();pool.join();man("attribution");at=attrib({n:r for n,r in done.items()});nsh=[f"train_{j:05d}.bin" for j in range(nxt)];sm=[os.path.join(OUT,x) for x in nsh[::max(1,len(nsh)//12)]]+[os.path.join(OUT,"hold.bin")]
 man("audit",{"attribution":{"file":"attribution.jsonl.gz","docs":at,"clean2":"attribution_clean2.jsonl.gz"}});au=os.path.join(OUT,"decontam_scan.json");rc=subprocess.run([sys.executable,"-m","amni.compute.decontam","scan",*sm,"--sample-mb",E("ADAM_T_AUDIT_MB","48"),"--strata","128","--workers",str(min(W,4)),"--out",au],cwd=_H0).returncode;A=json.load(open(au)) if os.path.exists(au) else {};AX=[DC.scan_file(f,float(E("ADAM_T_AUDIT_MB","48")),128,min(W,4)) for f in sm];pk=("path","mode","units","dirty_units","unit_rate","chars","dirty_chars","grams","hit_grams","by_group")
 man("done",{"attribution":{"file":"attribution.jsonl.gz","docs":at,"clean2":"attribution_clean2.jsonl.gz"},"decontam_audit":{"cmd":"python -m amni.compute.decontam scan <stratified shards> --sample-mb "+E("ADAM_T_AUDIT_MB","48")+" --strata 128","returncode":rc,"index_key":A.get("index",{}).get("key"),"same_index_as_build":A.get("index",{}).get("key")==ix.key,"results":[{k_:r[k_] for k_ in pk} for r in A.get("results",[])]},"decontam_audit_extended":{"cmd":"in-process decontam.scan_file on the same stratified shards with the build index (eval_sources + data/adam_t/eval)","index_key":ix.key,"results":[{k_:r[k_] for k_ in pk} for r in AX]}});p(f"done in {time.time()-t0:.0f}s")
def dlog(msg:str):open(os.path.join(_H0,"data","ks","logs","deletions.log"),"a").write(time.strftime("%Y-%m-%dT%H:%M:%S%z")[:-2]+":"+time.strftime("%z")[-2:]+" ADAM-T-DATA: "+msg+"\n")
def sha(path:str,a:int=0,n:int=None)->str:
 h=hashlib.sha256()
 with open(path,"rb") as f:
  f.seek(a);r=os.path.getsize(path)-a if n is None else n
  while r>0:b=f.read(min(r,1<<24));h.update(b);r-=len(b)
 return h.hexdigest()
def tokenize():
 bp=E("ADAM_T_BPE",os.path.join(_H0,"data","clean2","bpe4096.json"));exe=E("ADAM_T_TOK_BIN",os.path.join(_H0,".claude","worktrees","agent-ae741731ad611c7d7","crates","target","release","adam-t"));j=json.load(open(bp));voc=[bytes([b]) for b in range(256)];[voc.append(voc[a]+voc[b]) for a,b in j["merges"]];V=np.frombuffer(b"".join(voc),np.uint8);L=np.array([len(v) for v in voc],np.int64);O=np.r_[0,np.cumsum(L)[:-1]]
 def same(tk:np.ndarray,raw:str)->bool:
  r=np.memmap(raw,np.uint8,"r");q=0
  for i in range(0,len(tk),1<<23):
   x=tk[i:i+(1<<23)].astype(np.int64);l=L[x];c=np.cumsum(l);d=V[np.repeat(O[x]-(c-l),l)+np.arange(int(c[-1]))]
   if q+len(d)>len(r) or not np.array_equal(d,r[q:q+len(d)]):return False
   q+=len(d)
  return q==len(r)
 m=json.load(open(os.path.join(OUT,"manifest.json")));assert m["status"]=="done",m["status"];sh=m["shards"];tf=os.path.join(OUT,"train.bpe4096.u16");sp=os.path.join(OUT,"tokens.json");TD=os.path.join(OUT,"tok");os.makedirs(TD,exist_ok=True);es=sha(exe);bs=sha(bp)
 T=json.load(open(sp)) if os.path.exists(sp) else {"bpe":rel(bp),"bpe_sha256":bs,"encoder":rel(exe),"encoder_sha256":es,"cmd":"RAYON_NUM_THREADS=%s adam-t tokenize %s <train_XXXXX.bin> <out.u16>"%(E("ADAM_T_TOK_THREADS","8"),rel(bp)),"decode":"voc=bytes 0..255 + merges from bpe4096.json; tokens are little-endian u16","shards":[]};R=T["shards"];save=lambda:(json.dump(T,open(sp+".part","w"),indent=1),os.replace(sp+".part",sp))
 cur=os.path.getsize(tf)//2 if os.path.exists(tf) else 0;assert cur==(R[-1]["tok_end"] if R else cur),("train.bpe4096.u16 size does not match tokens.json",cur)
 if not R and cur:
  k=0;a=0
  while k<len(sh) and a<cur and os.path.exists(os.path.join(OUT,"tok4096",f"train_{k:05d}.u16")):
   f=os.path.join(OUT,"tok4096",f"train_{k:05d}.u16");n=os.path.getsize(f)//2;h=sha(f);assert sha(tf,2*a,2*n)==h,("ADAM-T prefix mismatch",k);raw=os.path.join(OUT,sh[k]["file"]);assert same(np.memmap(f,np.uint16,"r"),raw),("decode mismatch",k)
   R.append({"shard":k,"raw_file":sh[k]["file"],"raw_bytes":sh[k]["bytes"],"raw_sha256":sh[k]["sha256"],"tok_start":a,"tok_end":a+n,"tokens":n,"sha256":h,"bytes_per_token":round(sh[k]["bytes"]/n,4),"by":"ADAM-T agent tok4096/ (verified here)","verified_byte_exact":True});a+=n;k+=1;p(f"adopted ADAM-T token shard {k-1}: {n} tokens, byte-exact")
  assert a==cur,("ADAM-T prefix does not cover train.bpe4096.u16",a,cur);save()
 for r in R:
  raw=os.path.join(OUT,r["raw_file"])
  if os.path.exists(raw) and r["verified_byte_exact"]:os.remove(raw);dlog(f"deleted data/adam_t/stream/{r['raw_file']} ({r['raw_bytes']} raw UTF-8 bytes, sha256 {r['raw_sha256']}) after verifying its BPE-4096 token shard (train.bpe4096.u16 tokens [{r['tok_start']},{r['tok_end']}), sha256 {r['sha256']}) decodes back byte-exact");r["raw_deleted"]=True
 save()
 for k in range(len(R),len(sh)):
  raw=os.path.join(OUT,sh[k]["file"]);tmp=os.path.join(TD,f"train_{k:05d}.u16");t0=time.time();o=subprocess.run([exe,"tokenize",bp,raw,tmp],env={**os.environ,"RAYON_NUM_THREADS":E("ADAM_T_TOK_THREADS","8")},capture_output=True,text=True);assert o.returncode==0,o.stderr[-2000:]
  tk=np.memmap(tmp,np.uint16,"r");n=len(tk);ok=same(tk,raw);assert ok,("decode mismatch",k);h=sha(tmp);del tk
  with open(tf,"ab") as g,open(tmp,"rb") as f:shutil.copyfileobj(f,g,1<<24);g.flush();os.fsync(g.fileno())
  a=R[-1]["tok_end"] if R else 0;assert os.path.getsize(tf)==2*(a+n) and sha(tf,2*a,2*n)==h,("append verify failed",k)
  R.append({"shard":k,"raw_file":sh[k]["file"],"raw_bytes":sh[k]["bytes"],"raw_sha256":sh[k]["sha256"],"tok_start":a,"tok_end":a+n,"tokens":n,"sha256":h,"bytes_per_token":round(sh[k]["bytes"]/n,4),"by":"build_adam_t_stream tokenize","verified_byte_exact":True,"secs":round(time.time()-t0,1)});os.remove(tmp);os.remove(raw);dlog(f"deleted data/adam_t/stream/{sh[k]['file']} ({sh[k]['bytes']} raw UTF-8 bytes, sha256 {sh[k]['sha256']}) after verifying its BPE-4096 token shard (train.bpe4096.u16 tokens [{a},{a+n}), sha256 {h}) decodes back byte-exact");R[-1]["raw_deleted"]=True;save();p(f"token shard {k}: {n} tokens ({sh[k]['bytes']/n:.3f} B/tok), byte-exact, appended, raw deleted, {time.time()-t0:.0f}s")
 hb=os.path.join(OUT,"hold.bpe4096.u16");T["hold"]={"file":"hold.bpe4096.u16","tokens":os.path.getsize(hb)//2,"sha256":sha(hb),"raw_file":"hold.bin","raw_sha256":m["hold"]["sha256"],"verified_byte_exact":same(np.memmap(hb,np.uint16,"r"),os.path.join(OUT,"hold.bin")),"by":"ADAM-T agent (verified here)"};assert T["hold"]["verified_byte_exact"]
 T["train"]={"file":"train.bpe4096.u16","tokens":R[-1]["tok_end"],"bytes":os.path.getsize(tf),"sha256":sha(tf),"raw_bytes":sum(r["raw_bytes"] for r in R),"bytes_per_token":round(sum(r["raw_bytes"] for r in R)/R[-1]["tok_end"],4)};save()
 tb=os.path.join(OUT,"train.bin");os.path.exists(tb) and (os.remove(tb),dlog(f"deleted data/adam_t/stream/train.bin (reflink concatenation of the {len(R)} raw train shards, {sum(r['raw_bytes'] for r in R)} bytes; every shard verified byte-exact against train.bpe4096.u16 token ranges in data/adam_t/stream/tokens.json)"))
 t4=os.path.join(OUT,"tok4096");os.path.isdir(t4) and (shutil.rmtree(t4),dlog("deleted data/adam_t/stream/tok4096/ (ADAM-T agent per-shard token files 0-10, byte-identical to train.bpe4096.u16 tokens [0,%d), with its permission)"%R[min(10,len(R)-1)]["tok_end"]));os.path.isdir(TD) and not os.listdir(TD) and os.rmdir(TD)
 m["status"]="done-bpe4096";m["tokens"]=T;m["train_bin"]={"file":None,"deleted":"raw byte train shards and train.bin deleted after byte-exact token verification; see tokens and data/ks/logs/deletions.log","raw_bytes":T["train"]["raw_bytes"]};m["format"]["tokens"]="train.bpe4096.u16 / hold.bpe4096.u16: little-endian u16 BPE-4096 ids (data/clean2/bpe4096.json, crates/adam-t bpe.rs, lossless); token shard k = tokens [tok_start,tok_end) in tokens.shards; hold.bin stays raw for bpb evals; <shard>.idx.npy byte offsets refer to the deleted raw shards (provenance only)";json.dump(m,open(os.path.join(OUT,"manifest.json.part"),"w"),indent=1);os.replace(os.path.join(OUT,"manifest.json.part"),os.path.join(OUT,"manifest.json"));p(json.dumps(T["train"]))
if __name__=="__main__":tokenize() if sys.argv[1:]==["tokenize"] else main()
