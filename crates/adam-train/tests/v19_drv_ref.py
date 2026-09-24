import os,sys,json,zlib,numpy as np
H=os.environ.get("AMNI_HOME","/home/anmire/ai/Amni-Ai");sys.path[:0]=[p for p in [os.environ.get("REF_DIR",""),os.path.join(H,"scripts"),H] if p]
for k,v in {"ARCH":"cpu","QT":"1","COPY":"13","LEVELS":"21","HOPS":"4","LOG2_ENTRIES":"16","CTX":"8192","BS":"8","STEPS":"0","CORPUS":"chat3","PROF":"0"}.items():os.environ.setdefault(k,v)
os.environ.setdefault("SYNSETS",os.path.join(H,"data","big2","synsets.txt"))
m=__import__(os.environ.get("REF_SCRIPT","ray_hash_field_ti_v19"))
rows=[json.loads(l) for l in open(os.path.join(H,"data","chat3","hold_eval.jsonl"))];lim=int(os.environ.get("LIMIT","0"));rows=rows[:lim] if lim else rows
xs=[];ls=[];nbs=[];crc=[];mem={k:[] for k in ("ent","rel","vlen","stamp","val","cand","qent","qrel","qtyp")}
for i in range(0,len(rows),m.BS):
 chunk=[r["dialogue"] for r in rows[i:i+m.BS]];raws=[(t+" ").encode("ascii","replace") for t in chunk];L=max(len(r) for r in raws);x=np.full((m.BS,m.Tn),10,np.int32)
 for j,r in enumerate(raws):x[j,:len(r)]=list(r)
 m.X.from_numpy(x);m.drivers(len(chunk),L,m.TOPIC_A);n=len(chunk)*L
 w=m.W.to_numpy()[:n];idx=np.where(w!=0,m.IDX.to_numpy()[:n],0).astype(np.int32);cpb=m.CPB.to_numpy()[:n].astype(np.int32)
 crc.append([zlib.crc32(idx.tobytes()),zlib.crc32(w.astype(np.float32).tobytes()),zlib.crc32(cpb.tobytes())]);xs.append(x[:,:m.Tn].astype(np.uint8));ls.append(L);nbs.append(len(chunk))
 for k,f in (("ent",m.ENT),("rel",m.REL),("vlen",m.VLEN),("stamp",m.STAMP),("val",m.VAL),("cand",m.CAND)):mem[k].append(f.to_numpy()[:m.BS].astype(np.int64))
 for k,f in (("qent",m.QENT),("qrel",m.QREL),("qtyp",m.QTYP)):mem[k].append(f.to_numpy()[:m.BS].astype(np.int64))
 print(f"batch {i//m.BS} L={L} crc={crc[-1]}",flush=True)
np.savez(sys.argv[1],x=np.stack(xs),L=np.array(ls,np.int64),nb=np.array(nbs,np.int64),crc=np.array(crc,np.int64),**{k:np.stack(v) for k,v in mem.items()})
print("written",sys.argv[1],len(ls),"batches",flush=True)
