import os,sys,json,struct,importlib
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from amni.compute import wikifact as WF
E=[(31,"Belgium",["Kingdom of Belgium"],"Belgium",220),(239,"Brussels",["City of Brussels"],"Brussels",180),(42,"Douglas Adams",["Douglas Noel Adams"],"Douglas Adams",120),(350,"Cambridge",[],"Cambridge",150),(5,"human",[],"",0),(11424,"film",[],"",0),(7366,"song",[],"",0),(100,"The Big Bang Theory",["TBBT"],"The Big Bang Theory",90),(101,"Kunal Nayyar",[],"Kunal Nayyar",60),(102,"Rajesh Koothrappali",["Raj Koothrappali","Raj"],"",0),(103,"Jim Parsons",[],"Jim Parsons",70),(104,"Sheldon Cooper",[],"Sheldon Cooper",40),(200,"Thriller",[],"Thriller (song)",40),(201,"Michael Jackson",["MJ"],"Michael Jackson",250),(202,"Rod Temperton",[],"Rod Temperton",30),(203,"Thriller",[],"Thriller (album)",80),(11573,"metre",[],"",0),(300,"Titanic",[],"Titanic (1997 film)",100),(301,"James Cameron",[],"James Cameron",120),(302,"Titanic",[],"Titanic (1953 film)",8),(303,"Jean Negulesco",[],"Jean Negulesco",10)]
def T(y,m,d,p):return ((y+(1<<39))<<24)|(m<<16)|(d<<8)|(p<<1)
F={31:[(36,0,1,239,[]),(1082,0,3,11825551.0,[(585,2,T(2023,1,1,11))]),(1082,1,3,11959000.0,[(585,2,T(2025,1,1,11))])],42:[(31,0,1,5,[]),(569,0,2,T(1952,3,11,11),[]),(19,0,1,350,[]),(570,0,2,T(2001,5,11,11),[]),(2048,0,3,1.96,[])],100:[(161,0,1,103,[(453,1,104)]),(161,0,1,101,[(453,1,102)])],200:[(31,0,1,7366,[]),(175,0,1,201,[]),(86,0,1,202,[]),(577,0,2,T(1983,11,12,11),[]),(577,0,2,T(1982,11,30,11),[])],203:[(175,0,1,201,[]),(577,0,2,T(1982,11,30,11),[])],300:[(31,0,1,11424,[]),(57,0,1,301,[]),(577,0,2,T(1997,12,19,11),[])],302:[(31,0,1,11424,[]),(57,0,1,303,[])]}
def mini(d:Path)->Path:
 E2=sorted(E);q2i={q:i for i,(q,*_) in enumerate(E2)};fs,fp,fr,fk,fv,fx,fqo,qp,qk,qv,efs,efn=[],[],[],[],[],[],[],[],[],[],[],[]
 for q,*_ in E2:
  efs.append(len(fs));efn.append(len(F.get(q,[])))
  for p,r,k,v,Q in F.get(q,[]):
   fs.append(q2i[q]);fp.append(p);fr.append(r);fk.append(k);fx.append(q2i[11573] if p==2048 else 0xFFFFFFFF);fqo.append(len(qp));fv.append(q2i[v] if k==1 else struct.unpack("<Q",struct.pack("<d",v))[0] if k==3 else v)
   for a,b,c in Q:qp.append(a);qk.append(b);qv.append(q2i[c] if b==1 else c)
 fqo.append(len(qp));A=lambda n,x,t:np.save(d/f"{n}.npy",np.asarray(x,t))
 [A(n,x,t) for n,x,t in (("e_qid",[e[0] for e in E2],np.uint32),("e_sl",[e[4] for e in E2],np.uint16),("e_fs",efs,np.uint32),("e_fn",efn,np.uint32),("f_s",fs,np.uint32),("f_p",fp,np.uint16),("f_r",fr,np.uint8),("f_k",fk,np.uint8),("f_v",fv,np.uint64),("f_x",fx,np.uint32),("f_qo",fqo,np.uint32),("q_p",qp,np.uint16),("q_k",qk,np.uint8),("q_v",qv,np.uint64))]
 lo,to=[0],[0];(d/"e_lab.bin").write_bytes(b"".join(e[1].encode() for e in E2));(d/"e_tit.bin").write_bytes(b"".join(e[3].encode() for e in E2));(d/"str.bin").write_bytes(b"")
 [lo.append(lo[-1]+len(e[1].encode())) or to.append(to[-1]+len(e[3].encode())) for e in E2];A("e_lo",lo,np.uint64);A("e_to",to,np.uint64)
 (d/"names.tsv").write_text("".join(f"{i}\t0\t{e[1]}\n"+"".join(f"{i}\t1\t{a}\n" for a in e[2])+(f"{i}\t2\t{e[3]}\n" if e[3] and e[3]!=e[1] else "") for i,e in enumerate(E2)),encoding="utf-8")
 WF.build(d);return d
@pytest.fixture(scope="module")
def wf(tmp_path_factory):
 d=mini(tmp_path_factory.mktemp("wf"));os.environ["AMNI_WIKIFACT_DIR"]=str(d);m=importlib.reload(WF);yield m;os.environ.pop("AMNI_WIKIFACT_DIR",None);importlib.reload(WF)
def test_norm_hash_dates():
 assert WF.nrm("Charlotte's Web!")=="charlottes web" and WF.nrm("Pokémon  &  Me")=="pokemon and me" and WF.h64("a")!=WF.h64("b")
 assert WF.fdate(T(1952,3,11,11))[:2]==["March 11, 1952","11 March 1952"] and WF.fdate(T(1952,3,11,11),True)==["1952"] and WF.fdate(T(-480,0,0,9))==["480 BC"] and WF.fdate(T(1990,0,0,8))==["1990s"]
 assert WF.fnum(2117522.0)[0]=="2,117,522" and WF.fnum(1.96,"metre")[0]=="1.96 metre" and WF.lem("sang")=="sing" and WF.lem("born")=="bear"
def test_store_build(wf):
 S=wf.store();assert S is not None and S.man["license"].startswith("CC0") and S.ix(42)>=0 and S.name(S.ix(42))=="Douglas Adams" and S.man["prop_facts"]["P31"]==4 and sum(S.man["prop_facts"].values())==len(S.f_p)
 assert {e for e,_ in S.ents("raj koothrappali")}=={S.ix(102)} and "tbbt" in S.names(S.ix(100)) and {e for e,_ in S.ents("thriller")}=={S.ix(200),S.ix(203)}
def test_answers(wf):
 a=lambda q:(wf.answer(q) or (None,))[0]
 assert a("What is the capital of Belgium?")=="Brussels"
 assert a("When was Douglas Adams born?")=="March 11, 1952" and a("what year was douglas adams born")=="1952"
 assert a("where was douglas adams born")=="Cambridge" and a("when did douglas adams die")=="May 11, 2001"
 assert a("who plays raj koothrappali on big bang theory")=="Kunal Nayyar" and a("who played sheldon cooper in the big bang theory")=="Jim Parsons"
 assert a("who did jim parsons play in the big bang theory")=="Sheldon Cooper"
 assert a("who sang thriller")=="Michael Jackson" and a("who wrote the song thriller")=="Rod Temperton"
 assert wf.ask("what is the population of belgium")["answer"]=="11,959,000" and wf.ask("how many people live in belgium")["temporal"] and a("what is the population of belgium") is None
 assert a("who directed titanic")=="James Cameron" and a("when did thriller come out")=="November 30, 1982"
def test_declines(wf):
 for q in ("who wants a heart in the wizard of oz","what is the capital of atlantis","who plays the girl hyena in the big bang theory","why is the sky blue","","who is the voice of batman in gotham"):
  assert wf.answer(q) is None
def test_answer_contract(wf):
 r=wf.answer("what is the capital of belgium");assert isinstance(r,tuple) and len(r)==3 and 0.5<=r[1]<=1 and "Wikidata Q31 P36" in r[2] and "CC0" in r[2]
 assert wf.ask("who played sheldon cooper in the big bang theory")["rule"]=="cast"
def test_no_store_is_inert(tmp_path):
 os.environ["AMNI_WIKIFACT_DIR"]=str(tmp_path/"none");m=importlib.reload(WF)
 try:assert m.store() is None and m.answer("what is the capital of belgium") is None
 finally:os.environ.pop("AMNI_WIKIFACT_DIR",None);importlib.reload(WF)
@pytest.mark.skipif(not (WF.DIR/"a_h.npy").exists(),reason="full store not built")
def test_full_store_probes():
 m=importlib.reload(WF);a=lambda q:(m.answer(q) or (None,))[0]
 assert a("When was Douglas Adams born?")=="March 11, 1952" and a("what is the capital of belgium")=="Brussels" and a("who directed jurassic park")=="Steven Spielberg"
