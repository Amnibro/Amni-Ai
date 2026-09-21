"""Ray mipmap for code: type → shape → related pages → stitch → sandbox rewrite → post.

LOD 0 is the named page. Coarser LODs take every Nth ray cell, then adjacent python
pages, so neighboring concepts sit in the same blend, the way a texture mipmap
averages a neighborhood instead of a single texel.
"""
from __future__ import annotations
import ast,re
from amni.compute.ptex_1t_store import CODE_SUBRANGES_1T
from amni.compute.resident_code import (
 _STOP,_WEAK,extract_python_defs,score_src,coverage_ok,walk_query_cells,python_pages_for_cells,_defs_from_pages,
 retrieve_resident_code,
)

_LO, _HI = 0, 3584

def problem_kind(query:str,anchors:list)->str:
 q=(" "+(query or "")+" "+" ".join(anchors or [])+" ").lower()
 if any(k in q for k in (" dijkstra "," shortest "," bfs "," dfs "," graph "," path ")):return "graph"
 if any(k in q for k in (" knapsack "," subsequence "," edit "," dynamic "," dp "," backpack ")):return "dp"
 if any(k in q for k in (" gcd "," divisor "," modulo "," euclid ")):return "euclid"
 if any(k in q for k in (" tree "," heap "," stack "," queue "," cache "," rotation ")):return "struct"
 if any(k in q for k in (" palindrome "," anagram "," string ")):return "string"
 return "fn"

def output_format(query:str,anchors:list,kind:str)->dict:
 q=(query or "").lower()
 shape="class" if ("class" in q or kind=="struct") else "def"
 clean=[a.lower() for a in (anchors or []) if a.lower() not in _STOP and a.lower() not in _WEAK and len(a)>=3]
 name=""
 for w in ("knapsack","dijkstra","gcd","longest_increasing_subsequence"):
  if w in q.replace(" ","_") or w.replace("_"," ") in q:name=w;break
 if not name:
  name="_".join(clean[:3]) if shape=="def" else "".join(x.capitalize() for x in clean[:3])
 if not name:name="solve" if shape=="def" else "Solution"
 if kind=="graph":ret="mapping of node to distance or a path"
 elif kind=="dp":ret="numeric optimum or recovered sequence"
 elif kind=="euclid":ret="integer"
 elif kind=="struct":ret="object with the asked methods"
 else:ret="return value of the stitched function"
 return {"shape":shape,"name":name,"kind":kind,"returns":ret}

def stamp(src:str)->set:
 flags=set();low=(src or "").lower()
 if "heapq" in low or "heappush" in low or "priority" in low:flags.add("priority")
 if "graph" in low or "neighbor" in low or "edge" in low:flags.add("graph")
 if "while" in low and "%" in src:flags.add("euclid")
 if src.lstrip().startswith("class "):flags.add("struct")
 try:
  t=ast.parse(src)
  nfor=sum(1 for n in ast.walk(t) if isinstance(n,ast.For))
  if nfor>=2:flags.add("nested")
 except Exception:
  pass
 if "dp" in low or ("nested" in flags and any(k in low for k in ("max(","min("))):flags.add("dp")
 return flags

_KIND_FLAGS={"graph":{"graph","priority"},"dp":{"dp","nested"},"euclid":{"euclid"},"struct":{"struct"},"string":set(),"fn":set()}

def _clamp_pages(pages)->list:
 out=[];seen=set()
 for p in pages:
  p=int(p)
  if p<_LO:p=_LO
  if p>=_HI:p=_HI-1
  if p not in seen:
   seen.add(p);out.append(p)
 return out

def mipmap_pages(query:str,anchors:list,store,lam,a:int,c:int,nwalk:int)->dict:
 cells=walk_query_cells(query,lam,a,c,max(8,nwalk))
 coarse=cells[::max(1,len(cells)//3)] or cells
 lod0=python_pages_for_cells(coarse)
 lod1=python_pages_for_cells(cells)
 lod2=[]
 for p in lod1:
  lod2.extend([p-2,p-1,p,p+1,p+2])
 lod3=[]
 try:
  from amni.compute.routine_bank import python_page_for_key,ROUTINES
  for a0 in (anchors or []):
   if len(a0)>=3:lod0.append(python_page_for_key(a0.lower()))
  kind=problem_kind(query,anchors)
  for rec in ROUTINES:
   tag=" ".join(rec.get("tags") or ())
   if kind=="graph" and "dijkstra" in rec["name"]:lod3.append(python_page_for_key(rec["name"]))
   elif kind=="dp" and rec["name"] in ("knapsack","longest_increasing_subsequence"):lod3.append(python_page_for_key(rec["name"]))
   elif kind=="euclid" and rec["name"]=="gcd":lod3.append(python_page_for_key(rec["name"]))
   elif kind=="struct" and any(t in (rec.get("tags") or ()) for t in ("tree","rotate","bst","node","struct")):lod3.append(python_page_for_key(rec["name"]))
 except Exception:
  pass
 return {
  "cells":cells,
  "lod0":_clamp_pages(lod0),
  "lod1":_clamp_pages(lod1),
  "lod2":_clamp_pages(lod2),
  "lod3":_clamp_pages(lod3),
 }

def _pool(store,pages:list,per_page:int=6,cap:int=96)->list:
 rows=[];seen=set()
 for src,p in _defs_from_pages(store,pages[:80]):
  m=re.match(r"(?:def|class)\s+(\w+)",src)
  nm=m.group(1) if m else src[:40]
  if nm in seen:continue
  seen.add(nm);rows.append((src,p))
  if len(rows)>=cap:break
 return rows

_FAMILY_NEEDLES={
 "graph":(b"def dijkstra",b"def bfs",b"def dfs",b"heappush",b"def shortest"),
 "dp":(b"def knapsack",b"dp =",b"def lis",b"subsequence"),
 "euclid":(b"def gcd",b"a % b"),
 "struct":(b"def rotate",b"left_rotate",b"right_rotate",b"class Node",b"def insert",b"def delete",b"red_black",b"redblack"),
 "string":(b"def palindrome",b"s[::-1]"),
 "fn":(),
}

def family_mmap(store,kind:str,cap:int=16)->list:
 buf=getattr(store,"mmap_obj",None)
 if not buf or not hasattr(store,"get_page_offset"):return []
 base=store.get_page_offset(0);dim=int(store.page_dim)
 rows=[];seen=set()
 for needle in _FAMILY_NEEDLES.get(kind) or ():
  pos=0;n=0
  while n<8:
   idx=buf.find(needle,pos)
   if idx<0:break
   n+=1;pos=idx+max(1,len(needle))
   if idx<base:continue
   p=(idx-base)//dim
   if p<_LO or p>=_HI:continue
   sl=buf[idx:idx+3500].decode("latin1","ignore")
   for src in extract_python_defs(sl):
    m=re.match(r"(?:def|class)\s+(\w+)",src)
    if not m or m.group(1) in seen:continue
    seen.add(m.group(1));rows.append((src,p))
    if len(rows)>=cap:return rows
 return rows

def _loads(src:str)->set:
 try:
  t=ast.parse(src)
 except Exception:
  return set()
 defined=set()
 for n in ast.walk(t):
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):defined.add(n.name)
  elif isinstance(n,ast.arg):defined.add(n.arg)
  elif isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store):defined.add(n.id)
 return {n.id for n in ast.walk(t) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id not in defined}

def rename_top(src:str,new_name:str)->str:
 return re.sub(r"(?m)^(def|class)\s+[A-Za-z_]\w*",r"\1 "+new_name,src,count=1)

def stitch(primary:str,pool:list,fmt:dict)->str:
 name=fmt.get("name") or ""
 kind=fmt.get("kind") or ""
 m_prim=re.search(r"(?:def|class)\s+([A-Za-z_]\w*)",primary)
 prim_name=m_prim.group(1) if m_prim else ""
 if kind=="struct" or fmt.get("shape")=="class" or (prim_name and name and (prim_name==name or name in prim_name or prim_name in name)):
  body=primary
 else:
  body=rename_top(primary,name) if name else primary
 have=set()
 try:
  t=ast.parse(body)
  have={n.name for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 except Exception:
  pass
 need=_loads(body)
 by_name={}
 for src,_p in pool:
  m=re.match(r"(?:def|class)\s+(\w+)",src)
  if m:by_name.setdefault(m.group(1),src)
 helpers=[]
 prefer=("Node","left_rotate","right_rotate","insert","delete") if kind=="struct" else ()
 for nm in prefer:
  if nm in have or nm not in by_name:continue
  helpers.append(by_name[nm]);have.add(nm)
 for src,_p in pool:
  m=re.match(r"(?:def|class)\s+(\w+)",src)
  if not m:continue
  nm=m.group(1)
  if nm in have or nm==name:continue
  if nm in need:
   helpers.append(src);have.add(nm)
   if len(helpers)>=6:break
 blob="\n\n".join(helpers+[body])
 ast.parse(blob)
 return blob

def _page_text(store,p:int)->str:
 try:
  return store.read_tile_mmap(int(p),tile_size=int(store.page_dim)).decode("latin1","ignore")
 except Exception:
  return ""

def harvest_tests(store,pages:list,fn:str,old_names:list)->str:
 from amni.compute.routine_bank import extract_tests
 chunks=[]
 for p in pages[:16]:
  t=extract_tests(_page_text(store,p))
  if t:chunks.append(t)
 text="\n".join(chunks)
 for old in old_names:
  if old and old!=fn:
   text=re.sub(r"\b"+re.escape(old)+r"\b",fn,text)
 filtered=[]
 for ln in text.splitlines():
  s=ln.strip()
  if not s:continue
  if "is not None" in s and s.startswith("assert "):continue
  if fn.lower() not in s.lower():continue
  filtered.append(s)
 text="\n".join(filtered)
 lines=[]
 for ln in text.splitlines():
  s=ln.strip()
  if not s:continue
  if "is not None" in s and s.startswith("assert "):continue
  lines.append(s)
 seen=set();out=[]
 for ln in lines:
  if ln in seen:continue
  seen.add(ln);out.append(ln)
 return "\n".join(out[:12])

def compose_code(query:str,anchors:list=None,store=None,lam=None,a:int=3,c:int=17,nwalk:int=24)->dict:
 kind=problem_kind(query,anchors or [])
 fmt=output_format(query,anchors or [],kind)
 miss={"code":None,"via":"miss","page":None,"score":0,"cells":[],"n_cands":0,"parse_ok":False,"harness":"","kind":kind,"format":fmt,"lods":{},"ok":False,"cycles":0,"stderr":""}
 if store is None or not getattr(store,"mmap_obj",None):
  return miss
 rec=retrieve_resident_code(query,anchors,store,lam=lam,a=a,c=c,nwalk=nwalk)
 pages=mipmap_pages(query,anchors or [],store,lam,a,c,nwalk)
 all_pages=pages["lod0"]+pages["lod1"]+pages["lod2"]+pages["lod3"]
 pool=_pool(store,pages["lod3"]+pages["lod0"])+family_mmap(store,kind)+_pool(store,pages["lod1"]+pages["lod2"])
 seen=set();dedup=[]
 for src,p in pool:
  m=re.match(r"(?:def|class)\s+(\w+)",src)
  nm=m.group(1) if m else src[:24]
  if nm in seen:continue
  seen.add(nm);dedup.append((src,p))
 pool=dedup
 flags_need=_KIND_FLAGS.get(kind,set())
 ranked=[]
 for src,p in pool:
  sc=score_src(src,anchors)
  st=stamp(src)
  if flags_need and (st & flags_need):sc+=8
  if kind=="struct" and any(k in src.lower() for k in ("rotate","rotation","red_black","redblack","rbnode","left_rotate","right_rotate")):sc+=14
  if not coverage_ok(src,anchors):sc=0
  ranked.append((sc,src,p,st))
 ranked.sort(key=lambda x:x[0],reverse=True)
 primary=None;page=None;via="blend";score=0
 if rec.get("code") and int(rec.get("score") or 0)>=12 and coverage_ok(rec["code"],anchors):
  primary=rec["code"];page=rec.get("page");via=rec.get("via") or "ray";score=int(rec.get("score") or 0)
 elif ranked and ranked[0][0]>=4:
  score,primary,page,_st=ranked[0]
  via="mipmap"
 if not primary:
  return {**miss,"cells":pages["cells"],"n_cands":len(pool),"lods":{k:len(pages[k]) for k in ("lod0","lod1","lod2","lod3")}}
 old=[]
 m=re.match(r"(?:def|class)\s+(\w+)",primary)
 if m:old.append(m.group(1))
 try:
  code=stitch(primary,[(s,p) for _sc,s,p,_st in ranked],fmt)
 except Exception:
  code=primary
 harness=rec.get("harness") or ""
 if rec.get("code")!=primary:
  harness=""
 if not harness:
  test_pages=[page] if page is not None else []
  test_pages+=pages.get("lod3") or []
  keep=fmt.get("shape")!="def" or kind=="struct"
  harness=harvest_tests(store,test_pages,(old[0] if (keep and old) else fmt["name"]),[] if keep else old)
 ok=False;cycles=0;err=""
 if harness:
  try:
   from amni.compute.routine_bank import verify_code
   vr=verify_code(code,harness)
   if vr.get("ok"):
    ok=True
   else:
    err=(vr.get("stderr") or "")[:240]
    from amni.compute.code_debugger_engine import CodeDebuggerEngine
    dbg=CodeDebuggerEngine(timeout=3)
    out=dbg.debug_and_converge(query,code=code,harness=harness,max_cycles=2)
    code=out.get("code") or code
    ok=bool(out.get("verified"))
    cycles=int(out.get("cycles") or 0)
    if not ok and out.get("log"):
     err=(out["log"][-1].get("stderr") or err)[:240]
  except Exception as e:
   err=type(e).__name__+":"+err
 try:
  ast.parse(code);parse_ok=True
 except Exception:
  parse_ok=False
 return {
  "code":code,"via":via,"page":page,"score":score,"cells":pages["cells"],"n_cands":len(pool),
  "parse_ok":parse_ok,"harness":harness or "","kind":kind,"format":fmt,
  "lods":{k:len(pages[k]) for k in ("lod0","lod1","lod2","lod3")},
  "ok":ok,"cycles":cycles,"stderr":err,
 }
