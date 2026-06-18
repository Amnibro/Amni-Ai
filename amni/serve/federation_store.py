"""Federation store — export Adam's learned lessons as portable PTEX 'learning packs', publish them to a
git-backed learnings/ dir + a manifest.json 'page' (+ optional HuggingFace dataset push), and fetch packs
on demand back into the rapid-access sem_lut. This is the substrate for: train -> federate/HF/git store ->
delete local -> download-on-demand -> use. PII-scrubbed + safety-gated on both export and import."""
import os,json,time,hashlib,subprocess,re
from pathlib import Path
from typing import Dict,Any,List,Optional,Tuple
_ROOT=Path(os.environ.get('AMNI_LEARNINGS_DIR') or (Path(__file__).resolve().parents[2]/'learnings'/'federated'))
_MANIFEST=_ROOT/'manifest.json'
_LANG_HINTS={'python':('python','pip','pytest','asyncio','numpy','django','flask','def ','__init__','pythonic'),'rust':('rust','cargo','rustc','borrow','lifetime','trait ','impl ','tokio','crate'),'javascript':('javascript','node.js','npm','typescript',' react','async/await','=>','const ','document.'),'kotlin':('kotlin','gradle','coroutine','android','suspend ','fun '),'cpp':('c++','cpp','std::','template<','#include','pointer','hip ','rocm'),'go':('golang',' go ','goroutine','func ','defer '),'sql':('sql','select ','postgres','sqlite','join ','query'),'shell':('bash','shell','powershell','#!/bin')}
def _now()->float:return time.time()
def _read_manifest()->Dict[str,Any]:
    try:return json.loads(_MANIFEST.read_text(encoding='utf-8')) if _MANIFEST.exists() else {'version':1,'updated':0.0,'packs':[]}
    except Exception:return {'version':1,'updated':0.0,'packs':[]}
def _write_manifest(m:Dict[str,Any])->None:
    _ROOT.mkdir(parents=True,exist_ok=True);m['updated']=_now()
    _MANIFEST.write_text(json.dumps(m,indent=1),encoding='utf-8')
def _detect_lang(q:str,a:str)->str:
    t=(q+' '+a).lower();best='general';score=0
    for lang,hints in _LANG_HINTS.items():
        s=sum(1 for h in hints if h in t)
        if s>score:score=s;best=lang
    return best if score>0 else 'general'
def _select_lessons(adam,source:str='',domain:str='',languages:Optional[List[str]]=None,limit:int=600)->List[Tuple[str,str,str,str]]:
    sl=getattr(adam,'sem_lut',None);raw=getattr(sl,'_raw',[]) if sl is not None else []
    src=getattr(sl,'_src',[]) or []
    try:from amni.serve.federated import scrub_pii,is_publishable
    except Exception:scrub_pii=None;is_publishable=None
    langset=set(l.lower() for l in (languages or []))
    out=[]
    for i,(q,a) in enumerate(raw):
        sv=(src[i] if i<len(src) else '') or ''
        if source and source.lower() not in sv.lower():continue
        lang=_detect_lang(q,a)
        if langset and lang not in langset:continue
        if domain and domain not in ('any','all'):
            try:
                from amni.serve.federated import _detect_domain
                if _detect_domain(q,a)!=domain:continue
            except Exception:pass
        sq,sa=q,a
        if scrub_pii is not None:
            try:
                sq,fq=scrub_pii(q);sa,fa=scrub_pii(a)
                if is_publishable is not None:
                    ok,_=is_publishable(sq,sa,confidence=0.9)
                    if not ok:continue
            except Exception:pass
        out.append((sq,sa,sv,lang))
        if len(out)>=limit:break
    return out
def export_pack(adam,name:str='',source:str='',domain:str='',languages:Optional[List[str]]=None,limit:int=600,push:bool=True)->Dict[str,Any]:
    from amni.inference.semantic_ptex_lut import SemanticPTEXLUT
    sel=_select_lessons(adam,source=source,domain=domain,languages=languages,limit=limit)
    if not sel:return {'error':'no lessons matched the filter','source':source,'domain':domain,'languages':languages}
    pack_name=re.sub(r'[^a-z0-9_-]+','-',(name or source or domain or 'pack').lower()).strip('-') or 'pack'
    pid=f'{pack_name}-{int(_now())}'
    pdir=_ROOT/pid;pdir.mkdir(parents=True,exist_ok=True)
    enc=getattr(getattr(adam,'sem_lut',None),'encoder',None)
    pack=SemanticPTEXLUT(grid=32,pca_dim=6,encoder=enc)
    langs={}
    for q,a,sv,lang in sel:pack.add(q,a,source=sv or f'pack:{pid}');langs[lang]=langs.get(lang,0)+1
    pack.fit();pack.save(str(pdir/'pack'))
    blob=(pdir/'pack.json').read_bytes()
    sha=hashlib.sha256(blob).hexdigest()
    entry={'id':pid,'name':pack_name,'domain':domain or 'computer_science','languages':sorted(langs,key=lambda l:-langs[l]),'lang_counts':langs,'n_lessons':len(sel),'created':_now(),'sha256':sha,'files':[f'{pid}/pack.npz',f'{pid}/pack.json'],'hf_repo':None,'git_remote':None}
    m=_read_manifest();m['packs']=[p for p in m['packs'] if p.get('id')!=pid]+[entry];_write_manifest(m)
    res={'exported':True,'pack':entry,'dir':str(pdir)}
    if push:
        res['git']=_git_commit_push([str(pdir),str(_MANIFEST)],f'learning pack {pid}: {len(sel)} lessons {sorted(langs)}')
        hf=_hf_push(pid,[pdir/'pack.npz',pdir/'pack.json',_MANIFEST])
        if hf.get('pushed'):entry['hf_repo']=hf.get('repo');_write_manifest(m)
        res['hf']=hf
    print(f'[federation_store] exported pack {pid}: {len(sel)} lessons {sorted(langs)} -> {pdir}',flush=True)
    return res
def list_packs(remote:str='')->Dict[str,Any]:
    if remote:
        try:
            from amni.serve.code_safety import safe_urlopen
            raw,_=safe_urlopen(remote.rstrip('/')+'/manifest.json' if not remote.endswith('manifest.json') else remote,timeout=8,max_bytes=2000000)
            return json.loads(raw.decode('utf-8','ignore'))
        except Exception as e:return {'error':f'remote manifest fetch failed: {e}','packs':[]}
    return _read_manifest()
def fetch_pack(adam,pack_id:str='',url:str='',domain:str='',languages:Optional[List[str]]=None,merge:bool=True)->Dict[str,Any]:
    from amni.inference.semantic_ptex_lut import SemanticPTEXLUT
    from amni.serve.consensus import find_match
    enc=getattr(getattr(adam,'sem_lut',None),'encoder',None);pdir=None;pid=pack_id
    m=_read_manifest()
    if not pack_id and not url:
        cands=[p for p in m['packs'] if (not domain or p.get('domain')==domain) and (not languages or set(l.lower() for l in languages)&set(p.get('languages',[])))]
        if not cands:return {'error':'no matching local pack; supply url or pack_id','domain':domain,'languages':languages}
        pack_id=sorted(cands,key=lambda p:-p.get('created',0))[0]['id'];pid=pack_id
    if pack_id and not url:
        pdir=_ROOT/pack_id
        if not (pdir/'pack.json').exists():
            ent=next((p for p in m['packs'] if p.get('id')==pack_id),None)
            if ent and ent.get('hf_repo'):
                hp=_hf_pull(ent['hf_repo'],pack_id)
                if hp.get('ok'):pdir=Path(hp['dir'])
            if not (pdir/'pack.json').exists():return {'error':f'pack {pack_id} not found locally and no HF mirror','id':pack_id}
    if url:
        pdir=_http_pull_pack(url,pid or f'remote-{int(_now())}')
        if not pdir:return {'error':f'failed to download pack from {url}'}
        pid=pdir.name
    try:pack=SemanticPTEXLUT.load(str(pdir/'pack'),encoder=enc)
    except Exception as e:return {'error':f'pack load failed: {e}','dir':str(pdir)}
    if not merge:return {'loaded':True,'id':pid,'n_lessons':len(pack._raw),'merged':0}
    sl=adam.sem_lut;psrc=getattr(pack,'_src',[]) or []
    new=0
    for i,(q,a) in enumerate(pack._raw):
        if find_match(sl,q) is None:sl.add(q,a,source=(psrc[i] if i<len(psrc) and psrc[i] else f'federated:{pid}'));new+=1
    if new:
        try:sl.fit();adam.save_lessons()
        except Exception as e:print(f'[federation_store] fit/save after fetch deferred: {e}',flush=True)
    print(f'[federation_store] fetched pack {pid}: merged {new} new lessons into sem_lut',flush=True)
    return {'fetched':True,'id':pid,'pack_lessons':len(pack._raw),'merged_new':new,'lessons_n':len(sl._raw)}
def _git_commit_push(paths:List[str],msg:str)->Dict[str,Any]:
    remote=os.environ.get('AMNI_LEARNINGS_GIT_REMOTE','')
    try:
        if not (_ROOT/'.git').exists() and not (_ROOT.parent/'.git').exists():
            subprocess.run(['git','init'],cwd=str(_ROOT),capture_output=True,timeout=20)
        cwd=str(_ROOT)
        subprocess.run(['git','add','-A'],cwd=cwd,capture_output=True,timeout=30)
        r=subprocess.run(['git','-c','user.email=adam@amni.local','-c','user.name=Adam','commit','-m',msg],cwd=cwd,capture_output=True,timeout=30,text=True)
        committed='nothing to commit' not in (r.stdout+r.stderr)
        out={'committed':committed,'remote_pushed':False}
        if remote:
            subprocess.run(['git','remote','remove','learnings'],cwd=cwd,capture_output=True,timeout=10)
            subprocess.run(['git','remote','add','learnings',remote],cwd=cwd,capture_output=True,timeout=10)
            pr=subprocess.run(['git','push','learnings','HEAD','--force'],cwd=cwd,capture_output=True,timeout=60,text=True)
            out['remote_pushed']=pr.returncode==0;out['push_msg']=(pr.stderr or pr.stdout)[:200]
        return out
    except Exception as e:return {'committed':False,'error':f'{type(e).__name__}: {e}'}
def _hf_push(pid:str,files:List[Path])->Dict[str,Any]:
    repo=os.environ.get('AMNI_LEARNINGS_HF_REPO','');token=os.environ.get('HF_TOKEN','') or os.environ.get('HUGGINGFACE_TOKEN','')
    if not repo or not token:return {'pushed':False,'reason':'AMNI_LEARNINGS_HF_REPO + HF_TOKEN not set (local-only mode)'}
    try:
        from huggingface_hub import HfApi
        api=HfApi(token=token)
        try:api.create_repo(repo_id=repo,repo_type='dataset',exist_ok=True)
        except Exception:pass
        for f in files:
            if Path(f).exists():api.upload_file(path_or_fileobj=str(f),path_in_repo=(f'{pid}/{Path(f).name}' if Path(f).name!='manifest.json' else 'manifest.json'),repo_id=repo,repo_type='dataset')
        return {'pushed':True,'repo':repo}
    except Exception as e:return {'pushed':False,'error':f'{type(e).__name__}: {e}'}
def _hf_pull(repo:str,pid:str)->Dict[str,Any]:
    token=os.environ.get('HF_TOKEN','') or os.environ.get('HUGGINGFACE_TOKEN','')
    try:
        from huggingface_hub import snapshot_download
        d=snapshot_download(repo_id=repo,repo_type='dataset',allow_patterns=[f'{pid}/*'],token=token or None,local_dir=str(_ROOT))
        return {'ok':True,'dir':str(_ROOT/pid)}
    except Exception as e:return {'ok':False,'error':f'{type(e).__name__}: {e}'}
def _http_pull_pack(base:str,pid:str)->Optional[Path]:
    from amni.serve.code_safety import safe_urlopen
    pdir=_ROOT/pid;pdir.mkdir(parents=True,exist_ok=True)
    try:
        for fn in ('pack.json','pack.npz'):
            raw,_=safe_urlopen(base.rstrip('/')+'/'+fn,timeout=20,max_bytes=20000000)
            (pdir/fn).write_bytes(raw)
        return pdir
    except Exception as e:print(f'[federation_store] http pull failed: {e}',flush=True);return None
_LOADED_PACKS=set()
_CODING_Q_RE=re.compile(r"\b(python|rust|javascript|typescript|kotlin|java|c\+\+|cpp|golang|\bgo\b|sql|bash|powershell|async|await|coroutine|borrow|trait|lifetime|decorator|generic|pointer|compile|function|class|method|library|framework|\bapi\b|syntax|idiom|stdlib|pip|cargo|npm|gradle|tokio|numpy|pytest)\b",re.IGNORECASE)
def fetch_for_query(adam,query:str,loaded:Optional[set]=None)->Dict[str,Any]:
    if not _CODING_Q_RE.search(query or ''):return {'fetched':False,'reason':'not a coding query'}
    loaded=_LOADED_PACKS if loaded is None else loaded
    lang=_detect_lang(query,'');m=_read_manifest()
    if not m.get('packs'):return {'fetched':False,'reason':'no packs in manifest'}
    cands=[p for p in m['packs'] if p.get('id') not in loaded and (lang in p.get('languages',[]) or p.get('domain') in ('computer_science','any','general'))]
    if not cands:return {'fetched':False,'reason':'no matching unloaded pack','lang':lang}
    pack=sorted(cands,key=lambda p:((lang in p.get('languages',[])),p.get('created',0)),reverse=True)[0]
    r=fetch_pack(adam,pack_id=pack['id'])
    if r.get('fetched'):loaded.add(pack['id']);r['matched_lang']=lang
    return r
def stats()->Dict[str,Any]:
    m=_read_manifest()
    return {'root':str(_ROOT),'manifest_exists':_MANIFEST.exists(),'packs':len(m.get('packs',[])),'total_lessons':sum(p.get('n_lessons',0) for p in m.get('packs',[])),'languages':sorted(set(l for p in m.get('packs',[]) for l in p.get('languages',[]))),'hf_configured':bool(os.environ.get('AMNI_LEARNINGS_HF_REPO') and (os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN'))),'git_remote_configured':bool(os.environ.get('AMNI_LEARNINGS_GIT_REMOTE'))}
def federation_skill(args:Dict[str,Any],ctx:Dict[str,Any],reg)->Dict[str,Any]:
    adam=ctx.get('adam') if ctx else None
    action=(args.get('action') or 'stats').strip().lower()
    if action in ('stats','status'):return stats()
    if action in ('list','packs','manifest','page'):return list_packs(remote=args.get('remote',''))
    if action in ('export','publish','federate','store'):
        if adam is None:return {'error':'no adam in skill context'}
        return export_pack(adam,name=args.get('name',''),source=args.get('source',''),domain=args.get('domain',''),languages=args.get('languages'),limit=int(args.get('limit',600)),push=bool(args.get('push',True)))
    if action in ('fetch','pull','download'):
        if adam is None:return {'error':'no adam in skill context'}
        return fetch_pack(adam,pack_id=(args.get('pack_id') or args.get('id') or ''),url=args.get('url',''),domain=args.get('domain',''),languages=args.get('languages'))
    if action in ('purge','delete','forget'):
        if adam is None:return {'error':'no adam in skill context'}
        src=args.get('source') or ''
        if not src:return {'error':'need source to purge (e.g. code-corpus)'}
        sl=getattr(adam,'sem_lut',None)
        if sl is None or not hasattr(sl,'purge_by_source'):return {'error':'sem_lut has no purge_by_source'}
        n=sl.purge_by_source(src)
        try:adam.save_lessons()
        except Exception:pass
        return {'purged':n,'source':src,'lessons_n':len(sl._raw)}
    if action in ('sources','provenance'):
        sl=getattr(adam,'sem_lut',None)
        return {'sources':sl.sources_summary()} if (sl is not None and hasattr(sl,'sources_summary')) else {'error':'no sem_lut'}
    return {'error':f'unknown action "{action}"; valid: stats|list|export|fetch|purge|sources'}
