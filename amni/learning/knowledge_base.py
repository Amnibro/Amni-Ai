"""KnowledgeBase: PTEX-encoded lossless LUT for function/text recall.

Inverts the typical SFT loop: instead of learning facts via gradient descent (slow, lossy,
GPU-bound), STORES facts as PTEX-encoded byte sequences with an address index. Adam handles
reasoning + composition; the KB handles recall.

Storage format (per page):
    pages/page_<idx:06d>.kb.ptex    raw bytes, 4096*4096*4 = 67,108,864 bytes per page
                                    (treats RGBA channels as 4-byte stream; same disk shape
                                    as ExperienceAtlas pages but used for direct text storage)

Index (JSON, root-level):
    {
        "schema_version": 1,
        "pages": [{"filename": "page_000000.kb.ptex", "used_bytes": N}, ...],
        "entries": {
            "<key>": {"page_idx": 0, "offset": 12345, "length": 678, "meta": {...}}
        },
        "n_entries": <count>,
        "created": <iso8601>,
        "tokenizer": "utf-8_bytes"
    }

Encoding: byte-level UTF-8 (no tokenization). 1 char = 1-4 bytes = 1-4 fp16/uint8 slots.
Faster + smaller than embedding-based stores for exact-recall workloads.

Usage:
    kb = KnowledgeBase('experiences/devdocs_kb')
    kb.add('python.pathlib.Path.read_text', 'Read entire file as string. Returns str. ...')
    kb.lookup('python.pathlib.Path.read_text')              # exact-key
    kb.lookup_substring('read file as string')              # full-text scan (slow at scale)
    kb.lookup_prefix('python.pathlib.')                     # range lookup via index
    kb.stats()                                              # disk + count + page utilization
"""
import json,struct,time,mmap,os,atexit
from pathlib import Path
from typing import Dict,List,Optional,Iterator,Tuple
_PAGE_W=4096
_PAGE_H=4096
_PAGE_BYTES=_PAGE_W*_PAGE_H*4
_MAGIC=b'KBPTEX01'
_AUTOSAVE_EVERY=int(os.environ.get('AMNI_KB_AUTOSAVE_EVERY','100'))
_REPLACE_RETRIES=int(os.environ.get('AMNI_KB_REPLACE_RETRIES','12'))
_REPLACE_BACKOFF=float(os.environ.get('AMNI_KB_REPLACE_BACKOFF','0.05'))
class KnowledgeBaseError(Exception):pass
class KnowledgeBase:
    def __init__(self,root_dir):
        self.root=Path(root_dir)
        self.root.mkdir(parents=True,exist_ok=True)
        self.pages_dir=self.root/'pages'
        self.pages_dir.mkdir(exist_ok=True)
        self.index_path=self.root/'index.json'
        self._dirty=False
        self._adds_since_save=0
        if self.index_path.exists():
            self.index=json.loads(self.index_path.read_text(encoding='utf-8'))
        else:
            self.index={'schema_version':1,'magic':_MAGIC.decode(),'page_w':_PAGE_W,'page_h':_PAGE_H,'pages':[],'entries':{},'n_entries':0,'tokenizer':'utf-8_bytes','created':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            self._save_index()
        self._mmaps={}
        self._writers={}
        atexit.register(self._atexit_flush)
    def _save_index(self):
        tmp=self.index_path.with_suffix('.tmp')
        payload=json.dumps(self.index,indent=2)
        last_err=None
        for i in range(_REPLACE_RETRIES):
            try:
                tmp.write_text(payload,encoding='utf-8')
                tmp.replace(self.index_path)
                self._dirty=False
                self._adds_since_save=0
                return
            except PermissionError as e:
                last_err=e
                time.sleep(_REPLACE_BACKOFF*(2**min(i,6)))
        raise KnowledgeBaseError(f'index save failed after {_REPLACE_RETRIES} retries: {last_err}')
    def flush(self):
        for w in self._writers.values():
            try:w.flush()
            except Exception:pass
        if self._dirty:self._save_index()
    def _atexit_flush(self):
        try:self.flush()
        except Exception:pass
    def _maybe_save(self):
        self._dirty=True
        self._adds_since_save+=1
        if self._adds_since_save>=_AUTOSAVE_EVERY:self._save_index()
    def _new_page(self):
        idx=len(self.index['pages'])
        fname=f'page_{idx:06d}.kb.ptex'
        path=self.pages_dir/fname
        with open(path,'wb') as f:f.write(b'\x00'*_PAGE_BYTES)
        meta={'filename':fname,'idx':idx,'used_bytes':0,'created':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        self.index['pages'].append(meta)
        self._save_index()
        return idx,path,meta
    def __enter__(self):return self
    def __exit__(self,*a):self.flush()
    def _current_page(self):
        if not self.index['pages']:return self._new_page()
        meta=self.index['pages'][-1]
        return meta['idx'],self.pages_dir/meta['filename'],meta
    def _ensure_capacity(self,n_bytes):
        idx,path,meta=self._current_page()
        if meta['used_bytes']+n_bytes>_PAGE_BYTES:return self._new_page()
        return idx,path,meta
    def _get_writer(self,idx,path):
        w=self._writers.get(idx)
        if w is None:
            w=open(path,'r+b')
            self._writers[idx]=w
        return w
    def _close_writers(self):
        for w in self._writers.values():
            try:w.flush();w.close()
            except Exception:pass
        self._writers.clear()
    def add(self,key:str,text:str,meta:Optional[Dict]=None,allow_overwrite=True):
        if not key:raise KnowledgeBaseError('empty key')
        if key in self.index['entries']:
            if not allow_overwrite:raise KnowledgeBaseError(f'key exists: {key}')
            self._mmaps.clear()
        data=text.encode('utf-8') if isinstance(text,str) else bytes(text)
        if len(data)>_PAGE_BYTES:raise KnowledgeBaseError(f'entry too large for one page: {len(data)} > {_PAGE_BYTES}')
        idx,path,page_meta=self._ensure_capacity(len(data))
        f=self._get_writer(idx,path)
        f.seek(page_meta['used_bytes'])
        f.write(data)
        offset=page_meta['used_bytes']
        page_meta['used_bytes']+=len(data)
        self.index['entries'][key]={'page_idx':idx,'offset':offset,'length':len(data),'meta':meta or {}}
        self.index['n_entries']=len(self.index['entries'])
        self._maybe_save()
        return key
    def add_batch(self,items):
        n=0
        for it in items:
            self.add(it['key'],it['text'],meta=it.get('meta'))
            n+=1
            if n%1000==0:print(f'  [kb] added {n}')
        self.flush()
        return n
    def _mmap_page(self,page_idx):
        if page_idx in self._mmaps:return self._mmaps[page_idx]
        path=self.pages_dir/self.index['pages'][page_idx]['filename']
        f=open(path,'rb')
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        self._mmaps[page_idx]=mm
        return mm
    def lookup(self,key:str)->Optional[str]:
        e=self.index['entries'].get(key)
        if e is None:return None
        mm=self._mmap_page(int(e['page_idx']))
        data=mm[int(e['offset']):int(e['offset'])+int(e['length'])]
        try:return data.decode('utf-8')
        except UnicodeDecodeError:return data.decode('utf-8',errors='replace')
    def lookup_prefix(self,prefix:str)->List[Tuple[str,str]]:
        out=[]
        for k in self.index['entries']:
            if k.startswith(prefix):
                v=self.lookup(k)
                if v is not None:out.append((k,v))
        return out
    def lookup_substring(self,needle:str,case_insensitive=True,max_results=20)->List[Tuple[str,str]]:
        if case_insensitive:needle_l=needle.lower()
        out=[]
        for k in self.index['entries']:
            v=self.lookup(k) or ''
            if case_insensitive:hit=needle_l in k.lower() or needle_l in v.lower()
            else:hit=needle in k or needle in v
            if hit:
                out.append((k,v))
                if len(out)>=max_results:break
        return out
    def keys(self)->List[str]:return list(self.index['entries'].keys())
    def __len__(self):return int(self.index['n_entries'])
    def __contains__(self,key):return key in self.index['entries']
    def stats(self)->Dict:
        n_pages=len(self.index['pages'])
        used=sum(int(p['used_bytes']) for p in self.index['pages'])
        capacity=n_pages*_PAGE_BYTES
        return {'n_entries':len(self),'n_pages':n_pages,'used_bytes':used,'capacity_bytes':capacity,'utilization':used/capacity if capacity else 0,'avg_bytes_per_entry':used/len(self) if len(self) else 0}
    def close(self):
        self.flush()
        self._close_writers()
        for mm in self._mmaps.values():
            try:mm.close()
            except Exception:pass
        self._mmaps.clear()
