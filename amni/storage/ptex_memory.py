import os,json,time,hashlib,mmap,re
from pathlib import Path
_MAGIC=b'PTEX_MEM_V1\x00'
class PtexMemoryAtlas:
    def __init__(self,path):
        self.path=Path(path)
        self.idx_path=self.path.with_suffix('.idx.json')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self._mmap=None;self._fd=None
        self._load_or_init()
    def _load_or_init(self):
        if self.path.exists() and self.idx_path.exists():
            with open(self.idx_path) as f:self._index=json.load(f)
        else:
            self._index={'magic':_MAGIC.decode('latin-1'),'entries':[],'total_bytes':len(_MAGIC),'created':time.time()}
            with open(self.path,'wb') as f:f.write(_MAGIC)
            self._save_index()
    def _save_index(self):
        tmp=self.idx_path.with_suffix('.tmp')
        with open(tmp,'w') as f:json.dump(self._index,f,separators=(',',':'))
        os.replace(tmp,self.idx_path)
    def _invalidate_mmap(self):
        if self._mmap is not None:self._mmap.close();self._mmap=None
        if self._fd is not None:self._fd.close();self._fd=None
    def _ensure_mmap(self):
        if self._mmap is None and self.path.stat().st_size>0:
            self._fd=open(self.path,'rb')
            self._mmap=mmap.mmap(self._fd.fileno(),0,access=mmap.ACCESS_READ)
    def append(self,text,meta=None):
        data=text.encode('utf-8') if isinstance(text,str) else bytes(text)
        pad=(4-len(data)%4)%4
        padded=data+b'\x00'*pad
        offset=self._index['total_bytes']
        with open(self.path,'ab') as f:f.write(padded)
        eid=len(self._index['entries'])
        entry={'id':eid,'offset':offset,'len':len(data),'sha':hashlib.sha256(data).hexdigest()[:16],'ts':time.time()}
        if meta:entry['meta']=meta
        self._index['entries'].append(entry)
        self._index['total_bytes']+=len(padded)
        self._save_index()
        self._invalidate_mmap()
        return eid
    def read(self,entry_id):
        if entry_id<0 or entry_id>=len(self._index['entries']):return None
        self._ensure_mmap()
        e=self._index['entries'][entry_id]
        data=bytes(self._mmap[e['offset']:e['offset']+e['len']])
        return data.decode('utf-8',errors='replace')
    def read_raw(self,entry_id):
        if entry_id<0 or entry_id>=len(self._index['entries']):return None
        self._ensure_mmap();e=self._index['entries'][entry_id]
        return bytes(self._mmap[e['offset']:e['offset']+e['len']])
    def read_meta(self,entry_id):
        if entry_id<0 or entry_id>=len(self._index['entries']):return None
        return self._index['entries'][entry_id].get('meta')
    def iter_all(self):
        for e in self._index['entries']:yield e['id'],self.read(e['id']),e.get('meta')
    def iter_recent(self,n=10):
        for e in self._index['entries'][-n:]:yield e['id'],self.read(e['id']),e.get('meta')
    def filter(self,pred):
        out=[]
        for e in self._index['entries']:
            if pred(e):out.append((e['id'],self.read(e['id']),e.get('meta')))
        return out
    def search_words(self,query_words,exclude_meta_subjects=(),max_results=10,min_overlap=2):
        qw=set(w.lower() for w in query_words if len(w)>2)
        if not qw:return []
        out=[]
        for e in self._index['entries']:
            m=e.get('meta') or {}
            if m.get('subject') in exclude_meta_subjects:continue
            txt=self.read(e['id'])
            if not txt:continue
            tw=set(re.findall(r"[a-z0-9]{3,}",txt.lower()))
            overlap=len(qw&tw)
            if overlap>=min_overlap:
                out.append((overlap,e['id'],txt,m))
        out.sort(key=lambda x:-x[0])
        return out[:max_results]
    def __len__(self):return len(self._index['entries'])
    def close(self):self._invalidate_mmap()
    def stats(self):
        return {'n_entries':len(self._index['entries']),'total_bytes':self._index['total_bytes'],'path':str(self.path),'idx_path':str(self.idx_path)}
class PtexDeltaWriter:
    def __init__(self,learnings_dir):
        self.dir=Path(learnings_dir)
        self.dir.mkdir(parents=True,exist_ok=True)
        self._atlas=PtexMemoryAtlas(self.dir/'memory.ptex')
        self._subjects_seen=set()
        for _,_,m in self._atlas.iter_all():
            if m and 'subject' in m:self._subjects_seen.add(m['subject'])
    def write_delta(self,subject,facts,sources=None,coverage=1.0):
        if not facts:return {'ok':False,'reason':'no facts'}
        n=0
        for f in facts:
            txt=f.get('text','') if isinstance(f,dict) else str(f)
            if not txt or len(txt.strip())<8:continue
            meta={'subject':subject,'sources':sources or [],'coverage':float(coverage)}
            if isinstance(f,dict):
                for k,v in f.items():
                    if k!='text':meta[k]=v
            self._atlas.append(txt,meta=meta)
            n+=1
        if n>0:self._subjects_seen.add(subject)
        return {'ok':n>0,'subject':subject,'n_facts':n,'version':1}
    def all_subjects(self):
        seen=set()
        for _,_,m in self._atlas.iter_all():
            if m and 'subject' in m:seen.add(m['subject'])
        return sorted(seen)
    def read_delta(self,subject):
        items=self._atlas.filter(lambda e:(e.get('meta') or {}).get('subject')==subject)
        if not items:return None
        class _Page:
            def __init__(self,subject,items):
                self.subject=subject
                self.facts=[{'text':t,**(m or {})} for _,t,m in items]
                self.version=len(items)
                self.sources=[]
                self.coverage=1.0
        return _Page(subject,items)
    def search_words(self,query_words,exclude_subjects=(),max_results=10,min_overlap=2):
        return self._atlas.search_words(query_words,exclude_meta_subjects=tuple(exclude_subjects),max_results=max_results,min_overlap=min_overlap)
    @property
    def _delta_index(self):
        out={}
        for s in self.all_subjects():
            p=self.read_delta(s)
            if p:out[s]=p
        return out
    def stats(self):return self._atlas.stats()
