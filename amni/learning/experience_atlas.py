import json,struct,time,hashlib,uuid,numpy as np
from pathlib import Path
from typing import List,Optional,Dict,Iterable,Iterator
_MAGIC=b'EXPATLAS'
_VERSION=1
_PAGE_W=4096
_PAGE_H=4096
_PAGE_BYTES=_PAGE_W*_PAGE_H*4
_REC_HEADER_FMT='<IIBBBI'
_REC_HEADER_SZ=struct.calcsize(_REC_HEADER_FMT)
class ExperienceError(Exception):pass
class ExperienceRecord:
    __slots__=('rec_id','timestamp','outcome','subject_id','reserved','prompt','response','system')
    def __init__(self,rec_id,timestamp,outcome,subject_id,reserved,prompt,response,system=''):
        self.rec_id=rec_id;self.timestamp=timestamp;self.outcome=outcome
        self.subject_id=subject_id;self.reserved=reserved
        self.prompt=prompt;self.response=response;self.system=system
    def __repr__(self):return f'<Exp #{self.rec_id} subj={self.subject_id} outcome={self.outcome} prompt={self.prompt[:30]!r}...>'
class ExperienceAtlas:
    def __init__(self,root_dir,subject='global'):
        self.root_dir=Path(root_dir);self.subject=subject
        self.subject_dir=self.root_dir/'experiences'/subject
        self.subject_dir.mkdir(parents=True,exist_ok=True)
        self.index_path=self.subject_dir/'index.json'
        self.pages_dir=self.subject_dir/'pages'
        self.pages_dir.mkdir(exist_ok=True)
        if self.index_path.exists():
            self.index=json.loads(self.index_path.read_text(encoding='utf-8'))
        else:
            self.index={'subject':subject,'magic':'EXPATLAS','version':_VERSION,'page_w':_PAGE_W,'page_h':_PAGE_H,'next_rec_id':0,'pages':[],'created':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            self._save_index()
    def _save_index(self):
        tmp=self.index_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.index,indent=2),encoding='utf-8')
        tmp.replace(self.index_path)
    def _current_page_path(self):
        if not self.index['pages']:return None
        return self.pages_dir/self.index['pages'][-1]['filename']
    def _new_page(self):
        idx=len(self.index['pages'])
        fname=f'page_{idx:06d}.exp.ptex'
        path=self.pages_dir/fname
        with open(path,'wb') as f:f.write(b'\x00'*_PAGE_BYTES)
        meta={'filename':fname,'idx':idx,'records':[],'used_bytes':0,'created':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        self.index['pages'].append(meta)
        self._save_index()
        return path,meta
    def append(self,prompt,response,outcome=0,subject_id=0,timestamp=None,system=''):
        timestamp=int(timestamp if timestamp is not None else time.time())
        sys_b=system.encode('utf-8') if isinstance(system,str) else bytes(system or b'')
        prompt_b=prompt.encode('utf-8') if isinstance(prompt,str) else bytes(prompt)
        response_b=response.encode('utf-8') if isinstance(response,str) else bytes(response)
        body=struct.pack('<I',len(sys_b))+sys_b+prompt_b+b'\x00\x00\x00\x00'+response_b
        if len(body)>0xFFFFFFFF:raise ExperienceError(f'record body too large {len(body)}')
        if len(prompt_b)>0xFFFFFFFF:raise ExperienceError(f'prompt too large {len(prompt_b)}')
        rec_id=int(self.index['next_rec_id']);self.index['next_rec_id']=rec_id+1
        header=struct.pack(_REC_HEADER_FMT,rec_id,timestamp,int(outcome)&0xFF,int(subject_id)&0xFF,0,len(prompt_b))
        rec=header+body
        rec_size=len(rec)+4
        page_path=self._current_page_path()
        page_meta=self.index['pages'][-1] if self.index['pages'] else None
        if page_meta is None or page_meta['used_bytes']+rec_size>_PAGE_BYTES:
            page_path,page_meta=self._new_page()
        with open(page_path,'r+b') as f:
            f.seek(page_meta['used_bytes'])
            f.write(struct.pack('<I',len(rec)))
            f.write(rec)
        page_meta['records'].append({'rec_id':rec_id,'offset':page_meta['used_bytes'],'size':rec_size})
        page_meta['used_bytes']+=rec_size
        self._save_index()
        return rec_id
    def __iter__(self):return self.iter_records()
    def iter_records(self):
        for page_meta in self.index['pages']:
            page_path=self.pages_dir/page_meta['filename']
            with open(page_path,'rb') as f:
                for rec_meta in page_meta['records']:
                    f.seek(int(rec_meta['offset']))
                    sz=struct.unpack('<I',f.read(4))[0]
                    blob=f.read(sz)
                    yield self._parse_record(blob)
    def _parse_record(self,blob):
        rec_id,ts,outcome,subj_id,reserved,prompt_len=struct.unpack(_REC_HEADER_FMT,blob[:_REC_HEADER_SZ])
        body=blob[_REC_HEADER_SZ:]
        if len(body)>=4:
            sys_len_bytes=body[:4]
            sys_len=struct.unpack('<I',sys_len_bytes)[0]
            if 4+sys_len+prompt_len+4<=len(body):
                system=body[4:4+sys_len].decode('utf-8',errors='replace')
                rest=body[4+sys_len:]
            else:
                system='';rest=body
        else:
            system='';rest=body
        prompt=rest[:prompt_len].decode('utf-8',errors='replace')
        response=rest[prompt_len+4:].decode('utf-8',errors='replace')
        rec=ExperienceRecord(rec_id,ts,outcome,subj_id,reserved,prompt,response)
        rec.system=system
        return rec
    def __len__(self):return int(self.index['next_rec_id'])
    def stats(self):
        return {'subject':self.subject,'n_records':len(self),'n_pages':len(self.index['pages']),'total_bytes':sum(p['used_bytes'] for p in self.index['pages']),'avg_bytes_per_record':(sum(p['used_bytes'] for p in self.index['pages'])/max(1,len(self))) if len(self) else 0}
    def to_records_list(self,outcomes_filter=None):
        out=[]
        for r in self.iter_records():
            if outcomes_filter is not None and r.outcome not in outcomes_filter:continue
            out.append({'prompt':r.prompt,'response':r.response,'system':getattr(r,'system','') or '','category':self.subject,'rec_id':r.rec_id,'outcome':r.outcome,'timestamp':r.timestamp})
        return out
    @classmethod
    def list_subjects(cls,root_dir):
        d=Path(root_dir)/'experiences'
        if not d.exists():return []
        return [p.name for p in d.iterdir() if p.is_dir()]
    def export_bundle(self,out_path,contributor_id=None,note=''):
        contributor_id=contributor_id or f'anon-{uuid.uuid4().hex[:8]}'
        records=list(self.iter_records())
        n=len(records)
        bundles=[]
        chunks=[]
        byte_offset=0
        for r in records:
            sys_b=(getattr(r,'system','') or '').encode('utf-8')
            blob=struct.pack(_REC_HEADER_FMT,r.rec_id,r.timestamp,r.outcome,r.subject_id,r.reserved,len(r.prompt.encode('utf-8')))
            blob+=struct.pack('<I',len(sys_b))+sys_b+r.prompt.encode('utf-8')+b'\x00\x00\x00\x00'+r.response.encode('utf-8')
            bundles.append({'rec_id':r.rec_id,'byte_offset':byte_offset,'byte_length':len(blob)})
            chunks.append(blob)
            byte_offset+=len(blob)
        header={'format':'expatlas-bundle/v1','subject':self.subject,'contributor_id':contributor_id,'note':note,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'n_records':n,'records':bundles}
        header_json=json.dumps(header).encode('utf-8')
        out=Path(out_path)
        with open(out,'wb') as f:
            f.write(_MAGIC)
            f.write(struct.pack('<BQ',_VERSION,len(header_json)))
            f.write(header_json)
            for c in chunks:f.write(c)
        return out
    def import_bundle(self,bundle_path):
        b=Path(bundle_path).read_bytes()
        if not b.startswith(_MAGIC):raise ExperienceError(f'not an expatlas bundle: {bundle_path}')
        version=b[8]
        if version!=_VERSION:raise ExperienceError(f'unsupported bundle version {version}')
        hl=struct.unpack('<Q',b[9:17])[0]
        header=json.loads(b[17:17+hl].decode('utf-8'))
        payload=b[17+hl:]
        imported=0
        for r in header['records']:
            blob=payload[int(r['byte_offset']):int(r['byte_offset'])+int(r['byte_length'])]
            rec=self._parse_record(blob)
            self.append(rec.prompt,rec.response,outcome=rec.outcome,subject_id=rec.subject_id,timestamp=rec.timestamp,system=getattr(rec,'system',''))
            imported+=1
        return imported
