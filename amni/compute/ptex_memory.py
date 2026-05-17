import hashlib,json,numpy as np
from pathlib import Path
DEFAULT_SLOT_SIZES={'memory':3072,'context':3072,'learnings':256,'errors':256}
CHANNELS=tuple(DEFAULT_SLOT_SIZES.keys())
def content_key(x)->int:
    if isinstance(x,str):b=x.encode('utf-8')
    elif isinstance(x,(bytes,bytearray)):b=bytes(x)
    elif isinstance(x,np.ndarray):b=np.ascontiguousarray(x).tobytes()
    elif isinstance(x,(list,tuple)):b=str(x).encode('utf-8')
    else:b=str(x).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(b,digest_size=8).digest(),'little')
class PtexMemory:
    def __init__(self,root,slot_sizes=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.slot_sizes=dict(slot_sizes or DEFAULT_SLOT_SIZES)
        for ch in self.slot_sizes:assert ch in CHANNELS,f'unknown channel {ch}'
        self.tables={ch:{} for ch in self.slot_sizes}
        self._load()
    def _ch_dir(self,ch):d=self.root/ch;d.mkdir(parents=True,exist_ok=True);return d
    def _idx_path(self,ch):return self._ch_dir(ch)/'index.json'
    def _page_path(self,ch):return self._ch_dir(ch)/'page.atex16'
    def _load(self):
        for ch in self.slot_sizes:
            ip=self._idx_path(ch)
            if ip.exists():
                d=json.loads(ip.read_text());self.tables[ch]={int(k):int(v) for k,v in d.items()}
            else:self.tables[ch]={}
    def _save(self,ch):
        self._idx_path(ch).write_text(json.dumps({str(k):v for k,v in self.tables[ch].items()}))
    def write(self,ch,key,vec):
        assert ch in self.slot_sizes,f'unknown channel {ch}'
        sd=self.slot_sizes[ch];v=np.ascontiguousarray(np.asarray(vec).astype(np.float16)).reshape(-1)
        assert v.size==sd,f'channel {ch} expects slot_size={sd}, got {v.size}'
        slot=self.tables[ch].get(int(key))
        if slot is None:slot=len(self.tables[ch]);self.tables[ch][int(key)]=slot
        pp=self._page_path(ch);off=slot*sd*2
        cur=bytearray(pp.read_bytes()) if pp.exists() else bytearray()
        if len(cur)<off+sd*2:cur.extend(b'\x00'*(off+sd*2-len(cur)))
        cur[off:off+sd*2]=v.tobytes();pp.write_bytes(bytes(cur));self._save(ch);return slot
    def read(self,ch,key):
        if int(key) not in self.tables[ch]:return None
        sd=self.slot_sizes[ch];slot=self.tables[ch][int(key)];off=slot*sd*2
        pp=self._page_path(ch)
        if not pp.exists():return None
        with open(pp,'rb') as f:f.seek(off);data=f.read(sd*2)
        return np.frombuffer(data,dtype=np.float16).copy() if len(data)==sd*2 else None
    def has(self,ch,key):return int(key) in self.tables[ch]
    def keys(self,ch):return list(self.tables[ch].keys())
    def stats(self):
        out={}
        for ch in self.slot_sizes:
            n=len(self.tables[ch]);sz=self.slot_sizes[ch]
            pp=self._page_path(ch);bytes_on_disk=pp.stat().st_size if pp.exists() else 0
            out[ch]={'entries':n,'slot_size_fp16':sz,'page_bytes':bytes_on_disk,'expected_bytes':n*sz*2}
        return out
    def clear(self,ch=None):
        chs=(ch,) if ch else tuple(self.slot_sizes.keys())
        for c in chs:
            self.tables[c]={};self._save(c)
            pp=self._page_path(c)
            if pp.exists():pp.unlink()
def rgba16_view(vec_fp16):
    v=np.ascontiguousarray(vec_fp16,dtype=np.float16).reshape(-1)
    pad=(-v.size)%4
    if pad:v=np.concatenate([v,np.zeros(pad,dtype=np.float16)])
    return v.view(np.uint16).reshape(-1,4)
def from_rgba16(rgba16,n_weights):
    return np.ascontiguousarray(rgba16,dtype=np.uint16).reshape(-1)[:n_weights].view(np.float16).copy()
