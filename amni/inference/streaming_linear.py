import os,sys,json,ctypes as _ct,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from pathlib import Path
from collections import OrderedDict
_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(_ROOT))
from amni.compute.reffelt4 import decode_rgba4_to_fp16,decode_rgba16_quad_to_fp16
_HIP_OPT_IN=os.environ.get('AMNI_HIP_GEMV_ON','0')=='1'
_HIP=None
_HIP_ENG=None
def _hip_engine():
    global _HIP
    if _HIP is not None or not _HIP_OPT_IN:return _HIP
    try:
        from amni.compute import ari_engine as _ae
        if _ae._load():_HIP=_ae;return _HIP
    except Exception as e:
        print(f'[streaming_linear] HIP engine unavailable: {e}',file=sys.stderr)
    return None
def _hip_eng_singleton(eng_mod):
    global _HIP_ENG
    if _HIP_ENG is None:_HIP_ENG=eng_mod.ARIEngine()
    return _HIP_ENG
def _torch_dtype_from_str(s):return {'bfloat16':torch.bfloat16,'float16':torch.float16,'float32':torch.float32,'uint16':torch.uint16}[s]
_GF17_PLANE_KEYS=('d0','d1','d2','d3')
class TensorRegistry:
    def __init__(self,bake_dir,budget_bytes=60*1024*1024,device='cuda',enable_prefetch=False):
        self.bake_dir=Path(bake_dir)
        with open(self.bake_dir/'manifest.json') as f:self.manifest=json.load(f)
        self.budget=budget_bytes;self.device=device;self._lru=OrderedDict();self._sizes={};self._mmaps={}
        self._gf17_mmaps={};self._residual_mmaps={}
        self._fetch_count=0;self._evict_count=0;self._bytes_loaded=0;self._prefetch_hits=0
        self._pinned_keys=set();self._inflight={};self._last_use={}
        self._is_gf17=self.manifest.get('reffelt_scheme')=='gf17_digit_planes'
        self._is_rgba16q=self.manifest.get('reffelt_scheme')=='rgba16_quad'
        self._is_palette=self.manifest.get('reffelt_scheme')=='palette'
        self._is_tilepack=self.manifest.get('scheme')=='tile_bitpack'
        self._prefetch_stream=torch.cuda.Stream() if enable_prefetch and not self._is_tilepack and device.startswith('cuda') and torch.cuda.is_available() else None
        self._residual_overlay_count=0
        self.active_subjects=('global',)
    def _mmap_for(self,key):
        if key not in self._mmaps:
            e=self.manifest['tensors'][key]
            self._mmaps[key]=np.memmap(self.bake_dir/e['ptex_path'],dtype=np.uint8,mode='r',shape=(e['page_h'],e['page_w'],4))
        return self._mmaps[key]
    def _mmap_u16(self,key):
        if key not in self._mmaps:
            e=self.manifest['tensors'][key]
            self._mmaps[key]=np.memmap(self.bake_dir/e['ptex_path'],dtype=np.uint16,mode='r',shape=(e['page_h'],e['page_w'],4))
        return self._mmaps[key]
    def _decode_rgba16q_to_torch(self,key,target_shape):
        e=self.manifest['tensors'][key];mm=self._mmap_u16(key);nw=int(np.prod(target_shape))
        flat_fp16=decode_rgba16_quad_to_fp16(mm,nw);src_dtype=e['source_dtype'];t=torch.from_numpy(flat_fp16.copy())
        return t.reshape(target_shape).to(self.device) if src_dtype=='float16' else t.view(_torch_dtype_from_str(src_dtype)).reshape(target_shape).to(self.device)
    def _decode_tilepack_to_torch(self,key,target_shape):
        from amni.inference.tilepack import load_tilepack
        return load_tilepack(self.bake_dir,self.manifest['tensors'][key]).reshape(target_shape).to(self.device)
    def _mmap_gf17(self,key):
        if key not in self._gf17_mmaps:
            e=self.manifest['tensors'][key]
            n=int(e['n_pixels'])
            self._gf17_mmaps[key]=np.memmap(self.bake_dir/e['gf17_path'],dtype=np.uint8,mode='r',shape=(4*n,))
        return self._gf17_mmaps[key]
    def set_active_subjects(self,subjects):
        self.active_subjects=tuple(subjects) if subjects else ('global',)
        self._residual_mmaps.clear()
        self._lru.clear();self._sizes.clear()
        import gc;gc.collect()
    def _residual_paths_for_key(self,key):
        e=self.manifest['tensors'][key]
        out=[]
        paths_dict=e.get('residual_paths') or ({'global':e['residual_path']} if e.get('residual_path') else {})
        for s in self.active_subjects:
            rel=paths_dict.get(s)
            if rel:
                rp=self.bake_dir/rel
                if rp.exists():out.append((s,rp))
                continue
            if s=='global':
                gf17_p=Path(e.get('gf17_path',''))
                if gf17_p:
                    rp=self.bake_dir/gf17_p.parent/(gf17_p.stem+'.gf17res')
                    if rp.exists():out.append((s,rp))
            else:
                gf17_p=Path(e.get('gf17_path',''))
                if gf17_p:
                    rp=self.bake_dir/gf17_p.parent/(gf17_p.stem+f'.{s}.gf17res')
                    if rp.exists():out.append((s,rp))
        return out
    def _mmap_residuals_active(self,key):
        e=self.manifest['tensors'][key]
        n=int(e['n_pixels'])
        cache_key=(key,self.active_subjects)
        if cache_key in self._residual_mmaps:return self._residual_mmaps[cache_key]
        mmaps=[]
        for subject,rp in self._residual_paths_for_key(key):
            mm=np.memmap(rp,dtype=np.uint8,mode='r',shape=(4*n,))
            mmaps.append((subject,mm))
        self._residual_mmaps[cache_key]=mmaps
        return mmaps
    def _mmap_residual(self,key):
        active=self._mmap_residuals_active(key)
        if not active:return None
        return active[0][1] if len(active)==1 else None
    def _decode_gf17_to_fp16(self,key):
        e=self.manifest['tensors'][key]
        gf17=self._mmap_gf17(key);n=int(e['n_pixels']);po=e['plane_offsets']
        base_d=[np.array(gf17[int(po[k]):int(po[k])+n],dtype=np.uint16) for k in _GF17_PLANE_KEYS]
        active=self._mmap_residuals_active(key)
        if not active:
            u16=(base_d[0].astype(np.uint32)+base_d[1].astype(np.uint32)*17+base_d[2].astype(np.uint32)*289+base_d[3].astype(np.uint32)*4913).astype(np.uint16)
            return u16.view(np.float16)
        self._residual_overlay_count+=len(active)
        if len(active)==1:
            subject,res=active[0]
            d=[(base_d[i]+np.array(res[int(po[k]):int(po[k])+n],dtype=np.uint16))%17 for i,k in enumerate(_GF17_PLANE_KEYS)]
            u16=(d[0].astype(np.uint32)+d[1].astype(np.uint32)*17+d[2].astype(np.uint32)*289+d[3].astype(np.uint32)*4913).astype(np.uint16)
            return u16.view(np.float16)
        sd=e.get('source_dtype','float16')
        base_u16=np.minimum(base_d[0].astype(np.uint32)+base_d[1].astype(np.uint32)*17+base_d[2].astype(np.uint32)*289+base_d[3].astype(np.uint32)*4913,65535).astype(np.uint16)
        if sd=='bfloat16':base_fp=torch.from_numpy(base_u16.view(np.float16)).view(torch.bfloat16).to(torch.float32).numpy()
        else:base_fp=base_u16.view(np.float16).astype(np.float32)
        sum_delta=np.zeros_like(base_fp)
        for subject,res in active:
            eff_d=[(base_d[i]+np.array(res[int(po[k]):int(po[k])+n],dtype=np.uint16))%17 for i,k in enumerate(_GF17_PLANE_KEYS)]
            eff_u16=np.minimum(eff_d[0].astype(np.uint32)+eff_d[1].astype(np.uint32)*17+eff_d[2].astype(np.uint32)*289+eff_d[3].astype(np.uint32)*4913,65535).astype(np.uint16)
            recon_fp=torch.from_numpy(eff_u16.view(np.float16)).view(torch.bfloat16).to(torch.float32).numpy() if sd=='bfloat16' else eff_u16.view(np.float16).astype(np.float32)
            delta=np.where(np.isfinite(recon_fp-base_fp),recon_fp-base_fp,0.0).astype(np.float32)
            sum_delta+=delta
        target_fp=base_fp+sum_delta/len(active)
        if sd=='bfloat16':target_u16=torch.from_numpy(target_fp).to(torch.bfloat16).view(torch.float16).numpy().view(np.uint16)
        else:target_u16=target_fp.astype(np.float16).view(np.uint16)
        return target_u16.view(np.float16)
    def _decode_gf17_rows_to_fp16(self,key,row_indices):
        e=self.manifest['tensors'][key]
        gf17=self._mmap_gf17(key);n=int(e['n_pixels']);po=e['plane_offsets']
        cols=int(e['shape'][-1])
        rows=np.asarray(row_indices,dtype=np.int64)
        base_d=[]
        for k in _GF17_PLANE_KEYS:
            plane=gf17[int(po[k]):int(po[k])+n].reshape(-1,cols)
            base_d.append(np.array(plane[rows],dtype=np.uint16))
        active=self._mmap_residuals_active(key)
        if not active:
            u16=(base_d[0].astype(np.uint32)+base_d[1].astype(np.uint32)*17+base_d[2].astype(np.uint32)*289+base_d[3].astype(np.uint32)*4913).astype(np.uint16)
            return u16.view(np.float16)
        self._residual_overlay_count+=len(active)
        if len(active)==1:
            subject,res=active[0]
            d=[]
            for i,k in enumerate(_GF17_PLANE_KEYS):
                rplane=res[int(po[k]):int(po[k])+n].reshape(-1,cols)
                rrows=np.array(rplane[rows],dtype=np.uint16)
                d.append((base_d[i]+rrows)%17)
            u16=(d[0].astype(np.uint32)+d[1].astype(np.uint32)*17+d[2].astype(np.uint32)*289+d[3].astype(np.uint32)*4913).astype(np.uint16)
            return u16.view(np.float16)
        sd=e.get('source_dtype','float16')
        base_u16=np.minimum(base_d[0].astype(np.uint32)+base_d[1].astype(np.uint32)*17+base_d[2].astype(np.uint32)*289+base_d[3].astype(np.uint32)*4913,65535).astype(np.uint16)
        if sd=='bfloat16':base_fp=torch.from_numpy(base_u16.view(np.float16)).view(torch.bfloat16).to(torch.float32).numpy()
        else:base_fp=base_u16.view(np.float16).astype(np.float32)
        sum_delta=np.zeros_like(base_fp)
        for subject,res in active:
            eff_d=[]
            for i,k in enumerate(_GF17_PLANE_KEYS):
                rplane=res[int(po[k]):int(po[k])+n].reshape(-1,cols)
                rrows=np.array(rplane[rows],dtype=np.uint16)
                eff_d.append((base_d[i]+rrows)%17)
            eff_u16=np.minimum(eff_d[0].astype(np.uint32)+eff_d[1].astype(np.uint32)*17+eff_d[2].astype(np.uint32)*289+eff_d[3].astype(np.uint32)*4913,65535).astype(np.uint16)
            recon_fp=torch.from_numpy(eff_u16.view(np.float16)).view(torch.bfloat16).to(torch.float32).numpy() if sd=='bfloat16' else eff_u16.view(np.float16).astype(np.float32)
            delta=np.where(np.isfinite(recon_fp-base_fp),recon_fp-base_fp,0.0).astype(np.float32)
            sum_delta+=delta
        target_fp=base_fp+sum_delta/len(active)
        if sd=='bfloat16':target_u16=torch.from_numpy(target_fp).to(torch.bfloat16).view(torch.float16).numpy().view(np.uint16)
        else:target_u16=target_fp.astype(np.float16).view(np.uint16)
        return target_u16.view(np.float16)
    def _decode_to_torch(self,rgba_bytes,e,target_shape):
        flat_fp16=decode_rgba4_to_fp16(np.ascontiguousarray(rgba_bytes).reshape(-1,4))
        src_dtype=e['source_dtype']
        t=torch.from_numpy(flat_fp16.copy())
        return (t.reshape(target_shape).to(self.device) if src_dtype=='float16' else t.view(_torch_dtype_from_str(src_dtype)).reshape(target_shape).to(self.device))
    def _decode_gf17_to_torch(self,key,target_shape):
        e=self.manifest['tensors'][key]
        flat_fp16=self._decode_gf17_to_fp16(key)
        src_dtype=e['source_dtype']
        t=torch.from_numpy(flat_fp16.copy())
        return (t.reshape(target_shape).to(self.device) if src_dtype=='float16' else t.view(_torch_dtype_from_str(src_dtype)).reshape(target_shape).to(self.device))
    def _decode_palette_to_fp16(self,key):
        e=self.manifest['tensors'][key];K=int(e['K']);b=int(e['idx_bits']);n=int(e['n'])
        raw=np.memmap(self.bake_dir/e['palette_path'],dtype=np.uint8,mode='r')
        pal=np.ascontiguousarray(raw[:K*2]).view('<u2')
        packed=raw[K*2:]
        out=np.empty(n,dtype=np.uint16);CH=8_000_000;woff=0;boff=0;w=np.arange(b,dtype=np.uint32)
        while woff<n:
            m=min(CH,n-woff);nb=(m*b+7)//8
            cb=np.ascontiguousarray(packed[boff:boff+nb])
            bits=np.unpackbits(cb)[:m*b].reshape(m,b).astype(np.uint32)
            idx=(bits*(1<<w)).sum(1)
            out[woff:woff+m]=pal[idx]
            woff+=m;boff+=nb;del bits,idx,cb
        del raw;return out.view(np.float16)
    def _decode_palette_to_torch(self,key,target_shape):
        e=self.manifest['tensors'][key];K=int(e['K']);b=int(e['idx_bits']);n=int(e['n']);sd=e['source_dtype'];dev=self.device
        if not (isinstance(dev,str) and dev.startswith('cuda') and torch.cuda.is_available()):
            flat=self._decode_palette_to_fp16(key);t=torch.from_numpy(flat.copy())
            return (t.reshape(target_shape).to(dev) if sd=='float16' else t.view(_torch_dtype_from_str(sd)).reshape(target_shape).to(dev))
        raw=np.memmap(self.bake_dir/e['palette_path'],dtype=np.uint8,mode='r')
        pal=torch.from_numpy(np.ascontiguousarray(raw[:K*2]).view('<u2').astype(np.int32)).to(dev)
        packed=raw[K*2:];sh=torch.arange(7,-1,-1,device=dev,dtype=torch.int32);w=(torch.ones((),dtype=torch.int32,device=dev)<<torch.arange(b,device=dev,dtype=torch.int32))
        out=torch.empty(n,dtype=torch.int16,device=dev);CH=8_000_000;woff=0;boff=0
        while woff<n:
            m=min(CH,n-woff);nb=(m*b+7)//8
            pk=torch.from_numpy(np.ascontiguousarray(packed[boff:boff+nb]).copy()).to(dev).to(torch.int32)
            bits=((pk.unsqueeze(1)>>sh)&1).reshape(-1)[:m*b].reshape(m,b)
            idx=(bits*w).sum(1).to(torch.long);out[woff:woff+m]=pal[idx].to(torch.int16)
            woff+=m;boff+=nb;del pk,bits,idx
        del raw;return out.view(torch.float16 if sd=='float16' else _torch_dtype_from_str(sd)).reshape(target_shape)
    def invalidate(self,key):
        if key in self._lru:del self._lru[key]
        if key in self._sizes:del self._sizes[key]
        if key in self._inflight:del self._inflight[key]
        for k in list(self._residual_mmaps.keys()):
            if (isinstance(k,tuple) and k[0]==key) or k==key:self._residual_mmaps.pop(k,None)
        import gc;gc.collect()
    def pin(self,key):self._pinned_keys.add(key)
    def manifest_total_bytes(self):
        db={'float16':2,'bfloat16':2,'float32':4,'uint16':2,'int8':1}
        return sum(int(np.prod(e['shape']))*db.get(e.get('source_dtype','float16'),2) for e in self.manifest['tensors'].values())
    def autosize_budget(self,cap_bytes=None,headroom=1.05):
        b=int(self.manifest_total_bytes()*headroom);self.budget=min(b,int(cap_bytes)) if cap_bytes else b;return self.budget
    def pin_hot(self,patterns=('embed_tokens','lm_head','model.norm','layers.0.')):
        hot=[k for k in self.manifest['tensors'] if any(p in k for p in patterns)]
        for k in hot:self.pin(k)
        return hot
    def warmup(self):
        for k in self.manifest['tensors']:
            try:self.get_full(k)
            except Exception:pass
        return self.stats()
    def thrash_per_token(self,tokens=1):return self._evict_count/max(1,tokens)
    def drop_resident(self,key):self._lru.pop(key,None);self._sizes.pop(key,None)
    def note_use(self,key):
        if not self.device.startswith('cuda') or not torch.cuda.is_available():return
        evt=torch.cuda.Event();evt.record(torch.cuda.current_stream())
        self._last_use[key]=evt
    def schedule_prefetch(self,key):
        if self._prefetch_stream is None:return
        if key in self._lru:self._lru.move_to_end(key);return
        if key in self._inflight:return
        if key not in self.manifest['tensors']:return
        e=self.manifest['tensors'][key]
        if self._is_gf17:flat_fp16=self._decode_gf17_to_fp16(key)
        elif self._is_palette:flat_fp16=self._decode_palette_to_fp16(key)
        elif self._is_rgba16q:flat_fp16=decode_rgba16_quad_to_fp16(self._mmap_u16(key),int(np.prod(e['shape'])))
        else:
            mm=self._mmap_for(key)
            n=int(e['n_pixels'])
            flat_fp16=decode_rgba4_to_fp16(np.ascontiguousarray(mm.reshape(-1,4)[:n]))
        host=torch.from_numpy(flat_fp16.copy()).pin_memory()
        with torch.cuda.stream(self._prefetch_stream):
            t=host.to(self.device,non_blocking=True)
            src_dtype=e['source_dtype']
            if src_dtype!='float16':t=t.view(_torch_dtype_from_str(src_dtype))
            t=t.reshape(e['shape'])
            evt=torch.cuda.Event();evt.record(self._prefetch_stream)
        self._inflight[key]=(t,evt,host)
    def get_full(self,key):
        if key in self._lru:self._lru.move_to_end(key);return self._lru[key]
        if key in self._inflight:
            t,evt,_host=self._inflight.pop(key)
            evt.synchronize()
            self._lru[key]=t
            size=t.numel()*t.element_size();self._sizes[key]=size
            self._fetch_count+=1;self._bytes_loaded+=size;self._prefetch_hits+=1
            self._evict_if_over_budget(protect=key)
            return t
        e=self.manifest['tensors'][key]
        if self._is_gf17:t=self._decode_gf17_to_torch(key,e['shape'])
        elif self._is_palette:t=self._decode_palette_to_torch(key,e['shape'])
        elif self._is_rgba16q:t=self._decode_rgba16q_to_torch(key,e['shape'])
        elif self._is_tilepack:t=self._decode_tilepack_to_torch(key,e['shape'])
        else:
            mm=self._mmap_for(key)
            n=int(e['n_pixels'])
            flat=mm.reshape(-1,4)[:n]
            t=self._decode_to_torch(flat,e,e['shape'])
        self._lru[key]=t
        size=t.numel()*t.element_size();self._sizes[key]=size
        self._fetch_count+=1;self._bytes_loaded+=size
        self._evict_if_over_budget(protect=key)
        return t
    def get_rows(self,key,row_indices):
        e=self.manifest['tensors'][key]
        if self._is_rgba16q:
            cols=int(e['shape'][-1]);ri=np.asarray(row_indices,dtype=np.int64)
            flat=np.ascontiguousarray(self._mmap_u16(key)).reshape(-1)[:int(np.prod(e['shape']))].view(np.float16).reshape(-1,cols)
            t=torch.from_numpy(np.ascontiguousarray(flat[ri]).copy());src_dtype=e['source_dtype']
            if src_dtype!='float16':t=t.view(_torch_dtype_from_str(src_dtype))
            return t.reshape(ri.shape[0],cols).to(self.device)
        if self._is_gf17:
            flat_fp16=self._decode_gf17_rows_to_fp16(key,row_indices)
            n_rows=len(row_indices) if hasattr(row_indices,'__len__') else int(row_indices.shape[0])
            cols=int(e['shape'][-1])
            src_dtype=e['source_dtype']
            t=torch.from_numpy(flat_fp16.copy())
            if src_dtype!='float16':t=t.view(_torch_dtype_from_str(src_dtype))
            return t.reshape(n_rows,cols).to(self.device)
        if self._is_palette or self._is_tilepack:
            full=self.get_full(key)
            ri=torch.as_tensor(np.asarray(row_indices,dtype=np.int64),device=full.device,dtype=torch.long)
            return (full if full.dim()==2 else full.reshape(-1,int(e['shape'][-1])))[ri]
        mm=self._mmap_for(key);ri=np.asarray(row_indices,dtype=np.int64)
        cols=int(e['shape'][-1])*int(e.get('u16_per_elem',1))
        gather=(ri[:,None]*cols+np.arange(cols,dtype=np.int64)[None,:]).reshape(-1)
        rows_rgba=np.ascontiguousarray(mm.reshape(-1,4)[gather])
        flat_fp16=decode_rgba4_to_fp16(rows_rgba.reshape(-1,4))
        src_dtype=e['source_dtype']
        t=torch.from_numpy(flat_fp16.copy())
        if src_dtype!='float16':t=t.view(_torch_dtype_from_str(src_dtype))
        return t.reshape(ri.shape[0],int(e['shape'][-1])).to(self.device)
    def _evict_if_over_budget(self,protect=None):
        if sum(self._sizes.values())<=self.budget:return
        evicted_any=False
        while sum(self._sizes.values())>self.budget and len(self._lru)>1:
            for k in list(self._lru.keys()):
                if k==protect:continue
                if k in self._pinned_keys:continue
                evt=self._last_use.pop(k,None)
                if evt is not None:evt.synchronize()
                del self._lru[k];del self._sizes[k];self._evict_count+=1
                evicted_any=True
                break
            else:break
        if evicted_any and self.device.startswith('cuda'):torch.cuda.empty_cache()
    def evict(self,key):
        if key in self._lru:del self._lru[key];del self._sizes[key]
    def evict_all(self):
        if self.device.startswith('cuda') and torch.cuda.is_available():torch.cuda.synchronize()
        self._lru.clear();self._sizes.clear();self._last_use.clear()
        if self.device.startswith('cuda'):torch.cuda.empty_cache()
    def total_resident(self):return sum(self._sizes.values())
    def stats(self):return {'resident_bytes':self.total_resident(),'fetches':self._fetch_count,'evictions':self._evict_count,'bytes_loaded':self._bytes_loaded,'prefetch_hits':self._prefetch_hits,'cached_keys':list(self._lru.keys()),'pinned_keys':list(self._pinned_keys),'inflight':list(self._inflight.keys())}
class StreamingLinear(nn.Module):
    def __init__(self,registry,weight_key,bias_key=None,prefetch_keys=None):
        super().__init__()
        self.registry=registry;self.weight_key=weight_key;self.bias_key=bias_key
        self.prefetch_keys=prefetch_keys or []
        self._bias_cached=registry.get_full(bias_key) if bias_key and bias_key in registry.manifest['tensors'] else None
        self._hip_tex=None;self._hip_NK=None;self._hip_skip=False;self._dropped=False
    def _try_bind_hip(self,w):
        if self._hip_skip or self._hip_tex is not None:return
        eng_mod=_hip_engine()
        if eng_mod is None:self._hip_skip=True;return
        if w.dim()!=2 or w.dtype not in (torch.float16,torch.bfloat16):self._hip_skip=True;return
        try:
            N,K=int(w.shape[0]),int(w.shape[1])
            w_fp16=w.detach().contiguous().to(torch.float16) if w.dtype==torch.bfloat16 else w.detach().contiguous()
            w_cpu=w_fp16.cpu().numpy().view(np.uint16).reshape(-1)
            n_pix=(w_cpu.size+3)//4
            page_w=4096
            page_h=(n_pix+page_w-1)//page_w
            page=np.zeros((page_h,page_w,4),dtype=np.uint16)
            page.reshape(-1)[:w_cpu.size]=w_cpu
            self._hip_eng=_hip_eng_singleton(eng_mod)
            self._hip_page=page
            self._hip_tex=self._hip_eng.bind_texture16(page,N*K)
            self._hip_NK=(N,K)
            self._hip_lib=eng_mod._lib
            self._hip_orig_dtype=w.dtype
        except Exception as e:
            print(f'[streaming_linear] HIP bind failed for {self.weight_key}: {e}',file=sys.stderr)
            self._hip_skip=True
            self._hip_page=None
    def forward(self,x):
        if self._hip_tex is not None and x.dtype in (torch.float16,torch.bfloat16) and x.is_cuda:
            N,K=self._hip_NK
            xs=x.shape
            tokens=int(np.prod(xs[:-1]))
            if tokens==1 and xs[-1]==K:
                in_dtype=x.dtype
                xv=(x.to(torch.float16) if in_dtype==torch.bfloat16 else x).contiguous().view(-1)
                y=torch.empty(N,dtype=torch.float16,device=x.device)
                self._hip_lib.ari_gemv_rgba16_fp16(_ct.c_void_p(int(xv.view(torch.uint16).data_ptr())),self._hip_tex.idx,_ct.c_void_p(int(y.view(torch.uint16).data_ptr())),N,K)
                if self._bias_cached is not None:y=y+(self._bias_cached.to(torch.float16) if self._bias_cached.dtype!=torch.float16 else self._bias_cached)
                if in_dtype==torch.bfloat16:y=y.to(torch.bfloat16)
                if not self._dropped:self.registry.drop_resident(self.weight_key);self._dropped=True
                self.registry.note_use(self.weight_key)
                return y.view(*xs[:-1],N)
        for k in self.prefetch_keys:self.registry.schedule_prefetch(k)
        w=self.registry.get_full(self.weight_key)
        if not self._hip_skip and self._hip_tex is None:self._try_bind_hip(w)
        if self._hip_tex is not None and x.dtype in (torch.float16,torch.bfloat16) and x.is_cuda:
            N,K=self._hip_NK
            xs=x.shape
            tokens=int(np.prod(xs[:-1]))
            if tokens==1 and xs[-1]==K:
                in_dtype=x.dtype
                xv=(x.to(torch.float16) if in_dtype==torch.bfloat16 else x).contiguous().view(-1)
                y=torch.empty(N,dtype=torch.float16,device=x.device)
                self._hip_lib.ari_gemv_rgba16_fp16(_ct.c_void_p(int(xv.view(torch.uint16).data_ptr())),self._hip_tex.idx,_ct.c_void_p(int(y.view(torch.uint16).data_ptr())),N,K)
                self.registry.note_use(self.weight_key)
                if self._bias_cached is not None:y=y+(self._bias_cached.to(torch.float16) if self._bias_cached.dtype!=torch.float16 else self._bias_cached)
                if in_dtype==torch.bfloat16:y=y.to(torch.bfloat16)
                return y.view(*xs[:-1],N)
        out=F.linear(x,w,self._bias_cached)
        self.registry.note_use(self.weight_key)
        return out
def install_prefetch_chain(model,horizon=6):
    keys=[]
    for name,m in model.named_modules():
        if isinstance(m,StreamingLinear):keys.append((m,m.weight_key))
    n=len(keys)
    for i,(m,_) in enumerate(keys):
        upcoming=[]
        for j in range(i+1,min(i+1+horizon,n)):
            wk=keys[j][1]
            if wk!=m.weight_key:upcoming.append(wk)
        m.prefetch_keys=upcoming
    return n
class StreamingEmbedding(nn.Module):
    def __init__(self,registry,weight_key,embed_scale=None):
        super().__init__()
        self.registry=registry;self.weight_key=weight_key
        self.embed_scale=embed_scale
    def forward(self,input_ids):
        ids=input_ids.flatten().detach().cpu().numpy().astype(np.int64)
        unique_ids,inverse=np.unique(ids,return_inverse=True)
        rows=self.registry.get_rows(self.weight_key,unique_ids)
        out=rows[torch.from_numpy(inverse).to(rows.device)].view(*input_ids.shape,-1)
        if self.embed_scale is not None:out=out*self.embed_scale.to(out.dtype) if hasattr(self.embed_scale,'to') else out*self.embed_scale
        return out
class StreamingTiedLMHead(nn.Module):
    def __init__(self,registry,embed_key,tile_rows=4096):
        super().__init__()
        self.registry=registry;self.embed_key=embed_key;self.tile_rows=tile_rows
    def forward(self,hidden):
        e=self.registry.manifest['tensors'][self.embed_key]
        V,H=int(e['shape'][0]),int(e['shape'][1])
        if self.tile_rows>=V:
            w=self.registry.get_full(self.embed_key)
            return F.linear(hidden,w,None)
        out=torch.empty(*hidden.shape[:-1],V,dtype=hidden.dtype,device=hidden.device)
        r0=0
        while r0<V:
            r1=min(r0+self.tile_rows,V)
            idx=np.arange(r0,r1,dtype=np.int64)
            tile=self.registry.get_rows(self.embed_key,idx)
            out[...,r0:r1]=F.linear(hidden,tile,None)
            r0=r1
        return out
def swap_modules(model,registry,verbose=False,lmhead_tile_rows=4096):
    manifest=registry.manifest['tensors']
    swapped_linear=0;swapped_embed=0;swapped_lmhead=0
    embed_key='model.embed_tokens.weight'
    has_lm_head_weight='lm_head.weight' in manifest
    def _swap(parent_name,parent):
        nonlocal swapped_linear,swapped_embed,swapped_lmhead
        for cn,child in list(parent.named_children()):
            full=f'{parent_name}.{cn}' if parent_name else cn
            if isinstance(child,nn.Linear):
                wk=f'{full}.weight';bk=f'{full}.bias' if child.bias is not None else None
                if full=='lm_head' and not has_lm_head_weight and embed_key in manifest:
                    setattr(parent,cn,StreamingTiedLMHead(registry,embed_key,tile_rows=lmhead_tile_rows));swapped_lmhead+=1
                    if verbose:print(f'  TIED_LMHEAD {full} -> {embed_key}')
                elif wk in manifest:
                    setattr(parent,cn,StreamingLinear(registry,wk,bk));swapped_linear+=1
                    if verbose:print(f'  LINEAR {full}')
            elif isinstance(child,nn.Embedding):
                wk=f'{full}.weight'
                if wk in manifest:
                    es=getattr(child,'scalar_embed_scale',None) if hasattr(child,'scalar_embed_scale') else None
                    if es is None:
                        eb=getattr(child,'embed_scale',None)
                        if eb is not None and hasattr(eb,'item'):
                            try:es=float(eb.item()) if not eb.is_meta else None
                            except Exception:es=None
                    setattr(parent,cn,StreamingEmbedding(registry,wk,embed_scale=es));swapped_embed+=1
                    if verbose:print(f'  EMBED {full} scale={es}')
            else:
                _swap(full,child)
    _swap('',model)
    return {'linear':swapped_linear,'embed':swapped_embed,'lmhead':swapped_lmhead}
def materialize_remaining_params(model,registry,device='cuda',verbose=False):
    from accelerate.utils import set_module_tensor_to_device
    manifest=registry.manifest['tensors']
    materialized=0;skipped=0;mismatches=[]
    _pshape={n:tuple(p.shape) for n,p in model.named_parameters()}
    for name,param in list(model.named_parameters()):
        if name not in manifest:
            skipped+=1
            if verbose:print(f'  SKIP_NO_MANIFEST {name}')
            continue
        e=manifest[name]
        if registry._is_tilepack:
            from amni.inference.tilepack import load_tilepack;t=load_tilepack(registry.bake_dir,e)
        else:
            if registry._is_gf17:flat_fp16=registry._decode_gf17_to_fp16(name)
            elif registry._is_palette:flat_fp16=registry._decode_palette_to_fp16(name)
            else:
                mm=registry._mmap_for(name)
                n=int(e['n_pixels'])
                flat_fp16=decode_rgba4_to_fp16(np.ascontiguousarray(mm.reshape(-1,4)[:n]))
            t=torch.from_numpy(flat_fp16.copy())
            src_dtype=e['source_dtype']
            if src_dtype!='float16':t=t.view(_torch_dtype_from_str(src_dtype))
        try:t=t.reshape(e['shape'])
        except Exception as _re:print(f'  [materialize RESHAPE-FAIL] {name}: cannot reshape {t.numel()} elems to {e["shape"]}: {_re}',flush=True);mismatches.append((name,'reshape',e['shape']));continue
        want=_pshape.get(name)
        if want is not None and tuple(t.shape)!=want:
            if len(mismatches)<=25:print(f'  [materialize MISMATCH] {name}: baked {tuple(t.shape)} vs model {want}',flush=True)
            mismatches.append((name,tuple(t.shape),want));continue
        set_module_tensor_to_device(model,name,device,value=t)
        materialized+=1
        if verbose:print(f'  MATERIALIZE {name} shape={e["shape"]}')
    if mismatches:print(f'  [materialize] DONE materialized={materialized} skipped_nomanifest={skipped} MISMATCHES={len(mismatches)}',flush=True)
    return {'materialized':materialized,'skipped':skipped,'mismatches':len(mismatches)}
