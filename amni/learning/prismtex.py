import json,struct,time,hashlib,uuid,numpy as np
from pathlib import Path
from typing import List,Optional,Dict,Iterable
from amni.learning.gf17_writer import LearningWriter,WeightAccessError,AsimovProtectedError,_PLANE_KEYS
_MAGIC=b'PRISMTEX'
_VERSION=1
def _u16arr_to_f32(u16,src_dtype):
    if src_dtype=='bfloat16':
        import torch
        return torch.from_numpy(u16.view(np.float16)).view(torch.bfloat16).to(torch.float32).numpy()
    return u16.view(np.float16).astype(np.float32)
def _f32arr_to_u16(arr,src_dtype):
    if src_dtype=='bfloat16':
        import torch
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16).view(torch.float16).numpy().view(np.uint16)
    return arr.astype(np.float16).view(np.uint16)
class PrismTexError(Exception):pass
class PrismTexBundle:
    def __init__(self,header,payload):
        self.header=header;self.payload=payload
    @property
    def source_sha256(self):return self.header.get('source_sha256','')
    @property
    def contributor_id(self):return self.header.get('contributor_id','')
    @property
    def tensor_names(self):return [t['name'] for t in self.header.get('tensors',[])]
    @property
    def is_empty(self):return len(self.header.get('tensors',[]))==0
    def get_residual_bytes(self,name):
        for t in self.header['tensors']:
            if t['name']==name:
                off=int(t['byte_offset']);ln=int(t['byte_length'])
                return self.payload[off:off+ln]
        raise PrismTexError(f'tensor {name} not in bundle')
    @classmethod
    def export_from_bake(cls,bake_dir,contributor_id=None,tensor_names=None,note='',subject='global'):
        bake=Path(bake_dir)
        manifest=json.loads((bake/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('reffelt_scheme')!='gf17_digit_planes':raise PrismTexError(f'expected gf17 bake, got {manifest.get("reffelt_scheme")}')
        contributor_id=contributor_id or f'anon-{uuid.uuid4().hex[:8]}'
        all_tensors=manifest['tensors']
        def _rel_for_subject(v):
            paths=v.get('residual_paths')
            if paths and subject in paths:return paths[subject]
            return v.get('residual_path') if subject=='global' else None
        names=tensor_names or [k for k,v in all_tensors.items() if _rel_for_subject(v)]
        tensors=[];chunks=[];byte_offset=0
        for name in names:
            info=all_tensors.get(name)
            if info is None:continue
            rel=_rel_for_subject(info)
            if not rel:continue
            rp=bake/rel
            if not rp.exists():continue
            if info.get('asimov_immutable'):continue
            n=int(info['n_pixels'])
            data=rp.read_bytes()
            if len(data)!=4*n:raise PrismTexError(f'residual {name} size mismatch: {len(data)} vs {4*n}')
            tensors.append({'name':name,'n_pixels':n,'shape':info['shape'],'byte_offset':byte_offset,'byte_length':len(data),'plane_offsets':info['plane_offsets']})
            chunks.append(data);byte_offset+=len(data)
        header={'format':'prismtex/v1','source_sha256':manifest.get('source_sha256',''),'contributor_id':contributor_id,'subject':subject,'note':note,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'reffelt_k4':manifest.get('reffelt_k4',[1,17,289,4913]),'tensors':tensors}
        payload=b''.join(chunks)
        return cls(header,payload)
    def to_bytes(self):
        header_json=json.dumps(self.header,sort_keys=True).encode('utf-8')
        return _MAGIC+struct.pack('<BQ',_VERSION,len(header_json))+header_json+self.payload
    def write(self,path):
        Path(path).write_bytes(self.to_bytes())
        return Path(path)
    @classmethod
    def read(cls,path):
        b=Path(path).read_bytes()
        if not b.startswith(_MAGIC):raise PrismTexError(f'not a PrismTex file: {path}')
        version=b[8]
        if version!=_VERSION:raise PrismTexError(f'unsupported version {version}')
        hl=struct.unpack('<Q',b[9:17])[0]
        header=json.loads(b[17:17+hl].decode('utf-8'))
        payload=b[17+hl:]
        return cls(header,payload)
    @classmethod
    def merge(cls,bundles:List['PrismTexBundle'],contributor_id=None,note='merged')->'PrismTexBundle':
        if not bundles:raise PrismTexError('empty merge')
        first=bundles[0]
        src=first.source_sha256
        for b in bundles[1:]:
            if b.source_sha256!=src:raise PrismTexError(f'source_sha256 mismatch: {b.source_sha256[:16]} vs {src[:16]}')
        names=[]
        for b in bundles:
            for t in b.header.get('tensors',[]):
                if t['name'] not in names:names.append(t['name'])
        merged_tensors=[];merged_payload=[];byte_offset=0
        for name in names:
            ref_info=None
            sum_arr=None
            for b in bundles:
                try:
                    raw=b.get_residual_bytes(name)
                    info=next(t for t in b.header['tensors'] if t['name']==name)
                except (PrismTexError,StopIteration):continue
                arr=np.frombuffer(raw,dtype=np.uint8)
                if sum_arr is None:sum_arr=arr.astype(np.uint16);ref_info=info
                else:
                    if arr.shape!=sum_arr.shape:raise PrismTexError(f'shape mismatch on {name}: {arr.shape} vs {sum_arr.shape}')
                    sum_arr=(sum_arr+arr.astype(np.uint16))%17
            final=sum_arr.astype(np.uint8).tobytes()
            merged_tensors.append({'name':name,'n_pixels':ref_info['n_pixels'],'shape':ref_info['shape'],'byte_offset':byte_offset,'byte_length':len(final),'plane_offsets':ref_info['plane_offsets']})
            merged_payload.append(final);byte_offset+=len(final)
        header={'format':'prismtex/v1','source_sha256':src,'contributor_id':contributor_id or f'merge-{uuid.uuid4().hex[:8]}','note':note,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'reffelt_k4':first.header.get('reffelt_k4',[1,17,289,4913]),'merged_from':[b.contributor_id for b in bundles],'tensors':merged_tensors}
        return cls(header,b''.join(merged_payload))
    @classmethod
    def merge_fp16_avg(cls,bundles:List['PrismTexBundle'],base_bake_dir,contributor_id=None,note='fp16-avg-merged')->'PrismTexBundle':
        if not bundles:raise PrismTexError('empty merge')
        first=bundles[0]
        src=first.source_sha256
        merged_subject=first.header.get('subject','global')
        for b in bundles[1:]:
            if b.source_sha256!=src:raise PrismTexError(f'source_sha256 mismatch: {b.source_sha256[:16]} vs {src[:16]}')
            bs=b.header.get('subject','global')
            if bs!=merged_subject:raise PrismTexError(f'subject mismatch in merge: bundles must share subject (got {merged_subject!r} and {bs!r})')
        w=LearningWriter(base_bake_dir)
        if w.manifest.get('source_sha256','')!=src:raise PrismTexError(f'base bake source_sha256 mismatch: {w.manifest.get("source_sha256","")[:16]} vs {src[:16]}')
        names=[]
        for b in bundles:
            for t in b.header.get('tensors',[]):
                if t['name'] not in names:names.append(t['name'])
        merged_tensors=[];merged_payload=[];byte_offset=0
        for name in names:
            try:info=w.tensor_info(name)
            except Exception:continue
            if info.get('asimov_immutable'):continue
            n=int(info['n_pixels']);po=info['plane_offsets'];sd=w._src_dtype(info)
            base_arr=np.fromfile(w._path(info),dtype=np.uint8)
            base_d=[base_arr[int(po[k]):int(po[k])+n].astype(np.uint16) for k in _PLANE_KEYS]
            base_u32=base_d[0].astype(np.uint32)+base_d[1].astype(np.uint32)*17+base_d[2].astype(np.uint32)*289+base_d[3].astype(np.uint32)*4913
            base_u16=np.minimum(base_u32,65535).astype(np.uint16)
            base_fp=_u16arr_to_f32(base_u16,sd)
            sum_delta=np.zeros_like(base_fp);count=0;ref_info=None
            for b in bundles:
                try:
                    raw=b.get_residual_bytes(name)
                    bi=next(t for t in b.header['tensors'] if t['name']==name)
                except (PrismTexError,StopIteration):continue
                if ref_info is None:ref_info=bi
                r_arr=np.frombuffer(raw,dtype=np.uint8)
                r_d=[r_arr[int(po[k]):int(po[k])+n].astype(np.uint16) for k in _PLANE_KEYS]
                eff_d=[(b_d+r)%17 for b_d,r in zip(base_d,r_d)]
                eff_u32=eff_d[0].astype(np.uint32)+eff_d[1].astype(np.uint32)*17+eff_d[2].astype(np.uint32)*289+eff_d[3].astype(np.uint32)*4913
                eff_u16=np.minimum(eff_u32,65535).astype(np.uint16)
                recon_fp=_u16arr_to_f32(eff_u16,sd)
                delta=recon_fp-base_fp
                delta=np.where(np.isfinite(delta),delta,0.0).astype(np.float32)
                sum_delta+=delta;count+=1
            if count==0 or ref_info is None:continue
            avg_delta=sum_delta/count
            target_fp=base_fp+avg_delta
            target_u16=_f32arr_to_u16(target_fp,sd).astype(np.uint32)
            target_d=[(target_u16//k)%17 for k in (1,17,289,4913)]
            base_d_signed=[bd.astype(np.int32) for bd in base_d]
            new_residual=[((td.astype(np.int32)-bd)%17).astype(np.uint8) for bd,td in zip(base_d_signed,target_d)]
            full=np.zeros(4*n,dtype=np.uint8)
            for i,k in enumerate(_PLANE_KEYS):full[int(po[k]):int(po[k])+n]=new_residual[i]
            merged_tensors.append({'name':name,'n_pixels':n,'shape':ref_info['shape'],'byte_offset':byte_offset,'byte_length':int(full.size),'plane_offsets':ref_info['plane_offsets']})
            merged_payload.append(full.tobytes());byte_offset+=full.size
        header={'format':'prismtex/v1','source_sha256':src,'contributor_id':contributor_id or f'fp16avg-{uuid.uuid4().hex[:8]}','subject':merged_subject,'note':note,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'reffelt_k4':first.header.get('reffelt_k4',[1,17,289,4913]),'merged_from':[b.contributor_id for b in bundles],'merge_strategy':'fp16_average','tensors':merged_tensors}
        return cls(header,b''.join(merged_payload))
    def apply_to_bake(self,bake_dir,clobber=True):
        bake=Path(bake_dir)
        w=LearningWriter(bake)
        if w.manifest.get('source_sha256','')!=self.source_sha256:raise PrismTexError(f'bake source_sha256 mismatch: {w.manifest.get("source_sha256","")[:16]} vs {self.source_sha256[:16]}')
        target_subject=self.header.get('subject','global')
        applied=0;refused=0;additive_warned=False
        for t in self.header['tensors']:
            name=t['name']
            try:info=w.tensor_info(name)
            except Exception:continue
            if info.get('asimov_immutable'):refused+=1;continue
            n=int(t['n_pixels'])
            new_data=self.payload[int(t['byte_offset']):int(t['byte_offset'])+int(t['byte_length'])]
            new_arr=np.frombuffer(new_data,dtype=np.uint8)
            rp=w._residual_path(info,subject=target_subject,create=True)
            existing=np.memmap(rp,dtype=np.uint8,mode='r+',shape=(4*n,))
            if clobber:existing[:]=new_arr
            else:
                if not additive_warned and existing.any():
                    import warnings;warnings.warn('apply_to_bake(clobber=False) on non-empty residuals does mod-17 digit stacking which collapses for N>1 contributors. Use PrismTexBundle.merge_fp16_avg([bundles], base_bake) for correct N-way federation. This call is proceeding with the legacy stacking behavior.',DeprecationWarning,stacklevel=2)
                    additive_warned=True
                existing[:]=(existing.astype(np.uint16)+new_arr.astype(np.uint16))%17
            existing.flush()
            del existing
            import gc;gc.collect()
            applied+=1
        return {'applied':applied,'refused':refused,'tensors':[t['name'] for t in self.header['tensors']]}
