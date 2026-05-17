import json,hashlib,os,numpy as np
from pathlib import Path
from typing import Iterable,Tuple,Optional
_PLANE_KEYS=('d0','d1','d2','d3')
TIER_RULES={
    'asimov':       {'level':0,'authority':'system',  'writable':False,'rationale':'5 immutable laws — never modified'},
    'commandments': {'level':1,'authority':'anthony', 'writable':False,'rationale':'output voice — anthony only'},
    'ascension':    {'level':2,'authority':'anthony', 'writable':False,'rationale':'directive/purpose — anthony only'},
    'foundation':   {'level':3,'authority':'system',  'writable':False,'rationale':'structural stability — auto-managed'},
    'wisdom':       {'level':4,'authority':'swarm',   'writable':True, 'rationale':'subject-routable knowledge — federated'},
}
DEFAULT_TIER_PATTERNS=(
    ('asimov',('embed_tokens',)),
    ('commandments',('lm_head',)),
    ('ascension',('model.norm.weight','model.final_norm','final_norm')),
    ('foundation',('_layernorm.','.norm.','rotary_emb','.bias')),
)
def classify_tensor_tier(tensor_name,patterns=None):
    if patterns is None:patterns=DEFAULT_TIER_PATTERNS
    for tier,pats in patterns:
        for p in pats:
            if p in tensor_name:return tier
    return 'wisdom'
def _u16_to_native(u16,src_dtype):
    if src_dtype=='float16':return float(np.array([u16],dtype=np.uint16).view(np.float16)[0])
    if src_dtype=='bfloat16':
        import torch
        return float(torch.from_numpy(np.array([u16],dtype=np.uint16).view(np.float16)).view(torch.bfloat16)[0])
    return float(np.array([u16],dtype=np.uint16).view(np.float16)[0])
def _native_to_u16(value,src_dtype):
    if src_dtype=='float16':return int(np.array([value],dtype=np.float16).view(np.uint16)[0])
    if src_dtype=='bfloat16':
        import torch
        t=torch.tensor([float(value)],dtype=torch.bfloat16)
        return int(t.view(torch.float16).numpy().view(np.uint16)[0])
    return int(np.array([value],dtype=np.float16).view(np.uint16)[0])
class WeightAccessError(Exception):pass
class AsimovProtectedError(PermissionError):pass
class LearningWriter:
    def __init__(self,bake_dir):
        self.bake_dir=Path(bake_dir)
        mp=self.bake_dir/'manifest.json'
        if not mp.exists():raise WeightAccessError(f'no manifest at {mp}')
        self.manifest=json.loads(mp.read_text(encoding='utf-8'))
        if self.manifest.get('reffelt_scheme')!='gf17_digit_planes':raise WeightAccessError(f'expected gf17_digit_planes, got {self.manifest.get("reffelt_scheme")}')
        self._k4=tuple(self.manifest.get('reffelt_k4',(1,17,289,4913)))
        self._cache={}
        if os.environ.get('AMNI_VERIFY_INTEGRITY_ON_LOAD','0')=='1':self.verify_integrity(raise_on_mismatch=True)
    def verify_integrity(self,manifest_path=None,raise_on_mismatch=True):
        from amni.learning.integrity import verify_immutable_integrity
        return verify_immutable_integrity(self.bake_dir,manifest_path=manifest_path,raise_on_mismatch=raise_on_mismatch)
    def reload(self):
        mp=self.bake_dir/'manifest.json'
        self.manifest=json.loads(mp.read_text(encoding='utf-8'))
        self._cache.clear()
    def tensor_tier(self,tensor_name):
        info=self.manifest['tensors'].get(tensor_name)
        if info is None:return None
        t=info.get('tier')
        if t is None:return 'asimov' if info.get('asimov_immutable') else 'wisdom'
        return t
    def tier_info(self,tier):return TIER_RULES.get(tier,{})
    def is_writable_by(self,tensor_name,requestor='swarm'):
        tier=self.tensor_tier(tensor_name)
        if tier is None:return False
        rules=TIER_RULES.get(tier,{})
        if rules.get('writable',False):return rules.get('authority')==requestor or requestor=='anthony'
        return rules.get('authority')==requestor and rules.get('authority')!='system'
    def is_immutable(self,tensor_name,requestor='swarm'):
        return not self.is_writable_by(tensor_name,requestor)
    def list_by_tier(self,tier):return [k for k in self.manifest['tensors'] if self.tensor_tier(k)==tier]
    def tier_summary(self):
        out={t:0 for t in TIER_RULES}
        for k in self.manifest['tensors']:
            t=self.tensor_tier(k) or 'wisdom'
            out[t]=out.get(t,0)+1
        return out
    def assign_tiers(self,patterns=None,save=True):
        n_assigned=0
        for name,info in self.manifest['tensors'].items():
            tier=classify_tensor_tier(name,patterns)
            info['tier']=tier
            info['tier_authority']=TIER_RULES[tier]['authority']
            info.pop('asimov_immutable',None)
            info.pop('asimov_reason',None)
            n_assigned+=1
        if save:self._save_manifest()
        return n_assigned
    def list_tensors(self):return list(self.manifest['tensors'].keys())
    def tensor_info(self,tensor_name):
        info=self.manifest['tensors'].get(tensor_name)
        if info is None:raise WeightAccessError(f'unknown tensor {tensor_name}')
        return info
    def _path(self,info):return self.bake_dir/info['gf17_path']
    def _src_dtype(self,info):return info.get('source_dtype','float16')
    def _encode_digits(self,native_value,src_dtype):
        u=_native_to_u16(native_value,src_dtype)
        return (u%17,(u//17)%17,(u//289)%17,(u//4913)%17)
    def _decode_digits(self,d0,d1,d2,d3,src_dtype):
        u16=int(d0)+int(d1)*17+int(d2)*289+int(d3)*4913
        return _u16_to_native(u16,src_dtype)
    def read_weight(self,tensor_name,flat_idx):
        info=self.tensor_info(tensor_name)
        n=int(info['n_pixels'])
        if flat_idx<0 or flat_idx>=n:raise WeightAccessError(f'flat_idx {flat_idx} out of range [0,{n})')
        po=info['plane_offsets']
        with open(self._path(info),'rb') as f:
            digits=[]
            for key in ('d0','d1','d2','d3'):
                f.seek(int(po[key])+flat_idx);digits.append(f.read(1)[0])
        return float(self._decode_digits(*digits,src_dtype=self._src_dtype(info)))
    def write_weight(self,tensor_name,flat_idx,new_value):
        if self.is_immutable(tensor_name,requestor='swarm'):raise AsimovProtectedError(f'tier-protected tensor ({self.tensor_tier(tensor_name)}) refuses writes: {tensor_name}')
        info=self.tensor_info(tensor_name)
        n=int(info['n_pixels'])
        if flat_idx<0 or flat_idx>=n:raise WeightAccessError(f'flat_idx {flat_idx} out of range [0,{n})')
        d=self._encode_digits(new_value,self._src_dtype(info))
        po=info['plane_offsets']
        with open(self._path(info),'r+b') as f:
            for key,digit in zip(('d0','d1','d2','d3'),d):
                f.seek(int(po[key])+flat_idx);f.write(bytes([int(digit)]))
        readback=self.read_weight(tensor_name,flat_idx)
        return readback
    def write_weights_batch(self,tensor_name,updates:Iterable[Tuple[int,float]]):
        if self.is_immutable(tensor_name,requestor='swarm'):raise AsimovProtectedError(f'tier-protected tensor ({self.tensor_tier(tensor_name)}) refuses writes: {tensor_name}')
        info=self.tensor_info(tensor_name)
        n=int(info['n_pixels'])
        po=info['plane_offsets']
        sd=self._src_dtype(info)
        applied=0
        with open(self._path(info),'r+b') as f:
            for flat_idx,new_value in updates:
                if flat_idx<0 or flat_idx>=n:raise WeightAccessError(f'flat_idx {flat_idx} out of range [0,{n})')
                d=self._encode_digits(new_value,sd)
                for key,digit in zip(('d0','d1','d2','d3'),d):
                    f.seek(int(po[key])+flat_idx);f.write(bytes([int(digit)]))
                applied+=1
        return applied
    def mark_immutable(self,tensor_names,save=True):
        for name in tensor_names:
            info=self.tensor_info(name)
            info['asimov_immutable']=True
        if save:self._save_manifest()
    def unmark_immutable(self,tensor_names,save=True):
        for name in tensor_names:
            info=self.tensor_info(name)
            info.pop('asimov_immutable',None)
        if save:self._save_manifest()
    def list_immutable(self):return [k for k,v in self.manifest['tensors'].items() if v.get('asimov_immutable')]
    def auto_protect_foundational(self,patterns=None,save=True):
        if patterns is None:patterns=('embed_tokens','lm_head','_layernorm.','model.norm','.bias','rotary_emb')
        protected=[]
        for name in self.manifest['tensors']:
            for p in patterns:
                if p in name:
                    info=self.manifest['tensors'][name]
                    info['asimov_immutable']=True
                    info['asimov_reason']='foundational'
                    protected.append(name)
                    break
        if save:self._save_manifest()
        return protected
    def _save_manifest(self):
        tmp=self.bake_dir/'manifest.json.tmp'
        with open(tmp,'w',encoding='utf-8') as f:json.dump(self.manifest,f,indent=2)
        tmp.replace(self.bake_dir/'manifest.json')
    def _residual_paths_dict(self,info):
        d=info.get('residual_paths')
        if d:return d
        legacy=info.get('residual_path')
        if legacy:return {'global':legacy}
        return {}
    def _residual_path(self,info,subject='global',create=False):
        paths=self._residual_paths_dict(info)
        rel=paths.get(subject)
        if rel:
            rp=self.bake_dir/rel
            if rp.exists():return rp
        if not create:return None
        sk=Path(info['gf17_path']).stem
        suffix=f'.{subject}' if subject!='global' else ''
        rp=self.bake_dir/'tensors'/f'{sk}{suffix}.gf17res'
        n=int(info['n_pixels'])
        with open(rp,'wb') as f:f.write(np.zeros(4*n,dtype=np.uint8).tobytes())
        if 'residual_paths' not in info:info['residual_paths']={}
        info['residual_paths'][subject]=f'tensors/{sk}{suffix}.gf17res'
        if subject=='global':info['residual_path']=info['residual_paths'][subject]
        self._save_manifest()
        return rp
    def has_residuals(self,tensor_name,subject=None):
        info=self.tensor_info(tensor_name)
        paths=self._residual_paths_dict(info)
        if subject is None:return len(paths)>0
        return subject in paths
    def list_residual_tensors(self,subject=None):
        out=[]
        for k,v in self.manifest['tensors'].items():
            paths=self._residual_paths_dict(v)
            if subject is None and len(paths)>0:out.append(k)
            elif subject is not None and subject in paths:out.append(k)
        return out
    def list_subjects(self):
        subjects=set()
        for v in self.manifest['tensors'].values():
            for s in self._residual_paths_dict(v).keys():subjects.add(s)
        return sorted(subjects)
    def read_residual_digits(self,tensor_name,flat_idx,subject='global'):
        info=self.tensor_info(tensor_name)
        n=int(info['n_pixels'])
        if flat_idx<0 or flat_idx>=n:raise WeightAccessError(f'flat_idx {flat_idx} out of range [0,{n})')
        rp=self._residual_path(info,subject=subject,create=False)
        if rp is None:return (0,0,0,0)
        po=info['plane_offsets']
        with open(rp,'rb') as f:
            digits=[]
            for key in _PLANE_KEYS:
                f.seek(int(po[key])+flat_idx);digits.append(f.read(1)[0])
        return tuple(digits)
    def write_residual_digits(self,tensor_name,flat_idx,digits:Tuple[int,int,int,int],subject='global'):
        if self.is_immutable(tensor_name,requestor='swarm'):raise AsimovProtectedError(f'tier-protected tensor ({self.tensor_tier(tensor_name)}) refuses residual writes: {tensor_name}')
        info=self.tensor_info(tensor_name)
        n=int(info['n_pixels'])
        if flat_idx<0 or flat_idx>=n:raise WeightAccessError(f'flat_idx {flat_idx} out of range [0,{n})')
        for d in digits:
            if d<0 or d>16:raise WeightAccessError(f'residual digit {d} out of GF(17) range [0,16]')
        rp=self._residual_path(info,subject=subject,create=True)
        po=info['plane_offsets']
        with open(rp,'r+b') as f:
            for key,digit in zip(_PLANE_KEYS,digits):
                f.seek(int(po[key])+flat_idx);f.write(bytes([int(digit)]))
        return tuple(digits)
    def add_residual_digits(self,tensor_name,flat_idx,deltas:Tuple[int,int,int,int],subject='global'):
        existing=self.read_residual_digits(tensor_name,flat_idx,subject=subject)
        new=tuple((int(e)+int(d))%17 for e,d in zip(existing,deltas))
        return self.write_residual_digits(tensor_name,flat_idx,new,subject=subject)
    def encode_target_array_as_residuals(self,tensor_name,target_array,additive=False,subject='global'):
        if self.is_immutable(tensor_name,requestor='swarm'):raise AsimovProtectedError(f'tier-protected tensor ({self.tensor_tier(tensor_name)}) refuses residual writes: {tensor_name}')
        info=self.tensor_info(tensor_name)
        n=int(info['n_pixels'])
        sd=self._src_dtype(info)
        base_arr=np.fromfile(self._path(info),dtype=np.uint8)
        po=info['plane_offsets']
        base_d=[base_arr[int(po[k]):int(po[k])+n].astype(np.uint16) for k in _PLANE_KEYS]
        target_flat=np.asarray(target_array).reshape(-1)
        if target_flat.size!=n:raise WeightAccessError(f'target size {target_flat.size} != n_pixels {n}')
        if sd=='bfloat16':
            import torch
            t=torch.from_numpy(target_flat.astype(np.float32) if target_flat.dtype!=np.float32 else target_flat).to(torch.bfloat16)
            target_u16=t.view(torch.float16).numpy().view(np.uint16)
        elif sd=='float16':
            target_u16=target_flat.astype(np.float16).view(np.uint16)
        else:
            target_u16=target_flat.astype(np.float16).view(np.uint16)
        target_d=[((target_u16.astype(np.uint32))//k%17).astype(np.int32) for k in (1,17,289,4913)]
        base_d_signed=[bd.astype(np.int32) for bd in base_d]
        new_residual=[((td-bd)%17).astype(np.uint8) for bd,td in zip(base_d_signed,target_d)]
        rp=self._residual_path(info,subject=subject,create=True)
        existing=np.memmap(rp,dtype=np.uint8,mode='r+',shape=(4*n,))
        if additive:
            for i,k in enumerate(_PLANE_KEYS):
                seg=existing[int(po[k]):int(po[k])+n]
                seg[:]=((seg.astype(np.uint16)+new_residual[i].astype(np.uint16))%17).astype(np.uint8)
        else:
            for i,k in enumerate(_PLANE_KEYS):
                existing[int(po[k]):int(po[k])+n]=new_residual[i]
        existing.flush()
        del existing
        import gc;gc.collect()
        return n
    def read_effective_weight(self,tensor_name,flat_idx,subjects=('global',)):
        info=self.tensor_info(tensor_name)
        n=int(info['n_pixels'])
        if flat_idx<0 or flat_idx>=n:raise WeightAccessError(f'flat_idx {flat_idx} out of range [0,{n})')
        po=info['plane_offsets']
        with open(self._path(info),'rb') as f:
            base=[]
            for key in _PLANE_KEYS:
                f.seek(int(po[key])+flat_idx);base.append(f.read(1)[0])
        eff=[int(b) for b in base]
        for s in subjects:
            r=self.read_residual_digits(tensor_name,flat_idx,subject=s)
            eff=[(e+int(rd))%17 for e,rd in zip(eff,r)]
        return float(self._decode_digits(*eff,src_dtype=self._src_dtype(info)))
    def clear_residuals(self,tensor_name,subject=None,save=True):
        info=self.tensor_info(tensor_name)
        paths=self._residual_paths_dict(info)
        targets=[subject] if subject else list(paths.keys())
        cleared=0
        for s in targets:
            rp=self._residual_path(info,subject=s,create=False)
            if rp is None:continue
            n=int(info['n_pixels'])
            arr=np.memmap(rp,dtype=np.uint8,mode='r+',shape=(4*n,))
            arr[:]=0;arr.flush()
            del arr
            import gc;gc.collect()
            cleared+=1
        return cleared
    def remove_residuals(self,tensor_name,subject=None,save=True):
        info=self.tensor_info(tensor_name)
        paths=self._residual_paths_dict(info)
        targets=[subject] if subject else list(paths.keys())
        removed=0
        for s in targets:
            rp=self._residual_path(info,subject=s,create=False)
            if rp is None:continue
            rp.unlink(missing_ok=True)
            if 'residual_paths' in info:info['residual_paths'].pop(s,None)
            if s=='global':info.pop('residual_path',None);info.pop('residual_bytes',None)
            removed+=1
        if 'residual_paths' in info and not info['residual_paths']:info.pop('residual_paths')
        if save:self._save_manifest()
        return removed
