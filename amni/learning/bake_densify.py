import json,time,gc,numpy as np
from pathlib import Path
from amni.compute.reffelt4 import encode_fp16_to_rgba4,decode_rgba4_to_fp16
from amni.learning.prismtex import _u16arr_to_f32,_f32arr_to_u16
from amni.learning.gf17_writer import classify_tensor_tier,TIER_RULES
class DensifyError(Exception):pass
class TierProtectedError(PermissionError):pass
class BakeDensifier:
    def __init__(self,bake_dir):
        self.bake_dir=Path(bake_dir)
        mp=self.bake_dir/'manifest.json'
        if not mp.exists():raise DensifyError(f'no manifest at {mp}')
        self.manifest=json.loads(mp.read_text(encoding='utf-8'))
        if self.manifest.get('reffelt_scheme')!='rgba4':raise DensifyError(f'expected rgba4 bake, got {self.manifest.get("reffelt_scheme")}')
        self._bdir=self.bake_dir/'densify_backups';self._idx=self._bdir/'index.json'
    def _info(self,name):
        info=self.manifest['tensors'].get(name)
        if info is None:raise DensifyError(f'unknown tensor {name}')
        return info
    def _tier(self,name):
        info=self._info(name)
        return info.get('tier') or ('asimov' if info.get('asimov_immutable') else classify_tensor_tier(name))
    def writable(self,name):return bool(TIER_RULES.get(self._tier(name),{}).get('writable',False))
    def _natural(self,info):
        sh=info['shape'];return len(sh)==2 and int(info['page_w'])==int(sh[-1]) and int(info['page_h'])*int(info['page_w'])==int(info['n_pixels'])
    def _page(self,info,mode):return np.memmap(self.bake_dir/info['ptex_path'],dtype=np.uint8,mode=mode,shape=(int(info['page_h']),int(info['page_w']),4))
    def _idx_load(self):return json.loads(self._idx.read_text(encoding='utf-8')) if self._idx.exists() else {}
    def _idx_save(self,d):self._bdir.mkdir(exist_ok=True);self._idx.write_text(json.dumps(d,indent=1),encoding='utf-8')
    def read_region(self,name,r0,r1):
        info=self._info(name)
        if not self._natural(info):raise DensifyError(f'{name} not natural row-addressable (shape={info["shape"]} page=({info["page_h"]}x{info["page_w"]}))')
        pw=int(info['page_w']);rgba=np.ascontiguousarray(self._page(info,'r')[r0:r1]).reshape(-1,4)
        real=_u16arr_to_f32(decode_rgba4_to_fp16(rgba).view(np.uint16),info.get('source_dtype','float16'))
        return real.reshape(r1-r0,pw)
    def densify_region(self,name,r0,r1,delta,backup=True):
        if not self.writable(name):raise TierProtectedError(f'tier {self._tier(name)!r} refuses densify: {name}')
        info=self._info(name)
        if not self._natural(info):raise DensifyError(f'{name} not natural row-addressable')
        if info.get('u16_per_elem',1)!=1:raise DensifyError(f'{name} u16_per_elem!=1 unsupported in phase1')
        sd=info.get('source_dtype','float16');ph,pw=int(info['page_h']),int(info['page_w'])
        if r0<0 or r1>ph or r1<=r0:raise DensifyError(f'row range [{r0},{r1}) out of [0,{ph})')
        page=self._page(info,'r+');tile=page[r0:r1]
        if tile.shape[-1]!=4:raise DensifyError('k-invariant violated: rgba4 must keep 4 planes')
        m=(r1-r0)*pw;d=np.asarray(delta,dtype=np.float32).reshape(-1)
        d=np.broadcast_to(d,(m,)) if d.size in (1,m) else None
        if d is None:raise DensifyError(f'delta size must be 1 or {m}')
        bid=None
        if backup:
            self._bdir.mkdir(exist_ok=True);bid=f'{Path(info["ptex_path"]).stem}__r{r0}_{r1}__{int(time.time()*1000)}'
            (self._bdir/f'{bid}.bak').write_bytes(np.ascontiguousarray(tile).tobytes())
            ix=self._idx_load();ix.setdefault(name,[]).append({'id':bid,'r0':r0,'r1':r1,'pw':pw});self._idx_save(ix)
        old=_u16arr_to_f32(decode_rgba4_to_fp16(np.ascontiguousarray(tile).reshape(-1,4)).view(np.uint16),sd)
        new=old+d;new_u16=_f32arr_to_u16(new,sd);new_rgba=encode_fp16_to_rgba4(new_u16.view(np.float16)).reshape(r1-r0,pw,4)
        if new_rgba.shape[-1]!=4:raise DensifyError('k-invariant violated post-encode')
        tile[:]=new_rgba;page.flush();del page;gc.collect()
        rr=self.read_region(name,r0,r1).reshape(-1)
        num=float(np.dot(old,rr));den=float(np.linalg.norm(old)*np.linalg.norm(rr))+1e-12
        return {'tensor':name,'rows':r1-r0,'cols':pw,'weights':m,'planes':4,'scheme':'rgba4','moved_cos':num/den,'backup_id':bid}
    def rollback(self,name,backup_id=None):
        ix=self._idx_load();entries=ix.get(name,[])
        targets=[e for e in entries if backup_id is None or e['id']==backup_id]
        if not targets:return 0
        info=self._info(name);page=self._page(info,'r+');n=0
        for e in sorted(targets,key=lambda x:x['id'],reverse=True):
            bp=self._bdir/f'{e["id"]}.bak'
            if not bp.exists():continue
            tile=np.frombuffer(bp.read_bytes(),dtype=np.uint8).reshape(e['r1']-e['r0'],e['pw'],4)
            page[e['r0']:e['r1']]=tile;bp.unlink(missing_ok=True);n+=1
        page.flush();del page;gc.collect()
        ix[name]=[e for e in entries if e not in targets];self._idx_save(ix)
        return n
