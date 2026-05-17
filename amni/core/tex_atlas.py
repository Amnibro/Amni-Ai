import numpy as np,struct,json as _json,os
from pathlib import Path
from typing import Dict,List,Tuple,Optional
from collections import defaultdict
TEX_MAX=int(os.environ.get('AMNI_TEX_MAX','4096'))
TEX_PIXELS=TEX_MAX*TEX_MAX
BYTES_PER_PX=4
SLOTS_PER_REL=20
SLOTS_COOC=50
SLOTS_VEC_CHUNK=32
TEX_MAGIC=0x544D5558
class TexPage:
    __slots__=('name','width','height','data','path')
    def __init__(self,name:str,width:int=TEX_MAX,height:int=TEX_MAX):
        self.name=name
        self.width=width
        self.height=height
        self.data=np.zeros((height,width,4),dtype=np.uint8)
        self.path=""
    def set_f32(self,row:int,col:int,val:float):
        if row>=self.height or col>=self.width: return
        b=np.float32(val).tobytes()
        self.data[row,col]=np.frombuffer(b,dtype=np.uint8)
    def get_f32(self,row:int,col:int)->float:
        if row>=self.height or col>=self.width: return 0.0
        return np.frombuffer(self.data[row,col].tobytes(),dtype=np.float32)[0]
    def set_u16_pair(self,row:int,col:int,id_val:int,weight_u16:int):
        if row>=self.height or col>=self.width: return
        self.data[row,col,0]=id_val&0xFF
        self.data[row,col,1]=(id_val>>8)&0xFF
        self.data[row,col,2]=weight_u16&0xFF
        self.data[row,col,3]=(weight_u16>>8)&0xFF
    def get_u16_pair(self,row:int,col:int)->Tuple[int,int]:
        if row>=self.height or col>=self.width: return 0,0
        id_val=int(self.data[row,col,0])|(int(self.data[row,col,1])<<8)
        wt=int(self.data[row,col,2])|(int(self.data[row,col,3])<<8)
        return id_val,wt
    def set_i32(self,row:int,col:int,val:int):
        if row>=self.height or col>=self.width: return
        b=np.int32(val).tobytes()
        self.data[row,col]=np.frombuffer(b,dtype=np.uint8)
    def get_i32(self,row:int,col:int)->int:
        if row>=self.height or col>=self.width: return 0
        return int(np.frombuffer(self.data[row,col].tobytes(),dtype=np.int32)[0])
    def flat_u8(self)->np.ndarray:
        return self.data.reshape(-1)
    def flat_f32(self)->np.ndarray:
        return self.data.reshape(-1,4).view(np.float32).flatten()
    def save(self,dir_path:str):
        p=Path(dir_path)
        p.mkdir(parents=True,exist_ok=True)
        self.path=str(p/f"{self.name}.tex")
        with open(self.path,'wb') as f:
            np.array([TEX_MAGIC],dtype=np.uint32).tofile(f)
            np.array([self.width,self.height],dtype=np.int32).tofile(f)
            self.data.tofile(f)
    @classmethod
    def load(cls,path:str)->'TexPage':
        with open(path,'rb') as f:
            mg=np.fromfile(f,dtype=np.uint32,count=1)[0]
            assert mg==TEX_MAGIC,f"bad tex magic: {mg:#x}"
            dims=np.fromfile(f,dtype=np.int32,count=2)
            w,h=int(dims[0]),int(dims[1])
            page=cls(Path(path).stem,w,h)
            page.data=np.fromfile(f,dtype=np.uint8,count=w*h*4).reshape(h,w,4)
            page.path=path
            return page
    def used_rows(self)->int:
        return int(np.any(self.data.reshape(self.height,-1),axis=1).sum())
    def size_bytes(self)->int:
        return self.width*self.height*4
class TexAtlas:
    def __init__(self,name:str="atlas"):
        self.name=name
        self.pages:Dict[str,TexPage]={}
        self.meta:dict={}
    def add_page(self,page:TexPage):
        self.pages[page.name]=page
    def get_page(self,name:str)->Optional[TexPage]:
        return self.pages.get(name)
    def save(self,dir_path:str):
        p=Path(dir_path)
        p.mkdir(parents=True,exist_ok=True)
        for pg in self.pages.values():
            pg.save(str(p))
        self.meta["pages"]={n:{"w":pg.width,"h":pg.height,"path":pg.path,"used_rows":pg.used_rows()} for n,pg in self.pages.items()}
        self.meta["name"]=self.name
        with open(str(p/"tex_atlas_meta.json"),'w') as f: _json.dump(self.meta,f)
    @classmethod
    def load(cls,dir_path:str)->'TexAtlas':
        p=Path(dir_path)
        with open(str(p/"tex_atlas_meta.json")) as f: meta=_json.load(f)
        atlas=cls(meta.get("name","atlas"))
        atlas.meta=meta
        for name,info in meta.get("pages",{}).items():
            pg=TexPage.load(info["path"])
            atlas.pages[name]=pg
        return atlas
    def total_bytes(self)->int:
        return sum(pg.size_bytes() for pg in self.pages.values())
def f16_to_u16(val:float)->int:
    return int(np.float16(val).view(np.uint16))
def u16_to_f16(val:int)->float:
    return float(np.uint16(val).view(np.float16))
def pack_lexicon_textures(lexicon,cooc_sparse:Dict[int,Dict[int,float]],out_dir:str,rel_slots:int=SLOTS_PER_REL,cooc_slots:int=SLOTS_COOC)->TexAtlas:
    from amni.core.lexicon import REL_TYPES,N_RELS
    n_words=lexicon.vocab_size
    dim=lexicon.dim
    print(f"packing lexicon into TMU textures: {n_words} words, {dim}d")
    rows_needed=n_words
    pages_needed=(rows_needed+TEX_MAX-1)//TEX_MAX
    print(f"  words need {pages_needed} page(s) of {TEX_MAX} rows")
    atlas=TexAtlas("lexicon_tmu")
    meta_cols=4
    meta_pages=[]
    for pi in range(pages_needed):
        pg=TexPage(f"meta_{pi}",meta_cols,TEX_MAX)
        row_start=pi*TEX_MAX
        row_end=min(row_start+TEX_MAX,n_words)
        for nid in range(row_start,row_end):
            r=nid-row_start
            w=lexicon.nonce_to_word.get(nid)
            if not w: continue
            wn=lexicon.word_nonces[w]
            pg.set_i32(r,0,nid)
            pg.set_i32(r,1,wn.pos_id)
            pg.set_i32(r,2,wn.domain_id)
            pg.set_f32(r,3,wn.freq)
        atlas.add_page(pg)
        meta_pages.append(pg)
    print(f"  meta textures: {len(meta_pages)} pages ({meta_cols} cols)")
    rel_cols=N_RELS*rel_slots
    actual_rel_cols=min(rel_cols,TEX_MAX)
    rel_pages=[]
    for pi in range(pages_needed):
        pg=TexPage(f"rels_{pi}",actual_rel_cols,TEX_MAX)
        row_start=pi*TEX_MAX
        row_end=min(row_start+TEX_MAX,n_words)
        for nid in range(row_start,row_end):
            r=nid-row_start
            w=lexicon.nonce_to_word.get(nid)
            if not w: continue
            entry=lexicon.entries.get(w)
            if not entry: continue
            rels=entry.all_relations()
            for rel_type,rel_words in rels.items():
                base_col=rel_type*rel_slots
                for si,rw in enumerate(sorted(rel_words)[:rel_slots]):
                    rwn=lexicon.lookup(rw)
                    if not rwn: continue
                    col=base_col+si
                    if col>=actual_rel_cols: break
                    wt=f16_to_u16(1.0)
                    pg.set_u16_pair(r,col,rwn.nonce_id,wt)
        atlas.add_page(pg)
        rel_pages.append(pg)
    print(f"  relationship textures: {len(rel_pages)} pages ({actual_rel_cols} cols)")
    cooc_pages=[]
    for pi in range(pages_needed):
        pg=TexPage(f"cooc_{pi}",cooc_slots,TEX_MAX)
        row_start=pi*TEX_MAX
        row_end=min(row_start+TEX_MAX,n_words)
        for nid in range(row_start,row_end):
            r=nid-row_start
            neighbors=cooc_sparse.get(nid,{})
            sorted_nb=sorted(neighbors.items(),key=lambda x:-x[1])[:cooc_slots]
            for ci,(target_nid,weight) in enumerate(sorted_nb):
                wt=f16_to_u16(min(weight,1.0))
                pg.set_u16_pair(r,ci,int(target_nid)&0xFFFF,wt)
        atlas.add_page(pg)
        cooc_pages.append(pg)
    print(f"  co-occurrence textures: {len(cooc_pages)} pages ({cooc_slots} cols)")
    vec_chunks=(dim+SLOTS_VEC_CHUNK-1)//SLOTS_VEC_CHUNK
    vec_pages=[]
    for chunk_i in range(vec_chunks):
        d_start=chunk_i*SLOTS_VEC_CHUNK
        d_end=min(d_start+SLOTS_VEC_CHUNK,dim)
        chunk_cols=d_end-d_start
        for pi in range(pages_needed):
            pg=TexPage(f"vec_{chunk_i}_{pi}",chunk_cols,TEX_MAX)
            row_start=pi*TEX_MAX
            row_end=min(row_start+TEX_MAX,n_words)
            for nid in range(row_start,row_end):
                r=nid-row_start
                w=lexicon.nonce_to_word.get(nid)
                if not w: continue
                wn=lexicon.word_nonces[w]
                for di in range(chunk_cols):
                    pg.set_f32(r,di,float(wn.vector[d_start+di]))
            atlas.add_page(pg)
            vec_pages.append(pg)
    print(f"  vector textures: {len(vec_pages)} pages ({vec_chunks} chunks x {SLOTS_VEC_CHUNK} dims)")
    atlas.meta["layout"]={
        "n_words":n_words,"dim":dim,"rel_slots":rel_slots,"cooc_slots":cooc_slots,
        "vec_chunk_size":SLOTS_VEC_CHUNK,"vec_chunks":vec_chunks,"pages_per_word_range":pages_needed,
        "n_rels":N_RELS,"meta_cols":meta_cols,"rel_cols":actual_rel_cols
    }
    atlas.save(out_dir)
    total_mb=atlas.total_bytes()/(1024*1024)
    print(f"  total texture atlas: {total_mb:.1f} MB ({len(atlas.pages)} pages)")
    return atlas
def load_tex_to_gpu(page:TexPage,device:str="cuda")->dict:
    import torch
    flat=page.flat_u8()
    gpu_buf=torch.from_numpy(flat.copy()).to(device)
    from amni.compute.hip_lut_kernel import create_tex_u8
    tex_obj=create_tex_u8(gpu_buf.data_ptr(),gpu_buf.numel())
    return {"gpu_buf":gpu_buf,"tex_obj":tex_obj,"page":page}
def load_atlas_pages_gpu(atlas:TexAtlas,page_names:List[str],device:str="cuda")->Dict[str,dict]:
    loaded={}
    for name in page_names:
        pg=atlas.get_page(name)
        if pg: loaded[name]=load_tex_to_gpu(pg,device)
    return loaded
def unload_tex_gpu(loaded:dict):
    from amni.compute.hip_lut_kernel import destroy_tex
    if "tex_obj" in loaded: destroy_tex(loaded["tex_obj"])
