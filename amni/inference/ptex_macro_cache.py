"""PtexMacroCache — intern a full response into ONE PTEX page addressed by a single macro-token, so a cached sentence/response costs 1 token (a page fetch) instead of N model forward passes. Anthony's "temporarily tokenize a full response into ptex as a single token to accelerate stuff." Token ids -> base-17 Reffelt digits (5 tiers, covers vocab<1.4M) -> RGBA8 PTEX page (lossless, creed-aligned: 1 channel = 1 base-17 digit). expand() reconstructs ids BIT-EXACT. Strict-additive accelerator: recall = O(page) memcpy, never slower than inference, deterministic + fixable. Session-scoped (temporary); optional disk persist via save/load."""
import numpy as np,os,time
K5=np.array([1,17,289,4913,83521],dtype=np.int64)
class PtexMacroCache:
    def __init__(s,tok=None,page_w=256):
        s.tok=tok;s.page_w=int(page_w);s._pages={};s._meta={};s._index={};s._next=1
    def _ids_to_page(s,ids):
        a=np.asarray(ids,dtype=np.int64)
        if a.size==0:return np.zeros((1,s.page_w,4),np.uint8)
        d=((a[:,None]//K5[None,:])%17).astype(np.uint8).reshape(-1)
        pad=(-len(d))%4
        if pad:d=np.concatenate([d,np.zeros(pad,np.uint8)])
        px=d.reshape(-1,4);h=(len(px)+s.page_w-1)//s.page_w
        page=np.zeros((h*s.page_w,4),np.uint8);page[:len(px)]=px
        return page.reshape(h,s.page_w,4)
    def _page_to_ids(s,page,n):
        flat=page.reshape(-1)[:5*n].astype(np.int64).reshape(n,5)
        return (flat*K5[None,:]).sum(1)
    def intern(s,text):
        if text in s._index:return s._index[text]
        ids=list(s.tok.encode(text,add_special_tokens=False)) if s.tok is not None else []
        mid=s._next;s._next+=1
        s._pages[mid]=s._ids_to_page(ids);s._meta[mid]={'n':len(ids)};s._index[text]=mid
        return mid
    def expand_ids(s,mid):
        return s._page_to_ids(s._pages[mid],s._meta[mid]['n']).tolist()
    def expand(s,mid):
        ids=s.expand_ids(mid)
        return s.tok.decode(ids,skip_special_tokens=True) if s.tok is not None else ids
    def page_bytes(s,mid):return int(s._pages[mid].nbytes)
    def stats(s):return {'macros':len(s._pages),'tokens_total':int(sum(m['n'] for m in s._meta.values()))}
    def save(s,root):
        os.makedirs(root,exist_ok=True)
        for mid,page in s._pages.items():np.save(os.path.join(root,f'macro_{mid}.npy'),page)
        np.save(os.path.join(root,'_meta.npy'),np.array([(mid,s._meta[mid]['n']) for mid in s._pages],dtype=np.int64))
    def load(s,root):
        mp=os.path.join(root,'_meta.npy')
        if not os.path.exists(mp):return 0
        n=0
        for mid,ntok in np.load(mp):
            p=os.path.join(root,f'macro_{int(mid)}.npy')
            if os.path.exists(p):
                s._pages[int(mid)]=np.load(p);s._meta[int(mid)]={'n':int(ntok)};s._next=max(s._next,int(mid)+1)
                if s.tok is not None:s._index[s.expand(int(mid))]=int(mid)
                n+=1
        return n
if __name__=='__main__':
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('bakes/gemma4_12b_nvfp4_atex')
    c=PtexMacroCache(tok)
    resps=['The answer is (D). The field below an infinite charged plane is sigma/(2*epsilon_0) pointing downward, equal in magnitude to the field above.','17 * 23 = 391.','Paris is the capital of France, and the boiling point of water is 100 degrees Celsius.']
    ok=0
    for r in resps:
        mid=c.intern(r);back=c.expand_ids(mid);ref=list(tok.encode(r,add_special_tokens=False));same=back==ref;ok+=same
        print(f'mid={mid}: ONE macro-token <- {c._meta[mid]["n"]} real tokens | id-roundtrip={"LOSSLESS" if same else "FAIL"} | page={c.page_bytes(mid)}B',flush=True)
        print(f'   expand->text: {c.expand(mid)[:80]!r}',flush=True)
    big=' '.join(resps*30);mid=c.intern(big);N=c._meta[mid]['n']
    t0=time.perf_counter();[c.expand_ids(mid) for _ in range(2000)];te=(time.perf_counter()-t0)/2000*1e6
    t0=time.perf_counter();[tok.encode(big,add_special_tokens=False) for _ in range(50)];tt=(time.perf_counter()-t0)/50*1e6
    print(f'\nexpand a {N}-token response from 1 macro-token: {te:.1f}us  |  re-tokenize same text: {tt:.0f}us  |  (model generation of {N} tokens ~= {N*30/1000:.1f}s)',flush=True)
    print(f'ROUNDTRIP {ok}/{len(resps)} lossless | {c.stats()}',flush=True)
    print('PTEX_MACRO_OK' if ok==len(resps) else 'CHECK',flush=True)
