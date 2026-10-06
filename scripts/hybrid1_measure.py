import os,sys,json,time,argparse,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)]
from amni.serve import blender_atex as B
import torch
def meas(e,v,prefill=2048,decode=512,seed=7):
    nt=os.environ.get("AMNI_BLENDER_NOTHR");os.environ["AMNI_BLENDER_NOTHR"]="1";g=torch.Generator().manual_seed(seed);ids=torch.randint(1000,min(30000,e.tc.vocab_size-1),(1,prefill),generator=g)
    e.generate(ids[:,:64],4,ign=True);torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();v3=v.cur();p0=v.pk;v.pk=v.cur();tm={};e.generate(ids,decode,ign=True,tm=tm);torch.cuda.synchronize();pk=v.pk;v.pk=max(p0,pk)
    nt is None and os.environ.pop("AMNI_BLENDER_NOTHR",None);tc=e.tc
    return {"opts":e.o,"after_warmup_proc_vram_GB":round(v3/1e9,3),"kv_cache_GB_at_ctx":round(e.kvb/1e9,4),"ctx_tokens":prefill+decode,"peak_proc_vram_GB":round(pk/1e9,3),"torch_max_alloc_GB":round(torch.cuda.max_memory_allocated()/1e9,3),"torch_max_reserved_GB":round(torch.cuda.max_memory_reserved()/1e9,3),"prefill_tok_s":round(tm["n_prompt"]/tm["prefill_s"],1),"decode_tok_s":round(tm["n_new"]/tm["decode_s"],2),"prefill_s":round(tm["prefill_s"],2),"decode_s":round(tm["decode_s"],2),"host_pinned_GB":round(e.host_bytes/1e9,3),"bake_manifest_GB":round(e.man["bake_bytes"]/1e9,3),"bits":e.man["bits"],"gs":e.man["gs"],"fmt":e.man.get("fmt"),"layers":tc.num_hidden_layers,"hidden":tc.hidden_size,"vocab":tc.vocab_size,"tied":bool(e.tied),"layer_types":sorted(set(getattr(tc,"layer_types",None) or ["full_attention"]))}
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--bake",required=True);ap.add_argument("--opts",default="{}");ap.add_argument("--prefill",type=int,default=2048);ap.add_argument("--decode",type=int,default=512);ap.add_argument("--out",required=True);ap.add_argument("--name",default="");a=ap.parse_args()
    v=B.Vram();v0=v.cur();torch.cuda.init();torch.zeros(1,device="cuda");v1=v.cur();v.start();o=json.loads(a.opts);o.pop("sys",None);t=time.time();e=B.Atex(a.bake,**o);torch.cuda.synchronize();ld=time.time()-t;gc.collect();torch.cuda.empty_cache();v2=v.cur();wa=torch.cuda.memory_allocated()
    r={"name":a.name or os.path.basename(a.bake.rstrip("/")),"load_s":round(ld,1),"vram_ctx_overhead_GB":round((v1-v0)/1e9,3),"proc_vram_after_ctx_GB":round(v1/1e9,3),"weights_vram_GB":round((v2-v1)/1e9,3),"proc_vram_after_load_GB":round(v2/1e9,3),"torch_alloc_after_load_GB":round(wa/1e9,3),**meas(e,v,a.prefill,a.decode)};v.stop()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);json.dump(r,open(a.out,"w"),indent=1);print(json.dumps(r))
