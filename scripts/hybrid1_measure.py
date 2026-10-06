import os,sys,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)];os.environ["AMNI_BLENDER_NOTHR"]="1"
ap=argparse.ArgumentParser();ap.add_argument("--bake",required=True);ap.add_argument("--opts",default="{}");ap.add_argument("--prefill",type=int,default=2048);ap.add_argument("--decode",type=int,default=512);ap.add_argument("--out",required=True);ap.add_argument("--name",default="");a=ap.parse_args()
from amni.serve import blender_atex as B
v=B.Vram();v0=v.cur()
import torch,gc
torch.cuda.init();torch.zeros(1,device="cuda");v1=v.cur()
o=json.loads(a.opts);o.pop("sys",None);t=time.time();e=B.Atex(a.bake,**o);torch.cuda.synchronize();ld=time.time()-t;gc.collect();torch.cuda.empty_cache();v2=v.cur();wa=torch.cuda.memory_allocated()
g=torch.Generator().manual_seed(7);ids=torch.randint(1000,min(30000,e.cfg.vocab_size-1),(1,a.prefill),generator=g)
e.generate(ids[:,:64],4,ign=True);torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
v.pk=v.cur();v.start();tm={};e.generate(ids,a.decode,ign=True,tm=tm);pk=v.stop()
kvq=e.kvb;cfg=e.cfg;tc=cfg.get_text_config() if hasattr(cfg,"get_text_config") else cfg
r={"name":a.name or os.path.basename(a.bake.rstrip("/")),"opts":o,"load_s":round(ld,1),"vram_ctx_overhead_GB":round((v1-v0)/1e9,3),"weights_vram_GB":round((v2-v1)/1e9,3),"torch_alloc_after_load_GB":round(wa/1e9,3),"torch_reserved_after_load_GB":round(torch.cuda.memory_reserved()/1e9,3),"kv_cache_GB_at_ctx":round(kvq/1e9,4),"ctx_tokens":a.prefill+a.decode,"peak_vram_GB_incl_ctx":round((pk-v0)/1e9,3),"peak_vram_GB_excl_ctx":round((pk-v1)/1e9,3),"torch_max_alloc_GB":round(torch.cuda.max_memory_allocated()/1e9,3),"torch_max_reserved_GB":round(torch.cuda.max_memory_reserved()/1e9,3),"prefill_tok_s":round(tm["n_prompt"]/tm["prefill_s"],1),"decode_tok_s":round(tm["n_new"]/tm["decode_s"],2),"prefill_s":round(tm["prefill_s"],2),"decode_s":round(tm["decode_s"],2),"bake_manifest_GB":round(e.man["bake_bytes"]/1e9,3),"bits":e.man["bits"],"gs":e.man["gs"],"layers":tc.num_hidden_layers,"hidden":tc.hidden_size,"vocab":tc.vocab_size,"tied":bool(e.tied)}
Path(a.out).parent.mkdir(parents=True,exist_ok=True);json.dump(r,open(a.out,"w"),indent=1);print(json.dumps(r))
