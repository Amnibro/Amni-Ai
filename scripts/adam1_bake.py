"""adam1-bake: convert any HuggingFace transformer to a GF(17) Adam-1 bake.
One command: download safetensors -> RGBA PTEX -> GF(17) digit planes -> tier-classified manifest.
Usage:
    python scripts/adam1_bake.py --hf-id Qwen/Qwen2.5-1.5B-Instruct --out bakes/qwen25_1_5b_gf17
    python scripts/adam1_bake.py --hf-id meta-llama/Llama-3.2-1B --out bakes/llama32_1b_gf17 --hf-token <tok>
"""
import argparse,sys,time,shutil,os,json
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_ROOT))
def _download(hf_id,cache_dir,token=None):
    from huggingface_hub import snapshot_download
    print(f'[adam1-bake] downloading {hf_id} -> {cache_dir}',flush=True)
    t0=time.time()
    path=snapshot_download(repo_id=hf_id,cache_dir=cache_dir,token=token,allow_patterns=['*.json','*.txt','*.safetensors','*.model'])
    print(f'  done in {time.time()-t0:.1f}s -> {path}',flush=True)
    return Path(path)
def _bake_safetensors_to_ptex(src,intermediate_dir,model_name):
    from scripts.v5_0_3_bake import bake
    print(f'[adam1-bake] safetensors -> RGBA PTEX (lossless GF(17) digits)',flush=True)
    t0=time.time()
    rc=bake(src,intermediate_dir,model_name)
    if rc!=0:raise RuntimeError(f'PTEX bake failed rc={rc}')
    print(f'  done in {time.time()-t0:.1f}s',flush=True)
def _convert_to_gf17_planes(intermediate_dir,out_dir):
    sys.argv=['ptex_to_gf17','--src-bake',str(intermediate_dir),'--out-bake',str(out_dir)]
    from scripts.v5_5_21_ptex_to_gf17 import main as gf17_main
    print(f'[adam1-bake] PTEX -> GF(17) digit planes (Adam-1 native format)',flush=True)
    t0=time.time()
    rc=gf17_main()
    if rc!=0:raise RuntimeError(f'GF17 conversion failed rc={rc}')
    print(f'  done in {time.time()-t0:.1f}s',flush=True)
def _copy_tokenizer_files(src,out):
    out=Path(out);src=Path(src)
    files=['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json','config.json','generation_config.json','chat_template.jinja','processor_config.json','preprocessor_config.json']
    n=0
    for f in files:
        s=src/f
        if s.exists():
            shutil.copy2(s,out/f);n+=1
    if n>0:print(f'[adam1-bake] copied {n} tokenizer/config files alongside bake')
def _assign_tiers(out):
    from amni.learning import LearningWriter
    w=LearningWriter(out)
    n=w.assign_tiers()
    s=w.tier_summary()
    print(f'[adam1-bake] tier classification ({n} tensors total):')
    for tier,count in s.items():print(f'    {tier:13s}: {count}')
def main():
    ap=argparse.ArgumentParser(description='adam1-bake: convert HF transformer to GF(17) Adam-1 bake')
    ap.add_argument('--hf-id',required=True,help='HuggingFace model id (e.g. Qwen/Qwen2.5-1.5B-Instruct)')
    ap.add_argument('--out',required=True,help='destination bake directory')
    ap.add_argument('--cache-dir',default='downloaded_models',help='where to cache downloaded safetensors')
    ap.add_argument('--hf-token',default=None,help='HF auth token for gated models')
    ap.add_argument('--keep-intermediate',action='store_true',help='keep the intermediate RGBA PTEX bake (default: delete)')
    ap.add_argument('--skip-download',action='store_true',help='use existing local snapshot (--hf-id must be a path)')
    ap.add_argument('--skip-tier-assign',action='store_true',help='do not auto-classify foundational tiers')
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists():
        print(f'[adam1-bake] WARNING: {out} exists. Remove it manually first if you want to re-bake.',file=sys.stderr)
        return 1
    src=Path(args.hf_id) if args.skip_download else _download(args.hf_id,args.cache_dir,args.hf_token)
    intermediate=out.parent/(out.name+'.ptex_intermediate')
    if intermediate.exists():shutil.rmtree(intermediate)
    model_name=args.hf_id.replace('/','_')
    print('='*60);print(f'[adam1-bake] {args.hf_id}');print(f'  src       = {src}');print(f'  out       = {out}');print('='*60)
    _bake_safetensors_to_ptex(src,intermediate,model_name)
    _convert_to_gf17_planes(intermediate,out)
    if not args.keep_intermediate:
        shutil.rmtree(intermediate);print(f'[adam1-bake] cleaned intermediate {intermediate}')
    else:
        print(f'[adam1-bake] kept intermediate at {intermediate}')
    _copy_tokenizer_files(src,out)
    if not args.skip_tier_assign:_assign_tiers(out)
    manifest=json.load(open(out/'manifest.json'))
    print()
    print('='*60)
    print(f'[adam1-bake] DONE')
    print(f'  bake at {out}')
    print(f'  {manifest["tensor_count"]} tensors, {manifest["total_params"]:,} params')
    print(f'  fp16 baseline: {manifest["fp16_baseline_bytes"]/1024/1024:.1f} MB')
    print(f'  gf17 stored:   {manifest["gf17_total_bytes"]/1024/1024:.1f} MB')
    print(f'  ratio:         {manifest["compression_ratio"]:.3f}x')
    print('='*60)
    print(f'\nNext steps:')
    print(f'  Inference:  python scripts/adam1_serve.py --bake {out} --model {src} --port 8000')
    print(f'  Quickstart: python examples/quickstart.py --bake {out} --model {src}')
    return 0
if __name__=='__main__':sys.exit(main())
