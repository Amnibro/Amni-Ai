import os,sys,json,subprocess,argparse
from pathlib import Path
from functools import reduce
_ROOT=Path(__file__).resolve().parents[1]
def st_header(path):
    with open(path,'rb') as f:
        n=int.from_bytes(f.read(8),'little');return {k:v for k,v in json.loads(f.read(n)).items() if k!='__metadata__'}
def ensure_src(repo,dest):
    d=Path(dest)
    if any(d.glob('*.safetensors')):print(f'[launch] src present {d}',flush=True);return d
    from huggingface_hub import snapshot_download
    print(f'[launch] downloading {repo} (~weights, may take a while)...',flush=True)
    snapshot_download(repo_id=repo,local_dir=str(d),allow_patterns=['*.safetensors','*.json','*.txt','tokenizer*','*.model','*.jinja'])
    return d
def expected(src):
    names={};params=0
    for s in sorted(Path(src).glob('*.safetensors')):
        for k,v in st_header(s).items():
            if k in names:continue
            sh=v.get('shape',[]);names[k]=sh;params+=reduce(lambda a,b:a*b,sh,1) if sh else 0
    return len(names),params
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',default='ibm-granite/granite-4.1-8b');ap.add_argument('--dest',default='downloaded_models/granite-4.1-8b');ap.add_argument('--out',default='bakes/granite41_8b_gf17');ap.add_argument('--name',default='granite_4_1_8b_gf17');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'_expected.json').write_text(json.dumps({'model':a.name,'dest':a.dest,'phase':'downloading','total':0,'total_params':0}))
    src=ensure_src(a.repo,a.dest);tot,par=expected(src)
    (out/'_expected.json').write_text(json.dumps({'model':a.name,'dest':a.dest,'phase':'baking','total':tot,'total_params':par}))
    print(f'[launch] expected tensors={tot} params={par:,}; baking -> {out}',flush=True)
    py=str(_ROOT/'.venv/Scripts/python.exe');py=py if Path(py).exists() else sys.executable
    r=subprocess.run([py,str(_ROOT/'scripts/v5_0_3_bake.py'),'--src',str(src),'--out',str(out),'--model-name',a.name])
    print(f'[launch] bake exit={r.returncode}',flush=True);sys.exit(r.returncode)
if __name__=='__main__':main()
