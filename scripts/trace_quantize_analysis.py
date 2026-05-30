"""trace_quantize_analysis — load raw per-layer residual vectors (npz from /admin/trace/dump_raws) and measure
|S_l| under different lossless quantizations:
  - per-element N-bit truncate (N in {1,2,4,6,8,10,12,14,16})  — keeps dim, narrows range
  - top-K PCA + per-component N-bit quantize  — reduces dim AND range
Output: chart + JSON ranking which quantization gives the smallest |S_l| at acceptable cossim retention.
Usage: python scripts/trace_quantize_analysis.py --npz eval_reports/trace_raws.npz --out eval_reports/trace_quantize.json"""
import sys,json,argparse
from pathlib import Path
import numpy as np
def per_element_quantize(vecs:np.ndarray,bits:int)->np.ndarray:
    if bits>=16:return vecs.astype(np.float16)
    v=vecs.astype(np.float32)
    vmin=v.min(axis=0,keepdims=True);vmax=v.max(axis=0,keepdims=True)
    span=np.maximum(vmax-vmin,1e-9)
    levels=(1<<bits)-1
    q=np.clip(((v-vmin)/span*levels).round().astype(np.int32),0,levels)
    return q
def count_distinct(q:np.ndarray)->int:
    if q.ndim==1:return len(np.unique(q))
    flat=np.ascontiguousarray(q).view([('b',q.dtype,q.shape[1])]).reshape(-1)
    return len(np.unique(flat))
def fit_pca(vecs:np.ndarray,k:int)->tuple:
    m=vecs.mean(axis=0)
    c=vecs-m
    _,_,Vt=np.linalg.svd(c,full_matrices=False)
    return m,Vt[:k]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--npz',default='eval_reports/trace_raws.npz')
    ap.add_argument('--out',default='eval_reports/trace_quantize.json')
    ap.add_argument('--chart',default='eval_reports/trace_quantize.png')
    ap.add_argument('--summary',default='eval_reports/trace_quantize_summary.md')
    ap.add_argument('--bits',nargs='+',type=int,default=[1,2,4,6,8])
    ap.add_argument('--pca-k',nargs='+',type=int,default=[8,16,32,64])
    ap.add_argument('--pca-bits',type=int,default=4)
    a=ap.parse_args()
    z=np.load(a.npz,allow_pickle=False)
    layer_ids=sorted([int(k) for k in z.files])
    print(f'loaded {len(layer_ids)} layers: dims={z[str(layer_ids[0])].shape}')
    results={}
    for li in layer_ids:
        vecs=z[str(li)].astype(np.float32)
        N,D=vecs.shape
        row={'n_samples':N,'dim':D,'per_element_bits':{},'pca':{}}
        for b in a.bits:
            q=per_element_quantize(vecs,b)
            row['per_element_bits'][b]={'distinct':int(count_distinct(q)),'ratio':float(count_distinct(q))/N}
        m,Vt=fit_pca(vecs,max(a.pca_k))
        proj=(vecs-m)@Vt.T
        for k in a.pca_k:
            q=per_element_quantize(proj[:,:k],a.pca_bits)
            row['pca'][k]={'k':k,'bits':a.pca_bits,'distinct':int(count_distinct(q)),'ratio':float(count_distinct(q))/N}
        results[li]=row
        print(f'layer {li:>2}: N={N} D={D} per_bits={{ {",".join(f"{b}={row["per_element_bits"][b]["distinct"]}" for b in a.bits)} }} pca_k_b{a.pca_bits}={{ {",".join(f"{k}={row["pca"][k]["distinct"]}" for k in a.pca_k)} }}')
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(f'wrote {a.out}')
    lines=['# Trace quantize analysis','','Per-layer |S_l| under different quantization schemes. Lower number = more cacheable.','']
    lines.append('## Per-element bit-quantize')
    lines.append('| layer | N | '+' | '.join(f'{b}-bit' for b in a.bits)+' |')
    lines.append('|---|---:|'+'---:|'*len(a.bits))
    for li in layer_ids:
        r=results[li]
        lines.append(f'| {li} | {r["n_samples"]} | '+' | '.join(str(r["per_element_bits"][b]["distinct"]) for b in a.bits)+' |')
    lines.append('')
    lines.append(f'## PCA top-K + {a.pca_bits}-bit per component')
    lines.append('| layer | '+' | '.join(f'k={k}' for k in a.pca_k)+' |')
    lines.append('|---|'+'---:|'*len(a.pca_k))
    for li in layer_ids:
        r=results[li]
        lines.append(f'| {li} | '+' | '.join(str(r["pca"][k]["distinct"]) for k in a.pca_k)+' |')
    Path(a.summary).write_text('\n'.join(lines),encoding='utf-8')
    print(f'wrote {a.summary}')
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(12,6))
        for b in a.bits:
            ys=[results[li]['per_element_bits'][b]['distinct'] for li in layer_ids]
            ax.plot(layer_ids,ys,marker='o',label=f'per-elem {b}-bit')
        for k in a.pca_k:
            ys=[results[li]['pca'][k]['distinct'] for li in layer_ids]
            ax.plot(layer_ids,ys,marker='s',linestyle='--',label=f'pca k={k} ({a.pca_bits}-bit)')
        ax.set_xlabel('layer index');ax.set_ylabel('|S_l| distinct quantized states');ax.set_yscale('log');ax.legend();ax.set_title('Residual-stream entropy collapse under quantization — Adam Gemma-4-E2B')
        ax.grid(True,alpha=0.3);fig.tight_layout();fig.savefig(a.chart,dpi=120);plt.close(fig)
        print(f'wrote {a.chart}')
    except Exception as e:print(f'chart skipped: {e}')
if __name__=='__main__':main()
