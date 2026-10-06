import os,sys,json,glob
from pathlib import Path
H=Path(os.environ.get("HYBRID1_HOME","/home/anmire/ai/hybrid1"));R=H/"runs";B=H/"bakes"
J=lambda p:json.load(open(p))
pc=lambda v:"-" if v is None else f"{100*v:.1f}"
def fid():
    L=["| bake | bits/group | bake GB | layer cos mean | layer cos min | layer rel-err (weighted) | code roundtrip exact |","|---|---|---|---|---|---|---|"]
    for d in sorted(B.glob("*")):
        m=d/"bake_manifest.json"
        if not m.exists():continue
        x=J(m);f=x["fidelity"]["layers"];L.append(f"| {d.name} | {x['bits']}/{x['gs']} | {x['bake_bytes']/1e9:.2f} | {f['cos_mean']:.4f} | {f['cos_min']:.4f} | {100*f['rel_err_weighted']:.1f}% | {f['codes_exact_all']} |")
    return "\n".join(L)
def meas(pat="meas_*.json"):
    L=["| config | weights GB (torch alloc) | KV GB @ctx | peak GB (torch reserved) | peak GB (sysfs delta, noisy) | prefill tok/s | decode tok/s | ctx tokens |","|---|---|---|---|---|---|---|---|"]
    for p in sorted(R.glob(pat)):
        x=J(p);L.append(f"| {p.stem[5:]} | {x['torch_alloc_after_load_GB']:.2f} | {x['kv_cache_GB_at_ctx']:.3f} | {x['torch_max_reserved_GB']:.2f} | {x['peak_vram_GB_excl_ctx']:.2f} | {x['prefill_tok_s']:.0f} | {x['decode_tok_s']:.1f} | {x['ctx_tokens']} |")
    return "\n".join(L)
def probe(pat="probe_*.json"):
    L=["| config | faithful (strict) % | no-new-content % | hedge when uncovered % | blend coherent % | chat answered % | chat coherent % | decline % | emergency-text FP % | distinct-2 | opener-3 repeat % |","|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(R.glob(pat)):
        S=J(p)["summary"];b=S["blend_probe"];c=S["chat_probe"]["all"];d=S["diversity"]
        L.append(f"| {p.stem[6:]} | {pc(b['faithful_para']['p'])} | {pc(b['no_new_content_para']['p'])} | {pc(b['hedge_uncov']['p'])} | {pc(b['coherent']['p'])} | {pc(c['answered']['p'])} | {pc(c['coherent']['p'])} | {pc(c['decline']['p'])} | {pc(c['em_fp']['p'])} | {d['distinct2_chat_probe']:.3f} | {pc(d['opener3_repeat_rate'])} |")
    return "\n".join(L)
if __name__=="__main__":print({"fid":fid,"meas":meas,"probe":probe}[sys.argv[1]](*sys.argv[2:]))
