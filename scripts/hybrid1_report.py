import os,sys,json
from pathlib import Path
H=Path(os.environ.get("HYBRID1_HOME","/home/anmire/ai/hybrid1"));R=Path(os.environ.get("HYBRID1_RUNS") or H/"runs");B=H/"bakes"
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
    L=["| config | weights GB (torch alloc) | KV GB @ctx | peak GB torch reserved | peak GB whole process (fdinfo) | prefill tok/s | decode tok/s | ctx tokens |","|---|---|---|---|---|---|---|---|"]
    for p in sorted(R.glob(pat)):
        x=J(p);L.append(f"| {p.stem[5:]} | {x['torch_alloc_after_load_GB']:.2f} | {x['kv_cache_GB_at_ctx']:.3f} | {x['torch_max_reserved_GB']:.2f} | {x.get('peak_proc_vram_GB',float('nan')):.2f} | {x['prefill_tok_s']:.0f} | {x['decode_tok_s']:.1f} | {x['ctx_tokens']} |")
    return "\n".join(L)
def probe(pat="probe_*.json"):
    L=["| config | faithful (strict) % | no-new-content % | hedge when uncovered % | blend coherent % | chat answered % | chat coherent % | decline % | emergency-text FP % | distinct-2 | opener-3 repeat % |","|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(R.glob(pat)):
        S=J(p)["summary"];b=S["blend_probe"];c=S["chat_probe"]["all"];d=S["diversity"]
        L.append(f"| {p.stem.split("_",1)[1]} | {pc(b['faithful_para']['p'])} | {pc(b['no_new_content_para']['p'])} | {pc(b['hedge_uncov']['p'])} | {pc(b['coherent']['p'])} | {pc(c['answered']['p'])} | {pc(c['coherent']['p'])} | {pc(c['decline']['p'])} | {pc(c['em_fp']['p'])} | {d['distinct2_chat_probe']:.3f} | {pc(d['opener3_repeat_rate'])} |")
    return "\n".join(L)
def e2e(*names):
    EV=Path(os.environ.get("HYBRID1_WT","/home/anmire/ai/Amni-Ai/.claude/worktrees/agent-a5bd8c5f36a3c80c7"))/"data"/"adam_t"/("ev"+"al")
    L=["| run | answered | good | fact | ctx | coherent | 911 lead | near-miss FP | decline | dup | distinct-2 | blend attempts | blend passes | blend pass rate |","|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r=J(EV/f"report_{n}_e2e_tune_dry.json");m=r["m"];it=r["suites"]["chat"]["items"];bl=sum(int((x.get("ctx_n") or {}).get("hbl",0)) for x in it);bp=sum(int((x.get("ctx_n") or {}).get("hbp",0)) for x in it)
        L.append(f"| {n} | {pc(m['chat_ans'])} | {pc(m['chat_good'])} | {pc(m['chat_fact'])} | {pc(m['chat_ctx'])} | {pc(m['chat_coh'])} | {pc(m['chat_em'])} | {pc(m['chat_nmfp'])} | {pc(m['chat_dec'])} | {pc(m['chat_dup'])} | {m['chat_d2']:.3f} | {bl} | {bp} | {pc(bp/bl) if bl else '-'} |")
    return "\n".join(L)
def rows(*names):
    g=lambda d,*k:__import__("functools").reduce(lambda x,y:(x or {}).get(y),k,d);f2=lambda v:"-" if v is None else f"{v:.2f}";A=["| config | source files (dev_ / probe_ + config + .json) | measure-phase peak GB `blender2.meas.peak_proc_vram_GB` | whole-run peak GB `blender2.proc_vram_peak_all_GB` | KV+state GB @2,560 `meas.kv_cache_GB_at_ctx` | pinned host GB | prefill tok/s | decode tok/s | ppl `blender2.ppl.ppl` | dev faithful % | dev no-new % | dev hedge % | dev blend coherent % | dev chat decline % |","|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"];P=["| config | probe8 faithful % `summary.blend_probe.faithful_para.p` | no-new % | hedge % | blend coherent % | chat answered % | chat coherent % | decline % | emergency-text FP % | distinct-2 | opener-3 repeat % | same process as the VRAM row |","|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        d=J(R/f"dev_{n}.json") if (R/f"dev_{n}.json").exists() else {};q=J(R/f"probe_{n}.json") if (R/f"probe_{n}.json").exists() else {};x=(q or d).get("blender2") or {};m=x.get("meas") or {};sd=d.get("summary") or {};sq=q.get("summary") or {}
        A.append(f"| {n} | dev_{n}.json, probe_{n}.json | {f2(m.get('peak_proc_vram_GB'))} | {f2(x.get('proc_vram_peak_all_GB'))} | {m.get('kv_cache_GB_at_ctx','-')} | {f2(m.get('host_pinned_GB'))} | {m.get('prefill_tok_s','-')} | {m.get('decode_tok_s','-')} | {g(x,'ppl','ppl') or '-'} | {pc(g(sd,'blend_probe','faithful_para','p'))} | {pc(g(sd,'blend_probe','no_new_content_para','p'))} | {pc(g(sd,'blend_probe','hedge_uncov','p'))} | {pc(g(sd,'blend_probe','coherent','p'))} | {pc(g(sd,'chat_probe','all','decline','p'))} |")
        P.append(f"| {n} | {pc(g(sq,'blend_probe','faithful_para','p'))} | {pc(g(sq,'blend_probe','no_new_content_para','p'))} | {pc(g(sq,'blend_probe','hedge_uncov','p'))} | {pc(g(sq,'blend_probe','coherent','p'))} | {pc(g(sq,'chat_probe','all','answered','p'))} | {pc(g(sq,'chat_probe','all','coherent','p'))} | {pc(g(sq,'chat_probe','all','decline','p'))} | {pc(g(sq,'chat_probe','all','em_fp','p'))} | {g(sq,'diversity','distinct2_chat_probe') or '-'} | {pc(g(sq,'diversity','opener3_repeat_rate'))} | {bool(q) and bool(d) and q.get('blender2')==d.get('blender2')} |")
    return "\n".join(A)+"\n\n"+"\n".join(P)
def equiv(*tags):
    L=["| file | config | reference | prompts token-identical to reference | logit max-abs-diff on identical prompts | max-abs-diff incl. divergence step | divergences (prompt index @ first differing step) | degenerate repeats |","|---|---|---|---|---|---|---|---|"]
    for t in tags:
        x=J(R/f"equiv_{t}.json")
        for k,v in x["cfgs"].items():L.append(f"| equiv_{t}.json | {k} | {v['ref']} | {v['n_identical_to_ref']}/{v['n_cmp']} | {v['maxabs_identical_max']} | {v['maxabs_all_max']} | {', '.join(str(d['i'])+'@'+str(d['div']) for d in v['diverged'][:8])+(' ...' if len(v['diverged'])>8 else '')} | {len(v['degenerate_repeat_prompts'])} |")
    return "\n".join(L)
def select(*names):
    g=lambda d,*k:__import__("functools").reduce(lambda x,y:(x or {}).get(y),k,d);D={}
    for n in names:
        d=J(R/f"dev_{n}.json");x=d["blender2"];S=d["summary"];D[n]={"f":100*g(S,"blend_probe","faithful_para","p"),"nn":100*g(S,"blend_probe","no_new_content_para","p"),"h":100*g(S,"blend_probe","hedge_uncov","p"),"c":100*g(S,"blend_probe","coherent","p"),"dec":100*g(S,"chat_probe","all","decline","p"),"ppl":x["ppl"]["ppl"],"vr":x["meas"]["peak_proc_vram_GB"],"va":x["proc_vram_peak_all_GB"],"tok":x["meas"]["decode_tok_s"]}
    for n,v in D.items():r=D.get(n.rsplit("_",2)[0]+"_q8_S");v["pr"]=v["ppl"]/r["ppl"] if r else None;v["gate"]=v["h"]>=80 and v["c"]>=95 and v["dec"]<=5 and v["pr"] is not None and v["pr"]<=1.15;v["sum"]=v["f"]+v["nn"]
    G=[n for n in D if D[n]["gate"]];F0=max(v["f"] for v in D.values());N0=max(v["nn"] for v in D.values());F3=max((D[n]["f"] for n in G),default=0);N3=max((D[n]["nn"] for n in G),default=0);S5=max((D[n]["sum"] for n in G),default=0)
    e0=[n for n in G if D[n]["f"]>=F0-5 and D[n]["nn"]>=N0-5];e3=[n for n in G if D[n]["f"]>=F3-5 and D[n]["nn"]>=N3-5];e5=[n for n in G if D[n]["sum"]>=S5-5]
    pick=lambda E:min(E,key=lambda n:(round(D[n]["vr"]/0.05),-D[n]["tok"])) if E else "none"
    L=["| config | dev faithful % | dev no-new % | sum | dev hedge % | dev blend coherent % | dev chat decline % | ppl / own int8 S | passes gates | eligible: plan as written | eligible: amendment 3 | eligible: amendment 5 | measure-phase peak GB | whole-run peak GB | decode tok/s |","|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    L+=[f"| {n} | {v['f']:.1f} | {v['nn']:.1f} | {v['sum']:.1f} | {v['h']:.1f} | {v['c']:.1f} | {v['dec']:.1f} | {v['pr']:.3f} | {v['gate']} | {n in e0} | {n in e3} | {n in e5} | {v['vr']:.3f} | {v['va']:.3f} | {v['tok']} |" for n,v in D.items()]
    return "\n".join(L)+f"\n\nBest dev faithful over all rows {F0:.1f}, over gate-passing rows {F3:.1f}; best no-new over all {N0:.1f}, over gate-passing {N3:.1f}; best summed score over gate-passing {S5:.1f}.\n\n- Plan as written selects: **{pick(e0)}**\n- Amendment 3 selects: **{pick(e3)}**\n- Amendment 5 selects: **{pick(e5)}**"
if __name__=="__main__":print({"fid":fid,"meas":meas,"probe":probe,"e2e":e2e,"rows":rows,"equiv":equiv,"select":select}[sys.argv[1]](*sys.argv[2:]))
