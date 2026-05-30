"""harness_config — the FROZEN benchmark contract (checklist B2). Changing any value here = a new harness version;
bump HARNESS_VERSION so old results stay comparable only within their own version. Locks decoding, per-bench
token budgets, prompt-template version, and fixed item counts so iteration N and N-1 see identical conditions.
Gold answers are used ONLY for scoring downstream — never injected into generation."""
HARNESS_VERSION='h1.0.0'
PROMPT_TEMPLATE_VERSION='p1'
DECODING={'canonical':{'do_sample':False},'self_consistency':{'do_sample':True,'k':5}}
PER_BENCH={
 'mmlu_pro':{'max_tokens':512,'kind':'mcq','limit':200},
 'gpqa_diamond':{'max_tokens':512,'kind':'mcq','limit':198},
 'math500':{'max_tokens':1024,'kind':'boxed','limit':200},
 'humanevalplus':{'max_tokens':768,'kind':'code','limit':164},
 'mbppplus':{'max_tokens':768,'kind':'code','limit':200},
 'gsm8k':{'max_tokens':640,'kind':'numeric','limit':200},
 'arc':{'max_tokens':8,'kind':'mcq','limit':200},
}
CANONICAL_BENCHES=['mmlu_pro','gpqa_diamond','math500','humanevalplus']
CONTINUITY_BENCHES=['gsm8k','arc']
def stamp(bake_version='',server_version='',date='',ram_free_gb=None,vram_free_gb=None,extra=None):
    s={'harness_version':HARNESS_VERSION,'prompt_template_version':PROMPT_TEMPLATE_VERSION,'bake_version':bake_version,'server_version':server_version,'date':date,'ram_free_gb':ram_free_gb,'vram_free_gb':vram_free_gb,'decoding':'canonical_greedy'}
    if extra:s.update(extra)
    return s
def describe():
    lines=[f'Harness {HARNESS_VERSION} (prompts {PROMPT_TEMPLATE_VERSION}) — canonical decoding = greedy (do_sample=False)']
    lines.append('Canonical benches: '+', '.join(CANONICAL_BENCHES))
    lines.append('Continuity (saturating, context-only): '+', '.join(CONTINUITY_BENCHES))
    for b,c in PER_BENCH.items():lines.append(f'  {b}: kind={c["kind"]} max_tokens={c["max_tokens"]} limit={c["limit"]}')
    return '\n'.join(lines)
