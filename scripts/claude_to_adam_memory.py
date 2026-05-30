"""claude_to_adam_memory — push Claude's session context into Adam's PTEX-backed memory via the running server.
Two write surfaces:
  - /teach     : structured Q/A facts (feeds Adam's tier-1 LUT for natural-question recall)
  - /memory/notes : free-form session log entries (searchable, taggable)
Usage:
  python scripts/claude_to_adam_memory.py --note 'Fast-path patch deployed; tok/s 17.5→27 (+57%)' --tag perf --tag session-2026-05-27
  python scripts/claude_to_adam_memory.py --teach 'What is Adam cossim rule?' 'cossim<1 = gibberish; never modify the base model bake.'
  python scripts/claude_to_adam_memory.py --bootstrap   # one-shot push of today's critical context"""
import sys,json,argparse,urllib.request
def _post(url,payload,timeout=30):
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'content-type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())
def teach(url,q,a):return _post(url+'/teach',{'question':q,'answer':a})
def note(url,text,tags=None,session_id=''):return _post(url+'/memory/notes',{'text':text,'tags':tags or [],'session_id':session_id})
_BOOTSTRAP_FACTS=[
    ('What is the cossim rule for Adam?','cossim<1 = gibberish, period. Never modify the base model bake (no purge, filter, re-bake, distill, sub-GF17 quantization). All perf work must be BIT-EXACT operation-by-operation acceleration (LUTs/precomputes/restructured-math producing identical outputs).'),
    ('Should Claude purge bad training data and re-bake Adam?','No. That breaks the cossim=1 axiom. Lossless ops only — LUT activation/rsqrt/softmax-exp, skip bit-exact zeros, restructure GEMV TMU throughput. Base model stays untouched.'),
    ('What is the streaming_linear fast-path patch?','Once _hip_tex is bound, forward() skips schedule_prefetch loop + get_full(weight_key) + note_use — jumps straight to the HIP texture kernel. Backup at backups/streaming_linear.py.v0001_pre_fastpath.bak. Deployed 2026-05-27, lifted tok/s 17.5 -> 27 (+57%), warmup 195s -> 103s.'),
    ('What is Adam ARC-Challenge accuracy?','82.0% on n=50 ARC-Challenge (single-shot, CoT, gemma4_e2b_it_gf17, RX 7800 XT, ~3GB VRAM). Within ~1pp of llama-3.1-8b 83.4, +26pp over gemma-2-2b 55.7.'),
    ('What is Adam GSM8K accuracy?','84.0% on n=50 GSM8K-CoT (single-shot K=1, post-CoT-suppression fix commit 26c9a03, max_tokens=4096). Of 8 failures, 3 were empty outputs (budget) and 5 were reasoning errors. Pre-fix run was 28% — the lift was harness CoT enablement, not model change.'),
    ('What is Claude session memory directive?','Claude must store session context in both markdown memory files (C:\\Users\\antho\\.claude\\projects\\... memory/) AND Adam PTEX via /teach + /memory/notes. Dogfood the architecture; keep context pure across reboots; let Adam see Claude\'s session notes via the same nonce-address LUT.'),
    ('What is Anthony PTEX perf vision?','Don\'t ceiling-think Adam against matmul/GEMV. PTEX can precompute whole-step transforms — per pixel 256^4 states, per atlas 268M coords. Per-token cost can fall to ~40 FLOPs; with 27 TFLOPs that\'s 675B tok/s theoretical (orders of magnitude above current 27 tok/s). Never quote a ceiling without qualifying "for known-approach kernel".'),
]
_BOOTSTRAP_NOTES=[
    ('claude-2026-05-27: deployed streaming_linear fast-path; tok/s 17.5 -> 27 (+57%); warmup 195 -> 103s; eval_reports/tok_per_sec_probe.json holds the numbers',['perf','session-2026-05-27','deployed']),
    ('claude-2026-05-27: ARC-C n=50 82.0%, GSM8K-CoT n=50 84.0% (K=1, single-shot) — eval_reports/adam_real_run_n50.json + adam_gsm8k_cot.json',['benchmark','session-2026-05-27']),
    ('claude-2026-05-27: modern_bench.py + sample.json scaffolded (MMLU-Pro/MATH-500/HumanEval+/GPQA-Diamond) — offline 5/5 PASS; not yet live-run',['benchmark','queued','session-2026-05-27']),
    ('claude-2026-05-27: bench_suites.py --self-consistency K flag wired (hits /chat do_sample=True K times, majority-vote on numeric); not yet live-run',['benchmark','queued','session-2026-05-27']),
    ('claude-2026-05-27: HARD RULE saved (feedback_amni_ai_lossless_baseline_only) — cossim<1=gibberish, never touch bake, lossless ops only',['rule','session-2026-05-27']),
    ('claude-2026-05-27: PTEX-precompute vision saved (feedback_amni_ai_think_outside_matmul + project_amni_ai_quality_tmap reframed) — next-pass candidates: activation LUT, rsqrt LUT, softmax exp LUT, sparse-zero skip, GEMV TMU restructuring',['perf','queued','session-2026-05-27']),
    ('claude-2026-05-27: directive — Claude should also push session context to Adam PTEX via /teach + /memory/notes (this script); dogfood the architecture',['rule','session-2026-05-27']),
]
def bootstrap(url):
    teach_out=[];note_out=[]
    for q,a in _BOOTSTRAP_FACTS:
        try:r=teach(url,q,a);teach_out.append({'q':q[:60],'r':r})
        except Exception as e:teach_out.append({'q':q[:60],'error':str(e)[:120]})
    for t,tags in _BOOTSTRAP_NOTES:
        try:r=note(url,t,tags=tags,session_id='claude-2026-05-27');note_out.append({'text':t[:60],'r':r})
        except Exception as e:note_out.append({'text':t[:60],'error':str(e)[:120]})
    return {'teach':teach_out,'notes':note_out}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url',default='http://127.0.0.1:7700')
    ap.add_argument('--bootstrap',action='store_true')
    ap.add_argument('--teach',nargs=2,metavar=('Q','A'))
    ap.add_argument('--note',default=None)
    ap.add_argument('--tag',action='append',default=[])
    ap.add_argument('--session-id',default='')
    a=ap.parse_args();url=a.url.rstrip('/')
    if a.bootstrap:print(json.dumps(bootstrap(url),indent=2,default=str));return
    if a.teach:print(json.dumps(teach(url,a.teach[0],a.teach[1]),indent=2,default=str));return
    if a.note:print(json.dumps(note(url,a.note,tags=a.tag,session_id=a.session_id),indent=2,default=str));return
    ap.print_help()
if __name__=='__main__':main()
