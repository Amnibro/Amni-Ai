"""run_security_suite — one-command regression over every hardening step from the security-hardening loop (S1-S35).
Each test module uses plain assert-functions (no pytest dep), so this imports each, runs every `test_*`, and reports a
pass/fail summary. Exit code is non-zero if anything fails. Run: `python tests/run_security_suite.py` (set PYTHONUTF8=1).
The mapping below doubles as the control inventory — see docs/SECURITY.md for the threat model behind each."""
import sys,importlib,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
SUITE=[
 ('S1  malicious-code danger-scan + sandbox isolation','tests.test_code_safety_v6_10_140'),
 ('S2  detect_personal soft-PII hardening','tests.test_detect_personal_hardening_v6_10_141'),
 ('S3  publish/ingest PII gates','tests.test_publish_ingest_pii_v6_10_142'),
 ('S4  manual /teach PII gate','tests.test_teach_gate_v6_10_143'),
 ('S5  sandbox resource caps (output/cpu/mem)','tests.test_sandbox_resource_caps_v6_10_144'),
 ('S6  crawl content sanitize + source allowlist','tests.test_ingest_sanitize_v6_10_145'),
 ('S7  lesson-pollution audit + federation audit log','tests.test_lesson_audit_v6_10_146'),
 ('S8  quarantine polluted lessons','tests.test_quarantine_v6_10_147'),
 ('S9  federation PULL receive-side verify','tests.test_federation_pull_verify_v6_10_148'),
 ('S10 rate-limit public endpoints','tests.test_rate_limit_v6_10_149'),
 ('S11 secrets/PII egress scrubber','tests.test_egress_scrub_v6_10_150'),
 ('S12 deepened AST danger-scan','tests.test_danger_scan_deep_v6_10_151'),
 ('S13 periodic auto-audit/quarantine','tests.test_periodic_audit_v6_10_153'),
 ('S14 egress scrub on HTTPException/SSE/dict','tests.test_egress_endpoints_v6_10_154'),
 ('S15 path-traversal prefix-bypass fix (file skills)','tests.test_path_traversal_guard_v6_10_152'),
 ('S16 chat output byte cap','tests.test_output_cap_v6_10_155'),
 ('S17 structural Luhn/SSN PII detectors','tests.test_structural_pii_v6_10_156'),
 ('S18 sandbox network-egress block + run_python reroute','tests.test_sandbox_no_network_v6_10_157'),
 ('S19 shell-injection (validation/exec mismatch) fix','tests.test_shell_injection_v6_10_158'),
 ('S20 python/pip shell-allowlist exec lockdown','tests.test_shell_interpreter_exec_v6_10_159'),
 ('S21 SSRF guard (scheme + private-IP block)','tests.test_ssrf_guard_v6_10_160'),
 ('S22 daemon fetch SSRF coverage','tests.test_daemon_fetch_ssrf_v6_10_161'),
 ('S23 DNS-rebinding pin + redirect re-check','tests.test_ssrf_pin_redirect_v6_10_162'),
 ('S25 federation_import receive-side verify','tests.test_federation_import_verify_v6_10_163'),
 ('S26 publish send-side dangerous-code/injection block','tests.test_publish_sendside_safety_v6_10_164'),
 ('S27 federation import DoS cap','tests.test_federation_import_dos_v6_10_165'),
 ('S28 input size caps (/chat,/teach)','tests.test_input_size_caps_v6_10_166'),
 ('S29 session-endpoint path-traversal fix','tests.test_session_path_traversal_v6_10_167'),
 ('S30 ingest_pdf arbitrary-file-read gate','tests.test_ingest_pdf_gate_v6_10_168'),
 ('S31 vision/voice upload size caps','tests.test_upload_size_caps_v6_10_169'),
 ('S32 /v1 + /complete input caps','tests.test_v1_completion_caps_v6_10_170'),
 ('S33 on-disk audit-log scrub','tests.test_log_scrub_v6_10_171'),
 ('S34 reasoning-stream secret scrub','tests.test_reasoning_secret_scrub_v6_10_172'),
 ('S35 scan-skill teach-loop PII scrub','tests.test_scan_pii_scrub_v6_10_173'),
 ('S37 secret-file deny-list (file skills)','tests.test_secret_file_deny_v6_10_174'),
 ('S38 MCP external-surface input cap','tests.test_mcp_input_cap_v6_10_175'),
 ('S39 MCP tools/call rate-limit','tests.test_mcp_rate_limit_v6_10_176'),
 ('S40 /v1 + ollama-compat rate-limit + cap','tests.test_compat_rate_limit_v6_10_177'),
 ('S41 embed-batch cap (/api/embed)','tests.test_embed_batch_cap_v6_10_178'),
 ('S42 trace dump_raws arbitrary-write fix','tests.test_trace_dump_path_v6_10_179'),
 ('S43 path-param sweep (vision read + out_path gate)','tests.test_path_param_sweep_v6_10_180'),
 ('S44 web-skill result sanitize','tests.test_web_result_sanitize_v6_10_181'),
 ('S45 Asimov-law source write-protect','tests.test_law_write_protect_v6_10_182'),
 ('S46 security-core module write-protect','tests.test_security_core_protect_v6_10_183'),
 ('S47 source-integrity tamper detection','tests.test_source_integrity_v6_10_184'),
 ('S48 PTEX load-time input validation','tests.test_ptex_load_filter_v6_10_185'),
]
def run():
    total_pass=0;total_fail=0;failures=[]
    for label,mod in SUITE:
        try:m=importlib.import_module(mod)
        except Exception as e:
            print(f'[ERR ] {label}: import failed: {type(e).__name__}: {e}');total_fail+=1;failures.append((label,'import',e));continue
        fns=[f for f in dir(m) if f.startswith('test_')]
        p=0;f=0
        for fn in fns:
            try:getattr(m,fn)();p+=1
            except Exception as e:f+=1;failures.append((label,fn,e))
        total_pass+=p;total_fail+=f
        print(f'[{"PASS" if f==0 else "FAIL"}] {label:<52} {p:>2}/{len(fns)}')
    print('-'*72)
    print(f'TOTAL: {total_pass} passed, {total_fail} failed across {len(SUITE)} hardening steps')
    if failures:
        print('\nFAILURES:')
        for label,fn,e in failures:print(f'  - {label} :: {fn}: {type(e).__name__}: {e}')
    return 0 if total_fail==0 else 1
if __name__=='__main__':sys.exit(run())
