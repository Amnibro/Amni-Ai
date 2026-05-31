# Amni-Ai / Adam — Security Posture

Adam is a self-learning, web-crawling, tool-using AI that runs locally and federates lessons to peers. That combination
means three things must hold: **personal data must never enter the lesson PTEX or leave the box**, **untrusted code
(generated or crawled) must never run unsandboxed**, and **untrusted content (crawled, federated, scanned) must never
poison the lesson store or propagate onward**. This document inventories the controls that enforce those properties.

Run the full regression in one command:

```
PYTHONUTF8=1 python tests/run_security_suite.py
```

(34 hardening steps, 194 assert-style tests, exit non-zero on any failure. No pytest dependency.)

## Threat model

| Threat | Vector | Primary controls |
|---|---|---|
| **PII leak** | personal data reaching the lesson PTEX (federation-eligible) or any response | record-time local-vs-global routing, `detect_personal` (regex + structural Luhn/SSN), `scrub_pii`, `is_publishable`, egress scrub, scrub-at-ingest on all three ingest paths |
| **Malicious code run** | Adam runs generated/crawled code in its debug/test cycle, or via `run_python`/`shell` | AST `danger_scan` (refuse HIGH), isolated sandbox (`-I -B`, stripped env, temp cwd), resource rlimits + output cap, network-egress block, shell allowlist + metachar reject + interpreter lockdown |
| **Store poisoning** | crawled/federated/scanned content injecting prompt-hijacks, dangerous code, or markup into lessons | `sanitize_ingest`, `verify_incoming`, `audit_lessons` + periodic auto-quarantine, send-side `is_publishable` block |
| **SSRF / file read** | poisoned URL → internal services or `file://`; path params escaping data dirs | `ssrf_check` (scheme + private-IP) + IP-pin + redirect re-check; workdir containment on all file/path skills + session/PDF endpoints |
| **DoS** | unbounded input/output/import exhausting memory/CPU | input size caps (all text/upload endpoints), output byte cap, federation-import entry/size caps, rate limiting |

## Controls by area

### PII boundary (leak prevention)
- **Routing** — `conversation_atlas.record()` routes `is_personal` turns to a local-only slot; `federation_pull()` reads only the global slot. Personal data is structurally excluded from anything federation-eligible.
- **Detection** — `conversation.detect_personal`: markers + names + addresses + soft-PII (DOB/handles/passport/etc.) **plus structural** Luhn-valid credit cards and checksum-valid SSNs (phrasing-independent; kills false positives that would drop legit lessons). [S2, S17]
- **Scrubbing** — `federated.scrub_pii` (emails/phones/keys/paths/UUID + structural cc/ssn); `pii_egress.scrub` is the choke-point for web/crawl/news egress (patterns + the user's own PersonalAtlas tokens). [S3, S11]
- **Ingest parity** — crawl (`_ingest_one_url`), federation (`federation_import`), and the `scan` skill all run `scrub_pii` + `sanitize_ingest` at intake, so raw secrets never reach the local store. [S3, S6, S25, S35]
- **Egress** — `scrub_egress` on uncaught errors, `HTTPException` details, streamed SSE error events, returned error dicts, `/health`/`/stats` workdir; on-disk audit logs (shell history, skill calls) scrubbed before write; `scrub_secrets` (keys+homedir, answer-safe) on the live reasoning stream. [S11, S14, S33, S34]
- **Write gate** — manual `/teach` rejects `detect_personal` content (the one ungated path into the PTEX). [S4]

### Malicious code-run prevention
- **Static** — `code_safety.danger_scan` (AST): dangerous imports, os/subprocess/socket/shutil calls, `eval`/`exec`/`compile`/`__import__`, file-write, getattr-obfuscation, `__subclasses__`/`__bases__` escapes, dynamic getattr/setattr on powerful modules, dangerous decorators. Refuses HIGH before any run. [S1, S12]
- **Sandbox** — `run_in_sandbox` → `run_capped`: isolated `python -I -B`, stripped env, throwaway temp cwd, POSIX rlimits (AS/DATA/CPU/FSIZE/NPROC=0), output-cap kill, and a prepended shim that monkeypatches `socket`/`_socket` to block all network egress. The `run_python` skill delegates here (was raw `subprocess.run` with full env). [S1, S5, S18]
- **Shell** — `_gate_shell` allowlist + reject shell metacharacters (`; & | < > $ \` %VAR%`) that defeated the token-validated/`shell=True`-executed mismatch; `python`/`pip` restricted to read-only invocations (no `-c`/`-m`/`pip install`); skill re-validates internally. [S19, S20]
- **File access** — every path-taking skill (`file_read`/`write`, `code_edit`, `scan`, `format_code`, `symbols`, `ingest_pdf`) + the session endpoints enforce resolved workdir containment (`is_relative_to`, not prefix-`startswith`). Fixed real arbitrary-read holes in the file-skill prefix-bypass, session-`sid` traversal, and ungated `ingest_pdf`. [S15, S29, S30]

### Federation (bidirectional + content)
- **Receive** — `verify_incoming` gates every pulled/imported lesson (scrub → reject sensitive-PII → reject injection/markup → reject dangerous code), on BOTH `pull_lessons` and `coding_ledger.federation_import`; import is bounded (max entries + per-entry size, dropped before the AST scan). [S9, S25, S27]
- **Send** — `is_publishable` blocks PII/personal, script tags, AND dangerous-code/injection (symmetric with receive), on both flat + atlas publish paths. Every publish/pull is provenance-logged (counts only, never raw). [S7, S26]
- **Self-heal** — the daemon periodically re-audits the lesson store and auto-quarantines anything polluted. [S8, S13]

### Network (SSRF)
- `code_safety.ssrf_check` / `safe_urlopen`: http/https only (`file://` refused), resolve-and-block private/loopback/link-local/reserved/metadata IPs, **pin** the resolved IP for the connection (no DNS-rebinding) keeping SNI/cert via the real hostname, and follow redirects **manually** re-checking every hop. Every outbound fetch (crawl, ingest, federation pull, DDG search) routes through it. [S21, S22, S23, S24]

### Integrity / tamper resistance
- **Weights** — `learning/integrity.py` records + verifies sha256 of every asimov/foundation/ascension `.gf17` tensor (fail-fast `IntegrityError` on mismatch). **Axiom text** — `a1/asimov.py` `_AXIOM_INTEGRITY` sha256-checks at layer init.
- **Guardrail source** — the Asimov law source (`inference/asimov.py`, `a1/asimov.py`) and the integrity checker (`learning/integrity.py`) are **write-protected** against `code_edit`/`file_write`/`code_diff` (so the self-improvement loop or an MCP client can't rewrite the laws or disable the checker); readable but not writable, override `AMNI_ALLOW_LAW_EDIT=1`. [S45]

### Availability (DoS)
- Input size caps (HTTP 413) on `/chat`, `/chat/stream`, `/teach`, `/complete`, `/v1/chat/completions` (sums across messages), and base64/text caps on vision/voice. Output byte cap on `/chat/stream`. Per-key sliding-window rate limit on `/chat` + `/teach`. Federation-import entry/size caps. [S10, S16, S28, S31, S32, S27]

## Notable real vulnerabilities found & fixed (not just hypothetical)
- **Path-traversal arbitrary file read** — file-skill prefix-bypass (S15), session-endpoint `sid` (`../../etc/passwd`) read/delete (S29), ungated `ingest_pdf` local-path read (S30), ungated `vision` image-path read (S43).
- **Arbitrary file WRITE** — `/admin/trace/dump_raws` wrote a client-supplied `path` with no containment → overwrite any file on disk (S42); fixed with root-containment + `.npz`/`.npy` suffix restriction.
- **Shell command injection** — gate validated a `shlex`-tokenized view but executed the raw string with `shell=True` (S19); `python -c`/`pip install` unsandboxed-exec bypass (S20).
- **SSRF** — crawler had no guard against `169.254.169.254`/loopback/`file://`, plus redirect-SSRF and DNS-rebinding windows (S21–S24).
- **Secret/PII persistence** — full `os.environ` leaked into the `run_python` child (S18); raw command output + skill args written verbatim to on-disk logs (S33).

## Tuning (env vars)
`AMNI_SANDBOX_MAX_OUTPUT`, `AMNI_SANDBOX_MEM_MB`, `AMNI_SANDBOX_NO_NET`, `AMNI_CRAWL_TRUSTED_ONLY`, `AMNI_RATE_CHAT`,
`AMNI_RATE_TEACH`, `AMNI_SCRUB_EGRESS`, `AMNI_MAX_OUTPUT_BYTES`, `AMNI_MAX_INPUT_CHARS`, `AMNI_MAX_UPLOAD_BYTES`,
`AMNI_FED_MAX_ENTRIES`, `AMNI_FED_MAX_LESSON_BYTES`, `AMNI_ALLOW_PRIVATE_FETCH`, `AMNI_SHELL_ALLOW_EXEC`,
`AMNI_SANDBOX_NO_NET`, `security_audit_period_s` (daemon config).

_Defaults are safe; overrides loosen for trusted local-dev use. See `docs/loop_security_progress.md` for the per-step
log (S1–S35) including verification details and the threat each control closes._
