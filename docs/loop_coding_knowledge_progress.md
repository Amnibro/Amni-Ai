# Loop: Adam coding-knowledge self-learning — progress

Autonomous /loop building Adam into a self-learning coding brain. Each wake: read this, do the next chunk, update this, schedule next. Server is kept DOWN during builds (machine is RAM-sensitive); crawl runs happen only when explicitly launched lean (`AMNI_ROUTED_LESSONS=1 AMNI_NO_DAEMON=1`).

## Foundations already in place (pre-loop)
- Routed map-PTEX lesson store (`routed_lessons.py`) — partitioned by (language,topic), incremental fit, cold-pack zero-load. Flag `AMNI_ROUTED_LESSONS=1`.
- GPU queue + embedders pinned CPU (`gpu_queue.py`) — no HIP race.
- Layered self-debug (`self_debug.py`) — AST lint + adversarial probe + real subprocess run. Wired into the serve coding loop.
- Programming bootstrap crawler (`programming_seeds.py` + `learning_daemon.run_programming_bootstrap`) — on-demand, controlled.

## Plan (iterations)
- [x] **I1 — Expand crawl coverage**: every major language, canonical reference sources (direct URLs: official docs, awesome-lists), HuggingFace code datasets, direct-URL ingest path, `bootstrap_all`.
- [ ] **I2 — Lesson inference**: error A ~ error B → apply B's fix to A (similarity over the error/lesson store).
- [ ] **I3 — Pollution/quality self-check**: periodically lint/compile-check stored code lessons, dedup, quarantine bad ones.
- [ ] **I4 — Learn-as-it-codes loop**: on every coding task, record the (task, mistake, fix) → teach; on retry recall the fix.
- [ ] **I5 — Federation**: share/pull routed coding packs as federated PTEX (per-language banks).
- [ ] **I6 — Engineering practices + prompt-following**: enforce a debug/lint/test/review checklist + style/spec adherence in the coding cycle.
- [ ] **I7 — HuggingFace code-dataset direct ingestion** (the_stack/codeparrot samples by license).

## Log
- 2026-05-30 I1: started.
- 2026-05-30 I1: DONE. `programming_seeds.py` → 30 languages, 315 search topics, 20 canonical direct sources (TheAlgorithms MIT repos + awesome-lists, license-tagged), 5 HF code datasets. Added `learning_daemon._ingest_one_url` (direct canonical-URL ingest) + `run_programming_bootstrap(max_topics,max_sources)` now crawls canonical sources first, then topics — controlled (bounded, yields to user, routed-store). Tests: `test_programming_bootstrap_v6_10_139.py` (8). 24 green. NEXT: I2 lesson-inference (error A~B → analogous fix).
