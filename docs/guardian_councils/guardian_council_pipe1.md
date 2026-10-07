# Guardian council: PIPE-1 (2026-10-06)

Question: how to implement the CEO pipeline (answer LUT -> retrieval -> context -> img2img-style refine -> corrections into the knowledge store) on top of the v4 harness and STORE-2 without breaking emergency-first, grounding or the paradigm.

## Proposals
- **Architect**: no new orchestrator beside the harness. The harness already runs LUT sources before passages; PIPE-1 is a set of flag-guarded hooks: `Draft:` material assembly, a strength field on the material object consumed by the existing `blend()`/`gate()`, and an overlay-aware STORE-2 call inside `fill_s1`. One new serve module (`pipe1.py`) for the strength map, corrections and the overlay glue; one compute module for the overlay.
- **Sentinel**: facts never change at any s: must-keep terms, no new digits and no new capitalised names are hard gate terms at every strength, and the s = 0 template is the fallback. Corrections are refused when the turn or the previous turn is an emergency, when the answer was an emergency template, or when the fact is safety-relevant. Global promotion only by reference verification or admin; poisoning stays personal and visibly hedged.
- **Scholar**: pre-register the s formula, the per-route choice rule and every metric; score fact preservation and grounding with scorers that do not reuse the gate code; isolate the eval set author; compare against the v4 baseline on the same items.
- **Engineer**: the overlay is an A5-style GF(17) ribbon in numpy (0/1 band, banded elimination over GF(17), seeded back-substitution), cells packed four digits per Reffelt pixel, the whole page sealed with `at_rest.Vault`. Global overlay sharded with a fixed shard count so compaction re-solves only touched shards and equals a rebuild byte for byte. Decoding parameters pass through the existing JSON-lines `gen` op (ADAM-T already reads temperature/top_k/seed).
- **Pathfinder**: store coordinates, not values: verified corrections map a question nonce to a (QID, PID, value index) coordinate of the reference record, so the global overlay never holds user text; only personal overlays and the quarantine hold the user's literal value, both encrypted and local.

## Vote
Unanimous on hooks-in-the-harness over a parallel orchestrator, hard must-keep at every s, coordinates in the global overlay and encrypted personal overlays. Split (Engineer vs Pathfinder) on the overlay key for unresolvable questions: question-nonce keys (normalized content lemmas) chosen, with the coordinate key used whenever STORE-2 resolves the question.
