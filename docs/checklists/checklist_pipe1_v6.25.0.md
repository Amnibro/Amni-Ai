# Checklist PIPE-1 (v6.25.0)

- [x] Read CLAUDE.md, harness.md (v3/v4), STORE-2 prereg/results/code, TORUS-A A5 results, user_toroid.md (encryption), blender_atex.py, CTO memory notes
- [x] Worktree reset onto feat/ray-field-serve (94c6462)
- [x] Pre-registration `docs/adam_t/pipe1_prereg.md` committed before any eval item or measurement
- [x] Guardian council `docs/guardian_councils/guardian_council_pipe1.md`
- [x] Backups to `backups/pipe1/` before editing (adam_t_harness, adam_t_client, blender_atex, user_memory, store1 serve, changelog, architecture_map)
- [x] `amni/compute/gf17_overlay.py`: A5-style GF(17) solved overlay (learn/forget/compact/rebuild, Reffelt pixel pages)
- [x] `amni/serve/pipe1.py`: strength map, draft templates, overlay-aware STORE-2 LUT, correction detection/processing, quarantine, admin promote, undo
- [x] Harness hooks (flag `AMNI_PIPE1`): Draft material, s-aware `blend()`/`gate()`, retrieval only on missing/low-confidence drafts, corrections first, chat refine
- [x] Decoding plumbing: `Proto.text`, `AdamT.gen`, `Blender.gen`, `Atex.generate` (temperature, top_k, top_p, seed)
- [x] `user_memory.forget` gains an optional clock (undo of one correction)
- [x] Unit tests `tests/test_pipe1.py`
- [x] Eval set authored by an isolated agent, decontaminated, annotated, split, sha256-locked, committed
- [x] Dev strength sweep (s = 0..1) + v4 baseline + no-Task ablation; per-route s chosen by the pre-registered rule
- [x] Code frozen; test scored once
- [x] Store and harness suites in the main tree environment
- [x] Results doc `docs/adam_t/pipe1_results.md`, changelog entry, architecture_map entry (shared tree, after backup)
- [x] All processes stopped, scratch cleaned, deletions logged (done after the final commit; see report)
- Gate outcome: 4/8 on test (G1, G4, G7, G8 pass; G2, G3, G5, G6 fail). Not claimed complete: awaiting CTO review and the GPU-blender re-run.
