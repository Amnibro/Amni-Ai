# Checklist: TORUS-A2b (A2 relational torus re-run), v1

Branch `worktree-agent-a5667e1f0c990af2a` (reset onto `feat/ray-field-serve` 7c7c9b5), 2026-09-30. Protocol: `docs/adam_t/torus_a2b_prereg.md`. Council: `docs/guardian_councils/guardian_council_torus_a2b.md`.

## Lever grids (fixed before the lever runs)
- Bounce: sources {tan, nbr, tan+nbr}; up to 4 bounces per head; bounce weight bw in {1, 2, 4} (bounce sum divided by bounce count, times bw/4, added to the direct score x1); aggregation mean (max only if mean wins).
- Mix: 2 and 3 (pairs d, d^1; triples d, d^1, d^2 within aligned blocks of 4 dims); re-solve with everything else fixed.
- Levels: match level of h and c over 64/256/1024 cells (identity landing, neighbouring cells within a face) plus a learned per-relation coarse landing cell (64-level argmax table); per-relation log-odds LUT x lw, lw in {2, 4, 8}.
- Calibration: 16 buckets vs the A2 softmax (tau 64).

## Steps
- [x] Read CLAUDE.md, the spec, TORUS-A results, crates/adam-torus and the A2-A4 spec (18f9195).
- [x] Worktree reset onto feat/ray-field-serve 7c7c9b5.
- [x] Guardian council.
- [x] Pre-registration written and committed before any A2b data build or run.
- [x] Backups: torus_solve.py, torus_data.py, sphere_noleak.py, torus_a.rs, architecture_map.md, changelog.md -> backups/*.torus_a2b_2026-09-30.bak.
- [x] torus_data.py: A2b sampler (aux relations, degree-constrained split).
- [x] sphere_noleak.py: dataset/output paths; A2b no-leak sphere built.
- [x] torus_solve.py: data/out paths, test mode, save/load, chunked scoring, bounce, mix, lvl, calib, ECE, sphere recall from a given sphere.
- [x] Leak check (lift-only dev): max relation AUC 0.928 (P17, popularity prior), hypernym 0.573; no leak.
- [x] Dev ablation: baseline, bounce, mix, lvl, calib, plus amendment-1 per-relation popularity weight. Frozen config in the prereg.
- [ ] Final: train, dev LUTs, single test scoring.
- [ ] Recall@20 torus vs A2b no-leak sphere.
- [ ] torus_a l2pack: PTEX pages + bit-exact roundtrip.
- [ ] Results doc, architecture_map.md + changelog.md entries, deletions log, target dir deleted, processes stopped.
