# Checklist: TORUS-B (unlabeled-ray relation discovery), v1

Branch `worktree-agent-ada5a1c435dedcf1f` (reset onto `feat/ray-field-serve` 521a607), 2026-10-05. Protocol: `docs/adam_t/torus_b_prereg.md` (Amendment 1). Results: `docs/adam_t/torus_b_results.md`. Council: `docs/guardian_councils/guardian_council_torus_b.md`.

- [x] Read CLAUDE.md, the proposal, TORUS-A/A2b results and prereg, and the merged crate and data.
- [x] Worktree reset onto feat/ray-field-serve.
- [x] Guardian council.
- [x] Pre-registration written and committed before any run (068cb46).
- [x] Backups: torus_solve.py, architecture_map.md, changelog.md -> backups/*.torus_b_2026-10-05.bak (torus_a.rs unchanged, not backed up).
- [x] torus_solve.py: --unlab K hard EM, --warm reseed, --sib, --orient, --ge, --tan, --bal, --oracle, nmi/kpp, parse_known_args.
- [x] torus_b.py: clusterers M1/M2/M3, density LUT, matched negatives, controls, B1 metrics, composition, direction and inverse, A2b comparison, export, test lock.
- [x] Dev controls and oracle diagnostics -> Amendment 1 (numbers, frozen config R, G4 on hop-1-held-out) committed before the final solves and any test scoring.
- [x] Dev ablation (K, ge, tan, D, clusterer, tori, warm-up, sibling edges, lam, rho, balance, LUT transforms, orientation control, oracle).
- [x] Final solves for seeds 17 and 23 (train only); test scored once per seed (lock files).
- [x] A2b labeled comparison, 10 novel clusters, storage, l2pack bit-exact roundtrip (both seeds).
- [x] Results doc, architecture_map.md + changelog.md entries (shared tree, backed up first), deletions log, own cargo target dir and dev models deleted, no processes left running, own import/undefined-name check clean (pyflakes is not installed here; `scratchpad/lint.py` AST check used).
- [ ] User confirmation that it works (CTO to merge).
