# Checklist: TORUS-B (unlabeled-ray relation discovery), v1

Branch `worktree-agent-ada5a1c435dedcf1f` (reset onto `feat/ray-field-serve` 521a607), 2026-10-05. Protocol: `docs/adam_t/torus_b_prereg.md`. Council: `docs/guardian_councils/guardian_council_torus_b.md`.

- [x] Read CLAUDE.md, the proposal, TORUS-A/A2b results and prereg, and the merged crate and data.
- [x] Worktree reset onto feat/ray-field-serve.
- [x] Guardian council.
- [x] Pre-registration written and committed before any run.
- [ ] Backups: torus_solve.py, torus_a.rs (unchanged expected), architecture_map.md, changelog.md -> backups/*.torus_b_2026-10-05.bak.
- [ ] torus_solve.py: --unlab K hard EM, --orient, --ge, --tan, metrics (nmi, purity), parse_known_args.
- [ ] torus_b.py: clusterers M1/M2/M3, density LUT, matched negatives, controls, B1 metrics, composition, direction, A2b comparison, export.
- [ ] Model-free controls on dev (metadata NMI, popularity AUC, lift-only) -> Amendment 1 (numbers) committed before the dev ablation of the model.
- [ ] Dev ablation (K, ge, tan, D, clusterer, tori, warm-up, orient control).
- [ ] Frozen config committed; test scored once per seed (17, 23).
- [ ] A2b labeled comparison; novel clusters; storage; l2pack bit-exact roundtrip.
- [ ] Results doc, architecture_map.md + changelog.md entries (shared tree, backed up first), deletions log, target dir deleted, processes stopped, pyflakes clean.
