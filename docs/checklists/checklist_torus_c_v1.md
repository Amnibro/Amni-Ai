# Checklist: TORUS-C (ray angle + polarity relation signature), v1

Branch `worktree-agent-abaf78a481b1ca204` (reset onto `feat/ray-field-serve` 8327342), 2026-10-06. Protocol: `docs/adam_t/torus_c_prereg.md`. Results: `docs/adam_t/torus_c_results.md`. Council: `docs/guardian_councils/guardian_council_torus_c.md`.

- [x] Read CLAUDE.md, proposal, TORUS-A/A2b/B prereg and results, torus_solve.py, torus_b.py, torus_a.rs interface.
- [x] Worktree reset onto feat/ray-field-serve.
- [x] Guardian council.
- [x] Pre-registration committed before any C solve or scoring (de080b8).
- [x] Backups: torus_solve.py, torus_b.py, architecture_map.md, changelog.md -> backups/*.torus_c_2026-10-06.bak (and *_end.bak before the final entries).
- [x] New split (fresh seed 6029) + rebuilt no-leak sphere -> data/torus/c/.
- [x] torus_solve.py: --psub, --seedf/--seedfix, --rnoe, --rolen, --lamr/--rhor, --ro ang, group-wise edge sets, relation-subspace E-step; B config reproduces bit-exactly with --psub 0 (checked: X, TF, assign equal).
- [x] torus_c_seed.py: label-free structural role seeds.
- [x] torus_b.py: --dir/--dset, relation dims, angle-code clusterers, residual NMI, polarity on antisymmetric relations, strata, seeds-only and coarse-address clusterers; B regression on the old split reproduces resb numbers exactly.
- [x] Dev controls (shuffled, random, metadata, seeds-only, B config R, oracle) -> Amendment 1 committed before the final solves.
- [x] Dev ablation: split subspaces, seeds, angle code, polarity, tori, sphere coarse address, composed rays, K, P, lam/rho.
- [x] Final solves for seeds 17 and 23 (train only); test scored once per seed (lock files); B config R scored on the same split.
- [x] 10 largest clusters, storage, l2pack bit-exact roundtrip (both seeds).
- [x] Results doc, architecture_map.md + changelog.md entries (shared tree, backed up first), deletions log, own cargo target dir and dev models deleted, no processes left running, own import/undefined-name check clean (pyflakes not installed; AST check used: no undefined names, no comments, no blank lines).
- [ ] User confirmation that it works (CTO to merge).
