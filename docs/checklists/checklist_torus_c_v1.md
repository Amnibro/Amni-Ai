# Checklist: TORUS-C (ray angle + polarity relation signature), v1

Branch `worktree-agent-abaf78a481b1ca204` (reset onto `feat/ray-field-serve` 8327342), 2026-10-06. Protocol: `docs/adam_t/torus_c_prereg.md`. Results: `docs/adam_t/torus_c_results.md`. Council: `docs/guardian_councils/guardian_council_torus_c.md`.

- [x] Read CLAUDE.md, proposal, TORUS-A/A2b/B prereg and results, torus_solve.py, torus_b.py, torus_a.rs interface.
- [x] Worktree reset onto feat/ray-field-serve.
- [x] Guardian council.
- [ ] Pre-registration committed before any C solve or scoring.
- [ ] Backups: torus_solve.py, torus_b.py, architecture_map.md, changelog.md -> backups/*.torus_c_2026-10-06.bak.
- [ ] New split (fresh seed) + rebuilt no-leak sphere -> data/torus/c/.
- [ ] torus_solve.py: --psub, --seedf/--seedfix, group-wise edge sets, relation-subspace E-step; B config reproduces exactly with --psub 0.
- [ ] torus_c_seed.py: label-free structural role seeds.
- [ ] torus_b.py: --dir/--data, relation-subspace dims, angle-code clusterer, residual NMI, polarity on antisymmetric relations, strata, seeds-only baseline.
- [ ] Dev controls (shuffled, random, metadata, seeds-only, B config R) -> Amendment 1 (gate numbers) committed before the final solves.
- [ ] Dev ablation: split subspaces, seeds, angle code, polarity, tori, sphere coarse address, composed rays, K.
- [ ] Final solves for seeds 17 and 23 (train only); test scored once per seed (lock files); B config R scored on the same split.
- [ ] 10 largest clusters, storage, l2pack bit-exact roundtrip.
- [ ] Results doc, architecture_map.md + changelog.md entries (backed up first), deletions log, own cargo target dir and dev models deleted, no processes left running, import/undefined-name check.
- [ ] User confirmation that it works (CTO to merge).
