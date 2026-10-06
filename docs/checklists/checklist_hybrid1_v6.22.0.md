# Checklist HYBRID-1 v6.22.0 (open dense blender on ATEX, 2026-10-05)

- [x] Read CLAUDE.md, architecture_map, harness docs, ATEX/Granite changelog entries
- [x] ROCm PyTorch venv outside the repo (`~/ai/hybrid1/venv`, torch 2.14.1+rocm7.2, HIP device 0 = Navi 44)
- [x] Candidate verification from model cards (licence, size, architecture, revision)
- [x] Guardian council (`docs/guardian_councils/guardian_council_hybrid1.md`)
- [x] Generalised bit-plane group-quant kernel `amni/inference/qgroup_gemv.py` + tests `tests/test_qgroup_gemv.py`
- [x] Baker `scripts/bake_hybrid1.py` with per-tensor fidelity report
- [x] Engine and flagged backend `amni/serve/blender_atex.py`; `AMNI_BLENDER=atex:<bake>`; hook in `adam_t_client.client()` (backup in `backups/`)
- [x] Dev set `scripts/build_hybrid1_dev.py` (CC0, 6-gram clean)
- [x] Evaluation plan frozen (`docs/adam_t/hybrid1_eval.md`)
- [ ] Bakes and fidelity: Granite 3B, MiniCPM5-2B, Qwen3-4B-Instruct-2507 at 4/3/2 bits
- [ ] VRAM/speed measurements and lever study
- [ ] probe8 scoring (once per frozen configuration), ADAM-T r2/r3 baselines
- [ ] E2E chat suite: hybrid, open alone, ADAM-T r3 + harness
- [ ] `docs/adam_t/hybrid1_models.md`, `hybrid1_results.md`
- [ ] NOTICE attribution, deletions.log entries
- [ ] architecture_map.md and changelog.md dated entries (backups first)
- [ ] Linter, tests, cleanup of processes and scratch
