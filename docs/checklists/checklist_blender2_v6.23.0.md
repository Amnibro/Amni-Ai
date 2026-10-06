# Checklist BLENDER-2 v6.23.0 (current-generation blender, verified VRAM x quality, 2026-10-06)

Priority (CEO update 2026-10-06): Granite 4.2 3B first, MiniCPM5-2B second, Qwen3.5 Gated DeltaNet port third (time-boxed about 1 day).

- [x] Read CLAUDE.md, HYBRID-1 results (incl. CTO audit caveats), eval plan, models doc, env, blender backend, kernel, baker, decision memory
- [x] Worktree reset onto feat/ray-field-serve (b62845b)
- [x] ROCm venv rebuilt at ~/ai/hybrid1/venv from docs/adam_t/hybrid1_env.txt; tests/test_blender_atex.py and tests/test_qgroup_gemv.py run in it
- [x] Candidate verification from model cards and the HF API (licence, date, architecture, params, revision); Qwen org checked for a dense Qwen3.8 <= 4B
- [x] Guardian council (docs/guardian_councils/guardian_council_blender2.md)
- [ ] Backups of every edited tracked file in backups/ (*.vblender2.bak)
- [ ] Determinism: reproduce the intermittent garbage (full config sequence x >= 5 fresh processes), find and fix the cause, 5 consecutive clean runs
- [ ] Equivalence harness: >= 50 prompts, greedy token equality or first divergence step plus logit max-abs-diff
- [ ] Granite 4.2 3B: verified bake int8, calibrated int4 (GPTQ), equivalence, same-config VRAM x quality (streamed, minimal-VRAM)
- [ ] MiniCPM5-2B: same
- [ ] Qwen3.5 Gated DeltaNet port (time-boxed), KV/state VRAM measured
- [ ] Dev-set model selection (never probe8); probe8 scored once per frozen configuration, from the same process configuration as the VRAM number
- [ ] docs/adam_t/blender2_results.md with the Pareto and file/field citations; docs/adam_t/blender2_models.md
- [ ] Recommended bake under bakes/ (manifest + NOTICE), NOTICE attribution, AMNI_BLENDER works with it (default off)
- [ ] deletions.log entries for every deletion; disk peak under 40 GB; only venv + recommended bake + manifest kept
- [ ] architecture_map.md and changelog.md dated entries (backups first)
- [ ] Linter (pyflakes), tests, all own processes stopped
