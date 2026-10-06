# Checklist STORE-2 (v6.24.0)

- [x] Read CLAUDE.md, STORE-1 prereg/results/code, TORUS-A2b and TORUS-C results (incl. CTO audit), architecture_map.md
- [x] Pre-registration `docs/adam_t/store2_prereg.md` committed before any eval item (d87216b)
- [x] Eval set: 570 authored by an isolated agent, decontaminated (6-gram) vs locked suites + STORE-1 set, strict absent verification, split + sha256 lock, committed (128fff6)
- [x] Piece vocabulary: 83,521 = 17^4 ids (512 byte-fallback + 83,009 BPE), trainer config recorded (`data/store2/pieces/trainer.json`)
- [x] Measurements: text 0-order vs context (`tstat_full.json`), fact predictors and ablations (`fstat_*.json`), torus bound (`torus_bound.json`)
- [x] Backups to `backups/store2/` before editing (adam-store1 lib/bin, compute/serve store1.py, harness, test_store1)
- [x] Guardian council `docs/guardian_councils/guardian_council_store2.md`
- [x] Rust: base-17 rANS coder, symmetric count/encode/decode procedure, BPE pieces + case channel, cycle-checked fact prediction, rANS codebooks, header/core/label/name/qualifier/string streams, hot table, piece-key name table, reader + C ABI `s2_*`, build/parity/bench
- [x] Full parity of the final drive
- [x] Python: `amni/compute/store1.py` Drive gains the `s2` prefix (piece keys); `amni/serve/store1.py` STORE-2 mode (mined property map, piece-nonce keys, stricter absence, drive-only class check)
- [x] Harness: `AMNI_STORE2=1` (default off) route `store2`
- [x] Dev ablations, selection by the pre-registered rule, F choice by dev latency
- [x] Test scored once with frozen code
- [x] Tests: `tests/test_store1.py`, `tests/test_store2.py`, harness suites in the main tree environment
- [x] Results doc, changelog entry, architecture_map entry (shared tree, after backup)
- [x] Rust target dir deleted, deletions logged, processes stopped
