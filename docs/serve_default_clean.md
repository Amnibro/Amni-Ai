# Serve default: license-clean stack (SERVE-DEFAULT + GROK-PURGE)

Date: 2026-09-25. Branch `feat/ray-field-serve`.

## What changed

The `ray` backend (`AMNI_GEN_BACKEND=ray`) now picks the clean stack when no env override is set.

| Setting | Old default | New default | Where |
|---|---|---|---|
| Pack (`AMNI_RAY_PACK`) | `rayhash_v16_big2_L21_e20_hop4_q5_s150k_pk_q5_i8.npz` (big2, 35% Grokipedia) | `exports/gf17_continuum/rayhash_v19e_clean_chat4e_r4_s30k_sid.npy` (`DEFAULT_PACK`) | `amni/compute/ray_field.py` |
| Base / knowledge pack | none | `rayhash_clean1v2_base_v18_s100k_sid.npy` (`BASE_PACK`): KS coder `ks2-clean1`, bpb test, tile_synth naturalness scorer (`TS_NAT_PACK`), parity tests | `ray_field.py`, `ks_codec.py`, `tile_synth.py` |
| Lexicon (`AMNI_RAY_LEX`) | `big2` | `clean1` (`DEFAULT_LEX`) | `ray_field.py` |
| Engine (`AMNI_RAY_IMPL`) | `taichi` | `rust` (`DEFAULT_IMPL`, the `adam_ray` PyO3 module) | `ray_field.engine()` |
| Model id (`AMNI_RAY_MODEL`) | `adam:ray-v16` | `adam:ray-v19e-clean` (`RAY_MODEL`) | `amni/serve/gguf_runtime.py`, `scripts/amni_serve.py` (`POST /models/activate`) |
| KS local-learn sources (`AMNI_LEARN_SRCS`) | `wikipedia,grokipedia` | `wikipedia` | `amni/compute/ray_learn.py` |
| Answer citation | `Grokipedia` or `Wikipedia` by URL | always `Wikipedia` | `gguf_runtime._cite` |
| `v19_rust_eval.py` / `nightly_eval.py` `LEX` | `big2` | `clean1` | scripts |
| `AMNI_RAY_HIST` | `24` | unchanged | `amni/serve/user_memory.py` |

`ks1-v16` in `ks_codec.CODERS` is pinned to `LEGACY_PACK` (`exports/legacy_grok/...v16_big2...npz`), so `data/ks/store` (1 shard, frozen `ks1-v16` coder) still decodes. `KSCodec()` with no arguments still means `ks1-v16` until `store_clean` is complete (see KS status).

The Taichi `RayField` class cannot run v19 packs. Use it only with an explicit v16/v18 `AMNI_RAY_PACK` and `AMNI_RAY_IMPL=taichi`.

`adam_ray` was rebuilt from `crates/adam-ray-py` at 89f7a6d (`cargo build --release -j 6 -p adam-ray-py`, installed with an atomic rename to `adam_ray.cpython-312-x86_64-linux-gnu.so` at the repo root, as `crates/build_py.sh` does). It loads v19e (`L=46`, `copy=16`). The previous module is in `backups/adam_ray.cpython-312-x86_64-linux-gnu.so.v_servedefault.bak`.

## Env switches

```bash
AMNI_GEN_BACKEND=ray                       # select the ray backend (the default backend is still the Granite bake)
AMNI_RAY_PACK=<path>_sid.npy|.npz          # pack, default v19e clean
AMNI_RAY_LEX=clean1                        # lexicon dir under data/
AMNI_RAY_IMPL=rust|taichi                  # engine, default rust
AMNI_RAY_ARCH=cpu|vulkan                   # device, default vulkan
AMNI_RAY_MODEL=adam:ray-v19e-clean         # model id reported by /healthz and /models
AMNI_LEARN_SRCS=wikipedia                  # KS sources for local learning
TS_NAT_PACK=<path>                         # tile_synth naturalness pack, default BASE_PACK
```

## Rollback

Old defaults via env only. This serves a big2 (Grokipedia) model, so it breaks the GROK-PURGE order; use it only in an emergency.

```bash
AMNI_GEN_BACKEND=ray AMNI_RAY_IMPL=taichi AMNI_RAY_LEX=big2 AMNI_RAY_MODEL=adam:ray-v16 \
AMNI_RAY_PACK=$PWD/exports/legacy_grok/rayhash_v16_big2_L21_e20_hop4_q5_s150k_pk_q5_i8.npz \
AMNI_LEARN_SRCS=wikipedia .venv-linux/bin/python scripts/amni_serve.py
```

Previous clean-lineage pack (v19d, same engine and lexicon): `AMNI_RAY_PACK=$PWD/exports/gf17_continuum/rayhash_v19d_clean_chat4c_s60k_sid.npy`.

Code rollback: `git revert bfa170e` (defaults) and the GROK-PURGE commit that adds this file (test and script repoints). Module rollback:

```bash
cp backups/adam_ray.cpython-312-x86_64-linux-gnu.so.v_servedefault.bak .adam_ray.tmp.so && mv -f .adam_ray.tmp.so adam_ray.cpython-312-x86_64-linux-gnu.so
```

Every edited file also has a copy in `backups/<name>.v_servedefault.bak`.

Restore the retired packs to their old paths (every move is recorded in `exports/legacy_grok/MANIFEST.json`):

```bash
.venv-linux/bin/python -c "import json,os;[os.rename(os.path.join('exports/legacy_grok',m['file']),m['from']) for m in json.load(open('exports/legacy_grok/MANIFEST.json'))['files']]"
```

## GROK-PURGE

Moved to `exports/legacy_grok/`, names unchanged: 809 files, 57,472,750,010 bytes (53.5 GiB), 319 pack/report stems. Nothing was deleted. A pack moved when any of these held:

- its name contains `big2`, `big3`, `grok` or `pycode`;
- its json/log names big2, big3, pycode or Grokipedia (as the corpus, a `multi` salt, or a loader line such as "Loaded ... Grokipedia STEM texts");
- `cfg.corpus` is big2, big3, chat1, chat2, chat3, chat4 or pycode (each of these corpora mixes in big2 bytes directly);
- it was warm-started from a moved pack (v16 chat1/code/pycode, v18 chat2, v19c2 chat3, v19d chat4*);
- it is a byte dump of the big2 hold set (`dump_*.npz`), or it was built from the Grokipedia civics/STEM loader (`ray_ngram_k8.npz`, `ray_ngram_meta.json`, `adam_omni_model.json`).

By family: v13 12.1 GiB, v3 11.1, prune 6.0, v9 4.5, v11 3.7, v8 2.0, v16 2.0, v19d-chat4 1.7, ab 1.7, v10 1.1, tritdeq 1.0, plus smaller runs.

The rayhash packs with weights that stay in `gf17_continuum` are `clean1v2_base_v18_s100k`, `v19d_clean_chat4c_s60k`, `v19e_clean_chat4e_r4_s30k` and the two `v13_big4` packs (FineWeb only). `v19d_clean_chat6c_s60k` was still training when the packs were moved.

### Open finding: the v19e "clean" pack is not fully clean

`data/chat4c` and `data/chat4e` (the v19d-clean and v19e fine-tune corpora) put 28 MB of `data/chat2` chunks into training. `data/chat2` was built on 2026-09-23 with a 35% big2 prose mix: 10.3 MB, about 42% of which is Grokipedia. Their own prose slice is clean1: the random offset carries a 32-bit length signature. The chat2 chunks are not. In prose-probe checks, 5 of 16 chat4e probes were found only in big2 and in no clean1 byte. Estimate: about 14% of chat4e is big2 text, about 6% Grokipedia. `data/chat6c` (chat5 distractors plus clean1 prose) showed 0 of 12 big2-only probes. v19e stays the default because this mission named it. The fix is the queued chat6c retrain (CHAT6C-TRAIN / F8); swap `DEFAULT_PACK` when that pack passes eval.

Other leftovers, reported only:

- `data/big3/train.bin` (2.07 GB) still holds all of big2, Grokipedia included.
- `data/chat1` through `data/chat4e` and `data/chat6c` meta files name big2 bytes. chat6c's are clean1 bytes stored under the old key name.

### Tests changed

- Skipped with "legacy_grok retired": `tests/test_ray_field_parity.py::test_parity`, `crates/adam-ray-py/tests/test_parity.py::test_v16`, `crates/adam-ray/tests/parity.rs::parity_v16` (`#[ignore]`), `crates/adam-ray/tests/tables.rs::tables_v16_exact` (`#[ignore]`).
- Moved to clean equivalents: `v18` parity/tables/qa/bench now use `clean1v2_base_v18` with the clean1 lexicon (`crates/adam-ray-py/tests/{ray_ref,test_parity,bench}.py`, `crates/adam-ray/tests/{parity,tables}.rs`, `crates/adam-qa/tests/parity.rs`, `crates/adam-ray/examples/bench.rs`, which defaults to v19e). `tests/test_ray_field_bpb.py` uses `BASE_PACK` on the clean1 hold, with a threshold of `< 1.85` (it measures 1.797 at 10 batches). `tests/test_ray_field_gen.py` uses `engine()`, and `tests/test_ray_field_backend.py` expects `RAY_MODEL`.
- Untracked scripts repointed from `rayhash_v18_chat2_qt_s20k` to the clean base: `scripts/{x0_code_eval,ks_eval,v19_baseline,v19_conv_eval}.py`, `crates/adam-tiles/scripts/x0_code_eval_min.py`.
- Left alone: `tests/ray_field_ref.py`. It is the v16 reference generator for the skipped parity test, and its default path no longer exists.

### Knowledge Store

At 01:33 on 2026-09-25, `data/ks/store_clean/wikipedia` had 8 of 56 shards done (00000 to 00007, each with `meta.json`). Six more were partial (00008, 00009, 00010, 00025, 00026, 00036), and the KS2 encoders are still running. It is not complete, so the defaults were not switched.

TODO once all 56 shards are done:

- set `KSCodec()` to default to `ks2-clean1`;
- point `KSIndex`'s store root at `store_clean`;
- switch `scripts/ks_wiki_defs.py` to `store_clean`;
- move `ray_learn._wc` (the KS term idf) off `data/big2/wordcounts.txt`. That table is a word list, not model weights. It stays unchanged while KS2 runs so every `store_clean` shard is indexed with the same idf.

After that, retire `ks1-v16` and `data/ks/store`, pending the CEO's deletion call.

### Other big2 defaults (legacy trainers and analysis, not serving)

- `scripts/ray_hash_field_ti_v18.py`: big2 wordcounts.
- `scripts/boundary_slot_ceiling.py` and `scripts/gf17_group_calc_test.py`: `CORPUS=big2`.
- `crates/adam-train/tests/{fwd_parity.py,v19_drv_ref.py,v19_fwd.rs}`: big2 lexicon.
- `crates/adam-ray/src/field.rs`: `RayField::new(None...)` and the default `bpb` hold still point at big2. This file is off-limits during F8. The Python wrappers always pass explicit paths.
- `amni/compute/ks_answer.py` (Berrik) reranks with `AMNI_RAY_LEX` defaulting to big2. It is off unless `AMNI_ANSWER_RERANK=1`.

### Things named "grok"

Not deleted. None of these serve Grokipedia data.

- Grokipedia ingest code: `scripts/ks_download.py` (`grok()`), `scripts/ks_ingest.py` (grokipedia branches, guarded), `scripts/build_big_corpus.py` (`grok()` source), `scripts/build_clean1_corpus.py` (grok source at share 0), `scripts/stream_omni_dataset.py` (`load_grokipedia_civics_samples`), `scripts/train_adam_omni_ray.py`, `scripts/fill_1t_public_source.py`.
- Grok the xAI model/CLI, not the dataset: `amni/a1/triumvirate.py` (`grok` oracle), `amni/delve/adapters.py` (Delve `grok` adapter), `amni/serve/jarvis_web.py` (`/delve @grok` hint), `evals/hard_v1/adapters/grok_45.py`, `scripts/eval_hard_v1.py`, `workspace/duel_grok_check.py`.
- Grok-bot team reports: `reports/grok_*.json`, `grok-review.md`, `logs/grok_*`.
- `ray_learn.MIRROR` still lists `grokipedia` as a mirror domain to reject. That keeps it out.
