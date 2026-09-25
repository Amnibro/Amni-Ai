# adam-ray page source: streaming tables from an `adam-gf17` store (lane S2)

`adam-ray` can run from a coded `.gf17` page store instead of the four `.npy` tables. Its results are bit-identical to the `.npy` path on CPU and on GPU, and its resident set is bounded by a page budget.

## Use

```bash
cd crates && cargo build --release -j 6 -p adam-ray --example pages
target/release/examples/pages pack $AMNI_HOME/exports/gf17_continuum/<name>            # -> <name>.gf17, coded, tile 128
target/release/examples/pages <pack> <lexdir> <hold.bin> [bytes=2000] [tokens=200] [cpu|vulkan] [both|bpb|gen|upload] [prompt]
AMNI_HOME=$AMNI_HOME cargo test --release -j 6 -p adam-ray --test pages -- --nocapture  # npy vs gf17, budgets 256 MB and 48 MB
```

- **Rust.** `RayField::new(Some("<name>.gf17"), ..)`. When no path is given, `AMNI_RAY_PAGES` takes precedence over `AMNI_RAY_PACK`.
- **Python.** Use `adam_ray.RayField(..., pages="<name>.gf17")`. `RayField.pages` returns the store path, and `RayField.page_stats()` returns the counters. `amni/compute/ray_field.py::engine()` passes `AMNI_RAY_PAGES` through when `AMNI_RAY_IMPL=rust`.
- **Environment:**
  - `AMNI_RAY_PAGE_BUDGET_MB` (default 256) sets the page budget.
  - `AMNI_RAY_PAGE_TIERS` (default 4) sets the tiers. 4 is exact; 2 (d3+d2) is a usable preview; 1 (d3) is a placeholder only.
- **Cargo feature.** The feature is `gf17`, on by default. It adds the `adam-gf17` and `libc` dependencies. With `--no-default-features` the engine builds as before, and a `.gf17` path returns an error.

## Design

- **Source enum.** `pack::Tables` gains `src: src_gf17::Src`, which is either `Npy` (the resident `cells` built from the npy mmap, unchanged) or `Gf17(Pages)`.
  - The engine reads a cell through `Tables::at(l, e)`.
  - `cpu::logits` calls `Tables::want(plan, mtaps)` once per step, before it applies the taps.
  - `gpu::Gpu::new` builds each level through `Tables::level_cells(z)`.
  - For `Npy`, all three calls are the old slice reads. The float accumulation order is untouched, which is why the results are bit-identical.
- **Cell.** The ternariser that turns 3 sid bytes plus 3 sw f16 into the engine cell `[u32; 2]` is factored out as `pack::mkcell`. The npy load and the page source share it.
- **Page unit: a level, pinned on first touch.**
  - The first time a level is needed, a single-threaded page server decodes it with `Store::level` (sid exact, sw at the tier setting) and builds its `t` cells (8 MiB at t = 2^20). The level is then pinned for the life of the engine.
  - Up to `(budget − reserve) / (8·t)` levels are pinned, with reserve = budget / 8.
  - Every step touches the same levels: plan levels 2..20, the qt levels, hops and memory taps. That is 23 levels for v18 and 28 for v19d and v19e.
  - The 13 to 16 copy levels of v19d and v19e are read only through `cw` and are never materialised as cells. This is where most of the RSS saving over npy comes from.
- **Streaming beyond the budget.**
  - Levels touched after the pin budget is spent are read per cell. `want` batches all of a step's missing cells into one `Store::get` of 5 words per cell (2 sid words, 3 sw words), and the built cells go into a sharded, direct-mapped cell cache (16 B per slot, reserve / 2).
  - `Store`'s own decoded-tile LRU gets reserve / 2.
  - Every 128 fetch batches the server calls `Store::unmap()` (drops the PTEs; the page cache stays) and `malloc_trim`. Without this, mmap fault-around and arena retention inflate RSS.
- **Loaded eagerly at open.**
  - `dp`: 256 rows of level 0 through `get`, level 1 through `level`. 33.5 MB.
  - `bias`.
  - `cw`: slot 0 of each copy level, 2 MiB per copy level.
- **Threading.**
  - The `Store` is owned by one `gf17-pages` thread. It runs jobs sent over a channel inside a dedicated rayon pool with the same size as the global pool.
  - Engine workers block on a channel receive, which never work-steals.
  - An earlier `Mutex<Store>` version deadlocked: a worker holding the lock inside `Store::get`'s nested `par_iter` stole another beam's `logits` job, and that job blocked on the same lock.
- **GPU upload.** `Gpu::new` builds the `Rg32Uint` cell texture one level at a time from the store (decode, then `write_texture`). The whole cell table never exists in host RAM. The tier setting gives a coarse upload (d3, or d3+d2); 4 tiers are exact.

## Changes to existing files (kept minimal; the logic is in new files)

| file | change |
|---|---|
| `adam-ray/src/pack.rs` | `Tables.src` field; `mkcell` factored out of the load closure; `.gf17` branch in `load` |
| `adam-ray/src/cpu.rs` | `cell` reads `tb.at(..)`; `logits` calls `tb.want(..)` before the taps |
| `adam-ray/src/gpu.rs` | the cell texture closure calls `tb.level_cells(z)` |
| `adam-ray/src/field.rs` | `AMNI_RAY_PAGES` env fallback |
| `adam-ray/src/lib.rs` | `pub mod src_gf17` |
| `adam-ray-py/src/lib.rs` | `pages=` kwarg, `pages` getter, `page_stats()` |
| `amni/compute/ray_field.py` | pass `AMNI_RAY_PAGES` to `adam_ray.RayField` |
| `adam-gf17/Cargo.toml`, `src/lib.rs` | `npy` feature no longer depends on `adam-ray`; this removes the cargo dependency cycle, and a small `npy()` header reader replaces `adam_ray::npy::Npz`. `Store.pcap` (plane-index cap, set to all planes by the page source) and `Store::unmap()` |
| `crates/Cargo.toml` | `adam-gf17` added to the workspace members |

New files: `adam-ray/src/src_gf17.rs`, `adam-ray/examples/pages.rs` (pack plus measure), `adam-ray/tests/pages.rs`, and this note.

## Results (2026-09-25; Ryzen 9 9950X3D, heavily loaded by two training jobs; RAYON_NUM_THREADS=6; throughput ±30% run to run)

The evaluation text is 2,000 bytes of `data/clean1/hold.bin` at offset 1,234,567. The batched bpb is `RayField::bpb` with 16 beams × 128 contexts. The decode is `stream` with topk = 1, 200 bytes, no veto. The lexicon is `data/clean1`.

### Parity: exact everywhere

| pack | bpb (2,000 B) | logits FNV (512,000 f32) | batched bpb | decode |
|---|---|---|---|---|
| v18 `rayhash_clean1v2_base_v18_s100k` | 1.87846216514683695 both | `164a8b26e0d16c20` both | 1.79108088018359468 both | identical |
| v19d `rayhash_v19d_clean_chat4c_s60k` (retired to `exports/legacy_grok`, F8) | 2.16964848225962292 both | `f0143d8cdd9d0061` both | 2.05765664863378728 both | identical |
| v19e `rayhash_v19e_clean_chat4e_r4_s30k` (retired to `exports/legacy_grok`, F8) | 2.33756774526122468 both | `7c07e044662f9ca4` both | 2.20949163860542930 both | identical |
| v19d `rayhash_v19d_clean_chat6c_s60k` (serve default 3d7893e to F8) | 2.27442265713151581 both | `e3fd3541e21f1405` both | 2.09451936920121939 both | identical |
| v19f `rayhash_v19f_clean_chat7e_s40k` (serve default since F8, MASKV) | 2.32975395604304225 both | `74279bd2692d8e62` both | 2.15184850423366480 both | identical |
| v19e, GPU (RX 9060 XT, Vulkan) | 2.33756775530370398 both | `0cc4f4687384d0b7` both | 2.20949164007896037 both | identical |

- Streamed budgets of 128, 64 and 48 MB are also exact (`tests/pages.rs`: 0 of 512,000 logits differ, and the decode is equal).
- The Python path (`ray_field.engine` with `AMNI_RAY_PAGES`) gives the same SHA-1 over all logits.
- The stores themselves: `checkpoint_roundtrip` passes on v19e with the new npy reader.

### Store size (`pages pack`, coded, tile 128)

| pack | npy | `.gf17` | pack time |
|---|---|---|---|
| v18 | 331,351,552 B | 142,955,165 B (0.431×) | 0.8 s |
| v19d | 472,909,312 B | 182,584,958 B (0.386×) | 1.1 s |
| v19e | 501,220,864 B | 189,114,373 B (0.377×) | 1.3 s |
| v19d chat6c | 472,909,312 B | 184,400,081 B (0.390×) | 1.3 s |

### RSS, CPU (MB): after load / after the 2,000-byte eval / after the 200-token decode (peak)

| pack | npy (fully resident) | gf17, 256 MB budget | gf17, 128 MB | gf17, 64 MB |
|---|---|---|---|---|
| v18 | 281 / 344 / 283 (peak 803) | 71 / 326 / 265 (peak 381) | | |
| v19d | 446 / 512 / 451 (peak 1,176) | 115 / 369 / 344 (peak 424) | | |
| v19e | 474 / 541 / 479 (peak 1,251) | 122 / 376 / 351 (peak 432) | 119 / 310 / 280 (peak 397) | 114 / 283 / 224 (peak 366) |
| v19d chat6c | 446 / 512 / 451 (peak 1,175) | 117 / 371 / 346 (peak 429) | | |
| v19e, decode only | 476 (peak 1,250) | 350 (peak 353) | | |

- **What gf17 RSS consists of.** Pinned levels (23 × 8 MiB for v18, 28 × 8 MiB for v19) plus fixed state outside the budget: `dp` 33.5 MB, lexicon 17 MB, `cw` (26 to 32 MB for v19), and the engine's eval buffers. The eval buffers are the ~60 MB transient that npy shows too, between eval and decode.
- **Peak.** The npy path peaks at 2.6 to 2.9× its steady state while it copies the tables. The page path never materialises a table.

### Throughput, CPU (B/s for eval, tok/s for decode)

"Cold" means the first pass, which for gf17 includes the one-time level pins: 0.24 s for v18, 0.30 s for v19d and v19e. "Warm" means the same pass run again.

| pack | eval cold npy / gf17 | eval warm npy / gf17 | batched 16×128 npy / gf17 | decode warm npy / gf17 |
|---|---|---|---|---|
| v18 | 212K / 7.9K | 187K / 180K | 64K / 61K | 144K / 150K |
| v19d | 178K / 7.5K | 135K / 172K | 67K / 62K | 87K / 114K |
| v19e | 160K / 7.3K | 158K / 129K | 56K / 51K | 118K / 105K |
| v19e, 128 MB (14 pinned, 32 streamed) | | 22.6K | 1.4K | 55K |
| v19e, 64 MB (7 pinned) | | 3.3K | 0.9K | 18K |

- **Warm throughput.** Once the used levels are pinned, a cell read is a slice read: warm gf17 is within noise of npy.
- **Streamed levels are expensive on a miss.** One cold cell costs two 128² tile decodes (sid plus sw, 4 planes each), about 130 to 250 µs under this load. Budgets below the used-level set (~224 MB at t = 2^20) therefore run 5 to 50× slower. They are for bounded RSS on bigger models, not for speed.

### Cold start (page cache dropped with `POSIX_FADV_DONTNEED` before each run)

| | npy | gf17, 256 MB |
|---|---|---|
| CPU load, v18 / v19d / v19e | 0.43 / 0.43 / 0.45 s | 0.09 / 0.16 / 0.16 s (plus 0.24 to 0.32 s of pins on the first step) |
| CPU time to first token, v19e (decode only) | 0.458 s | 0.478 s (0.161 s open + 0.318 s pinning 28 levels) |
| GPU, v19e: tables + upload | 0.421 + 0.569 = 0.99 s, peak 1,251 MB | 0.168 + 0.753 (0.428 s of it level decode) = 0.92 s, peak 184 MB |
| GPU `RayField::load`, and time to first token | 1.38 s, 1.39 s | 1.29 s, 1.30 s |
| GPU upload with tiers 2 / 1 (bpb) | | 1.11 s (2.486) / 0.91 s (11.07, unusable) |

GPU steady state is identical on both paths, because they use the same textures: about 9.6K B/s single-stream eval, 38K to 52K B/s batched, 3.5K to 5.4K tok/s decode.

## Design note: on-demand GPU paging (not built)

Today the GPU path uploads every level from the store at load. That bounds host RAM, but VRAM still holds the full `l × t × 8` cell texture: 386 MB for v19e. Full on-demand paging would work as follows:

1. **Page table.** Replace the `Rg32Uint` `D2Array` cell texture with:
   - a resident pool of N level-slices (`D2Array`, N ≤ l);
   - a small `R32Uint` page table, level → slot or `ABSENT`.

   The shader reads the page table first. An absent level returns zero taps (rule 3: distant pages return zeros). The GPU also appends the level id to a `miss` storage buffer, deduplicated with an atomic bitmask of l bits.
2. **Host loop.** After each dispatch batch, the host reads back the miss mask (l bits, piggy-backed on the logits readback). It decodes the missing levels through `Pages::level` on the page server, uploads them into free or LRU slots with `write_texture` (one level is 8 MiB, about 1 ms over PCIe 4), updates the page table, and re-runs the steps that touched absent levels.
   - Keeping exactness needs a re-run policy: either stall the step until its levels are resident, or accept "zero until resident" answers during warm-up.
   - For serving, stall on the first step (it touches the ~28 hot levels, which is the same 0.3 s as CPU pinning). Later misses are rare, because the used-level set is fixed per geometry.
3. **Tier refinement (rule 4).**
   - Keep one `R8Uint` digit-plane array per tier instead of pre-ternarised cells. Decode `R + 17G + 289B + 4913A` and ternarise in WGSL, with MID fill for tiers not yet resident.
   - Upload d3 plus the exact `sid` first, then d2, then d1 and d0.
   - This moves `mkcell` into the shader, and costs 4 B per slot per tier instead of 8 B per cell. Only the page table and the per-slot tier count change as tiers arrive.
   - Exactness holds once all 4 tiers are resident. Tiers 1 and 2 measured above give the quality of each stage.
4. **Budget.** VRAM slots N = min(l, vram_budget / (8·t)). Copy levels never need slots, because Host reads them through `cw`. So N = 28 already serves v19e fully, at 224 MiB instead of 368 MiB.

The upload-from-store path built here is step 0 of this plan: `Tables::level_cells` is the per-level decode that on-demand paging would call.
