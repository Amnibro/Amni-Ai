# adam-gf17: lossless GF(17) Reffelt page store (lane S prototype)

`adam-gf17` stores Adam's tables (`_sid`, `_sw`, `_dp`, `_bias`) as GF(17) RGBA pages on NVMe and reads them back bit-exact through an mmap reader with a bounded hot-tile LRU. The tables can be larger than RAM or VRAM: a reader touches only the pages that a fetch needs, and it can load the d3 digit plane first, then refine to exact.

There are two on-disk modes of the same logical PTEX page:
- **raw** (`tile = 0`, file version 1) stores one digit per byte, which is 2.0× the `.npy` size;
- **coded** (`tile > 0`, file version 2) entropy-codes each digit plane per tile, which is 0.37× to 0.43× the `.npy` size.

Both decode to the identical RGBA page (`rgba()` / `page()`), and both are bit-exact.

It is CPU only. It depends on nothing that `adam-ray` or `adam-train` will change, apart from the `.npy` reader (`adam_ray::npy::Npz`), which sits behind the default `npy` feature so that `adam-ray` can later depend on this crate without a cycle.

## Codec

- **Unit.** Each table's bytes, little-endian, are cut into 16-bit words, level by level. So a word is one f16 weight, two u8 ids, or half of an f32. A level with an odd byte count gets a zero pad byte.
- **Key.** Every table uses `raw` by default: key = word. This matches `amni/compute/reffelt4.py::encode_fp16_to_rgba4` bit for bit, and `tests/test_py.py` checks it.
  - `ord` is optional, per table, and applies to f16 only: key = `b | 0x8000` for b ≥ 0, else `!b`. It maps f16 order onto integer order, so a truncated key never changes sign for |w| > 3e-4 and never decodes to NaN.
  - Both keys are bijections, so a roundtrip is exact with either.
- **Digits.** `REFFELT_K4 = [1, 17, 289, 4913]`, and d_k = key / K4[k] % 17. 17^4 = 83,521 ≥ 65,536, so d3 ≤ 13.
- **Pixel (rule 5).** One word is one RGBA8 pixel: R = d0, G = d1, B = d2, A = d3, one digit per channel. Inverse: `R + 17G + 289B + 4913A`.
- **Page.** A page is 4096 × 4096 pixels, 16,777,216 words. Word w of a level lives on page w / 2^24 at pixel (x, y) = (w % 4096, w / 4096 % 4096). The last page of a level keeps only the rows it uses: rows = ceil(words / 4096) ≤ 4096.
- **Tier truncation.** With n loaded tiers (d3 down to d(4−n)), each missing digit is filled with mid-range 8, and the value is clamped to 65535:
  - `raw` zeroes a non-finite f16 result, and the count is reported;
  - `ord` clamps the key to the finite range [0x0400, 0xFBFF], so it is always finite.
  - n = 4 is exact.

## Page file (`.gf17`)

The file is little-endian. In raw mode (version 1) every plane is 4096-byte aligned.

| offset | bytes | field |
|---|---|---|
| 0 | 8 | magic `GF17PTX1` |
| 8 | 4 | version = 1 |
| 12 | 4 | page width = 4096 |
| 16 | 4 | page height = 4096 |
| 20 | 4 | flags = 0 (raw) or 1 (coded) |
| 24 | 8 | directory offset |
| 32 | 8 | directory length |
| 40 | 4 | coded only: tile width |
| 44 | 4 | coded only: tile height (= width) |
| 48 | 4 | coded only: hash of the directory |
| 52 | 4 | coded only: hash of header bytes 0..52 |
| ..4096 | | zero |
| 4096.. | | planes, in stream order (below) |
| dir | | directory |

**Planes are planar RGBA.** A page's four channels are stored as four separate u8 planes of `rows × 4096` bytes. The logical pixel is still one value, as in the `page()` / `rgba()` view. The planes are grouped into four tier streams, written in this order:

1. **stream 3 (coarse):**
   - every plane (d3, d2, d1, d0) of the *exact* tables: `sid`, which holds coordinates (rule 3), and `bias`;
   - the d3 plane of every value table (`sw`, `dp`).
2. **stream 2:** the d2 planes of the value tables.
3. **stream 1:** the d1 planes.
4. **stream 0:** the d0 planes.

A coarse reader streams one contiguous region and has exact coordinates plus d3 weights (rule 4: d3 first). Within a stream, the order is table, then level, then page (then tier, for exact tables).

**Directory:**
- `u32 ntables`, then for each table:
  - `u16 name_len`, then `name`;
  - `u8 dtype_bytes` (1 = u8, 2 = f16, 4 = f32), `u8 ord`, `u8 exact`, `u8 ndim`;
  - `u64 shape[ndim]`.
- `u32 nentries`, then for each entry (44 bytes, one per plane):
  - `u32 table`, `u32 level`, `u32 page`, `u32 tier`, `u32 rows`;
  - `u64 cell0`, `u64 cell1`: the cell range, where a cell is `shape[1]` and holds `prod(shape[2:])` elements;
  - `u64 offset`;
  - coded only: `u64 length` (52-byte entries).

Level is `shape[0]` for tables with more than one dimension. A 1-D table has one level.

## Coded mode (version 2)

The logical format does not change: PTEX pages, one GF(17) digit per RGBA channel, planar tier streams in the same order (stream 3, then 2, 1, 0). Only the bytes of each plane are compressed, and `rgba()` reconstructs the page pixel for pixel.

- **Tiles.** A plane (`rows × 4096` digits) is cut into `tile × tile` pixel tiles, row-major over the page; the last tile row is `rows % tile` high. Each tile decodes on its own, so a fetch touches one tile per plane, not the page. The default in the tools is 128 (256 also works; see Results).
- **Model.** Every plane has one static frequency table per page, with 17 contexts of 17 symbols (M = 2^10):
  - d3 is conditioned on the d3 digit 32 pixels earlier in the tile scan (the same rANS lane, so the lanes stay independent);
  - d2, d1 and d0 are conditioned on the next-higher digit of the same pixel (d3 → d2 → d1 → d0). A tier therefore needs only the tiers above it, which matches the d3-first stream order (rule 4).
- **Coder.** Static rANS over base-17 symbols: 32-bit state, 16-bit renormalisation, 32 interleaved lanes (symbol i uses lane i mod 32).
  - Decode runs 4 AVX2 vectors of 8 lanes, with one gather from a fused 32-bit table (symbol, frequency, slot bias) per 8 symbols. A scalar path handles the tail and CPUs without AVX2.
  - A tile whose digits are all equal is stored as one byte.
- **Plane blob:**
  - `u32 ntiles`, `u32 hash(bytes 8..header end)`, `u16 freq[17 × 17]`;
  - `u32 tile end offsets[ntiles + 1]` (relative to the blob), then `u32 tile hash[ntiles]`;
  - the tile streams: `u32 state[32]`, then the renormalisation words.
- **Integrity.** Every tile, plane header, the directory and the file header carry a 32-bit hash. A decode also checks that every rANS lane returns to its initial state and that the stream is used exactly. Any mismatch is an `io::Error` naming the table, level, page, digit plane and tile. It is never a panic or a silent wrong value.
- **Writing.** `Writer` spills the encoded planes to `<out>.spill` as levels arrive, then `finish()` lays them out in stream order behind a 4 KiB header, writes the directory and deletes the spill. So a table larger than RAM can still be written level by level.

## Reader

- `Store::open(path, budget_bytes)` maps the whole file read-only with `MADV_RANDOM`. It reads either mode.
- **Errors.** `get`, `fetch`, `fetch_many`, `level`, `unpack`, `pdig` and `rgba` return `io::Result`. Raw mode never fails after `open`; coded mode fails on a checksum or rANS state mismatch.
- **Coded hot set.** In coded mode the LRU is over decoded tiles, 4 × tile² bytes each, and the budget bounds them.
  - A fetch dedupes the tiles in its batch.
  - Plane headers are parsed into a plane index (a 70 KB decode table per plane), in parallel for large batches. The index is capped at budget / 8.
  - A fetch issues `WILLNEED` for all tile ranges, decodes the missing tiles (inline for up to 2, on the rayon pool otherwise), then `DONTNEED`s the compressed bytes.
  - A cached tile that holds fewer tiers is refined in place: d3 first, then d2 only when asked.
  - Evicted tile buffers are recycled.
- `reset(level)`: level 0 drops the decoded tiles, 1 also drops the mapping and the page cache, and 2 also drops the plane index. `cold()` = `reset(2)`.
- `pdig(t, l, p, tiers)` returns a page's four digit planes, borrowed in raw mode and decoded in coded mode.
- **Hot set.** The hot set is an intrusive LRU over 64 KiB file tiles, which is also the kernel's fault-around window. Each tile has a 16-bit mask of the 4 KiB rows touched so far.
  - A fetch touches the planes it needs, then issues one `MADV_WILLNEED` per not-yet-seen 4 KiB row. The reads for the up-to-8 planes of an exact `(sid, sw)` fetch therefore go to NVMe in parallel.
  - A fetch then reads the digits.
  - Evicted tiles get `MADV_DONTNEED`. Resident mapped pages only ever lie in hot tiles, so RSS ≤ budget + fixed overhead. The overhead is 10 bytes per 64 KiB tile of file.
- `get(&[(table, level, word)], tiers)` is the batched word fetch. `fetch_many(&[(level, cell, slot)], tiers)` / `fetch(level, cell, slot)` return `(sid u8, sw f16 bits)`. `tiers < 4` returns the truncated approximation for value tables. Exact tables are always exact.
- `level(t, l, tiers, out)` / `unpack(t, tiers)` do a bulk parallel decode. In coded mode each tile is decoded and written straight into the output words, with no intermediate page planes. They prefetch the next level's planes with `WILLNEED`, and drop the decoded planes with `DONTNEED`.
- `cold()` drops the mapping and the file's page cache (`POSIX_FADV_DONTNEED`), and resets the LRU.
- `rgba(t, l, p)` returns the interleaved RGBA8 page, for GPU upload or for parity with `reffelt4.py`.
- `pack(prefix, out, ord, tile)` packs a checkpoint (`tile = 0` is raw). `write(out, &[(Table, bytes)], tile)` packs any arrays. `Writer::create(path, tabs, drop, tile)`, `put(t, l, words)` and `finish()` stream level by level, which is how the synthetic 40 GB table was written without holding it in RAM.

## Python (`adam_gf17`)

```bash
cd crates && adam-gf17/build_py.sh
```

This copies `adam_gf17.cpython-*.so` to the repo root, like `build_py.sh` does for `adam_ray`. Set `DEST=<dir>` to put it elsewhere. The root `.gitignore` does not list `/adam_gf17*.so` yet.

```python
import adam_gf17
st = adam_gf17.pack("exports/gf17_continuum/rayhash_v19d_clean_chat4c_s60k", "/path/v19d.gf17")
st = adam_gf17.pack(prefix, "/path/v19d_c.gf17", tile=128)
st = adam_gf17.Store("/path/v19d.gf17", budget_mb=256)
arrs = adam_gf17.unpack(st)
coarse = adam_gf17.unpack(st, tiers=1)
sid, w = st.fetch(level, cell, slot)
ids, ws = st.fetch_many(levels, cells, slots)
rgba = st.page("sw", 0, 0)
st.tables; st.stats; st.cold(); st.cold(0)
st.tile; st.size; st.streams
```

- `pack` defaults to `ord=False` and `out=prefix + ".gf17"`.
- `unpack` returns `{name: ndarray}` with the original dtypes and shapes.
- `st.page("sw", 0, 0)` has shape (rows, 4096, 4), as uint8.
- `st.stats` is `(hits, misses, evictions, hot_bytes, budget, rss)`.
- `st.tile` is 0 for raw mode; `st.size` is the file size; `st.streams` is the on-disk bytes of tier streams `[0, 1, 2, 3]`.
- `st.cold(level=2)` calls `reset(level)`.
- Corruption raises `IOError`.

## Tests and tools

```bash
cd crates
cargo test --release -j 8 -p adam-gf17
TMPDIR=target/gf17 ADAM_GF17_PREFIX=$AMNI_HOME/exports/gf17_continuum/rayhash_v19d_clean_chat4c_s60k cargo test --release -j 8 -p adam-gf17 --test roundtrip
DEST=target/py bash adam-gf17/build_py.sh && ADAM_GF17_SO_DIR=crates/target/py python crates/adam-gf17/tests/test_py.py
cargo build --release -j 8 -p adam-gf17 --example gf17
target/release/examples/gf17 pack    <prefix> <out.gf17> raw|ord [tile]
target/release/examples/gf17 tiers   <prefix> <store.gf17> $AMNI_HOME/data/clean1 <scratch dir>
target/release/examples/gf17 bench   <store.gf17> <n> <budget_mb> [<prefix> | synth] [<seq cap GB>]
target/release/examples/gf17 synth   <out.gf17> <levels> <log2 cells> [tile]
target/release/examples/gf17 density <store.gf17> <prefix>
```

`RAYON_NUM_THREADS` sets the pool size. The bench runs its sequential pass with 1 thread and with the pool size.

The `cargo test` suite covers:
- `codec_exhaustive`: all 65,536 words under both keys; `ord` is monotone; the truncation error bound; truncated `ord` is always finite.
- `store_roundtrip_multi_page`: for raw and for coded tiles 64 and 256:
  - u8/f16/f32 tables, odd byte counts, a 2-page level, a skewed table and an all-zero table (constant tiles, single-symbol contexts);
  - the LRU budget, cell ranges, truncated sign;
  - coded == raw for every tier count 1 to 4 and for every RGBA page.
- `coded_tile_random_access`: one fetch decodes exactly one tile (miss +1, hot = one tile); tier refinement 1 → 2 → 4 on the cached tile; last partial tile rows; a 2,000-word batch that evicts inside the batch.
- `coded_corruption_detected`: a flipped byte in a tile, a frequency table, the directory or the header gives `Err`, and only for the reads that touch it (other tiles and coarser tiers still read).
- `checkpoint_roundtrip`: a real pack, raw and coded (256), when `ADAM_GF17_PREFIX` is set.

`test_py.py` checks the bindings end to end:
- RGBA page parity with `reffelt4.encode_fp16_to_rgba4`;
- a coded pack (tile 128): bit-exact unpack, fetch parity, RGBA page == raw page, d3-only == raw d3-only;
- corruption raising `IOError`.

## Coded-mode results (2026-09-25, lane S3)

The box is a Ryzen 9 9950X3D at load average 46 to 80 from other jobs, with 6 rayon threads. The raw-mode rows were re-measured in the same sweep, so the two modes are compared under the same load. Timings vary by about ±50% run to run at this load.

### Size

`raw` key. "Uncoded" is the version-1 page file.

| pack | raw `.npy` | uncoded `.gf17` | coded, tile 128 | coded, tile 256 | context-model entropy bound |
|---|---|---|---|---|---|
| v18 (L=28) | 331,351,552 B | 662,731,019 B (2.000×) | 142,955,165 B (**0.431×**) | 138,941,005 B (0.419×) | 0.410× |
| v19d (L=43) | 472,909,312 B | 945,851,819 B (2.000×) | 182,584,958 B (**0.386×**) | 176,142,345 B (0.372×) | 0.364× |

- The per-plane order-0 bound of the old README (0.65×) is beaten, because the contexts (per page, and conditioned on the digit above) carry most of the structure.
- The file is within 2 to 6% of the bound. The overhead is the lag-32 d3 context, M = 2^10, the 128 B of lane state per tile, and the tile index.
- The target of ≤ 0.8× raw is met with margin.

Per table, tile 128:

| table | v18 | v19d |
|---|---|---|
| sid | 0.55× | 0.44× |
| sw | 0.47× | 0.41× |
| dp | 0.19× | 0.19× |

Per tier stream, tile 128, as coded bytes (and coded / uncoded):

| stream | v18 | v19d |
|---|---|---|
| 3: exact `sid`/`bias` + d3 of `sw`/`dp` | 56.3 MB (0.189) | 73.3 MB (0.167) |
| 2: d2 | 27.2 MB (0.224) | 34.2 MB (0.203) |
| 1: d1 | 29.2 MB (0.240) | 37.1 MB (0.220) |
| 0: d0 | 30.3 MB (0.249) | 37.9 MB (0.224) |

The coarse pass (stream 3 only) now reads 0.17× of raw, down from 0.90×.

### Exactness

- `pack` + `unpack` is bit-exact on every table of both packs, with cos = 1.000000000000 on `sw` and `dp`.
- 2,000 random `fetch`es per bench run matched the `.npy` bit for bit.
- The coded RGBA pages equal the raw pages, and truncated tiers 1 to 3 equal the raw-mode truncation.
- `checkpoint_roundtrip` (raw + coded) passes on v19d.

### Random fetch latency

"Tile decoded" means the decoded-tile cache was dropped (`reset(0)`) before each fetch, with the page cache warm: this is the cost of a tile miss. "Cold data" means `reset(1)` (page cache dropped, plane index kept, as a serving process would run). "Cold" means `cold()` (everything dropped).

| store | budget | 1-tile decode p50 / p99 | `fetch` (sid+sw, 2 tiles) p50 / p99 | warm hit p50 / p99 | cold data p50 / p99 | cold p50 / p99 | batch-64 cold / warm p50 |
|---|---|---|---|---|---|---|---|
| v18 coded 128 | 256 MB | **51 / 83 µs** | 115 / 258 µs | 0.9 / 4.0 µs | 481 µs / 1.3 ms | 797 µs / 3.4 ms | 9.2 ms / 43 µs |
| v19d coded 128 | 256 MB | **99 / 405 µs** | 255 / 861 µs | 1.5 / 4.1 µs | 560 µs / 2.1 ms | 811 µs / 2.3 ms | 10.2 ms / 28 µs |
| v19d coded 128 | 2 GB | 106 / 385 µs | 238 / 889 µs | 1.2 / 2.3 µs | 604 µs / 2.5 ms | 903 µs / 2.7 ms | 13.8 ms / 24 µs |
| v18 coded 256 | 2 GB | 281 / 652 µs | 512 µs / 1.17 ms | 1.3 / 3.6 µs | 898 µs / 2.3 ms | 1.1 / 3.0 ms | 19.4 ms / 39 µs |
| v19d coded 256 | 2 GB | 371 µs / 2.0 ms | 505 µs / 1.22 ms | 1.0 / 1.9 µs | 925 µs / 2.9 ms | 1.1 / 3.0 ms | 21.2 ms / 30 µs |
| v18 raw | 256 MB | | | 10.8 / 39 µs | | 124 µs / 1.1 ms | 3.0 ms / 37 µs |
| v19d raw | 256 MB | | | 10.0 / 29 µs | | 108 µs / 1.5 ms | 2.2 ms / 8.7 µs |

- **Target (warm tile p99 < 1 ms).** Met with tile 128 in every run: 83 to 405 µs for one tile, and ≤ 0.9 ms for a two-tile `fetch`. Tile 256 misses it under this load (p99 0.65 to 2.0 ms), so 128 is the recommended tile.
- **A tile decode** is 128 × 128 × 4 digits at about 0.7 to 1.5 ns per digit here.
- **Earlier optimisation steps**, measured in isolation, in ns per digit:
  - scalar 4-lane rANS: 3.5 to 4.4;
  - AVX2 8 lanes: 1.6;
  - AVX2 32 lanes: 0.7 to 0.85.
- **Cold reads cost more than in raw mode**, about 0.5 to 0.9 ms against 0.1 ms. A cold coded fetch is one IO round trip for the plane headers (skipped when the plane index is resident), one for the tile bytes, and the decode. Raw needs one round trip and no decode. Batch-64 cold is 9 to 21 ms (5 to 12 ms with the plane index resident) against 2 to 3 ms. That is the price of the 5× smaller file.
- **Hot-set RSS.** VmHWM was 259 to 283 MB at a 256 MB budget. That is the decoded tiles, plus the plane index (about 70 KB per plane; 25 MB for v19d), plus file-backed pages. `cold()` calls `malloc_trim`.

### Sequential decode (`level` over every table), in GB/s of decoded table bytes

| store | 1 thread warm | 6 threads warm | 6 threads cold | coarse (d3 + exact), 6 threads warm |
|---|---|---|---|---|
| v18 coded 128 | 0.53 | 2.48 | 1.36 | 3.35 |
| v19d coded 128 | 0.63 to 0.74 | 2.25 to 3.12 | 1.73 to 2.17 | 2.54 to 3.18 |
| v18 coded 256 | 0.68 | 2.86 | 1.94 | 3.42 |
| v19d coded 256 | 0.67 | 2.50 | 1.87 | 2.33 |
| v18 raw | 4.20 | 3.95 | 0.57 | 3.91 |
| v19d raw | 2.61 | 3.21 | 0.73 | 4.26 |

- **Single thread:** coded decode is 3.5 to 8× slower than raw.
- **6 threads, warm:** coded reaches 60 to 97% of raw.
- **Cold (reading from NVMe):** coded is 2.4 to 3.4× faster, because it reads a fifth of the bytes.

### Denser fixed-rate digit packing (report only; not implemented, CEO decision)

Rule 5 puts one digit in each 8-bit channel. Packing several base-17 digits into a bit field keeps the planar tier streams but gives up the one-digit-per-channel byte layout. `gf17 density` computes the sizes from the real plane geometry (padded page rows, every plane packed on its own):

| packing (bits per digit) | v18 | v19d |
|---|---|---|
| rule 5: 1 digit / 8-bit channel (8.00) | 662.7 MB, 2.000× | 945.8 MB, 2.000× |
| 2 digits / 9 bits (4.50) | 372.8 MB, 1.125× | 532.0 MB, 1.125× |
| 3 digits / 13 bits (4.33) | 359.0 MB, 1.083× | 512.3 MB, 1.083× |
| **7 digits / 29 bits** (4.14; 17^7 = 410,338,673 ≤ 2^29) | **343.2 MB, 1.036×** | **489.8 MB, 1.036×** |
| 7/29, plus d3 as 8 base-14 digits / 31 bits (d3 ≤ 13) | 337.6 MB, 1.019× | 481.9 MB, 1.019× |
| 41 digits / 168 bits (4.10) | 339.4 MB, 1.024× | 484.5 MB, 1.024× |
| information limit for planar base-17 digits (4.0875) | 338.6 MB, 1.022× | 483.3 MB, 1.022× |
| whole word (4 digits = the 16-bit key itself, not planar) | 331.4 MB, 1.000× | 472.9 MB, 1.000× |
| entropy-coded (this lane, tile 128), for comparison | 143.0 MB, 0.431× | 182.6 MB, 0.386× |

- **Fixed rate cannot reach 0.8×.** Every fixed-rate packing stays at or above about 1.02× raw. A planar digit costs at least log2 17 bits, and 4 × 4.0875 > 16. Only entropy coding gets below the raw size.
- **7/29 halves the uncoded file** (0.52× of rule 5), and a digit is still addressable in O(1) with no decode tables. The cost is a 29-bit unpack instead of a byte load.
- **The rule-5 question for the CEO.** Is a packed or coded *on-disk* digit plane allowed, as long as the logical page (and the GPU-uploaded texture) is still one digit per RGBA channel? Coded mode already relies on that reading: `rgba()` rebuilds the exact pixel view. If rule 5 must also hold on disk, only raw mode complies.

## Raw-mode results (2026-09-25, lane S; 32-core box at load average 60 to 70 from other jobs, NVMe Fanxiang S910MAX)

### Roundtrip and size

Every table is bit-exact under both keys, and `cos = 1.0` (f64, printed to 12 digits) on `sw` and `dp`.

| pack | raw `.npy` | `.gf17` | coarse stream (exact tables + d3) | digit-entropy bound |
|---|---|---|---|---|
| `rayhash_clean1v2_base_v18_s100k` (L=28) | 331,351,552 B | 662,731,019 B (2.000×) | 297,811,968 B (0.899×) | 216.7 MB (0.654×) |
| `rayhash_v19d_clean_chat4c_s60k` (L=43) | 472,909,312 B | 945,851,819 B (2.000×) | 439,369,728 B (0.929×) | 301.5 MB (0.638×) |

- **Why 2×.** One base-17 digit per 8-bit channel is the cost of rule 5, and the file carries 4 bytes per 16-bit word.
- **The coarse stream** is mostly `sid`, because coordinates stay exact. The d3 planes alone are 0.25× of `sw` and `dp`.
- **Entropy.** The per-plane digit entropy (`raw` key, v18) is:

  | table | d3 | d2 | d1 | d0 |
  |---|---|---|---|---|
  | sw | 1.27 | 3.16 | 3.32 | 3.36 |
  | dp | 1.11 | 1.41 | 1.42 | 1.42 |
  | sid | 2.14 | 3.40 | 3.74 | 3.78 |

  All values are bits per digit, out of a maximum of log2 17 = 4.09. Coded mode (above) goes below this per-plane order-0 bound.
- **Decode speed** (unpack, warm) is 1.5 to 5.5 GB/s of page bytes.

### Tier truncation

These numbers characterise truncation only; the full tiers are lossless. The logits come from the `adam-ray` CPU engine on 5 texts (531 positions, chat and prose). The truncated `sw` and `dp` were decoded by this crate, while `sid` and `bias` stayed exact. Each value is max |Δlogit| / mean |Δlogit| / argmax agreement.

| pack, key | d3 | d3+d2 | d3+d2+d1 | all 4 |
|---|---|---|---|---|
| v18, raw | 36.6 / 4.83 / 59.5% | 7.2 / 0.47 / 87.8% | 11.0 / 0.038 / 96.0% | 0 / 0 / 100% |
| v18, ord | 54.4 / 14.9 / 42.7% | 12.5 / 0.43 / 88.3% | 12.2 / 0.038 / 94.7% | 0 / 0 / 100% |
| v19d, raw | 47.5 / 3.46 / 60.3% | 9.7 / 0.87 / 94.9% | 7.4 / 0.053 / 97.7% | 0 / 0 / 100% |
| v19d, ord | 63.7 / 14.3 / 64.2% | 9.1 / 1.03 / 94.9% | 6.8 / 0.053 / 97.7% | 0 / 0 / 100% |

- **Value error.** The median relative error of each weight is about 0.8 at d3, 0.03 to 0.07 at d3+d2 and 0.002 to 0.004 at d3+d2+d1.
- **`raw` NaNs.** A truncated `raw` value can land on NaN, and those values were zeroed:

  | pack | d3 | d3+d2+d1 |
  |---|---|---|
  | v18 | 463,562 values | 35,545 values (mostly −0.0 → NaN) |
  | v19d | 659,538 values | 81,608 values |

  `ord` never needs this.
- **Where the error comes from, `sw` versus `dp`:**
  - Truncating `sw` alone gives a small mean error: 0.004 to 0.008 at 3 tiers, 0.06 to 0.2 at d3.
  - The max error of `sw` stays around 7 to 12 even at 3 tiers. The engine ternarises each cell against 0.5·max and 0.75·max of its 3 slots, so a small weight error flips a trit.
  - `dp` dominates the mean error at d3, and reaches 99.4 to 100% argmax agreement at 3 tiers.
- **Takeaway.** d3 alone is a placeholder, not a usable model. d3+d2 is a usable preview, at about 88 to 95% argmax agreement. Answers need all 4 tiers.

### Fetch latency and throughput

- "Cold" means `cold()` (the mapping and page cache dropped) before every fetch, or before every batch of 64.
- "Warm" means the page cache is warm. With a 256 MB budget smaller than the working set, the LRU still evicts, so the warm path still takes a minor fault and a `WILLNEED`. With a 2 GB budget, every fetch is an LRU hit.
- Every fetched value matched the source bit for bit.

| store | budget | cold single p50 / p99 | warm single p50 / p99 | cold batch-64 p50 | warm batch-64 p50 / p99 |
|---|---|---|---|---|---|
| v19d, 0.95 GB | 256 MB | 123 µs / 897 µs | 12.4 µs / 25.8 µs | 2.49 ms | 7.8 µs / 13.6 µs |
| v19d, 0.95 GB | 2 GB | 121 µs / 1.8 ms | **0.5 µs / 1.2 µs** | 2.79 ms | 19 µs / 41 µs |
| v18, 0.66 GB | 256 MB | 124 µs / 1.5 ms | 7.4 µs / 30.2 µs | 2.55 ms | 12.8 µs / 31.7 µs |

Sequential decode, in GB/s of page bytes read:

| store | full tiers, cold | full tiers, warm | coarse (d3 + exact tables), cold |
|---|---|---|---|
| v18 | 2.10 | 4.85 | 0.92 |
| v19d | 1.09 | 2.64 | 0.82 |

The coarse pass is bound by decode, because it still writes every word.

### Scale test

- **Synthetic table.** `synth` wrote a 38.65 GB page file with the checkpoint layout: 32 levels × 2^26 cells × 3 slots, u8 `sid` plus f16 `sw`, random values from a counter hash. That is 19.3 GB of raw tables and 2.4× the 16 GB VRAM on disk.
- **Write.** It took 41.3 s (0.94 GB/s), with `fdatasync` and `DONTNEED` per plane so that it did not flood the page cache. The file was deleted after the run. 40 GB was the disk cap; the 10× target (160 GB) takes the same path, and only the tile arrays grow, to about 25 MB.
- **Checks.** Every fetched value matched the generator bit for bit.

| budget | fetches | cold p50 / p99 | warm p50 / p99 | VmRSS (file-backed) | VmHWM |
|---|---|---|---|---|---|
| 256 MB | 5,000 × 4 runs | 85 to 187 µs / 1.7 to 2.7 ms | 14.7 to 22 µs / 27.5 to 51.8 µs (one run 173 µs) | 32 MB (20 MB) | 32 MB |
| 64 MB | 2,000 | 103 µs / 3.0 ms | 23 µs / 63 µs | 19 MB (7 MB) | 19 MB |
| 2 GB | 2,000 × 3 runs | 89 µs / 2.4 ms | 1.0 to 1.2 µs / 2 to 3 µs (see the note below) | 77 MB (65 MB) | 77 MB |

- **The outlier.** One of the three 2 GB runs measured a warm p50 of 1.16 ms. The page cache had lost those pages between the warm-up pass and the timed pass, on a box that was 13 GB into swap. Pinning hot tiles with `mlock` would prevent this, but `RLIMIT_MEMLOCK` is 8 MB here.
- **Batch of 64 fetches:** cold p50 3.0 to 4.1 ms, warm p50 39 to 64 µs.
- **Sequential decode:** 1.05 GB/s cold and 3.8 GB/s warm, over 4.8 GB of `sw` planes.

## Integrating with adam-ray (after the R-lane merge)

1. **Load.** `pack::load` should accept `*.gf17` in either mode:
   - open a `Store` and `unpack(t, 4)?` into the existing `dv`/`sid`/`sw`/`bias` buffers;
   - add `adam-gf17 = { path = "../adam-gf17", default-features = false }`.

   The decode is bit-exact, so every parity test holds unchanged.
2. **Cells.** Factor the per-cell closure in `pack::load` (3 ids plus 3 sw, into the `[u32; 2]` trit/level cell) into a `pub fn cell(sid: [u8; 3], sw: [u16; 3]) -> [u32; 2]`. Both the resident path and the streamed path would then build cells the same way.
3. **CPU streaming.**
   - Behind a `CellSource` in `cpu.rs`, `Cpu::run` gathers `cells[l * t + e]` from either the resident `Vec` or a `Store`.
   - The streamed version plans one step's taps first, then issues one `fetch_many` per step, which covers ≤ 43 levels × 3 slots.
   - The warm cost is about 8 to 60 µs per step.
4. **GPU streaming.**
   - Replace the resident `Rg32Uint` cell texture with a page-table texture: (level, page) maps to a resident layer, or to "absent → zeros", which is rule 4.
   - Keep an `R8Uint` digit-plane array per tier, and decode `R + 17G + 289B + 4913A` in WGSL, with MID fill for tiers that are not resident yet.
   - Upload order is the stream order: coordinates plus d3, then d2, then d1 and d0 before answering.
   - The ternariser can move into the shader. Alternatively, store the engine cell form as a derived 4-word table in the same file, so the shader decodes it once.
5. **Trainer.** `adam-train` can emit `.gf17` next to the `.npy` checkpoints: call `write(out, &[(Table, bytes)], 128)` on the in-memory tables after each save.
   - Coded packing runs at about 2 to 4 s per checkpoint with 6 threads.
   - The GPU path would upload decoded tiles (`dtile` / `pdig`) into the per-tier `R8Uint` arrays, because the shader never sees the coded bytes.
6. **Tidy-up.** Add `/adam_gf17*.so` to `.gitignore`, and add an `adam-gf17` row to `crates/README.md`.
