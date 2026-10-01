# TORUS-A2b results (A2 relational torus, re-run), 2026-09-30

Protocol: `docs/adam_t/torus_a2b_prereg.md`. It was committed before any data build (bbd7190). Amendment 1 and the frozen config were committed before the test run (23a6aa7, 317ead0). Test was scored once. Everything ran on CPU only, in `systemd-run --user --scope -p MemoryMax=10G -p CPUQuota=300% nice -n 15`. Outputs are in `data/torus/a2b/`.

## Verdict: A2 PASS on the A2b test split, with thin margins

| gate | test | threshold | result |
|---|---|---|---|
| pooled hits@10 / frequency baseline | 0.2357 / 0.1130 = **2.085x** | >= 2x | PASS |
| pooled AUC of P(related) | **0.9073** | >= 0.90 | PASS |
| pooled ECE of P(related) | **0.0056** | <= 0.05 | PASS |
| per-relation ECE | 0.006-0.028 | <= 0.05 | PASS (10/10) |

Read this with four caveats:
- **Thin margins.** The margins are about 3 standard errors (ratio SE ~0.03, AUC SE ~0.003), from a single solver seed.
- **The popularity prior carries the pass.** It passes because of the per-relation popularity weight (amendment 1). The relational ray alone (mix 3 + calibration) read 1.91x and AUC 0.851 on dev, which fails both gates. On the weak relations the popularity prior does the work, not the torus.
- **AUC negatives favour popularity.** The AUC uses uniform random negatives from the candidate set. That favours any popularity prior: true tails are popular, random candidates mostly are not. Popularity-matched negatives would read lower.
- **A2b is an easier task than A2.** It is fully transductive, every held-out entity keeps at least 2 training facts, and persons get auxiliary facts. A2 and A2b numbers are not comparable.

## Data (seed 2930)
- 615,156 entities; 1,134,174 facts in 13 relations: the 10 scored ones plus P27, P20 and P106, which are training-only for the persons; 1,021,163 training facts.
- Held-out sets: 56,506 test and 56,505 dev. Every test and dev entity keeps at least 2 training facts, and every held-out tail is a training tail of its relation.
- 10% held out per relation, except two that ran out of eligible facts under the rule: `cn:CapableOf` reached 4.6% (1,052) and `wd:P171` 5.5% (7,091).
- Scored: 17,192 queries per split, 100% transductive and 100% inside the candidate set. In A2, 42% of the test queries had an unseen head or tail.
- No-leak sphere `a2b/sphere_noleak.npz`: rebuilt from the WordNet graph minus 47,708 edges (28,986 held-out pairs).
  - The solver's related edges exclude every dev and test pair.
  - 430 of the 259,480 tangent links pointed into a held-out pair and were dropped.
- Leak check (lift only, no solve): the maximum relation AUC was 0.928, for P17, from the popularity digit; hypernym was 0.573. No leak.

## Dev ablation (train split only; 17,192 dev queries; calibration cross-fitted 2-fold inside dev)

| config | hits@10 | ratio | AUC raw | AUC P(related) | ECE |
|---|---|---|---|---|---|
| lift only (no solve) | 0.0636 | 0.573 | 0.6797 | 0.7502 | 0.0070 |
| A2 frozen config (base) | 0.2017 | 1.819 | 0.7945 | 0.8485 | 0.0066 |
| **1 bounce** tan, bw 1 / 2 / 4 | 0.1981 / 0.1948 / 0.1896 | 1.786 / 1.756 / 1.710 | 0.797 / 0.796 / 0.793 | 0.8451 / 0.8419 / 0.8371 | 0.006-0.008 |
| bounce nbr, bw 1 / 2 / 4 | 0.2004 / 0.2001 / 0.1987 | 1.807 / 1.804 / 1.792 | 0.792 / 0.789 / 0.785 | 0.8465 / 0.8446 / 0.8423 | 0.004-0.007 |
| bounce tan+nbr, bw 1 | 0.2010 | 1.812 | 0.7947 | 0.8463 | 0.0064 |
| **2 mix** 2 (17^2 LUT per output dim) | 0.2068 | 1.865 | 0.7981 | 0.8486 | 0.0045 |
| mix 3 (17^3 LUT per output dim), **kept** | 0.2118 | 1.910 | 0.8022 | 0.8510 | 0.0038 |
| **3 levels** on base, lw 2 / 4 / 8 | 0.2018 / 0.2020 / 0.2022 | 1.819 / 1.821 / 1.823 | 0.795 | 0.8490 / 0.8492 / 0.8499 | 0.005-0.006 |
| levels on mix 3, lw 2 / 4 / 8 | 0.2122 / 0.2123 / 0.2124 | 1.913 / 1.914 / 1.915 | 0.803 | 0.8513 / 0.8517 / 0.8521 | 0.003-0.004 |
| **4 calibration** off (A2 LUT-exp softmax, mix 3) | 0.2118 | 1.910 | 0.8022 | 0.7737 | 0.4540 |
| calibration on (16-bucket PAV LUT), **kept** | 0.2118 | 1.910 | 0.8022 | 0.8510 | 0.0038 |
| **5 popularity weight** uniform 24 / 48 / 96 / 192 (mix 3) | 0.2217 / 0.2212 / 0.2005 / 0.1767 | 1.999 / 1.994 / 1.807 / 1.593 | 0.823 / 0.844 / 0.860 / 0.868 | 0.8649 / 0.8808 / 0.8934 / 0.8980 | 0.004-0.006 |
| per relation `192,96,96,192,192,192,48,12,24,192`, **kept, final** | 0.2321 | 2.092 | 0.8647 | 0.9071 | 0.0056 |

What each idea added:
- **Multi-bounce rays** (tangent cells, training neighbours, both) lost at every weight: -0.01 to -0.11 ratio and -0.002 to -0.011 AUC. Averaging landings from neighbours blurs the direct ray, because ICM already pulls related entities' positions together, so a neighbour's landing adds noise rather than evidence. A diagnostic explains the ceiling:

  | relation | held-out pairs reachable by any 2-hop training path |
  |---|---|
  | hypernym | 15% |
  | member_holonym | 9% |
  | P171 | 1% (removing a taxon's parent link disconnects its subtree) |
  | UsedFor | 25% |
  | CapableOf | 20% |
  | P19 / P17 / P131 | 68% / 87% / 73% |

- **Dimension mixing:**
  - Mix 3 gives +0.09 ratio and +0.0025 AUC; mix 2 gives +0.046, under the +0.05 bar.
  - The gain is mostly in P131 (0.28 to 0.33), part_holonym (0.12 to 0.15) and AtLocation.
  - Cost: solve time goes from 318 s to 1,130 s on 3 cores, and transforms from 1,088 to 314,432 digits per relation.
- **Abstraction levels** (64/256/1024 cells, plus a learned coarse landing cell per relation): at most +0.0014 AUC. Held-out pairs aren't sphere neighbours, because the no-leak sphere never saw them.
- **Calibration** is what makes P(related) usable: +0.077 pooled AUC, and ECE falls from 0.454 to 0.004.
- **Per-relation popularity weight:** +0.18 ratio and +0.056 AUC. For UsedFor, CapableOf and P171 the ray was below the frequency baseline, and the prior fixes the ranking. The selection is in-sample on dev (5 values per relation), so its dev row is optimistic; test agrees closely (2.085 vs 2.092, 0.907 vs 0.907).

## Test, scored once (`res_test_final.json`)

| relation | n | hits@10 | baseline | ratio | AUC raw | AUC P | ECE |
|---|---|---|---|---|---|---|---|
| wn:hypernym | 2000 | 0.0540 | 0.0380 | 1.42 | 0.846 | 0.845 | 0.020 |
| wn:part_holonym | 910 | 0.1060 | 0.0549 | 1.93 | 0.829 | 0.827 | 0.022 |
| wn:member_holonym | 1230 | 0.1984 | 0.0976 | 2.03 | 0.864 | 0.862 | 0.022 |
| cn:UsedFor | 2000 | 0.1085 | 0.1017 | 1.07 | 0.780 | 0.784 | 0.013 |
| cn:AtLocation | 2000 | 0.0938 | 0.0830 | 1.13 | 0.938 | 0.936 | 0.015 |
| cn:CapableOf | 1052 | 0.0347 | 0.0369 | 0.94 | 0.698 | 0.701 | 0.028 |
| wd:P19 | 2000 | 0.2762 | 0.0760 | 3.63 | 0.965 | 0.963 | 0.013 |
| wd:P17 | 2000 | 0.8878 | 0.4745 | 1.87 | 0.983 | 0.982 | 0.012 |
| wd:P131 | 2000 | 0.3758 | 0.0360 | 10.44 | 0.978 | 0.976 | 0.006 |
| wd:P171 | 2000 | 0.0415 | 0.0580 | 0.72 | 0.809 | 0.807 | 0.016 |
| **pooled** | 17192 | **0.2357** | **0.1130** | **2.085** | 0.865 | **0.907** | **0.006** |

- Relations at or above 2x on their own: P19, P131 and member_holonym; part_holonym and P17 are close. P17's ceiling is 1/0.4745 = 2.1x.
- CapableOf and P171 stay below the frequency baseline.

## Recall@20 against the A2b no-leak sphere (sphere protocols, rng seed 8)

| model | current graph: recall@20 / wn / hyp | C graph: recall@20 / wn / hyp |
|---|---|---|
| A2b no-leak sphere | **0.471** / **0.457** / **0.678** | 0.1221 / **0.430** / **0.690** |
| A2b torus (final) | 0.305 / 0.363 / 0.407 | **0.127** / 0.360 / 0.399 |

The torus beats the sphere only on the C-graph recall@20; the sphere's own layout graph favours the sphere on the other columns. The torus is built for relation rays, not undirected neighbourhoods.

## Layer 2 in PTEX pages (`torus_a l2pack`)
- File `data/torus/a2b/layer2_test_final.gf17l2p` (25,348,597 B): 28 sections, 47,707,265 GF(17) digits. Each section is pixel-aligned. Logical 4096x4096 RGBA pages with one digit per channel; physically one 17-bit Reffelt pixel word (`adam_gf17::pack17`), 4.25 bits per digit. That is 11.9M pixels, 0.71 of one logical page.
- Roundtrip: 0 mismatches through the RGBA page view (`page17`), 0 in the per-section decode (`px17` + `lut17`), 0 on 1M random-access reads; header and data hashes check. Bit-exact.
- Sections:
  - **positions** X: 615,156 x 64;
  - **transforms** TF: 13 x 64 x 17^3;
  - **scoring LUTs:** distance LUT S, per-relation popularity weight pw, per-candidate popularity digits pop0-9 (candidate id 5 digits + 1 digit), calibration LUTs cal0-9 (15 edges x 4 digits + 16 probabilities x 2 digits);
  - **sphere hierarchy and tangents:** cell1024 (6 digits per sphere node), tan_cnt, tan_dst (6 digits per link), lift_wd.

## Storage cost
- Final config, used by scoring: positions 64 + transforms 6.64 + popularity digits 1.87 + LUTs ~0 = **72.5 digits per entity**, 308 bits or 38.5 bytes per entity. That is 43.7 digits per training fact.
- Including the sphere cells and tangents (packed, but not used by the final config): 77.55 digits per entity.
- **Projection to the 1T / 128 GB target**, at 4.25 bits per digit:
  - 128 GB holds 2.41e11 digits. That is 3.3e9 entities of Layer 2 alone at 72.5 digits per entity. All of Wikidata (11.9M entities) would take 0.46 GB.
  - Transforms scale with relations, not entities: 314k digits per relation at mix 3, so 10k relations would take 1.7 GB. Mix 2 would be 18.5k digits per relation, and per-dim maps 1,088.
  - 1T stored digits would need 531 GB. As in A1, "1T params in 128 GB" only works if the params are generated rather than stored. Layer 2 generates P(related) for any (h, r, t) from 72.5 digits per entity, so the implied pair table is many orders of magnitude larger than what is stored.

## Where it left the paradigm (declared)
- **Inference is integer only:** LUT transforms over 1-3 coordinates, a distance LUT, adds, an integer popularity weight, then a bucket search and an integer LUT for P (P = v/288, two GF(17) digits). No matmul, no float.
- **Build-time floats** (stored results are integer digits):
  - the sphere lift (atan2, QR on xyz) at init;
  - float64 histogram normalisation in the transform fit and the mean-field tables, rounded to integers before the ICM kernel;
  - the calibration fit (PAV) and the level-LUT fit (log2);
  - the evaluation metrics;
  - the no-leak sphere itself, an existing float layout script, plus its float recall protocol.
- **Not yet SSD-streamed:** the scorer reads in-RAM uint8 arrays, not the mmapped pages. Layer 2 is packed and verified, but reading it page by page at score time (like Layer 1's `Table::get`) is not wired yet.

## Code
- `crates/adam-torus/py/torus_data.py --a2b`: the A2b sampler (auxiliary relations, degree-constrained interleaved dev/test split).
- `sphere_noleak.py --a2b`: the no-leak sphere for any dataset.
- `torus_solve.py` (A2 path bit-identical at `--mix 1`; checked on `a2_data` against the old file):
  - `--data`, `--mode test`, `--load`;
  - `--mix 1|2|3`;
  - `--bounce tan,nbr --bk --bw`;
  - `--lvl`;
  - `--calib`;
  - `--pw`;
  - `--export`;
  - ECE;
  - `--sphere --sph`.
- `crates/adam-torus/src/bin/torus_a.rs l2pack`: packs the exported digit arrays into PTEX pages and verifies the roundtrip.

## Next steps
1. Score directly from the packed pages: a Rust Layer-2 reader on an mmap with `px17` LUT reads, checked to reproduce `res_test_final.json` exactly. Then wire Layer 1 and Layer 2 together for Stage B.
2. Robustness:
   - a second solver seed;
   - AUC with popularity-matched negatives, so the popularity prior can't carry it;
   - a cross-fitted popularity-weight selection.
3. The weak relations need input signal, not more ray capacity:
   - P171: give the torus the taxon name (a species name starts with its genus name) and P105 rank as coordinates;
   - hypernym and CapableOf: gloss and lemma tokens as extra relations.

   The 2-hop reachability check should go into every future pre-registration.
4. Mix 3 cost: restrict it to the relations where it gains (P131, holonyms, AtLocation) and keep per-dim maps elsewhere. That would cut both solve time and transform digits.
