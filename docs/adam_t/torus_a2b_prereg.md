# TORUS-A2b pre-registration (written 2026-09-30, before any A2b data build, dev run or scoring)

Re-run of the Stage A gate A2 (relational torus, Layer 2) from `docs/adam_t/torus_space_proposal.md`. The A2 test split of `data/torus/a2_data.npz` is never scored again. All A2b outputs go to `data/torus/a2b/`.

## Data (fresh seed 2930)
- Scored relations (the same 10): `wn:hypernym`, `wn:part_holonym`, `wn:member_holonym`, `cn:UsedFor`, `cn:AtLocation`, `cn:CapableOf`, `wd:P19`, `wd:P17`, `wd:P131`, `wd:P171`. Same sources and node merging as A2 (`torus_data.py`).
- Auxiliary relations, training only, never scored: `wd:P27` (citizenship), `wd:P20` (place of death), `wd:P106` (occupation), for the sampled persons. They exist so a held-out birth place is inferable from other facts about the person.
- Wikidata subgraph (seed 2930): 150,000 persons sampled from P19 subjects that also have at least one P27, P20 or P106 fact; all their P19, P27, P20, P106 facts; P131 facts of their P19 and P20 places and of those regions (2 levels); P17 facts of every place in that closure; 100,000 sampled P171 subjects with their P171 facts and their parents' P171 facts.
- Training degree of an entity = number of distinct training facts (all 13 relations, both roles) that touch it. The sphere graph and other side information do not count.
- Split (seed 2930): all scored facts in one global random order. A fact is eligible for held-out if h != t, both h and t have training degree >= 3 at that moment (so >= 2 after removal), and t appears as a training tail of the same relation at least twice (so it stays a range-constrained candidate). An eligible fact goes to whichever of test or dev has fewer facts of that relation (ties to test) while that split is under its cap, ceil(10%) of the relation's facts each; otherwise it stays in training. Auxiliary facts are always training.
- So every dev and test entity has at least 2 training facts, and every held-out tail is a candidate. Achieved counts per relation are reported; relations that cannot reach 10% under the rule are reported as such, not refilled.

## No-leak sphere
- Rebuilt with the unmodified `scripts/concept_sphere.py` (paths redirected by `sphere_noleak.py`) from the WordNet graph minus every dev and test pair (any relation, both directions) whose endpoints are sphere nodes. Output `data/torus/a2b/sphere_noleak.npz`. Its positions, 64/256/1024 cells and tangent links are the only sphere data the torus uses.
- Co-occurrence corpus edges stay as side information (as in A2). The solver's sphere-graph "related" edges exclude every dev and test pair. A tangent link is dropped if any node in its target cell forms a dev or test pair with its source.
- Leak check before any lever: a lift-only dev run (no solve sweeps). If any relation scores dev AUC above 0.95 from the lift alone, the cause is found and fixed before continuing.

## Evaluation (unchanged from A2 except where stated)
- Up to 2,000 seeded queries per relation per split, tail prediction (h, r, ?). Candidates = training tails of r; filtered ranking (other known tails of (h, r) in train/dev/test removed); ties count as expected rank. Baseline = relation frequency (count as a training tail of r).
- AUC: one negative per query, a tail drawn uniformly from the candidates minus known true tails; Mann-Whitney, pooled over relations.
- P(related): the per-relation calibration LUT (16 buckets at score quantiles, PAV-monotone, Laplace-smoothed fraction of positives, stored as an integer out of 289 = two GF(17) digits). Calibrated against the 1:1 positive:negative evaluation distribution.
- ECE: 10 equal-width bins of P, |mean P - fraction positive| weighted by bin size, over the scored pairs (positives and their negatives).

## Gates (read once, on test)
- Pooled hits@10 >= 2 x pooled frequency-baseline hits@10.
- Pooled AUC of P(related) >= 0.90 (raw-score AUC also reported).
- Pooled ECE of P(related) <= 0.05; per-relation ECE reported against the same 0.05.
- Reported, not gated: per-relation hits@10, baseline, ratio, AUC, ECE; recall@20 of the A2b torus against the A2b no-leak sphere on both sphere-graph protocols (current 321k-edge graph and C 1.14M-edge graph, with the WordNet and hypernym sub-protocols, rng seed 8 as in A2).

## Model selection (dev only)
- Train on the training split only. Dev queries choose everything. The starting point is the frozen A2 config: D=64, sharp distance LUT, per-dim polynomial LUT transforms, cavity weights, 20 identity warm-up sweeps then 40 more, rho 0.5, lam 24, popularity digit weight 12, solver seed 17.
- Levers, tried in this order, each kept only if it wins on dev:
  1. Multi-bounce rays: head -> tangent neighbour cells (`--bounce tan`) and/or training-graph neighbours (`nbr`) -> relation transform -> landing; the bounce hits add an integer term to the direct ray score.
  2. Dimension-mixing transforms: each output coordinate is a GF(17) LUT of 2 (`--mix 2`, 17^2 table) or 3 (`--mix 3`, 17^3 table) input coordinates, with backoff to the per-dim marginal for unseen input combinations.
  3. Abstraction levels: coarse-to-fine cell evidence from the sphere's 64/256/1024 cells (`--lvl`), as a per-relation integer LUT.
  4. Per-relation calibration LUT for P(related) (`--calib 16`), compared with the A2 LUT-exp softmax.
- LUTs fitted on held-out facts (the level LUT and the calibration LUT) are 2-fold cross-fitted inside dev for the ablation (fit on one half of each relation's dev queries, scored on the other). The final model fits them on the whole dev split.
- Win rule: a lever is kept if it raises dev pooled calibrated AUC by >= 0.005 without lowering the dev hits@10 ratio by more than 0.05, or raises the hits ratio by >= 0.05 without lowering calibrated AUC by more than 0.005. Each lever's small grid is listed in the checklist before it runs.

## Final run
- The chosen config trains on the training split, fits the dev LUTs on the whole dev split, and scores test exactly once. Results go to `docs/adam_t/torus_a2b_results.md` and `data/torus/a2b/`.
- Layer-2 positions, transforms and LUTs are packed into 4096x4096 RGBA GF(17) PTEX pages with the adam-gf17 17-bit Reffelt pixel codec; the roundtrip must be bit-exact. Storage is reported in digits per entity and projected to the 1T / 128 GB target.

## Amendment 1 (2026-09-30, written after the lever-1/2/3 dev runs, before any test scoring)
- Finding on dev: for `cn:UsedFor`, `cn:CapableOf` and `wd:P171` the torus ranks below the plain frequency baseline, because the popularity digit has one global weight (12). A 2-hop reachability check on dev explains why: held-out pairs reachable by any 2-hop training path are 15% (hypernym), 9% (member_holonym), 1% (P171) vs 68-87% for P19/P17/P131; the ray has little to work with on those relations.
- Lever 5 (added, dev only): per-relation popularity weight `--pw` (an integer per relation on the popularity digit, part of the stored Layer-2 scoring LUT). Grid {12, 24, 48, 96, 192} run as uniform eval-only passes on the chosen model; per relation, the weight maximising dev hits@10 + dev raw AUC is kept. This selection is in-sample on dev (5 values per relation), so its dev row is optimistic; the test split stays the only honest read. The lever is kept only under the same win rule.
- Nothing else changes: gates, data, splits, test protocol.
