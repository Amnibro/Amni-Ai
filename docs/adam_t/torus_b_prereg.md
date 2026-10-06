# TORUS-B pre-registration: unlabeled-ray relation discovery (written 2026-10-05, before any B solve, dev run or scoring)

Spec lineage: `docs/adam_t/torus_space_proposal.md` (ray gives P(related)), TORUS-A and TORUS-A2b (`torus_a_results.md`, `torus_a2b_results.md`). All TORUS-B outputs go to `data/torus/b/`.

## Why this experiment exists
CEO correction, 2026-10-05: "Rays were supposed to define concept relations." In A2/A2b one polynomial or LUT transform was fitted per labeled relation, and the ray only applied it. A2b passed only through a per-relation popularity weight (ray-only: 1.91x hits@10, AUC 0.851 on dev). So the labels defined the relations, not the geometry. TORUS-B tests the CEO's intent directly: the ray between two concepts, d = (pos(b) - pos(a)) mod 17 in every dimension of Z17^D, is the relation; relation types are clusters of rays found without labels; the density of rays gives P(related).

## Questions
- B0 placement: can positions be solved from a relatedness graph with relation type erased (only "related" pairs against sampled non-edges), by integer relaxation, no backprop?
- B1 ray = relation: (1) do unlabeled ray clusters recover the true relation types (NMI, purity)? (2) does the ray-space density predict held-out edges against popularity-matched negatives, with no popularity weight? (3) can a pair's relation type be read from its ray alone? (4) do clusters exist that match no known label (novel structure)? (5) do rays encode direction and inverses? (6) do ray sums compose (two-hop facts)?
- Side questions: what did labels buy in A2b and what does geometry-only cost? Do multiple tori and the tangent links help?

## Data (frozen: the A2b dataset and split, nothing re-sampled)
- `data/torus/a2b/a2b_data.npz`: 615,156 entities, 1,134,174 facts in 13 relations (10 scored: wn:hypernym, wn:part_holonym, wn:member_holonym, cn:UsedFor, cn:AtLocation, cn:CapableOf, wd:P19, wd:P17, wd:P131, wd:P171; 3 training-only auxiliary: wd:P27, wd:P20, wd:P106). Split column: 0 train (1,021,163 facts), 1 dev (56,505), 2 test (56,506).
- Reusing the split is deliberate so B is comparable with A2b on identical held-out pairs. Disclosure: the A2b test numbers were already public when this was written. B's metrics differ from A2b's, no B quantity has been computed on test, and the test set is scored once per solver seed with frozen code (a lock file refuses a second test scoring of the same seed).
- Placement input graph ("relatedness graph"): all 1,021,163 training facts pooled, relation id erased. The stored orientation (head to tail) is kept, because a ray needs a start and an end. A dev control with random orientation measures what the orientation gives (`--orient 1`).
- Side information as in A2b, treated as dev levers: (a) the sphere graph "related" edges (735k, undirected, neutral, dev and test pairs removed) as pulls with a zero ray (`--ge`); (b) the no-leak concept-sphere lift as the starting positions (always on; the live `sphere.npz` is never used; `a2b/sphere_noleak.npz` was rebuilt from the WordNet graph minus every dev and test pair); (c) sphere tangent links as extra neutral neighbourhood edges (`--tan`, links into a held-out pair dropped).
- Labels (the 13 relation ids) are used only to evaluate (NMI, purity, label table, composition chain selection). They never enter the solver, the clusterers or the density LUT. The label table of B1.3 is built from the train labels and is the only place train labels enter, and it is an output readout, not a placement input.
- Entity `src` (0 WordNet/sphere, 1 ConceptNet-only, 2 Wikidata) is a dataset-of-origin tag. It is used only to match negatives and in the metadata control, never by the solver.

## B0 placement (integer relaxation, no backprop)
- Positions X in Z17^D (D dims = D/8 tori of 8 dims). Start: sphere lift (A2b), random for entities without a sphere position.
- Latent relations: K ray prototypes c_k in Z17^D (translations). Training edge (a,b) is assigned to k* = argmax_k sum_d S[(x_a,d + c_k,d - x_b,d) mod 17] (S = the sharp distance LUT of A2b). Hard EM, all integer: E-step assigns edges, M-step refits c_k as the per-dimension circular mode with the A2b cavity penalty (existing `fitT`, `--tm trans`), then the A2b ICM sweep moves positions with the A2b cavity (popularity-neutral) term. Dead clusters are re-seeded from the worst-fit edges.
- Solver = `crates/adam-torus/py/torus_solve.py` extended (`--unlab K`), not forked.

## B1 ray clusterers (all integer, no float weights stored)
- M1 EM prototypes from the B0 solve (`k*` above).
- M2 post-hoc k-modes: integer k-modes over training rays (init by k-modes++ on a 30k sample, per-dimension circular mode update, S-LUT distance), independent of the B0 assignments, so it tests whether the rays alone carry the structure.
- M3 coarse hash address: per-dimension ray coarse-quantised to 3 cells ({|r|<=1}, positive side, negative side), hashed per torus; the K most populated addresses are the clusters, the rest fall into the nearest by Hamming distance (the "abstraction cell" address of the CEO's proposal).
- Ray signature over t tori (t in {1,2,4,8}, 8 dims each) as a dev ablation on one solve.
- Density LUT (P(related)): cell = (cluster k, bucket b of the in-cluster integer score; 8 buckets at train-positive quantiles). P(cell) = (n_pos + 1)/(n_pos + n_neg + 2) from 300,000 train positives and 300,000 popularity-matched train negatives, stored as an integer out of 288 (two GF(17) digits). No per-relation or popularity weight exists in the score.

## Metrics (everything below is computed on all held-out facts of the split's 10 scored relations: 56,505 dev, 56,506 test)
- **Popularity-matched negatives** (headline): popularity = distinct training facts touching the entity; bin = floor(log2(1 + deg)); stratum = (src(t), bin). For a held-out (h, t) the negative tail t' is uniform in t's stratum (widened to bin +-1 if fewer than 50 eligible), excluding t, h and any entity with a known fact to h in any split, either direction. Stricter than the CTO's spec because the origin tag stands in for entity type. A degree-only matched variant and a uniform variant (the A2b protocol) are reported next to it.
- **AUC**: one matched negative per held-out edge, pooled and per relation, Mann-Whitney with tie midranks, on P(related) (and the raw integer ray score alongside).
- **hits@10**: 1 true tail against 99 matched negatives (for a random 20,000-edge subset of the split, seed fixed), ties as expected rank. Chance 0.10.
- **Popularity-only baselines**: score = log(1 + deg(t)) and deg(h)*deg(t), same negatives, same AUC and hits@10. Under matched negatives these must sit near 0.5 and 0.10 (a sanity check of the matching); under uniform negatives they show the A2b prior.
- **NMI** = I(C;L)/(0.5(H(C)+H(L))) between cluster C and true relation L, on train edges (13 labels) and on held-out edges (10 labels, clusters assigned by the clusterer's nearest rule). **Purity** = sum_k max_l n_kl / n.
- **Relation inference**: label table = majority train label per cluster. Per held-out relation r, recall_r = share of its held-out edges predicted r. Prior baseline = the train class prior pi_r (a predictor that draws the label from the prior has recall pi_r). A stronger baseline, the majority label per (src(h), src(t)) pair, is reported next to it.
- **Controls** (same prototypes, label table and density LUT refit where noted): (S) shuffled positions: X permuted within each `src` stratum, density LUT and label table refit on the shuffled train rays; (R) random rays: uniform random ray vectors; (M) metadata clustering: cluster = (src(h), src(t), 3-level degree bin of h, 3-level degree bin of t) with up to 81 clusters, more than K, so it is generous; (L) lift only: clusterers fit on the unsolved lifted positions.
- **Novel structure**: among clusters with >= 500 train edges, those whose normalised train-label entropy (13 labels) is >= 0.5 or whose majority share is < 0.5 and whose in-cluster mean ray score is at least half the maximum possible. The ten largest are shown with ten uniformly random member pairs each (names), unfiltered. If fewer than ten qualify the largest remaining clusters complete the list and are marked.
- **Direction**: orientation accuracy per relation = share of held-out edges where P(ray(a->b)) beats P(ray(b->a)), ties 0.5 (chance 0.5). Inverse structure: for each prototype c_k the integer distance of -c_k to the nearest prototype, against the same distance for random vectors; reversed training edges (b->a) of each relation are assigned and reported.
- **Composition (A3 style)**: chains P19>P17, P19>P131, hyp>hyp, P171>P171 over stored hops (both hops in training, composed pair not stored in any split), up to 2,000 chains per chain type, seed fixed. The two hops' clusters k1, k2 are the assigned clusters (geometry only; labels only pick which chains to test). Predicted landing = x_a + c_k1 + c_k2; candidates = tails of training edges assigned to k2; score = sum_d S[...]; filtered hits@10 against the frequency baseline in the same candidate set. Note: the sum of the true rays of a path telescopes to x_c - x_a for any positions; the test therefore uses the cluster prototypes.
- **Labeled vs unlabeled**: the frozen A2b mix-3 model `torus_dev_mix3.npz` scored ray-only (no popularity weight, relation given) on the same held-out pairs with the same matched negatives, and with uniform negatives. Reported as a post-hoc comparison, not tuned.

## Gates (read on test, once per seed, for both seeds; all numbers fixed below before any model result)
- G1 relation recovery. Held-out NMI of the headline clusterer (clusterer and K chosen on dev) >= 3x the shuffled-position control (S) AND >= 1.25x the metadata clustering (M) AND >= the absolute floor NMI_floor.
- G2 relation-free prediction. Pooled AUC of P(related) with popularity-matched negatives >= 0.85, AND hits@10 (1 vs 99 matched) >= 0.30 (3x chance). Both with no popularity term.
- G3 relation inference. recall_r >= 2 x pi_r on at least 6 of the 10 relations.
- G4 composition. Pooled hits@10 over the four chains >= 2x the frequency baseline.
- Reported, not gated: purity, per-relation everything, novel clusters, direction, inverses, tori and tangent ablations, storage, A2b comparison.
- The thesis "unlabeled rays define relations" is read as supported only if G1, G2 and G3 pass for both seeds. G4 is reported as the A3-style composition read. A failed gate is reported as failed with a diagnosis.
- NMI_floor is a placeholder (0.15) until Amendment 1 sets it from the model-free controls on dev.

## Model selection (dev only, solver seed 17; every row goes in the ablation table)
- Order: (1) K in {16, 32, 64} at D = 64, ge = 1, tan = 0, warm 0; (2) ge in {0, 1}; (3) tan in {0, 1}; (4) D in {16, 32, 64}; (5) clusterer M1/M2/M3; (6) tori in the signature t in {1, 2, 4, 8}; (7) warm-up in {0, 10} identity sweeps. Controls S, R, M, L and `--orient 1` on the chosen model.
- Win rule (as A2b): a setting is kept if it raises dev held-out NMI by >= 0.01 without lowering dev AUC (matched, P) by more than 0.01, or raises AUC by >= 0.01 without lowering NMI by more than 0.01. Cost breaks ties (fewer digits first).
- No LUT is fitted on dev or test: the density LUT and the label table use training edges only, so the dev-mode solve is exactly the test-mode training set.
- Final: the chosen configuration is frozen, solved from scratch for seeds 17 and 23 on the training split, and each solve is scored on test once.

## Storage and PTEX
- Digits per entity (positions D), the prototype table (K x D), the density LUT (K x 8 buckets x 2 digits plus 7 thresholds x 3 digits per cluster), the label table (K digits). Packed with `torus_a l2pack` (adam-gf17 17-bit Reffelt pixel codec, 4096x4096 RGBA logical pages); the roundtrip must be bit-exact. Reported next to A2b's 72.5 digits per entity.

## Paradigm declaration
Inference is integer only (digit differences mod 17, distance LUT, adds, argmax, an integer LUT). Build-time floats are reported as in A2b (sphere lift, cavity histograms rounded to integers before the ICM kernel, NMI and AUC evaluation). Anything outside the paradigm is named in the results.
