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

## Amendment 1 (2026-10-05, written after the dev ablation and the dev controls, before any test scoring and before the final solves)
Everything below was measured on the dev split (solver seed 17, one dev check at seed 23) and the training split. No test quantity has been computed.

### What dev showed (why the protocol changes)
- Plain latent-relation EM (prereg B0 as written) from the sphere lift gives clusters that carry no relation labels: held-out NMI 0.03-0.05 at K = 8, 16, 32, 64, no better than k-modes on the unsolved lift (0.047). Train AUC 0.90, held-out matched AUC 0.57: the clusters memorise train edges.
- Oracle diagnostics (labels seeded into the assignment, dev only, never in a headline): translation prototypes with the true labels fixed re-derive the labels at train NMI 0.67 and held-out NMI 0.27 (7 of 10 relations at 2x prior); oracle-initialised EM that is then free to move settles at held-out NMI 0.21. The labelled partition also has a higher train ray fit (0.666) than EM from random (0.61), so the objective prefers relations but the search does not find them. Metadata (origin pair x degree bins) alone reaches held-out NMI 0.449, higher than anything geometry produced even with labels in the placement.
- Two levers close most of the generalisation gap and are added to the registered lever list: warm identity sweeps (`--warm`, A2b recipe) and sibling edges (`--sib`: pairs of training heads that share a tail, and pairs of tails that share a head, as neutral zero-ray pulls, built from training facts only, held-out pairs removed; a label-free second-order neighbourhood prior). Also tried: balance auction (`--bal`), cavity weight `--lam`, update rate `--rho`, more sweeps, LUT instead of translation prototypes (`--tm lut`).
- The A2b labeled model scored ray-only (relation given, no popularity weight) under the same popularity-matched negatives reaches dev matched AUC 0.743 and hits@10 0.586. The CTO's example of AUC >= 0.85 with matched negatives therefore sits above what the labeled approach itself reaches. G2 is kept as written (an honest fail is informative), and a parity read G2' is added.

### Dev controls measured (reference configuration below)
| control | held-out NMI | matched AUC |
|---|---|---|
| shuffled positions (same prototypes) | 0.034 | 0.506 |
| random rays | 0.001 | 0.501 |
| k-modes on the unsolved lift | 0.047 | 0.497 |
| metadata clustering (origin pair x 3x3 degree bins) | 0.449 | n/a |
| popularity only, matched negatives / uniform negatives | n/a | 0.504 / 0.900 |
| A2b labeled, ray only, matched negatives | n/a | 0.743 |
| oracle-seeded EM, fixed labels (diagnostic) | 0.272 | 0.676 |

### Frozen reference configuration R (chosen by the one-factor-at-a-time win rule on dev; the table is in the results doc)
`torus_solve.py --unlab 32 --tm trans --S sharp --lam 24 --rho 500 --warm 10 --em 2 --sweeps 50 --sib 1 --ge 1 --tan 0 --bal 0 --D 64` (K = 32 translation prototypes, 8 tori of 8 dims, sibling edges and sphere graph edges as neutral pulls, no tangent edges, train-only solve), headline clusterer M1 (the EM prototypes), density LUT B = 8, positions D = 64. No variant beat R under the win rule: K 16/64, ge off, tangent edges, D 16/32, M2 and M3, rho 300, lam 12, 20 warm sweeps, LUT transforms, balance auction. Dev seed noise (R at seed 17 vs 23): NMI +-0.005, matched AUC +-0.014, so differences below 0.01 NMI and 0.015 AUC are inside the noise.

### Gates as they are read on test (replaces the Gates section for the numbers marked amended)
- G1 (amended): held-out NMI >= 0.15 AND >= 3x the shuffled control of the same model. The 1.25x metadata clause is demoted to a reported comparison: not even the oracle-seeded placement reaches it on dev (0.27 vs 0.449), so as a gate it would only measure the metadata, not the geometry. The 0.15 floor is 70% of the oracle-seeded EM level (0.21), set before any B result at test.
- G2 (unchanged): pooled matched-negative AUC >= 0.85 AND hits@10 (1 vs 99 matched) >= 0.30. G2' (added, read next to it): AUC >= 0.69 (labeled ray-only dev 0.743 minus 0.05) AND hits@10 >= 0.30.
- G3 (unchanged): recall_r >= 2 x pi_r on at least 6 of 10 relations, with the majority-label table. The balanced readout (cluster labelled by the relation that has the largest share of its edges there) is reported next to it.
- G4 (amended): composition is gated on the hop-1-held-out variant (hop 1 is a held-out edge of the evaluated split and its cluster is assigned by geometry; hop 2 is a training edge), pooled hits@10 >= 2x the frequency baseline. Reason: with both hops stored, the true rays telescope, and the stored-hops variant is reported as informational only.
- Pass/fail is read for each seed (17 and 23) separately; a gate passes overall only if it passes on both.
- The thesis reading is unchanged: supported only if G1, G2 and G3 pass on both seeds.

### Final procedure (frozen)
Solve R from scratch on the training split for seeds 17 and 23 (tags `final_s17`, `final_s23`), then for each seed run `torus_b.py --split test --final 1 --clu m1 --ctl 1 --novel 1 --comp 1 --a2b 1 --export 1` once (a lock file refuses a second scoring). M2, M3 and the tori ablation are dev-only.
