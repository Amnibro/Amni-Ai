# TORUS-B results: unlabeled-ray relation discovery, 2026-10-05

Protocol: `docs/adam_t/torus_b_prereg.md` (pre-registration committed before any solve, Amendment 1 committed before the final solves and before any test scoring). Code: `crates/adam-torus/py/torus_solve.py` (extended: `--unlab K`, `--warm`, `--sib`, `--orient`, `--ge`, `--tan`, `--bal`, `--oracle`) and the new `crates/adam-torus/py/torus_b.py`. Outputs: `data/torus/b/`. CPU only, every heavy step in `systemd-run --user --scope -p MemoryMax=10G -p CPUQuota=300% nice -n 15`.

## Verdict: the thesis "unlabeled rays define relations" is NOT supported

Geometry-only placement works as a relatedness model but does not recover relation types. Test, scored once per solver seed (17 and 23), frozen code:

| gate | seed 17 | seed 23 | threshold | result |
|---|---|---|---|---|
| G1 held-out NMI of ray clusters vs true relations | 0.128 (3x shuffled 0.100, floor 0.15) | 0.131 (3x shuffled 0.102) | >= 0.15 and >= 3x shuffled | **FAIL** (clears 3x shuffled, misses the floor) |
| G2 matched-negative AUC (P(related), no popularity term) | 0.759 | 0.742 | >= 0.85 | **FAIL** |
| G2 hits@10, 1 true vs 99 matched negatives | 0.507 | 0.503 | >= 0.30 | pass |
| G2' parity read (AUC >= 0.69 and hits >= 0.30) | 0.759 / 0.507 | 0.742 / 0.503 | added in Amendment 1 | **PASS** |
| G3 relations with recall >= 2x prior (majority label table) | 2 of 10 | 1 of 10 | >= 6 | **FAIL** (balanced readout: 5 and 4) |
| G4 composition, hop-1 held out, pooled hits@10 / frequency baseline | 0.491x | 0.501x | >= 2x | **FAIL** |

Gate passes overall only if it passes on both seeds: G1, G2, G3, G4 fail; G2' (parity with the labeled approach) passes. The two seeds differ by 0.003 in NMI and 0.017 in AUC, and test agrees with dev (dev NMI 0.128, AUC 0.758).

What the data says in one paragraph: rays between solved positions encode "these two concepts are related" (matched AUC 0.76, hits@10 0.51 against a popularity baseline at chance, and on par with the labeled A2b ray-only model), but they do not encode which relation it is. 72% of all edges fall into one cluster of near-zero rays ("related means near"). The remaining clusters are destination regions (people to the United Kingdom, to the United States, to "association football player"), not relation types, so one cluster mixes citizenship, birthplace, death place and occupation. Origin plus degree metadata alone reads relations better (NMI 0.449) than the geometry does (0.13).

## What was built
- **B0 placement without relation labels** (`torus_solve.py --unlab 32`): positions in Z17^64 (8 tori of 8 dims), 1,021,163 training facts pooled with relation ids erased, stored head-to-tail orientation kept (a control with random orientation is below). Hard EM, all integer: (E) each edge goes to the translation prototype c_k with the best sum of the sharp distance LUT over dims of (x_a + c_k - x_b) mod 17 (a Taichi kernel over edges x K x D); (M) the A2b `fitT` refits prototypes as per-dimension circular modes with the cavity penalty; the A2b ICM sweep moves positions. Dead clusters are re-seeded from the worst-fit edges. Start: no-leak sphere lift (`a2b/sphere_noleak.npz`), then 10 identity warm-up sweeps (A2b recipe), sibling edges (`--sib`: pairs of training heads sharing a tail and pairs of tails sharing a head, neutral zero-ray pulls built from training facts only), 735k sphere-graph edges as neutral pulls. 50 sweeps, K = 32, about 7 minutes on 3 cores. Deterministic: the final solve at seed 17 reproduces the dev solve exactly.
- **B1 ray clusterers** (`torus_b.py`): M1 the EM prototypes (headline), M2 integer k-modes over training rays (k-modes++ init, circular-LUT distance), M3 coarse hash address (3 cells per dim over one torus, top-K addresses). Density LUT: cell = (cluster, 8 buckets of the in-cluster integer score), P = (pos + 1)/(pos + neg + 2) from 300k train positives and 300k popularity-matched train negatives, stored as an integer out of 288. No float weights, no popularity term.
- **Popularity-matched negatives**: for a held-out (h, t) the negative tail is drawn uniformly from the 101 entities of the same origin tag closest in training degree (random tie order), excluding t, h and every entity with a known fact to h in any split. The popularity-only baseline confirms the matching: AUC 0.504 and hits@10 0.103 (chance 0.100). Under uniform negatives (the A2b protocol) the same popularity baseline reads AUC 0.901.
- **Controls**: shuffled positions (permuted within origin strata, same prototypes), random rays, k-modes on the unsolved lift, metadata clustering, and a supervised naive-Bayes readout on rays (labels used only to measure how much relation information the rays hold).

## Dev ablation (training split only, 56,505 dev edges, solver seed 17 unless marked; tags are files in `data/torus/b/`)
Columns: NMI of ray clusters vs relation labels on held-out and on train edges, train ray fit (mean best-prototype score / maximum), share of the largest cluster, matched AUC of P(related) and of the raw ray score, hits@10 (1 vs 99 matched), stored-orientation accuracy (chance 0.5), number of relations with recall >= 2x prior.

| run | change | NMI held | NMI train | fit | max share | AUC P | AUC raw | hits@10 | orient | r>=2x |
|---|---|---|---|---|---|---|---|---|---|---|
| k32 | **base**: K 32, no warm-up, no sibling edges | 0.042 | 0.046 | 0.610 | 0.10 | 0.569 | 0.572 | 0.207 | 0.550 | 1 |
| k8 / k16 / k64 | K | 0.028 / 0.035 / 0.049 | 0.035 / 0.042 / 0.049 | 0.63 / 0.61 / 0.61 | 0.19 / 0.13 / 0.09 | 0.585 / 0.573 / 0.561 | 0.607 / 0.586 / 0.567 | 0.240 / 0.214 / 0.193 | 0.55 | 1 |
| k32g0 | sphere-graph edges off | 0.022 | 0.071 | 0.613 | 0.11 | 0.546 | 0.554 | 0.177 | 0.536 | 1 |
| k32 m2 / m3 | clusterer k-modes / coarse hash | 0.042 / 0.038 | 0.046 / 0.052 | 0.59 / 0.82 | 0.10 / 0.09 | 0.564 / 0.523 | 0.569 / 0.510 | 0.200 / 0.143 | 0.55 / 0.52 | 1 / 1 |
| k32w10 | + 10 identity warm-up sweeps | 0.091 | 0.064 | 0.781 | 0.54 | 0.750 | 0.729 | 0.488 | 0.537 | 1 |
| k32w20 | 20 warm-up sweeps | 0.117 | 0.076 | 0.783 | 0.68 | 0.755 | 0.725 | 0.491 | 0.535 | 2 |
| k32sib | + sibling edges (no warm-up) | 0.069 | 0.105 | 0.676 | 0.21 | 0.672 | 0.682 | 0.382 | 0.624 | 2 |
| **k32w10sib = R** | warm 10 + sibling edges (frozen reference, also `final_s17`) | **0.128** | 0.100 | 0.813 | 0.72 | **0.758** | 0.723 | 0.512 | 0.513 | 2 |
| r_s23 | R at seed 23 (seed noise) | 0.133 | 0.093 | 0.827 | 0.73 | 0.744 | 0.703 | 0.507 | 0.515 | 1 |
| b_k16 / b_k64 | R with K 16 / 64 | 0.119 / 0.143 | 0.086 / 0.111 | 0.81 / 0.82 | 0.74 / 0.63 | 0.749 / 0.745 | 0.724 / 0.705 | 0.511 / 0.512 | 0.52 | 1 / 1 |
| b_ge0 | R, sphere-graph edges off | 0.096 | 0.077 | 0.822 | 0.79 | 0.779 | 0.765 | 0.545 | 0.507 | 0 |
| b_tan | R + sphere tangent links as neutral edges | 0.122 | 0.098 | 0.816 | 0.68 | 0.739 | 0.711 | 0.499 | 0.516 | 2 |
| b_D16 / b_D32 | R with 16 / 32 dims (2 / 4 tori) | 0.097 / 0.123 | 0.074 / 0.087 | 0.81 / 0.82 | 0.53 / 0.71 | 0.747 / 0.737 | 0.732 / 0.711 | 0.513 / 0.497 | 0.53 / 0.52 | 1 / 2 |
| b_cap8 / b_cap32 | R + balance auction on cluster size | 0.127 / 0.134 | 0.094 / 0.098 | 0.81 / 0.81 | 0.72 / 0.75 | 0.756 / 0.757 | 0.724 / 0.722 | 0.515 / 0.509 | 0.52 | 2 / 2 |
| r_rho300 | R, update rate 0.3 | 0.135 | 0.090 | 0.801 | 0.69 | 0.761 | 0.726 | 0.525 | 0.524 | 1 |
| r_lam12 | R, cavity weight 12 | 0.152 | 0.102 | 0.864 | 0.75 | 0.724 | 0.683 | 0.493 | 0.524 | 4 |
| r_w20 | R, 20 warm-up sweeps | 0.128 | 0.093 | 0.822 | 0.74 | 0.745 | 0.714 | 0.512 | 0.521 | 1 |
| r_lut | R with per-dim LUT transforms instead of translations | 0.137 | 0.220 | 0.910 | 0.36 | 0.747 | 0.609 | 0.490 | 0.589 | 2 |
| R m2 (8 tori) / m3 | clusterer k-modes / coarse hash on R's positions | 0.148 / 0.092 | 0.125 / 0.086 | 0.82 / 0.92 | 0.78 / 0.48 | 0.746 / 0.734 | 0.703 / 0.633 | 0.516 / 0.480 | 0.52 | 3 / 1 |
| R, 1 / 2 / 4 tori in the ray signature (M1) | tori ablation (post-hoc, same solve) | 0.106 / 0.118 / 0.128 | 0.084 / 0.091 / 0.098 | 0.83 / 0.82 / 0.81 | 0.67 / 0.70 / 0.72 | 0.727 / 0.733 / 0.748 | 0.714 / 0.723 / 0.729 | 0.487 / 0.499 / 0.512 | 0.51 | 1 / 1 / 1 |
| b_orient | R with random head/tail orientation (control) | 0.139 | 0.087 | 0.789 | 0.73 | 0.777 | 0.752 | 0.548 | 0.504 | 2 |
| k32w10lam48, k32w10b8 | lam 48; superseded log-size balance penalty | 0.046 / 0.107 | 0.062 / 0.071 | 0.67 / 0.77 | 0.23 / 0.82 | 0.655 / 0.752 | 0.666 / 0.724 | 0.316 / 0.491 | 0.54 / 0.52 | 1 / 1 |
| c_* (K 64 + warm + sib) | sweeps 80, rho 300, lam 12, warm 20, LUT | 0.141 / 0.140 / 0.155 / 0.153 / 0.140 | | | | 0.744 / 0.757 / 0.723 / 0.748 / 0.732 | | | | |
| orc_fix | **oracle diagnostic**: labels seeded and held fixed (K 13) | 0.272 | 0.673 | 0.666 | 0.28 | 0.676 | 0.691 | 0.405 | 0.675 | 7 |
| orc_em | oracle-seeded EM, then free to move | 0.208 | 0.568 | 0.727 | 0.25 | 0.688 | 0.683 | 0.411 | 0.687 | 7 |

Win rule (A2b): a change is kept if it raises dev NMI by >= 0.01 without lowering matched AUC by more than 0.01, or raises AUC by >= 0.01 without lowering NMI by more than 0.01. Seed noise is NMI +-0.005 and AUC +-0.014.
- **Kept**: 10 warm-up sweeps (AUC 0.569 to 0.750, NMI 0.042 to 0.091); sibling edges on top (NMI 0.091 to 0.128, AUC 0.750 to 0.758). Those two make R.
- **Not kept** (all inside the noise or failing the rule): K 16/64, sphere-graph edges off (AUC +0.021 but NMI -0.032), tangent links, D 16/32, M2 (NMI +0.020 but AUC -0.012), M3, rho 300, lam 12 (NMI +0.024, AUC -0.034), 20 warm-up sweeps, LUT transforms (NMI +0.009), balance auction.
- Pre-registered question "do multiple tori help": modestly. NMI 0.106, 0.118, 0.128, 0.128 and AUC 0.727, 0.733, 0.748, 0.758 for 1, 2, 4, 8 tori in the signature; D 16 / 32 / 64 give NMI 0.097 / 0.123 / 0.128 and AUC 0.747 / 0.737 / 0.758. The gain saturates around 4 tori for NMI.
- Pre-registered question "do tangent links help": no. Tangent links as neutral neighbourhood edges lowered NMI (0.128 to 0.122) and AUC (0.758 to 0.739). The sphere's neighbourhood information is already covered by the lift, the sphere-graph edges and the sibling edges.

## Dev controls (R)
| control | held-out NMI | matched AUC |
|---|---|---|
| shuffled positions | 0.034 | 0.506 |
| random rays | 0.001 | 0.501 |
| k-modes on the unsolved lift | 0.047 | 0.497 |
| metadata clustering (origin pair x 3x3 degree bins, up to 81 clusters) | 0.449 | n/a |
| popularity only, matched / uniform negatives | n/a | 0.504 / 0.901 |
| supervised naive-Bayes readout on R's rays (diagnostic, uses labels) | 0.168 to 0.182 | n/a |

## Test results per seed (scored once, `resb_final_s17_m1_test.json`, `resb_final_s23_m1_test.json`)
| metric | seed 17 | seed 23 |
|---|---|---|
| held-out NMI / purity (clusters vs 10 relations) | 0.128 / 0.371 | 0.131 / 0.378 |
| train NMI (13 labels) | 0.100 | 0.093 |
| controls: shuffled / random rays / lift-only k-modes NMI | 0.033 / 0.001 / 0.048 | 0.034 / 0.001 / 0.057 |
| metadata NMI (report-only) | 0.448 | 0.448 |
| AUC of P(related), popularity-matched (headline) | 0.759 | 0.742 |
| AUC raw ray score, matched | 0.722 | 0.702 |
| AUC degree-matched only / uniform negatives | 0.822 / 0.855 | 0.804 / 0.843 |
| train-edge AUC (overfit gap) | 0.892 | 0.880 |
| popularity-only AUC: matched / degree-matched / uniform | 0.504 / 0.503 / 0.901 | 0.504 / 0.503 / 0.901 |
| hits@10 (1 vs 99 matched): ray / popularity-only | 0.507 / 0.103 | 0.503 / 0.104 |
| relation inference accuracy: majority table / balanced / origin-pair metadata / supervised NB readout | 0.083 / 0.072 / 0.213 / 0.148 | 0.074 / 0.051 / 0.213 / 0.139 |
| relations with recall >= 2x prior: majority / balanced / origin-pair / supervised NB | 2 / 5 / 4 / 6 | 1 / 4 / 4 / 6 |
| stored-orientation accuracy (chance 0.5) | 0.516 | 0.513 |
| composition, hop-1 held out, pooled hits@10 vs frequency baseline | 0.026 vs 0.054 (0.49x) | 0.033 vs 0.066 (0.50x) |
| composition, both hops stored (informational, telescoping) | 0.309 vs 0.058 (5.3x) | 0.319 vs 0.064 (5.0x) |

Per relation, seed 17 (matched AUC; relation inference recall vs the class prior pi_r):
| relation | AUC unlabeled | AUC A2b labeled ray-only | pi_r | recall majority | recall balanced | origin-pair metadata | supervised NB |
|---|---|---|---|---|---|---|---|
| wn:hypernym | 0.535 | 0.609 | 0.077 | 0.437 | 0.000 | 1.000 | 0.010 |
| wn:part_holonym | 0.568 | 0.691 | 0.007 | 0.000 | 0.248 | 0.000 | 0.301 |
| wn:member_holonym | 0.751 | 0.844 | 0.010 | 0.000 | 0.000 | 0.000 | 0.146 |
| cn:UsedFor | 0.607 | 0.552 | 0.031 | 0.105 | 0.105 | 0.268 | 0.007 |
| cn:AtLocation | 0.632 | 0.703 | 0.020 | 0.000 | 0.116 | 0.172 | 0.225 |
| cn:CapableOf | 0.542 | 0.599 | 0.020 | 0.000 | 0.379 | 0.741 | 0.663 |
| wd:P19 | 0.819 | 0.906 | 0.119 | 0.000 | 0.034 | 0.000 | 0.093 |
| wd:P17 | 0.937 | 0.918 | 0.053 | 0.000 | 0.000 | 0.000 | 0.410 |
| wd:P131 | 0.963 | 0.978 | 0.063 | 0.000 | 0.000 | 0.000 | 0.284 |
| wd:P171 | 0.754 | 0.461 | 0.113 | 0.000 | 0.316 | 0.000 | 0.009 |

## Labeled (A2b) vs unlabeled (B), same test pairs, same matched negatives
The frozen A2b mix-3 model scored ray-only (relation given, no popularity weight, no calibration):

| | A2b labeled, ray only | B unlabeled (seed 17 / 23) |
|---|---|---|
| matched AUC | 0.744 | 0.759 / 0.742 |
| hits@10 vs 99 matched | 0.587 / 0.591 | 0.507 / 0.503 |
| uniform-negative AUC (A2b protocol, no popularity weight) | 0.771 | 0.855 / 0.843 |
| knows the relation type | yes, given | no: accuracy 0.07 to 0.08, NMI 0.13 |
| direction | by construction | none: 0.51 |
| storage | 72.5 digits per entity (positions 64, transforms 6.6, popularity 1.9) | 64.005 digits per entity |

- **What the labels bought**: relation identity, direction, and a better rank for the true tail among near-popularity candidates (hits@10 0.59 vs 0.51), and per-relation strength on P19 (0.906 vs 0.819) and the WordNet relations (+0.05 to +0.09). The A2b headline (2.085x, AUC 0.907) additionally needed a per-relation popularity weight, which pooled AUC under matched negatives does not reward (labeled ray-only drops to 0.744).
- **What geometry-only costs**: the relation type (the whole point), direction, and composition. Link prediction itself costs nothing: pooled matched AUC is on par, and unlabeled is better on P171 (0.754 vs 0.461, taxon siblings) and UsedFor.
- **Caveats on this comparison**: B uses sibling edges (second-order neighbourhood from training facts) that A2b did not have, and seed noise on AUC is +-0.015. The unlabeled model's uniform-negative AUC (0.855) is not evidence of relation learning; the sphere graph, sibling edges and warm-up make "related means near" strong.

## Novel structure: 10 clusters, unfiltered (seed 17, ranked by size among clusters with >= 500 train edges that do not match a single label, 10 random members each)
All ten qualified under the registered rule; members are uniformly random, nothing hand-picked. Label mix = share of the cluster's train edges by true relation.
1. **Cluster 5** (736,127 edges, cohesion 0.90): P106 24%, P171 15%, P27 14%, P19 12%. The near-zero ray, "related means near", with every relation in it: cotton balls -> plug ears (UsedFor); Eriotheca -> Bombacoideae (P171); Rob Eaton -> singer-songwriter (P106); Snowy River Shire -> Australia (P17); Ora Washington -> United States (P27).
2. **Cluster 27** (43,466): P27 27%, P106 23%, P19 18%. Mixed people-to-place and occupation: Maria Kaupas -> Chicago (P20); Dave Navarro -> composer (P106); Ernst von Manstein -> Germany (P27); Marilyn Avila -> Statesboro (P19).
3. **Cluster 13** (21,154): P27 52%, P19 21%. People to the United States: Ian Hecox, Eddie Rickenbacker, Jenny Simmons -> United States (P27); Douglas Mossman -> Honolulu (P19); Hugh Latimer Dryden -> Washington, D.C. (P20).
4. **Cluster 14** (17,278): hypernym 51%, CapableOf 16%, UsedFor 12%. WordNet verbs and nouns toward their abstractions, plus ConceptNet actions: lance.v.01 -> travel.v.01; cowpox -> animal_disease; director -> cut scene (CapableOf); battery -> make electricity (UsedFor).
5. **Cluster 9** (12,844): P27 40%, P19 21%, P106 14%. People to the United Kingdom or to a sport/profession: Donald Hewlett, Gordon Ramsay -> United Kingdom (P27); Habib Khan -> cricketer (P106); Rolf Schweizer -> Emmendingen (P19).
6. **Cluster 1** (10,952): P106 45%, P27 22%, P19 19%. People to occupations and Latin American and Turkish towns: Scott Brady -> boxer; Pablo Turturiello -> Montevideo (P19); Gödöllő -> Hungary (P17).
7. **Cluster 29** (8,891): P27 46%, P106 29%. French people and football players: Jules Costé, Émile Muselier -> France (P27); Simone Laudehr, Jan Frederiksen -> association football player (P106).
8. **Cluster 3** (8,124): P106 43%, P27 18%, P19 18%, P20 9%. Vienna and football players: Matthias Durst, Auguste Auspitz-Kolar -> Vienna (P20); Petro Stasyuk -> association football player (P106).
9. **Cluster 12** (6,969): P27 33%, P19 18%, CapableOf 14%. Eastern-European places and the Russian Empire: Samuil Feinberg, Andrey Voronikhin -> Russian Empire (P27); Yury Solomin -> Chita (P19); Oulu Province -> Russian SFSR (P17).
10. **Cluster 25** (5,667): hypernym 39%, part_holonym 11%, P19 10%. WordNet abstractions mixed with places: marriage_contract -> contract; gagman -> comedian; viborg -> denmark (part_holonym); Bernardo Sassetti -> Cascais (P20).

Reading: these are not candidate undiscovered relations. They are destination regions of the torus (a person to a country, to a profession, to a city region), so each mixes several known relations that point into the same region. The member lists are in `resb_final_s17_m1_test.json` (`novel_clusters`, with the full 10 members each).

## Direction and inverse (B5)
- Stored-orientation accuracy (does P(a to b) beat P(b to a) for a true edge) is 0.516 and 0.513 pooled, 0.35 to 0.57 per relation (chance 0.5). The random-orientation control (dev) is as good or better (AUC 0.777 vs 0.758, NMI 0.139 vs 0.128), so the stored head-to-tail orientation is not used by the geometry. The no-warm-up variant with sibling edges reads 0.62 but is worse everywhere else.
- Inverse structure: 47% of the prototypes have an inverse partner with similarity >= 0.5 (random vectors: best similarity 0.23 vs measured 0.57). The cause is trivial: the dominant near-zero prototype is its own negation. Every relation's forward edges and reversed edges fall into the same cluster 5 (reversed share 0.35 to 0.97). The data holds no inverse relation pair (hypernym and hyponym, part-of and has-part are not both stored), so inverse relations cannot be identified from it. They are negations by construction (ray(b to a) = -ray(a to b)), and the test would need synthetic inverse labels.

## Composition (B6, A3 style)
- Both hops stored: 5.3x and 5.0x the frequency baseline (hits@10 0.31 vs 0.06). This is the telescoping effect: both hop rays were fitted in training, so x_a + c_k1 + c_k2 lands near x_c. Informational only.
- Hop 1 held out (the gate): 0.49x and 0.50x. Chains P19 to P17 0.20x and 0.21x, P19 to P131 3.3x and 2.4x, hypernym twice 1.1x and 1.8x, P171 twice has a frequency baseline of 0.0 (hits 0.024 and 0.032). Pooled it fails: composing rays does not predict unseen two-hop facts better than candidate frequency.

## Why it does not recover relations (diagnosis)
1. **The objective prefers proximity over relation structure.** With warm-up, EM reaches a train ray fit of 0.81 by sending 72% of edges to one near-zero ray (cohesion 0.90). A zero ray fits every edge of every relation, so the fit-maximising solution makes relations indistinguishable. (The oracle runs below used the no-warm-up layout, so their fit of 0.666 is not directly comparable with 0.81; a like-for-like labeled run on R's layout was not done.)
2. **EM from the random start finds an arbitrary partition.** Without warm-up, held-out NMI is 0.03 to 0.05, the same as k-modes on the unsolved lift (0.047), and the train AUC 0.90 against held-out 0.57 shows memorised edges. The labeled partition scores a higher fit than the arbitrary one (0.666 vs 0.61), so the objective prefers relations but the search does not find them.
3. **Even labels in the placement do not make geometry beat metadata.** The oracle-seeded placement, with relation labels fixed, re-derives them at held-out NMI 0.27 (train 0.67), and 7 of 10 relations reach 2x prior, but origin plus degree metadata reads 0.449. Translation is a weak model for many-to-many relations (one person, several occupations; one country, millions of citizens) and several relations share the same (type, type) pair (P19 and P20, P27 and P19).
4. **Rays carry destination, not relation.** The non-zero clusters are regions (United Kingdom, United States, France, Vienna, football players), mixing citizenship, birth and death place. A supervised readout of the same rays (naive Bayes on per-dim histograms, labels allowed) reaches held-out NMI 0.17 to 0.18 and recall >= 2x prior on 6 relations, so there is some relation information in rays, more than unsupervised clustering extracts (0.13), and less than metadata.
5. **Direction is not used.** The solver is dominated by symmetric proximity, so stored orientation adds nothing (random orientation is equal or better).

## Storage and PTEX (`layer2_b_final_s17.gf17l2p`, `layer2_b_final_s23.gf17l2p`)
- Per entity: **64 digits** (the position in Z17^64). No popularity digits and no per-relation transforms.
- Ray-cluster table: **3,264 digits** in total: 32 prototypes x 64 digits (2,048), density LUT 32 x 8 x 2 digits (512), bucket thresholds 32 x 7 x 3 digits (672), label table 32 digits.
- File: 39,373,248 digits in 5 sections, 20,921,142 bytes at 4.25 bits per digit, 9.84M pixels (one logical 4096x4096 RGBA page). Packed with `torus_a l2pack` (adam-gf17 17-bit Reffelt pixel codec). Roundtrip bit-exact for both seeds: 0 mismatches through the page view, 0 in the per-section decode, 0 in 1M random reads, header and data hashes good.
- Versus A2b (72.5 digits per entity, transforms 6.6 digits per entity at mix 3): 12% fewer digits per entity. The cluster table is 3.3k digits against 314k per relation for the A2b mix-3 transforms.
- Projection (as A2b): 128 GB holds 2.41e11 digits, which is 3.8e9 entities of positions at 64 digits.

## Where it left the paradigm
- **Inference** is integer only: digit differences mod 17, the distance LUT, adds, argmax over K prototypes, an integer bucket search and an integer LUT for P (v/288). No matmul, no float weights stored. All scoring in `torus_b.py` is Taichi integer kernels (`ksc`, `csc`) plus numpy integer lookups.
- **Build time** uses floats, as A2b: the sphere lift (atan2, QR), float64 histograms in the prototype fit rounded to integers before the ICM kernel, quantile thresholds and the density LUT counts (rounded to integers), NMI, AUC and entropy for evaluation, and the supervised naive-Bayes diagnostic (log-probabilities scaled to integers).
- **Not SSD-streamed at scoring time**: the scorer reads in-RAM int32 arrays, not the mmapped pages. The pages are packed and verified, as in A2b.
- **Python orchestration**: edge sampling, k-modes++ initialisation, the confusion tables and the composition loops run in numpy, not in a Taichi kernel; GPU was never touched.
- Nothing silently fell back to a standard ML technique. Where the paradigm could not do something it is stated above: relation types do not emerge.

## Honest caveats
- The A2b test split was reused (same held-out pairs, same no-leak sphere), and the A2b test numbers were known before TORUS-B was written. No TORUS-B quantity was computed on test before the freeze.
- About 45 dev solves were run; dev numbers are optimistic by selection. The test numbers agree with dev to the third digit on NMI (0.128 vs 0.128) and within seed noise on AUC.
- The 0.15 NMI floor and the G2' parity read were written after the dev controls and the oracle diagnostics (Amendment 1), not before. Neither would change the verdict: with the original gates the metadata clause also fails, and G2 fails as originally written.
- Only the solver seed was varied across the two final seeds. The split is the same.
- Sibling edges are a second-order neighbourhood prior built from training facts. A2b did not have them, so the labeled-vs-unlabeled link-prediction comparison is not like for like.

## Next steps
1. **Separate the subspaces.** Give relation prototypes their own subspace and solve proximity (first and second order) in a different one, so "related means near" cannot absorb the ray. A split of D = 64 into 32 proximity dims and 32 translation dims is the cheapest test.
2. **Seed the clusters from structure, then let geometry refine**: edge role signatures (the head's and tail's other-fact profile) as the initial partition, reported as graph-structural, not geometric. The oracle runs show geometry keeps a labeled partition near NMI 0.57 on train; a structural start might land there without labels.
3. **Make the relation a pair of rays rather than one**: the unit that identifies P19 vs P20 is likely (ray from the person to the place, ray from the place to the person's country), a composed-ray signature (two-hop cluster pair), which the composition machinery can already evaluate.
4. **Ask the CEO whether the relation should be a cluster in (head coordinate, ray) space** (a LUT transform, which already raised train NMI to 0.22 to 0.25) rather than a pure translation. That is closer to A2b's per-relation transforms, with the transform discovered instead of supplied.
5. **Measure the ambiguity bound**: for each pair of relations, the share of edges whose (origin, degree-bin) signature is shared, as an upper bound on any label-free method that sees only entity types, and a like-for-like labeled run on R's warm-up and sibling layout.

## Files
- Code: `crates/adam-torus/py/torus_solve.py`, `crates/adam-torus/py/torus_b.py`.
- Results: `data/torus/b/resb_final_s17_m1_test.json`, `resb_final_s23_m1_test.json`, dev `resb_*_dev.json`, `resb_full_k32w10sib_m1.json`, solver logs `solve_*.log`, models `torus_final_s17.npz` and `torus_final_s23.npz`, packed pages `layer2_b_final_s17.gf17l2p` and `layer2_b_final_s23.gf17l2p`.
- Checklist: `docs/checklists/checklist_torus_b_v1.md`. Council: `docs/guardian_councils/guardian_council_torus_b.md`.
