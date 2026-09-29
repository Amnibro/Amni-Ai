# ADAM-T-COGNITION: relations, abstraction and inference outside the model

Status: design plus working prototype, 2026-09-29. CPU only, run under `systemd-run --user --scope -p MemoryMax=6G -p CPUQuota=200% nice -n 15`. No GPU was used, and nothing named `adam-t` was run or touched.

- **Code:** `amni/serve/cognition.py`. It is behind `AMNI_COGNITION=1` and is off by default.
- **Eval:** `scripts/adam_t_cognition_eval.py`.
- **Tests:** `tests/test_cognition.py`.
- **Data:**
  - `data/adam_t/eval/cognition/dev.jsonl` (310 items) and `test.jsonl` (145 held-out items);
  - reports `report_dev.json`, `report_test.json` and `report_creative_dev.json`.

## 1. Summary

The RCA (`rca_grounding.md`) found that ADAM-T copies one fact well but cannot bind or reason over several. So reasoning lives outside the model, as lookups and transform rules over stores Adam already has. The model only phrases the result. It receives a short list of checked links, one per line, each written as a single fact it can copy.

The layer implements:
- **six operations**, each returning typed paths with provenance: `relate`, `common`, `analogy`, `deduce`, `abstract`, and `connect` (the creative search);
- **a mode dial** that is a lookup/pattern classifier: factual, analytical or creative;
- **a creativity score:** path validity × a band over hyperbolic (Lorentz) semantic distance.

There is no matmul on the reasoning path: only sorted-array lookups, set intersections, CSR slices and a few scalar `cosh`/`acosh` calls. There is no new build on disk. One small in-memory index over the WordNet glosses is built at warm-up in about 3 s. Process RSS peaks at 2.3 GB, under the 5 GB budget.

Headline results. *Dev* is the tuning set; *held-out* was written after the code was frozen and run once.

| | dev (first run) | dev (final) | held-out (run once) |
|---|---|---|---|
| relate, top-path keyword hit / hit@3 | 64 / 84 | 73 / 87 | 43 / 77 |
| relate, top path judged plausible by hand | – | 27/30 (90%) | 27/30 (90%) |
| common, hit@1 / hit@3 | 68 / 92 | 74 / 90 | 87 / 100 |
| analogy, answer correct / in top 3 | 78 / 94 | 86 / 98 | 70 / 80 |
| abstraction, first step / any of 3 steps | 88 / 88 | 88 / 92 | 90 / 95 |
| mode accuracy (60 items) | 95.0 | 100 | 98.3 |
| latency per operation, p95 (max) | – | 14 ms (80) relate, 9 ms (18) common, 2 ms analogy, 1 ms abstract | 55 ms (93) relate |

**The keyword metric understates relate.** It checks the top path against a short hand-written list of connector words. On held-out items, 14 of the 17 keyword "misses" are correct links in other words, for example "Charles Dickens, notable work: Oliver Twist" or "Japan, highest point: Mount Fuji". So the hand judgment in §8.4 is the better measure.

**The analogy drop from dev to held-out (86 to 70) is real.** It is overfitting to the dev pairs plus sense-coverage gaps, detailed in §8.5.

## 2. The concept map

### 2.1 Nodes

A node is a coordinate, not a stored object. Each has a kind digit and an index into an existing store. Nothing is copied.

| kind | index | store (reused, unchanged) | size |
|---|---|---|---|
| `n` WordNet synset | row in `data/concepts/wordnet_graph.npz` | CSR over 348k typed edges; 117,659 synsets; lemma index | 3.6 MB |
| `w` Wikidata item | entity row in `data/wikifact/v2/store` | 11.8M entities, 53.4M facts (GF(17) pages), reverse index `r_v/r_f`, alias hash `a_h` | 4.3 GB, memory-mapped and paged lazily |
| `u` user entity | row in the user's `.torus` (`amni/serve/user_toroid.py`) | 5W slots: entity, attribute, value | per user |
| `t` text term | term id in the KS passage index `data/ks/index_rknow/wiki_v2` | 41.6M Wikipedia passages, postings, alias table | memory-mapped |
| (corpus word) | vocabulary id in `data/concepts/cooc_graph.npz` | 120k words, top-24 PMI neighbours | 12 MB |

**Linking "the same thing".** Links are made lazily, at query time:
- **Wikidata item to WordNet synset.** The item's label is looked up in the WordNet lemma index (`wanc`). If that fails, its first `instance of`/`subclass of` class is tried, one step below.
- **Sense agreement.** When a pair of terms is resolved, the two WordNet senses are chosen together, by the deepest shared ancestor with a penalty for sense rank. The Wikidata candidates are then re-sorted so that items whose anchor agrees with the chosen sense come first (`_pre`/`_align`). For "Mars and Venus" the top path is the planets ("inner planet of the Solar System"). The consistent god reading ("Dii Consentes") still appears as the second path.
- **Case agreement.** A lowercase query cannot pick a capitalised WordNet lemma: "salmon" can no longer resolve to the Salmon River.
- **User entities.** "my dog" becomes the node `your dog`. It gets a `type of: dog` link to the WordNet anchor of its head noun, and its stored slots come along ("your dog, name: Biscuit.").

### 2.2 Typed relations

| source | relations used | confidence `c` | allowed in factual mode |
|---|---|---|---|
| Wikidata (`wd`) | the entity-valued properties of the v2 KEEP list: capital, author, notable work, instance/subclass of, part of, flows into, … | 0.95 | yes |
| WordNet (`wn`) | type of, part of / has part, member of, made of, substance of, similar to, antonym, attribute, entails, causes, related form | 0.95 | yes |
| WordNet gloss (`gl`) | "X is defined as …, mentioning Y" | 0.85 | yes |
| user toroid (`us`) | your X, attribute: value; your X, type of: X | 0.90 | yes |
| Wikipedia sentence (`tx`) | one sentence mentioning both terms (title-matched: 0.85, else 0.72) | 0.72–0.85 | only when title-matched |
| Wikipedia bridge (`br`) | a word found in the co-mention sentences of at least 2 passages | ≤ 0.70 | no |
| corpus co-occurrence (`co`) | PMI association | ≤ 0.50 | no |

**Wikidata property priors.** Very general properties are down-weighted in ranking: `country` 0.8, `instance of` 0.8, `part of` 0.85, `located in` 0.85, `award received` 0.7. So "Tokyo, capital of: Japan" outranks "Tokyo, country: Japan".

**Hubs.** A node that thousands of facts point at, such as `human`, `France` or `taxon`, makes a trivial bridge. The specificity factor `1/(1+ln(1+indegree/100))` handles this. A class chain is scored by its most hub-like node, so "Einstein and Newton are both anatomically modern humans" loses to "both are physicists".

## 3. Hyperbolic coordinates (Lorentz model)

**The coordinates.** A WordNet synset gets a point in H³ on the hyperboloid, X = (cosh ρ, sinh ρ · u):
- **Rapidity** ρ = δ·n. The integer n is the synset's minimum hypernym depth, computed once at load by a vectorised frontier sweep. δ = `AMNI_COG_DELTA` = 0.5.
- **Direction** u is the synset's unit vector in `sphere.npz`, the existing WordNet sphere, with hypernym recall 0.58.
- The root sits at ρ = 0. Specific concepts sit far out.

**Exact rapidity storage.** Following `lorentz_prereg.md` §3, n is stored as an octave and a GF(17) discrete-log digit: `rz(n) = (n >> 4, 3^(n mod 16) mod 17)`.
- A boost by k (`boost`) adds k to the log digit, with carry into the octave.
- Abstraction is a boost by −1 per level, or by the exact depth difference when WordNet's multiple inheritance skips a level.
- Chains of boosts compose additively and exactly. `test_gf17_rapidity_roundtrip_and_additive_boosts` checks 300 random chains and the round trip for 4,096 values. Every `abstract()` result carries each step's `(n, rap)` and is checked with `rn(rap) == n`.

**Semantic distance** is the hyperbolic law of cosines taken at the pair's deepest shared ancestor L:

  D(a, b) = arcosh( cosh ρa′ · cosh ρb′ − sinh ρa′ · sinh ρb′ · cos θ ),  with ρa′ = δ(n_a − n_L), ρb′ = δ(n_b − n_L), cos θ = u_a · u_b

- Siblings under a deep ancestor with similar directions come out close.
- Concepts that only meet near the root come out far, bounded by δ·(n_a + n_b).
- Moving from a to b is a boost down to L and a boost back up: two exponent additions plus one rotation.

**Calibration**, from `adam_t_cognition_eval.py calib`:

| pair set | D p10 / p50 / p90 |
|---|---|
| 15 hand-picked siblings (dog/cat, oak/maple, …) | 0.00 / 0.06 / 5.64 |
| 85 dev relation pairs | 0.63 / 4.02 / 7.85 |
| 400 random noun pairs | 1.63 / 5.31 / 8.24 |

The siblings' p90 of 5.6 comes from two pairs where the first-sense rule picked an unrelated sense.

## 4. Operations

Every operation returns `{op, mode, ok, paths, lines, ms, …}`. Each path is a list of edges `{a, r, b, src, ref, c, ka, kb, line}`:
- `ref` is the provenance: the Wikidata `Q.. P.. Q..` triple, the WordNet synset and relation, or the Wikipedia article title;
- `ka`/`kb` are the Lorentz anchors;
- `line` is the verbalised link.

**Path scores:**
- validity `v = Π c_e`;
- hop penalty `0.85^(hops−1)`;
- specificity `sp`;
- analytical score `q = v · sp · hop`;
- creative score `cr = v · hop · B(D)` (§6).

**`relate(x, y)`**, the shortest meaningful typed path. It merges five generators:
1. **Wikidata, one hop:** a fact x→y or y→x.
2. **Wikidata, two hops:** a shared value (x→m←y); forward chains x→m→y through the 60 least hub-like values of x; co-subjects (m→x and m→y, e.g. a paper or event naming both).
3. **WordNet:** a direct typed edge; a shared non-taxonomic neighbour (part, member, substance); or the deepest shared ancestor, compressed into two transitive "type of" links.
4. **WordNet glosses:** x's definition names y, or the reverse.
5. **Wikipedia passages:**
   - intersect the postings of both terms;
   - rank passages by title match, then length;
   - read at most 24;
   - pick the best sentence that names both, preferring a lead position, a short gap between the two mentions and a relational verb, and penalising list pages;
   - words appearing in the co-mention sentences of at least 2 passages become "bridge" concepts.

Corpus PMI associations are added outside factual mode.

**`common(x, y)`**, the nearest shared ancestor or shared property. It returns:
- the WordNet deepest shared ancestor and shared parts;
- the Wikidata class-closure intersection (3 levels of `instance of`/`subclass of`, each scored by its most hub-like node);
- shared property values, excluding awards, series, children, cast and other noise properties;
- Wikipedia bridge words and, outside factual mode, shared PMI neighbours.

**`analogy(a, b, c)`** finds the transforms that take a to b, applies each to c and votes on the answers by label:
- **Wikidata property**, forward or reverse. Candidates follow Wikidata's preferred rank first. Reverse transforms weigh ×0.7, and an answer that shares a class with b weighs ×1.5.
- **WordNet typed edge.** Candidates are ordered by Lorentz consistency |Δn(c→d) − Δn(a→b)|, with a ×1.5 bonus when d and b share a deep ancestor.
- **Ancestor transform.** When b is an ancestor of a, take the ancestor d of c that sits at the same height below its shared ancestor with b as b does. This gives dog:mammal :: sparrow:bird, since mammal and bird are both directly under vertebrate.
- **Antonyms** across every sense of a and c.
- **Gloss templates.** The word before b in a's gloss ("young" dog) is looked up in c's gloss and the head noun is taken ("young domestic cat" → domestic cat).

**`deduce(x, y, rel)`** checks transitive or compositional chains:
- `isa`: WordNet hypernyms, or Wikidata `instance of`/`subclass of`;
- `in`: also WordNet part-holonyms and Wikidata located in / country / continent / part of.

It returns the chain ("Paris, part of: France." / "France, part of: Europe.") and `holds`.

**`abstract(x)`** steps up the hierarchy:
- for common nouns, and proper nouns WordNet knows, the deepest hypernym at each step, with an exact GF(17) boost per step;
- otherwise Wikidata `instance of` (preferred rank) and then `subclass of`.

Factual mode takes one step; analytical mode takes three.

**`connect(seed)`** is the creative search:
- concepts whose gloss is defined using the seed;
- concepts sharing a specific noun from the seed's gloss (gloss inverse document frequency ≥ 0.45);
- PMI neighbours of the seed's strongest associations.

Candidates are scored by `cr`, proper nouns weigh ×0.7, and at most one target per bridge word is taken first so the options are diverse.

## 5. Mode dial

`mode(message)` is a weighted pattern lookup. It uses no model.

| mode | cues | effect |
|---|---|---|
| factual (the default: pull back) | wh- questions with an entity; "capital of", "born", "how many/tall/long", "who wrote" | only `wd/wn/gl/us` and title-matched Wikipedia sentences; top path only; at most 2 lines; abstraction takes 1 step |
| analytical | an analytical operation parses (relate/common/analogy/abstract/deduce); "why", "compare", "difference", "in common", "is X a kind of", "how does X affect" | all sources except PMI in `relate`; up to 2 paths (the second only if its score is at least 0.5× the first); at most 4 lines |
| creative | poem, story, haiku, metaphor, "what if", imagine, brainstorm, invent, pretend, "as if it were", … | all sources; ranked by `cr`; "metaphors for X" runs `connect`, "poem about X and Y" runs `relate` in creative mode |

Scoring:
- Factual starts at 0.3, so it wins ties and anything with no cues.
- Each matched cue adds its weight: creative 2.0, analytical 1.5, factual 1.2 or 0.8.
- A parsed analytical operation adds 1.5.

The harness can use `mode` to choose decoding as well, for example greedy for factual and sampled for creative. That choice is the harness's.

## 6. Creativity as a measurable target

  **creativity = validity × B(D)**,  B(D) = (D/D₀) · e^(1 − D/D₀),  D₀ = 2.5

- **Validity** is the product of the edge confidences. Every link must be a checked fact from a store, so an invented link scores 0.
- **B** is a band over the Lorentz distance between the connected concepts:
  - B(0) = 0, so a trivial link (a synonym or sibling) scores nothing;
  - B peaks at 1 when D = D₀;
  - B decays for distant leaps: B(8) = 0.35, B(11) = 0.12.
- D₀ was fixed before calibration. It lies between the sibling median (0.06) and the random-pair median (5.3).
- **Random** connections lose twice: long or weak paths have low validity, and far leaps have low B.

**Fifteen creative-mode connections.** `scripts/adam_t_cognition_eval.py creative` computes them; the data is in `data/adam_t/eval/cognition/report_creative_dev.json`. The top pick per seed, then the bottom of each seed's scored list:

| seed | top connection (via) | V | D | score | my rating | bottom connection (via) | D | score |
|---|---|---|---|---|---|---|---|---|
| time | token ("instance") | 0.68 | 2.44 | 0.68 | weak (generic bridge) | humiliation ("instance") | 0.00 | 0.0001 |
| ocean | Vasco da Gama (crossed the Indian Ocean) | 0.85 | 2.51 | 0.85 | valid, literal | lake (PMI via "river") | 0.01 | 0.001 |
| loneliness | quarantine ("isolation") | 0.72 | 2.52 | 0.72 | **good metaphor** | anomie ("isolation") | 0.00 | 0.003 |
| city | blockbuster (a bomb for a city block) | 0.85 | 2.62 | 0.85 | surprising, valid | District of Columbia ("district") | 0.05 | 0.025 |
| fire | cyanide poisoning (smoke from fires) | 0.85 | 2.48 | 0.85 | valid, dark | Percheron ("artillery") | 10.5 | 0.086 |
| music | marching band | 0.85 | 2.44 | 0.85 | valid, literal | poetry ("communication") | 0.01 | 0.005 |
| memory | word (bits stored in computer memory) | 0.85 | 2.47 | 0.85 | **good cross-domain** | reliving ("experience") | 0.03 | 0.018 |
| heart | septal defect | 0.85 | 2.47 | 0.85 | valid, literal | Tennessee walker ("move") | 11.2 | 0.046 |
| moon | limb (edge of the disc of the sun or moon) | 0.85 | 2.51 | 0.85 | good | inhabitant (PMI via "planet") | 0.60 | 0.065 |
| tree | wood duck (nests in hollow trees) | 0.85 | 2.51 | 0.85 | **good** | Yggdrasil (the world tree) | 0.00 | 0.000 |
| love | pax (a greeting of Christian love) | 0.84 | 2.91 | 0.84 | valid, surprising | joy ("emotion") | 0.00 | 0.002 |
| bridge | conning tower (a raised bridge on a submarine) | 0.85 | 2.51 | 0.85 | good (a sense shift) | building complex ("structure") | 0.00 | 0.002 |
| winter | hibernation | 0.85 | 2.38 | 0.85 | **good** | summer ("equinox") | 0.00 | 0.0004 |
| light | doze ("a light fitful sleep") | 0.85 | 2.31 | 0.85 | pun on a sense; valid as wordplay | ultraviolet ("radiation") | 0.01 | 0.003 |
| money | tax bill | 0.85 | 2.52 | 0.85 | valid, literal | Spanish mackerel ("medium") | 11.0 | 0.057 |

**What the table shows:**
- The metric's two failure regions are exactly the ones it was built to reject. The bottom rows are either trivially close (D ≈ 0: summer/winter, joy/love, seven seas/ocean) or homograph leaps that are technically valid but random (D > 10: fire → Percheron via "artillery", money → Spanish mackerel via "medium").
- Of the 15 top picks, 6 are good, surprising and valid; 6 are valid but literal; 3 are weak.
- The main limit is that gloss-word bridges have no word-sense disambiguation. A shared *word* is not always a shared *sense*: "tender" joins money and sentiment. Creative mode tolerates this, and analytical and factual modes never use these bridges.

## 7. Verbalisation and integration

**Line format.** Each link is one line in the `<Label>, <relation>: <value>.` form the model already copies (the `_wl` Wikidata line format), or `<Title>: <sentence>` for a Wikipedia sentence, or `<term>: <gloss>.` for a WordNet gloss. Lines never mix two facts. Examples from the prototype:

```
How are bees and almonds connected?        (analytical)
  Pollination: Almond pollinating bees, approximately 60% of the bees in the US ... will be mixed with bees from thousands of other hives ...
  bee, type of: organism.
  almond, type of: organism.
Paris is to France as Tokyo is to what?    (answer: Japan)
  Paris, country: France.
  Tokyo, country: Japan.
Is Paris in Europe?                        (holds: true)
  Paris, part of: France.
  France, part of: Europe.
hot is to cold as tall is to what?         (answer: short)
  hot, opposite of: cold.
  tall, opposite of: short.
What do a whale and a bat have in common?
  whale, type of: placental.
  bat, type of: placental.
  whale, type of: mammal.
  Chiroptera, subclass of: mammal.
```

**API for ADAM-T-HARNESS.** The harness's internals are not edited. It calls:

```
cognition.context(message, history=None, user=None, budget=0)
  -> {mode, op, lines, ms, answer, holds}
```

- `lines` are ASCII-folded, each clipped to `AMNI_COG_LINE_BYTES` (200), and the list is cut to fit `budget` bytes.
- It returns `lines=[]` unless `AMNI_COGNITION=1`, and it never raises.
- `user` can be a user id (loaded through `user_toroid`, only if that flag is on) or a toroid `Store`.
- The lines are meant to go one per line into the harness's context block.
- The current `adam_t_client.template` joins `rc` items with spaces. Every line is a self-contained sentence ending in ".", so that also works, but newline joining matches the RCA's single-fact copying setup better.
- `cognition.run(message)` gives the full paths with provenance, for logging or UI.

**Flags:**
- `AMNI_COGNITION` (off);
- `AMNI_COG_WARM` (1: warm in a background thread when enabled);
- `AMNI_COG_DELTA` (0.5);
- `AMNI_COG_D0` (2.5);
- `AMNI_COG_LINE_BYTES` (200);
- `AMNI_CONCEPTS_DIR` (`data/concepts`);
- `AMNI_WIKIFACT_DIR` (existing).

## 8. Evaluation

### 8.1 Data

The dev set `dev.jsonl` has 310 items and the held-out set `test.jsonl` has 145. Both are CC0, written from scratch for this mission.

| type | dev | held-out | gold |
|---|---|---|---|
| relation, "how are X and Y connected" (4 templates) | 100 | 30 | connector keyword stems |
| relation, "what do X and Y have in common" (4 templates, one with "share") | 50 | 15 | connector keyword stems |
| analogy (3 templates, including `a : b :: c : ?`) | 50 | 20 | answer stems |
| abstraction (4 templates) | 50 | 20 | acceptable ancestors |
| mode classification | 60 (20 per mode) | 60 | label |

**Decontamination.** `adam_t_cognition_eval.py overlap` checks every string field of every item against every string field of `data/adam_t/eval/chat/chat_src.jsonl` (2,019 distinct 6-grams). Both sets have **0** overlapping 6-grams. Getting there took rewrites of common trivia phrasings such as "what is the boiling point of water": 5 dev items (7 rewrites) and 4 held-out items.

### 8.2 Protocol and honesty notes

- The metrics are keyword hits, so the gold lists are deliberately short. The hand judgments in §8.4 are the main measure of relation quality.
- **Dev run 1** was taken before any change based on dev items.
- **Dev runs 2 and 3** fixed failure classes seen in run 1. All fixes were generic:
  - sense order (sense number over lemma-count ties);
  - plural lookup ("bats" had matched WordNet's slang "bats");
  - Wikidata preferred rank (Tokyo is Japan's 9th `capital` value but the preferred one);
  - `searchsorted` dtype (the uint32 memmap was upcast on every call: 26 ms per call, 9 s per query);
  - hub-aware class chains, noise-property exclusion, type agreement in analogies, antonyms over all senses, case agreement, and "share"/article parsing.
- Mode went from 95% to 100% after two analytical cues and op-aware scoring were added.
- **The held-out set was written after the final code and run once.** No change was made after it.

### 8.3 Results

| op | set | answered | hit@1 | hit@3 | lines hit | parse | end-to-end hit | latency p50 / p95 / max (ms) |
|---|---|---|---|---|---|---|---|---|
| relate | dev | 100 | 73 | 87 | 85 | 99 | 85 | 5.4 / 14.5 / 79.6 |
| relate | held-out | 100 | 43 | 77 | 70 | 100 | 70 | 11.1 / 55.0 / 92.8 |
| common | dev | 100 | 74 | 90 | 86 | 100 | 86 | 3.5 / 8.6 / 17.6 |
| common | held-out | 100 | 87 | 100 | 87 | 100 | 87 | 6.9 / 11.1 / 11.1 |
| analogy | dev | 98 | 86 | 98 | 86 | 100 | 88 | 1.0 / 2.3 / 2.6 |
| analogy | held-out | 80 | 70 | 80 | 70 | 100 | 70 | 1.7 / 8.3 / 8.3 |
| abstract | dev | 100 | 88 | 92 | 88 | 100 | 92 | 0.2 / 0.7 / 1.2 |
| abstract | held-out | 100 | 90 | 95 | 90 | 100 | 95 | 0.6 / 1.6 / 1.6 |
| connect | 15 seeds | – | – | – | – | – | – | 13–98 per seed |

**Mode:**
- dev: 60/60 (run 1: 57/60, all misses analytical → factual);
- held-out: 59/60. The miss is "What drives the change between summer and winter?" → factual.

**Latency:**
- Measured on the first call per item, in a warm process on this PC's page cache, under the 2-core cap.
- Every operation stays under 200 ms. The worst relate is 93 ms; `connect` peaked at 217 ms before its fan-out was bounded and now tops out at 98 ms.
- Warm-up (nltk WordNet load, gloss index, npz files, index handles) takes 5.7–6.5 s once per process.
- A truly cold SSD cache was not measured, because dropping the page cache needs root.

**Top-path sources on dev** (relate + common): Wikidata 79, WordNet 61, Wikipedia sentence 29, gloss 26, bridge 6.

### 8.4 Plausibility and path validity (hand-checked)

**Dev.** Thirty relate/common items were drawn at random (`eval sample`, seed 7), and I judged each top path. The verdict is **27 plausible, 3 partial, 0 wrong**.

| item | top link(s) given to the model | judgment |
|---|---|---|
| penguin / ostrich, in common | both type of: bird | plausible ("flightless" would be better) |
| Sahara / Africa | Sahara, continent: Africa | plausible |
| bees / almonds | Pollination: "Almond pollinating bees, approximately 60% of the bees in the US …" | plausible |
| kidney / liver | both instance of: organ type; type of: internal organ | plausible |
| laptop / smartphone | both subclass of: mobile computer | plausible |
| Nile / Mediterranean | Nile, flows into: Mediterranean Sea | plausible |
| Einstein / relativity | gloss: "…formulated the special theory of relativity and the general theory…" | plausible |
| ice / water | ice, made of: water; "water frozen in the solid state" | plausible (a keyword miss) |
| rainbow / light | gloss: "an arc of colored light … caused by refraction" | plausible |
| guitar / harp | both type of: stringed instrument | plausible |
| Star Wars / George Lucas | Star Wars, creator: George Lucas | plausible |
| Edwin Hubble / Hubble Space Telescope | Hubble Space Telescope, named after: Edwin Hubble | plausible |
| tulips / Netherlands | "Windmills, tulips, clogs … symbolize the Netherlands" | plausible |
| coal / electricity | "Coal burnt in coal power stations to generate electricity…" | plausible |
| bread / wheat | "Bread is usually made from a wheat-flour dough…" | plausible |
| diamond / graphite | both subclass of: carbon-silicon family; native element mineral | plausible (odd Wikidata class; "both are carbon" would be better) |
| yen / Japan | Japan, currency: yen | plausible |
| Neil Armstrong / Moon | gloss: "first man to set foot on the Moon" | plausible |
| paper / trees | "Most paper companies also plant trees to help regrow forests." | **partial** (true, but misses "made from wood pulp") |
| Bach / organ | Johann Sebastian Bach, instrument: organ | plausible |
| whale / bat | both type of: placental (mammal) | plausible |
| clouds / rain | gloss: "drops of fresh water that fall as precipitation from clouds" | plausible |
| pandas / bamboo | gloss: "…mammal of bamboo forests"; "bamboo … major food source of the giant panda" | plausible (a keyword miss) |
| gold / silver | both subclass of: noble metal | plausible |
| Louis Armstrong / trumpet | instrument: trumpet | plausible |
| rose / tulip | both type of: vascular plant | **partial** (true but shallow) |
| photosynthesis / sunlight | "Sunlight is a key factor in photosynthesis…" | plausible |
| spiders / webs | "…predators that build curtain webs over their burrows" | plausible |
| sushi / Japan | "In Japan, green tea is invariably served together with sushi." | **partial** (the second line, country of origin: Japan, is the real answer) |
| piano / organ | both instance of: type of musical instrument; linked with: pedal | plausible |

**Held-out.** All 30 relate items were judged. The verdict is **27 plausible, 3 wrong**. All three failures are lexical co-mention accidents:
- "Night: people who prefer to be active at night are called night owls" (an idiom, not a relation);
- a hurricane that hit "Ocean City";
- an off-topic sentence about the 1888 Río de la Plata earthquake.

**Validity of individual links.** Every line in the 60 judged examples is a verbatim fact or sentence from its cited source, so provenance is 100%. One line is false for the *intended* entity: "Statue of Liberty, location: France." It is an entity-resolution error (a replica item), not a reasoning error. One dubious ontology claim ("carbon-silicon family") comes from Wikidata itself.

### 8.5 Failure analysis

1. **Sense ranking.** WordNet's sense order is only partly frequency-based. Cases:
   - "oak" resolves to the wood, "chess" to a grass, "hammer" to a gun part and "democracy" to a political orientation (abstraction misses);
   - "Neptune" resolves to the god (a valid reading).

   The sense chooser in pair operations fixes most of these in context. Single-term abstraction has no context.
2. **Lexical co-mention.** The `tx` generator takes any sentence naming both terms: "night owls", "Ocean City", "Kangaroo Island". Title match, verb and list-page features cut most of these. The rest are why factual mode admits only title-matched sentences.
3. **Analogy coverage.** Held-out misses:
   - cub:bear::piglet, wrist:hand::knee and violin:string::flute find no shared transform in WordNet;
   - light:dark::happy fails because the adjective sense of "light" is beyond the six senses scanned;
   - cat:mammal::trout gives "aquatic vertebrate", a correct co-hyponym of mammal but not "fish";
   - hammer:tool::pistol gives "instrument";
   - puppy:dog::kitten gives "young mammal", because WordNet files kitten under young mammal and not under cat. It is a known failure and is not in the tests.
4. **Creative bridges.** These have no word-sense disambiguation (§6).
5. **Graph coverage.** Commonsense relations are thin in the kept Wikidata properties. `parent taxon`, `uses`, `main food source` and `made from` are not in the v2 KEEP list, so "pandas eat bamboo" comes only from glosses and Wikipedia.

## 9. Next steps (not built)

- **Harness wiring.** Done in ADAM-T-HARNESS2 (`docs/adam_t/harness.md`, v2): lines are newline-joined, analytical replies are fact-checked or composed from the checked lines, and creative replies are coherence-checked. On this dev set, gold-keyword hit went from 0.096 to 0.840 through the live route. Mode-driven decoding is still open (the protocol has no sampling field).
- **Wikidata v3 KEEP.** Add P171 parent taxon, P1034 main food source, P186 made from, P366 has use, P2283 uses, P1542 has effect, P828 has cause and P461 opposite of. These fill the commonsense gap without new code: they are just more typed edges.
- **Word-sense disambiguation for gloss bridges.** Disambiguate gloss words with the same pair-sense chooser, which would remove the homograph bridges in creative mode.
- **A held-out analogy set per transform type,** to tune the vote weights on something other than dev.
