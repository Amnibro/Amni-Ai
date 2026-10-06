# STORE-1 coverage audit (aggregates only)

Source: `data/adam_t/eval/report_h1-hybrid-qwen3-4b-q8-C8_0_e2e_tune_dry.json` (chat suite; know n=102, explain n=59). Per-item detail lives in `coverage_audit.json` and is withheld here to keep the resolver work blind.

Primary class = first store, in the order wd_entity > wd_value > concepts > passages > compute, whose stored fact or text really gives the gold. "Strict" counts only direct hits; "lenient" also admits weak/indirect hits (loose property, partial join, cause-only fact, figurative text, computation that needs an outside fact or an unsupported operator).

## know: primary class (strict)
| class | n | share |
|---|---:|---:|
| wd_entity | 33 | 32.4% |
| wd_value | 14 | 13.7% |
| concepts | 38 | 37.3% |
| passages | 17 | 16.7% |
| compute | 0 | 0.0% |
| none | 0 | 0.0% |

## know: primary class (lenient)
| class | n | share |
|---|---:|---:|
| wd_entity | 37 | 36.3% |
| wd_value | 14 | 13.7% |
| concepts | 34 | 33.3% |
| passages | 17 | 16.7% |
| compute | 0 | 0.0% |
| none | 0 | 0.0% |

## explain: primary class (strict)
| class | n | share |
|---|---:|---:|
| wd_entity | 0 | 0.0% |
| wd_value | 0 | 0.0% |
| concepts | 17 | 28.8% |
| passages | 40 | 67.8% |
| compute | 0 | 0.0% |
| none | 2 | 3.4% |

## explain: primary class (lenient)
| class | n | share |
|---|---:|---:|
| wd_entity | 8 | 13.6% |
| wd_value | 0 | 0.0% |
| concepts | 25 | 42.4% |
| passages | 25 | 42.4% |
| compute | 0 | 0.0% |
| none | 1 | 1.7% |

## Multi-label reach (items each store can answer)
| class | know strict | know +weak | explain strict | explain +weak |
|---|---:|---:|---:|---:|
| wd_entity | 33 | 37 | 0 | 8 |
| wd_value | 14 | 14 | 0 | 0 |
| concepts | 78 | 79 | 17 | 31 |
| passages | 101 | 102 | 57 | 58 |
| compute | 4 | 8 | 0 | 0 |

Strict labels per item, know: {1: 16, 2: 45, 3: 40, 4: 1}; explain: {0: 2, 1: 40, 2: 17}.

## know: Wikidata coverage
| measure | n |
|---|---:|
| reachable from WD v3 (strict, entity or value) | 47 |
| reachable from WD v3 incl. weak | 51 |
| subject entity in WD, property/fact missing (wd_subject_only) | 51 |
| not WD-reachable and no single topic entity | 0 |
| not WD-reachable but answered by concepts or passages | 51 |

## know: current v3 resolver (`W.ask` on the raw question)
| measure | n |
|---|---:|
| returns an answer | 23 |
| answer correct (any confidence) | 21 |
| accepted at threshold (`W.answer`) | 18 |
| accepted and correct | 17 |
| returns a wrong answer | 2 |
| WD-reachable (strict) but not answered correctly | 26 |
| WD-reachable (strict) but not accepted+correct | 30 |

## Caveats
- Passage reach was found with retrieval plus targeted title lookups and then read by hand; it shows a passage exists in the index, not that the live `_know` ranking surfaces it.
- Concept reach includes WordNet glosses and lemma synonyms; a resolver would need gloss parsing or lemma lookup to use them.
- Superlative and count questions are not counted as WD-reachable even where the values for an aggregate exist.
- Explain items have no gold answers, only optional keyword hints. Reach means some store states the mechanism those keywords (or, failing them, the standard explanation) describe. This call is more subjective than for know items.
- The resolver figures use the raw question only. The chat client's rephrasing variants (e.g. a leading "Name ...") can recover a few more.
