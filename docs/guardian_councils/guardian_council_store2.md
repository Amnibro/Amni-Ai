# Guardian council: STORE-2 (2026-10-06)

Question: how to cut the STORE-1 drive toward 300 MB and 7 bits/fact while fixing the resolver's test-time drop and latency, inside the GF(17) paradigm.

## Proposals
- **Architect**: one symmetric coding procedure (count, encode and decode run the same code path), base-17 rANS whose output stream is GF(17) digits in Reffelt pixel words; streams per 32-entity block (header, core facts, labels, names, qualifiers), lazy fact values so a block's symbols decode without cross-entity lookups.
- **Sentinel**: every predictor that references another fact must be resolvable without cycles; the encoder must prove it. Parity over all facts, qualifiers, names and every name key before any size claim; test scored once; fallback answers must be hedged.
- **Scholar**: facts as rank in predicted candidate lists (rules, block recency, context lists, near-index, class-partitioned codebooks); measure each predictor's saving with ablations and charge its tables; report the torus honestly (it covers 615k entities and its layer costs 25 MB).
- **Engineer**: keep STORE-1's ribbon name table (F=2/3 per the CEO), piece-hash keys; split labels and the header (QID gap, sitelinks) into their own streams so the resolver's hot path decodes little; build with 3 cores under 12 GB.
- **Pathfinder**: mine the property map from Wikidata labels, aliases, descriptions, WordNet synonyms/derivations/glosses and wh-typed templates; keep the STORE-1 hand table only as a supplement and report "mined only".

## Vote
Unanimous on the symmetric rANS-17 codec, cycle-checked prediction, parity-first, and the mined property map with the hand table as an ablated supplement. Split vote (Engineer vs Scholar) on block size: 32 chosen for latency; 64 reported as the size alternative.
