# Proposal: restore numbers in the wiki_v2 knowledge index

Status: approved by the CTO on 2026-09-28 and executed as WIKI-NUMBERS (see "Execution" at the end). `wiki_v3` is built next to `wiki_v2` but **not swapped in**: the readiness gate failed (+6.0% passages), and R-KNOW tune/hold dropped because of decontam. Needs a CTO decision.

## What is wrong

ADAM-T's live context shows passages like "Located approximately  from the city center", "At the , it had 395,790 residents" and "(16 December 191719 March 2008)". The numbers, units and date dashes are missing from the index text itself, so a better ranker can't fix this. The model is being shown sentences with holes where the answer should be.

## Root cause

`data/ks/index_rknow/wiki_v2` is built from `data/ks/fetch/wikipedia`, which comes from `data/ks/clean/wikipedia`. Shards 00000-00040 (5.1M articles, about 87% of the docs) were ingested by `scripts/ks_download.py wiki()` from the Hugging Face `wikimedia/wikipedia` `20231101.en` parquet. That dataset was extracted from wikitext with its templates dropped. `{{convert}}`, `{{height}}`, census and `{{snd}}`/`&ndash;` date-range templates all disappear, and they carry most of Wikipedia's measurements, populations and life dates.

Shards 00041-00055 came from `HuggingFaceFW/finewiki` (`finewiki()` in the same script), which is extracted from rendered HTML, so the templates are expanded.

Measured on random lead passages (3,000 per source; `_HOLE` is the conservative unit-hole detector in `amni/serve/adam_t_client.py`; "broad" means any `word  word` or `word ,` template gap):

| source | unit/number hole | broad template hole |
|---|---|---|
| HF `wikimedia/wikipedia` 20231101 (shards 00000-00040) | 5.7% | 15.5% |
| FineWiki (shards 00041-00055) | 0.8% | 4.4% |

About 0.4% of leads also have glued date ranges ("191719 March").

## What already ships at read time (no rebuild)

`_fix()` in `amni/serve/adam_t_client.py`:
- re-splits glued date ranges ("16 December 1917 - 19 March 2008", "(1917-1985)");
- drops empty "( )" left by stripped pronunciations;
- marks number holes, so sentence trimming avoids them and passages with holes rank lower on how-many/how-much questions.

This hides some of the damage but can't bring back a number that was never stored.

## Proposed fix

Re-source shards 00000-00040 from FineWiki, then rebuild the fetch store and the R-KNOW index.

1. Run `finewiki()` in a mode that replaces text for the page ids already present, not only the missing ones. The current code filters with `have != pid`. The change is one condition plus a separate output dir. Reuse `fwtext()`.
2. Run `ks_ingest` (neutral filter, hygiene, dedupe, held-out) into `data/ks/clean/wikipedia_fw/`. Keep the held-out selection identical by title, or re-lock it. This needs a decision.
3. Rebuild `data/ks/fetch/wikipedia_fw` and `adam-rknow` into `data/ks/index_rknow/wiki_v3`. Then rebuild `d_first`, `bt_*` and `al_*`.
4. A/B `wiki_v2` against `wiki_v3` through `AMNI_RKNOW_INDEX` with `scripts/adam_t_retrieve_dev.py run` and the R-KNOW dev harness. Switch only if both hold or improve.

## Cost (estimates for this box, CPU only, niced)

| step | size / time |
|---|---|
| FineWiki enwiki download | 15 parquet files, 35.1 GiB total, streamed one at a time (about 2.5 GB peak on disk); 10-30 min on the network |
| convert + write shards | about 30-60 min CPU |
| clean / hygiene / dedupe | about 40-60 min (the CC clean ran at about 8 MB/s) |
| fetch store encode | about 20-40 min |
| `adam-rknow` index build | 82 s last time, peak RSS 4.5 GB, plus a few minutes for aux tables |
| extra disk while v2 and v3 coexist | about 26 GB (clean 7.8 + fetch 8.6 + index 9.2) |
| total wall | about 2.5-4 h, no GPU |

## Knock-on effects

- Doc and passage ids change. Ranking models (`rknow_rank.json`, `rknow_prr.json`) use features, not ids, so they carry over. The locked R-KNOW numbers should still be re-measured on v3.
- Titles and page ids keep coming from Wikipedia, so Wikidata links by title hash survive.
- The text is CC BY-SA 4.0 either way; attribution stays "Wikipedia".

## Execution (WIKI-NUMBERS, 2026-09-28)

Script: `scripts/ks_wiki_numbers.py` (`dl`, `clean`, `rknow`, `holes`, `verify`, `swap`, `purge`, plus `run --view`, which runs any script against the post-swap layout without touching the live paths). Everything ran CPU-only, with at most 2 workers, under `systemd-run --user --scope -p MemoryMax=8G -p CPUQuota=200% nice -n 15`.

### What was built (next to the live data, nothing swapped)

| path | size | what |
|---|---|---|
| `data/ks/clean/wikipedia_fw/` | 8.3 GB | cleaned text, 56 shards, same columns as `clean/wikipedia` plus `text_src`, `dc_text`, `dc_n`, `dc_groups`; `ATTRIBUTION.txt`, `license_manifest.json` |
| `data/ks/fetch/wikipedia_fw/` | 9.1 GB | fetch store + `titles.npz` |
| `data/ks/index/wikipedia_fw/` | 2.9 GB | KS page index. It has to match the fetch store's pages, because `ray_learn.local` / `KSIndex` read `fetch/wikipedia`. |
| `data/ks/index_rknow/wiki_v3/` | 9.8 GB | R-KNOW index incl. `d_first`, `bt_*`, `al_*`; `meta.json` records sources, FineWiki revision, licence, attribution, held-out hash and decontam stats |
| `data/ks/state_fw/` | 46 MB | per-shard stats, `clean_summary.json`, `holes_*.json`, `verify.json`, `ab_nodc.json` |

- **Source:** `HuggingFaceFW/finewiki` `data/enwiki` at revision `8bd13e72e6a002407649b3e898535f42ceb1aeb9`: 15 files, 35.1 GiB, 6,614,655 rows. The dataset card lists CC BY-SA 4.0 and GFDL. The HF parquet has no page ids, so articles were matched by exact title: 5,367,768 FineWiki rows matched a title in shards 00000-00040.
- **Text:** `fwtext()` as in shards 00041-00055, plus **markdown tables removed**. On a 7.7k-article sample, keeping tables adds +34% passages; without tables it is +4.8%. Tables were never in the 20231101 text.
- **Per article:** FineWiki text replaces the old text when it passes the unchanged KS `clean()` (neutral filter, hygiene, tail cut, PII masking). Otherwise the old 20231101 text is kept.

| shards 00000-00040 (5,601,638 docs) | count |
|---|---|
| FineWiki text | 5,140,552 (91.8%) |
| old text kept: no FineWiki page with that title | 315,303 |
| old text kept: FineWiki text < 200 bytes (clean stubs; the old text passed only because of its reference/category tail) | 145,590 |
| old text kept: nav / filter | 192 / 1 |
| exact duplicates dropped (all shards) | 205 |

- **Held-out set:** identical. 2,883 docs, sha256 of the sorted `shard\ttitle` list is `48da4660...03dc85`, old and new. `heldout/wikipedia` is untouched.
- **Decontam:** `amni.compute.decontam` default index (key `6e7f096b72b2`, 13-gram, 102 eval files). The article is kept and each sentence overlapping a hit is removed. That touched 17,157 docs and removed 25,298 sentences. About 10k hits each come from the chat6c-chat10j sets (boilerplate stub sentences), 3,095 from chat4 (e.g. US election boilerplate), 2,080 from code_eval (month rows) and 1,531 from adam_t. 380 of the 395 `rc_dev.json` source articles were scrubbed, because rc_dev's passages were copied out of `wiki_v2`.
- **Raw downloads:** each FineWiki file was deleted right after conversion (35.1 GiB), then the converted parts (7.7 GiB) were deleted after the clean build. Both are logged in `data/ks/logs/deletions.log`.

### Gates

| gate | wiki_v2 | wiki_v3 | result |
|---|---|---|---|
| readiness: passages within ±2% | 39,272,597 | 41,627,831 (+6.0%) | **fail** |
| number holes, lead passages 00000-00040 (`_HOLE`, n=3000) | 5.83% | 1.30% (FineWiki-sourced 0.98%, old-text 5.0%) | pass |
| broad template gaps / glued date ranges | 13.8% / 0.40% | 5.2% / 0.07% | |
| retrieval dev gold-in-context, know all/tune/hold | 84.8 / 84.6 / 85.3 | 84.8 / 84.6 / 85.3 | pass (multi 87.1 -> 88.7) |
| held-out set identical by hash | | identical | pass |

The +6.0% is genuine content (rendered templates, units, dates; +5% words on the sample), not duplication. The swap was not done, and `swap` refuses until `verify.json` passes.

### R-KNOW numbers (`scripts/rknow_analyze.py`, 400 questions per split; acc / R@1 / R@30)

| split | wiki_v2 | wiki_v3 (decontam) | wiki_v3 without decontam (A/B) |
|---|---|---|---|
| tune | 27.5 / 33.5 / 72.5 | 20.25 / 23.75 / 61.75 | 26.25 / 33.5 / 71.75 |
| hold | 22.0 / 35.25 / 75.0 | 17.75 / 23.75 / 64.0 | 20.75 / 33.5 / 74.5 |
| hold2 | 20.0 / 27.0 / 75.0 | 17.75 / 27.75 / 75.75 | 17.5 / 27.75 / 76.0 |
| hold3 | 21.25 / 30.25 / 74.25 | 18.5 / 32.25 / 74.0 | 18.5 / 32.0 / 74.0 |
| locked NQ (v20_m13) | 19.0 / 32.5 / 78.25 | 19.5 / 32.0 / 76.75 | 19.75 / 32.0 / 76.75 |

- The tune/hold retrieval drop is decontam. 46 of 49 (tune) and 48 of 54 (hold) R@30 losses are questions whose `rc_dev` gold passage was scrubbed. Without decontam, R@k matches v2.
- Independent of decontam, answer accuracy falls 1.25-2.75 points on tune/hold/hold2/hold3 while R@k is flat. The gold passage still ranks first, but the extractor/ranker, fitted on v2 text statistics, picks another candidate (e.g. "Bruce Willis" for The Sixth Sense). The locked NQ split is flat (+0.5). Nothing was tuned on these numbers.
- `tests/test_adam_t_route.py`: 33 passed on live `wiki_v2`, and 33 passed against `wiki_v3` via `run --view` (real-index test included).

### Decisions needed

1. Accept +6.0% passages as the readiness baseline, or keep ±2%. If ±2% stays, v3 cannot pass as long as FineWiki text is used.
2. Decontam against retrieval text. The shared decontam removes `rc_dev` passages (drawn from wiki_v2 itself) and ~10k boilerplate stub sentences, which costs tune/hold R@30 about 10 points. Options: keep it as built, exempt `rc_dev.json`-style passage sets, or skip decontam for the retrieval index. `wiki_v3_nodc` rebuilds in about 15 min: `ks_wiki_numbers.py fetch --variant _nodc && rknow --variant _nodc`.
3. Whether to accept the 1.25-2.75 point extraction loss, or refit `rknow_rank.json` / `rknow_prr.json` on v3 before swapping.
4. Swap: `ks_wiki_numbers.py swap` exchanges `fetch/wikipedia`, `index/wikipedia` and `index_rknow/wiki_v2` with the new dirs (renameat2 RENAME_EXCHANGE), then `purge old` deletes and logs the old copies (~20.7 GB). `clean/wikipedia` stays, because `store_clean` and the ADAM-T stream builder's held flags are aligned to it. The new text is in `clean/wikipedia_fw`.
5. Shards 00041-00055 still contain markdown tables (unchanged). Decide whether to strip them there too.
