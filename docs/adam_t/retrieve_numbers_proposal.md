# Proposal: restore numbers in the wiki_v2 knowledge index

Status: proposal only, needs a CTO OK before any rebuild. Nothing was rebuilt.

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
