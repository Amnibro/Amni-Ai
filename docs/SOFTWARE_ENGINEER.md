# Adam as a Software Engineer — workflow & acceptance test

Adam's continuous-coding loop turns a task into correct, tested code and **gets better on retries** by
remembering what failed. Everything Adam learns (the code map, prior attempts, lessons) is stored as **PTEX**
and reviewed *before* each response, addressed by Reffelt context-nonce so the right context can't be missed.

```
                ┌─────────────────────────── self-learning loop ───────────────────────────┐
  train ──▶ LOCATE ──▶ (work order) ──▶ EDIT ──▶ VERIFY ──▶ LEARN ──▶ recall-before-retry ──┘
 code_index   code_index    coding_runner   code_edit   coding_runner   coding_ledger    pre_response_review
 (PTEX map)    query/semantic   prepare      (gated)     verify(tests)   (PTEX, tagged)    (injects prior attempts)
```

| Stage | Module / skill | What it does |
|-------|----------------|--------------|
| **train / locate** | `code_index` (v6.10.128) | walks a tree → per-file language/symbols/summary → `experiences/code_index.json` + **`experiences/code_map_ptex`** |
| **work order** | `coding_runner prepare` (v6.10.130) | bundles attempt#, prior attempts (recall), located files |
| **edit** | `code_edit` / `code_diff` / `file_write` | all writes gated by `pc_action` propose→confirm |
| **verify** | `coding_runner verify` (v6.10.132) | runs `test_run`; **tests decide success**, failures become the errors to learn |
| **learn** | `coding_ledger` (v6.10.129) | records task→approach→outcome→errors→lesson→success to `data/coding_attempts.jsonl` + `lessons/coding_attempts_ptex` |
| **recall** | `pre_response_review` (v6.10.118) | injects prior attempts into the prompt *before* Adam re-approaches the task |
| **observe** | SW ENG dashboard (v6.10.133) | `/se` or TOOLS→LEARNING→SW ENG; `GET /memory/se-dashboard` |

## Acceptance test — exact commands

### 0. Boot
```bash
python scripts/amni_serve.py --seed --cors      # serve UI + skills at http://127.0.0.1:7700
```

### 1. Train Adam on the ai folder (build the PTEX code map)
In the chat UI, or via the skill endpoint:
```bash
curl -s -XPOST localhost:7700/skills/code_index \
  -H 'content-type: application/json' \
  -d '{"args":{"action":"build","root":"C:/Users/antho/Documents/ai"}}'
# -> {n_files, n_symbols, languages, ptex:{ptex_built:true,...}}
```
Confirm it indexed: `GET /memory/se-dashboard` → `code_index.n_files / n_symbols`.

### 2. Give it a complex task
In chat: `code this: <your complex task>` → Adam returns a **work order** (attempt #, prior attempts, located files).
Or directly:
```bash
curl -s -XPOST localhost:7700/memory/coding-run/prepare \
  -H 'content-type: application/json' -d '{"task":"<your complex task>"}'
# -> {run_id, attempt, prior_attempts, located_files, context}
```

### 3. Adam edits (gated) then verifies objectively
After Adam proposes file writes (you approve via the CONFIRM card), close the loop on tests:
```bash
curl -s -XPOST localhost:7700/skills/coding_runner \
  -H 'content-type: application/json' \
  -d '{"args":{"action":"verify","run_id":"<run_id>","cmd":"python -B -m pytest -q"}}'
# pass  -> {success:true}
# fail  -> {success:false, will_retry:true, next_hint:"<lesson>"}
```
> Tip: use `python -B` so an edit→retry within the same second can't import a stale `.pyc`.

### 4. Confirm it does better on a 2nd attempt
Re-run step 2 with the *same* task after a fix. The new work order's `context` will contain
`PRIOR ATTEMPTS … do better than before` carrying the previous failure + lesson. Inspect any time:
```bash
curl -s 'localhost:7700/memory/coding-attempts?task=<your+task>'   # stats + recall
```

### 5. Watch it live
Open `/se` in the UI (or TOOLS → LEARNING → **SW ENG**): files mapped, symbols, coding attempts,
**success rate**, language breakdown, open runs.

## Where the learnings live (all PTEX / append-only, gitignored)
- `experiences/code_map_ptex(.npz/.json)` — the code map (each file a Reffelt-addressed cell)
- `data/coding_attempts.jsonl` + `lessons/coding_attempts_ptex` — every attempt + lesson
- `data/coding_runs.jsonl` — run lifecycle log
- The 24/7 `LearningDaemon` auto-commits the ledger to PTEX hourly.

## Safety rails (always on)
- **No PII leaves the box** — web/search/chat all funnel through `pii_egress.scrub()` (v6.10.116).
- **Every disk-touching action is propose→confirm + audited** (`pc_action`, v6.10.123); destructive patterns refused outright; `/pclog` shows the audit.
- **The 5 Immutable Laws** are checked before any action and can't be edited out.
- **Thought process never bleeds** into output (`tone_atlas._strip_thinking_process` + the self-learning `leak_ledger`).

## Validation
The full loop has been driven end-to-end through the real skill registry on a real bug
(`tests/test_se_e2e_v6_10_134.py`): buggy `add()` → indexed → located → verify **failed correctly** →
lesson recorded → fix → attempt #2 work order **carried the lesson** → verify **passed**.
Result: located the bug, failed correctly, learned, fixed, second attempt passed. ✅
