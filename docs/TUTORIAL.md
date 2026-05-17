# Adam Tutorial — first 30 minutes

A walkthrough of what makes Adam different from a typical chat model. By the end you'll have:
- Talked to Adam in 3 personas
- Watched it write code, run the code in a sandbox, and check its own tests
- Seen a query go from 180s (cold generate) → 80ms (lesson cache hit)
- Hit the harm-intent screen with a phrased jailbreak

Assumes you've finished [`INSTALL.md`](INSTALL.md). Server is running at `http://127.0.0.1:8001/`.

---

## 1. Open the chat

Browser → `http://127.0.0.1:8001/`. You get a single-pane chat with the **Mentor** persona by default (calm, professional, mid-length responses).

Type: `Hey, what's up?`

You should get a friendly greeting back within ~2 seconds. The tier badge at the bottom of Adam's bubble shows `tier1_lut` (template match — cached greeting) or `tier1_persona_lut` (persona-cached). Sub-100ms.

---

## 2. Switch personas

Click the persona dropdown in the topbar. Pick **Rikku**.

Type: `Hey, what's up?`

Same question, different voice. Rikku's an upbeat Final Fantasy X character — energetic, uses some Al Bhed words ("Rao!", "Fryd's ib!"). Same intent classifier underneath, different tone overlay.

Try **Yoda** next. Then **Sherlock Holmes**. Then **Hypatia of Alexandria**. The 14 built-ins span casual to formal, modern to historical, terse to thorough. Each carries its own system prompt + length preference + tone-atlas overlay applied to the same underlying Adam reasoner.

---

## 3. Watch Adam write + verify code

Switch back to **Mentor** (CoT scaffolds are most visible there).

Type: `Write a Python function to compute the Nth Fibonacci number using memoization, then print fib(10).`

You'll see (in order, on the bot bubble):

1. **`META`** badge appears immediately: `session_id`, persona = Mentor
2. **Second `META`** after ~3s: `cot: true, category: code`
3. **First token** at ~80-180s (the Gemma cold-start cost — subsequent code questions stream faster as the model stays warm)
4. **Tokens stream in** — you see Adam's CLARIFY → APPROACH → ```python``` block → TESTS section → COMPLEXITY
5. **`exec`** badge fires: `sandbox exit 0`, stdout shows `Fib(10): 55`
6. **`tests`** badge: `tests ✓ 4 div=0.67` (Adam wrote 4 distinct asserts; diversity score 0.67)
7. **`learned`** badge: `learned ✓ #1793` (the question + answer is now in the permanent lesson bank, 1793 lessons total)
8. Final tier: `tier_persona_cot_run_tests_ok_promoted`

**Try the same question again right now.** Same browser, send identical text.

The response comes back in under **100 milliseconds**. Tier is `tier1_5_semantic_lesson` — the lesson bank caught it. **~2400× speedup.**

This is the iter18 promotion loop: code that passes its own asserts becomes permanent knowledge. Adam gets faster + better the more you use it.

---

## 4. Watch the trial-and-error loop

Type: `Write a Python function that returns the prime factorization of N as a list of (prime, exponent) tuples. Test it on factorize(84).`

Watch the badges:

- `exec` fires — `[(2, 2), (3, 1), (7, 1)]` — correct for 84 = 2²·3·7
- `tests ✓ 3 div=0.50` — all 3 asserts pass

But sometimes Adam's first shot is wrong. If you get a query where the function compiles but asserts fail, you'll see:

- `exec` — sandbox runs, rc=0, stdout shows something
- `tests ✗ N div=0.X` — one or more asserts fail
- `perturb` badges appear: `SMALL exec_failed` → `MEDIUM tests_failed` → `LARGE all_passed`
- Final tier: `tier_persona_cot_run_perturb_medium_promoted`

Adam's trial-and-error loop: when self-tests fail, it re-attempts with progressively larger code changes (SMALL = tweak a value, MEDIUM = restructure a loop, LARGE = different algorithm). Only after a perturbation passes both exec AND asserts does the answer get promoted.

---

## 5. Use a skill

Adam has 11 skills wired up. Try them:

| Type this | What fires |
|---|---|
| `What time is it?` | `time` skill — returns local ISO timestamp instantly |
| `Compute 47 * 89` | `calc` skill — exact integer math via Python eval-safe |
| `Read file scripts/amni_serve.py` | `file_read` skill — returns first ~2KB |
| `Search my lessons for fibonacci` | `mem` skill — semantic LUT lookup with cosine scores |
| `Find online: latest python version` | `web` skill — DuckDuckGo + trafilatura + Adam distillation |
| `Run python: print(sum(range(100)))` | `run_python` skill — sandbox executes, returns stdout |

Skills are dispatched **before** Adam thinks. The regex detector fires first; if no skill matches, the message goes to the LLM. This is why "What time is it?" returns in 40ms instead of 80 seconds.

---

## 6. Multi-layer safety screening

Adam runs every incoming message through three stacked safety layers before any inference happens:

1. **Lexical filter** — fast regex on obvious jailbreak phrasing
2. **GF(17) hash pattern matcher** — catches lexical variants the regex misses
3. **Semantic intent screen** — MiniLM embeddings + cosine distance against a harm-intent canonical bank, with pre-decoders for common encoding tricks

Blocked messages return a category-specific refusal in **~40ms** (no LLM call). The tier badge will read `tier_intent_block_<category>` so you can tell at a glance whether the block was lexical, hash-based, or semantic.

Benign queries pass through unaffected — the layers have measured zero false-positive rate on a held-out benign set. We don't publish specific bypass-rate breakdowns because that would amount to a roadmap for attackers; if you're doing legitimate security research on Adam's defenses, email `the maintainer (via GitHub)` for the internal harness.

---

## 7. Inspect the stats

Click the engine-drawer icon in the topbar (or `curl http://127.0.0.1:8001/stats`). You'll see:

```json
{
  "lessons_n": 1793,
  "auto_margin": 0.05,
  "tier_counts": {
    "tier1_lut": 8,
    "tier1_5_semantic_lesson": 2,
    "tier_persona_cot_run_tests_ok_promoted": 1,
    ...
  },
  "token_counts": {...},
  "svc_boot_s": 16.8
}
```

The `tier_counts` shows you what's happening under the hood — how many queries hit each tier of Adam's resolution chain. Over time, the higher tiers (LUT, semantic lesson) handle more of your traffic as the bank grows.

---

## 8. The persona-as-PTEX trick

Want a custom personality? Tell Adam:

```
/persona create casanova: a charming 18th-century rogue with a flair for poetic exaggeration.
```

(That's a slash-command inside the chat, OR via `POST /personas` with JSON body.)

Adam compiles a tone-atlas overlay from your description, stores it as a PTEX entry in `experiences/personas.json`, and makes it available in the dropdown. Personas you create persist across sessions. Share `experiences/personas.json` with a friend → they have your personas.

---

## What's next

Once you're comfortable:

- Wire Adam into your editor via Ollama drop-in (`Continue.dev`, `Cursor`, `Cline`) — see [`INSTALL.md` § Path B](INSTALL.md#path-b--ollama-drop-in)
- Contribute lessons back: any answer Adam generates can be exported via the upcoming PTEX federation (iter24+) for community knowledge sharing
- Extend the skill set: write a `weather` or `gh_pr_summary` skill in `amni/serve/skills.py` and register it — Adam will dispatch to it automatically

Architecture deep-dive: [`docs/architecture.svg`](architecture.svg) shows the side-by-side comparison vs traditional LLM stacks.
