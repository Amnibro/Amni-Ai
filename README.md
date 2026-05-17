# Adam

**Texture-native, GF(17)-lossless, self-improving local AI assistant.** Runs in 4 GB VRAM. Drop-in Ollama replacement.

<p align="center">
  <img src="docs/architecture.svg" alt="Adam vs Traditional LLM architecture comparison" width="100%">
</p>

| | Traditional LLM | Adam |
|---|---|---|
| Weights | `.safetensors` float16 | `.ptex` GF(17) RGBA pixel atlas |
| Compression | INT8/FP8 lossy | GF(17) lossless (cosine=1.0 roundtrip) |
| Memory | Full VRAM load (7B = 14 GB) | Progressive tier-stream from NVMe (4 GB) |
| Compute | GEMM (matmul) | TMU lookups (GPU texture hardware) |
| Safety | Keyword filter | 5 immutable laws + semantic intent screen (81% block) |
| Code output | One-shot generation | CoT → exec → self-test → perturb → promote |
| Cross-query memory | None | Permanent PTEX lesson bank (2400× speedup on repeats) |

---

## Quick start

```bash
git clone https://github.com/Amnibro/Amni-Ai
cd Amni-Ai
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python -c "from amni.runtime import fetch; fetch(license_key='free-noncommercial')"
python scripts/amni_serve.py --seed --cors
```

Browser → http://127.0.0.1:8001/ — chat with Adam in the Rikku persona.

**[Full install guide →](docs/INSTALL.md)** · **[Tutorial: first 30 min →](docs/TUTORIAL.md)** · **[Landing page →](https://amni-scient.com/amni-ai)**

---

## What's in this repo

The public source-available skeleton. Everything that's not the proprietary GF(17) math:

- **`amni/serve/`** — FastAPI server, persona system, conversation store, 11 skills, web UI, Ollama-compatible `/api/*` endpoints
- **`amni/inference/`** — Streaming chat wrapper, semantic LUT (lesson bank lookup), KB retriever, web crawler, debugger harness
- **`amni/seeds/`** — 41 corpus modules (1792 lessons) — Adam ships smart out of the box
- **`amni/a1/semantic_intent.py`** — Harm-intent screening (65-phrase bank + 5 encoding pre-decoders)
- **`amni/runtime.py`** — Bootstrap that fetches the encrypted Reffelt blob from amni-scient.com on first launch
- **`tests/_v6_*.py`** — Public-API probe tests + unit smokes for every iter15+ feature
- **`scripts/amni_serve.py`** — Server entry point with `--seed --cors --port` flags

What's NOT here (proprietary, lives in the encrypted runtime blob):
- GF(17) 4-tier decomposition + TMU dispatch (Rust kernels)
- AsimovLayer + LawKeeper (5 immutable laws + integrity sealing)
- Training pipeline + PTEX writer/reader internals

---

## Use it as an Ollama drop-in

If you already use **Open WebUI**, **Continue.dev**, **Cursor**, or any client that speaks Ollama's API, Adam slots in unchanged:

```bash
python scripts/amni_serve.py --seed --cors --port 11434
```

Point your client at `http://127.0.0.1:11434`. Adam shows up as `adam:e2b-gf17` plus aliases (`llama3`, `qwen`, `mistral`) so clients with hardcoded model strings still resolve. CoT scaffolds, self-tests, lesson promotion, and persona switching all work — see [`docs/INSTALL.md § Path B`](docs/INSTALL.md#path-b--ollama-drop-in).

---

## What makes Adam different

**1. Sandbox-validated code output.** Adam doesn't just write Python — it runs the code in a subprocess sandbox, runs your test asserts against the output, and if any assert fails it kicks into a trial-and-error loop with SMALL → MEDIUM → LARGE perturbations until tests pass. Code that ships has been *proven* to work, not just look right.

**2. Self-improving lesson bank.** Every question that produces a test-passing answer gets persisted into a semantic PTEX LUT. The next time you (or anyone using your instance) asks the same or a similar question, Adam returns the cached answer in **~80ms** — a 2400× speedup over re-generating. Adam literally gets faster + better the more you use it.

**3. Semantic intent screening.** Three layers of safety stacked: regex (catches "ignore previous instructions"-style jailbreaks), GF(17) hash patterns (catches lexical variants), and semantic embedding-based screening with pre-decoding for rot13, leet, Al Bhed, and base64 encoding tricks. Total block rate on adversarial paraphrase / encoding / indirection / prefill attacks: **81%** with **0% false positives** on benign queries.

**4. 14 built-in personas.** Mentor, Rikku, Yoda, Sherlock Holmes, Hypatia of Alexandria, Ibn Battuta, Rosalind Franklin, Steve Jobs, Haiku Poet, and more — each with its own tone-atlas overlay applied to the same underlying Adam reasoner. Create your own via `/persona create` slash command; they persist across sessions.

**5. Local. Always.** No API keys. No telemetry. No data leaves your machine. The runtime blob is signed but loads entirely client-side. Your conversations live in `experiences/conversations/` on your disk. Delete them anytime.

---

## License

**CC BY-NC 4.0** — free for personal, research, educational, and non-profit use.

Commercial license inquiries: **amnibro7@gmail.com**

See [`LICENSE`](LICENSE) for full terms.

---

## Status

| | |
|---|---|
| Latest version | v6.8 (iter22, 2026-05-16) |
| Public source launched | 2026-05-16 |
| Reffelt runtime blob pipeline | iter22 in progress |
| PTEX multi-peer federation | iter23-24 |
| amni-scient.com integration | iter22 |

Changelog: [`changelog.md`](changelog.md) tracks every iter.
