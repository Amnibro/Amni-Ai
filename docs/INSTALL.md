# Installing Adam

Adam runs locally on your machine. Three install paths depending on what you want.

## Hardware minimums

| Use | VRAM | RAM | Disk |
|---|---|---|---|
| Adam only (Gemma-4 E2B GF17) | 4 GB | 8 GB | 5 GB |
| Adam + web-crawler tier | 8 GB | 16 GB | 10 GB |
| Adam + persona overlays + lesson bank | 4 GB | 16 GB | 8 GB |

Tested on **AMD Radeon RX 7800 XT** (ROCm 7.2) and **NVIDIA RTX 3060+** (CUDA 12+). CPU-only inference works but is slow (~1 tok/s).

---

## Path A — Standalone (recommended)

The fastest way to talk to Adam right now.

```bash
git clone https://github.com/Amnibro/Amni-Ai
cd Amni-Ai
python -m venv .venv
.venv/Scripts/activate          # on Linux: source .venv/bin/activate
pip install -r requirements.txt
python -c "from amni.runtime import fetch; fetch(license_key='free-noncommercial')"
python scripts/amni_serve.py --seed --cors
```

The first `fetch()` call downloads the encrypted GF(17) runtime blob (~80 MB) from `amni-scient.com/adam/runtime/` and installs it into `~/.amni_runtime/`. This is a one-time cost.

Open `http://127.0.0.1:8001/` in a browser — Adam's chat UI loads with the Rikku persona by default. Switch persona via the topbar dropdown (14 built-in: Mentor, Rikku, Yoda, Scientist, Sherlock, Hypatia of Alexandria, etc.).

---

## Path B — Ollama drop-in

If you already use **Open WebUI**, **Continue.dev**, **LangChain Ollama backend**, or any client that speaks the Ollama HTTP API, Adam slots in unchanged. Adam's `/api/*` endpoints are shape-matched to Ollama's spec.

Start Adam on the standard Ollama port (replace 11434 if you've changed it):

```bash
python scripts/amni_serve.py --seed --cors --port 11434
```

Now point your Ollama client at `http://127.0.0.1:11434`. Adam shows up as `adam:e2b-gf17` plus common aliases (`llama3`, `qwen`, `mistral`) so clients with hardcoded model names still resolve.

### Open WebUI

1. Install Open WebUI: `pip install open-webui && open-webui serve`
2. Open `http://localhost:8080`, go to **Settings → Connections → Ollama API**, set base URL to `http://127.0.0.1:11434`
3. Refresh model list — `adam:e2b-gf17` appears. Select it.
4. Chat normally. CoT scaffolds, self-tests, lesson promotion all fire — you'll see badges in Adam's responses (when using the native UI; Open WebUI shows the rendered text).

### Continue.dev (VS Code)

In your VS Code settings.json:

```json
"continue.models": [{
  "title": "Adam",
  "provider": "ollama",
  "model": "adam:e2b-gf17",
  "apiBase": "http://127.0.0.1:11434"
}]
```

Hit `Ctrl+L`, ask Adam to refactor a function. The sandbox auto-runs and validates any Python it generates.

### LangChain

```python
from langchain_ollama import ChatOllama
adam = ChatOllama(base_url="http://127.0.0.1:11434", model="adam:e2b-gf17")
print(adam.invoke("Write a Python function that returns the Nth Fibonacci number.").content)
```

---

## Path C — Build from source (advanced)

If you want to extend the persona system, add skills, or modify the serve layer.

```bash
git clone https://github.com/Amnibro/Amni-Ai
cd Amni-Ai
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt -r requirements-dev.txt
```

You'll still need the runtime blob (`amni.runtime.fetch`) — the Reffelt math + Rust kernels ship as a signed encrypted bundle. The Python source for persona, skills, CoT scaffolds, perturb loop, semantic intent screen, and FastAPI server is all in the public repo and freely editable.

### Where to look

| What | Where |
|---|---|
| FastAPI server entry | `scripts/amni_serve.py` |
| Agent orchestration (skill dispatch, persona, CoT) | `amni/serve/agent.py` |
| Persona system + tone atlas | `amni/serve/persona.py`, `amni/serve/tone_atlas.py` |
| Skills (run_python, web, file_read, etc.) | `amni/serve/skills.py` |
| Web UI | `amni/serve/web.py` |
| Streaming inference wrapper | `amni/inference/streaming_chat.py` |
| Semantic LUT (lesson bank lookup) | `amni/inference/semantic_ptex_lut.py` |
| Harm-intent screening | `amni/a1/semantic_intent.py` |
| Seed corpora (1792 lessons) | `amni/seeds/` |
| Public-API probe tests | `tests/_v6_*.py` |

---

## Verifying it works

Three quick smoke tests after install:

```bash
# 1. Health check
curl http://127.0.0.1:8001/healthz
# {"status":"ok","lessons_n":1792,"skills_n":11,"version":"6.0.0"}

# 2. Single-turn chat
curl -X POST http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the capital of France?"}'

# 3. Streaming chat with code generation (watch for tier_persona_cot_run_tests_ok_promoted)
.venv/Scripts/python tests/_v6_8_promotion_e2e.py
```

The third test asks Adam to write a function, runs the function in a sandbox, validates its own asserts pass, and persists the (question→working-code) pair into the lesson bank. Asking the same question again returns in **~80ms** via LUT hit.

---

## Troubleshooting

**`RuntimeNotReadyError: Reffelt runtime not installed`** — run `python -c "from amni.runtime import fetch; fetch(license_key='free-noncommercial')"` to download the encrypted blob. Requires internet on first run only.

**`CUDA out of memory` or `HIP out of memory`** — set `AMNI_BUDGET_MB=3000` in your environment before launch. Adam will stream more aggressively from disk. Sub-3GB means you'll feel it on long responses.

**Port 8001 already in use** — pass `--port <N>` to `amni_serve.py`. The Ollama drop-in path uses 11434 by default if you launch Ollama-compatible.

**No GPU detected** — Adam falls back to CPU automatically but inference drops to ~1 tok/s. Install ROCm (AMD) or CUDA (NVIDIA) drivers + matching PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

**Persona is wrong / always Rikku** — pass `--default-persona mentor` (or any of the 14 names) at launch. Per-session override via `/persona <name>` slash command in chat.

---

## License + commercial use

Adam is **CC BY-NC 4.0** — free for personal, research, educational, and non-profit use. For commercial use (selling Adam access, embedding in a paid product, etc.), email **amnibro7@gmail.com** for a commercial license.

The runtime blob is encrypted + signed. Tampering with the signature invalidates your install. The `license_key='free-noncommercial'` value above is the public key for personal use; commercial keys are issued per-organization.
