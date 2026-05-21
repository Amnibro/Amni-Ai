# Installing Adam

Adam runs locally on your machine. The public clone is a complete, working install — no separate runtime fetch required.

> **Status (2026-05-17, iter29):** Source-available under CC BY-NC 4.0. Includes Reffelt GF(17) decomposition, AsimovLayer, Rust kernels (prebuilt Windows .pyd; Mac/Linux compile from `amni_kernels/src/`). The `from amni.runtime import fetch` call referenced in older docs is no longer required.

## Hardware minimums

| Use | VRAM | RAM | Disk |
|---|---|---|---|
| Adam only (Gemma-4 E2B GF17) | 4 GB | 8 GB | **~20 GB** (5 GB transferred via hf-xet dedup) |
| Adam + web-crawler tier | 8 GB | 16 GB | ~25 GB |
| CPU-only fallback | 0 | 16 GB | ~20 GB |

Tested on **AMD Radeon RX 7800 XT** (ROCm 7.2) and **NVIDIA RTX 3060+** (CUDA 12+). **CPU-only works** but inference drops to ~1 tok/s — install.py auto-detects and warns.

### Installing on an external drive (USB / NVME)

The ~20 GB bake goes wherever you point. Two options:

```bash
# move just the bake (model weights)
python install.py --bake-dir E:/external/Amni/bake

# move EVERYTHING (config, bakes, lessons, conversations)
python install.py --home E:/external/Amni
```

Or set env vars and run normally:

```bash
# Linux/Mac
export AMNI_BAKE=/mnt/external/Amni/bake
python install.py

# Windows PowerShell
$env:AMNI_BAKE = "E:\external\Amni\bake"
python install.py

# Windows cmd
set AMNI_BAKE=E:\external\Amni\bake
python install.py
```

The drive needs to be mounted before launching the server each time. Performance-wise: NVMe is ideal (streaming tier-loads happen during inference), spinning disk works but adds noticeable latency on first-token.

---

## Path A — Standalone (recommended)

The fastest way to talk to Adam right now. **Two installer options — pick one:**

### Option 1: GUI installer (easiest)

```bash
git clone https://github.com/Amnibro/Amni-Ai
cd Amni-Ai
python installer.py
```

Opens a window where you pick:
- **Install drive** (with free space shown per drive — important because the bake is ~20 GB; pick a drive with >25 GB free)
- **GPU vendor** (auto-detected, override available)
- **Persona**, launch options

Click "Start install" and the GUI streams every step's output to a console pane. The full install transcript auto-saves to `<install-dir>/install_log_<timestamp>.txt`. If anything fails, click **"Email log to support"** — it opens your mail client prefilled to `amnibro7@gmail.com` with platform info and the last 30 lines of output; attach the saved log file before sending.

On Windows you can double-click `installer.bat` instead of typing the python command. On Mac/Linux: `./installer.sh`.

The GUI installer is a thin pywebview wrapper (~5 MB, auto-installed on first run) that orchestrates `install.py` under the hood with the options you picked.

### Option 2: Headless CLI

```bash
git clone https://github.com/Amnibro/Amni-Ai
cd Amni-Ai
python install.py               # auto-downloads Gemma-4 E2B GF(17) bake from HuggingFace (~20 GB, one-time)
```

`install.py` handles **everything**: venv creation, vendor-correct PyTorch (CUDA / ROCm / CPU auto-detect), pip install of dependencies, optional Rust toolchain + amni_kernels native build, first-run bake download from HuggingFace, then launches the server and opens your browser at `http://127.0.0.1:11434/`. You do **not** need to manually create a venv or `pip install -r requirements.txt` first.

Headless install location can be picked with `--home <path>` (e.g. `--home D:/Adam`). Without it, the bake goes to `~/.amni-ai/` (your user-profile drive, usually C: on Windows — fine if you have >25 GB free there, but if you're on a small system SSD, the GUI installer is the easier way to put the bake elsewhere).

On Windows you can double-click `install.bat` instead of typing the python command. On Mac/Linux: `./install.sh`.

Adam's chat UI loads with the Rikku persona by default. Switch persona via the topbar dropdown (14 built-in: Mentor, Rikku, Yoda, Scientist, Sherlock, Hypatia of Alexandria, etc.).

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
pip install -r requirements.txt
pip install pytest  # if you want to run tests/_v6_*.py probes
```

The Python source for persona, skills, CoT scaffolds (iter15-26), perturb loop, semantic intent screen, sandbox-validated code path, lesson promotion, FastAPI server, web UI, and Ollama-compat endpoints is all in the public repo and freely editable. You can:

- Add new skills in `amni/serve/skills.py` (regex detector + handler — Adam dispatches automatically)
- Add new personas via `POST /personas` (no code changes needed)
- Extend seed lessons by dropping new modules into `amni/seeds/` and importing them in `amni/seeds/__init__.py`
- Modify the CoT scaffolds in `amni/serve/agent.py` (`_COT_CODE`, `_COT_MATH`, etc.)
- Tune the semantic intent layer thresholds via `AMNI_INTENT_COS` env var
- Add new SSE event handlers in `amni/serve/web.py` for any new agent events

What's NOT in the public source:
- Reffelt 4-tier GF(17) decomposition + TMU dispatch (`amni/compute/`, `amni/core/`)
- Rust kernels (`amni_kernels/`)
- AsimovLayer + LawKeeper file-integrity sealing (`amni/a1/asimov.py`, `amni/a1/lawkeeper.py`)
- Streaming weight loader (`amni/inference/streaming_linear.py`)
- Training pipeline (`amni/training/`)

When the runtime blob ships, the encrypted bundle decrypts into `~/.amni_runtime/` and the streaming-chat backend connects to it transparently.

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

**`ImportError: cannot import name 'amni_kernels'`** — on Mac/Linux, the prebuilt Windows `.pyd` won't load. Compile from source: `cd amni_kernels && cargo build --release` then move the `.so` next to `amni_kernels/__init__.py`. Requires Rust toolchain (rustup.rs).

**`CUDA out of memory` or `HIP out of memory`** — set `AMNI_BUDGET_MB=3000` in your environment before launch. Adam will stream more aggressively from disk. Sub-3GB means you'll feel it on long responses.

**Port 8001 already in use** — pass `--port <N>` to `amni_serve.py`. The Ollama drop-in path uses 11434 by default if you launch Ollama-compatible.

**No GPU detected** — Adam falls back to CPU automatically but inference drops to ~1 tok/s. Install ROCm (AMD) or CUDA (NVIDIA) drivers + matching PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

**Persona is wrong / always Rikku** — pass `--default-persona mentor` (or any of the 14 names) at launch. Per-session override via `/persona <name>` slash command in chat.

---

## License + commercial use

Adam is **CC BY-NC 4.0** — free for personal, research, educational, and non-profit use. For commercial use (selling Adam access, embedding in a paid product, etc.), email **amnibro7@gmail.com** for a commercial license.

The runtime blob is encrypted + signed. Tampering with the signature invalidates your install. The `license_key='free-noncommercial'` value above is the public key for personal use; commercial keys are issued per-organization.
