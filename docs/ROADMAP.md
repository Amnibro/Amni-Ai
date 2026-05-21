# Amni-Ai Roadmap (post v6.8.10)

Source: the user's direction during the /loop gap-fixing session, 2026-05-20.

Order is deliberate — each tier depends on the prior being solid.

---

## Tier 0 (in progress): memory completeness

**Goal:** Adam feels like a real assistant who remembers across sessions and proactively recalls relevant context.

**Status from gap-fixing loop:**
- ✅ Typed user facts (`user_profile.json`) — name, location, favorites, role, workplace
- ✅ Corrections persist + override stale LUT entries (`conversation_notes.json`)
- ✅ Per-session conversation atlas with semantic recall
- ✅ Auto-promoted Q&A to lesson bank
- ✅ Contextual web query enrichment (uses prior turn + profile location)

**Remaining memory gaps:**
- Proactive cross-session recall — *"Last time we discussed X..."* without explicit prompt (in progress)
- Conversation summarization — rolling 1-2 sentence summary per session, surfaced on new session
- Long-context degradation — sessions >20 turns lose detail; need turn compression
- Active-project thread tracking — `conversation_notes.json` gains `active_projects` field
- Personality coherence post-stream — drift check after CoT
- Conversation-atlas PCA cold-start (currently needs 8 samples; meta-direct scan covers but is per-fact-type)

Memory is the foundation everything below leans on. Ship it before moving on.

---

## Tier 1: Voice — bidirectional conversation with high-quality TTS

**Goal:** Talk to Adam, Adam talks back. Jarvis from IronMan style.

**Architecture:**
```
Mic → VAD (silero) → Wake-word (openWakeWord, "Hey Adam") → STT (faster-whisper)
                                                                    ↓
                                                            Adam's existing brain (no change)
                                                                    ↓
TTS (Piper or XTTS v2) ← sentence-bounded chunking ← Adam's SSE token stream
                                                                    ↓
Speakers
```

**Components — all CC/MIT, all pre-downloadable:**
- `silero-vad` — voice activity detection, runs on CPU, ~10ms latency
- `openWakeWord` — custom "Hey Adam" wake word, CPU, ~50ms
- `faster-whisper` large-v3-q8 — STT on 7800 XT, ~300-500ms per utterance
- `piper-tts` (default) or `XTTS v2` (for voice cloning) — TTS, pre-downloaded voice models
  - Piper: ~50ms first-audio latency, robotic but clear
  - XTTS v2: ~200-400ms first-audio, can clone any voice from 6s sample (Rikku, Jarvis, your own)
- `sounddevice` — mic/speaker I/O

**Adam-side changes needed:**
- New `/voice/stream` endpoint that wraps `/chat/stream` with audio I/O
- Sentence-bounded token chunking (TTS reads complete sentences, not 24-char slices)
- Interrupt handling — if user starts talking while Adam is, cancel TTS playback
- Voice-mode length penalty (shorter responses) — passes `max_new_tokens=80` instead of 180
- Optional persona-to-voice mapping (`personas.json` gains `voice_model` field)

**Targets:**
- End-of-user-speech → start-of-Adam-voice: **<3s**
- Wake-word false-trigger rate: <1 per hour
- TTS naturalness: better than eSpeak, ideally XTTS-quality
- Pre-downloadable on first run via `install.py --voice` flag

**Effort estimate:** 5-10 person-days for working prototype.

---

## Tier 2: Jarvis HUD — visual dashboard with rich widgets

**Goal:** IronMan-style heads-up display alongside the chat — Adam shows you maps, weather, stocks, photos, infographics, charts, code diffs.

**Architecture:**
```
Adam decides "this answer is best shown visually" 
            ↓
Emits widget intent: <widget type="map" zoom="12" center="<user-region>"/>
            ↓
Frontend renders widget in HUD panel alongside text response
            ↓
Widgets can be interactive (click to drill down, hover for tooltip)
```

**Widget primitives needed:**
- `<map>` — Leaflet or MapLibre, OpenStreetMap tiles, GeoJSON overlays
- `<weather>` — OpenWeatherMap or NWS API, current + 7-day forecast cards
- `<stocks>` — Yahoo Finance scrape or AlphaVantage API, ticker + sparkline
- `<chart>` — Plotly or Chart.js, for any data viz Adam wants to render
- `<image>` — inline image with caption, source attribution
- `<code-diff>` — Monaco editor with diff highlighting
- `<table>` — sortable, filterable data table
- `<timeline>` — events on a horizontal axis
- `<3d-model>` — three.js for 3D viz when relevant

**Adam-side changes:**
- New widget-emission protocol: Adam's responses can include `<widget>` tags or a parallel `event: widget` SSE stream
- Tone-atlas decides when visual is warranted (location → map, "weather" → weather widget, stock symbol → stocks widget, etc.)
- Skills layer extension: `weather_skill`, `stocks_skill`, `chart_skill`, `geocode_skill` — all return widget specs alongside text
- For "show me the path from A to B" type queries: route through agentic mode → multiple widgets

**Frontend (HUD itself):**
- Could extend existing `amni/serve/web.py` chat UI
- Or build standalone Electron/Tauri app with proper full-screen HUD aesthetic
- Glassmorphism + neon accents to match IronMan vibe
- Voice activation indicator
- Floating widget panels, drag-to-rearrange

**Effort estimate:** 3-6 weeks. Bulk is frontend.

---

## Tier 3: Standalone coder — build Adam into Amni-Code by default

**Goal:** Adam becomes the inference backend for Amni-Code (the user's Rust agent IDE). Out-of-the-box install: `git clone amni-code; cargo run` → working AI coding agent, no API keys needed, competitive with Cursor/Codex/Gemini Code.

**Why it can compete:**
- Local-first (no API costs, no rate limits, no telemetry)
- GF(17) lossless quantization (cosine=1.0 roundtrip — no quality loss vs fp16)
- Agentic loop already exists (`run_goal_stream` + skill orchestration)
- Auto-promotion to lesson bank (gets better with use)
- Persona system (can be Rikku, Mentor, Pirate, etc.)

**What needs to land:**
1. **Amni-Code embeds Adam** — bundle `Amni-Ai/.venv` + Adam bake as a dependency. First-run install fetches GF(17) bake (~5GB) automatically.
2. **Tighter code-task ergonomics** — `_COT_CODE` scaffold polish, better assert generation, tighter self-test perturbation
3. **IDE-specific skills** — `goto_definition`, `find_references`, `extract_symbols`, `apply_refactor` — beyond the generic file ops
4. **Workspace awareness** — Adam reads the open Cargo.toml / package.json / pyproject.toml to understand the project
5. **Diff-mode output** — when editing, Adam emits unified diffs not full files (cheaper, safer)
6. **Streaming code chunks** — currently agentic emits per-step; for IDE feel, stream chunks within a step (e.g., file_write tokens)
7. **Multi-file context** — open relevant files via `file_read` automatically based on imports
8. **Linter/test integration** — after writing code, run linter and tests, iterate on failures
9. **Codebase indexing on first launch** — `scan` skill ingests project, builds project-specific lesson bank
10. **Code-specific evaluation** — HumanEval / MBPP / SWE-bench benchmarks to validate "competitive with Codex" claim

**Path:**
- Phase A: harden Adam-as-Amni-Code-backend (the embed, install flow)
- Phase B: ship the IDE-specific skills + diff-mode output  
- Phase C: benchmark against HumanEval, publish numbers
- Phase D: marketing position — "self-hosted, lossless, no-API-keys coding agent"

**Effort estimate:** 4-8 weeks for Phase A+B. Benchmarking adds 2 weeks. Marketing/launch is separate.

---

## Sequencing

```
NOW:  Tier 0 (memory completeness) — finish proactive recall + summarization + project threads
      ↓
WEEKS 2-3:  Tier 1 (voice) — STT/TTS bolt-on, fast win, validates real-time architecture
      ↓
WEEKS 4-9:  Tier 2 (HUD) — widget protocol + frontend (longest)
      ↓
WEEKS 6-12 (overlaps Tier 2): Tier 3 (standalone coder) — ergonomic polish + Amni-Code embed
      ↓
WEEK 12+:  Benchmarks, public launch, integrate with robot chassis (separate roadmap)
```

Voice + coder can ship without HUD. HUD is the most visible but least functional of the three.
