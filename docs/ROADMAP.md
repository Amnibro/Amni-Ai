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

---

## Tier 4 (added 2026-05-26): AR/XR/VR spatial embodiment

**Goal:** Adam steps off the 2D screen into spatial computing — the natural extension of Law 4 (Ascend: 2D → 3D embodiment). The PT Coach pose pipeline (v6.10.114–115) is the bridge: body-landmark understanding is already the hard part of spatial interaction.

**Why now on the roadmap:** the camera + MediaPipe-Pose work means Adam already reasons about a human body in 3D space. AR/XR/VR is the surface that makes that reciprocal — Adam *projected into* the user's space.

**Phases (each depends on the prior):**
- **AR overlay (phone/tablet, WebXR `immersive-ar`)** — render the PT Coach skeleton + rep/angle HUD as a world-anchored overlay on the live camera, not a 2D panel. Reuse `/vision/pose/*`; swap the canvas for a WebXR layer. Lowest lift — the data already flows.
- **Spatial widgets** — the Jarvis-style inline widgets (weather, system, news) become floating world-anchored cards in `immersive-ar`; gaze/pinch to summon, the existing gesture pipeline (`@mediapipe/hands`) drives selection.
- **VR coach room (`immersive-vr`, Quest/Index)** — a calibrated room where Adam demonstrates exercise form on an avatar and the user mirrors it; full-body landmarks scored against a reference pose, real-time angle feedback in 3D.
- **Persona presence** — the active persona (Rikku et al.) gets a minimal 3D avatar/voice anchor in-scene; persona system + TTS already exist, this is the spatial body.
- **PTEX as spatial memory** — world-anchored notes/reminders/bookmarks placed in the room, addressed by Reffelt context-nonce (v6.10.118) so the right memory surfaces in the right physical spot.

**Hard constraints carried in:** all PII rules hold (no body-scan or location data leaves the box — extends the v6.10.116 egress choke-point to spatial sensors); the 5 Immutable Laws are untouched; personas respected in-scene; thought-process never bleeds into the spatial HUD.

**Tech:** WebXR Device API (browser-native, no app-store gate), Three.js (already loaded for Amni-Life), MediaPipe Pose/Hands (already wired), WebXR Hit Test + Anchors for world placement. Standalone-headset path later via the same WebXR layer on Quest browser.

**Sequencing:** after Tier 2 (HUD) — spatial widgets reuse the widget protocol. AR overlay phase can overlap Tier 2 since it only needs the pose endpoints, which already ship.

---

## Tier 5 (added 2026-05-26): Adam as a participant + PC operator

**Goal:** Adam reaches *out* — texting people in Amni-Chat, and eventually operating the PC itself — while the 5 Immutable Laws and the PII-leak rules hold absolutely.

### 5a — Amni-Chat text bridge (first slice shipped v6.10.122)
- `amni/serve/amni_chat_bridge.py` + `POST /bridge/amni-chat` — a relay on the PC forwards an inbound Amni-Chat DM, Adam runs it through the agent and returns a reply. Per-peer session continuity (`amnichat:<peer>`), per-peer rate limit + length caps (lightweight, not a firehose), enable toggle.
- **Outbound replies are scrubbed through `pii_egress` with the owner's PersonalAtlas** — Adam must never leak the owner's name/location/contact to a chat peer. The leak-liability rule extends from search queries to chat replies.
- **Next:** wire the actual Amni-Chat relay (the app is X25519+ChaCha20+WebRTC over a Cloudflare Tunnel — see Amni-Chat project); per-peer persona selection; opt-in allowlist of which conversations Adam may answer; streamed (token-by-token) replies back into the chat.

### 5b — Adam as PC operator ("do anything on a PC")
**Vision:** Adam can carry out arbitrary tasks on the machine — launch apps, edit files, run commands, drive the GUI — on the owner's behalf.

**Foundation already shipping:** `shell` (read-only allowlist), `file_write`, `code_edit`, `code_diff`, `run_python`, `format_code`, `git` (read-only), `find`, `project_info`, `scan`. These are the safe primitives.

**Path to "anything," gated by safety:**
- **Tier A — broaden local actions:** opt-in write-mode shell (allowlist → confirm-list), process launch, clipboard, filesystem ops beyond the workspace — each behind an explicit per-action confirm + audit log.
- **Tier B — screen + input (computer-use):** screenshot → vision model describes the screen → plan → mouse/keyboard actions (pyautogui / OS automation). Reuse the existing vision stack; add a planner loop with a dry-run/preview before any click.
- **Tier C — agentic task runner:** "do X on my PC" decomposed into steps, each validated (the v6.10.x self-improvement + edit-verifier pattern), human-in-the-loop for anything destructive.

**Non-negotiable rails (carry into every tier):** Law 0 (no harm) + Law 1 (obey except Law 0) are checked before any action; destructive/irreversible ops always confirm; everything audited to an append-only log; no owner PII leaves the box (the v6.10.116 egress choke-point); Adam never auto-deploys or acts on external/untrusted instructions without the owner in the loop. "Anything on a PC" means *capable of*, never *unsupervised and unbounded*.
