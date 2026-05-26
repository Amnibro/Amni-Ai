# Adam → Jarvis-grade Roadmap

Goal: Adam absolutely INCREDIBLE — autonomous coding, file gen, CLI, Amni-Code, TTS, inline data widgets (Jarvis-style), loop runs, automated content learning, ask-answer-ask coaching, anything-a-user-asks. Inspired by Suryansh Chourasia's Jarvis-CV (gesture + Three.js + MediaPipe).

Working state at iteration-1 start (2026-05-21, v6.9.5):
- 23 skills registered: time, calc, mem, web, file_read, file_write, code_edit, shell, git, test_run, code_diff, project_info, format_code, symbols, rename_symbol, auto_import, parse_error, export_session, prune_sessions, tts, stt, run_python, scan
- OpenAI tool-call protocol via /v1/chat/completions (Amni-Code drives Adam autonomously)
- ConversationAtlas, CodeAtlas, PersonalAtlas (PTEX cell-address LUTs)
- Persona (Rikku default), Ollama compat, MCP, voice/chat endpoint

## What's missing for "INCREDIBLE"

### A. Inline data widgets (Jarvis-style cards in chat)
Adam emits fenced ` ```widget` JSON blocks → frontend renders as glowing cards instead of text dumps. Same shape pattern as tool_protocol — proven, easy to extend.

Skills (iter 1):
- [x] `weather` — Open-Meteo (free, no key)
- [x] `system_stats` — psutil (CPU/mem/disk/GPU)
- [x] `time_card` — time widget with timezone

Skills (later):
- [ ] `news` — top headlines via DDG
- [ ] `stock` — finance lookup
- [ ] `calendar` — local ICS / system calendar
- [ ] `tasks_card` — Adam's own task queue
- [ ] `kbd_shortcuts_card` — context-aware shortcut helper

### B. Jarvis-style web UI shell (iter 2)
Replace plain `amni/serve/web.py` UI with neural-network-background HTML page:
- Neon-glow CSS, animated particle bg, neural-net SVG
- Widget renderer per protocol
- Voice button with live waveform
- Tactical-overlay aesthetic (target lock on focus area)
- Three.js optional 3D scene side panel

### C. Ask-answer-ask coaching mode (iter 3)
New `coach` skill — Socratic tutor loop. Adam asks questions on a topic, tracks user's answers, escalates difficulty, builds curriculum.
- Per-topic mastery model in PersonalAtlas (non-confidential — it's learning state, not PII)
- Hint/expand/skip controls
- Session summary at end

### D. Adam-driven scheduling (iter 4)
New `schedule_loop` skill — Adam creates its own recurring tasks (poll feeds, daily summaries, weekly retros). Lightweight thread + persistent JSONL store. Survives restarts.

### E. Content learning automation (iter 5)
Extend `scan` to handle:
- URLs (fetch → trafilatura → chunk → teach)
- PDFs (pypdf → chunk → teach)
- YouTube transcripts (yt-dlp → captions → chunk → teach)
- "Build curriculum from <topic>" — chain web → scan → coach

### F. Gesture / camera input (iter 6)
MediaPipe Hands in JS frontend → swipe/pinch/grab → trigger Adam skills.
- Webcam capture via getUserMedia
- Hand landmark stream → simple gesture classifier
- Map gestures to skill invocations (swipe = next, pinch = zoom card, grab = persistent panel)

### G. Polish (iter 7+)
- More widgets based on usage
- Multi-modal: paste image → describe → act
- Long-running task UI (progress bars, cancellable)
- Memory inspector ("show me what you know about me")
- Privacy dashboard for PersonalAtlas

## Per-iteration cadence

Each iteration:
1. Pick highest-ROI gap from above
2. Build + test + commit + push
3. Update this roadmap with checkmarks
4. Schedule wakeup ~25min later for next iter

## Done so far

- iter 1 (2026-05-21): widget protocol + weather + system_stats + time_card → v6.9.6
- iter 2 (2026-05-21): Jarvis UI shell at /jarvis route — neural-net canvas bg, neon-glow CSS, inline widget cards (weather/system/time), voice in+out, calls /v1/chat/completions → v6.9.7
- iter 3 (2026-05-21): Socratic coach mode — CoachAtlas (per-topic mastery + rolling weighted avg) + coach skill (start/ask/answer/hint/skip/summary/status) with mocked-Adam tested grading loop, difficulty escalation on 2-streak → v6.9.8
- iter 4 (2026-05-21): Adam-driven scheduling — ScheduleAtlas (persistent jobs.jsonl + outcomes capped to 5/job) + AdamScheduler thread (poll every 5s, fire due skill/prompt/webpoll jobs) + schedule_loop skill (add/list/get/cancel/enable/disable/runs/run_now/stats). 29 skills total. → v6.9.9
- iter 5 (2026-05-22): Content ingestion automation — ingest_url (trafilatura + HTML-strip fallback), ingest_pdf (pypdf conditional), ingest_youtube (yt-transcript-api conditional), build_curriculum (web search → ingest top-N → start coach session). 33 skills total. → v6.9.10
- iter 6 (2026-05-25): MediaPipe Hands gesture input in /jarvis — webcam capture, 21-point hand tracking via MediaPipe Hands (CDN), distance-based 6-gesture classifier (pinch/fist/open_palm/peace/point/thumb_up), 900ms cooldown, on-screen flash, mapped actions (toggle voice / clear chat / system check / cycle theme / next question / submit). Live landmark overlay in corner cam panel with FPS counter. Closes the Jarvis-CV reference from the original loop input. → v6.9.11
- iter 7 (2026-05-25): **24/7 self-improvement substrate** — Adam now learns continuously in the background, on its own. Six integrated modules: LearningAtlas (per-cell provenance/confidence/consensus/last_reinforced metadata), qa_extractor (5× density via atomic Q-A extraction), curiosity (PTEX sparse-region + low-mastery gap finder), consensus (multi-source verification → verified/debated flagging), sleep_consolidator (cluster + synthesize summary cells), LearningDaemon (24/7 thread orchestrating curiosity ticks + parallel ingest workers + sleep passes + spaced repetition; auto-yields when user is chatting). Toggle via $AMNI_NO_LEARNING_DAEMON. 34 skills total. → **v6.10.0**
- iter 8 (2026-05-25): **Knowledge graph + privacy backports** — KnowledgeGraph SPO triple store (subject/predicate/object with consensus, BFS path-between, fuzzy subject search, persistence). kg_extractor turns Q-A pairs into triples via Adam JSON-mode. kg_query skill: neighbors/out/in/predicate/path/search/add/forget/stats. Plus two deferred backports finally landed: (a) pre-save force-refit applied to ConversationAtlas + CodeAtlas (same v6.9.5 PersonalAtlas fix), (b) ConversationAtlas.recall include_local default flipped to False — closes the v6.5.0 paranoid PII test that was failing. 35 skills. → **v6.10.1**
- iter 9 (2026-05-25): **Memory Inspector** — surface what Adam knows across all 9 atlases inline in /jarvis. New HTTP layer amni/serve/memory_endpoints.py (GET /memory/snapshot|profile|kg|coach|daemon, POST /memory/forget + /memory/confirm). New PersonalAtlas.list_facts API. Slide-in side panel in /jarvis with MEMORY button: substrate overview, profile facts with confidential tags + forget buttons, pending clarifications with confirm buttons, KG top subjects (clickable to explore), coach mastery bars (clickable to resume), daemon counters. Trust through transparency. → **v6.10.2**
- iter 10 (2026-05-25): **Long-running task UI** — TaskRegistry (in-memory, thread-safe, register/update/complete/fail/request_cancel + ring-buffer of recent). HTTP layer /tasks GET + /tasks/<id> + /tasks/<id>/cancel POST. build_curriculum + ingest pipeline emit progress + honor cancel flag (graceful break between sources). Floating /jarvis task tray (slides up from bottom when active count > 0, auto-dismisses after 3.5s idle, per-task progress bar + × cancel). User regains visibility AND control over Adam's autonomy. → **v6.10.3**
- iter 11 (2026-05-25): **Multi-modal vision** — Adam handles image input. VisionService lazy-loads BLIP-base (~470MB) for captions + BLIP-VQA-base for image Q&A. describe(image_bytes) → caption, caption_with_question() → answer. Optional dep — clean error dict if transformers/Pillow missing, never crashes server. HTTP /vision/describe, /vision/ask, /vision/status (+ /vision/upload if python-multipart installed). describe_image skill. /jarvis composer accepts paste (Ctrl+V) + drag/drop image events: thumbnail inline in user bubble, caption appended, follow-up Qs auto-route to /vision/ask via keyword heuristic ("this/that/the image/color/shape"). 36 skills. → **v6.10.4**
- iter 12 (2026-05-25): **5 new inline widgets** — news (DDG news scrape, scrollable card with sources), stock (Yahoo Finance quotes with price/change/H/L/market state, color-coded ▲/▼), file_preview (head N lines mono pre with meta), disk_widget (per-partition bars), git_status (branch/dirty/ahead/behind/recent commits/unstaged files). Each gets a tailored neon-themed Jarvis renderer. 41 skills. → **v6.10.5**
- iter 13 (2026-05-25): **Local voice upgrade** — /jarvis now prefers local Piper TTS + faster-whisper STT over browser Web Speech. New voice_endpoints layer (GET /voice/status, POST /voice/speak, POST /voice/transcribe) wraps existing tts/stt skills with TaskRegistry integration. UI probes /voice/status on load; speak() falls through to /voice/speak first (audio_base64 → HTMLAudioElement); mic now uses MediaRecorder → /voice/transcribe pipeline; browser Web Speech kept as automatic fallback. Strict base64 validation, data-URL prefix tolerance, task-tray visibility on long Whisper runs. → **v6.10.6**
- iter 14 (2026-05-25): **File/folder watcher** — stdlib-only polling watcher (no watchdog dep). FileWatcher class: register watches (path, glob, recursive, coalesce_s, optional on_change_skill+args), single daemon thread polls every 5s, diffs mtime/size against last snapshot, emits created/modified/deleted events with per-watch ring buffer. watch skill: add/list/get/cancel/enable/disable/events/tick/stats. New 'watch' widget type with neon-themed Jarvis renderer (color-coded kind tags green/cyan/red). Pairs with the scheduler for "when X changes run Y" workflows. Disable via $AMNI_NO_FILE_WATCHER. 42 skills. → **v6.10.7**
- iter 15 (2026-05-25): **Amni-Code widget rendering** — cross-repo polish, Amnibro/Amni-Code → v2.5.0. Rust agent_loop_stream now extracts raw_msg.amni_widgets[] and emits a 'widget' SSE event per item. Frontend renderWidget() draws 12 widget types as themed cards inline in the assistant message, adopting the active Amni-Code theme (pink/crypt/haven/ai/etc) via CSS custom properties. Ask Adam "what's the weather" inside Amni-Code → you get a themed temperature card. Two surfaces (/jarvis + Amni-Code) now both speak the widget protocol. cargo check passes. (No Amni-Ai version bump — backend already emitted widgets correctly since v6.10.0; this was purely the receiving side catching up.)
- iter 16 (2026-05-26): **Continuous voice conversation + latency reductions** — CONVO button + VAD-driven mic + auto-rearm. State machine (idle/listening/recording/transcribing/thinking/speaking/error) with pulsing color-coded indicator. Banner shows live mic level meter. Five latency reductions in one iter: silence threshold 1200→600ms, min speech 300→150ms, **barge-in** (kill TTS when user starts talking mid-reply), **streaming chat + per-sentence TTS** (calls /chat/stream, segments by sentence boundary, parallel TTS+continued generation — Adam starts speaking before generation completes), Whisper pre-warm on convo start. Safety: max 3 consecutive transcribe fails → auto-disable, max 10s per utterance. 18/18 tests. → **v6.10.8**

## Open future ideas

- Multi-user separation (resolve the `recall_does_not_cross_session_for_personal` regression noted in v6.9.5)
- Backport pre-save force-refit to ConversationAtlas + CodeAtlas (same bug class as v6.9.5 fix)
- ConversationAtlas `__local_user__` privacy isolation rework
