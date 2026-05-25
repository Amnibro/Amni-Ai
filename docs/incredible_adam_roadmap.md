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

## Open future ideas

- Multi-user separation (resolve the `recall_does_not_cross_session_for_personal` regression noted in v6.9.5)
- Backport pre-save force-refit to ConversationAtlas + CodeAtlas (same bug class as v6.9.5 fix)
- ConversationAtlas `__local_user__` privacy isolation rework
