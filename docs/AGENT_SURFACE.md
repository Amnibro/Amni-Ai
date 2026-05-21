# Adam Agent Surface — endpoints, skills, clients

Quick reference for what's exposed at the HTTP layer above the GF(17) inference core. Refresh this when adding/removing skills or endpoints.

## HTTP endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Lightweight liveness + version + warmup state |
| GET | `/health` | Full diagnostics: adam state, skills, voice backends, GPU, personas |
| GET | `/warmup` | `{done, wall_s, error}` — async warmup status |
| GET | `/skills` | List all registered skills with descriptions + schemas |
| POST | `/skills/{name}` | Invoke a skill directly: `{args:{...}}` envelope |
| POST | `/chat/stream` | SSE streaming chat: meta → token → done. Auto-routes to agentic for build requests |
| POST | `/complete` | Inline code completion (IDE-style): `{prefix, suffix?, language?, max_tokens?, stop?}` |
| POST | `/voice/chat` | Audio↔chat↔audio one-call: `{audio_base64 OR text, return_audio?, session_id?}` |
| POST | `/profile/inference` | Throughput benchmark: `{prompt, max_new_tokens, runs}` → per-run + avg tok/s |
| GET | `/sessions` | List all session metadata |
| GET | `/sessions/{sid}` | Get full turns for a session (paginated by `?limit=N`) |
| DELETE | `/sessions/{sid}` | Delete a session |
| GET | `/personas` | List known personas |
| POST | `/persona` | Switch active persona: `{name}` |
| POST | `/ask` | Non-streaming agent.ask (legacy) |
| POST | `/teach` | Add a Q→A lesson to the bank: `{question, answer}` |
| GET | `/lessons` | Browse the lesson bank: `?q=&offset=&limit=` |
| GET | `/stats` | Per-tier counters + iteration metrics |
| GET | `/mcp` | MCP server endpoint (work-in-progress) |
| GET | `/api/tags` | Ollama-compatibility shim |

## Skills (24)

Skills register at boot via `default_registry()` and are invocable via `/skills/{name}` or by the agentic planner.

**Core inference + memory**
- `calc` — math expressions via Adam tier-3.7
- `mem` — query lesson bank
- `web` — DDG search + crawler distill
- `time` — current local time

**Code understanding**
- `symbols` — AST/regex symbol extraction (Python/Rust/JS/TS)
- `project_info` — workspace summary (languages, deps, git branch, dirty files)
- `parse_error` — Python/Rust/JS/TS/Go stack-trace extractor

**Code modification**
- `file_read` — chunked text file read (offset/limit, line_offset/line_limit)
- `file_write` — overwrite text file
- `code_edit` — find-and-replace edit (AST-validated for `.py`)
- `code_diff` — apply unified diff (with hunk validation + dry_run)
- `rename_symbol` — multi-file rename across workdir
- `auto_import` — detect undefined names + suggest stdlib imports
- `format_code` — run canonical formatter (ruff/black/rustfmt/prettier/gofmt)

**Execution / project**
- `run_python` — sandboxed Python execution (blocks fs-mutation/network/subprocess)
- `shell` — allowlisted read-only shell commands
- `git` — read-only git ops (status/log/diff/branch/blame/etc, mutation refused)
- `test_run` — auto-detect + run cargo/pytest/npm/go/make tests
- `scan` — walk path + ingest into lesson bank

**Voice**
- `tts` — text-to-speech (piper > pyttsx3 fallback)
- `stt` — speech-to-text (faster-whisper > vosk fallback, auto-detects audio format)

**Session management**
- `export_session` — dump session as markdown/text/json
- `prune_sessions` — delete old session jsonl (keeps N most-recent)
- `goal` — agentic multi-step orchestrator

## Client surfaces

### Terminal CLI (`scripts/adam_cli.py`)
```sh
python scripts/adam_cli.py "your prompt"
python scripts/adam_cli.py --session work1 "follow-up"
python scripts/adam_cli.py --persona Yoda "explain recursion"
python scripts/adam_cli.py --voice-record 5 --continuous   # mic → reply → speaker
python scripts/adam_cli.py --skill calc "17*23"
python scripts/adam_cli.py --json --no-color "..." | jq    # machine-readable
```

### HUD (`docs/hud/index.html`)
Open in browser. Connects to `http://127.0.0.1:11434`. Features:
- Markdown chat with code-block rendering + inline code styling
- Sessions sidebar (click-to-load + new + refresh)
- Persona switcher (top of chat)
- Mic recording via MediaRecorder → `/voice/chat`
- Live `/health` polling every 10s (status bar)
- Skills sidebar (click-to-insert into prompt)

### VSCode extension (`extensions/vscode/`)
- `InlineCompletionItemProvider` calling `/complete` as you type
- `adam.ask` (Ctrl+Alt+A) — open chat in output panel
- `adam.complete` (Ctrl+Alt+Space) — explicit completion request
- `adam.build` — feed selection to agentic loop
- `adam.health` — status bar item, click for diagnostics
- Settings: host, port, max-tokens, persona

## Test + benchmark scripts

| Script | Purpose |
|---|---|
| `scripts/smoke_test.py` | 17-check end-to-end test (run after every restart or before commit) |
| `scripts/eval_humaneval.py` | 10-problem mini-HumanEval benchmark (currently 91.9% pass) |
| `scripts/amni_serve.py` | The HTTP server itself |

## See also

- `docs/BENCHMARKS.md` — published numbers
- `docs/ROADMAP.md` — what's next
- `changelog.md` — full version history
- `architecture_map.md` — GF(17)/TMU/PTEX architecture (paradigm layer, not agent surface)
