# Adam (Amni-Ai) — VSCode Extension

Local AI coding assistant. Connects to your running Adam server (default `127.0.0.1:11434`).

## Features

- **Inline completions** (Copilot-style ghost text as you type) via `/complete`
- **Adam: Ask** (Ctrl+Alt+A) — open-question chat in output panel
- **Adam: Inline Complete at Cursor** (Ctrl+Alt+Space) — explicit completion request
- **Adam: Agentic Build** — feed selected text (or prompt) to Adam's agentic loop
- **Adam: Show Server Health** — status bar item, click for skill count + voice backends + GPU state

## Install (dev mode)

```sh
cd Amni-Ai/extensions/vscode
code --install-extension .   # or open folder in VSCode + F5 to launch Extension Development Host
```

## Configure

Settings → search "Adam":
- `adam.host` (default `127.0.0.1`)
- `adam.port` (default `11434`)
- `adam.inlineCompletion` (default `true`) — disable if too aggressive
- `adam.completionMaxTokens` (default `30`)
- `adam.persona` (default `Rikku`) — applied to chat commands

## Requires

A running Adam server: `cd Amni-Ai && python scripts/amni_serve.py --seed --cors --port 11434`

## Status

This is a scaffold (v0.1.0). Full features (vision input, MCP integration, sidebar chat panel) are roadmap items in `docs/ROADMAP.md`. Inline completions and explicit commands work today.
