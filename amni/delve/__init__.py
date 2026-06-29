"""Amni-Delve: the multi-AI roundtable, fused into Adam. adapters = declarative platform registry (claude/grok/gemini/codex/aider/cursor/ollama/adam), hub = the shared-transcript engine, ptex = learning capture. Keys live only in Anthony's env; nothing here persists or returns a secret."""
from amni.delve import adapters,hub,ptex
Hub=hub.Hub
__all__=["adapters","hub","ptex","Hub"]
