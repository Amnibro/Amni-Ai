@echo off
REM Launch Adam in a standalone desktop window (no terminal command, no launcher needed).
REM Double-click this file. It activates the local venv if present, then serves Adam and
REM opens a chromeless Chrome/Edge app window once the GF(17) weights finish warming.
setlocal
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
echo Starting Adam -- a standalone window will open when the model is ready (this takes a minute on first warmup)...
"%PY%" -m amni.cli serve --app-window %*
endlocal
