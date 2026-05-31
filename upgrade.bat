@echo off
REM One-click Adam upgrade for Windows: git pull + re-run installer (deps/kernels/model). KaTeX rides with the pull.
cd /d "%~dp0"
where python >nul 2>nul || (echo [error] python not found on PATH. Install Python 3.10+ and re-run. & exit /b 1)
python upgrade.py %*
