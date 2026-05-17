@echo off
REM One-line Adam installer for Windows. Detects Python, runs install.py.
where python >nul 2>nul
if errorlevel 1 (
    echo [error] Python not found. Install Python 3.10+ from https://python.org and re-run.
    exit /b 1
)
python install.py %*
