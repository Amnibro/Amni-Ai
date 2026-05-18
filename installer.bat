@echo off
where python >nul 2>nul
if errorlevel 1 (
    echo [error] Python not found. Install Python 3.10+ from https://python.org and re-run.
    pause
    exit /b 1
)
python "%~dp0installer.py" %*
