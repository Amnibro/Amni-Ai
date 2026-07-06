$ErrorActionPreference = 'SilentlyContinue'
$dir = 'C:\Users\antho\Documents\ai\Amni-Ai'
if (Test-Path "$dir\logs\.killswitch") { exit 0 }
if ((Get-NetTCPConnection -LocalPort 7700 -State Listen -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0) { exit 0 }
$lock = "$dir\logs\.launching"
if ((Test-Path $lock) -and (((Get-Date) - (Get-Item $lock).LastWriteTime).TotalMinutes -lt 6)) { exit 0 }
Set-Content -Path $lock -Value ((Get-Date).Ticks) -Force
$env:AMNI_NO_NVFP4 = '1'
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
$env:AMNI_NO_LEARNING_DAEMON = '1'
$env:AMNI_NO_CHORD = '1'
$env:AMNI_MCQ_SAMPLES = '1'
$env:AMNI_CHAT_BRIEF_TOKENS = '180'
$env:AMNI_VISION_DEVICE = 'cpu'
$env:AMNI_CHAT_IGNORE_EDS = '9da61606a9eb8881cf25b52edcbbf2b07d68034b8540bbfbf086881a817dbb80'
Start-Process -FilePath "$dir\.venv\Scripts\python.exe" -ArgumentList 'scripts/amni_serve.py','--host','0.0.0.0','--port','7700','--cors','--seed' -WorkingDirectory $dir -WindowStyle Hidden -RedirectStandardOutput "$dir\logs\amni_serve_out.log" -RedirectStandardError "$dir\logs\amni_serve_err.log"
