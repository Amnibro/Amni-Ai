$DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $DIR
$ROCM = if ($env:ROCM_HOME) { $env:ROCM_HOME } elseif (Test-Path "C:\Program Files\AMD\ROCm") { (Get-ChildItem "C:\Program Files\AMD\ROCm" | Sort-Object Name -Descending | Select-Object -First 1).FullName } else { "C:\Program Files\AMD\ROCm\6.2" }
$SHORT_ROCM = (& cmd /c "for %I in (`"$ROCM`") do @echo %~sI").Trim()
Write-Host "[gf17_hip] ROCM_HOME=$ROCM"
Write-Host "[gf17_hip] short path=$SHORT_ROCM"
$HIPCC = "$ROCM\bin\hipcc.bat"
if (-not (Test-Path $HIPCC)) { $HIPCC = "hipcc" }
Write-Host "[gf17_hip] compiling for gfx1100+gfx1101 (Windows DLL)..."
& $HIPCC --offload-arch=gfx1100 --offload-arch=gfx1101 -O3 -shared -o "$DIR\libgf17_hip.dll" "$DIR\gf17_hip.cpp" "-I$SHORT_ROCM\include" "-L$SHORT_ROCM\lib" -lamdhip64 "-Wl,/DEF:$DIR\gf17_hip.def"
if ($LASTEXITCODE -eq 0) { Write-Host "[gf17_hip] built: $DIR\libgf17_hip.dll ($((Get-Item "$DIR\libgf17_hip.dll").Length) bytes)" } else { Write-Host "[gf17_hip] build FAILED" -ForegroundColor Red }
Write-Host "[ari_hip] compiling for gfx1100+gfx1101 (Windows DLL)..."
& $HIPCC --offload-arch=gfx1100 --offload-arch=gfx1101 -O3 -shared -o "$DIR\libari_hip.dll" "$DIR\ari_hip.cpp" "-I$SHORT_ROCM\include" "-L$SHORT_ROCM\lib" -lamdhip64 "-Wl,/DEF:$DIR\ari_hip.def"
if ($LASTEXITCODE -eq 0) { Write-Host "[ari_hip] built: $DIR\libari_hip.dll ($((Get-Item "$DIR\libari_hip.dll").Length) bytes)" } else { Write-Host "[ari_hip] build FAILED" -ForegroundColor Red }
