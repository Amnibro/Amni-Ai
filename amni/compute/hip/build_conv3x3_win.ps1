$DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $DIR
$VENV='C:\Users\antho\AppData\Local\Programs\Python\Python312\Lib\site-packages'
$CORE="$VENV\_rocm_sdk_core"
$DEVEL="$VENV\_rocm_sdk_devel"
$HIPCC='hipcc'
Write-Host "[conv3x3] compiling for gfx1101..."
& $HIPCC --offload-arch=gfx1101 -O3 -shared -o "$DIR\libconv3x3.dll" "$DIR\conv3x3.cpp" "-I$CORE\include" "-L$CORE\lib" -lamdhip64 "-Wl,/DEF:$DIR\conv3x3.def"
if ($LASTEXITCODE -eq 0) { Write-Host "[conv3x3] built: $DIR\libconv3x3.dll ($((Get-Item "$DIR\libconv3x3.dll").Length) bytes)" } else { Write-Host "[conv3x3] build FAILED" -ForegroundColor Red; exit 1 }
