@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "CORE=C:\Users\antho\Documents\ai\Amni-Ai\.venv\Lib\site-packages\_rocm_sdk_core"
set "OUT=%~dp0"
"%CORE%\lib\llvm\bin\clang.exe" -x hip "%OUT%nvfp4_kernel.cu" -shared -o "%OUT%nvfp4hip.dll" --offload-arch=gfx1101 -O3 -std=c++17 -D__HIP_PLATFORM_AMD__=1 -fms-runtime-lib=dll --rocm-path="%CORE%\lib\llvm" --rocm-device-lib-path="%CORE%\lib\llvm\amdgcn\bitcode" -I"%CORE%\include" -fuse-ld=lld --ld-path="%CORE%\lib\llvm\bin\lld-link.exe" -L"%CORE%\lib" -lamdhip64
echo COMPILE_EXIT=%ERRORLEVEL%
