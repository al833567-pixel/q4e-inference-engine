@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
  echo VCVARS_FAILED
  exit /b 1
)
cd /d "%~dp0"
echo ===CL_VERSION===
cl /? 2>&1 | findstr /n /r "^" | findstr /r "^1[^0-9]"
echo ===MSVC_BUILD===
cl /nologo /EHsc /O2 regex_probe.cpp /Fe:regex_probe_msvc.exe
if errorlevel 1 (
  echo MSVC_BUILD_FAILED
  exit /b 1
)
echo ===MSVC_RUN===
regex_probe_msvc.exe
echo ===MSVC_RUN_EXITCODE:%ERRORLEVEL%===
