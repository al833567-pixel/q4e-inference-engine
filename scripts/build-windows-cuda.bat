@echo off
rem q4e v7 build (Windows, CUDA): C/C++ with clang-cl (the CPU kernels are much faster than with MSVC), CUDA with
rem nvcc and MSVC as its host compiler, cmake + ninja from the VS 2022 Build Tools. Output: engine-v7\build-clang\bin.
rem Static exe: Smart App Control blocks unsigned DLLs, a single unsigned exe runs. OpenMP off (ggml's threadpool).
rem
rem Tools (override before calling):
rem   CUDA_PATH      CUDA toolkit (set by the CUDA installer), e.g. ...\CUDA\v12.8
rem   VS_VCVARS      vcvars64.bat of VS 2022 (default: found with vswhere)
rem   LLVM_BIN       folder of clang-cl.exe (default: %ProgramFiles%\LLVM\bin)
rem   CUDA_ARCH      CUDA architectures (default: 75 = Turing, e.g. Tesla T10 / T4)
setlocal
if "%CUDA_ARCH%"=="" set "CUDA_ARCH=75"
if "%LLVM_BIN%"=="" set "LLVM_BIN=%ProgramFiles%\LLVM\bin"
if "%CUDA_PATH%"=="" (echo CUDA_PATH is not set & exit /b 1)
if "%VS_VCVARS%"=="" (
    for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set "VS_VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
)
rem PATH is extended BEFORE vcvars64: after it PATH holds quoted entries that break a later set "PATH=..."
set "PATH=%CUDA_PATH%\bin;%LLVM_BIN%;%PATH%"
call "%VS_VCVARS%" || exit /b 1
where cl >nul 2>nul || exit /b 1
where clang-cl >nul 2>nul || exit /b 1
cd /d "%~dp0..\engine-v7"
for /f "delims=" %%i in ('where cl') do set "MSVC_CL=%%i"
set "MSVC_CL=%MSVC_CL:\=/%"
set "MSVC_BIN=%MSVC_CL:/cl.exe=%"
rem Smart App Control blocks llvm-lib/lld-link/llvm-rc: archive, link and resources with the signed MSVC/SDK tools
for /f "delims=" %%i in ('where rc') do set "SDK_RC=%%i"
for /f "delims=" %%i in ('where mt') do set "SDK_MT=%%i"
set "SDK_RC=%SDK_RC:\=/%"
set "SDK_MT=%SDK_MT:\=/%"
cmake -S . -B build-clang -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_CUDA_HOST_COMPILER="%MSVC_CL%" -DCMAKE_AR="%MSVC_BIN%/lib.exe" -DCMAKE_LINKER="%MSVC_BIN%/link.exe" -DCMAKE_RC_COMPILER="%SDK_RC%" -DCMAKE_MT="%SDK_MT%" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH% -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF || exit /b 1
cmake --build build-clang -j 12 --target llama-server llama-cli llama-bench q4e-kbench || exit /b 1
for %%d in (cudart64_12 cublas64_12 cublasLt64_12) do copy /y "%CUDA_PATH%\bin\%%d.dll" build-clang\bin\ >nul
echo BUILD-OK
