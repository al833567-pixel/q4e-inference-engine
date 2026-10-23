@echo off
chcp 65001 >nul
title q4e v7 - Qwen3.8-Flash-Next IQ2_XS (dGPU + CPU, three-tier expert cache, MTP)
rem usage: run-flash-next.bat [model folder]   (default: ..\models\Qwen3.8-Flash-Next next to this repo)
rem
rem Everything except the routed experts runs on the GPU; the experts are cut into blocks (2 per expert) that live
rem in three tiers and move by use:
rem   VRAM : the hottest blocks (sized from the free VRAM at the first request)
rem   RAM  : the rest, locked in RAM - the CPU computes these while hot ones are copied to VRAM
rem   SSD  : whatever does not fit RAM
rem The scores are saved to <model>.q4e-profile, so the next start fills VRAM with your usual experts. The first
rem start writes <model>.q4e-down.bin (once). A missed expert with under 12%% of the token's router weight is
rem skipped (Q4E_SKIP=0.12 default, set Q4E_SKIP=0 for exact outputs).
rem Measured (Tesla T10 16 GB + Ryzen 7 H 255 + 48 GB): 32k 37 / 50 t/s (essay / code), 128k 33.5 / 46.

set "ENGINE=%~dp0..\engine-v7\build-clang\bin"
set "MODEL_DIR=%~1"
if "%MODEL_DIR%"=="" set "MODEL_DIR=%~dp0..\..\models\Qwen3.8-Flash-Next"

"%ENGINE%\llama-server.exe" ^
    -m "%MODEL_DIR%\Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf" ^
    -c 131072 ^
    -fa on ^
    -np 1 ^
    --device CUDA0 ^
    -ngl 99 ^
    --q4e ^
    --spec-type draft-mtp ^
    -md "%MODEL_DIR%\mtp-Qwen3.8-Flash-Next-Q4_0.gguf" ^
    --spec-draft-n-max 3 ^
    -ngld 99 ^
    -t 8 ^
    --host 127.0.0.1 ^
    --port 8080

pause
