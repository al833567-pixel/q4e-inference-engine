@echo off
chcp 65001 >nul
title q4e v7 - GLM-5.3-Flash (dGPU + CPU, three-tier expert cache)
rem usage: run-glm-5.3-flash.bat [model folder]   (default: ..\models\GLM-5.3-Flash next to this repo)
rem
rem Dense part on the GPU; the experts (80 GiB) in three tiers: VRAM (hottest ~10%%), RAM (locked), SSD (read on
rem demand by the CPU). One expert = one block (--q4e-blocks 1): GLM experts are ~7 MiB, splitting them was slower.
rem Q4E_SWAP_MB=32: smaller VRAM promotions per step, so they compete less with the CPU for the SSD.
rem -t 12: the CPU tier waits on SSD pages, 12 threads keep more reads in flight (-t 16 collapses).
rem A missed expert with under 12%% of the token's router weight is skipped (Q4E_SKIP=0.12 default).
rem Measured (Tesla T10 16 GB + Ryzen 7 H 255 + 48 GB, warm profile): ~11.5 / ~10.9 t/s (essay / code).

set "ENGINE=%~dp0..\engine-v7\build-clang\bin"
set "MODEL_DIR=%~1"
if "%MODEL_DIR%"=="" set "MODEL_DIR=%~dp0..\..\models\GLM-5.3-Flash"
set Q4E_SWAP_MB=32

"%ENGINE%\llama-server.exe" ^
    -m "%MODEL_DIR%\GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-00001-of-00003.gguf" ^
    -c 8192 ^
    -fa on ^
    -np 1 ^
    --device CUDA0 ^
    -ngl 99 ^
    --q4e ^
    --q4e-blocks 1 ^
    -t 12 ^
    --host 127.0.0.1 ^
    --port 8080

pause
