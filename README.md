# q4e inference engine (v7)

Large MoE models on one mid-size GPU + CPU: `engine-v7/` is llama.cpp (b11796) with **q4e**, a three-tier cache
of the routed experts.

- **VRAM**: a pool of the most used expert *blocks* (an expert cut into slices of its FFN rows), computed by the GPU
- **RAM**: the other blocks, locked in memory, computed by the CPU in the same step
- **SSD**: what does not fit RAM, read by the CPU through the file mapping

Blocks move between the tiers by use while generating (copies on a second stream), the scores are saved to
`<model>.q4e-profile` so the next start fills VRAM with the usual experts. The VRAM pool is sized from the free
memory of the GPU, so the same build adapts to other cards.

Tested models: **Qwen3.8-Flash-Next** IQ2_XS (arch `qwen4exp`, with its MTP draft model) and **GLM-5.3-Flash**
(arch `glm5-next`). Results and what was tried: [REPORT.md](REPORT.md).

## Results (Tesla T10 16 GB + Ryzen 7 H 255 + 48 GB RAM)

| model | setup | essay t/s | code t/s |
|---|---|---|---|
| Qwen3.8-Flash-Next IQ2_XS | q4e + MTP n=3, 32k ctx | **37** | **50** |
| Qwen3.8-Flash-Next IQ2_XS | q4e + MTP n=3, 128k ctx | 33.5 | 46 |
| Qwen3.8-Flash-Next IQ2_XS | llama.cpp b11796 (experts on CPU) | 15.5 | 16.4 |
| GLM-5.3-Flash (80 GiB experts) | q4e, 1 block per expert, -t 12, 8k ctx, warm profile | 11.5 | 10.9 |

## Build (Windows, CUDA)

Needs the VS 2022 Build Tools (cmake, ninja, MSVC), LLVM (clang-cl) and the CUDA toolkit.

```bat
set CUDA_PATH=<CUDA toolkit folder>
set CUDA_ARCH=75
scripts\build-windows-cuda.bat
```

Output: `engine-v7\build-clang\bin\llama-server.exe` (static, plus the CUDA runtime DLLs). Other platforms: the
usual llama.cpp CMake build of `engine-v7/`.

## Run

`run-flash-next.bat` starts with a 128k context (`-c 131072`), `run-glm-5.3-flash.bat` with 8k; `scripts/bench.ps1` measures at 32k unless `-Extra @('-c', ...)` is given.

```bat
scripts\run-flash-next.bat <folder with the Flash-Next GGUF + MTP GGUF>
scripts\run-glm-5.3-flash.bat <folder with the GLM-5.3-Flash GGUF>
```

The server is OpenAI compatible on `http://127.0.0.1:8080`. q4e is enabled by `--q4e` (implies the experts in host
memory), options:

| option / env | meaning |
|---|---|
| `--q4e-blocks N` | blocks per expert (0 = auto, blocks of up to 1 MiB; GLM: 1) |
| `--q4e-vram-mib N` / `--q4e-reserve-mib N` | VRAM pool size / VRAM left free (default: free VRAM - 640 MiB) |
| `--q4e-ram-mib N` | RAM tier (default: free RAM - 6 GiB) |
| `--q4e-profile FILE` | block scores (default `<model>.q4e-profile`, `none` = off) |
| `Q4E_SKIP=0.12` | a missed expert under this share of the token's router weight is skipped (0 = exact) |
| `Q4E_SWAP_MB=32` | VRAM promotions per step once the pool is full |
| `Q4E_CPU_FUSED=0` | separate ops instead of the fused CPU expert op |
| `Q4E_CUDA_FUSE=0` | no CUDA fusions of the qwen4exp small ops |
| `GGML_SCHED_TIMING=2` | per-backend timing in the `q4e: last 64` log lines |

The first start writes `<model>.q4e-down.bin` (the down projections repacked per block, once).

## Scripts

- `scripts/bench.ps1`: start a server, essay + code + ~2k prompt, print t/s
- `scripts/ab.ps1`: A/B runs of Flash-Next from the same expert profile

## License

`engine-v7/` is based on llama.cpp (MIT, see `engine-v7/LICENSE`).
