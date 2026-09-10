# s3_diag analysis — q4e-moe-vram K=10 vs K=0 (Qwen3.8-Flash-Next UD Q4_K_XL)

## 0. Run metadata

| Item | Value |
|---|---|
| Binary | `D:\Nova2_home\AI研究\推理工具\engine\build-cuda\bin\q4e-server.exe` |
| Version (exact, `--version`) | `version: 0.3.0-dev (build 39, commit 0ccc16c)` / `built with MSVC 19.44.35228.0 for Windows AMD64` — **build 39 as expected, gate passed** |
| Model (entry shard) | `D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` (10.4 MiB; loader opens the other 3 shards) |
| GPU (before runs) | RTX 3090, 24576 MiB total, **24036 MiB already used** by a pre-existing `D:\llama\llama-server.exe` (PID 16280) — see `gpu_state.txt`. Not killed (read-only diagnostic). |
| Log sizes | both runs: 385.2 KB, 5390 lines each |

Notes:
- Both runs **exited on their own after ~1.4 s** with the expected fit-check failure; no kill was needed, no server was left running, no source files touched.
- Layer count in this model is **49** (log: "offloading 47 repeating layers to GPU", "offloaded 49/49 layers to GPU"), not 38.
- The build 39 log does **not** contain any line matching `buffer type overridden` (verified by exact phrase and by `overrid*`, `buffer type`, `buft`, `to Host`, `Host buft` patterns). The per-tensor lines at `-lv 10` are only `create_tensor: loading tensor blk.N.<name>`; the closest placement-related lines are the `load_tensors: offloading ... to GPU` lines quoted below. All "overridden" counts in section 3 are therefore 0.

---

## RUN 1 — K=10 (`--q4e-moe-vram 10`)

### 1. Exit, fit-check, device lines (exact verbatim)

- **Exit code: 1** (non-zero, as expected; self-exit after fit-check failure)
- **Fit-check error line (exact):**

  ```
  0.01.406.656 E cmn  common_q4e_m: common_q4e_moe_vram_check: --q4e-moe-vram k=10 does not fit in VRAM: CUDA0 needs 78431 MiB, only 23141 MiB free
  ```

  → needs **78431 MiB**, only **23141 MiB free**.
- **"using device ... free" lines (exact, both occurrences):**

  ```
  0.00.510.314 I llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 3090) (0000:01:00.0) - 23336 MiB free
  0.01.056.004 D llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 3090) (0000:01:00.0) - 23336 MiB free
  ```

- Termination lines (verbatim, end of log):

  ```
  0.01.406.662 E cmn  common_init_: failed to load model 'D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
  0.01.406.663 E srv    load_model: failed to load model, 'D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
  0.01.406.669 I srv   operator (): operator (): cleaning up before exit...
  0.01.407.631 E srv  llama_server: exiting due to model loading error
  ```

### 2. Complete memory breakdown table (verbatim, L5383–L5385 of run1_k10_lv10.log)

```
0.01.361.190 I common_memory_breakdown_print: | memory breakdown [MiB] | total    free     self   model   context   compute    unaccounted |
0.01.361.194 I common_memory_breakdown_print: |   - CUDA0 (RTX 3090)   | 24575 = 23141 + (78431 = 78056 +     178 +     196) +      -76997 |
0.01.361.194 I common_memory_breakdown_print: |   - Host               |                  28128 = 28110 +       0 +      18                |
```

Read per the header (`total = free + (self = model + context + compute) + unaccounted`):
- **CUDA0:** total 24575, free 23141, self 78431 = **model 78056** + context 178 + compute 196, unaccounted −76997
- **Host:** total 28128, free 28110, self 0 (i.e. **model 0**, context 0, compute 0), unaccounted 18

Related placement lines (verbatim):

```
0.00.943.470 I load_tensors: offloading output layer to GPU
0.00.943.470 I load_tensors: offloading 47 repeating layers to GPU
0.00.943.471 I load_tensors: offloaded 49/49 layers to GPU
```

### 3. Buffer-type override lines

- **Count of lines containing "buffer type overridden": 0**
- Distinct destination buffer type names used: **none** (no such lines exist in this build's log)
- First 5 such lines: **N/A**
- Last 5 such lines: **N/A**
- Overridden lines matching `ffn_down_exps`: **0** — matching `ffn_gate_up_exps`: **0** (hence no layer grouping)
  - Context: the log does contain expert-tensor load lines (no override attached): `ffn_down_exps` in 288 lines, `ffn_gate_up_exps` in 96 lines, all of the form `D create_tensor: loading tensor blk.N.ffn_down_exps.weight` (N = 0..47).

### 4. Buft model MiB (from the breakdown table)

| buft | model MiB |
|---|---|
| Host (CPU) | **0** |
| CUDA0 | **78056** |

---

## RUN 2 — K=0 differential control (`--q4e-moe-vram 0`)

### 1. Exit, fit-check, device lines (exact verbatim)

- **Exit code: 1** (non-zero, as expected; self-exit after fit-check failure)
- **Fit-check error line (exact):**

  ```
  0.01.408.200 E cmn  common_q4e_m: common_q4e_moe_vram_check: --q4e-moe-vram k=0 does not fit in VRAM: CUDA0 needs 78431 MiB, only 23141 MiB free
  ```

  → needs **78431 MiB**, only **23141 MiB free** — **identical numbers to RUN 1** despite different K.
- **"using device ... free" lines (exact, both occurrences):**

  ```
  0.00.506.109 I llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 3090) (0000:01:00.0) - 23336 MiB free
  0.01.056.289 D llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 3090) (0000:01:00.0) - 23336 MiB free
  ```

- Termination lines (verbatim, end of log):

  ```
  0.01.408.206 E cmn  common_init_: failed to load model 'D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
  0.01.408.207 E srv    load_model: failed to load model, 'D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
  0.01.408.214 I srv   operator (): operator (): cleaning up before exit...
  0.01.409.092 E srv  llama_server: exiting due to model loading error
  ```

### 2. Complete memory breakdown table (verbatim, L5383–L5385 of run2_k0_lv10.log)

```
0.01.363.893 I common_memory_breakdown_print: | memory breakdown [MiB] | total    free     self   model   context   compute    unaccounted |
0.01.363.896 I common_memory_breakdown_print: |   - CUDA0 (RTX 3090)   | 24575 = 23141 + (78431 = 78056 +     178 +     196) +      -76997 |
0.01.363.897 I common_memory_breakdown_print: |   - Host               |                  28128 = 28110 +       0 +      18                |
```

Byte-for-byte identical to RUN 1 (only timestamps differ):
- **CUDA0:** total 24575, free 23141, self 78431 = **model 78056** + context 178 + compute 196, unaccounted −76997
- **Host:** total 28128, free 28110, self 0 (i.e. **model 0**, context 0, compute 0), unaccounted 18

Related placement lines (verbatim):

```
0.00.919.217 I load_tensors: offloading output layer to GPU
0.00.919.217 I load_tensors: offloading 47 repeating layers to GPU
0.00.919.218 I load_tensors: offloaded 49/49 layers to GPU
```

### 3. Buffer-type override lines

- **Count of lines containing "buffer type overridden": 0**
- Distinct destination buffer type names used: **none**
- First 5 such lines: **N/A**
- Last 5 such lines: **N/A**
- Overridden lines matching `ffn_down_exps`: **0** — matching `ffn_gate_up_exps`: **0** (hence no layer grouping)
  - Context: same as RUN 1 — `ffn_down_exps` in 288 lines, `ffn_gate_up_exps` in 96 lines, all `create_tensor: loading tensor blk.N.*` lines with no override attached.

### 4. Buft model MiB (from the breakdown table)

| buft | model MiB |
|---|---|
| Host (CPU) | **0** |
| CUDA0 | **78056** |

---

## 5. Conclusion

The K=10 run does **not** show 38 layers of expert tensors overridden to a host/CPU buft — in fact it shows no override activity at all: zero lines matching "buffer type overridden" (and no other override phrasing in the log), zero expert tensors (`ffn_down_exps`/`ffn_gate_up_exps`) attached to any override, the Host buft carries **0 MiB** of model memory, and the loader explicitly reports "offloaded 49/49 layers to GPU" with the entire 78056 MiB of model weights placed on CUDA0 (this model has 49 layers, not 38, so even a working 38-layer override would still have been visible as a non-zero Host model MiB and as override log lines). K=0 does **not** change the CUDA0 model MiB either: the fit-check requirement is byte-identical in both runs (`CUDA0 needs 78431 MiB, only 23141 MiB free`) and the breakdown tables are identical apart from timestamps, i.e. the `--q4e-moe-vram` knob (k=10 vs k=0) had no measurable effect on tensor placement in build 39 — the check function echoes the different k value, but the computed "needs" total and the buft distribution are unchanged, consistent with the override path either not being applied (e.g. bypassed by `--n-gpu-layers all`) or not implemented/reachable in this build.
