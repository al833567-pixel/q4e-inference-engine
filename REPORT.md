# q4e v7 報告

硬體：Tesla T10 16 GB（CUDA，Turing，PCIe 3.0 x4）＋ Ryzen 7 H 255（Zen 4，8C/16T，AVX-512）＋ 48 GB DDR5 ＋ NVMe。
引擎：`engine-v7/` = llama.cpp b11796 ＋ q4e。Windows 上以 clang-cl 靜態建置（`scripts/build-windows-cuda.bat`）。

## 1. 結果

| 模型 | 設定 | essay t/s | code t/s | 對照 |
|---|---|---|---|---|
| Qwen3.8-Flash-Next IQ2_XS | q4e + MTP n=3，32k | **37.3** | **49.7** | llama.cpp b11796（experts 在 CPU）：15.5 / 16.4 |
| Qwen3.8-Flash-Next IQ2_XS | q4e + MTP n=3，128k | 33.5 | 46.3 | |
| Qwen3.8-Flash-Next IQ2_XS | q4e，不用 MTP | 26.6 | 27.6 | |
| GLM-5.3-Flash | q4e，每專家 1 block，-t 12，暖 profile | 11.5 | 10.9 | 冷 profile 4.3–5.3 |

Flash-Next 的輸出（temperature 0）在所有無損最佳化前後逐字相同；`Q4E_SKIP=0.12`（略過權重很小的未命中專家）是唯一的有損預設，設 0 即完全精確。

## 2. q4e 怎麼運作

- **block**：每個 routed expert 切成數個 FFN 列區塊（Flash-Next 每專家 2 塊、各 320 列；GLM 1 塊）。block 本身就是一個小專家，專家輸出 = 各 block 輸出之和。down 投影是欄切片，第一次啟動時重排寫入 `<model>.q4e-down.bin`。
- **三層**：VRAM pool（最熱的 block，GPU 算）／RAM（鎖在記憶體，CPU 同一步算）／SSD（經 mmap 由 CPU 讀）。每步結束依分數把熱 block 搬進 VRAM（第二條 stream、256 KiB 一段，每步預算 32 MiB），分數存到 `<model>.q4e-profile`。
- **routing**：GPU 用 `GGML_OP_Q4E_ROUTE` 直接從 slot 表算出 pool 的 ids/weights（未命中為 -1，mmvq kernel 跳過）；CPU 端 routing op 寫出 CPU tier 的 ids 並記錄 selections。
- **CPU tier**：`GGML_OP_Q4E_EXPERTS` 一個 op 完成 gate/up/swiglu(或 swiglu_clamp)/down/加權和（與分開的 op 位元相同），AVX-512 多列 IQ2 核心。
- **排程**：CPU 先算（輸入在 router 之後即備妥），GPU 不需等 CPU 輸出的節點提前啟動；host↔device 輸入非同步複製（`GGML_SCHED_FAST_JOIN`）。

## 3. 每 token 時間（Flash-Next，MTP n=3，32k）

約 14.4 ms／token（每輪驗證 4 個 token、平均接受 ~3.4）：CPU 專家 3.4–3.9 ms（~50 GB/s，接近這台 RAM 的實際上限）、主機等 GPU ~7 ms、GPU busy ~57%。VRAM 命中率隨對話由 ~44% 升到 ~64%。

## 4. 有效的最佳化（依序）

| 項目 | 效果 |
|---|---|
| 專家切 block、三層快取、profile | 基礎（第一輪 18.4 / 21.3） |
| Q4E_SKIP 0.12、GPU 端 routing、-t 8 | ~20 / 27 |
| MTP n=3 草稿（draft-mtp） | +30–40% |
| 未命中 block 的 id = -1，mmvq 跳過 | 無損，少讀零槽 |
| 搬移分段 256 KiB、RAM lock 移到 worker | tiers 0.6–1.2 → 0.23 ms/token |
| hc_* 權重 BF16 → Q8_0（載入時） | dense 讀取減半 |
| CUDA 融合：scale+silu、x*sigmoid(g)、conv state、GDN gates、hc_post+norm、norm gate、rms+scale×2 | 31.6/41.9 → 33.3/44.2 |
| mmvq small-K 路徑擴到 2–4 欄 | iq4_xs [2560x4096]x4 45→31 us，+1.5% |
| CPU 融合專家 op（GGML_OP_Q4E_EXPERTS） | CPU 專家時間 -23%，34.1/45.3 → 36.0/48.0 |
| GLM：-t 12 | code 8.7 → 10.9 |

## 5. 試過但無效（已移除或預設關）

- 兩個 ubatch 交錯 pipeline（LLAMA_PIPE=1）：GPU busy 57%→89%，但 dense 權重讀兩次，變慢。
- hc_mix 融合 kernel、hc_inject 與 hc down 合併、activation q8_1 共用、GPU 專家 gate+up / down+wsum 融合 kernel、mmvq 每 block 1 列。
- iq2_s GPU 查表放 shared memory、byte 取法改寫、SIMD intrinsic 改寫（T10 上 iq2_s mat-vec 只有 ~25 GB/s，原因未明）。
- MTP 草稿詞表裁切（接受率下降）、更大 VRAM pool（reserve 256）、更小 ubatch、Q4E_DECAY/HYST 調整、zero-copy 小傳輸、spin。
- GLM：內建 NextN MTP（每輪驗證要讀更多專家）、SSD 預讀提示、融合 CPU op（GLM 大專家上慢 5%）、加大 RAM tier、SSD→RAM 搬移。
- iGPU（Radeon 780M，Vulkan）＋ dGPU、CPU 不算專家：可行但較慢（Flash-Next 24.7 / 28.7）。780M 的專家 kernel 頻寬與 CPU 相當，但每層 T10↔780M 交接的延遲每 token 約 15 ms，且 780M 只放得下 ~30 GiB；GLM 的 80 GiB 專家放不下。此方案已移除。

## 6. 限制

- 需要 MoE 模型且 experts 為一般 gate/up/down（不支援 merged gate_up）。
- 解碼最佳化針對 ≤4 token 的 ubatch（MTP 驗證）；大 batch（prompt）走一般路徑。
- 速度上限在這台機器：CPU 專家受 RAM 頻寬、GPU 受 T10 的 IQ2 解碼與每層同步延遲限制。
