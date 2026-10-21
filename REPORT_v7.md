# q4e v7 報告（2026-10-22）

硬體：Tesla T10 16 GB（CUDA）＋ Ryzen 7 H 255（Zen 4，8C/16T，AVX-512）（12 執行緒、AVX-512）＋ 48 GB RAM ＋ NVMe。
引擎：`engine-v7/`，以 llama.cpp b11796 為底，加入 q4e 三層專家快取。clang-cl 靜態建置，以符合 Smart App Control。

## 1. 結論

| 模型 | 設定 | tg 文章 | tg 程式 | pp | 對照 |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next IQ2_XS | q4e + MTP n=3 | **18.4** | **21.3** | 90–124 | 官方 b11796：15.5 / 16.4，pp 54 |
| Qwen3.8-Flash-Next IQ2_XS | q4e 不用 MTP | 18.0 | 16.9 | ~101 | Strata（T10+CPU）：46 / 54.5 |
| GLM-5.3-Flash Maya-S24 | q4e，B=1 | 2.1–2.3 | 2.0–2.1 | 26–34 | Maya：3.3–3.9 |

- Flash-Next 比官方快 +19%（文章）、+30%（程式），pp 快 1.7–2.3 倍。
- **50 t/s 的目標沒有達成。** 理由見第 5 節；Strata 的 46/54.5 走的是不同的路線。
- GLM 受 SSD 頻寬限制：專家共 80 GiB，VRAM+RAM 只有 43 GiB。q4e 在 GLM 上比 Maya 慢。

## 2. 每個 token 花在哪（Flash-Next，單 token 解碼，`GGML_SCHED_TIMING=1`，每 64 token 取平均）

| 階段 | ms |
|---|---|
| T10 計算（attention、dense、shared expert、router，以及 VRAM 命中的專家 block） | 27.5 |
| CPU 計算（RAM／SSD 命中的專家 block） | 14.5 |
| GPU←→CPU 資料搬移（CUDA copy-in 5.8 ＋ CPU copy-in 2.7） | 8.5 |
| 分層管理（統計、規劃、table 上傳、同步） | 3.2 |
| 排程與同步的空隙 | ~1.5 |
| **合計** | **~55 ms/token → 18 t/s** |

- 用 MTP 時，一次 verify pass 跑 4 個 token：
  - 目標模型約 104 ms（每 token 26 ms，GPU 與 CPU 重疊比較好）。
  - MTP draft 約 34 ms。
  - 平均每次接受 2.5–3.2 個 token，所以是 18.4–21.3 t/s。
- Prefill：每個 token 11 ms，2182 個 token 共 24 s。

## 3. 三層各放什麼

專家依 decayed score（每次使用 +1，每步 ×0.995）排序。分數高的放 VRAM，次高的鎖在 RAM，其餘留在 SSD，靠 mmap 按需讀取。

### Qwen3.8-Flash-Next（48 層，每層 512 專家，每個專家切 2 個 block）

| 層 | 容量 | 內容 | 解碼時命中率 |
|---|---|---|---|
| VRAM | 7.10 GiB（free VRAM − 640 MiB，自動調整） | 約 21% 的 block，就是本次對話最常用的那些。每 64 token 換入約 1 GiB | 70–88% |
| RAM | 25.9 GiB（VirtualLock） | 其餘全部的 block | 11–30% |
| SSD | 0 GiB | — | 0–2%（暖機期間） |

- block 大小：
  - iq2_s 層 0.72 MiB（34 層）
  - iq2_xxs 層 0.62 MiB（11 層）
  - iq1_m 層 0.56 MiB（3 層）
- 每個專家的 down 部分重新打包成 `<model>.q4e-down.bin`，block 才能切成連續的小塊。

### GLM-5.3-Flash（42 層 MoE，每個專家 1 個 block，6.7–7.2 MiB）

| 層 | 容量 | 內容 | 解碼時命中率 |
|---|---|---|---|
| VRAM | 7.78 GiB | 前 10% 的專家 | 27–42% |
| RAM | 35.2 GiB | 接下來的 44% | 22–46% |
| SSD | 37.1 GiB | 其餘 | 30–48% |

GLM 每個 token 約 460 ms：
- graph 410 ms，大多是 CPU 等 SSD 的 page fault。
- 分層管理 40–60 ms。

## 4. 「個人使用、對話只用到一部分專家」：實測工作集

| 模型 | token 數 | 用到的專家 | 90% 的使用量落在 | 95% | 99% |
|---|---|---|---|---|---|
| Flash-Next | 99 | 11.4 GiB | 6.7 GiB | 8.3 GiB | 10.7 GiB |
| Flash-Next | 355 | 17.4 GiB | 8.5 GiB | 10.9 GiB | 15.1 GiB |
| Flash-Next | 894 | 24.5 GiB | 12.0 GiB | 15.2 GiB | 20.4 GiB |
| GLM | 86 | 44.3 GiB | 28.0 GiB | 34.8 GiB | 42.4 GiB |

- 這個假設部分成立：使用量很集中，Flash-Next 有 90% 落在 12 GiB 內。
- 但工作集會跟著 token 數成長，換了話題也會跟著移動。VRAM 只有 7 GiB，所以命中率卡在 70–88%。
- GLM 光是 86 個 token，工作集就有 44 GiB，已經超過 VRAM+RAM 的 43 GiB。

## 5. 試過的方向與結果

| 方向 | 結果 |
|---|---|
| 專家切更小（Flash-Next 每專家 B=1/2/5 個 block） | 13.9 / 11.0 / 9.6 t/s（VRAM 層關閉時測）。切得越細越慢，因為每多一份就多一組 kernel 與搬移。自動模式選 ≤1 MiB 的 block（Flash-Next B=2，GLM B=1） |
| 非同步搬移：第二條 CUDA stream＋pinned staging＋worker thread | 搬移不再擋住解碼，分層管理從 >10 ms 降到 1–3 ms |
| GPU 前段與 CPU split 重疊（scheduler early launch） | 有效，已預設開啟 |
| AVX-512 多列 IQ2 kernel（移植自 Maya，MIT） | CPU 專家計算達 31.6 GB/s（IQ2_S）與 42.8 GB/s（Q2_0） |
| profile 檔：重開時直接把常用專家填進 VRAM | 第一個請求就有 60% 以上的命中率 |
| 自適應 RAM 層（SSD→RAM 換入） | GLM 上與 CPU 搶 SSD，3.1 降到 2.4，**預設關閉**（`Q4E_RAM_SWAP_MB`） |
| RAM 預算計入已映射頁 | 修正 Flash-Next 有 13 GiB 落在 SSD 的問題（14.1 回到 18.4 t/s） |
| BF16→F16（pre-Ampere） | +1%，改為 opt-in（`LLAMA_BF16_AS_F16=1`） |
| 16 執行緒、n_max 2/4、reserve 256、`--cpu-moe-draft`、poll | 都沒有改善 |
| 780M iGPU 當第四層 | 需要 Vulkan SDK（要系統管理員權限），而且 Smart App Control 擋未簽章 DLL。沒有做 |

### 為什麼到不了 50 t/s

- 光是 dense 部分，就要在 T10 上跑約 2000 個小 op，每 token 27 ms。即使 100% 命中 VRAM，上限也只有 ~35 t/s；加上 MTP 約 45 t/s。
- 要超過 50 t/s，需要的是 GPU 端的 op 融合或 micro-batch pipeline，而不是快取：
  - 減少 kernel 數量和 launch 開銷；
  - 或把 GPU 與 CPU 排成兩條 token 管線交錯執行。
- Strata 達到 46/54.5，是因為它走完全不同的執行器路線。

## 6. 使用

- `D:\AI model\bat\Run_q4e-Flash-Next.bat`（建議使用）
- `D:\AI model\bat\Run_q4e-GLM-5.3-Flash.bat`
- 選項：
  - `--q4e`
  - `--q4e-blocks N`
  - `--q4e-vram-mib`
  - `--q4e-ram-mib`
  - `--q4e-reserve-mib`（預設 640）
  - `--q4e-profile 路徑|none`
- 環境變數：
  - `Q4E_SWAP_MB`（每步換入 VRAM 的上限，預設 128）
  - `Q4E_HYST`（換入門檻，預設 1.1）
  - `Q4E_DECAY`（預設 0.995）
  - `Q4E_RAM_SWAP_MB`（預設 0）
  - `GGML_SCHED_TIMING=1`（每段 ms）
- 建置：`engine-v7\build-clang-t10.bat`

程式碼位置：
- `engine-v7/src/llama-q4e.{h,cpp}`
- `engine-v7/src/llama-graph.cpp`（`build_moe_ffn_q4e`）
- `engine-v7/ggml/src/ggml-cpu/`（`mul_mat_id` 與 kernel）
- `engine-v7/ggml/src/ggml-backend.cpp`（timing 與 overlap）
