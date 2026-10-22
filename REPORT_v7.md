# q4e v7 報告（2026-10-22，第二輪最佳化後）

- 硬體：Tesla T10 16 GB（CUDA）＋ Ryzen 7 H 255（Zen 4，8C/16T，AVX-512）＋ 48 GB RAM ＋ NVMe。
- 引擎：`engine-v7/`，以 llama.cpp b11796 為底，加入 q4e 三層專家快取。clang-cl 靜態建置，以符合 Smart App Control。

## 1. 結論

| 模型 | 設定 | tg 文章 | tg 程式 | pp | 對照 |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next IQ2_XS | q4e + MTP n=3，8 執行緒 | **19.1–20.5** | **26.5–28.0** | ~210 | 官方 b11796：15.5 / 16.4，pp 54 |
| Qwen3.8-Flash-Next IQ2_XS | q4e 不用 MTP | 19.2 | 20.0 | ~200 | Strata（T10+CPU）：46 / 54.5 |
| GLM-5.3-Flash Maya-S24 | q4e，每專家 1 塊 | **4.3–5.3** | | ~37 | Maya：3.3–3.9 |

- 與第一輪（18.4 / 21.3，pp 90–124；GLM 2.1–2.3）相比：
  - Flash-Next 文章 +8%、程式 +27%，pp 約 2 倍。
  - GLM 約 2.3 倍，現在比 Maya 快。
- Flash-Next 對官方：文章 +27%，程式 +66%，pp 約 4 倍。
- **50 t/s 的目標還沒有達成。** 理由見第 6 節。

## 2. 每個 token 花在哪（Flash-Next，`GGML_SCHED_TIMING=1`，每 64 token 平均）

### 單 token 解碼（約 46 ms → 21.5 t/s，server 實測 19–20）

| 階段 | 第一輪 ms | 現在 ms |
|---|---|---|
| T10 計算（dense、router、VRAM 命中的專家 block） | 27.5 | 24.7 |
| T10 等待並接收 CPU 的結果（copy-in） | 5.8 | 8.4 |
| CPU 計算（沒命中 VRAM 的專家 block） | 14.5 | **4.8** |
| CPU 接收輸入（copy-in） | 2.7 | 2.6 |
| 分層管理（統計、搬移規劃、table 上傳） | 3.2 | 2.8 |
| 空隙 | ~1.5 | ~2.8 |
| **合計** | **~55** | **~46** |

### MTP：一次 verify pass，4 個 token，約 87 ms

| 階段 | ms |
|---|---|
| T10 計算 | 40 |
| T10 copy-in | 9 |
| CPU 計算 | 16 |
| CPU copy-in | 4 |
| 分層管理 | 5 |
| 空隙 | ~5 |
| MTP draft（3 個草稿 token） | 6.6 |
| **合計** | **~87** |

平均每次 pass 接受 2.75 個 token（各位置接受率 0.79 / 0.56 / 0.39），換算約 31 t/s。server 實測 19–28 t/s：文章的接受率較低，另有 sampling 和 HTTP 的開銷。

## 3. 三層各放什麼

專家依 decayed score 排序（每次使用 +1，每步 ×0.995）：

- 分數最高的放 VRAM。
- 次高的鎖在 RAM。
- 其餘留在 SSD，靠 mmap 按需讀取。

### Qwen3.8-Flash-Next（48 層，每層 512 專家，每個專家切 2 個 block）

| 層 | 容量 | 內容 | 解碼時命中率 |
|---|---|---|---|
| VRAM | 7.1 GiB（用 MTP）／9.4 GiB（不用 MTP）；自動取 free VRAM − 640 MiB | 這次對話最常用的 21–28% block，每 64 token 換入約 1 GiB | 單 token 78–83%；MTP 63–70% |
| RAM | 23.6–25.9 GiB（VirtualLock） | 其餘全部 block | 17–34% |
| SSD | 0 GiB | — | 0–3% |

block 大小：
- iq2_s 層 0.72 MiB（34 層）
- iq2_xxs 層 0.62 MiB（11 層）
- iq1_m 層 0.56 MiB（3 層）

### GLM-5.3-Flash（42 層 MoE，每個專家 1 個 block，6.7–7.2 MiB）

| 層 | 容量 | 內容 | 解碼時命中率 |
|---|---|---|---|
| VRAM | 7.78 GiB | 前 10% 的專家 | 25–32% |
| RAM | 35.2 GiB | 接下來的 44% | 46% |
| SSD | 37.1 GiB | 其餘 | 22–29% |

GLM 每個 token 約 200–240 ms（第一輪約 460 ms）：
- graph 150–180 ms。
- 分層管理 50–60 ms。

## 4. 「個人使用、對話只用到一部分專家」：實測工作集

| 模型 | token 數 | 用到的專家 | 90% 的使用落在 | 95% | 99% |
|---|---|---|---|---|---|
| Flash-Next | 99 | 11.4 GiB | 6.7 GiB | 8.3 GiB | 10.7 GiB |
| Flash-Next | 355 | 17.4 GiB | 8.5 GiB | 10.9 GiB | 15.1 GiB |
| Flash-Next | 897 | 24.6 GiB | 12.2 GiB | 15.3 GiB | 20.4 GiB |
| GLM | 86 | 44.3 GiB | 28.0 GiB | 34.8 GiB | 42.4 GiB |

使用量確實集中，但工作集會隨對話成長、移動。VRAM 7–9 GiB 只裝得下約 90% 使用量的一半到七成。

## 5. 第二輪改了什麼

| 改動 | 結果 |
|---|---|
| **略過低權重的未命中專家**（`Q4E_SKIP`，預設 0.12：權重不到該 token 總路由權重 12% 的未命中專家不算，命中的照算） | Flash-Next CPU 每 token 12.6 → 5.2 ms；GLM 少讀很多 SSD，2.1 → 5.3 t/s。輸出仍通順。**這是有損的**，Strata 也有類似做法；設 `Q4E_SKIP=0` 即回到精確模式 |
| CPU 端 routing op：CPU 自己由 host 端 slot table 算出 ids／權重，並寫入選擇結果 | 省掉 selection 從 GPU 讀回 |
| GPU 端保留自己的 routing（`Q4E_ROUTE_GPU=1`，預設） | GPU 的命中 block 不必等 CPU 來回：MTP 17.9 / 25.2 → 18.8 / 27.1 |
| 執行緒 12 → 8 | CPU 那段很短，執行緒少，barrier 開銷也少：+1–2 t/s；16 執行緒反而掉到 16 / 21 |
| CUDA op profiler 改用 event 計時 | 不再每個 op 同步，數字比較準 |

### 試了但沒有用

| 改動 | 結果 |
|---|---|
| 專家切更細（GLM 每專家 2 塊） | VRAM 命中率不變（27%）且更慢。原因：被選中的專家，每一塊每次都要算，所以各塊熱度一樣，切細無法讓 VRAM 多放熱門部分 |
| MTP draft 的專家也交給 q4e（`Q4E_DRAFT_MIB`） | draft 每步慢 2.5 倍，主池反而變小：8.5 / 14.5 t/s，預設關閉 |
| KV cache q8_0 | 沒有差別 |
| MTP n=4 | 程式 27.4，但文章 16.7，平均沒比 n=3 好 |
| 第一輪已試：16 執行緒、reserve 256、`--cpu-moe-draft`、poll、BF16→F16、自適應 RAM 層 | 沒有改善 |

## 6. 為什麼還到不了 50 t/s

- 現在的瓶頸是 T10 本身：單 token 時 GPU 佔約 33 ms（計算 24.7 ＋ 等待 8.4），CPU 只剩約 5 ms。
- 這個架構（hyper-connections、linear attention、indexer）在 llama.cpp 裡，每個 token 要跑約 2000 個小 kernel。即使 CUDA graph 已經開著，dense 部分仍要 ~25 ms。
- 要到 50 t/s，需要 GPU 端的 kernel 融合，把每層幾十個小 op 合成幾個。這是自己寫 decode kernel 的工作，Strata 就是這麼做的；不是快取或分層能解決的。
- 次要的改善空間：把 GPU routing 的 ~14 個小 op 寫成一個 CUDA op，估計省 2–3 ms/token。

## 7. 使用

- `D:\AI model\bat\Run_q4e-Flash-Next.bat`（建議）
- `D:\AI model\bat\Run_q4e-GLM-5.3-Flash.bat`

選項：
- `--q4e`
- `--q4e-blocks N`
- `--q4e-vram-mib`
- `--q4e-ram-mib`
- `--q4e-reserve-mib`（預設 640）
- `--q4e-profile 路徑|none`

環境變數：
- `Q4E_SKIP`（預設 0.12，0 = 精確）
- `Q4E_ROUTE_GPU`（預設 1）
- `Q4E_SWAP_MB`（預設 128；GLM bat 設 32）
- `Q4E_HYST`（預設 1.1）
- `Q4E_DECAY`（預設 0.995）
- `Q4E_RAM_SWAP_MB`（預設 0）
- `Q4E_DRAFT_MIB`（預設 0）
- `GGML_SCHED_TIMING=1`（各段 ms）
- `GGML_CUDA_PROFILE=1` ＋ `GGML_CUDA_DISABLE_GRAPHS=1`（各 op 的 GPU 時間）

建置：`engine-v7\build-clang-t10.bat`。
- Smart App Control 偶爾會擋下新編出的 llama-server.exe。
- 遇到時，隨意改一行字串再編一次即可；不需要更改系統設定。

程式碼位置：
- `engine-v7/src/llama-q4e.{h,cpp}`
- `engine-v7/src/llama-graph.cpp`（`build_moe_ffn_q4e`、`q4e_route_*`）
- `engine-v7/ggml/src/ggml-cpu/`
- `engine-v7/ggml/src/ggml-backend.cpp`
- `engine-v7/ggml/src/ggml-cuda/ggml-cuda.cu`
