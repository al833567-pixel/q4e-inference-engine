# A-15.0 — v5 設計拍板＋參數語義（三層全 GPU MoE）

> 狀態：**拍板（2026-09-10）**。本文件＝v5 參數語義、fit 公式、LRU 逐出語義、依賴序的**單一權威來源**。
> 引擎側落點：`common.h`（`q4e_moe_vram`/`q4e_moe_ram`＋fit 結果結構）／`common.cpp`（`common_q4e_fit_vram`/`common_q4e_fit_ram`）／`common/arg.cpp`（兩旗標解析）／`tests/test-arg-parser.cpp`（解析＋fit 硬檢查）。
> 上游：[02_實作項目工作規劃.md](../../02_實作項目工作規劃.md) §二 v5 表＋§四 placement 紅線；[03B B-9 門](../../03B_項目現況_TaskB.md)。

## 1. 三層定義

| 層 | 媒體 | 角色 | 容量口徑 |
|---|---|---|---|
| **L1** | VRAM（RTX 3090 24 GB） | **expert pool**：每層 R 槽；token 的 top-10 專家命中 pool 內→GPU 直算（零 I/O） | `--q4e-moe-vram N`＝跨 48 層 VRAM 駐留 routed 專家數；每層 pool **R＝⌈N/48⌉**（上界＝每層專家數 512） |
| **L2** | pinned RAM（OS page cache backing） | **H2D 來源**：L1 miss 時從 L2 非同步提升進 pool 槽（`cudaMemcpyAsync`） | `--q4e-moe-ram N2`＝跨 48 層 L2 駐留 routed 專家數；每層 ⌈N2/48⌉（上界 512） |
| **L3** | SSD（GGUF 4 片，SATA 0.31–0.44 GB/s） | **冷層**：L2 miss→SSD 讀入 L2→再 H2D 進 L1 | 無參數；剩餘 512−⌈N2/48⌉/層 恆在 L3 |

非 MoE 固定部（attn/SSM/shexp/output）＋router 恆 VRAM（`--n-gpu-layers all`）；**PLE per_layer_token_embd＝27,465.95 MiB 恆 host**，不進任何 pool、不進 fit 公式。

## 2. 參數語義（v5）

| 旗標 | env | 語義 | 預設 | 範圍 | 裝不下 |
|---|---|---|---|---|---|
| `--q4e-moe-vram N` | `Q4E_MOE_VRAM` | VRAM 駐留 routed 專家數（跨 48 層；每層 pool R＝⌈N/48⌉） | **5000** | **0–99999** | **載入跳錯**（清晰錯誤，不悄悄降 N） |
| `--q4e-moe-ram N2` | `Q4E_MOE_RAM`（新） | L2 pinned RAM 駐留 routed 專家數（跨 48 層） | **4000** | **0–99999** | **載入跳錯**（清晰錯誤，不悄悄降 N2） |

- `N=0`＝pool 關閉（0 槽、0 MiB、恆 L3）；解析仍合法。
- N>512×48=24576 時 pool 自動上界＝512/層（全專家駐留），fit 公式照算（cap 已入實作）。
- 規劃文「≈104/層」「≈83/層」＝N/48 的整數近似顯示；**實作以 ⌈·/48⌉ 為準**：N=5000→R=105（pool 5040）、N2=4000→R=84（pool 4032）。
- 兩參數獨立：B-9 可調 N/N2 找上限；任一層裝不下＝載入跳錯（紅線＝不悄悄降）。

## 3. LRU 逐出語義

- **per-layer LRU**：每層 pool 獨立 84/105/512 槽；逐出該層最久未命中槽位。
- **L2 副本保留**：L1 槽逐出時**不做 D2H**——專家權重在 L2（pinned RAM）仍有副本（N2 覆蓋時）→逐出＝**純 metadata no-op**（slot→global 映射表更新、LRU 鏈更新、hit 簿記）。
- L2 未覆蓋的專家（N2 之外）：L1 逐出後下訪走 L3（SSD→L2 fill→H2D）。
- 提升路徑：L1 miss→（L2 hit？H2D 直提升：SSD→L2 fill 後 H2D）→寫入 LRU 最舊槽→映射更新。重疊度量／dedup／ngram 提前提升＝A-15.3 範圍（本步只定語義，不接線）。

## 4. fit 公式（純函式、可無 GPU 單元測試）

模型實測（qwen4exp，48 層 MoE、512 experts/層、top-10）：

| 項 | 值 |
|---|---|
| 每層 expert 權重 | gate 450 MiB (q4_K)＋up 450 MiB (q4_K)＋down 600 MiB (q5_1)＝**1550 MiB/層** |
| **每 expert bytes** | 1550 MiB ÷ 512＝**≈3.03 MiB/expert**（含 scales 實測帳） |
| 非 MoE 固定部（attn/SSM/shexp/output） | ≈4,342 MiB（恆 VRAM）＋ctx/compute |
| PLE per_layer_token_embd | 27,465.95 MiB（恆 host，**不計入** VRAM/RAM fit） |

公式（`n`＝N 或 N2；`R`＝每層 pool 槽）：

```
R          = ceil(n / n_layers)                       # 48 層；R 上界 = n_expert_per_layer(512)
pool_total = R * n_layers
required_mib = ceil(pool_total * expert_bytes / (1024*1024))
fit        = (required_mib <= free_mib)
```

- **VRAM-fit**：`required_mib(N) ≤ GPU free`；否則跳錯。完整 VRAM 預算＝required(N)＋非 MoE ≈4,342＋ctx/compute（既有 `common_get_device_memory_data` 管路量，A-15.1 接）。
- **RAM-fit**：`required_mib(N2) ≤ RAM free`；否則跳錯。
- 數值例（expert_bytes＝1550×2²⁰÷512）：N=5000→R=105、pool=5040、required≈**15,258 MiB（≈14.9 GiB）**；N2=4000→R=84、pool=4032、required≈**12,207 MiB（≈11.9 GiB）**。
- **不悄悄降**：not-fit 時 pool 尺寸保持（跳錯）；錯誤文字含旗標名、required/free MiB。

### C++ 簽名（已實作，`common.h`/`common.cpp`）

```cpp
struct common_q4e_fit_result { int64_t per_layer_pool, pool_total, required_mib; bool fits; std::string error; };
common_q4e_fit_result common_q4e_fit_vram(int n_layers, int n_expert_per_layer, int64_t expert_bytes, int n_vram, int64_t free_mib);
common_q4e_fit_result common_q4e_fit_ram (int n_layers, int n_expert_per_layer, int64_t expert_bytes, int n_ram,  int64_t free_mib);
```

## 5. 旗標語意分界聲明（build 42 vs v5）

| | **build 42（v4.1，凍結交付件，`207fe88`）** | **v5（A-15.0 起，build 43+）** |
|---|---|---|
| `--q4e-moe-vram K` | K＝**GPU 駐留層數**；預設 10；範圍 0–999；K 層 expert 恆 VRAM、其餘 38 層 lazy host/SSD（`tensor_buft_overrides` CPU buft 路徑） | **同一旗標名、語意重定義**：N＝**VRAM 駐留 routed 專家數**；預設 5000；範圍 0–99999；48 層 expert 全 GPU 算、pool 動態駐留 |
| `--q4e-moe-ram N2` | 不存在 | **新旗標**；L2 pinned RAM 專家數；預設 4000；範圍 0–99999 |
| 行為 | 已交付、B-8 復驗對象；**二進位凍結，v5 改動不影響它** | A-15.0 只落參數＋fit 公式＋測試；pool 機制 A-15.1~A-15.3 接；v4.1 的 `common_q4e_moe_vram_apply/check`＋`llm_add/insert_q4e_moe_vram_overrides` **原樣保留**（不刪、不改行為） |

> **同旗標兩語意**：以**二進位 build 號**分界（≤42＝K 層數語意；≥43＝N 專家數語意）。文件/RUNBOOK 各版本對應各自語意，不混用。

## 6. 依賴序＋各步驗收門（02 §二 v5 表，逐字摘）

A-15.0 → A-15.1 → **A-15.3（H2D 管線）** → **A-15.2（graph 手術，最高風險）** → A-15.4 → A-15.5 → A-15.6 → A-15.7 → **B-9**（B 側 GPU）。

| 步 | 驗收門 |
|---|---|
| **A-15.0（本步）** | 設計文件＋參數規格＋fit 公式＋C++ 參數解析/fit 單元測試（test-arg-parser 全綠） |
| A-15.1 pool 管理器 | C++ 全綠＋單元測試（pool 容量／LRU 順序／promote/evict／hit 分佈／真實 M 計數） |
| A-15.3 H2D 管線 | C++ 全綠＋C++ 測試（H2D bytes／重疊比／dedup／提前提升） |
| A-15.2 graph 手術 | C++ 全綠＋**oracle 對答案**（v5 placement 逐層輸出比對）＋B 側載入成功 |
| A-15.4 L2 pool＋SSD→L2 fill | C++ 全綠＋C++ 測試（L2 hit／fill bytes／SSD→L2） |
| A-15.5 server＋stats v5 | 端點全欄位上線＋C++ 測試 |
| A-15.6 bench/e2e v5 | B-9 直接跑 q4e-bench/q4e-e2e 取數 |
| A-15.7 交付包 v5 | 交付包可獨立執行；B 端逐項對「✓」 |
| **B-9** | ON 暖 t/s ≥2× B-8 實測 v4.1 ON（M≤R 時，目標 ≥8）＋L1 hit ≥80%（M≤R 時）＋oracle 對＋VRAM-fit（N=5000 載入成功、過大跳錯）＋e2e 5 閘＋冷載入 ≤~250 s＋VRAM 歸零 |

## 7. 風險

1. **20 GB RAM 下 L2 吃緊**：N2=4000→required≈11.9 GiB，加 OS/既有進程→RAM-fit 可能跳錯；B-9 視實機降 N2 或調 N（跳錯＝預期行為，非 bug）。
2. **WDDM**：`nvidia-smi` free 為樂觀值（驅動保留）；fit 用同一口徑，實占超限＝載入失敗（載入跳錯紅線保護）。
3. **SATA 0.31–0.44 GB/s**：L3 miss 路徑＝SSD bound；M≫R 時 v5 收益受限（仍優於 CPU 算）。
4. **A-15.2 graph 手術＝最高風險**（pool tensor＋id-remap＋promotion op）；驗收門含 oracle 逐層對答案。
5. **build 42 凍結**：v5 所有改動 build 43+；`tensor_buft_overrides` v4.1 路徑保留原樣，避免回歸。

## 8. 本步交付與證據

- 設計文件：本檔。
- 代碼：`common.h`（＋`q4e_moe_ram`、fit 結構與簽名；`q4e_moe_vram` 預設 10→5000、註解改專家數語意）；`common.cpp`（＋fit 實作 3 函式）；`common/arg.cpp`（`--q4e-moe-vram` 範圍 0–99999＋help 改語意；＋`--q4e-moe-ram`）。
- 測試：`tests/test-arg-parser.cpp` 三新區塊（v5 語意解析、`--q4e-moe-ram` 解析、fit 公式硬檢查）；v4.1 override 生成/insert 區塊保留為**回歸**（改標 build 42 路徑）。
- build＋test log：`build.log`（tail 200）＋`testargparser.log`（全量）同目錄。
