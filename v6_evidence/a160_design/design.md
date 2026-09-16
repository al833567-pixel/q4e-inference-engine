# A-16.0 — v6 設計拍板＋參數語義（100k Context 版）

範圍：v6 參數語義拍板（common_params 欄位＋CLI 解析）＋VRAM/RAM fit 公式＋A-16.1～A-16.8 介面契約。
本文只做**設計與契約**，不含 A-16.1 以後的實作。v5 凍結行為（`q4e_expert_cache`/pool/h2d/warm 既有語義）不變。

## 一、v6 參數總表

| 參數 | 預設 | 範圍（parser 層） | 語義 | 驗證 |
|---|---|---|---|---|
| `--q4e-moe-vram` N | 5000 | 0-99999 | **v6 重定義**：L1 全局 VRAM 池駐留 routed 專家**總數**（跨 48 層共享，非 per-layer pool）；0 = pool off | 整數＋範圍，否則 `std::invalid_argument`；loader 層 VRAM-fit 檢查，不足即載入失敗、永不靜默降級 |
| `--q4e-moe-ram` N2 | 4000 | 0-99999 | L2 Pinned RAM 駐留專家數（v5 語義保留不動）；0 = off | 同上 |
| `--q4e-staging-mb` S | 128 | 0-99999 | Staging Buffer（A/B 雙緩衝）**總大小** MB（A/B 合計 S MB，單緩衝 S/2）；0 = staging off | 整數＋範圍，否則 `std::invalid_argument` |
| `--q4e-layer-floor` F | 10 | 0-99999 | Layer-Aware LRU 每層保底 expert 數；0 = 純全局 LRU；F 超過每層 expert 數時**載入時 clamp** 到每層 expert 數 | 整數＋範圍（parser 層通用，不硬編模型事實）；clamp 由 A-16.1 於載入時執行 |
| `--q4e-ple-mmap` on\|off | on | on/off（含 1/0/true/false 別名） | PLE/ngram 保持 SSD mmap（lazy loading，不佔 RAM/VRAM）；off 時 `LOG_WRN` 警告 | 僅接受 on/off（別名），其他字串 `std::invalid_argument`；**紅線：生產恒 on**（off 只警告、不封鎖，屬生產紀律） |
| `--q4e-kv-ram` MB | 6000 | 0-99999 | KV Cache L2 Pinned RAM 預算 MB（100k 分頁 offload）；0 = off | 整數＋範圍，否則 `std::invalid_argument` |
| `--cache-type-k` TYPE | （上游預設） | TYPE 字串 | 上游既有參數（`common/arg.cpp` `-ctk/--cache-type-k`，已確認存在）；v6 100k 配置要求 `q8_0`（或 `q4_0`） | 上游既有驗證，不改 |
| `--ctx-size` | 0 | int | 100k 預算 = 100000 | 上游既有，不改 |

v5 既有 q4e 參數（`--q4e-prefetch` / `--q4e-ngram-prefetch` / `--q4e-warm-min-freq`）語義與預設**全部不動**。
`--q4e-moe-vram` 的 parser 驗證（整數 0-99999、負數/超範圍/非數字 → 拋 `std::invalid_argument` → parse 回 false）**完全保留**，只更新 help text 為 v6 全局池語義。

## 二、VRAM-fit 公式（24 GB, RTX 3090）

```
VRAM ≈ Attn/Router/Shared (48 層, ngl=all)
     + N × expert_bytes            N = --q4e-moe-vram, expert_bytes ≈ 3.07 MB（01 文檔模型事實）
     + staging                     --q4e-staging-mb（A/B 雙緩衝合計）
     + KV active window            L1 VRAM 僅駐留當前計算所需 KV（例如 8k tokens）
```

代入 v6 基準（N=5000、staging 128 MB、KV 8k q8_0）：

| 項目 | 計算 | 大小 |
|---|---|---|
| Expert pool | 5000 × 3.07 MB = 15350 MB | ≈15.0 GiB（01 文檔預算值 ≈15.4 GiB，含餘裕） |
| Staging | 128 MB | ≈0.12 GiB |
| KV active window | 100k q8_0 ≈ 35 GB → 8k ≈ 35 × 8/100 | ≈2.8 GiB |
| Attn/Router/Shared | 殘差 | ≈5.2 GB |
| **合計** | | **≈23.5 GB ≤ 24 GB ✓** |

結論：N=5000 全局池 + 128 MB staging + 8k KV window 在 24 GB 內成立（02 文檔驗收 #3）。
載入期必須做 hard fit check：VRAM 不足 → 載入失敗並明確指向 `--q4e-moe-vram`（v5 紀律：永不靜默降級）。
**實作現況（2026-09-16，A-16.9 註記）**：本節與 §四 的設計物理路徑（L1 池緩衝＋CUDA gather＋promotion＋graph 手術＝整層拆開、逐 expert 全局池）**尚未實現**——A-16.2/16.3/16.4 交付物為 pure C++ bookkeeping＋CPU 契約（`q4e_gather_launch_cuda` 為 stub、無 q4e CUDA kernel、server 僅將 staging_mb/layer_floor 傳入 stats shim），server 實際推理目前走 **v5 per-layer 駐留路徑**（per-layer buft＋prefetch）。A-16.9 load path 修正（v6 N → `ceil(N/512)` 整層駐留）是對**現有 v5 物理路徑**的相容修正，**不是** §四 設計佈局的落地；設計物理路徑（整層拆開）待另行實作。
v5 的 `common_q4e_fit_vram`（per-layer R = ceil(N/48)）是凍結的 v5 行為，保留供 v5 回歸；v6 全局池的 fit 檢查由 A-16.1 以「N 全局」語義實作（公式同式，但不再除以 48 層）。

## 三、RAM-fit 公式（~20 GB）

```
RAM ≈ q4e-kv-ram (KB L2 Pinned)
    + q4e_moe_ram × expert_bytes   (Expert L2 pool, Pinned RAM)
    + OS 保留
```

| 配置 | KV L2 | Expert L2 | 合計 | 判定 |
|---|---|---|---|---|
| **100k 極限配置**（01 文檔） | 6000 MB ≈ 5.9 GiB | **N2=3000** × 3.07 MB ≈ 9.0 GiB（01 文檔 ≈9.2 GiB） | ≈15 GiB + OS ≈ **17 GB** | ≤ 20 GB ✓ |
| 預設組合 | 6000 MB ≈ 5.9 GiB | 4000 × 3.07 MB ≈ 12.0 GiB | ≈17.9 GiB + OS | **超過** 20 GB 預算 ✗ |

**拍板**：預設值（N2=4000、kv-ram=6000）適用於非 100k 場景或較小 KV L2；
**100k 極限配置必須在 RUNBOOK 顯式傳 `--q4e-moe-ram 3000`**（或等比調低 `--q4e-kv-ram`）。
兩參數的預設值互不推導，避免隱式耦合；組合合法性由部署紀律（02 文檔紅線）把關。

## 四、A-16.1～A-16.8 介面期望（契約，不實作）

- **A-16.1 全局動態池管理器**
  - 輸入：`q4e_moe_vram`（全局 N）、`q4e_layer_floor`（F，載入時 clamp 到每層 expert 數，本模型 512）。
  - 語義：單一 L1 池跨 48 層共享；slot 身份 = (layer, expert_id)；Global LRU 逐出，但每層保底 F 個不逐出（F=0 退化为純全局 LRU）。
  - 紀律：載入期 VRAM-fit 硬檢查（§二公式，全局 N 語義）；不足 → 失敗並指名 `--q4e-moe-vram`，永不靜默降級。
  - 輸出：hit/miss/evict/floor-hit 計數器，供 A-16.6 `/q4e/stats` 使用。
- **A-16.2 VRAM 內 Gather（Staging Buffer A/B）**
  - 輸入：`q4e_staging_mb`（A/B **合計** MB；單緩衝 = S/2）。
  - 語義：把 L1 池中選中的專家（碎片地址）拷入連續 staging 區，再對該區做 `MUL_MAT_ID`；A/B 雙緩衝讓「下一批 gather」與「本批計算」重疊。
  - 邊界：S=0 → staging off，退回逐專家直讀（效能降級，需 LOG_WRN）；staging 空間不足 → 分批 gather，不得 OOB。
- **A-16.3 H2D 提升管線**
  - 輸入：`q4e_moe_ram`（L2 Pinned 槽數）。
  - 語義：SSD → L2 fill；L2 → L1 非同步 H2D（經 A-16.2 staging 落位）；與 v5 凍結的 warm/prefetch 掛鉤（`q4e_prefetch`/`q4e_ngram_prefetch`/`q4e_warm_min_freq`）相容，不改其既有行為。
  - 輸出：fill/H2D 佇列深度、带宽計數器（A-16.6）。
- **A-16.4 Graph 手術**
  - 48 層全 GPU；動態 slot 映射（L1 全局池 → graph 內 expert 地址）；promotion op（L2→L1）。
  - 驗收：逐層輸出對 oracle（02 文檔 #4）。
- **A-16.5 100k KV Cache 分頁 Offload**
  - 輸入：`q4e_kv_ram`（L2 Pinned 預算 MB）、`--ctx-size 100000`、`--cache-type-k q8_0`（上游既有）。
  - 語義：L1 VRAM active window（如 8k）→ L2 Pinned（paged，預算 = q4e_kv_ram）→ L3 SSD pagefile；量化強制 q8_0/q4_0。
  - 驗收：100k 載入不 OOM＋KV 命中統計（02 文檔 #2）。
- **A-16.6 Server 接線＋`/q4e/stats` v6**
  - 上線欄位：expert 池 hit/miss/evict（A-16.1）、staging 使用率與批次深度（A-16.2）、L2 fill/H2D 統計（A-16.3）、KV L1/L2/L3 分頁命中與位元組（A-16.5）。
- **A-16.7 q4e-bench / q4e-e2e v6**：含 100k 壓力測試；B-16 直接跑取數（ON 暖 t/s ≥ 8，理想 ≥ 40）。
- **A-16.8 交付包 v6**：README/01/02/03A/03B 全落地；RUNBOOK 必含 §三 的 100k 組合（`--q4e-moe-ram 3000`）。

## 五、未決問題／風險（移交 A-16.1+ 決策）

1. **RAM 預設組合超預算**：N2=4000 + kv-ram=6000 ≈ 18 GiB，超 100k 極限配置（§三）。RUNBOOK 必須顯式組合，A-16.5/A-16.8 落地前不得用預設跑 100k。
2. **staging 計數口徑**：本設計拍板為 A/B **合計** = S MB（單緩衝 S/2）；A-16.2 實作若改口徑須同步更新本文與 help text。
3. **layer-floor clamp**：parser 層不硬編 512（模型事實）；clamp 在 A-16.1 載入期以實際每層 expert 數執行。
4. **`--q4e-ple-mmap off`**：只 `LOG_WRN`、不封鎖（紅線屬生產紀律）；若日後要硬封鎖需另立參數決策。
5. **v5 fit 函數共存**：`common_q4e_fit_vram/ram`（per-layer 語義）與 v6 全局池 fit（A-16.1）將並存，命名與錯誤訊息需明確區分 v5/v6，避免混淆。
