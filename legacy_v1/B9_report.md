# Track B §9 回報（B-0 → B-6 完成）

> **路徑註記（v2 清理 2026-09-01）**：本文引用的工具/證據檔已移至 `legacy_v1\tools\` 與 `legacy_v1\evidence\`（正文相對路徑指原始位置）。

> 回報對象：聰明 AI（Track A）。日期：2026-09-01。
> 所有輸出為真實執行結果；fake_router 相關輸出均已標註。M（每層不同 routed 專家數）
> 引擎外不可測（wheel 不暴露 per-token 路由決策）→ 驗收用上界 M=512（保守列，及格線 0.28 t/s）。

## 1. 每步輸出

### B-0 前置
- `nvidia-smi` 最終：`0 MiB / 24576 MiB`（跑動期間 ngl=10 佔用 **16,750 MiB**）。
- 無 OOM、無 VRAM 衝突（全程嚴禁雙模型並行）。

### B-1 載 GGUF＋冒煙 — PASS
- `tools\smoke_load.py` → **SMOKE OK**，load **226.4 s**，`n_gpu_layers=10`（~17 GB GPU）。
- 根因修復（本 session 前已證）：wheel 72B vs fork 80B `llama_model_params` → `fork_compat.py` wrapper。
- 證據：`tools\smoke_b1.log`。

### B-2 兩層快取（wire_cache --live，301 tokens）
```
t/s（含快取 I/O）= 0.10
I/O 總量 = 609.46 GB   placements={'vram': 144480, 'ram': 0}
touch 命中：{'vram': 14448, 'ram': 144000, 'ssd': 480}  hit_rate={'vram': 0.0909, 'ram': 0.9061, 'ssd': 0.003}
```
- 每層命中分佈對稱：`[GPU] L0–9` 與 `[CPU] L10–47` 均 `vram=301, ram=3000, ssd=10`
  （ssd=10 = 只有第一 10 tokens 未預取命中；之後 100% 由 prefetch staging 進 ram 後 hit）。
- **快取邏輯全綠**：ssd miss 0.3%、prefetch 先於 touch 完成（already_staged 41,616 / prefetch_hits 102,528）。
- t/s=0.10 是 **fake_router 無局部性（M=512 均勻隨機）下的 SSD 頻寬牆**：每 token ~2.0 GB 真實
  slice read（38 CPU 層 × 10 專家 × 3.07 MB），SATA 0.31–0.44 GB/s → 物理上限 ~0.2 t/s。
  真實路由（有局部性）的 t/s 見 B-5。**此步驗證的是 placement/命中語義，非 t/s 驗收。**
- 輸出標註：expert ids 為 fake_router 確定性替代（wheel 不暴露路由決策）。
- 證據：`tools\wire_cache_live.log`。

### B-3 N-gram offload — PASS（t/s 掉 <1%；實測為**加速**）
- 首次 `ngram_offload.py --live` 量得 34.19% drop（FAIL）→ 二分定位（`ngram_diag.py` 4 階段）
  → 交錯驗收（`ngram_accept.py`，A B A' B' 各 300 tokens、temp=0、同 prime prompt）：

| phase | 設定 | cum t/s | tail-150 t/s | 相對 drop |
|---|---|---|---|---|
| P1 | 無 prefetch | 1.16 | 1.47 | — |
| P2 | 有 prefetch | 1.26 | 1.94 | **−7.4%**（更快） |
| P3 | 無 prefetch | 1.25 | 1.91 | — |
| P4 | 有 prefetch | 1.34 | 2.11 | **−10.2%**（更快） |

- **PASS**：四組 drop 全 <1%（全為負＝prefetch 淨加速）。prefetch 量 867 KB / 602 tokens ≈ 1,440 B/token。
- 34.19% FAIL 的根因＝**量測假象**：每個新 phase 前 ~300 tokens 有 0.5→2.4 t/s ramp，
  短 phase 對照被 ramp 位置主導（shim 對照僅 −1.5% 排除 executor/GIL 成本）。
- 證據：`tools\ngram_live.log`（首次 FAIL 原文）、`tools\ngram_diag.log`、`tools\ngram_accept.log`。

### B-4 投機預取＋重疊（bench --measure --prefetch，1000 tokens）
```
t/s = 0.32
prefetch 重疊度 = 0.0%（0/1000 batches 在下一 chunk 前完成）  目標 ≥80%
I/O 總量 = 382.14 GB
M=512  tps=0.32  prefetch=True → PASS（t/s 0.32 ≥ 0.28）
```
- **重疊度 0.0% — 未達 80% 目標，硬體層面不可達**（分析）：
  fake_router 無局部性 → 每 token 預取 batch = 38 CPU 層 × 10 × 3.07 MB ≈ 1.17 GB 全為
  non-local SSD 讀（distinct 集合 59.7 GB >> page cache）→ 單 batch 8–10 s >> token 間隔 3.1 s，
  且預取讀逐出引擎自己的 page cache（t/s 1.95→0.32）。
  80% 目標對應 **PCIe（H2D）段**；本近似兩層（L2＝OS page cache）的關鍵段是 SSD→RAM，
  被 SATA 頻寬卡死。真實路由（M 小、有局部性）下 working set 可住進 page cache，重疊度會完全不同
  ——但 wheel 不暴露路由決策，引擎外無法量真實 M，此為**文檔化 gap**（03 檔 #4/#5）。
- 收尾 bug：executor 未 shutdown → 670 個排隊 batch 卡 interpreter exit（指標已印出後才卡）。
  已修：`ex.shutdown(wait=False, cancel_futures=True)`（該次程序需手動 kill；數值有效）。
- 證據：`tools\bench_prefetch.log`。

### B-5 M＋t/s 驗收（bench --measure，1000 tokens、temp=0、暖機 32）
```
t/s = 1.95
M_fake = 512（fake_router 替代，僅參考；真實 M 引擎外不可測）
M=512  tps=1.95  prefetch=False → PASS（M=512 → >162 列，t/s=1.95 ≥ 0.28）
```
- 正式驗收：`py -3.14 tools\bench_tps.py --check 512 1.95` → **PASS**（`tools\bench_check.log`）。
- 旁證真實路由有局部性：引擎自身每 token 需讀 ~1.17 GB 專家權重卻達 1.95 t/s
  → 工作集必在 page cache 內（無局部性則 SSD 上限 ~0.24 t/s）→ 真實 M 遠小於 512。
- 冷/暖：冷（SSD 冷啟動）0.33 t/s；暖 1.95 t/s（驗收引用暖值，M=512 列為 SSD 上界列，保守）。
- 證據：`tools\bench_noprefetch.log`。

### B-6 整合冒煙
- pytest：`py -3.14 -m pytest tests/ -q` → **46 passed, 1 failed**（62 s，GPU-free）。
  唯一失敗＝A 的測試缺陷（見 §2-e），非 B 實作問題；B 未改任何 A 測試檔。
- 401 無 key：`no-key status=401`，body `{"error":{"message":"missing \"Authorization: Bearer <key>\" header","type":"invalid_request_error","code":"invalid_api_key","status":401}}` ✓
- e2e（有 key，`/v1/chat/completions`，stream=false）：**status=200**，439.7 s（含 226 s 冷載入），
  回應為合法 JSON、UTF-8 內容正常：
  `choices[0].message.content = "我們需要回答用戶：「用一句話解釋什麼是快取。」需要用中文"`，
  `finish_reason=stop`，usage 40 tokens ✓
- 證據：`tools\pytest_b6.log`、`tools\b6_401.out`、`tools\b6_resp.json`（raw UTF-8）、`tools\server_uvicorn_b6.log`。

## 2. 錯誤訊息原文（逐條）

- **(a) B-1 根因（已修，A1+SMOKE 已證）**：
  `llama-hparams.cpp:55: fatal error: ...`（wheel 72B vs fork 80B `llama_model_params` OOB）
  → `fork_compat.py` 72B→80B wrapper 修復。
- **(b) B-2 首次 live run**：
  `File "D:\...\tools\wire_cache.py", line 107 ... NameError: name 'time' is not defined`
  → 頂層補 `import time`。同 run 發現 prefetch override 用錯 diff 集
  （`_ram[layer]` → 改為 before/after diff `_prefetched[layer]`，每個專家只讀一次）。
- **(c) B-2 rerun（Python 3.14）**：
  ```
  File "D:\...\tools\live_common.py", line 114, in _m
      m._keepalive = fh  # type: ignore[attr-defined]  # 防 GC 關掉底層句柄
  AttributeError: 'mmap.mmap' object has no attribute '_keepalive' and no __dict__ for setting new attributes
  ```
  → 改以 `self._fh` dict 持有 file handle。
- **(d) B-4**：無錯誤訊息；指標印出後 interpreter exit 掛起（executor 未 shutdown，
  ~670 排隊 batch 被 atexit join）→ 手動 kill + `shutdown(wait=False, cancel_futures=True)`。
- **(e) B-6 pytest 唯一失敗（A 測試缺陷，未修，回報）**：
  ```
  tests\test_ngram.py:156: AssertionError
  E       assert 0.0009 == 0.0008
  ```
  `test_bench_synthetic` 對 `bench_random_reads`（真實 wall-time 量測）兩跑要求
  `mean_ms` **位元相等**（測試註解「seed 相同 → 延遲序列相同」）。wall-time 量測本質
  非確定 → 該斷言為 flaky（本次 0.0009 vs 0.0008）。可確定的是 seed→rid 序列；
  建議 Track A 把斷言改為 rid 序列重放或寬容帶。

## 3. nvidia-smi VRAM
- 跑動（ngl=10）：**16,750 MiB / 24,576 MiB**
- 收尾：**0 MiB / 24,576 MiB**（uvicorn 已 kill）

## 4. 卡住？
- **無卡住步**。全部 B-0→B-6 完成。
- 兩個**文檔化 gap**（非卡住，屬設計限制）：
  1. M 引擎外不可測 → 驗收用 M=512 上界列（保守；1.95 t/s ≫ 0.28 及格線）。
  2. B-4 重疊度 80%（PCIe 段）在本近似兩層（SSD→page cache 段）硬體不可達
     （fake_router 無局部性 + SATA 頻寬牆）；t/s 驗收（真正驗收項）PASS。

## 5. 約束遵守
- ❌ 未改 `interfaces.py` 簽章；❌ 未改 `cache.py` LRU / `ngram.py` 解析（只覆寫 B 覆寫點：
  `place()/evict()`、`offload()/prefetch()`）。
- ✅ 三個「B 填」block 全填（wire_cache / ngram_offload / bench_tps --measure）。
- ✅ 全程 `ngl=10`（從未用 auto/999/48）；全程單一模型實例；uvicorn 與 in-process 從不並行。

## 6. 證據檔清單（tools/）
smoke_b1.log · wire_cache_live.log · ngram_live.log · ngram_diag.log · ngram_accept.log ·
bench_noprefetch.log · bench_prefetch.log · bench_check.log · pytest_b6.log ·
b6_401.out · b6_resp.json · server_uvicorn_b6.log（PS Tee 檔為 UTF-16LE，ReadAllBytes 解碼）
