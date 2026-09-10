# A-15.3 H2D 提升管線（q4e_expert_h2d）設計決策

上游：design.md v5 規格（48 層 × 512 routed experts；L1 VRAM / L2 pinned RAM / L3 SSD）；A-15.1 pool 契約（凍結）。
代碼：`engine/common/q4e-expert-warm.{h,cpp}` 同檔追加（v4.1 `q4e_expert_warm` + `cb_eval` 凍結不動，A-15.1 慣例）。
測試：`engine/tests/test-q4e-expert-warm.cpp` 新增 5 節 93 條硬斷言（H1–H5）。

## 接口（新增，零侵入）

- `q4e_expert_h2d_backend`：`virtual copy_expert(layer, expert, bytes)`，預設 no-op；仿 A-15.1 `q4e_expert_cache_backend` 模式。
  真實 L2→L1 CUDA 複製是 B 側實作；管線核心純 CPU、可回放。
- `q4e_h2d_config`：expert_bytes（job 大小）、queue_cap（預設 64）、copy_latency（ticks，預設 1）、n_layers/n_experts（未綁 pool 時的 range 邊界）。
- `q4e_expert_h2d(backend, cfg)`：
  - `bind(pool)`：issue 前讀 `pool->state_of()`；job 完成後調 `pool->promote()` 一次
  - `tick()`/`now()`：**邏輯 tick 時鐘**，caller 驅動；禁用 wall clock（確定性、可回放）
  - `hint(layer, expert, bytes=-1)`：ngram 提前提升；bytes 可逐次覆蓋；false = dedup / dropped / out-of-range
  - `access(layer, expert)`：實際消費路徑；已 L1 直接返回；inflight 計 overlap；queued 不計；否則 lazy issue
  - `drain_once()`：測試縫（無線程）——先 FIFO issue 全部 queued（調 backend，h2d_issued/h2d_bytes++），再完成到 ready_tick 的 inflight（h2d_completed++，pool->promote）
  - stats：h2d_issued / h2d_bytes / h2d_completed / h2d_dropped / dedup_skips / overlap_hits / accesses / overlap_ratio

## 口徑（測試斷言精確數）

1. job 狀態 queued→inflight→done；job=(layer, expert, bytes, issue_tick, ready_tick)；ready = issue 時 tick + copy_latency。
2. dedup：key=(layer, expert)；已 L1（pool state_of，或 standalone done set）或 queued/inflight → dedup_skips++、不重複入隊；
   **dropped 的 job 不打標** → 隊空後可重試（H2 斷言）。
3. overlap：`access()` 撞見 inflight → overlap_hits++；撞見 queued 不計（copy 尚未開始）。
   `overlap_ratio = overlap_hits / max(1, accesses)`；窗口＝管線生命期（本步無 reset）。
4. `access()` 一律計 accesses；out-of-range（綁 pool 用 pool cfg，否則用 cfg）no-op 不計數（H5 斷言）。
5. 每完成一個 job 只調一次 `pool.promote()`：L2→L1（l2_to_l1++）或 **L3→L2 fill 預留**（l2_fills++，實際 SSD 讀由 A-15.4 填）。
   L3 expert 到 L1 需兩段（fill 預留 job → H2D job），H3 斷言。

## 決策 / 偏差

- 無線程、無鎖：v4.1 有 worker 線程故用 atomics；本類無線程，caller（A-15.5 接線）串行化 hint/access/drain。
- 不做 L2→L1 數據搬運（backend 的職責）、不做 L3→L2 SSD 讀（A-15.4）、不接 server/graph（A-15.5/A-15.2）。
- 首輪 clean-first 建置在 [465/481] 失敗（drain_once deque 迭代器誤用 `it->second`，C2039×2）；
  fix 0f9fc74 後增量建置 212/212 全綠（見 build.log）。
