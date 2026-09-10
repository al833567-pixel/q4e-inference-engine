# A-15.1 - 專家 pool 駐留管理器（設計決策）

> 狀態：**done**。代碼＝`engine/common/q4e-expert-cache.{h,cpp}`（新增 `q4e_expert_pool`）；測試＝`engine/tests/test-q4e-expert-cache.cpp`（＋9 支 pool 測試，復用既有檔）。
> 上游：[a150_design/design.md](../a150_design/design.md) §1/§3（三層定義＋LRU 逐出語義）。

## 設計決策

1. **新增 struct `q4e_expert_pool`，不改 `q4e_expert_cache`**：legacy cache＝v2/v3 凍結契約（A-9 cb_eval hook 在用；其 touch()＝move-based 單駐留、被既有測試凍結）。v5 語義不同（L2 副本保留、L1 逐出＝metadata no-op）；硬改會破凍結測試＋hook 路徑。同檔案新 struct＝貼合「擴充此檔」的指令、復用既有 LRU helper、不引入新子系統。
2. **狀態派生、不另存**：per-(layer,expert) 狀態＝最高駐留層：l1_pos 有→L1；否則 l2_pos 有→L2；否則 L3（SSD 檔恆在、無限、不追蹤）。無需獨立狀態表。
3. **copy 語義（非 move）**：design.md §3「L2 副本保留、免 D2H」→L2→L1 提升**不移除** L2 條目（H2D 是 copy、pinned buffer 保留）；L2 槽逐出＝釋放槽位，專家若已在 L1 則留 L1、否則落 L3。L1 逐出只動 L1 鏈（metadata no-op）。
4. **access()＝hit 計數＋提升**：L1 hit＝LRU 刷新；L2 hit＝L2 刷新＋L1 insert（H2D hook，A-15.3 做 copy）＋計 l2_to_l1；L3 hit＝L2 fill（預留槽）＋L1 insert＋計 l3_reads。`l1_slots=0`（N=0）或 `l2_slots=0`（N2=0）乾淨降級（pool 關/SSD→VRAM 直取）。
5. **promote()＝A-15.3 的顯式 hook**：L3→L2 fill 預留 或 L2→L1；已在 L1＝no-op。純狀態轉移＋計數；**零 CUDA 呼叫**（本步純 CPU 紀律）。
6. **round M 計數**：`begin_round(layer)`/`end_round(layer)`；round 活躍期 access() 去重入 per-layer set；end_round 提交 global（rounds / m_sum / m_max）＝「取簡單」口徑；per-layer 隔離由 layer 參數保證。
7. **`per_layer_slots()`**：⌈n/48⌉ 上界 512；N=5000→105、N2=4000→84（＝design.md §2 數值）。ctor 同時把 config 槽位 cap 到 n_experts_per_layer。

## 驗收

- build：flat mirror `D:\q4e_build\src`、`--clean-first`、**479 步、exit 0**（q4e-server＋test-arg-parser＋test-q4e-expert-cache 全重編；見 build.log）。
- `test-q4e-expert-cache.exe`：**159 checks 全綠、exit 0**（legacy 98＋新 61；pool 容量上限/LRU 順序/promote+evict 轉移/hit 分佈/M distinct 各≥1 斷言＋disabled caps＋range＋reset）。
- `test-arg-parser.exe`：**all tests OK、exit 0**（v4.1 回歸＋v5 fit 硬檢查，未動）。
- build 號：＝flat mirror git rev count。flat HEAD 推進到 A-15.1 mirror commit（44→45，flat commit `ed342e3`）後重跑 CMake configure（build-info.cmake 於 configure 時取 `git rev-list --count HEAD`）→二進位實測：

  ```
  version: 0.3.0-dev (build 45, commit ed342e3)
  built with MSVC 19.44.35228.0 for Windows AMD64
  ```

## 偏離/疑慮

- 無代碼偏離。build 號＝45（＝協議預期；flat 44＋mirror commit 1）。
- main repo 比 flat 多 1 個 A-15.0 尾 commit（211861b，docs/evidence）；本步 main 代碼 commit 與 flat mirror commit 為**兩個 repo 的兩個 hash**（parent 不同）；二進位 `--version` 報 flat 端 hash（build-info 在 flat 源樹取 git）。
- `q4e_expert_pool` 尚未接任何 server/graph 路徑（A-15.2/A-15.3/A-15.5 範圍）；本步＝純資料結構＋測試。
