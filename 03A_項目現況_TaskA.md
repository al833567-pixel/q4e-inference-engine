# 推理工具 — 項目現況 · Task A (v6 100k 主線)

## 〇、Task A 角色
無 GPU 可用：設計＋寫碼（C++）＋CPU build＋測試＋交付包。判斷權全在 A 側。

## 一、當前狀態（9/17 覆核修正）
**⚠️ 卡住點（阻擋 Phase 4）：test-q4e-moe-pool 不穩定崩潰 — 15 連跑 8 次 AV（0xC0000005），是 crash 不是 hang**
- 9/16 22:58 Phase 3 驗收：16/16 全綠（moe-pool 31686 checks，單次執行）。9/17 10:10 首次 stress：5 bad＋1 ok（`v6_work/repro_err_*`／`repro_ok`）。
- 9/17 bisect（`v6_work/p3bisect/`）：v6 測試以 `#if 0` 禁用（只跑 v5，4501 checks）→ **仍 8/15 bad**；多執行緒與單執行緒（`Q4E_TEST_SINGLE_THREAD`）皆 8/15；bad 全部 exit=**-1073741819（0xC0000005 access violation）**＝**崩潰，非 hang**（stress 無 timeout 包，loop 能跑完 15 次即證明是 crash）。
- 崩潰點（DBG 標記）：`test_dynamic_oracle_sequence` round 1 的 sched reset 之後、graph build（`build_moe_ffn`）期間／`test_dynamic_multilayer` 入口後（dyn_env 析構後）；輸出中斷、無 FAILED 訊息。
- **程式碼歸屬（已核實）**：bisect 建置跑到的程式碼，工作區 diff 全為「純新增＋更名」（test 檔 v5 區段僅插入 hunk；llama-graph／models／qwen3next 純新增；q4e-h2d-pipeline 純 rename）→ 崩潰是**凍結 v5 潛在 bug（`common/q4e-expert-warm.h`：q4e_expert_h2d／q4e_pool_dynamic）或 v5 測試自身 UB**，9/17 首次 15 連跑才浮現（v6_work 無更早 stress 證據），**非 Phase 2/3 回歸**。
- **需 A 側裁決**：「凍結 v5 不動」紀律與修 bug 衝突 → 需明確授權「可動 `q4e-expert-warm.h`（v5 凍結檔）僅為修此 crash：1 檔 1 bug，不弱化任何斷言」。
**工作區狀態（未 commit；最後 commit `0a25cec`＝計畫文件）**：
- `tests/test-q4e-moe-pool.cpp`：v6 測試在 2 個 `#if 0` 區塊（L779-1132 Phase 2；L1134-1405 Phase 3，共 27185 斷言目前被關）；v6 include 被註解（L12）；16 支 DBG fprintf 殘留（L495-737）；`Q4E_TEST_SINGLE_THREAD` 開關（L492-501）；main() 4 個 v6 呼叫被註解（L1414-1417）。
- `common/q4e-pool-backend.{h,cpp}`：新檔（untracked），Phase 3a pool backend（fill/H2D CPU 版＋sync_round）；bisect 建置未 link。
- src Phase 2/3 代码在工作區：llama-graph（＋`q4e_v6_pool_fill`/`q4e_v6_pool_state`/`q4e_v6_slot_map`）、qwen3next（L135 守衛讀 `q4e_v6_pool_slots`，池放 CPU buft）、llama.h L330（`q4e_v6_pool_slots` 欄位）、models.h。
- Phase 4（4a/4b）未動：`common/common.cpp` L1293-1389 仍只有 A-16.9 v5 路徑；`/q4e/stats` 仍為既有 v6 群組（fake counters）。
- `diag/gguf_*.py`（9/17）：Qwen3.8-Flash-Next gguf 結構檢查（B-16 準備），與卡住點無關。
**v6 物理路徑 Phase 1-3 代码完成（待解卡後重新驗收）**：Phase 3＝runtime 動態 promotion（`external_l1`/`attach_l1`＋`graph_slot`＋`sync_round`）＋3b 流水併入（max_inflight=2）＋`test_v6_runtime_decode_sequence`（3 步 decode×2 層，每步 pool==full==oracle，16 promotions：15 fills＋1 L2 hit，10 evictions）；Phase 2＝`build_layer_ffn` 手術（守衛 `q4e_v6_pool_slots>0`）＋靜態池 fill＋每層 slot_table。31686 checks 曾一次全綠（9/16）。
**v6 A 側交付（A-16.0~A-16.9）不變**：A-16.0 拍板＋13 個原子步＋交付包，17 測試全綠（CPU）；B-16-2 VRAM-fit 裁定（A-16.9）✅ **已 commit `61bc074`**（現況舊記「未 commit」已過期）；設計對齊＝實作 v6 物理路徑（設計原意），A-16.9 整層映射保留為 v5-物理 fallback；關鍵發現＝重用 `build_moe_ffn_pool`（已 oracle 驗證原語）。
歷史版本已全面廢棄。
**下一步：Phase 0（P0-1→P0-8）解卡不穩定崩潰 → Phase 3 重新驗收 → Phase 4（P4-1→P4-6）→ 交 B 側 B-16。**

## 二、v6 交付項目 (全完成)
| 項 | 名稱 | 狀態 | 內容／驗收／commit |
|---|---|---|---|
| A-16.0 | v6 設計拍板＋參數語義 | ✅ `f5b15e9` | 重定義 `--q4e-moe-vram N`；新增 `--q4e-staging-mb`、`--q4e-layer-floor`、`--q4e-ple-mmap`、`--cache-type-k q8_0`、`--q4e-kv-ram`。design.md＋test-arg-parser 全綠。 |
| A-16.1 | 全局動態池管理器 | ✅ `d467f05`/`91abfb0` | Global LRU＋Layer-Aware Floor (防層間擠壓)。188 斷言＋接入 common 庫。 |
| A-16.2 | VRAM 內 Gather 設計 | ✅ `2e597b5`/`ecfb070` | Staging Buffer 雙緩衝 (A/B) 管理邏輯 (118)＋CUDA Kernel 介面定義 (34)。 |
| A-16.3 | H2D 提升管線 | ✅ `7de45e0` | L2 Pinned RAM＋SSD→L2 fill＋非同步 H2D (118，非同步 job queue＋L2 LRU 保護)。 |
| A-16.4 | Graph 手術 (最高風險) | ✅ `5f19be6`/`d75f4c4` | 48 層全 GPU＋動態 slot 映射 (78)＋promotion op (427，含 oracle 覆蓋)。 |
| A-16.5 | **100k KV Cache 分頁 Offload** | ✅ `6207f7c`/`abc7afc` | 分頁管理器 (947862，含 100k oracle 滑窗)＋KV bridge (40)。**16.5b-2 深度接線併入 A-16.6**（CUDA 路徑屬 B-16）。 |
| A-16.6 | Server 接線＋stats | ✅ `a891982`/`9445169` | Expert L1/L2/L3 命中＋KV 分頁統計 (139/50)，`/q4e/stats` v6 群組上線。 |
| A-16.7 | e2e/bench v6 | ✅ `c82fbc1`/`9ccded1` | 客戶端工具更新（bench 30＋e2e 五門 28，含 `--v6-kv100k` 100k 壓力）。 |
| A-16.8 | 交付包 v6 | ✅ `f4aaecd`/`321c1f8` | 8a 5 文件對齊（README/01/02/03A/03B）＋8b RUNBOOK_B_v6 編號對齊＋最終清理驗證。 |
| A-16.9 | **B-16-2 VRAM-fit 修正** | ✅ `61bc074` | `common_q4e_moe_vram_apply` 改經 `llm_q4e_moe_vram_keep_layers`（v6 全局 N→整層 ceil(N/n_expert)）再插 buft override；`common_q4e_moe_vram_check` 訊息 k→N；test-arg-parser 新增 8 斷言（N=0/1/512/513/5000/24576/99999/-1）；design.md §2 補物理落地註記。凍結的 v4.1 add/insert 函式與既有 pinned 測試不動。驗證：build-msvc 267 步 build exit 0＋17 測試全綠。 |

## 三、現況總表 (17 測試全綠，CPU)
| 測試 | 斷言數 | 備註 |
|---|---|---|
| test-arg-parser | — | v6 6 參數解析＋驗證 |
| test-q4e-global-pool | 188 | 全局池核心 |
| test-q4e-staging | 118 | 雙緩衝管理 |
| test-q4e-gather | 34 | kernel 介面 args build/validate |
| test-q4e-h2d-pipeline | 5690 | 非同步管線 |
| test-q4e-graph-map | 78 | 動態 slot 映射 |
| test-q4e-promotion | 427 | **含 oracle 覆蓋** |
| test-q4e-kv-paging | 947862 | **含 100k oracle 滑窗** |
| test-q4e-kv-bridge | 40 | params→paging 組裝 |
| test-q4e-stats-v6 | 139 | 七元件快照＋JSON |
| test-q4e-server-v6 | 50 | server 接線 |
| test-q4e-bench-v6 | 30 | 五元件 throughput＋100k 壓力 |
| test-q4e-e2e-v6 | 28 | **五門**（池組裝／覆蓋路徑 oracle／管線壓力／KV 100k 滑窗／stats-JSON） |
| test-q4e-expert-cache（v5 凍結） | 159 | 回歸 |
| test-q4e-expert-warm（v5 凍結） | 195 | 回歸（`q4e_expert_h2d`/`q4e_pool_dynamic` 定義處＝卡住點嫌疑檔） |
| test-q4e-ngram（v5 凍結） | 2989 | 回歸 |
| test-q4e-moe-pool（v5 凍結＋v6 P2/P3） | 31686 | **v6 部分目前被 `#if 0` 關（bisect 中，只跑 4501 v5）**；v5 dyn 區段不穩定 crash 8/15（0xC0000005）待 Phase 0 解卡 |

**新元件（engine/common/）**：`q4e-global-pool`、`q4e-staging`、`q4e-gather`（介面）、`q4e-h2d-pipeline`、`q4e-graph-map`、`q4e-promotion`、`q4e-kv-paging`、`q4e-kv-bridge`、`q4e-stats-v6`、`q4e-server-v6`、`q4e-pool-backend`（新，Phase 3a，未 commit）；工具：`q4e-bench --v6/--v6-kv100k`、`q4e-e2e --v6/--v6-kv100k`；server：`/q4e/stats` 回 v6 群組。

**範圍說明（16.5b-2）**：A-16.5b-2（PagedAttention 深度接線）併入 A-16.6 範圍處理——A 側 16.5 交付＝分頁管理器＋bridge＋server 接線（C++ 測試全綠，達 02 接受準）；**CUDA attention 路徑與 GPU 驗證屬 B 側（RUNBOOK B-16）**。

## 四、環境與約束
- CPU/RAM: i7-9700 / ~20 GB RAM / SATA SSD。
- GPU: 無 (A 側)。
- 架構紅線：48 層全 GPU 算；L1 全局池＋VRAM 內 Gather；PLE 必須 SSD mmap；100k KV 必須量化＋分頁。
- Build 陷阱：`GGML_CPU_REPACK=OFF`；`BUILD_SHARED_LIBS=OFF`；必包 `vcvars64.bat`。
- **沙箱 build 附註**：本機 DSH 沙箱禁 ninja 直跑（pipe-spawn 封鎖＋CJK stat）；交付 build 用 `ninja -t commands`＋純 ASCII .bat（%REPO% 注入）。B 側正常環境不受限（ninja 可直跑）。
- **Stress 工具**：`v6_work/p3bisect/stress.bat`（15 連跑＋DONE_BAD 統計）、`run_bisect.bat`（單檔重編 bisect 建置）。

## 五、執行計畫 v2（9/17 重新分配：更多步驟、極簡化、一步一門）
**原則（v2 紀律，比 v1 更嚴）**
1. 嚴格串列；一步＝至多 1 個檔案（同檔分步）。每步只做一件事。
2. 每步**恰好 1 個驗收門**；門不過 → 回退該檔（`git checkout -- <file>`）→ 帶證據報 A 側裁決；**不準自行換第二個檔試**。
3. 不弱化 v5 oracle 斷言（4501 門）；不加新測試檔；不 commit/push（checkpoint commit 需用戶批准）。
4. P0-7 完成前 moe-pool 只跑 v5 部分——任何「全綠」都不算完整驗收。

**Phase 0 — 解卡不穩定崩潰（當前卡住點，阻擋一切）**
| 步 | 內容（touch） | 驗收門 |
|---|---|---|
| P0-1 | 無程式碼（證據歸檔）：核對 3 個 stress 目錄（`stress/`、`stress1t/`、`stress1t_true/`）全部 bad run 的 exit code | bad 全為 0xC0000005；有異樣 exit 分開記錄入 03A |
| P0-2 | test-q4e-moe-pool.cpp（只改 main()）：加 CLI filter `exe [test-name]` 只跑指定測試，預設全跑；不改任何測試邏輯 | full 跑＝4501 checks（bisect 建置）；單測試可跑 |
| P0-3 | 無程式碼：用 P0-2 filter 逐測試 stress×15（slot_of／all_in_pool／out_of_pool／mixed／oracle_sequence／multilayer） | 圈出 crashing 測試＋各 bad rate（預期 oracle_sequence／multilayer） |
| P0-4 | 無程式碼（crash stack 捕捉）：VS debugger 或 windbg 跑 crashing 測試，抓第一次崩潰 stack（function＋module） | 拿到 stack；指名模組（ggml／llama-common／test 本體） |
| P0-5 | test-q4e-moe-pool.cpp（bisect 輔助，P0-7 必還原）：以 env 門控 probe round（r==1）與 rounds 數（尾端遞減）＋跳過尾部固定 counters CHECK，逐組 stress | 最小失敗序列：N rounds bad≥5/15 且 N-1 rounds bad=0/15 |
| P0-6 | **1 檔 1 bug（先經 A 側授權動 v5 凍結檔）**：依 P0-4/P0-5 證據修 root cause（候選：`common/q4e-expert-warm.h`、v5 測試本體、ggml——擇一，禁止盲修） | 最小 repro 15/15 clean；moe-pool 全量 3 連跑全綠 |
| P0-7 | test-q4e-moe-pool.cpp（只清理）：刪全部 16 支 DBG；還原 2 個 `#if 0`（v6 測試恢復）；還原 L12 include；P0-5 的 env 門控刪除；保留 Q4E_TEST_SINGLE_THREAD 開關（無害） | `all 31686 checks passed` ×3 連跑 |
| P0-8 | 無程式碼（驗收門）：full binary（v6 啟用）stress×15＋16 測試全套 | DONE_BAD=0＋16/16 全綠 → **Phase 0 完成；commit checkpoint（需批准）** |

**Phase 4 — server 接線＋stats（重切微步）**
| 步 | 內容（touch） | 驗收門 |
|---|---|---|
| P4-1 | common/common.cpp（`q4e_moe_vram_apply` 區段，~L1293-1389）：`--q4e-moe-vram N>0` → `mparams.q4e_v6_pool_slots=N`（欄位已存在，llama.h L330）；N=0 → 0 | test-arg-parser：N=5000→欄位==5000；N=0→0；既有 8 斷言仍綠 |
| P4-2 | common/common.cpp（同區段）：v6 開（slots>0）時不插 A-16.9 v5 buft override（N=0 fallback 原樣） | N>0→無 buft override；N=0→override 如前（既有斷言綠） |
| P4-3 | 無程式碼（驗證）：N>0 時模型走 Phase 2 pool path（qwen3next L135 已讀欄位，確認無其他阻擋） | moe-pool v6 oracle 測試綠 |
| P4-4 | common/q4e-server-v6.cpp（1 檔）：`/q4e/stats` v6 群組 fake counters → 真實 `q4e_global_pool_stats`＋pipeline counters | test-q4e-stats-v6 綠＋新斷言：注入已知 counters＝JSON 值 |
| P4-5 | common/q4e-server-v6.cpp（1 檔）：同 JSON 群組補 promotion counters（l2_fills／l2_to_l1／l1_evictions） | 斷言：3 欄位＝注入值 |
| P4-6 | 無程式碼（驗收門）：16/16＋moe-pool stress×15＋q4e-e2e/bench v6 實跑 | 全綠 → **Phase 4 完成；commit checkpoint（需批准）** |

**Phase 5 — B 側（不變）**
| 步 | 內容 | 驗收 |
|---|---|---|
| P5-1 | CUDA：gather kernel（若用 staging）＋cudaMemcpyAsync H2D＋B-16 GPU 驗證 | B 側 GPU 測試（RUNBOOK_B_v6） |

**風險（v2）**
1. P0-6 root cause 未定（凍結 v5 潛在 bug 或 v5 測試 UB）——P0-4 的 crash stack 是定案證據；無 stack 前禁止改動任何 C++。
2. 需動 v5 凍結檔修 bug → 須先經 A 側授權（見第一節裁決項）。
3. 工作區 v6 測試目前 `#if 0` 關閉 → P0-7 前 moe-pool「全綠」不可當驗收。

**子代理分工順序（v2）**：Phase 0（P0-1→P0-8，串列，每步一個子代理或同一人）→ A 側 review → Phase 4（P4-1→P4-6）→ A 側 review → 交 B 側 P5-1。

## 六、v1 計畫（已被 v2 取代，留檔）
目標：落地 v6 設計佈局（48 層全 GPU 算＋逐 expert 全局池＋動態 promotion）。重用紀律（AGENTS.md）：重用 `build_moe_ffn_pool`、v6 bookkeeping、v5 oracle 測試範式；不重造輪子、不加新測試檔、凍結 v5、不 commit/push。
| 步 | 內容 | 驗收（CPU） |
|---|---|---|
| 1a | 池張量配置：up/gate/down_pool `[n_ff, n_embd, N]` | 小池(N=8)寫入已知權重，驗證 layout |
| 1b | slot_table：(layer,expert)→slot | insert/evict 後 slot_table == `q4e_global_pool::slot_of` |
| 1c | fill backend（SSD→L2）CPU 版 | L2 slot 內容 == per-layer tensor expert slice |
| 1d | H2D backend（L2→L1）CPU 版 | pool slot 內容 == L2 slot 內容 |
| 1e | fill/H2D 接進 `q4e_h2d_pipeline`＋promotion op | test-q4e-h2d-pipeline 真 backend 全綠 |
| 2a | ✅ graph 手術：`build_layer_ffn` 用 `build_moe_ffn_pool` | ✅ oracle：池含 top-k 時 pool==full |
| 2b | ✅ 靜態池初始 fill＋slot_table | ✅ test_v6_static_fill＋test_v6_global_pool_oracle |
| 3a | ✅ 每層 runtime：router→top-k→promote→slot_table→池計算 | ✅ test_v6_runtime_decode_sequence（16 promotions） |
| 3b | ✅ 流水（併入 3a） | ✅ max_inflight=2＋L2 LRU 重疊 |
| 4a | server 接線（→ 拆為 P4-1~P4-3） | — |
| 4b | `/q4e/stats` 真實池統計（→ 拆為 P4-4/P4-5） | — |
| 5a | CUDA（→ P5-1） | B 側 |
v1 風險：2a graph 手術最高風險；3a per-layer 同步點（48 次/token）靠 3b 流水遮罩。
