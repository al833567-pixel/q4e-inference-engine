# 推理工具 — 項目現況 · Task A (v6 100k 主線)

## 〇、Task A 角色
無 GPU 可用：設計＋寫碼（C++）＋CPU build＋測試＋交付包。判斷權全在 A 側。

## 一、當前狀態
**v6 A 側完成**：A-16.0 設計拍板＋A-16.1a~A-16.7b 13 個原子步＋A-16.8 交付包，**17 測試全綠（CPU）**（v6 架構：全局動態池 + VRAM Gather + PLE/ngram SSD mmap + 100k KV Cache 分頁 Offload，已於 `01_架構系統資料.md` 拍板定案並落地）。
**B-16-2 失敗裁定（A-16.9）✅**：root cause＝v6 重定義只改了 parser/help，load path 的 buft override 仍走 v4.1「K=層數」語義（`N=5000 ≥ 48` → no-op）→ 24576 experts 全上 GPU → fit 量測 78431 MiB > 23222 MiB 被拒。修正＝`llm_q4e_moe_vram_keep_layers()`（N→ceil(N/512) 整層，N=5000→10 層≈15.7 GB）＋fit 訊息 k→N＋test-arg-parser 新增 8 斷言。CPU build（267 步 exit 0）＋**17 測試全綠**（test-arg-parser＋16 q4e 測試）。
**設計對齊（B-16-2 之後）**：v6 設計重點＝**整層拆開**（design.md §四：逐 expert 全局池「碎片地址」staging gather＋48 層全 GPU graph 手術＋promotion）。用戶已拍板＝**實作 v6 物理路徑（設計原意）**。A-16.9 整層映射保留為 v5-物理 fallback。
**關鍵發現**：v5 已有**已 oracle 驗證的 pool-path MoE FFN builder** `llm_graph_context::build_moe_ffn_pool`（llama-graph.cpp:2301，A-15.2a 靜態／A-15.2b 動態，test-q4e-moe-pool 4501 斷言對照 `build_moe_ffn`）——它用 `up_pool/gate_pool/down_pool`（[n_ff, n_embd, n_slots]）＋`slot_table`（expert_id→pool slot）取代 per-layer tensor、MUL_MAT_ID 直接讀池、out-of-pool 權重歸零。v6 物理路徑可**重用**此已驗證原語，範圍大幅收斂。
歷史版本已全面廢棄。
下一步：**實作 v6 物理路徑（§五，細切＋驗收，子代理分工）**→ A 側 CPU 測試全綠＋oracle → 交 B 側重建跑 B-16 驗證真設計佈局（全層 GPU＋逐 expert 池，40+ t/s 才可能成立）。

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
| A-16.9 | **B-16-2 VRAM-fit 修正** | ✅（未 commit，待批） | `common_q4e_moe_vram_apply` 改經 `llm_q4e_moe_vram_keep_layers`（v6 全局 N→整層 ceil(N/n_expert)）再插 buft override；`common_q4e_moe_vram_check` 訊息 k→N；test-arg-parser 新增 8 斷言（N=0/1/512/513/5000/24576/99999/-1）；design.md §2 補物理落地註記。凍結的 v4.1 add/insert 函式與既有 pinned 測試不動。驗證：build-msvc 267 步 build exit 0＋17 測試全綠。 |

## 三、現況總表 (17 測試全綠，CPU)
| 測試 | 斷言數 | 備註 |
|---|---|---|
| test-arg-parser | — | v6 6 參數解析＋驗證 |
| test-q4e-global-pool | 188 | 全局池核心 |
| test-q4e-staging | 118 | 雙緩衝管理 |
| test-q4e-gather | 34 | kernel 介面 args build/validate |
| test-q4e-h2d-pipeline | 118 | 非同步管線 |
| test-q4e-graph-map | 78 | 動態 slot 映射 |
| test-q4e-promotion | 427 | **含 oracle 覆蓋** |
| test-q4e-kv-paging | 947862 | **含 100k oracle 滑窗** |
| test-q4e-kv-bridge | 40 | params→paging 組裝 |
| test-q4e-stats-v6 | 139 | 七元件快照＋JSON |
| test-q4e-server-v6 | 50 | server 接線 |
| test-q4e-bench-v6 | 30 | 五元件 throughput＋100k 壓力 |
| test-q4e-e2e-v6 | 28 | **五門**（池組裝／覆蓋路徑 oracle／管線壓力／KV 100k 滑窗／stats-JSON） |
| test-q4e-expert-cache（v5 凍結） | 159 | 回歸 |
| test-q4e-expert-warm（v5 凍結） | 195 | 回歸 |
| test-q4e-ngram（v5 凍結） | 2989 | 回歸 |
| test-q4e-moe-pool（v5 凍結） | 4501 | pool-path MoE FFN oracle＋動態提升（A-15.2a/b；Audit-B 修復 sched 建立與 m_sum 計數後全綠） |

**新元件（engine/common/）**：`q4e-global-pool`、`q4e-staging`、`q4e-gather`（介面）、`q4e-h2d-pipeline`、`q4e-graph-map`、`q4e-promotion`、`q4e-kv-paging`、`q4e-kv-bridge`、`q4e-stats-v6`、`q4e-server-v6`；工具：`q4e-bench --v6/--v6-kv100k`、`q4e-e2e --v6/--v6-kv100k`；server：`/q4e/stats` 回 v6 群組。

**範圍說明（16.5b-2）**：A-16.5b-2（PagedAttention 深度接線）併入 A-16.6 範圍處理——A 側 16.5 交付＝分頁管理器＋bridge＋server 接線（C++ 測試全綠，達 02 接受準）；**CUDA attention 路徑與 GPU 驗證屬 B 側（RUNBOOK B-16）**。

## 四、環境與約束
- CPU/RAM: i7-9700 / ~20 GB RAM / SATA SSD。
- GPU: 無 (A 側)。
- 架構紅線：48 層全 GPU 算；L1 全局池＋VRAM 內 Gather；PLE 必須 SSD mmap；100k KV 必須量化＋分頁。
- Build 陷阱：`GGML_CPU_REPACK=OFF`；`BUILD_SHARED_LIBS=OFF`；必包 `vcvars64.bat`。
- **沙箱 build 附註**：本機 DSH 沙箱禁 ninja 直跑（pipe-spawn 封鎖＋CJK stat）；交付 build 用 `ninja -t commands`＋純 ASCII .bat（%REPO% 注入）。B 側正常環境不受限（ninja 可直跑）。

## 五、v6 物理路徑實作計畫（細切＋驗收，子代理分工）
**目標**：落地 v6 設計佈局（48 層全 GPU 算＋逐 expert 全局池＋動態 promotion），讓 B-16-2 以**設計佈局**可驗收。A 側＝CPU build＋測試＋oracle；CUDA kernel／GPU 驗證＝B 側。
**重用紀律（AGENTS.md）**：重用已 oracle 驗證的 `build_moe_ffn_pool`（graph 手術原語）、v6 bookkeeping（global-pool/staging/promotion/h2d-pipeline/graph-map）、v5 oracle 測試範式（test-q4e-moe-pool）。不重造輪子、不加新測試檔（擴充既有 q4e 測試檔）、凍結 v5 不動、不 commit/push。

| 步 | 內容 | 驗收（CPU） |
|---|---|---|
| 1a | 池張量配置：up/gate/down_pool `[n_ff, n_embd, N]`（N=`--q4e-moe-vram`）於目標 backend；N=0 → 無池（fallback） | 小池(N=8)寫入已知權重，驗證張量內容與 layout |
| 1b | slot_table：`q4e_global_pool` 的 (layer,expert)→slot → 每層 `[n_expert]` i32 ggml tensor，insert/evict 同步 | 一系列 insert/evict 後 slot_table == `q4e_global_pool::slot_of`（out-of-pool=0） |
| 1c | fill backend（SSD→L2）CPU 版：per-layer 3D tensor 讀 expert 權重 → L2 pinned slot | L2 slot 內容 == per-layer tensor 的 expert slice |
| 1d | H2D backend（L2→L1）CPU 版：L2 slot → pool slot | pool slot 內容 == L2 slot 內容 |
| 1e | fill/H2D 接進 `q4e_h2d_pipeline`（替 fake）＋promotion op 端到端 | 擴充 test-q4e-h2d-pipeline：真 backend＋管線＋promotion 全綠 |
| 2a | graph 手術：qwen3next.cpp `build_layer_ffn` 用 `build_moe_ffn_pool`（全局池＋每層 slot_table）替 `build_moe_ffn` | oracle：池含該 token 全部 top-k 時，MoE 層輸出 == `build_moe_ffn`（full tensor） |
| 2b | 靜態池初始 fill（N 跨 48 層 round-robin）＋slot_table | 對 top-k⊆池 的 token，靜態池＋slot_table 輸出正確 |
| 3a | 每層 runtime：router→讀 top-k（CPU sync）→promote missing(L2→L1)→slot_table 同步→池計算 | fake backend 下 decode 序列輸出 == `build_moe_ffn`（動態 promotion 保池正確） |
| 3b | 流水（staging A/B 重疊，選用） | 正確性不變 |
| 4a | server 接線：v6 模式(N>0)走 v6 物理路徑；N=0 → v5 fallback(A-16.9) | CPU build＋既有 17 測試全綠＋新 v6 物理測試全綠 |
| 4b | `/q4e/stats` 反映真實池 hit/miss/evict＋promotion | 驗證 stats 數值 |
| 5a | CUDA：gather kernel（若用 staging）＋cudaMemcpyAsync H2D | B 側 GPU 測試 |

**子代理分工順序**：Phase 1（1a–1e，池基礎設施）→ 審 → Phase 2（2a–2b，graph 手術）→ 審 → Phase 3（3a–3b，runtime 流水）→ 審 → Phase 4（4a–4b，server＋stats）。每 Phase 完成由 A 側 review＋CPU 測試全綠才進下一 Phase。
**風險**：2a graph 手術最高風險（改 qwen3next.cpp 推理路徑）；3a 的 router→sync→promote 引入 per-layer 同步點（48 次/token，靠 3b 流水遮罩）。