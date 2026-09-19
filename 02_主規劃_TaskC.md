# 推理工具 — 主規劃 · Task C 單軌（v6.1 專家級動態快取主線）

> **狀態：規劃完成，待用戶「開始」訊號。未經訊號不得動工（不改碼、不 build、不跑測試、不上 GPU）。**

## 〇、本規劃取代聲明

- **廢除並已刪除**：`02_實作項目工作規劃.md`、`執行順序.md.md`、`RUNBOOK_B_v6.md`、`03A_項目現況_TaskA.md`、`03B_項目現況_TaskB.md`（git 歷史可查，技術事實已收编入本文件）。
- **單一執行者：Task C**。不再有 A/B 分工、不再有「零判斷交接」「待開始訊號」的雙軌流程；唯一的外部門檻＝用戶的「開始」與 checkpoint commit 批准。
- **目標重申（用戶拍板）**：**動態快取的粒度＝每層裡的小專家（routed expert），不是整層快取**。只把**活躍工作集（~6B 量級）**放 GPU；專家位址天然破碎（48 層×512 槽內散布），**非連續問題必須在 VRAM 內以 Gather 解決**。A-16.9 的整層映射（`61bc074`）**降級為緊急備援，非目標路徑**。
- **承接口徑**：A-16.0~A-16.9 已交付（16 commits `f5b15e9`→`61bc074`，17 測試 CPU 綠）＝元件庫與參數語義資產；物理路徑 P1-P3 代碼在工作區未 commit（checkpoint `16530f3`）；moe-pool v5-dyn 不穩定 crash＝唯一技術遺留炸彈（見 C1）。

## 一、不變的目標語義（自 01 架構文件承接）

| 項 | 值 |
|---|---|
| 模型 | Qwen3.8-Flash-Next GGUF 4 片 103.7 GiB（`D:\AI_Model\Qwen3.8-Flash-Next\`）；MoE 48 層，每層 512 routed（≈3.07 MB/個）＋1 shared，router 選 10 |
| 專家級動態池 | L1（VRAM 全局池）N 槽跨 48 層共享、Layer-Aware LRU（floor F=10）＋L2（Pinned RAM N2）＋L3（SSD） |
| **破碎執行** | 選中專家散布於池槽 → **VRAM 內 Gather**：staging 雙緩衝（A/B）連續化 → `MUL_MAT_ID` over staging → slot_table 動態映射 → promotion op 回填 |
| **活躍集預算** | 「活躍 6B」解讀：每 token 活躍 routed ≈48×10×~5.1M≈2.4B 參數（＋shared/attn 量級湊數 B）；對應 L1 池權重 **N≈1200（≈3.7 GiB，=6B 參數口徑）~2000（≈6.1 GiB，=6 GB 權重口徑）**。**起始配置 N=2000，C5 真機以 VRAM-fit 門＋t/s 實測收斂定案** |
| 100k KV | PagedAttention 分頁：L1 active window 8k＋L2 pinned 6 GB＋L3 pagefile；強制 q8_0（≈35 GB→分頁 Offload） |
| PLE/ngram | 28.8 GB 恆 SSD mmap（紅線，不佔 RAM/VRAM） |
| 對外 | OpenAI 相容 `/v1`＋API key＋`/q4e/stats`（真實池統計） |
| 部署 | 單一 `q4e-server.exe`；i7-9700 / RTX 3090 24 GB / ~20 GB RAM / SATA SSD |

## 二、Phase C0 — 現場固化（動工第一步）

| 步 | 內容 | 門 |
|---|---|---|
| C0-1 | 還原工作區 DBG 插樁：`llama-graph.cpp`（17 行 DBG）、`q4e-expert-warm.cpp`（3 行 DBG COPY）、`test-q4e-moe-pool.cpp` 的插樁部分；P0-2 的 CLI filter 與 P0-5 env 門控**擇優保留**（有用則整併成乾淨 patch） | `git diff` 只剩被整併的工具改動 |
| C0-2 | checkpoint commit（**需用戶批准**）：物理路徑 P1-P3 代碼＋保留的測試工具＋證據歸檔聲明（`v6_work/p0`、`b16_2_server.err.log`、`15_results.txt` 保留不刪） | 樹乾淨、HEAD 可追 |
| C0-3 | 基線重現：CPU build（既有配方）→ 17 既有測試全綠重現一次 | 17/17 綠 |

## 三、Phase C1 — 崩潰解卡（換工具、設預算、隔離主線）

**證據回顧**：三種 corruption 類 exit（`0xC0000005` AV／`0xC0000409` /GS 堆疊 overrun／`0xC0000374` heap corruption）；**6 個 v5-dyn 測試各自單獨 ×15 全乾淨，僅全套依序執行才崩**；崩潰點散彈（`build_lora_mm(gate_inp)`、pool PTR 後、`build_moe_ffn` 期間）；20 輪 fprintf 插樁未收斂。**結論：heap 腐敗延遲爆發，崩潰點≠下毒點。**

| 步 | 內容 | 門 |
|---|---|---|
| C1-1 | **停用 fprintf 人海**。改用 PageHeap：`gflags /p /enable test-q4e-moe-pool.exe /full`（零改碼，腐敗當下即 AV）＋ VS debugger/WinDbg 抓 first-chance AV stack | 拿到確定 stack（function＋module） |
| C1-2 | 依 stack 定案 root cause；若 PageHeap 無效，升級 MSVC `/fsanitize=address` 或 CRT debug heap 二選一 | 最小 repro（哪個測試序列＋多少輪）必崩 |
| C1-3 | **測試二進制隔離（結構性解卡，無論 C1-2 結果）**：v6 物理路徑測試拆出獨立 binary，主線不再被 v5-dyn 套件綁死 | v6 測試 binary 全綠 ×3 連跑 |
| C1-4 | v5-dyn crash 修復（限時預算＝1 個工作 session）：凍結禁令解除（單軌後整棵樹都歸 Task C 管），**但修復仍須 stack 指證，禁止盲修**；審計 v6 共用的 copy 原語（`q4e_pool_promote_into`）是否同源 | moe-pool v5+ v6 全套 15×3 乾淨；或正式裁決「v5-dyn 隔離＋known-issue 記錄」放行主線 |

## 四、Phase C2 — 物理路徑驗收（承接原 P1-P3）

| 步 | 門 |
|---|---|
| Phase 2/3（池 fill＋graph 手術＋runtime 動態 promotion＋流水 max_inflight=2）在新隔離 binary 重驗 | 31686 checks 全綠（×3）＋每步 `pool==full==oracle` |
| stress：全套 ×15 | bad=0 ×3 |

## 五、Phase C3 — server 接線（承接原 P4，語義反轉）

| 步 | 內容 | 門 |
|---|---|---|
| C3-1 | `common_q4e_moe_vram_apply` 重接：`--q4e-moe-vram N>0` → **`q4e_v6_pool_slots=N`（專家級真池為主路徑）**；A-16.9 整層路徑降為顯式 fallback（僅當真池路徑被裁決不可用時啟用，啟用與否由 C5 後覆盤決定，不悄悄生效） | test-arg-parser：N=2000→slots==2000；N=0→0；既有斷言綠 |
| C3-2 | v6 開（slots>0）時不插整層 buft override；qwen3next 守衛（`q4e_v6_pool_slots>0`）走 Phase 2 池 path | moe-pool v6 oracle 測試綠 |
| C3-3 | `/q4e/stats` fake counters → 真實 `q4e_global_pool_stats`＋promotion counters（l2_fills／l2_to_l1／l1_evictions）＋KV 分頁統計 | 注入已知 counters＝JSON 值；16/16＋e2e/bench v6 實跑全綠 |

## 六、Phase C4 — CUDA build＋Gather kernel 實作（收編原 B-16-1）

| 步 | 內容 | 門 |
|---|---|---|
| C4-1 | gather kernel CUDA 實作（staging 版）：非連續池槽 → 連續 staging；`cudaMemcpyAsync` H2D 管線接真 GPU | 單元＋oracle 對 CPU 版 |
| C4-2 | CUDA build（已證配方，見 §九）：含 A-16.9 之後全部 HEAD | server.exe 冒煙 `--version` commit==HEAD |
| C4-3 | PagedAttention 深度接線（原 16.5b-2，屬此處）：KV 量化分頁走真 GPU attention 路徑 | KV 分頁統計真實、100k 滑窗 oracle 綠 |

## 七、Phase C5 — 真機驗收（收編原 B-16 全部門，Task C 親自跑）

**前置**：VRAM=0（**9/19 實測已空 0/24576 MiB**，需再驗）；無殘留 server；GGUF 4 片可讀。

| # | 門 | 準 |
|---|---|---|
| 1 | VRAM-fit：`--q4e-moe-vram 2000` 載入成功，VRAM ≤ 24 GB | 不觸發 fit 拒載；超出→降 N 收斂，紅線不動 |
| 2 | 100k Context：`--ctx-size 100000 --cache-type-k q8_0` 載入不 OOM，KV 分頁運作 | 分頁統計非零、無 OOM |
| 3 | Oracle 對答案：48 層全 GPU（attn/router/shared）＋專家池＋Gather 逐層輸出＝oracle | max_err=0（池內）／out_of_pool 誤差符合設計 |
| 4 | 效能：ON 暖 t/s ≥ 8（理想 L1 全命中 ≥ 40） | `/q4e/stats` 命中分佈佐證 |
| 5 | 回歸門：e2e 5 閘綠＋冷載入 ≤~250 s＋收尾 VRAM 歸零 | 全綠 |
| 6 | 活躍集收斂：N=1200/2000 兩檔 t/s＋命中分佈對比，定案生產 N | 數據入本文件 §十一 |

失敗＝錯誤原文＋證據歸檔回報用戶，不猜、不調參越權。

## 八、Phase C6 — 交付收尾

README/01 對齊實況（含本文件狀態更新）；`git tag v6.1-<date>`（需批准）；殘留清理（`_testtmp`、根目錄散落 log 歸檔至 `v6_work/`）。

## 九、已證環境配方（單軌後自用，全部實證過）

- **Build 紀律**：必包 `vcvars64.bat`；必 `-DBUILD_SHARED_LIBS=OFF -DGGML_CPU_REPACK=OFF`；`-DGGML_CUDA=OFF`（CPU 側）。
- **CJK 樹陷阱**：ninja 作為 executor 直跑此樹必敗（MSYS2 ninja ANSI 重編碼 `FindFirstFileExA` 錯；native MSVC ninja `mkdir Illegal byte sequence`）。唯一配方＝`ninja -t commands` 純文字提取 → 純 ASCII `.bat`（CJK 前綴以 `%REPO%` env 注入）→ `cmd /c` 背景 job；detached 樹會被沙箱收走，禁 `Start-Process -Wait`。
- **環境三修**：(a) `CCACHE_DIR` 必須在 D:（C: AppData 路徑 ccache 靜默掛死，暖備份在 `v6_work/b16/diag/ccache_copy`，7626 項）；(b) harness pwsh 無 INCLUDE/LIB → `.bat` 內 `call vcvars64.bat`；(c) PATH 前置 `v6_work/b16/bin`（native gzip shim，cygwin 系 gzip 沙箱下必死）。
- **沙箱附註**：DSH 沙箱禁 pipe-spawn 子程捕獲（EPERM）；讀寫限 workspace；checkpoint commit 需用戶批准。

## 十、紅線（單軌版）

1. **架構**：48 層 attn/router/shared 恆全 GPU；routed 專家＝專家級動態池＋VRAM 內 Gather（**破碎執行是產品語義，整層快取非目標**）；PLE/ngram 恆 SSD mmap；100k KV 恆量化＋分頁 Offload。
2. **GPU**：永不過兩個模型實例；絕不殺 harness 之服務程序；驗收前必驗 VRAM=0。
3. **證據**：任何「全綠」以全套×3 連跑為準；crash 修復必須 stack 指證；失敗回報原文。
4. **狀態**：以本文件為唯一進度正典；commit checkpoint 需用戶批准。

## 十一、風險與決策點

| # | 風險 | 對策 |
|---|---|---|
| 1 | v5-dyn crash 的下毒點若在 v6 共用原語（`q4e_pool_promote_into` 等）→ v6 物理路徑同病 | C1-1 的 PageHeap stack 是定案證據；C1-4 修復後 v6 oracle 重驗 |
| 2 | 真機 t/s 不達 8 | 槓桿＝L2 PCIe 補齊率：加大 staging、ngram 預取命中率、N2 檔位；紅線不動 |
| 3 | N=2000 VRAM-fit 爆 | 降 N 收斂（活躍集語義不變）；KV window 8k 可縮至 4k 換預算 |
| 4 | 「6B」口徑歧義（參數 vs 權重 GiB） | §一雙口徑並陳，C5-6 真機數據定案 |

## 十二、進度狀態（本文件維護）

| Phase | 狀態 |
|---|---|
| C0 現場固化 | ✅ DBG 還原（llama-graph/q4e-expert-warm 歸零）；保留 P0-2 CLI filter＋setvbuf 無緩衝 stderr；基線 16/16 全綠（moe-pool v6 區段仍 `#if 0`，全序不穩定屬 C1）；規劃重整 commit `172f2fc` |
| C1 崩潰解卡 | ✅ root cause＝測試端 dyn_round 在 res.reset() 後重用 gctx，快取 ctx0/gf 懸掛（UAF，同址重用掩護故 8/15 隨機崩）；MSVC ASan 全閉包重編抓到 stack 定案；修法＝reset 後重同步 ctx0/gf（零斷言變更）；v5×15＋full×15 bad=0、ASan×3 綠；commit `f2d7bee` |
| C2 物理路徑驗收 | ✅ 31686 checks ×15 全綠（超出 ×3 門）＋stress 15/15；即 C1 修後 full 序列 |
| C3 server 接線 | 🟡 C3-1/2 完成：`--q4e-moe-vram N>0` → `q4e_v6_pool_slots=N`（真池主路徑，整層 fallback 改 `Q4E_MOE_VRAM_FALLBACK=1` env 門控）；16/16＋moe-pool 31686×3 綠。C3-3 stats 活體接線併入 C4-0 runtime wiring |
| C4 CUDA＋Gather kernel | ⏳（含 C4-0：decode 迴圈 runtime promotion 接線——src/ 尚未接 sync_round/池 backend，為真池生效前最大缺口） |
| C5 真機驗收 | ⏳（GPU 由用戶暫用中，待通知釋放） |
| C6 交付收尾 | ⏳ |
