# 推理工具 — 主規劃 · Task C 單軌（v6.1 專家級動態快取主線）

> **v7 註記（2026-10-22）**：本文件描述 v6。目前主線為 `engine-v7/`（llama.cpp b11796 + q4e），架構、實測與限制見 [REPORT_v7.md](REPORT_v7.md)。

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
| **混合雙路徑（9/19 用戶拍板）** | **命中→GPU 池算；未命中→CPU 用原始 512 專家算（歸零語義廢除）**。`build_moe_ffn_pool` 同圖雙路：GPU 路＝現有 mm_id over 池槽；CPU 路＝mm_id over 原層專家張量（CPU mmap），`cpu_ids=selected×(1-in_pool)`、`w_cpu=w×(1-in_pool)`，兩路相加＝完整輸出。池純粹是加速器，正確性不再依賴命中率 |
| **破碎執行** | 選中專家散布於池槽 → **VRAM 內 Gather**：staging 雙緩衝（A/B）連續化 → `MUL_MAT_ID` over staging → slot_table 動態映射 → promotion op 回填 |
| **活躍集預算（9/19 用戶拍板）** | **池預算＝12B（6B 的兩倍）→ 預設 N=4000 槽 ≈ 12.3 GiB**（≈24B 權重參數量級）；`--q4e-moe-vram` 可設。VRAM 總帳：池 12.3＋attn/router/shared/KV/compute ≈ 14 GB ≤ 24 GB 紅線內 |
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
| C4-1 | 池 buffer 上運算裝置（select_buft）＋註冊進 model（fit 自動計帳）＋fill backend-aware | 31686 ×3 綠 |
| C4-2 | runtime promotion 接進 decode（observe→LRU→填槽→改表；graph output 讀回選擇） | 全套綠＋promo 測試 43 斷言 |
| C4-3 | 專家張量 CPU override（池路徑不把 77GB 推上 GPU） | VRAM-fit N=2000 過＋真機 decode 通 |
| **C4-4** | **混合雙路徑 MoE（9/19 用戶拍板）**：`build_moe_ffn_pool` 加 CPU 路（原層專家張量 mm_id，`cpu_ids=selected×(1-in_pool)`、`w_cpu=w×(1-in_pool)`，兩路相加）；歸零語義廢除 | 雙路徑 oracle：混合命中/未命中＝完整答案（現有 out_of_pool 測試語義改為全 oracle）＋全套綠 |
| **C4-5** | **池預設 12B（N=4000 ≈ 12.3 GiB）**，`--q4e-moe-vram` 可設；README/01/arg-parser 預設值同步 | 預設斷言 4000；VRAM 總帳 ≤ 24 GB |
| C4-6 | CUDA 重編＋真機重驗：正確性（不再歸零）＋t/s＋prefill 時間 | 見 C5 門 |

## 七、Phase C5 — 真機驗收（收編原 B-16 全部門，Task C 親自跑）

**前置**：VRAM=0；無殘留 server；GGUF 4 片可讀。

| # | 門 | 準 |
|---|---|---|
| 1 | VRAM-fit：`--q4e-moe-vram 4000` 載入成功，VRAM ≤ 24 GB | 不觸發 fit 拒載；超出→降 N 收斂，紅線不動 |
| 2 | 正確性：混合命中/未命中下輸出與 CPU 全量路徑一致（抽樣比對） | 雙路徑使歸零消失；短文生成語義合理 |
| 3 | 效能：ON 暖 t/s ≥ 8；prefill 不再 SSD 填池風暴（61 token 從 141 s 量級降到 ~10 s 內） | `/q4e/stats` 命中分佈佐證（C3-3 完成後） |
| 4 | 100k Context：`--ctx-size 100000 --cache-type-k q8_0` 載入不 OOM，KV 分頁運作 | 分頁統計非零、無 OOM |
| 5 | 回歸門：e2e 閘綠＋收尾 VRAM 歸零 | 全綠 |
| 6 | 活躍集收斂：N=2000/4000 兩檔 t/s＋命中分佈對比，定案生產 N | 數據入本文件 §十一 |

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

## 十二、進度狀態（本文件維護；9/20 用戶拍板「混合雙路徑＋池 12B」後更新）

| Phase | 狀態 |
|---|---|
| C0 現場固化 | ✅ DBG 還原；基線 16/16 綠；commit `172f2fc` |
| C1 崩潰解卡 | ✅ root cause＝測試端 gctx 快取 ctx0/gf 懸掛（UAF）；ASan 定案；v5×15＋full×15 bad=0；commit `f2d7bee` |
| C2 物理路徑驗收 | ✅ 31686 checks ×15 全綠＋stress 15/15 |
| C3 server 接線 | ✅ 真池主路徑＋CPU override（`9b9b30c`/`e4cb1c8`）；C3-3 stats 活體接線移入剩餘工作 |
| C4 CUDA＋Gather | 🟡 大部分完成（見 §十三 剩餘工作）；混合雙路徑＋池 12B 預設（`862258e`）；**池整合已移植到生產 arch `qwen4exp`＋池配置延後到權重 buffer 之後＋alloc 失敗自動退回 host 池**（`546189d`）；CPU 全套綠＋moe-pool 32241 綠 |
| C5 真機驗收 | 🔴 待 CUDA 重編後重驗（歷史教訓：qwen3next 上的「成功」其實池未建立） |
| C6 交付收尾 | ⏳ 全 pass 後：清暫存＋舊 llama.cpp 對比數據報告＋打包 |

## 十三、剩餘工作清單（9/20 盤點）

**主線（必做，完成即打包）：**

| # | 工作 | 門 |
|---|---|---|
| R1 | CUDA 重編（含 `546189d`）＋冒煙 commit==HEAD | build OK、`--version` 正確 |
| R2 | 真機載入：**驗證池真的建在 CUDA0**（log 出現 `q4e v6 pool: ... total=11718.8 MiB buft=CUDA0` **且之後無 CUDA error**）；若 alloc 仍敗→驗證 host 池退路生效（`retrying on host`）且整機可跑 | 載入成功＋VRAM 帳合理（池 11.7GB＋KV＋雜項） |
| R3 | 正確性：混合雙路徑下真機輸出合理（不再歸零）——短文生成語義連貫；如可取得 CPU 全量基準則抽樣比對 | 語義正確 |
| R4 | 效能重測：prefill 時間（61 token，對比 CPU-MoE 時代的 141 s）＋ON 暖 decode t/s（門 ≥8）；池命中後 GPU 路徑比重 | prefill 大幅下降；t/s 達門或記錄實測值 |
| R5 | C3-3 stats 活體接線：`/q4e/stats` 從合成元件改掛真實 `q4e_v6->rt` 的 pool stats（hits/misses/evictions/rounds）＋runtime IO | 真機請求後 stats 非零且與 log 一致 |
| R6 | 收尾門：e2e 綠＋收尾 VRAM 歸零 | 全綠 |

**打包（R1~R6 全綠後，按用戶 9/20 指示）：**

| # | 工作 |
|---|---|
| P1 | git commit 剩餘修正 |
| P2 | 清開發暫存：`v6_work/`（gitignore 內，磁碟可留可刪）、`_testtmp/`、build 目錄不入庫確認、根目錄散落 log |
| P3 | 對比驗證：舊 `D:\llama\llama-server.exe` vs 新 `q4e-server.exe`（同模型、同 prompt、同 ctx）：載入時間／prefill t/s／decode t/s／VRAM 峰值／輸出樣例，留數據 |
| P4 | 數據報告（`REPORT_v6.1.md`：架構、門表結果、對比表、已知限制）＋打包準備 GitLab |

**已知風險／備註：**
- WDDM：即便延後分配，CUDA0 上 11.7GB 仍可能失敗（診斷證實單獨行程可成功；若 R2 再敗→host 池退路保正確，效能門以 R4 實測為準，t/s 不達門時回報用戶裁決）。
- 舊 llama.cpp 對比基準：舊版無 q4e 池語義，對比表重點＝「能否載入 103.7GB 模型」與 t/s 量級差異，非同構對比。
- runtime promotion 目前為同步填槽（SSD→host→GPU 逐專家 copy）；非同步管線（h2d-pipeline 接入）為後續效能優化項，不擋打包。
