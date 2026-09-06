# Track B §9 回報 v4（B-0 → B-7 完成）

> 回報對象：Task A（聰明 AI）。日期：2026-09-06。
> 執行方式：照 [RUNBOOK_B_v2.md](RUNBOOK_B_v2.md) **零判斷執行**（每步＝指令＋預期輸出＋備案）；全部輸出為真實執行結果；失敗步照備案「錯誤原文＋證據」回報，**未調參、未改 CMake 參數、未改 `engine/` source、ngl 恆 10**。
> v1 回報（wheel/Python 方法學）在 `legacy_v1/B9_report.md`，本文為 **v4 C++ 產品**（`engine/` q4e 樹）的 §9。

## 0. 總覽（M-C 8 項驗收，02 §六）

| # | 驗收項 | 判定 | 證據 |
|---|---|---|---|
| 1 | ngl=10＋VRAM≈16,750＋`/v1` e2e 綠（200/401/SSE） | **PASS** | B-2：16,724 MiB、~206 s；B-3：5 閘全綠 |
| 2 | 真實 M 可測（核心看得到 router） | **PASS** | touches 非零（B-3 時 73,494 → B-6 時 327,436）→ 解掉 v1「M 引擎外不可測」gap |
| 3 | 暖 t/s ≥ `estimate_tps(M_real)`×0.7 | **FAIL** | 1.5692 < 10.773（M=44 列 15.39×0.7；M_real_lower=26.647）→ §4 回報 |
| 4 | 預取重疊度（PCIe 段）≥80% | **本 build 不可量** | `prefetch_issued=0`（A-5 接線＝no-op 狀態機）→ §4 回報現況，**未編造數字** |
| 5 | N-gram 預取 t/s 掉幅 <1% | **本 build 不可量** | 同 #4（無 live 預取可量） |
| 6 | 快取命中報告（vram/ram/ssd）＋I/O bytes＋真實 M | **PASS** | B-6：vram 43.51% / ram 43.11% / ssd 13.29%；io 1,242,921,984,000 B；M_real_lower=26.647 |
| 7 | A 側 C++ 測試全綠 | **PASS（A 側證據）** | A-3 82 checks＋A-4 2957 checks＋A-5 e2e（A 端交付，B 未重跑） |
| 8 | 冷載入 ≤~250 s＋收尾 VRAM 歸零 | **PASS** | B-2：~206 s（v1 參考 226.4 s）；B-7：0 MiB、無殘留程序 |

**結論：8 項中 5 PASS、1 FAIL（#3）、2 本 build 不可量（#4/#5，待 A 決定是否先補 live 預取接線再驗）。** B-4 硬線（ON≥OFF）同為 FAIL（見 §1 B-4 與 §4）。

## 1. 每步輸出

### B-0 前置 — PASS
- driver 610.47、CUDA UMD 13.3、nvcc V13.3.73；GGUF 4 片合計 111,334,654,784 B（≈103.69 GiB）；無殘留 q4e/llama 程序；VRAM 0 MiB / 24,576 MiB。

### B-1 q4e CUDA build — PASS
- RUNBOOK 原樣指令（vcvars64＋`-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DGGML_CPU_REPACK=OFF`，Ninja）→ **474/474 全綠**，日志尾 `[474/474] Linking CXX executable bin\q4e-server.exe`（107,766 行，零 build 錯誤；日志內 4 個 "FAILED" 字串＝3× CMake `Performing Test … - Failed`（CMAKE_HAVE_LIBC_PTHREAD / HAS_AVX512_1 / HAS_AVX512_2，Windows/MSVC 正常結果）＋1× line 113992 `-- UI: download dist.tar.gz from b17 failed`（web-UI 靜態資源下載失敗，**不影響 API**，build 仍 474/474 全綠）。
- 產物：`engine\build-cuda\bin\q4e-server.exe` 44,027,392 B；`--version` → `0.3.0-dev (build 17, commit 1144ddd)`、`built with MSVC 19.44.35228.0 for Windows AMD64`。
- 觀察 1（非門失敗）：背景 job 層 exit=1，但日志無對應錯誤、產物有效可執行 → 記為 harness/pipeline 假象。
- 觀察 2（新增環境特性）：WinLibs-MSYS2 ninja 1.13.2 在本 CJK repo 路徑下**增量 build 必掛**（存在 `.ninja_log` 時重建檢查用 ANSI API 掃 CJK 絕對路徑 → `FindFirstFileExA(...) syntax is incorrect`）；首次 full build（無 `.ninja_log`）正常。本 task 不需重編→無影響；workaround 已記錄於 `evidence_b1/root_cause_b1.md`（刪 `.ninja_log`／官方 standalone ninja／非 CJK 路徑）。
- 證據：`b1_retry_build.log`（41,998,042 B）、`evidence_b1/`（root_cause_b1.md＋TryCompile 備份＋b1_build.log）。

### B-2 真 GGUF 載入（ngl=10）— PASS
- `Start-Process q4e-server.exe --model …00001-of-00004.gguf --host 127.0.0.1 --port 8124 --api-key bkey --n-gpu-layers 10 --ctx-size 2048 --parallel 1`。
- health=200，**~206 s**（15 s 輪詢粒度，190–206 s 窗；v1 參考 226.4 s）。
- nvidia-smi：`16724MiB / 24576MiB`（v1 16,750 MiB）。
- 日志（llama.cpp 慣例寫 stderr）：`llama_server: model loaded`、`listening on http://127.0.0.1:8124`。
- 服務保持運行供 B-3（PID 14616）。
- 證據：`b2_server.log`（空）、`b2_server.err.log`。

### B-3 `/v1` e2e 5 閘 — PASS（5/5 綠）
- (a) `/health` → 200。
- (b) 非流式 chat（64-token 快取問）→ 200，`usage.completion_tokens=52`（`content` 與 `reasoning_content` 皆有 tokens；`prompt_tokens=62`，cached 58）。
- (c) 無 key → **401**。
- (d) SSE → 52 行 `data:`（51 事件＋末行 `data: [DONE]`）。
- (e) `/q4e/stats` → 200，**touches=73,494（非零＝cb_eval 鉤子確有觸發）**；hits vram 28,431 / ram 30,887 / ssd 13,972；hit_rate 0.3868/0.4203/0.1901；placement_events 95,499；io_bytes_estimate 293,372,928,000；occ_vram_avg 20.0 / occ_ram_avg 138.1；prefetch_issued 0。
- 執行註記（非參數變更）：本 harness 會剝離單引號 `-d '{…}'` 的雙引號（`curl --trace` 證實）→ 同一字節載荷改以 `-d "@body_*.json"` 發送（URL/方法/header 不變）；載荷檔 `body_b.json`/`body_c.json`/`body_d.json` 保留為證據。

### B-4 快取開 vs 關硬線 — **FAIL（ON < OFF，照備案回報、未調參）**

| 端 | completion_tokens | time_total (s) | t/s | 端側 eval（log 原文折算） |
|---|---|---|---|---|
| ON（q4e :8124，hook 開） | 256 | 163.136 | **1.5692** | 158,181 ms/256 = 1.61 t/s（620 ms/token；tg 由 1.41→1.60 爬升，1.41＝B-4 即時觀察，log 快照見 1.54→1.60） |
| OFF（oracle :8125，無 hook） | 256 | 144.634 | **1.7700** | 137,285 ms/256 = 1.86 t/s（538 ms/token） |

- 差距 **−11.3%**（ON 較慢）。兩端均 256 tokens、`finish_reason=length`、prompt 61 tokens。
- 切換紀律：kill q4e → 5 s → nvidia-smi 確認 **VRAM 歸零** 後才起 oracle（從未雙實例並載）；oracle 冷載 210.8 s、health 200；收尾 kill llama-server。
- 版本事實：oracle `b10685-17252c769`、q4e `b17-1144ddd`（RUNBOOK 註記「oracle 10685、q4e b10729 基」）。
- 備案執行：ON<OFF → **回報**（兩端 t/s、tokens、time_total、雙方 log 尾段 30 行，全文見子代理回報與 §2-c），**未自行調參**。
- 產出：`b6_stats.json`（271 B）、`b5_q4e.json`（950 B）、`b5_oracle.json`（963 B）三檔齊。
- 證據：`b4_off.log`（空）、`b4_off.err.log`、載荷 `body_on256.json`/`body_b5_64.json`。

### B-5 oracle 對答案 — PASS（一致或可解釋）
- `content` 兩端**位元相同**：`A cache is a small, fast storage layer that temporarily holds frequently accessed data to speed up future requests.`
- `reasoning_content` 差 **1 token**：q4e「…Ensure concise.」vs oracle「…Keep concise.」→ 可解釋（b17 vs b10685 kernel/logits 差異＋兩跑 prompt-cache 狀態不同：cached 58 vs 42、prompt_n 4 vs 20）。
- 其餘欄位差異（raw）：`usage.cached_tokens` 58 vs 42；`prompt_ms` 7383 vs 18232；`predicted_ms` 33161 vs 29098；`predicted_per_second` 1.538 vs 1.753；`system_fingerprint`/`id`/`created` 各異。
- 證據：`b5_q4e.json`、`b5_oracle.json`（全文）。

### B-6 性能基準（02 §六）— 第 1 行 FAIL；第 2/3 行本 build 不可量
- `estimate_tps.py`（`py -3.14 -X utf8`）全表已抓（見 §6 證據）。關鍵列：M=20→412.86、**M=44→15.39**、M=90→10.92、M=142→9.91、M=164→7.12、M=256→0.81、M=512→0.45 t/s。
- `b6_stats.json`（B-4 計數跑後）：`{"layers":48,"touches":327436,"hits_vram":142480,"hits_ram":141152,"hits_ssd":43528,"hit_rate_vram":0.4351,"hit_rate_ram":0.4311,"hit_rate_ssd":0.1329,"prefetch_issued":0,"occ_vram_avg":20.0,"occ_ram_avg":142.0,"placement_events":404597,"io_bytes_estimate":1242921984000}`
- M_real 代理（RUNBOOK 指定式）：M_real_lower = touches ÷ (48×256) = 327,436 ÷ 12,288 = **26.647**。
  （事實註記：RUNBOOK 括號預期「top-10 路由下 ≈10–11」與實測 touches 不符；已照指定式套實測數。）
- 驗收線：表中最小 M ≥ 26.647 → **M=44**（15.39）×0.7 = **10.773**；暖 t/s 1.5692 ≥ 10.773 → **NO**。
- 第 2/3 行：`prefetch_issued=0` → 本 build **無 live 預取可量**（A-5 接線＝狀態機 no-op）→ 依 RUNBOOK「回報 A 現況，**不編造重疊數字**」。

### B-7 收尾 — PASS
- `Get-Process … q4e-server|llama-server` → **空**；nvidia-smi → **0 MiB / 24,576 MiB**、`No running processes found`。
- GGUF 4 片在 D: SSD：10,946,624 / 49,859,583,136 / 49,376,141,504 / 12,087,983,520 B（合計 111,334,654,784 B＝B-0 一致；PLE 表隨分片留 SSD，離載不佔 VRAM）。

## 2. 錯誤/異常訊息原文（逐條）

- **(a) B-1 job exit=1（harness 假象）**：474/474 全綠、日志零錯誤、產物有效 → 非門失敗，記錄在案。
- **(b) B-1 增量 ninja CJK 失敗（新增環境特性）**：`cmake --build`／`ninja -C`（有 `.ninja_log`）→ `ninja: error: FindFirstFileExA(../../../../AI研究/推理工具/engine/common): The filename, directory name, or volume label syntax is incorrect`（ANSI API 不處理 CJK；增量 rebuild-edge 掃描專用；full build 路徑正常）。
- **(c) B-4 硬線 FAIL 原文**：ON 256 tokens/163.136 s vs OFF 256 tokens/144.634 s（見 §1 B-4 表；雙方 log 尾段 30 行全文在執行回報，核心行：q4e `eval time = 158181.04 ms / 256 tokens`；oracle `eval time = 137285.08 ms / 256 tokens`）。
- **(d) B-6 驗收線 FAIL**：1.5692 < 10.773（計算見 §1 B-6）；另 RUNBOOK 的 M 括號預期（10–11）與實測 M_real_lower（26.647）不符——兩條都屬**回報事實**，profile MoE 分發/kernel 由 A 決定是否展開。
- **(e) B-3 harness 引號剝離**：單引號 `-d` JSON 雙引號被 harness 剝除（`curl --trace` 證實送出體無引號）→ 改 `-d @file`（字節相同載荷），非產品缺陷。

## 3. nvidia-smi VRAM 全程
- B-2 跑動（ngl=10）：**16,724 MiB / 24,576 MiB**（v1：16,750）
- B-4 切換點（kill q4e 後）：**0 MiB**（確認歸零才起 oracle）
- B-7 收尾：**0 MiB / 24,576 MiB**、無程序

## 4. 卡住？— 無卡住步；兩件「回報 A 現況」＋一件 FAIL 判定
1. **B-4 硬線 ON<OFF（−11.3%）**：已照備案回報、未調參。A 端可查的原始事實：q4e 側 tg 由 1.41 爬至 1.60（ramp 未完全收斂；1.41 為 B-4 即時觀察，log 快照 `b2_server.err.log` 見 1.54→1.60、n_gen 222→251）、端側 eval 1.61 vs 1.86 t/s、版本差 b17 vs b10685。RUNBOOK 註記：版本差過大→回報事實由 A 判定。
2. **B-6 #3 FAIL（1.5692 < 10.773）＋ M 錨點不符**：M_real_lower=26.647（ touches 累計制式）落 M=44 列（15.39×0.7=10.773）。注意 estimate_tps 表本身為**串聯期望時間模型**（M=44 列 15.39 t/s 對應 VRAM 2×熱＋RAM 全覆蓋 44 專家的理想兩層快取），與本 build 單層快取（occ_ram_avg 142 已達 RAM 上限、occ_vram_avg 20）的實測 1.57 t/s 差距，屬**架構層面**而非單一 bug——是否 profile/對齊兩層設計由 A 判定（RUNBOOK 備案：不及→profile MoE 分發/kernel＋回報原文，已照辦）。
3. **#4/#5 不可量**：`prefetch_issued=0`（no-op 狀態機）→ live 預取重疊/N-gram 掉幅無物理量可測；是否先補 live 接線再驗由 A 決定（RUNBOOK 原話）。

## 5. 約束遵守
- ✅ 全程 **ngl=10**（B-2 起動、B-4 OFF 端 oracle 亦 `-ngl 10`）；從未 999/48/auto。
- ✅ CMake 參數零變更（RUNBOOK 原樣）；**未改 `engine/` source/test**。
- ✅ 永不過兩模型實例同時載入（B-4 切換：kill→VRAM 歸零→再起）。
- ✅ 失敗步＝錯誤原文＋證據回報；不猜、不繞、不調參、不編造數字。
- ✅ 證據全保留（清單見 §6）。

## 6. 證據檔清單（repo 根 `D:\Nova2_home\AI研究\推理工具\`）

| 檔 | 大小 | 內容 |
|---|---|---|
| `b1_retry_build.log` | 41,998,042 B | B-1 全綠 build 日志（474/474） |
| `evidence_b1\` | — | root_cause_b1.md（掛起根因＋重跑結果＋CJK 增量發現）、TryCompile-610ap5-backup、b1_build.log |
| `b2_server.log` / `b2_server.err.log` | 0 / 11,755 B | B-2 服務日志（llama.cpp 慣例走 stderr） |
| `body_b.json` / `body_c.json` / `body_d.json` | 131/60/130 B | B-3 請求載荷（@file 發送） |
| `body_on256.json` / `body_b5_64.json` | 140/131 B | B-4 請求載荷 |
| `b6_stats.json` | 271 B | B-4 後 `/q4e/stats`（B-6 輸入） |
| `b5_q4e.json` / `b5_oracle.json` | 950/963 B | B-5 對答案兩端全文 |
| `b4_off.log` / `b4_off.err.log` | 0 / 8,929 B | B-4 OFF 端 oracle 日志 |
| `engine\build-cuda\` | — | 全量 CUDA build 樹＋`bin\q4e-server.exe`（44,027,392 B） |
| estimate_tps 全表 | — | 於 B-6 執行回報全文（`py -3.14 -X utf8 D:\Nova2_home\AI研究\冷熱架構與t_s估算\estimate_tps.py`） |
