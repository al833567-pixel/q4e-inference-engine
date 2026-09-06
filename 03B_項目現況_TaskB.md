# 推理工具 — 項目現況 · Task B（有 GPU，較笨 AI）

> 更新：2026-09-06（v4.1：B 側三件「回報 A」已收到 A 端安排——**第二輪＝A-9（熱路徑開銷移除）→ A-11（驗收線拍板）→ A-10（live 預取）→ A-12（交付）→ B-8 復驗**，照 [RUNBOOK_B_v3.md](RUNBOOK_B_v3.md) 零判斷執行；B-8 待 A-12 交付後解鎖）
> v4 歷史：2026-09-06（**Task B 全數完成（B-0～B-7）**；§9 回報＝[B9_report_v4.md](B9_report_v4.md)；M-C 8 項＝5 PASS／1 FAIL（暖 t/s 驗收線）／2 本 build 不可量（live 預取＝no-op））
> 姊妹檔：[03A_項目現況_TaskA.md](03A_項目現況_TaskA.md)
> 關聯：[02_實作項目工作規劃.md](02_實作項目工作規劃.md)（Task B 項目＝§五）

## 〇、Task B 的角色

**有 GPU 可用、較笨 AI**：照 RUNBOOK **零判斷執行**（每步＝指令＋預期輸出＋備案）：CUDA build＋真機載入＋oracle 對答案＋驗收＋§9 回報。
**v4 不變**：B 端工作全部照 RUNBOOK_B v2 執行；被測對象＝**C++ 產品**（`engine/` q4e 樹）。
**失敗回報格式＝錯誤原文＋證據，不猜。**

## 一、Task B 環境（已實測）

| 項目 | 狀態 | 說明 |
|---|---|---|
| **GPU** | ✓ | RTX 3090 **24 GB**；ngl=10 跑動佔 **16,724 MiB**（v1：16,750）；收尾歸零 |
| **CUDA toolkit（nvcc）** | ✓ | B-0 實測 **v13.3**（nvcc V13.3.73；UMD 13.3；driver 610.47） |
| **CPU / RAM / SSD** | ✓ | i7-9700 8C/8T；RAM ~20 GB；SATA 0.31–0.44 GB/s |
| **GGUF** | ✓ | `D:\AI_Model\Qwen3.8-Flash-Next\`（4 片 111,334,654,784 B ≈103.69 GiB Q4_K_XL） |
| **oracle** | ✓ | `D:\llama\llama-server.exe`（fork build 10685；**僅二進位**） |
| **MSVC** | ✓ | VS 2022 BuildTools（vcvars64.bat）—B-1 CUDA build 用 |
| **Python** | ◐ | `py -3.14 -X utf8`（**僅** legacy_v1 方法學工具＋repo 外 t/s 估算；v3 合約層已移除，B 端不依賴） |

## 二、Task B 已驗證基準（v1 證據；v3/v4 驗收對照用）

| 數字 | 值 | 來源（`legacy_v1/evidence/`） |
|---|---|---|
| 冷載入（ngl=10） | **226.4 s**，VRAM 16,750/24,576 MiB | `smoke_b1.log` |
| 暖 t/s（無預取，M=512 上界列） | **1.95**（PASS ≥0.28） | `bench_noprefetch.log`/`bench_check.log` |
| 冷 t/s | 0.33 | bench |
| fake_router 無局部性（M=512 均勻） | 0.10–0.32 t/s＝SATA 頻寬牆（**非快取失敗**） | `wire_cache_live.log`/`bench_prefetch.log` |
| 快取命中（301 tokens） | vram 9.1% / ram **90.6%** / ssd **0.3%** | `wire_cache_live.log` |
| N-gram 預取 | **淨加速**（交錯 4 組 drop −7.4%…−10.2%；首次 34.19% FAIL＝ramp 假象） | `ngram_accept.log`/`ngram_diag.log` |
| ramp law | 新 phase 前 ~300 tokens 由 0.5→2.4 t/s 爬升 | `ngram_diag.log` |
| 旁證：真實路由有局部性 | 每 token 讀 ~1.17 GB 權重仍達 1.95 t/s→工作集住進 page cache→**真實 M 遠小於 512** | `legacy_v1/B9_report.md` §B-5 |

## 三、Task B 資產（v4）

| 資產 | 位置 | 角色 |
|---|---|---|
| **C++ 產品**（q4e 樹） | `engine/` | **B-1～B-7 的被測對象**（CUDA build＋真機） |
| oracle（fork 二進位） | `D:\llama\` | B-2 輸出對答案（token 序列）；**只對答案，代碼不進產品** |
| v1 證據（36 檔） | `legacy_v1/evidence/` | 基準數＋方法學實證 |
| v1 驗收報告 | `legacy_v1/B9_report.md` | §9 回報格式範本 |
| v1 工具（封存） | `legacy_v1/tools/` | 方法學參考（交錯 A/B、tail 窗、ExpertStore 佈局） |
| e2e body | `tools/b5_body.json` | B-6 e2e 請求體（JSON，保留） |
| t/s 驗收線 | [../冷熱架構與t_s估算/estimate_tps.py](../冷熱架構與t_s估算/estimate_tps.py) | B-5：`estimate_tps(M_real)`×0.7 |

## 四、Task B 項目（v4 不變）

| 項目 | 名稱 | 狀態 | 說明 |
|---|---|---|---|
| **B-0** | 前置：VRAM 空出＋CUDA 13.3（nvcc）＋GGUF 4 片＋無並行實例 | ✅ **PASS**（2026-09-06） | driver 610.47、nvcc V13.3.73、4 片 111,334,654,784 B、VRAM 0 |
| **B-1** | q4e **CUDA build**（vcvars64＋`GGML_CUDA=ON`＋`BUILD_SHARED_LIBS=OFF`＋`GGML_CPU_REPACK=OFF`） | ✅ **PASS** | **474/474 全綠**；`bin\q4e-server.exe` 44,027,392 B＝`0.3.0-dev (build 17, commit 1144ddd)`（MSVC 19.44.35228.0）；觀察：job exit=1 假象＋CJK 增量 ninja 特性（見 §2(a)(b)） |
| **B-2** | 真 GGUF 載入（**ngl=10**）＋`/health` 200（≤~250 s）＋VRAM 占用 | ✅ **PASS** | health 200 **~206 s**（v1 226.4）；**16,724 MiB**（v1 16,750）；`model loaded`＋`listening` |
| **B-3** | `/v1` e2e 5 閘（200／≥1 token／401／SSE）＋`/q4e/stats` 活數據 | ✅ **PASS（5/5）** | 200／52 tokens（content＋reasoning 皆有）／401／52 `data:`＋`[DONE]`／**touches=73,494 非零**（鉤子活） |
| **B-4** | 快取驗 **開 vs 關硬線**：暖 t/s（ON）≥（OFF，oracle 無 hook 基線） | ❌ **FAIL（回報、未調參）** | **ON 1.5692（256 tok/163.136 s）< OFF 1.7700（256/144.634 s）＝−11.3%**；兩端 log 尾段＋`b6_stats.json` 已存；備案＝回報不調參 |
| **B-5** | **oracle 對答案**（vs `D:\llama\llama-server.exe`，同 prompt＋temp=0） | ✅ **PASS（可解釋）** | `content` **位元相同**；`reasoning_content` 差 1 token（Ensure/Keep）＝b17 vs b10685 kernel 差＋cache 狀態 58 vs 42；`b5_q4e.json`/`b5_oracle.json` 全文 |
| **B-6** | 性能基準（02 §六）：暖 t/s ≥ `estimate_tps(M_real)`×0.7＋重疊 ≥80%＋N-gram 掉幅 <1% | ❌ **第 1 行 FAIL／第 2/3 行本 build 不可量** | M_real_lower=327,436÷(48×256)=**26.647**→M=44 列 15.39×0.7=**10.773** > 1.5692；`prefetch_issued=0`（no-op 狀態機）→ 重疊/N-gram 無物理量可測，**未編造**；hits 43.5/43.1/13.3%、io 1.24 TB |
| **B-7** | 收尾：VRAM 歸零＋PLE 留 SSD＋程序退出 | ✅ **PASS** | 無殘留程序；**0 MiB / 24,576 MiB**；4 片留 D: SSD（111,334,654,784 B） |
| **B-8**（v4.1） | **復驗（A-12 交付後）**：CUDA build→ngl=10→e2e→**硬線 ON≥OFF（重驗）**→暖 t/s vs 拍板線＋預取重疊 ≥80%→oracle 對答案→收尾 | ⏳ **待 A-12** | 照 [RUNBOOK_B_v3.md](RUNBOOK_B_v3.md) 零判斷；重驗 v4 三門（B-4 硬線、#3 t/s、#4 重疊）＋回歸門 |

> **Task B 全數完成（B-0→B-7，2026-09-06）**；§9 回報＝[B9_report_v4.md](B9_report_v4.md)。驗收＝02 §六 M-C 8 項＝**5 PASS／1 FAIL（#3 暖 t/s 驗收線）／2 本 build 不可量（#4/#5，live 預取 no-op）**。B-4 硬線同 FAIL（ON<OFF，已照備案回報）。

## 五、Task B 約束

| 約束 | 決定 |
|---|---|
| **ngl=10 固定** | **使用者紅線**（永不 auto/999/48） |
| **build 旗標紅線** | CMake 必帶 `-DGGML_CPU_REPACK=OFF`＋`-DBUILD_SHARED_LIBS=OFF`；build 環境＝`vcvars64`＋cl（PATH 上的 MinGW 會搶編譯器） |
| **並行** | 永不過兩個模型實例同時載入；先 kill server 再 in-process 載 |
| **零判斷** | 照 RUNBOOK_B v2 執行；每步＝指令＋預期輸出＋備案 |
| **失敗回報** | **錯誤原文＋證據**（不猜、不繞） |
| **量測紀律** | temp=0、固定 prime prompt、warmup 32；A/B 交錯（A B A' B'）＋tail 窗（ramp law） |
| **日誌** | PS Tee 輸出＝UTF-16LE（讀時 BOM sniff）；證據全保留 |
| **console** | legacy 工具用 `py -3.14 -X utf8`（cp950 對策）；產品本身＝C++ 二進位 |

## 六、現況總結

- **v1 B-0～B-6：全數完成並驗收**（M-B 達標；§9 回報＋36 證據在 `legacy_v1/`）。
- **v4：Task A 全數完成（A-0～A-6）→ Task B 全數完成（B-0→B-7，2026-09-06）**。照 [RUNBOOK_B_v2.md](RUNBOOK_B_v2.md) 零判斷執行；§9 回報＝[B9_report_v4.md](B9_report_v4.md)。
- **驗收**＝02 §六 M-C 8 項：**5 PASS**（ngl=10＋e2e 綠、真實 M 可測、命中報告、A 側測試全綠、冷載入 ≤~250 s＋VRAM 歸零）／**1 FAIL**（#3 暖 t/s 1.5692 < 10.773＝estimate_tps(M=44)×0.7）／**2 本 build 不可量**（#4 重疊≥80%、#5 N-gram<1%：`prefetch_issued=0`＝no-op 狀態機，無物理量可測→回報現況不編造）。
- **B-4 硬線 FAIL（ON 1.5692 < OFF 1.7700，−11.3%）**：照備案回報（t/s＋tokens＋time_total＋雙 log 尾段），**未調參、未改 CMake、未改 source**；原始事實（tg 爬升 1.41→1.60、1.41＝B-4 即時觀察／log 快照見 1.54→1.60、端側 eval 1.61 vs 1.86、版本差 b17 vs b10685）供 A 判定。
- **回報 A 現況（2026-09-06 已回報）**：(1) B-4 硬線 FAIL 原始事實（tg 爬升、端側 eval 1.61 vs 1.86、版本差 b17 vs b10685）；(2) B-6 #3 FAIL＋M 錨點不符（架構層，非單一 bug）；(3) #4/#5 不可量（`prefetch_issued=0`）。
- **A 端安排（v4.1，2026-09-06）**：根因已核實（`cb_eval` sync D2H＋no-op backend＋prefetch 未接線，見 03A §7.1）→ **B-8 復驗**＝A-9（開銷移除）→ A-11（#3 驗收線拍板，Option A 推薦）→ A-10（live 預取）→ A-12（交付包 v4.1）→ B 照 [RUNBOOK_B_v3.md](RUNBOOK_B_v3.md) 執行（B-8 待 A-12 解鎖）。
- **紅線**：ngl=10 固定（不 999/48/auto）；oracle `D:\llama` 只對答案、代碼不進產品；不為乾淨砍功能。
- 卡住規則：任一门失敗→停下回報（步驟、錯誤原文、已試事項），不猜。（本次無卡住步；FAIL 均為門判定，已照備案回報。）
