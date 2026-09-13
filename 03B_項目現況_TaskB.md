# 推理工具 — 項目現況 · Task B（有 GPU，較笨 AI）

> 更新：2026-09-12（**本檔只留未完成項目**；已完成工作→[已完成工作.md](已完成工作.md)；commit 全錄→`git log --oneline`）
> 姊妹檔：[03A_項目現況_TaskA.md](03A_項目現況_TaskA.md)
> 關聯：[02_實作項目工作規劃.md](02_實作項目工作規劃.md)；[RUNBOOK_B_v3.md](RUNBOOK_B_v3.md)

## 〇、Task B 的角色

**有 GPU 可用、較笨 AI**：照 RUNBOOK **零判斷執行**（每步＝指令＋預期輸出＋備案）：CUDA build＋真機載入＋oracle 對答案＋驗收＋§9 回報。**失敗回報格式＝錯誤原文＋證據，不猜。**

## 一、未完成項目

| 項 | 名稱 | 狀態 | 內容／門 |
|---|---|---|---|
| **B-8**（v4.1） | **復驗（A-12 交付後）**：照 [RUNBOOK_B_v3.md](RUNBOOK_B_v3.md) 零判斷 | **⏳ 進行中（2026-09-12 重跑、build 42 GPU 復驗：B-8-0/1/2 ✅（冷載入 286 s、health 200、VRAM 21,606 MiB fit PASS）→ ⛔ B-8-3 G5 FAIL `layers_touched=38 (want layers=48)` → B-8-4~7 未啟動；證據 `b8_evidence/step3_rerun_b42`、問題紀錄見下）→ ✅ A 側裁決＝38 為預期值＋G5 placement-aware 修復交付（2026-09-12、新 `q4e-e2e.exe`、[b8_evidence/stepA_g5/adjudication.md](b8_evidence/stepA_g5/adjudication.md)）、**2026-09-12 B 側重跑：B-8-3 ✅ G0–G6 全 PASS（`Q4E_E2E_ALL_PASS`、G5＝`layers_touched=38/48 (expected 38)`）＋B-8-5 oracle 對答案 ✅ → ⛔ B-8-4 硬線 FAIL（ON 1.6366 t/s ＜ OFF 2.3002 t/s、−28.8%）＋B-8-6 門 1 FAIL／門 2 PASS（overlap 0.93）／門 3 ngram 未量（停於卡住規則）；B-8-7 收尾 ✅；證據 `b8_evidence/step4_rerun_g5fix`、問題紀錄見下 → 待 A 側裁決 → **✅ A 側裁決完成（2026-09-12：修復 B-8-4-F1 build 47 已交付；門 3 維持 GPU 量；重跑指示見下「A 側裁決」節）**）→ ✅ B-8-1 build 完成 → ⛔ B-8-2 BLOCKED（fork bug）→ ✅ A 側修復 round 1（`0ccc16c`/build 39）→ ⛔ 重跑 FAIL（78,431/78,768 MiB）→ ✅ A 側 H1/H2 裁決＝H1 成立＋修復 round 2 已交付（2026-09-10，`207fe88`/build 42）**：真根因＝`common_params_parse_ex` 先 pad `tensor_buft_overrides` 4096 個 null、q4e pattern append 在 null 之後→loader 套用 **0 筆**（regex 無誤）；新 exe＝`engine\build-cuda\bin\q4e-server.exe`（build 42）＋test-arg-parser 全綠（含新 padded-buffer insert 測試）；**新預期 VRAM＝21,315 MiB**（fit no_alloc；model 20,106.45 on CUDA0＋ctx/compute）——B 側重跑 B-8-2 原文預期＝VRAM-fit 通過（≈21,3xx ≤ free 22,884）＋health 200 ≤~250 s＋nvidia-smi 實占 ≈20,1xx–21,3xx MiB；**前置：用戶再釋放 GPU（llama-server PID 13392 已回來占 23,680/24,576 MiB、絕不殺）**；**環境警示＝GPU 驅動慢性 WATCHDOG＋9/10 09:00 BSOD（0x133，minidump 091026-42859-01.dmp）→ 重跑期間機器崩潰＝環境問題、回報原文即可**；全細節＝[b8_evidence/stepA_h1h2/adjudication.md](b8_evidence/stepA_h1h2/adjudication.md)。**（原 round 1 紀錄）B-8-2 重跑（2026-09-09 20:3x，build 39）＝⛔ FAIL（VRAM-fit 清晰錯誤）**——fork 修復實機生效（不再 assert 崩、已抵達 VRAM-fit check 給清晰錯誤＝RUNBOOK 預期形態），但兩個獨立阻擋：① GPU 被 harness 自家模型服務（PID 2992 `llama-server` port 8080）占用 ~24 GB（絕不殺紅線→**待用戶釋放**）② check 實測 ngl=all+K=10 placement 需 **78,431 MiB** ＞ 全空 GPU 24,576 MiB、且為預期 ≈20.6 GB 之 3.8× → **B-8-2 空 GPU 重跑（2026-09-10 00:53，build 39）＝⛔ FAIL 再確認**：`CUDA0 needs 78768 MiB, only 22884 MiB free`（WDDM 樂觀值；全空 24,576；vs 20:3x 78,431 Δ≈+0.4%＝確定性 placement 測量、與 GPU 占用無關）→ **原文必 FAIL，⛔ 待 A 側 H1/H2 裁決（spec 自含，見下方兩節證據）；前置②用戶釋放 GPU＝✅ 已完成（llama-server 已消失、VRAM 0 MiB）**。B-8-1 沙箱映射根因已解（`danger-full-access` 後 build EXIT=0，`q4e-server.exe` 42 MB commit `b2dacb5`）；B-8-0 全綠。**B-8-2 真機載入 crash（非超時）**：`common.cpp:1768 GGML_ASSERT(...Tensor buffer overrides not terminated with empty pattern) failed`——`--q4e-moe-vram 10` 經 `common_q4e_moe_vram_apply`(common.cpp:1311)→`llm_add_n_cpu_ffn_overrides(38,...)`(common.h:1144) 只推 38 筆非 null pattern、**未加 null-terminator**，緊接 `common_model_params_to_llama`(common.cpp:1349) 斷言 `back().pattern==nullptr` 即崩；fork 自測(test-arg-parser.cpp:400) 只驗 `size()==38` 故遮蔽此不變式。**影響所有 k<n_layers（K=10 必崩）**，且崩於 VRAM-fit check(common.cpp:1352) 之前→非預期的「清晰 VRAM 錯誤」而是 crash。VRAM 0MiB、無程序、port 8124 空。**備案：不降 K、不改 engine source→待用戶/A 側決定**（最小修法：`common_q4e_moe_vram_apply` 回傳前補 `{nullptr,nullptr}`）。下方 B-8-1／B-8-2 阻斷為問題紀錄 | **重驗 v4 三門**＋回歸門：(1) **B-4 硬線**：暖 t/s **ON ≥ OFF**（A-9 已移除 sync D2H 根因）；(2) **#3′**：ON ≥ OFF 且 **ON ≥ 1.95 t/s**（A-11 拍板線）；(3) **#4 重疊**：`prefetch_overlap_ratio` **≥80%**（A-14.1 提供欄位；定義＝觸碰前已完成預取的 touches 佔比）＋`prefetch_io_issued>0`；**#5 N-gram＝A-14.4 ✓ 可量**（`5b09765`／`c92606e`；CPU 預驗 PASS max drop +0.22%；B 側 GPU 量）；回歸門＝冷載入≤~250 s＋VRAM 歸零＋oracle 對答案＋e2e 5 閘。**工具**：`q4e-bench`（ON/OFF t/s＋JSON）＋`q4e-e2e`（取代 curl 手動步）。**設定**：ON＝q4e `--n-gpu-layers all --q4e-moe-vram 10`（K=10 預設、**K 可設定**、B-8 可量 12/13 找上限）；OFF＝oracle（上游原生旗標）`-ngl all -ncmoe 38`（B-8-0 先驗 oracle 有 `-ncmoe`；若無→q4e `--no-q4e-prefetch` 同 placement）。**FAIL＝停下回報原文，不調參。** |

#### ✅ B-8-1 阻斷已解（2026-09-09 09:59；根因＝沙箱路徑映射，非 build 參數／engine）— 問題紀錄

**解開方式**：用戶開 `danger-full-access`（**關閉 DSH 檔案沙箱**）後，fresh configure **直接通過** C/CXX/CUDA ABI probe（`Configuring done (25.8s)`→`Generating done`→`Build files … D:/Nova2_home/…`），**主 `build.ninja` workdir 已是真 `D:`**（非 `D$:`）。證明 `D$:` 是 **沙箱把 workspace 映射成 `D$` 驅動** 造成，非 CMake／ninja／編譯器問題。隨之 `cmake --build build-cuda --target q4e-server` **EXIT=0**（649.8 s）→ `bin\q4e-server.exe`（44,076,544 B，commit `b2dacb5`）。**全程未改任何 CMake 參數／engine／cache。** 以下為當時（沙箱開啟）的排查紀錄，留作環境問題證據。

<details><summary>（原）B-8-1 阻斷證據（2026-09-08）</summary>

**卡住位置**：RUNBOOK B-8-1 的 build 指令 `cmake -S engine -B engine/build-cuda -G Ninja … && cmake --build …` 在 **CMake configure 階段**卡死於 `-- Detecting C compiler ABI info`（C-ABI TryCompile probe），其後無任何輸出。

**錯誤原文（cmake stdout，到此停住）**：
```
-- The C compiler identification is MSVC 19.44.35228.0        ← MSVC 版
-- The CXX compiler identification is MSVC 19.44.35228.0
-- Detecting C compiler ABI info                                ← 卡在此
```
（GCC 版，因未先跑 vcvars64：`-- The C compiler identification is GNU 16.1.0` → 同樣卡在 `Detecting C compiler ABI info`。）

**根因（已實證，非猜）**：CMake 的 Ninja generator 把 workspace workdir 寫進 probe 的 `build.ninja` 成 **`D$:`** 驅動形式（`cmake_ninja_workdir = D$:/Nova2_home/AI研究/…/TryCompile-<id>/`）。`D$` **不是有效 Windows 驅動字母**（`Test-Path D$:\`＝False；`D$:\Nova2_home\AI研究`＝False；無任何 `subst`），ninja 解析不到該 workdir → **pre-launch 阻斷**（0 CPU、0 子程序、不產 `.exe`）。相對地，**真 `D:\` workdir 可用**：手動重放同一 link 指令（workdir＝真 `D:\…` probe 目錄、vcvars64 環境）**1.07 s 成功**、產出 `cmTC_b1239.exe`。

**已排除（全數實測）**：
| 變量 | 結果 |
|---|---|
| ninja 版本 | mingw64 1.13.2 **與** native VS 1.12.1 **都**卡（0 CPU、0 子程序） |
| 編譯器 | MSVC cl 19.44 **與** GCC 16.1.0 **都**卡在同一 probe |
| manifest 編碼 | CP950 **與** UTF-8（transcode 後）**都**卡 |
| 陳舊 `.ninja_lock` | 移除無效（ninja 重創） |
| build-msvc（A 側） | 同型 C-ABI probe **成功**（現 `CMakeScratch` 已空＝probe 通過後自清）→ 機制本身在此環境可用；**本卡為 build-cuda 獨有的 `D$:` workdir 形式** |
| 主 `build.ninja`／`rules.ninja` | **不存在**（configure 卡在 probe，還沒到生成主 manifest） |

**為何不自行修（依「不調 CMake 參數／不猜」紅線）**：
1. `D$` 非有效驅動字母 → 無法用 `subst` 建立它讓 ninja 解析。
2. 要改寫 `build.ninja` 的 workdir，但 ABI probe 是 CMake **configure 期間**自己跑、自己生成 manifest 後立即 spawn ninja——**無法在中間攔截**；且 CMake 每次 configure 都用新隨機 id 重跑 probe，pre-seed 無效。
3. 換 CMake generator（如 NMake/VS）＝改 build 參數，**踩紅線**（RUNBOOK 指定 `-G Ninja`）。

**待用戶決定（擇一）**：
- **(A)** 授權換 generator（`-G "NMake Makefiles"` 或 VS generator）重試 B-8-1（**偏離 RUNBOOK `-G Ninja`**，需明確放行）；
- **(B)** 授權在 configure 後、build 前，以 A 側 `transcode_ninja.ps1` 同型工具把**所有** `build.ninja`／`rules.ninja` 的 `D$:` workdir 改寫為真 `D:\`（含 probe）——但 probe 在 configure 中無法攔截，需改由「先讓 configure 跑過 probe」的方式（可能仍需 generator 變更）；
- **(C)** 在**非 CJK、非沙箱映射**的 build 目錄重試（但 workspace 全為 CJK 且沙箱 root＝`D:\Nova2_home\AI研究`，無可寫 ASCII 位置）；
- **(D)** 由用戶／更上層環境層面修正沙箱把 workspace 映射成 `D$:` 的問題（root 層面）。

**現況（當時）**：`build-cuda` 已清（wipe、0 殘留程序）；`build-msvc/bin` CPU 基線完整（98 支 exe）；未動任何 CMake 參數／engine／cache。

</details>

#### ⛔ B-8-2 阻斷證據（2026-09-09 10:07；真機載入 crash＝fork `--q4e-moe-vram` 未加 null-terminator，非超時／非 VRAM-fit）

**指令**（RUNBOOK B-8-2 原文，逐字執行）：`Start-Process engine\build-cuda\bin\q4e-server.exe` + `--model D:\AI_Model\Qwen3.8-Flash-Next\…-00001-of-00004.gguf --host 127.0.0.1 --port 8124 --api-key bkey --n-gpu-layers all --q4e-moe-vram 10 --ctx-size 2048 --parallel 1`，poll `/health` ≤300 s。

**結果**：`health=000` 一路到 300 s 期限（**非超時後成功、是啟動 ~6 s 即崩**）；`nvidia-smi` **0MiB / 24576MiB、No running processes**；無 `q4e-server` 程序、port 8124 空。

**錯誤原文**（`b8_2_server.err.log`，761 B／5 行；`b8_2_server.log`=0 B）：
```
0.00.030.551 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.039.861 I srv    load_model: loading model 'D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
0.04.589.074 I srv    load_model: q4e: live prefetch backend active (model=…-00001-of-00004.gguf)
0.04.681.335 I q4e: ngram prefetch ready (file=…-00002-of-00004.gguf, rows=320001536, row=90 B)
D:\Nova2_home\AI研究\推理工具\engine\common\common.cpp:1768: GGML_ASSERT(params.tensor_buft_overrides.back().pattern == nullptr && "Tensor buffer overrides not terminated with empty pattern") failed
```

**根因（source 證據，非猜）**：
1. `common_init_result`（common.cpp:1341）→ GPU offload 時 `common_q4e_moe_vram_apply`（:1346/:1293）。
2. `--q4e-moe-vram 10` → `llm_add_q4e_moe_vram_overrides(n_layers, 10, params.tensor_buft_overrides)`（:1311）；n_layers=48 → n_cpu=38。
3. → `llm_add_n_cpu_ffn_overrides(38, LLM_FFN_EXPS_REGEX, …)`（common.h:1144-1151）：`for(i<38) overrides.push_back({非null regex, cpu_buft})`——**只推 38 筆非 null pattern，loop 後未 push `{nullptr,nullptr}`**。
4. 緊接 `auto mparams = common_model_params_to_llama(params)`（:1349）→ **common.cpp:1768** `GGML_ASSERT(back().pattern == nullptr)` 對 38 筆（最後一筆 pattern 非 null）→ **fail → abort**。
5. 崩於 `common_q4e_moe_vram_check`（:1352，本該給「清晰 VRAM 錯誤」的 VRAM-fit 檢查）**之前**→故 crash 而非 runbook 預期的清晰 VRAM 錯誤。
6. fork 自測 `test-arg-parser.cpp:396-414`：`llm_add_q4e_moe_vram_overrides(48,10,ovs)` 後只 `assert(ovs.size()==38)`——**不驗 `ovs.back().pattern==nullptr`**，故測不出此不變式違反（size 不變式與 terminator 不變式互相遮蔽）。

**影響範圍**：任何 `k < n_layers` 皆崩（K=10 在此 48 層模型必崩）；`k ≥ n_layers`（n_cpu≤0）不觸發 override 故不崩——但改 K 違反「不改 K／placement」紅線，且改動 B-8 ON 設定語意。

**備案（RUNBOOK B-8-2 原文）**：超時/失敗 → 回報原文，**不自動降 K**；且紅線禁改 engine source。故 **B-8-2 ON 於本 engine build（commit b2dacb5 / build 38）無法照原文載入**。

**⬇ 交接 A 側 spec（自含；A 側執行，B 側零改 engine）⬇**

**目標**：修 `--q4e-moe-vram k<n_layers` 路徑未補 null-terminator 致 `common_model_params_to_llama`（common.cpp:1768）abort；讓 `--n-gpu-layers all --q4e-moe-vram 10` 能載入。

**改法（二選一；推薦 A）**：
- **A（最小、最局部）**：`engine/common/common.cpp` `common_q4e_moe_vram_apply()`（:1293），在 `return true;`（:1312）前補一行：
  ```cpp
      llm_add_q4e_moe_vram_overrides(n_layers, params.q4e_moe_vram, params.tensor_buft_overrides);
      if (!params.tensor_buft_overrides.empty()) {
          params.tensor_buft_overrides.push_back({nullptr, nullptr});  // null-terminate for common_model_params_to_llama
      }
      return true;
  ```
  此改法**不動** `llm_add_n_cpu_ffn_overrides`／`llm_add_q4e_moe_vram_overrides`，故 `test-arg-parser.cpp` 的 `ovs.size()==38` 斷言**不變**、不需改測。
- **B（改生成端）**：`engine/common/common.h` `llm_add_n_cpu_ffn_overrides()`（:1144-1151）for-loop 後補 `overrides.push_back({nullptr, nullptr});`——但此會把 `llm_add_q4e_moe_vram_overrides(48,10,ovs)` 的 size 變 39，需同步改 `test-arg-parser.cpp:401` `assert(ovs.size()==38)`→`39`（`38→28` 那筆 :413 同理 →29），較易漏。

**A 側重建**：`cmd /c '"…\vcvars64.bat" && cmake --build engine\build-cuda --target q4e-server'`（增量；build dir 已存在）。交付新 `q4e-server.exe`（build 號應 > 38）。

**B 側再驗**：收到新 exe → 重跑 **B-8-2 原文**（ngl=all＋K=10）→ 預期 `health=200 ≤~250 s`＋VRAM ≈20,0xx–21,0xx MiB＋log 有 model loaded/listening → 再走 B-8-3…B-8-7。**B 側不重做 B-8-1 前置、不改 K／placement。**

**回歸**：A 側修後建議跑 `test-arg-parser`（q4e-moe-vram override 生成段）確認 size 斷言仍綠（改法 A 應無變動）。

#### ⛔ B-8-2 重跑證據（2026-09-09 20:3x；build 39；fork 修復實機生效＋新阻擋：GPU 被占＋check 需 78,431 MiB）

**指令**：RUNBOOK B-8-2 原文逐字（build 39 exe；log 落 `b8_run2\s1_b82\`）。**結果**：`health=000` 全程 300 s（載入階段 server 自行退出）；**不再 common.cpp:1768 assert 崩→A 側 null-terminator 修復實機生效**；錯誤為 RUNBOOK 預期「清晰 VRAM 錯誤」形態，但數字異常（見下）。

**錯誤原文**（`b8_2_server.err.log` 全 9 行、關鍵 2 行）：
```
0.01.647.653 E cmn  common_q4e_m: common_q4e_moe_vram_check: --q4e-moe-vram k=10 does not fit in VRAM: CUDA0 needs 78431 MiB, only 23142 MiB free
0.01.649.682 E srv  llama_server: exiting due to model loading error
```
（log 前 4 行正常：verbosity / loading model / `q4e: live prefetch backend active` / `ngram prefetch ready (rows=320001536, row=90 B)` → 載入已過原崩潰點、抵達 VRAM-fit check。）

**事實（不猜）**：
1. **check 代碼路徑**：`common_q4e_moe_vram_check`（common.cpp:1319-1342）：required＝`dmds.model+context+compute`（:1334），出自 `common_get_device_memory_data`（fit.cpp:155→:29 `_impl`）——`mparams_copy=*mparams`（**含 build 39 的 38 筆 q4e CPU overrides＋null-terminator**）、`no_alloc`＋`LLAMA_LOAD_MODE_NONE` 載 model 後 `llama_get_memory_breakdown(ctx)` 逐 buft 加總（fit.cpp:75-97）→ **78,431 MiB 是（若 override 生效的）實際 placement 測量值**。
2. **`free=23,142 MiB` 為 `ggml_backend_dev_memory`（CUDA）值、WDDM 下樂觀**（同時刻 nvidia-smi used≈23,972 MiB→真實 free ≈600 MiB；GPU 被 harness 自家模型服務 PID 2992 `llama-server` --port 8080 -ngl 99 -c 150000 --no-mmap（9/9 19:25 啟動）占 ~24 GB）。
3. **即使 GPU 全空（24,576 MiB）：78,431 > 24,576 → B-8-2 原文（ngl=all+K=10）照原文過不了 VRAM-fit check**，且為 RUNBOOK 預期 ≈20,0xx–21,0xx MiB／§二 估 ≈20.6 GB 之 3.8×。
4. 清理驗證：無 q4e 殘留、VRAM 回基準（23,972 MiB）、port 8124 空。

**⬇ 交接 A 側 spec（自含；A 側執行，B 側零改 engine）⬇**

**目標**：解釋 ngl=all+K=10 之 VRAM-fit required＝**78,431 MiB**（build 39 實測）vs 預期 **≈20.6 GB** 之 3.8× 落差，並讓 B-8-2 原文能在 24 GB 3090 上載入（或重定 placement）。

**兩假說（A 側裁決；B 側不猜）**：
- **H1（override 未生效）**：`llm_add_q4e_moe_vram_overrides`→`llm_add_n_cpu_ffn_overrides`（common.h:1144、`LLM_FFN_EXPS_REGEX`）產出的 38 筆 per-layer pattern **未匹配本 GGUF 實際 expert tensor 名稱** → 全部 experts 留在 CUDA → required≈全 attn＋全 experts≈78 GB。驗證法：dump GGUF tensor 名稱（metadata-only、勿載權重）逐筆對 regex；或 `-lv debug` 跑載入看 `common_memory_breakdown_print` per-buft breakdown（fit.cpp:146）。
- **H2（估算錯）**：pattern 生效、K=10 placement 確實 ≈78 GB → 20.6 GB 估算錯 → 紅線設定 ngl=all+K=10 於 24 GB 卡**物理裝不下** → **重定 K 或 ngl（＝placement 改變＝A＋用戶決定；RUNBOOK／03B 紅線同步更新）**。

**A 側交付**：(a) required 組成之 per-buft breakdown（明文解釋）；(b) 若 H1→修 pattern 重構（build 40、交付新 exe）；若 H2→A＋用戶重定 B-8-2 設定＋新預期 VRAM；(c) 新預期 VRAM 數字（B 側驗證線）。

**B 側前置＋復驗**：(1) A 側交付到位（⏳ pending）；(2) **用戶釋放 GPU＝✅ 已完成**（2026-09-10 00:53 前：llama-server 已消失、nvidia-smi 0 MiB；原狀態 PID 2992 占 ~24 GB；B 側紅線絕不殺 `llama-server`）。**B 側已於空 GPU 重跑 B-8-2 原文＝⛔ FAIL（needs 78,768 MiB＞24,576；見上方「空 GPU 重跑證據」）→ 待前置 (1) A 側交付到位（build 40／重定 placement＋新預期 VRAM）→ 再重跑 B-8-2 原文** → `health=200`＋VRAM 合 A 側新預期 → 續 B-8-3…B-8-7。**B 側不調 K／placement。**

#### ⛔ B-8-2 空 GPU 重跑證據（2026-09-10 00:53；build 39；GPU 全空 24,576 MiB；RUNBOOK B-8-2 原文）

**指令**：RUNBOOK B-8-2 原文逐字（build 39 exe、ngl=all＋K=10；log 落 `b8_run3\s_b82\`→歸檔 `b8_evidence\step2c_rerun_b39_gnufree\`；輪詢迴圈於 server 自退時 early-break——server 已自行退出、health 永不再達 200、與 300 s 全輪詢邏輯等價）。**結果**：`health=000`（首輪 poll 503→server 自退→connection refused；自退於啟動後 ~1.6 s）；**fork 修復仍實機生效（無 common.cpp:1768 assert 崩、抵達 VRAM-fit check）**。

**錯誤原文**（`b8_2_server.err.log` 14 行、關鍵 1 行）：
```
0.01.641.788 E cmn  common_q4e_m: common_q4e_moe_vram_check: --q4e-moe-vram k=10 does not fit in VRAM: CUDA0 needs 78768 MiB, only 22884 MiB free
```
（log 前行正常：verbosity / CORS 警告 / `loading model` / `q4e: live prefetch backend active` / `ngram prefetch ready (rows=320001536, row=90 B)` → 載入已過原崩潰點、抵達 VRAM-fit check。）

**事實（不猜）**：
1. **跑前 nvidia-smi＝0 MiB used / 24,576 free（無任何 GPU 程序）**；跑後 VRAM 回 0、無 q4e 殘留、port 8124 空。
2. **needs 78,768 vs 20:3x 之 78,431（Δ≈+0.4%）**＝`no_alloc` placement 確定性測量、**與 GPU 占用無關** → 20:3x 結論於空 GPU 再確認：**B-8-2 原文（ngl=all+K=10）必 FAIL VRAM-fit check**（78,768＞24,576 全空；預期 ≈20.6 GB 之 3.8×）。
3. `free=22,884 MiB` 仍為 WDDM 樂觀值（`ggml_backend_dev_memory`）；實際全空總量 24,576 MiB。
4. **B-8-2 狀態＝⛔ 待 A 側 H1/H2 裁決**（交接 spec 自含於上一節）。前置：(1) A 側交付到位＝⏳ pending（git HEAD `97619a0` 無 A 側新 commit、無 build 40）；(2) 用戶釋放 GPU＝✅ 已完成。

#### ⛔ B-8-3 G5 FAIL（2026-09-12、build 42 GPU 復驗；待 A 側裁決）— 問題紀錄

**錯誤原文**（A-14.3 交付工具 `q4e-e2e`）：G0–G4 全 PASS（G1 health 200；G2 401；G3 200、completion_tokens=52；G4 SSE 11 行＋`[DONE]`），**G5 FAIL**：
```
G5 FAIL: layers_touched=38 (want layers=48)
Q4E_E2E_FAIL G5
```
（E2E_EXIT=1。依 RUNBOOK B-8-3 備案「任一 gate FAIL → 停止並回報（工具 log-dir 證據＋stats 原文）、不猜」→ **B-8 停於此、B-8-4~7 未啟動**。）

**執行條件**：`q4e-e2e --server engine\build-cuda\bin\q4e-server.exe --model <4 片 GGUF> --host 127.0.0.1 --port 8123 --api-key a141key --ctx-size 2048 --max-tokens 64 --health-timeout 480 --log-dir b8_run4\e2e`；temp=0、固定 prompt、空 GPU（開跑實測 0 MiB、無 q4e-server/llama-server）、無並行實例。

**本輪 B-8-0/1/2（全 ✅；B-8-2 兩筆偏差已記錄）**：
- B-8-0：`--version`＝`version: 42 (207fe88)`；exe 44,068,352 B；`--help` 含 `--q4e-moe-vram K`（MoE VRAM 快取預算、預設 10、0–999）／`--q4e-ngram-prefetch`。
- B-8-1：4 片總 111,334,654,784 B（10,946,624 / 49,859,583,136 / 49,376,141,504 / 12,087,983,520）與前一致；GPU 0 MiB；無殘留。
- B-8-2：`--q4e-moe-vram 10 --n-gpu-layers all --ctx-size 2048 --port 8124`；health 200、**冷載入 286 s**（偏差①＝超 ~250 s、與重啟後冷 page cache＋SATA 0.31–0.44 GB/s 一致、非備案停止條件）；VRAM **21,606 MiB**（偏差②＝A 側 fit 21,315、Δ+291 WDDM overhead；≤ free 22,884 → **VRAM-fit PASS**）；kill PID 11684 → VRAM 0 MiB、無殘留。證據 `b8_evidence/step3_rerun_b42/s_b82/`（`b8_2_server.err.log`：`q4e: live prefetch backend active`／`ngram prefetch ready (rows=320001536, row=90 B)`／`model loaded`／listening 8124）。

**G5 交叉證據（全原文、零判斷）**：
1. GPU run stats 原文（`b8_evidence/step3_rerun_b42/s_b83_e2e/stats.json`）：`{"layers":48,"touches":21087,"hits_vram":7721,"hits_ram":5568,"hits_ssd":6621,"hit_rate_vram":0.3661,"hit_rate_ram":0.264,"hit_rate_ssd":0.314,"prefetch_issued":1559,"occ_vram_avg":15.8,"occ_ram_avg":108.2,"placement_events":28060,"io_bytes_estimate":86200320000,"layers_touched":38,"hook_skips_device_ids":0,"q4e_moe_vram":10,"q4e_prefetch":1,"ngram_prefetch":2,"ngram_hints":60,"ngram_rows_warmed":880,"ngram_rows_dropped":0,"ngram_bytes_warmed":79200,"prefetch_io_issued":6705,"prefetch_io_bytes":20977356800,"prefetch_io_completed":6704,"prefetch_io_dropped":0,"prefetch_events_dropped":1470,"prefetch_overlap_hits":13142,"prefetch_overlap_ratio":0.5815044247787611,"hook_skips_weights_device":650}`
2. 工具 G5 gate 源碼（`engine/tools/q4e-e2e/main.cpp:723-724`）：`if (layers_touched != layers) { bad.push_back("layers_touched=" + ... + " (want layers=" + ...) }`——gate＝「touched 層＝全部層」。G5 其餘項（q4e_prefetch=1／touches>0／layers>0／prefetch_io_issued>0／overlap_hits>0）本次全綠。
3. A 側 CPU 基準（`evidence_a141/stats.json`、A-14.1、ngl=0）：`layers_touched=48`（＝layers）→ gate 以 CPU 基準校準。
4. 舊 fork v1 載入 log（B-8-2 證據、同 K=10）：`llm_add_n_cpu_ffn_overrides(38, ...)`＝38 層 FFN 留 CPU 側；48 − 10（K）＝38（數字吻合、非因果斷言）。
5. GPU run `hook_skips_weights_device=650`（CPU 基準為 0）；A-9 設計（A 側表）：cb_eval 對 device-resident ids 不做 sync D2H、只簿記 CPU 層。
6. 回歸門（本輪可測項）：VRAM 歸零 ✅；冷載入 286 s（偏差①已記錄）；G3/G4 ✅；**B-8-4~7（t/s／overlap／ngram／oracle）未跑 → 回歸門未完成**。

**問 A 側（待裁決）**：`layers_touched=38` 於 GPU ngl=all＋K=10 下是否為預期值（10 層專家 VRAM-resident→不參與 CPU 簿記→48−10=38、與事實 4/5 一致），抑或為回歸？若為預期值 → `q4e-e2e` G5 是否應改為 placement-aware gate（如 `layers_touched == layers −（device-resident 層數）`）？B 側不改 engine/ 源碼、test、交付工具。

#### ✅ A 側裁決（2026-09-12、G5 修復已交付、B 側重跑完成 G0–G6 全 PASS）

**裁決＝`layers_touched=38` 為預期值（非回歸）＋G5 改 placement-aware gate（已交付）**。source 證據：`cb_eval`（`q4e-expert-cache.cpp:499-504`）對 device-resident 層 `note_hook_skip()` 不 `touch()`→`layers_touched` 恒只計 CPU 側層；`hook_skips_weights_device>0`⇔存在 VRAM-resident 層。全細節＋兩處對建議 snippet 的偏離理由（純 K 判斷式會誤殺 A-14.1 CPU 基線、且 B-9 K=5000 必踩同 gate）＝[b8_evidence/stepA_g5/adjudication.md](b8_evidence/stepA_g5/adjudication.md)。**交付件**：`engine\build-msvc\bin\q4e-e2e.exe`（SHA-256 `EDD6E0BC…F0610F`、367,104 B、2026-09-12 14:28 build；`q4e-server.exe` build 42 未動）；A 側離線驗證：B-8-3 stats（lt=38、K=10、skips=650）→expected 38 **PASS**＋A-14.1 CPU 基線（lt=48、skips=0）→expected 48 **PASS**。**B 側下一步**：重跑 B-8-3 原文（新 `q4e-e2e`）→預期 G0–G6 全 PASS（G5 印 `layers_touched=38/48 (expected 38)`）→續 B-8-4~7；任一 gate 仍 FAIL→回報原文。

**B-8 狀態**：⏳ B 側重跑完成（2026-09-12）：B-8-3 ✅ G0–G6 全 PASS → **⛔ B-8-4 硬線／B-8-6 門 1 FAIL（ON 1.6366 ＜ OFF 2.3002 t/s、且 ON ＜ 1.95）→ ✅ A 側裁決完成（2026-09-12：v4.1 warm 策略淨負向；修復 B-8-4-F1＝min-freq warm gate、build 47 已交付；門 3 裁決＝維持 GPU 量）待 B 側 GPU 重跑（B-8-2 加 `--q4e-warm-min-freq 2`）**；門 2 overlap 0.93 PASS；門 3 ngram 未量（A 側裁決＝補測）；B-8-5 ✅；B-8-7 收尾 ✅；證據 `b8_evidence/step4_rerun_g5fix`＋裁決 [b8_evidence/stepA_b84/adjudication.md](b8_evidence/stepA_b84/adjudication.md)。

#### ⛔ B-8-4 硬線／B-8-6 門 1 FAIL（2026-09-12 重跑、新 `q4e-e2e`）— 問題紀錄（✅ 已裁決＝見下方「A 側裁決」節；待 B 側以 build 47＋`--q4e-warm-min-freq 2` 重跑）

**重跑條件**：新 `q4e-e2e.exe`（SHA-256 `EDD6E0BC…F0610F`、B 側實測一致）；`q4e-server.exe` build 42 未動（commit `207fe88`）；開跑 GPU 0 MiB 空；GGUF 4 片 111,334,654,784 B。

**B-8-3 重跑 ✅（e2e、port 8123、工具自起/殺 server PID 16180、14:40:46→14:49:44、exit 0）**：

```text
G0 PASS: no residual server process; port 8123 free
START pid=16180
G1 PASS: /health 200
G2 PASS: 401 without auth
G3 PASS: 200 completion_tokens=52
G4 PASS: SSE 11 data lines + [DONE]
G5 PASS: q4e_prefetch=1 touches=20522 layers_touched=38/48 (expected 38) prefetch_io_issued=6610 prefetch_overlap_hits=12715
G5 ungated: prefetch_io_bytes=20696166400 prefetch_io_completed=6609 prefetch_io_dropped=0 prefetch_events_dropped=1533 prefetch_overlap_ratio=0.5784804367606915 hook_skips_weights_device=650 hits_vram=7449 hits_ram=5385 hits_ssd=6536
G6 PASS: own child pid=16180 gone after 7.2s
Q4E_E2E_ALL_PASS
```

G5 行與 A 側裁決預測逐字一致（`expected 38`）。

**B-8-4 ON 相（q4e-server 8124、PID 7192、B-8-2 原文 args）**：載入 14:51:07→14:57:19（**372 s**、偏差①）；載入後 VRAM 21,606 MiB；256 warm-up 丟棄後計量 256：
- `time_total=156.418471`、`completion_tokens=256` → **ON 1.6366 t/s**；server log 同 run：`eval time = 154358.33 ms / 256 tokens ( 605.33 ms per token, 1.65 tokens per second)`。
- **B-8-5** 64 run → `b8_5_q4e.json`（52 tok）。kill 7192 → VRAM 0 MiB ✅。
- **`/q4e/stats`（256 計量 run 後、全欄位原文）**：

```json
{"layers":48,"touches":194000,"hits_vram":88920,"hits_ram":81470,"hits_ssd":22090,"hit_rate_vram":0.4584,"hit_rate_ram":0.4199,"hit_rate_ssd":0.1139,"prefetch_issued":2020,"occ_vram_avg":15.8,"occ_ram_avg":112.4,"placement_events":227247,"io_bytes_estimate":698102784000,"layers_touched":38,"hook_skips_device_ids":0,"q4e_moe_vram":10,"q4e_prefetch":1,"ngram_prefetch":2,"ngram_hints":512,"ngram_rows_warmed":3968,"ngram_rows_dropped":0,"ngram_bytes_warmed":357120,"prefetch_io_issued":10091,"prefetch_io_bytes":31585075200,"prefetch_io_completed":10091,"prefetch_io_dropped":0,"prefetch_events_dropped":814,"prefetch_overlap_hits":182196,"prefetch_overlap_ratio":0.9299510004083299,"hook_skips_weights_device":5160}
```

**B-8-4 OFF 相（oracle 8125、`-ngl all -ncmoe 38`、PID 12984）**：載入 15:07:27→15:12:40（**313 s**、偏差①）；載入後 VRAM 21,952 MiB；256 warm-up 丟棄後計量 256：
- `time_total=111.294254`、`completion_tokens=256` → **OFF 2.3002 t/s**；oracle log 同 run：`eval time = 108765.54 ms / 256 tokens ( 426.53 ms per token, 2.34 tokens per second)`。
- **B-8-5** 64 run → `b8_5_oracle.json`（52 tok）。kill 12984 → VRAM 0 MiB ✅。

**B-8-4 硬線：ON 1.6366 t/s ＜ OFF 2.3002 t/s → ⛔ FAIL（−28.8%）**。上輪 B-4（v1）＝ON 1.5692 / OFF 1.7700（−11.3%）→ 差距擴大。tg_3s 3-秒窗：ON 在 0.40–5.23 t/s 振盪（例：13.33.198→0.61、13.36.264→4.57、13.20.369→0.93），OFF 窗多在 2.19–4.06（僅數據、不猜因）。prompt 側（64 run、20 新 token）：ON `prompt eval time = 30704.80 ms / 20 tokens (1535.24 ms per token)` vs OFF `9535.18 ms / 20 tokens (476.76 ms per token)`（3.2×、僅數據）。

**B-8-5 oracle 對答案 ✅**：64 run 同一 body（temp 0）雙方 `completion_tokens=52`、`finish_reason=stop`、`prompt_tokens=62`；`choices[0].message.content` 與 `reasoning_content` **逐字相同**（content＝"A cache is a small, fast storage layer that temporarily holds frequently accessed data to speed up future requests."；reasoning_content＝"We need to answer user's request: \"What is a cache? Explain in one sentence.\" Need final one sentence. Keep concise.\n"）。

**B-8-6 門評估**：

| 門 | 實測 | 標準 | 結果 |
|---|---|---|---|
| 1 暖 t/s（ON）≥ OFF 且 ≥ 1.95 | ON 1.6366 / OFF 2.3002 | ≥ OFF 且 ≥ 1.95 | **⛔ FAIL**（兩者皆未達） |
| 2 預取重疊 ≥80% | `prefetch_overlap_ratio=0.92995`、`prefetch_io_issued=10091＞0` | ≥0.8 | ✅ PASS |
| 3 N-gram 掉落率＜1% | **未量**——停於 B-8-4 卡住規則（門 3 需 `q4e-ngram accept` 額外 GPU run） | ＜1% | — 待 A 側裁決是否補測 |

**B-8-7 收尾回歸 ✅**：無 `q4e-server`／`llama-server` 殘留；VRAM 0 MiB；GGUF 4 片 111,334,654,784 B 完好。

**偏差（皆非阻斷、記錄）**：
1. 載入時間 ON 372 s／OFF 313 s（與 B-8-2 之 286 s 同類；超出驗收 #8「≤ ~250 s」期望；環境性＝SATA 0.31–0.44 GB/s 冷 page cache、同 B-8-2 偏差①）。
2. health poll 期限 300 s（RUNBOOK 原文）→ 480 s（為容納實測 372 s 載入；僅安全方向、停止條件不變）。
3. ON 相 server＝B-8-2 原文重起（B-8-2 實例已在 B-8-2 收尾 kill；RUNBOOK 所指「B-8-2 已起並 warm-up 之 8124 server」即該實例）。
4. B-8-6 門 3（`q4e-ngram accept`）未量：依卡住規則停於 B-8-4 硬線 FAIL。

**兩端 log tail 30 行**（RUNBOOK 備案要求）＝[b8_evidence/step4_rerun_g5fix/b8_run5/b8_4_on_server.err.log](b8_evidence/step4_rerun_g5fix/b8_run5/b8_4_on_server.err.log)（tail 30＝256 計量 run tg  progression＋eval block）與 [b8_4_off.err.log](b8_evidence/step4_rerun_g5fix/b8_run5/b8_4_off.err.log)；全證據（e2e console/stats/SSE/gen、b8_4/b8_5/b8_6 JSON、pid、body）＝`b8_evidence/step4_rerun_g5fix/b8_run5/`。

**問 A 側（待裁決）**：
1. ON 1.6366 ＜ OFF 2.3002 t/s（−28.8%）且 ON ＜ 1.95 → B-8-4 硬線／B-8-6 門 1 FAIL 如何裁決？（A-9 同步 D2H 移除＋A-14.1 真後端皆已生效：`q4e_prefetch=1`、`prefetch_io_issued=10091`、overlap 0.93、`layers_touched=38`、`hook_skips_weights_device=5160`。）
2. 是否補跑 B-8-6 門 3（`q4e-ngram accept` GPU 量測）？
3. 是否需要 A 側 profiling／進一步證據（ON tg_3s 低窗 0.40–0.61 t/s 段、prompt 側 1535 vs 477 ms/tok）？B 側照 RUNBOOK 原文執行。

#### ✅ A 側裁決（2026-09-12、B-8-4-F1 修復已交付 build 47、待 B 側 GPU 重跑）

**裁決＝v4.1 warm 策略淨負向（非 placement／kernel／環境）**：prev-token reuse warm 把**整個 distinct 工作集各 cold-warm 一次**（實測 `io_issued=10091`×3.13 MB＝**31.6 GB 純盤 I/O**、`io_dropped=0`＝全做完），其中一次性（one-shot）expert 對 compute **零幫助**（OFF 也只讀一次、3.13 MB 冷讀即棄）→ 搶共享 SATA（~400 MB/s）帶寬＋把 hot core 頁擠出 page cache → compute 關鍵路徑飢荒 → ON 被自己的 warm thread 拖到 1.6366 ＜ OFF 2.3002。獨立佐證：A-14.3 CPU 同二進位 A/B（ON 0.845/OFF 1.682，warm thread −50%）；A-14.1 CPU（io 40.8 GB、drops 4181）；OFF 2.30 t/s≈367 MB/s≈SATA 滿（OFF 已盤界，ON 加盤流量必輸）。**全證據鏈＝[b8_evidence/stepA_b84/adjudication.md](b8_evidence/stepA_b84/adjudication.md)**。

**修復 B-8-4-F1（min-freq warm gate）**：prev-token reuse warm 只 warm「選中 ≥N 次」的 expert；新旗標 **`--q4e-warm-min-freq N`**（預設 1＝舊行為凍結；**B 側重跑用 N=2**＝跳過一次性 expert、保留重複使用者）。stats 新增 `prefetch_warm_gated`（應 >0＝gate 生效證據）＋`prefetch_min_freq`（additive，G0–G6 不受影響）。**A 側驗證**：build **47**（`D:\q4e_build` 鏡像，mirror HEAD `5efd937`；`q4e-server.exe` 44,072,448 B；`--version`＝`0.3.0-dev (build 47, commit 5efd937)` MSVC 19.44）；單元測試全綠＝test-q4e-expert-warm **195 checks**（含新 `test_freq_gate`）＋test-q4e-expert-cache **159**＋test-arg-parser **all OK**；CPU A/B run 1（`b8_evidence/stepA_b84/cpu_ab/`）＝**ON（min_freq=2）0.4916 t/s**（`warm_gated=14,328` 生效、overlap 0.8825、io 39.96 GB/drops 0；distinct 12,743 vs A-14.1 13,019＝此模型 one-shot tail 小、gate 省 I/O ~2%＝必要但二階）；**OFF 已補測完成**（`cpu_ab3_off`、重啟後）＝**OFF 0.882 t/s**（290.3 s/256、q4e 關）；run 2（`cpu_ab2/`、重啟後）ON warmup 900 s timeout（<0.285 t/s）；A-14.3（9/8 淨機）OFF 1.682/ON 0.845 → **CPU 全數據每組 ON<OFF（含 gate 後）＋warm thread 把 ON 推過超時邊界（OFF 無此問題）→ gate 單獨大概率不足以翻轉 ON<OFF；v4.1 warm backend 在本硬體（工作集≫RAM）淨負向→v2 備案（warm 預設停用／I/O rate cap／真 lookahead）大概率需要**；CPU 為弱儀器（跨 run ±30% 雜訊）但方向一致。**B 側 GPU 仍為交付門（最終裁決）**。**階段目標（新、9/12 夜）**：build 47／min-freq gate 不改照跑；**B-8 完整跑完 → 確定命中率（快層命中率 `(hits_vram+hits_ram)/touches`＋SSD miss 率 `hits_ssd/touches`＋`prefetch_overlap_ratio`，ON 相 stats 原文）→ 才據此判斷 V5 是否有做的必要**（決策表＝adjudication §5.0：高命中率+ON<OFF＝v4.1 天花板→建議 V5；低命中率＝cache 失效→建議 V5；高命中率+ON≥OFF＝V5 暫不急需）；命中率＋ON/OFF 兩組數齊前不下 V5 結論。**⚠️ 機器風險**：9/12 **07:30 PM BSOD**（`C:\Windows\Minidump\091226-77640-01.dmp`）終止 run 1 OFF 相＋9/10 09:00 BSOD（0x133）＝**2 次 BSOD/2 日**；GPU 現被 harness `llama-server`（Qwen3.8-27B）佔 **23,608/24,576 MiB** → **B 側 GPU 實跑需先釋放（人類處理；A 側絕不 kill）**；建議低負載時段＋每步落盤＋中斷即原文回報。

**對 B 側三問之裁決**：
1. **B-8-4 硬線 FAIL＝修復後重跑**（新 exe＋`--q4e-warm-min-freq 2`）；若 GPU 仍 ON＜OFF → 原文回報、A 側 v2 迭代（hot-set re-warm／rate cap／min_freq 掃值），不調參。
2. **門 3＝維持量**（RUNBOOK 原定義＝B 側 GPU `q4e-ngram accept` 原文回報）；A-14.4 CPU accept PASS（max drop +0.22%）為預驗證據、不替代 GPU 量。
3. **A 側 profiling 已做**＝根因診斷（adjudication.md §二；數據點全列）；無需 B 側補資料。

**B 側重跑指示（零判斷；B-8-1~B-8-7 全原樣＋唯一新增）**：
1. **B-8-1 重build**（RUNBOOK 原指令）→ 新 `q4e-server.exe`；記 `--version`（build 47）＋SHA-256＋`--help` 應含 `--q4e-warm-min-freq N`。
2. **B-8-2 載入**：RUNBOOK 原參數＋**`--q4e-warm-min-freq 2`**（唯一新增旗標；health 紀律、VRAM 預期 21,315±WDDM 不變——本修復不改 placement）。
3. **B-8-3**：原樣（新 `q4e-e2e`；stats 新增 2 欄為 additive）。
4. **B-8-4**：原樣（ON≥OFF 硬線；2×256 tokens 紀律）。
5. **B-8-5**：原樣。
6. **B-8-6**：門 1＋門 2 原樣；**門 3＝`q4e-ngram accept` GPU 量**（原文回報）。
7. **B-8-7**：原樣。
8. **證據必含**：新 exe SHA-256、`b8_6_stats.json` 全欄位（含 `prefetch_warm_gated`＞0＋`prefetch_min_freq=2`）、ON/OFF t/s＋log 尾段。
9. **紅線不變**：placement `--n-gpu-layers all --q4e-moe-vram 10`；不並行雙實例；kill 只按 PID；**絕不殺 `llama-server`**；B 側不改 engine source。

### B-9（v5）復驗（三層全 GPU MoE；待 Task A A-15.0~A-15.7 完成）

| 項 | 名稱 | 狀態 | 內容／門 |
|---|---|---|---|
| **B-9**（v5） | **v5 復驗（三層全 GPU）**：照 RUNBOOK 零判斷（build＋載入 v5 placement＋oracle 對答案＋L1/L2/L3 命中分佈＋t/s＋回歸門） | **⏳ 待 A-15.0~A-15.7（Task A）完成** | **主門**：ON 暖 t/s **≥ 2× B-8 實測 v4.1 ON**（M≤R 時，目標 **≥8**）；**L1 hit ≥80%**（M≤R 時）；**oracle 對答案**（48 層全 GPU＋pool，逐層比對）；**VRAM-fit**（N=5000≤24 GB，過大跳錯）；回歸門＝e2e 5 閘＋冷載入≤~250 s＋VRAM 歸零。**設定**：`--n-gpu-layers all --q4e-moe-vram 5000 --q4e-moe-ram 4000`（B-9 可調 N/N2 找上限）。**FAIL＝停下回報原文，不調參。** |

> **v5 目標數字（對照用）**：v4.1 實測 2–4 t/s（v1 錨 1.95）→ v5 目標 **≥8 t/s**（M≤R≈104/層 時，L1 全命中、48 層全 GPU 算）。M 由 B-8 量出（`/q4e/stats` distinct 專家/層/輪）。

### B-8 上下文（v4 基線數字；對照用）

| 項 | v4 實測 | 來源 |
|---|---|---|
| ON / OFF 暖 t/s（256 tok） | **1.5692 / 1.7700**（B-4 FAIL −11.3%） | [已完成工作.md §三](已完成工作.md) |
| 冷載入 | ~206 s；VRAM 16,724 MiB | B-2 |
| touches（B-3） | 73,494（鉤子活） | B-3 |
| hits 分佈（B-6） | vram 43.5% / ram 43.1% / ssd 13.3%；io 1.24 TB（簿記） | git `24f11f7` 之 `b6_stats.json` |
| v1 暖基準（#3′ 第二線） | **1.95 t/s**（M=512 上界列） | `legacy_v1/evidence/bench_noprefetch.log` |

## 二、Task B 環境（已實測）

| 項目 | 狀態 | 說明 |
|---|---|---|
| **GPU** | ✓ | RTX 3090 **24 GB**；舊配置 ngl=10 實測 **16,724 MiB**；新配置（ngl=all+K=10）＝**build 42：A 側 fit 21,315／B 側實測 21,606 MiB（2026-09-12、Δ+291 WDDM overhead、fit PASS）**（A 側 fit `no_alloc` 投影＝model 20,106.45 on CUDA0＋ctx/compute ≈1,208.6；build 39 之 78,431→78,768 MiB 已裁決＝**H1 override no-op**、非 placement 問題）；收尾歸零；**9/10 實測：llama-server（PID 13392）占 23,680/24,576 MiB＝重跑前需用戶釋放** |
| **CUDA toolkit（nvcc）** | ✓ | **v13.3**（nvcc V13.3.73；driver 610.47） |
| **CPU / RAM / SSD** | ✓ | i7-9700 8C/8T；RAM ~20 GB；SATA 0.31–0.44 GB/s |
| **GGUF** | ✓ | `D:\AI_Model\Qwen3.8-Flash-Next\`（4 片 111,334,654,784 B ≈103.69 GiB Q4_K_XL） |
| **oracle** | ✓ | `D:\llama\llama-server.exe`（fork build 10685；**僅二進位**） |
| **MSVC** | ✓ | VS 2022 BuildTools（vcvars64.bat）—CUDA build 用 |
| **harness 注意** | ⚠ | harness 自身的模型服務＝`D:\llama\llama-server.exe` 常駐；**殘留檢查只配對 `q4e-server`，絕不殺 `llama-server`**。**9/9 實測：resident 時實際占 ~24 GB VRAM**（PID 2992、-ngl 99 -c 150000 --no-mmap、port 8080、19:25 啟動）→ **B-8-2 無法與之並行→需用戶先釋放**；2026-09-10 00:53 前曾消失（空 GPU 重跑＝78,768 FAIL）；**9/10 12:4x 後又回來（PID 13392、占 23,680/24,576 MiB，本 session 實測）→ 空 GPU 條件需用戶再次釋放**；**2026-09-12：機器已重啟（driver 616.92）、llama-server 不在、開跑實測 0 MiB＝GPU 釋放 ✅** |

## 三、Task B 約束

| 約束 | 決定 |
|---|---|
| **placement 紅線** | `--n-gpu-layers all`（attn/router/shared 恆 VRAM）＋`--q4e-moe-vram K`（**K＝MoE VRAM 快取預算**，預設 10、範圍 0–999、**VRAM 裝不下→跳錯**、K 可設定）；舊 `-ncmoe/-cmoe/-ncffn` 已連 core 移除（01 §六） |
| **build 旗標紅線** | CMake 必帶 `-DGGML_CPU_REPACK=OFF`＋`-DBUILD_SHARED_LIBS=OFF`；build 環境＝`vcvars64`＋cl（PATH 上的 MinGW 會搶編譯器） |
| **並行** | 永不過兩個模型實例同時載入；先 kill server 再 in-process 載 |
| **零判斷** | 照 RUNBOOK_B_v3 執行；每步＝指令＋預期輸出＋備案 |
| **失敗回報** | **錯誤原文＋證據**（不猜、不繞） |
| **量測紀律** | temp=0、固定 prime prompt、warmup 32；A/B 交錯（A B A' B'）＋tail 窗（ramp law） |
| **日誌** | PS Tee 輸出＝UTF-16LE（讀時 BOM sniff）；證據全保留 |
| **C++ 工具** | B-8 起改用交付包內的 `q4e-bench`／`q4e-e2e`（取代 legacy Python 工具＋手動 curl） |

## 四、指針

- B-0～B-7＋B9-v4 報告（全完成）：[已完成工作.md §三](已完成工作.md)
- v1 基準數字（冷載入/暖 t/s/命中分佈/ngram）：[已完成工作.md §一](已完成工作.md)
- B-8 執行手冊：[RUNBOOK_B_v3.md](RUNBOOK_B_v3.md)
