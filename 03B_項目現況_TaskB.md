# 推理工具 — 項目現況 · Task B（有 GPU，較笨 AI）

> 更新：2026-09-10（**本檔只留未完成項目**；已完成工作→[已完成工作.md](已完成工作.md)；commit 全錄→`git log --oneline`）
> 姊妹檔：[03A_項目現況_TaskA.md](03A_項目現況_TaskA.md)
> 關聯：[02_實作項目工作規劃.md](02_實作項目工作規劃.md)；[RUNBOOK_B_v3.md](RUNBOOK_B_v3.md)

## 〇、Task B 的角色

**有 GPU 可用、較笨 AI**：照 RUNBOOK **零判斷執行**（每步＝指令＋預期輸出＋備案）：CUDA build＋真機載入＋oracle 對答案＋驗收＋§9 回報。**失敗回報格式＝錯誤原文＋證據，不猜。**

## 一、未完成項目

| 項 | 名稱 | 狀態 | 內容／門 |
|---|---|---|---|
| **B-8**（v4.1） | **復驗（A-12 交付後）**：照 [RUNBOOK_B_v3.md](RUNBOOK_B_v3.md) 零判斷 | **✅ B-8-1 build 完成 → ⛔ B-8-2 BLOCKED（fork bug）→ ✅ A 側修復已交付（2026-09-09）**：A 側依改法 A 修復（`0ccc16c`）＋重構 **build 39**＋test-arg-parser 全綠；新 exe 在 `engine\build-cuda\bin\q4e-server.exe`。**B-8-2 重跑（2026-09-09 20:3x，build 39）＝⛔ FAIL（VRAM-fit 清晰錯誤）**——fork 修復實機生效（不再 assert 崩、已抵達 VRAM-fit check 給清晰錯誤＝RUNBOOK 預期形態），但兩個獨立阻擋：① GPU 被 harness 自家模型服務（PID 2992 `llama-server` port 8080）占用 ~24 GB（絕不殺紅線→**待用戶釋放**）② check 實測 ngl=all+K=10 placement 需 **78,431 MiB** ＞ 全空 GPU 24,576 MiB、且為預期 ≈20.6 GB 之 3.8× → **B-8-2 空 GPU 重跑（2026-09-10 00:53，build 39）＝⛔ FAIL 再確認**：`CUDA0 needs 78768 MiB, only 22884 MiB free`（WDDM 樂觀值；全空 24,576；vs 20:3x 78,431 Δ≈+0.4%＝確定性 placement 測量、與 GPU 占用無關）→ **原文必 FAIL，⛔ 待 A 側 H1/H2 裁決（spec 自含，見下方兩節證據）；前置②用戶釋放 GPU＝✅ 已完成（llama-server 已消失、VRAM 0 MiB）**。B-8-1 沙箱映射根因已解（`danger-full-access` 後 build EXIT=0，`q4e-server.exe` 42 MB commit `b2dacb5`）；B-8-0 全綠。**B-8-2 真機載入 crash（非超時）**：`common.cpp:1768 GGML_ASSERT(...Tensor buffer overrides not terminated with empty pattern) failed`——`--q4e-moe-vram 10` 經 `common_q4e_moe_vram_apply`(common.cpp:1311)→`llm_add_n_cpu_ffn_overrides(38,...)`(common.h:1144) 只推 38 筆非 null pattern、**未加 null-terminator**，緊接 `common_model_params_to_llama`(common.cpp:1349) 斷言 `back().pattern==nullptr` 即崩；fork 自測(test-arg-parser.cpp:400) 只驗 `size()==38` 故遮蔽此不變式。**影響所有 k<n_layers（K=10 必崩）**，且崩於 VRAM-fit check(common.cpp:1352) 之前→非預期的「清晰 VRAM 錯誤」而是 crash。VRAM 0MiB、無程序、port 8124 空。**備案：不降 K、不改 engine source→待用戶/A 側決定**（最小修法：`common_q4e_moe_vram_apply` 回傳前補 `{nullptr,nullptr}`）。下方 B-8-1／B-8-2 阻斷為問題紀錄 | **重驗 v4 三門**＋回歸門：(1) **B-4 硬線**：暖 t/s **ON ≥ OFF**（A-9 已移除 sync D2H 根因）；(2) **#3′**：ON ≥ OFF 且 **ON ≥ 1.95 t/s**（A-11 拍板線）；(3) **#4 重疊**：`prefetch_overlap_ratio` **≥80%**（A-14.1 提供欄位；定義＝觸碰前已完成預取的 touches 佔比）＋`prefetch_io_issued>0`；**#5 N-gram＝A-14.4 ✓ 可量**（`5b09765`／`c92606e`；CPU 預驗 PASS max drop +0.22%；B 側 GPU 量）；回歸門＝冷載入≤~250 s＋VRAM 歸零＋oracle 對答案＋e2e 5 閘。**工具**：`q4e-bench`（ON/OFF t/s＋JSON）＋`q4e-e2e`（取代 curl 手動步）。**設定**：ON＝q4e `--n-gpu-layers all --q4e-moe-vram 10`（K=10 預設、**K 可設定**、B-8 可量 12/13 找上限）；OFF＝oracle（上游原生旗標）`-ngl all -ncmoe 38`（B-8-0 先驗 oracle 有 `-ncmoe`；若無→q4e `--no-q4e-prefetch` 同 placement）。**FAIL＝停下回報原文，不調參。** |

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
| **GPU** | ✓ | RTX 3090 **24 GB**；舊配置 ngl=10 實測 **16,724 MiB**；新配置（ngl=all+K=10）A 側估算 **≈20.6 GB**（build 39 VRAM-fit check 實測 required＝**78,431→78,768 MiB**（空 GPU 重跑再確認）、3.8× 落差→見下方 B-8-2 重跑／空 GPU 重跑證據、待 A 側）；收尾歸零 |
| **CUDA toolkit（nvcc）** | ✓ | **v13.3**（nvcc V13.3.73；driver 610.47） |
| **CPU / RAM / SSD** | ✓ | i7-9700 8C/8T；RAM ~20 GB；SATA 0.31–0.44 GB/s |
| **GGUF** | ✓ | `D:\AI_Model\Qwen3.8-Flash-Next\`（4 片 111,334,654,784 B ≈103.69 GiB Q4_K_XL） |
| **oracle** | ✓ | `D:\llama\llama-server.exe`（fork build 10685；**僅二進位**） |
| **MSVC** | ✓ | VS 2022 BuildTools（vcvars64.bat）—CUDA build 用 |
| **harness 注意** | ⚠ | harness 自身的模型服務＝`D:\llama\llama-server.exe` 常駐；**殘留檢查只配對 `q4e-server`，絕不殺 `llama-server`**。**9/9 實測：resident 時實際占 ~24 GB VRAM**（PID 2992、-ngl 99 -c 150000 --no-mmap、port 8080、19:25 啟動）→ **B-8-2 無法與之並行→需用戶先釋放**；**2026-09-10 00:53 前已消失（nvidia-smi 0 MiB、無程序）→ 空 GPU 重跑條件已滿足** |

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
