# 推理工具 — 項目現況 · Task A（無 GPU，較聰明 AI）

> 更新：2026-09-02（v4：**C++ 唯一實現**；A-3 ✓、A-4 收尾中、A-7 重整動工）
> 姊妹檔：[03B_項目現況_TaskB.md](03B_項目現況_TaskB.md)（GPU 側）
> 關聯：[02_實作項目工作規劃.md](02_實作項目工作規劃.md)；[01_架構系統資料.md](01_架構系統資料.md)

## 〇、Task A 的角色

**無 GPU 可用、較聰明 AI**：設計＋寫碼（**q4e 核心 C++**）＋CPU build＋C++ 測試＋交付包。
**v4 新增**：C++ 化初始化（A-7）＝把進行到一半的有效工作全部落到 C++、移除 Python 合約層、整理目錄。
全部工作可在無 GPU 環境完成。判斷權全在 A 側。

## 一、Task A 環境（已實測）

| 項目 | 狀態 | 說明 |
|---|---|---|
| **MSVC（VS 2022 BuildTools）** | ✓ | vcvars64.bat 在位（cl 不在 PATH→build 時 invoke） |
| **CMake** | ✓ | 4.4.2 |
| **Ninja** | ✓ | 1.13.2 |
| **MinGW g++（WinGet WinLibs）** | ⚠ | **在 PATH 上，會搶 CMake 的編譯器**→ **紀律：build 必包在 `vcvars64.bat` 環境**（MSVC 14.44.35207 選中） |
| **CUDA toolkit** | ◐ | v13.3（nvcc 在 PATH）—**A 端不使用**（無 GPU）；B 端 build 用 |
| **CPU / RAM** | ✓ | i7-9700 8C/8T（無 HT）；RAM 可用 ~20 GB |
| **SSD（D 槽）** | ✓ | SATA 0.31–0.44 GB/s（GGUF 所在） |
| **GPU** | **✗（A 端）** | GPU 工作全歸 Task B（見 03B） |
| **GGUF（檔案在）** | ✓ | `D:\AI_Model\Qwen3.8-Flash-Next\` 4 片 ≈103.7 GB Q4_K_XL（A 端 CPU 冒煙＋PLE 真檔驗證） |
| **D:\llama fork** | ◐ | build 10685 **僅二進位**；＝**oracle**（輸出對答案），代碼不進產品 |
| **模型研究文件** | ✓ | [../QwenQwen3.8-Flash-Next/](../QwenQwen3.8-Flash-Next/)＝A-2/A-4 規格來源 |
| **t/s 基準** | ✓ | [../冷熱架構與t_s估算/](../冷熱架構與t_s估算/)（repo 外，保留） |
| **Python** | ◐ | `py -3.14` 仍在（**僅** repo 外工具/legacy_v1 用；**v3 合約層已移除（A-7）**） |
| **git** | ✓ | main；`86b572a`（A-3）；每步 commit |

## 二、Task A 資產（v4：C++ 唯一）

| 資產 | 位置 | 角色 |
|---|---|---|
| **q4e 引擎樹（C++）** | `engine/`（b10729 clone＋改名＋改核心） | **唯一實現** |
| A-3 快取核心＋單測 | `engine/common/q4e-expert-cache.{h,cpp}`＋`engine/tests/test-q4e-expert-cache.cpp` | 82 checks 全綠；14+2 向量凍結於測試（原 cache.py 已移除） |
| A-4 PLE 核心＋單測 | `engine/common/q4e-ngram.{h,cpp}`＋`engine/tests/test-q4e-ngram.cpp` | 2957 checks 全綠；FNV/RNG/解碼向量凍結於測試（原 ngram.py 已移除） |
| A-2 插入點分析 | [docs_A2_insertion_points.md](docs_A2_insertion_points.md) | MoE dispatch＋cb_eval hook＋專家佈局＋PLE 路徑＋server 端點 |
| 專家佈局實測常數 | `legacy_v1/tools/live_common.py`（ExpertStore） | stride 921,600/1,228,800 B；專家 3,072,000 B；shared 5,222,400 B |
| v1 封存 | `legacy_v1/`（spec 工具＋36 證據＋RUNBOOK＋B9 報告） | 基準數字＋方法學（不受 A-7 影響） |
| e2e 請求體 | `tools/b5_body.json` | B-6 用（保留） |

## 三、Task A 項目（v4）

| 項目 | 名稱 | 狀態 | 說明 |
|---|---|---|---|
| **A-0** | 規劃/合約（02 v4＋03A/03B） | **✓ 本輪** | 分工、合約、路線、基底紀律、**C++ 唯一決策** |
| **A-1** | source 準備：clone 最新 llama.cpp→`engine/`（`q4e`）→CPU build | **✓ 完成** | clone b10729（`engine/PROVENANCE.md`）；**MSVC 14.44 static Release 385/385 全過**；`q4e-server.exe --version` 可執行 |
| **A-2** | qwen4exp：grep 上游（原生收）→CPU 冒煙 | **✓ 完成** | b10729 `src/models/qwen4exp.cpp` 完整實作；**CPU 冒煙 PASS：16 tokens**（prompt 0.42 t/s；gen **0.69 t/s**；`engine/smoke_A2_cpu.log`） |
| **A-3** | 動態快取核心（C++）：狀態機＋cb_eval hook＋單測 | **✓ 完成（commit `86b572a`）** | `q4e-expert-cache.{h,cpp}`＋`llama_context_params::cb_eval` hook；**82 checks 全綠** |
| **A-4** | PLE 離載（C++）：q4e-ngram（GGUF 解析＋FNV row-id＋4.5-bit 解碼＋mmap 讀＋預取＋bench） | **◐ 收尾** | 單元 **2957 checks 全綠**（本輪修 3 bug：CreateFileMapping 權限、MT19937 twist OOB、CPython seeding；見 §四 #10–12）；**待辦**：真檔 row0 oracle 常數嵌入→真檔測試＋bench→清診斷碼→commit |
| **A-7** | **C++ 化重整（初始化）**：移除 Python 合約層＋temp 目錄；README/02/03A/03B 更新；git 清理 | ✗ | **先於 A-5**（使用者指定順序：先文件→再清 Python） |
| **A-5** | q4e-server（C++）：註冊 cb_eval hook＋q4e 統計端點＋**CPU e2e**（200/401/SSE＋真 GGUF ≥1 token） | ✗ | `/v1` 合約＝上游 server 既有（見 02 §三） |
| **A-6** | 交付包：RUNBOOK_B v2＋README＋03A/03B＋commit | ✗ | B 可獨立執行 |

> 執行順序：**A-4 收尾 → A-7 → A-5 → A-6**。

## 四、Task A 缺口／風險／陷阱

1. ~~上游最新是否收 qwen4exp~~ **已解**：b10729 原生收。
2. ~~PLE 注入點未落檔~~ **已解**：[docs_A2_insertion_points.md](docs_A2_insertion_points.md)。
3. CPU e2e（A-5）1 token 慢 → 預算 ~10 min，超時＝回報。
4. ~~CMake 4.x 相容性~~ 已解（A-1）。
5. ~~`GGML_CPU_REPACK`~~ **紀律**：build 必 `GGML_CPU_REPACK=OFF`（44.35 GB > 20 GB RAM）。**B 端同須**。
6. **同目錄 in-place reconfigure 陷阱**：改 CMake 參數＝刪 build 目錄重來。
7. **`BUILD_SHARED_LIBS` 陷阱**：MSVC Windows 預設 ON→test 整塊跳過。**紀律：configure 必顯式 `-DBUILD_SHARED_LIBS=OFF`＋`-DLLAMA_BUILD_TESTS=ON`**。
8. **沙箱擋 grandchild process**：已改 `danger-full-access`＋approval=never→build 正常。
9. ~~pytest tmp_path 紀律~~ **不適用**（v3 Python 層移除後；C++ 測試用 repo 內 `_testtmp/` 或直寫 temp 檔案）。
10. **CreateFileMapping 權限（2026-09-02 實測，已修）**：檔案 handle 以 `GENERIC_READ` 開啟時，`CreateFileMappingA(..., PAGE_READWRITE, ...)` → **ERROR_ACCESS_DENIED**（→ uncaught `std::runtime_error` → `std::terminate` → **0xC0000409 BEX64**，WER 指 `ucrtbase.dll abort()`）。**修法＝`PAGE_READONLY`**（本表只讀）。
11. **MT19937 第二 twist loop OOB（2026-09-02 實測，已修）**：`mt[k] = mt[k+251]`（k∈[227,623)）→ **讀越界最多 264 words**（stack/heap 垃圾→RNG 序列錯；layout 不同時直接 crash）。正確＝`mt[k-227]`（＝`mt[k+(M-N)]`）。
12. **CPython `random.Random(int)` seeding（2026-09-02 確認，已修）**：int seed 走 **`init_by_array`**（seed 拆 **32-bit little-endian chunks**、key 長度＝chunk 數、seed=0→key=[0]）；**不是** `init_genrand`、也不是固定 8-chunk key。RNG parity 目標＝CPython 3.14 `random`。
13. **0xC0000409 診斷法（可重用）**：`Get-WinEvent`（WER）＋`Report.wer` 看 BEX 偏移→objdump `ucrtbase.dll` 確認 crash 指令＝`abort()` 內的 `int $0x29`（SIGABRT trap）→**加 `std::set_terminate` 處理器**即可判定「uncaught exception」（本輪實證）。

## 五、約束（A 端）

| 約束 | 決定 |
|---|---|
| **目標紅線** | **Qwen3.8-Flash-Next 可完整載入使用**；**不為乾淨砍功能** |
| **C++ 唯一（v4）** | 產品實現＝C++（`engine/`）；Python 僅 legacy_v1/repo 外；合約 oracle 凍結於 C++ 測試 |
| **基底紀律** | 產品代碼只進 `engine/`；`D:\llama` fork＝oracle，代碼不進產品 |
| **無 GPU** | A 端全部＝CPU build＋CPU 跑；GPU 事→Task B |
| **ngl=10** | B 端設定紅線（A 端 config 預設同值） |
| **快取判斷** | CPU 式（LRU＋確定性預取），不訓練模型 |
| **git** | 每步 commit；可回滾 |
| **引擎代碼風格** | ASCII only；註解 1–2 行（ASD-STE100）；复用上游基建 |

## 六、現況總結

- **v1 A-0～A-5：全數完成並驗收**（M-A）。
- **v4：A-0 ✓、A-1 ✓、A-2 ✓、A-3 ✓（`86b572a`）、A-4 ◐（單元 2957 checks 全綠；真檔收尾中）**。
- **本輪修 3 個 C++ bug**（#10–12）→ 單元測試由 crash（0xC0000409）轉 2957/2957 全綠。
- **下一步**：A-4 收尾（真檔 oracle＋commit）→ **A-7 重整（移 Python＋清目錄）** → A-5 → A-6 → Task B 解鎖。
- 卡住規則：任一门失敗→停下回報（步驟、錯誤原文、已試事項），不猜。
