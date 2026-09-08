# 推理工具 — 項目現況 · Task A（無 GPU，較聰明 AI）

> 更新：2026-09-08（**本檔只留未完成項目**；已完成工作→[已完成工作.md](已完成工作.md)；commit 全錄→`git log --oneline`）
> **新策略（2026-09-06 拍板）＝A-14 全 C++ 化**：交付鏈（工具＋測試＋e2e＋bench）100% C++，我們的工作範圍零 .py/.ps1（上游 `engine/*.py` 不納入；`legacy_v1/*.py` 於 C++ 替代全綠後刪除、證據保留）。
> 姊妹檔：[03B_項目現況_TaskB.md](03B_項目現況_TaskB.md)（GPU 側）
> 關聯：[02_實作項目工作規劃.md](02_實作項目工作規劃.md)；[01_架構系統資料.md](01_架構系統資料.md)

## 〇、Task A 的角色

**無 GPU 可用、較聰明 AI**：設計＋寫碼（**q4e 核心 C++**）＋CPU build＋C++ 測試＋交付包。全部工作可在無 GPU 環境完成。判斷權全在 A 側。

## 一、未完成項目（A-14 全 C++ 化；子項含原 A-10/A-13）

| 子項 | 名稱 | 狀態 | 內容／驗收 |
|---|---|---|---|
| **A-14.4**（＝A-13） | ngram live 接線＋diag 4 階段／accept P1–P4 C++ 化 | **✓（`5b09765` engine 接線＋`c92606e` q4e-ngram 工具；evidence_a144）** | ngram 預測 hint→prefetch；解鎖 M-C #5（N-gram 預取 t/s 掉 <1%） |
| **A-14.5** | 清理：刪 `legacy_v1/*.py`（8）＋`a9_e2e.ps1`＋2 temp smoke（**12 刪除**；輸出證據保留）；**保留** `legacy_v1/evidence/`＋B9 報告 | **✓（`4e08ef2`）** | 交付樹零 .py/.ps1（上游 `engine/` 除外，文件註明） |
| **A-14.6** | 終門：交付包宣告「全 C++」＋README/02/03A/03B/已完成工作/RUNBOOK 全落地 | **✓（本次 commit）** | A-12 交付條件 |

> **執行順序**：A-14.0 ✓ → A-14.1 ✓ → A-14.2 ✓ → A-14.3 ✓ → **A-12（交付 v4.1）✓** → A-14.4 ✓ → **A-14.5 ✓（`4e08ef2`）** → **A-14.6 ✓（本次 commit）**；**Task A 全項 ✓（2026-09-08）；剩 B-8（Task B，[03B](03B_項目現況_TaskB.md)）**
> **範圍拍板（2026-09-06）**：上游 `engine/*.py` 218 檔（conversion/gguf-py/scripts/server tests/examples）＝**不納入**；`legacy_v1/*.py` C++ 替代全綠後**刪除**（evidence 保留）；`config.yaml` 已移除（設定＝q4e-server CLI flags）。
> **Task A 全項完成（2026-09-08）；剩餘工作＝B-8（Task B）**（A-14.6 終門＝本次 commit 宣告「全 C++」）。

## 二、Task A 環境（已實測；build/e2e 用）

| 項目 | 狀態 | 說明 |
|---|---|---|
| **MSVC（VS 2022 BuildTools）** | ✓ | vcvars64.bat 在位（cl 不在 PATH→build 時 invoke） |
| **CMake** | ✓ | 4.4.2 |
| **Ninja** | ✓ | 1.13.2（CJK 路徑增量陷阱→刪 `.ninja_log` 備案；根因見 git `24f11f7` 之 `evidence_b1/root_cause_b1.md`） |
| **MinGW g++（WinGet WinLibs）** | ⚠ | **在 PATH 上，會搶 CMake 的編譯器**→**紀律：build 必包在 `vcvars64.bat` 環境** |
| **CUDA toolkit** | ◐ | v13.3（nvcc 在 PATH）—**A 端不使用**（無 GPU）；B 端 build 用 |
| **CPU / RAM** | ✓ | i7-9700 8C/8T（無 HT）；RAM 可用 ~20 GB |
| **SSD（D 槽）** | ✓ | SATA 0.31–0.44 GB/s（GGUF 所在） |
| **GPU** | **✗（A 端）** | GPU 工作全歸 Task B |
| **GGUF（檔案在）** | ✓ | `D:\AI_Model\Qwen3.8-Flash-Next\` 4 片 ≈103.7 GB Q4_K_XL |
| **D:\llama fork** | ◐ | build 10685 **僅二進位**；＝**oracle**（輸出對答案），代碼不進產品 |
| **模型研究文件** | ✓ | [../QwenQwen3.8-Flash-Next/](../QwenQwen3.8-Flash-Next/) |
| **t/s 基準** | ✓ | [../冷熱架構與t_s估算/](../冷熱架構與t_s估算/)（repo 外，保留） |
| **git** | ✓ | main；每步 commit（engine 變更）；HEAD 見 `git log` |

## 三、約束（A 端）

| 約束 | 決定 |
|---|---|
| **目標紅線** | **Qwen3.8-Flash-Next 可完整載入使用**；**不為乾淨砍功能** |
| **全 C++（2026-09-06 拍板）** | 交付鏈＝C++ 唯一（A-14）；上游 `engine/*.py` 不納入；`legacy_v1/*.py` 替代後刪 |
| **基底紀律** | 產品代碼只進 `engine/`；`D:\llama` fork＝oracle，代碼不進產品 |
| **無 GPU** | A 端全部＝CPU build＋CPU 跑；GPU 事→Task B |
| **placement** | `--n-gpu-layers all`＋`--q4e-moe-vram K`（K 預設 10、範圍 0–999、VRAM 裝不下跳錯、K 可設定）；舊 `-ncmoe/-cmoe/-ncffn` **連 core 移除**（01 §六） |
| **快取判斷** | CPU 式（LRU＋確定性預取），不訓練模型 |
| **git** | 每步 commit；可回滾 |
| **引擎代碼風格** | ASCII only；註解 1–2 行（ASD-STE100）；复用上游基建 |

## 四、build 陷阱（仍適用）

1. **`GGML_CPU_REPACK`**：build 必 `GGML_CPU_REPACK=OFF`（44.35 GB > 20 GB RAM）。**B 端同須**。
2. **同目錄 in-place reconfigure 陷阱**：改 CMake 參數＝刪 build 目錄重來。
3. **`BUILD_SHARED_LIBS` 陷阱**：MSVC Windows 預設 ON→test 整塊跳過。**紀律：configure 必顯式 `-DBUILD_SHARED_LIBS=OFF`＋`-DLLAMA_BUILD_TESTS=ON`**。
4. **MinGW 搶編譯器**：PATH 上的 g++ 會蓋過 MSVC→build 必包 `vcvars64.bat`。
5. **CJK 路徑 ninja 增量**：偶發 stale→刪 `.ninja_log` 重跑（A-9 本輪未觸發；B-1 曾觸發）。
6. **CreateFileMapping 權限**（A-14.1 backend 適用）：handle 以 `GENERIC_READ` 開啟時 mapping 必 **`PAGE_READONLY`**（`PAGE_READWRITE`→ERROR_ACCESS_DENIED→0xC0000409）。
7. **沙箱**：本 session＝danger-full-access＋approval disabled；build 正常。
8. **header 依賴追蹤缺失**：`engine/build-msvc` 無 ninja .d depfile（CJK/ACP-950 環境）→header 改動不觸發重編→stale .obj（ODR 不一致）；**紀律：改到 header 後 build 必 `--clean-first`（或核對 .obj mtime）**；2026-09-08 實踩（S6 e2e 假失敗）→`S6-rebuild.log` 修復證據。

## 五、指針

- 已完成工作（A-0～A-9、A-11、B-0～B-7、v1 基準、決策紀錄）：[已完成工作.md](已完成工作.md)
- A-14.1 設計（issued 定義、雙緩衝、warm 語義、v1 對照）：[02 §十](02_實作項目工作規劃.md)
- B-4 FAIL 根因與核實：[已完成工作.md §四](已完成工作.md)
