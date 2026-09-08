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

> **執行順序（v4.1，全 ✓）**：A-14.0 ✓ → A-14.1 ✓ → A-14.2 ✓ → A-14.3 ✓ → **A-12（交付 v4.1）✓** → A-14.4 ✓ → **A-14.5 ✓（`4e08ef2`）** → **A-14.6 ✓（本次 commit）**；**Task A（v4.1）全項 ✓（2026-09-08）；剩 B-8（Task B，[03B](03B_項目現況_TaskB.md)）**
> **範圍拍板（2026-09-06）**：上游 `engine/*.py` 218 檔（conversion/gguf-py/scripts/server tests/examples）＝**不納入**；`legacy_v1/*.py` C++ 替代全綠後**刪除**（evidence 保留）；`config.yaml` 已移除（設定＝q4e-server CLI flags）。
> **Task A（v4.1）全項完成（2026-09-08）；B-8 後接 v5（B 方案，三層全 GPU）＝A-15.0~A-15.7（見下）**。

## 一-v5、未完成項目（B 方案＝三層全 GPU MoE；2026-09-08 拍板，完整版）

> 目標：v4.1「K 層 GPU＋38 層 CPU 算」→ **48 層 expert 全 GPU 算＋專家級動態駐留**（L1 VRAM pool＋L2 pinned RAM＋L3 SSD；miss→H2D 提升、LRU 逐出；重用 warm 管線＋ngram hint）。細節／驗收／風險見 [02 §二 v5 表](02_實作項目工作規劃.md)。

| 項 | 名稱 | 內容／驗收 |
|---|---|---|
| **A-15.0** | v5 設計拍板＋參數語義 | 重定義 `--q4e-moe-vram N`（＝VRAM 駐留 routed 專家數，預設 5000≈104/層）＋新 `--q4e-moe-ram N2`（L2，預設 4000）＋三層定義＋LRU 逐出（L2 副本保留→L1 逐出 metadata no-op）＋VRAM-fit/RAM-fit 公式；驗收＝設計文件＋參數解析/fit 單元測試 |
| **A-15.1** | 專家 pool 駐留管理器 | 擴充 `q4e-expert-cache`：per-(layer,expert) L1/L2/L3＋per-layer LRU pool（R 槽）＋promote/evict＋**hit 分佈（L1/L2/L3）＋真實 M 計數**；驗收＝C++ 全綠＋單元測試 |
| **A-15.3** | H2D 提升管線 | 改指 `q4e-expert-warm`：pinned staging＋非同步 H2D 進 pool 槽＋重疊度量＋dedup＋ngram 提前提升；驗收＝C++ 全綠＋測試（H2D bytes/重疊比） |
| **A-15.2**（core，最高風險） | graph 手術 | `qwen4exp` arch：每層 expert **pool tensor**（gate/up×[2560,640,R]、down×[640,2560,R]）＋**id-remap**（global→pool slot）＋**promotion op**（H2D＋slot→global 映射）＋`MUL_MAT_ID` over pool（**48 層全 GPU**）；驗收＝C++ 全綠＋**oracle 對答案**＋B 側載入成功 |
| **A-15.4** | L2 pinned RAM pool＋SSD→L2 fill | pinned pool（N2）＋OS page cache backing＋L2 miss→SSD 讀入 L2→H2D；驗收＝C++ 全綠＋測試 |
| **A-15.5** | server 接線＋`/q4e/stats` v5 | L1/L2/L3 hit 分佈＋H2D bytes＋pool 佔用＋**真實 M**＋重疊比；驗收＝端點全欄位＋C++ 測試 |
| **A-15.6** | q4e-bench / q4e-e2e v5 | ON/OFF t/s＋L1/L2/L3 命中＋M＋JSON 報告；驗收＝B-9 直接跑取數 |
| **A-15.7** | 交付包 v5 | README/01/02/03A/03B/RUNBOOK 全落地「三層全 GPU」＋「A 側已驗證」表；驗收＝交付包可獨立執行 |

> **v5 執行順序（依賴序）**：A-15.0 → A-15.1 → **A-15.3（H2D 管線）** → **A-15.2（graph 手術，最高風險）** → A-15.4 → A-15.5 → A-15.6 → A-15.7 → **B-9（Task B，[03B](03B_項目現況_TaskB.md)）**。
> **前置**：B-8（v4.1 復驗）先跑，量出**真實 M**＋v4.1 實測 t/s（v5 目標＝2×）；M≤R（≈104/層）→v5 收益最大（L1 全命中）。

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
8. **header 依賴追蹤缺失**：`engine/build-msvc` 無 ninja .d depfile（CJK/ACP-950 環境）→header 改動不觸發重編→stale .obj（ODR 不一致）；**紀律：改到 header 後 build 必 `--clean-first`（或核對 .obj mtime）**；2026-09-08 實踩（S6 e2e 假失敗）→`S6-rebuild.log` 修復證據（暫存 log 已清除、歸檔 git `bcaf1c9`，`git show bcaf1c9:S6-rebuild.log` 取回）。

## 五、指針

- 已完成工作（A-0～A-9、A-11、B-0～B-7、v1 基準、決策紀錄）：[已完成工作.md](已完成工作.md)
- A-14.1 設計（issued 定義、雙緩衝、warm 語義、v1 對照）：[02 §十](02_實作項目工作規劃.md)
- B-4 FAIL 根因與核實：[已完成工作.md §四](已完成工作.md)
