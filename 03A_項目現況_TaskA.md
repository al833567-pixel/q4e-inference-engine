# 推理工具 — 項目現況 · Task A（無 GPU，較聰明 AI）

> 更新：2026-09-01（v3：Task A/B 分工；03 拆成 03A/03B 兩檔）
> 姊妹檔：[03B_項目現況_TaskB.md](03B_項目現況_TaskB.md)（GPU 側）
> 關聯：[02_實作項目工作規劃.md](02_實作項目工作規劃.md)；[01_架構系統資料.md](01_架構系統資料.md)

## 〇、Task A 的角色

**無 GPU 可用、較聰明 AI**：設計＋寫碼（q4e 核心＋qwen4exp 支援＋server）＋CPU build＋spec/測試＋交付包。
全部工作可在無 GPU 環境完成（CPU build＋CPU 跑＋文件對照）。判斷權全在 A 側。

## 一、Task A 環境（已實測）

| 項目 | 狀態 | 說明 |
|---|---|---|
| **Python 3.14.6** | ✓ | `py -3.14`；pytest 9.1.1、numpy、fastapi/uvicorn、PyYAML |
| **MSVC（VS 2022 BuildTools）** | ✓ | vcvars64.bat 在位（cl 不在 PATH→build 時 invoke） |
| **CMake** | ✓ | 4.4.2 |
| **Ninja** | ✓ | 1.13.2 |
| **MinGW g++（WinGet WinLibs）** | ⚠ | **在 PATH 上，會搶 CMake 的編譯器**→ MinGW headers 缺 `CreateFile2`（httplib 0.54.1 編不過）。**對策：build 必須包在 `vcvars64.bat` 環境＋`-DCMAKE_C(XX)_COMPILER=cl`**（已實測 MSVC 14.44.35207 選中）；B 端同理 |
| **CUDA toolkit** | ◐ | v13.3（nvcc 在 PATH）—**A 端不使用**（無 GPU）；B 端 build 用（B-0 確認版本） |
| **CPU / RAM** | ✓ | i7-9700 8C/8T（無 HT）；RAM 可用 ~20 GB |
| **SSD（D 槽）** | ✓ | SATA 0.31–0.44 GB/s（GGUF 所在） |
| **GPU** | **✗（A 端）** | GPU 工作全歸 Task B（見 03B） |
| **GGUF（檔案在）** | ✓ | `D:\AI_Model\Qwen3.8-Flash-Next\` 4 片 ≈103.7 GB Q4_K_XL（**載入驗證＝B 端**；A 端做 CPU 冒煙） |
| **D:\llama fork** | ◐ | build 10685 **僅二進位**（無 source）；＝**oracle**（輸出對答案），代碼不進產品 |
| **模型研究文件** | ✓ | [../QwenQwen3.8-Flash-Next/](../QwenQwen3.8-Flash-Next/)（架構 01–08＋layer_memory）＝A-2/A-4 規格來源 |
| **t/s 基準** | ✓ | [../冷熱架構與t_s估算/](../冷熱架構與t_s估算/)（estimate_tps.py、tensor_audit.py） |
| **git** | ✓ | main；`dd74e71` 基線→`a9619b4` 清理；每步 commit |

## 二、Task A 資產（v2/v3 角色）

| 資產 | 位置 | 角色 |
|---|---|---|
| 快取 spec（LRU＋預取＋過濾＋統計，14 向量） | `cache.py`＋`tests/test_cache.py` | **A-3 核心單測 oracle** |
| PLE 合約（解析＋4.5-bit 解碼＋FNV hash） | `ngram.py`＋`tests/test_ngram.py` | **A-4 合約 oracle** |
| API 合約（/v1＋SSE＋401＋OpenAI 錯誤） | `server.py`＋`auth.py`＋`tests/test_server.py` | **A-5 `/v1` 合約 oracle** |
| 合約層 | `interfaces.py`＋`config.py`＋`config.yaml` | 簽章＋參數（ngl=10 紅線） |
| spec 工具 | `tools/bench_ngram.py`、`tools/b5_body.json` | A-4 CPU 對照；B-6 e2e body |
| 專家佈局實測常數 | `legacy_v1/tools/live_common.py`（ExpertStore） | A-3/A-4 直接用（stride 921,600/1,228,800 B；專家 3,072,000 B；shared 5,222,400 B） |
| v1 封存 | `legacy_v1/`（spec 工具＋36 證據＋RUNBOOK＋B9 報告） | 基準數字＋方法學 |
| **pytest** | **47/47 全綠** | spec＋合約測試（清理後實跑驗證） |

## 三、Task A 項目（v3）

| 項目 | 名稱 | 狀態 | 說明 |
|---|---|---|---|
| **A-0** | 規劃/合約（02 v3＋03A/03B） | **✓ 本輪** | 分工、合約、路線、基底紀律 |
| **A-1** | source 準備：clone 最新 llama.cpp→`engine/`（`q4e`）→CPU build | **✓ 完成** | clone b10729（458681e；`engine/PROVENANCE.md`）；改名（project→`q4e`、exe→`q4e-server`）；**MSVC 14.44 static Release 385/385 全過**（job pwsh-44，`engine/build_A1.log`）；**`q4e-server.exe --version` 可執行**（version: 0.3.0-dev，MSVC 19.44.35228 Windows AMD64）；`llama-cli.exe` 亦在（A-2 冒煙用）；**坑**：PATH 的 MinGW 會搶 CMake 編譯器→必須 vcvars64＋cl |
| **A-2** | qwen4exp：grep 上游→CPU 冒煙 | **◐ 第一步完成** | **上游原生收 qwen4exp**（b10729：`src/models/qwen4exp.cpp` 完整實作＋`LLM_ARCH_QWEN4EXP`＋PLE/QSA；gguf-py 已映射 `Qwen4ExpForCausalLM→qwen4exp`）→ A-2 縮小為 **CPU 冒煙：真 GGUF 載入＋≥1 token**（預算 ~10 min/token） |
| **A-3** | 動態快取核心（C++）：cache.py 合約＋14 向量單測 | ✗ | touch() 接真實 router（真實 M） |
| **A-4** | PLE 離載（C++）：ngram.py 合約＋CPU live | ✗ | SSD 表讀＋預取 |
| **A-5** | q4e-server（C++）：/v1＋auth＋metrics＋**CPU e2e** | ✗ | 200/401/SSE＋真 GGUF 1 token |
| **A-6** | 交付包：RUNBOOK_B v2＋README＋03A/03B 更新＋commit | ✗ | B 可獨立執行 |

## 四、Task A 缺口／風險

1. ~~上游最新是否收 qwen4exp~~ **已解**：b10729 原生收（2026-09-01 grep 確認）→ 不需自實作架構。
2. PLE 注入點＋MoE dispatch hook 點＝偵察中（subagent 報告→`docs_A2_insertion_points.md`）。
3. CPU e2e（A-5）1 token 慢（秒~十秒級）→ 預算 ~10 min，超時＝回報。
4. 上游 build 系統版本差異（CMake 4.x 相容性）→ A-1 內處理。

## 五、約束（A 端）

| 約束 | 決定 |
|---|---|
| **目標紅線** | **Qwen3.8-Flash-Next 可完整載入使用**（載入＋推理＋/v1＋快取）；**不為乾淨砍功能**（砍其他架構＝選項，預設不做，須複驗） |
| **基底紀律** | 產品代碼只進新 clone source；`D:\llama` fork＝oracle，**代碼不進產品** |
| **無 GPU** | A 端全部＝CPU build＋CPU 跑；GPU 事→Task B |
| **ngl=10** | B 端設定紅線（A 端 config 預設同值） |
| **快取判斷** | CPU 式（LRU＋確定性預取），不訓練模型 |
| **git** | 每步 commit；可回滾 |

## 六、現況總結

- **v1 A-0～A-5：全數完成並驗收**（M-A；spec/合約/工具/測試 47/47）。
- **v3：A-0 完成；下一步＝A-1（clone 最新 llama.cpp→`engine/` 改名 `q4e`→CPU build）**。
- 卡住規則：任一门失敗→停下回報（步驟、錯誤原文、已試事項），不猜。
