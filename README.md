# 推理工具 — Qwen3.8-Flash-Next（qwen4exp）推理

> **現況（2026-09-08）**：**Task A 全完成**：A-14.0～A-14.6＋A-12 ✓（`169fc67`…`4e08ef2`）；**交付包＝全 C++**（server＋q4e-e2e＋q4e-bench＋q4e-ngram＋測試；交付樹零 .py/.ps1（`engine/` 上游除外））；**B-8 復驗就緒**（[RUNBOOK_B_v3.md](RUNBOOK_B_v3.md)，B 側 GPU）；**v5（B 方案，三層全 GPU）已規劃**（A-15.0~A-15.7＋B-9，B-8 後接，見 [02 §二](02_實作項目工作規劃.md)＋[執行順序.md](執行順序.md)）。
> **策略**：**C++ 唯一**（`engine/` q4e 樹）；**A-14 全 C++ 化**＝交付鏈 100% C++、我們的工作範圍零 .py/.ps1（上游 `engine/*.py` 不納入；`legacy_v1/*.py` 替代後刪、證據保留）。
> **紅線**：設定＝`--n-gpu-layers all`＋`--q4e-moe-vram K`（**K＝MoE VRAM 快取預算**，預設 10、範圍 0–999、VRAM 裝不下跳錯、K 可設定；舊 `-ncmoe/-cmoe/-ncffn` 已連 core 移除，見 01 §六）；本模型完整載入使用（不為乾淨砍功能）；oracle `D:\llama` 只對答案；B 端零判斷；kill 只按 PID。

## 文件

| 文件 | 說明 |
|---|---|
| [01_架構系統資料.md](01_架構系統資料.md) | 架構、模型事實（專家佈局）、/v1 設計、記憶體佈局、元件清單 |
| [02_實作項目工作規劃.md](02_實作項目工作規劃.md) | **現況＋未完成項目主表＋驗收基準（M-C）＋紅線紀律** |
| [03A_項目現況_TaskA.md](03A_項目現況_TaskA.md) | Task A（無 GPU）未完成項目（**v5＝A-15.0~A-15.7**）＋環境＋build 陷阱 |
| [03B_項目現況_TaskB.md](03B_項目現況_TaskB.md) | Task B（GPU）未完成項目（**B-8、B-9（v5）**）＋環境＋約束 |
| [執行順序.md](執行順序.md) | **Task A／B numbered 執行順序**（v4.1 ✓ → B-8 → v5 A-15.x → B-9） |
| [已完成工作.md](已完成工作.md) | 已完成工作彙整（v1 基準＋Task A＋Task B＋決策紀錄）；歷史細節＝`git log` |
| [RUNBOOK_B_v3.md](RUNBOOK_B_v3.md) | **Task B 復驗手冊（B-8，零判斷）**：硬線/t/s/重疊三門＋回歸門 |
| [docs_A2_insertion_points.md](docs_A2_insertion_points.md) | qwen4exp 插入點分析（MoE dispatch＋cb_eval hook＋專家佈局＋server 端點） |
| [legacy_v1/](legacy_v1/) | v1 封存（B9 報告＋evidence；.py 工具已於 A-14.5 刪） |

## 目錄結構

```
推理工具/
├── 01/02/03A/03B/README/已完成工作.md   # 架構 / 規劃 / 現況 / 已完成
├── 執行順序.md                          # Task A/B numbered 執行順序（v5 規劃）
├── RUNBOOK_B_v3.md                     # B-8 零判斷手冊
├── docs_A2_insertion_points.md         # 插入點分析
├── engine/                             # **唯一實現（C++，q4e 樹）**
│   ├── common/
│   │   ├── q4e-expert-cache.{h,cpp}    # 專家快取狀態機（vram/ram/ssd LRU＋touch/prefetch）
│   │   ├── q4e-expert-warm.{h,cpp}     # A-14.1 live 預取 backend
│   │   └── q4e-ngram.{h,cpp}           # PLE 離載（GGUF 解析＋FNV row-id＋4.5-bit 解碼＋mmap＋預取）
│   ├── tools/
│   │   ├── q4e-e2e/                    # A-14.2 C++ e2e 客戶端（G0–G6）
│   │   └── q4e-bench/                  # A-14.3 C++ bench 客戶端（ON/OFF＋JSON）
│   ├── tests/
│   │   ├── test-q4e-expert-cache.cpp   # 98 checks（向量凍結）
│   │   ├── test-q4e-expert-warm.cpp    # 59 checks（A-14.1 backend）
│   │   └── test-q4e-ngram.cpp          # 2989 checks（FNV/RNG/解碼向量凍結）
│   └── build-msvc/                     # MSVC Release build（bin/*.exe）
└── legacy_v1/                          # v1 封存（B9 報告／evidence）
```

## build（A 端，CPU）

```powershell
# 環境：vcvars64（MSVC 14.44）＋ CMake ＋ ninja（PATH 上的 MinGW 會搶編譯器，必須包起來）
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cmake -S engine -B engine/build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CPU_REPACK=OFF -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON && cmake --build engine/build-msvc'
# 測試
engine\build-msvc\bin\test-q4e-expert-cache.exe   # 98 checks
engine\build-msvc\bin\test-q4e-ngram.exe          # 2989 checks（$env:Q4E_NGRAM_REAL_DIR 指向 4 片目錄＝真檔）
```

## 狀態

- **v4 完成**：Task A（A-0～A-9＋A-11）✓；Task B（B-0～B-7）✓（2026-09-06）。M-C 8 項＝5 PASS／1 FAIL（#3→已改 #3′＝ON≥OFF 且 ≥1.95 t/s）；#4 欄位已上線（A-14.1 ✓，A 側 CPU 實測 0.61，80% 門在 B-8 GPU 量）、#5 A-14.4 ✓ 可量（`5b09765`／`c92606e`，CPU accept verdict PASS、max drop +0.22%；B-8 GPU 終驗）。
- **v4.1 排程**：A-9 ✓ → A-11 ✓ → A-14.0 ✓（`169fc67`）→ A-14.1 ✓（`243dcac`…`9a9472b`）→ A-14.2 ✓（`19eaae3`）→ A-14.3 ✓（`471abae`）→ A-12 ✓（`ed84537`／`6fa43e7`）→ A-14.4 ✓（`5b09765`／`c92606e`）→ A-14.5 ✓（`4e08ef2`）→ **A-14.6 ✓（本次，終門）** → **B-8 復驗（下一步，[RUNBOOK_B_v3.md](RUNBOOK_B_v3.md)，B 側 GPU）**。
- **v5（B 方案，三層全 GPU，2026-09-08 拍板）**：B-8 後接 **A-15.0~A-15.7（Task A）＋B-9（Task B）**。48 層 expert 全 GPU 算＋專家級動態駐留（**L1 VRAM pool N=5000≈104/層＋L2 pinned RAM N2=4000＋L3 SSD**；miss→H2D 提升、LRU 逐出；重用 warm 管線＋ngram hint）。目標 **t/s ≥8**（v4.1 實測 2–4 的 2×，M≤R 時）＋L1 hit ≥80%。最高風險＝A-15.2 graph 手術（pool tensor＋id-remap＋promotion op）。細節見 [02 §二 v5 表](02_實作項目工作規劃.md)／[執行順序.md](執行順序.md)。
- git：main；歷史與已完成證據＝`git log`＋[已完成工作.md](已完成工作.md)。
