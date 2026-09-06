# 推理工具 — Qwen3.8-Flash-Next（qwen4exp）推理

> **現況（2026-09-06）**：v4 兩軌全完成（M-C 5 PASS／1 FAIL 已改 #3′ 線／2 待接線）；B-4 硬線 ON<OFF 根因已修（A-9）。**下一動工＝A-14.1**（live 預取＋real backend）。
> **策略**：**C++ 唯一**（`engine/` q4e 樹）；**A-14 全 C++ 化**＝交付鏈 100% C++、我們的工作範圍零 .py/.ps1（上游 `engine/*.py` 不納入；`legacy_v1/*.py` 替代後刪、證據保留）。
> **紅線**：ngl=10 固定；本模型完整載入使用（不為乾淨砍功能）；oracle `D:\llama` 只對答案；B 端零判斷；kill 只按 PID。

## 文件

| 文件 | 說明 |
|---|---|
| [01_架構系統資料.md](01_架構系統資料.md) | 架構、模型事實（專家佈局）、/v1 設計、記憶體佈局、元件清單 |
| [02_實作項目工作規劃.md](02_實作項目工作規劃.md) | **現況＋未完成項目主表＋驗收基準（M-C）＋紅線紀律** |
| [03A_項目現況_TaskA.md](03A_項目現況_TaskA.md) | Task A（無 GPU）未完成項目＋環境＋build 陷阱 |
| [03B_項目現況_TaskB.md](03B_項目現況_TaskB.md) | Task B（GPU）未完成項目（B-8）＋環境＋約束 |
| [已完成工作.md](已完成工作.md) | 已完成工作彙整（v1 基準＋Task A＋Task B＋決策紀錄）；歷史細節＝`git log` |
| [RUNBOOK_B_v3.md](RUNBOOK_B_v3.md) | **Task B 復驗手冊（B-8，零判斷）**：硬線/t/s/重疊三門＋回歸門 |
| [docs_A2_insertion_points.md](docs_A2_insertion_points.md) | qwen4exp 插入點分析（MoE dispatch＋cb_eval hook＋專家佈局＋server 端點） |
| [legacy_v1/](legacy_v1/) | v1 基準證據＋方法學（B9 報告＋36 證據；.py 工具於 A-14.5 刪） |

## 目錄結構

```
推理工具/
├── 01/02/03A/03B/README/已完成工作.md   # 架構 / 規劃 / 現況 / 已完成
├── RUNBOOK_B_v3.md                     # B-8 零判斷手冊
├── a9_e2e.ps1                          # CPU e2e（A-14.2 q4e-e2e 取代後刪）
├── docs_A2_insertion_points.md         # 插入點分析
├── engine/                             # **唯一實現（C++，q4e 樹）**
│   ├── common/
│   │   ├── q4e-expert-cache.{h,cpp}    # 專家快取狀態機（vram/ram/ssd LRU＋touch/prefetch）
│   │   └── q4e-ngram.{h,cpp}           # PLE 離載（GGUF 解析＋FNV row-id＋4.5-bit 解碼＋mmap＋預取）
│   ├── tests/
│   │   ├── test-q4e-expert-cache.cpp   # 98 checks（向量凍結）
│   │   └── test-q4e-ngram.cpp          # 2989 checks（FNV/RNG/解碼向量凍結）
│   └── build-msvc/                     # MSVC Release build（bin/*.exe）
└── legacy_v1/                          # v1 封存（B9 報告／evidence／.py 工具）
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

- **v4 完成**：Task A（A-0～A-9＋A-11）✓；Task B（B-0～B-7）✓（2026-09-06）。M-C 8 項＝5 PASS／1 FAIL（#3→已改 #3′＝ON≥OFF 且 ≥1.95 t/s）／2 待接線（#4 待 A-14.1、#5 待 A-14.4）。
- **v4.1 排程**：A-9 ✓ → A-11 ✓ → **A-14.1（live 預取＋real backend；`prefetch_io_issued`/`prefetch_overlap_ratio`）** → A-14.2（`q4e-e2e`）→ A-14.3（`q4e-bench`）→ A-12（交付包 v4.1）→ **B-8 復驗（[RUNBOOK_B_v3.md](RUNBOOK_B_v3.md)）** → A-14.4（ngram live）→ A-14.5（清理）→ A-14.6（終門）。
- git：main；歷史與已完成證據＝`git log`＋[已完成工作.md](已完成工作.md)。
