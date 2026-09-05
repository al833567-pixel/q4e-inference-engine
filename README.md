# 推理工具 — Qwen3.8-Flash-Next（qwen4exp）推理

> **v4（2026-09-02）**：自有引擎 **`q4e`** ＝ clone llama.cpp 最新 source → 改名 → **改核心**（專家級動態快取＋PLE 離載）。**C++ 唯一實現**（`engine/`）；v3 Python 合約層已移除，合約 oracle 常數凍結於 C++ 測試。
> **目標（紅線）**：Qwen3.8-Flash-Next **可完整載入使用**（載入＋推理＋/v1＋快取）；不為乾淨砍功能（砍其他架構＝選項，預設不做）。
> **分工**：**Task A**（無 GPU，較聰明 AI）設計＋寫碼（C++）＋CPU build＋交付包；**Task B**（GPU，較笨 AI）照 RUNBOOK 真機驗收。`D:\llama` fork＝**oracle only**（代碼不進產品）。
> v1 路線封存 → [`legacy_v1/`](legacy_v1/)。

## 文件

| 文件 | 說明 |
|---|---|
| [01_架構系統資料.md](01_架構系統資料.md) | 架構、記憶體佈局、/v1 設計、效能目標（v4：架構不變；技術基底已更新為 C++） |
| [02_實作項目工作規劃.md](02_實作項目工作規劃.md) | **v4 工作規劃**：Task A（A-0～A-7）/ Task B（B-0～B-7）＋介面合約＋驗收基準 |
| [03A_項目現況_TaskA.md](03A_項目現況_TaskA.md) | 項目現況 · Task A（無 GPU，較聰明 AI） |
| [03B_項目現況_TaskB.md](03B_項目現況_TaskB.md) | 項目現況 · Task B（有 GPU，較笨 AI） |
| [RUNBOOK_B_v2.md](RUNBOOK_B_v2.md) | **Task B 零判斷執行手冊**（B-0～B-7：指令＋預期輸出＋備案） |
| [docs_A2_insertion_points.md](docs_A2_insertion_points.md) | qwen4exp 插入點分析（MoE dispatch＋cb_eval hook＋專家佈局＋PLE 路徑） |
| [legacy_v1/](legacy_v1/) | v1（A/B 兩軌）RUNBOOK＋驗收報告＋工具＋36 份證據（封存） |

## 目錄結構（v4）

```
推理工具/
├── 01/02/03 .md                    # 架構 / v4 規劃 / 現況
├── engine/                         # **唯一實現（C++，q4e 樹）**
│   ├── common/
│   │   ├── q4e-expert-cache.{h,cpp}   # A-3：專家級動態快取核心（vram/ram/ssd LRU＋place/evict/touch/prefetch）
│   │   └── q4e-ngram.{h,cpp}          # A-4：PLE 離載（GGUF 解析＋FNV row-id＋4.5-bit 解碼＋mmap＋預取＋bench）
│   ├── tests/
│   │   ├── test-q4e-expert-cache.cpp  # 90 checks（向量凍結）
│   │   └── test-q4e-ngram.cpp         # 2989 checks（FNV/RNG/解碼向量凍結）
│   └── build-msvc/                   # MSVC Release build（bin/*.exe）
├── tools/
│   └── b5_body.json                  # e2e 請求體（B-3 用）
└── legacy_v1/                        # v1 封存（tools / evidence / RUNBOOK / B9 報告）
```

## build（A 端，CPU）

```powershell
# 環境：vcvars64（MSVC 14.44）＋ CMake 4.4.2 ＋ ninja
cmake -S engine -B engine/build-msvc -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF `
  -DGGML_CPU_REPACK=OFF -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build engine/build-msvc
# 測試
engine\build-msvc\bin\test-q4e-expert-cache.exe   # 82 checks
engine\build-msvc\bin\test-q4e-ngram.exe          # 2957 checks（fake GGUF；$env:Q4E_NGRAM_REAL_DIR 指向 4 片目錄＝真檔）
```

## 交付包（Task A → Task B）

> **Task A 全數完成（A-0～A-6）→ Task B 解鎖**。B 端（GPU、零判斷 AI）照下面執行：

| 文件 | 說明 |
|---|---|
| **[RUNBOOK_B_v2.md](RUNBOOK_B_v2.md)** | **B 端零判斷手冊**：B-0～B-7（前置確認→CUDA build→ngl=10 真機→/v1 e2e＋/q4e/stats→快取硬線→oracle 對答案→性能基準→收尾），每步＝指令＋預期輸出＋備案（失敗→回報原文，不猜） |
| [config.yaml](config.yaml) | B 側設定紅線：`ngl: 10`、GGUF 4 片（~103.7 GB）、C++ 模組＋`/q4e/stats` 端點指標 |
| [evidence_A5_cpu_e2e.md](evidence_A5_cpu_e2e.md) | A 側 CPU e2e 證據：真 GGUF＋200/401/SSE＋16 tokens＋`/q4e/stats` 活數據（`cb_eval` hook 真觸發） |
| [02_實作項目工作規劃.md](02_實作項目工作規劃.md) §六 | **驗收基準（M-C 8 項）**：ngl=10＋/v1 綠＋真實 M＋暖 t/s×0.7＋重疊 ≥80%＋N-gram <1%＋命中報告＋A 側測試全綠＋冷載入 ≤~250 s／VRAM 歸零 |

## 狀態

- **v1（A/B 兩軌）：全數完成並驗收**（M-A、M-B）；證據＝`legacy_v1/evidence/`（36 檔）＋`legacy_v1/B9_report.md`
- **v4：Task A 全數完成**——A-0 ✓、A-1 ✓、A-2 ✓（冒煙 16 tokens）、A-3 ✓（82 checks，`86b572a`）、A-4 ✓（2957 checks，`85907f6`）、A-7 ✓（`01f9202`）、A-5 ✓（CPU e2e 全綠，`5d98d54`）、A-6 ✓（交付包）
- **下一步：Task B 照 [RUNBOOK_B_v2.md](RUNBOOK_B_v2.md) 執行（READY）**
- 紅線：ngl=10 固定；永不過兩個模型實例並行；目標＝本模型完整載入使用（非砍多模型）
