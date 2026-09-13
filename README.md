# 推理工具 — Qwen3.8-Flash-Next（qwen4exp）推理（v6 主線）

> **現況**：**v6 主線已拍板**（[01_架構系統資料.md](01_架構系統資料.md)：全局動態池＋VRAM 內 Gather＋三層專家快取＋48 層全 GPU）。**Task A 準備啟動 A-16.0**（v6 設計拍板＋參數語義）；**Task B 等待 v6 交付包**後啟動 B-16 真機復驗。
> **歷史版本**：v4.1/v5 **已全面廢棄**，本 repo 不保留其歷史程式碼與除錯紀錄（證據/報告/舊手冊已清除；git log 可查）。
> **策略**：**C++ 唯一**（`engine/` q4e 樹）；交付樹零 .py/.ps1（上游 `engine/` 開發工具除外）。
> **紅線**：48 層 expert **全 GPU 算**；L1 **全局池**＋VRAM 內 Gather；**禁用 Per-Layer Pool**；B 端零判斷；build 必包 `vcvars64.bat`＋`-DBUILD_SHARED_LIBS=OFF -DGGML_CPU_REPACK=OFF`。
> **測試模型**：`D:\AI_Model\Qwen3.8-Flash-Next\`（4 片 `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf`，≈103.7 GiB）。

## 文件

| 文件 | 說明 |
|---|---|
| [01_架構系統資料.md](01_架構系統資料.md) | **v6 架構**：工具定位、模型事實、三層專家快取、記憶體佈局、執行參數 |
| [02_實作項目工作規劃.md](02_實作項目工作規劃.md) | **分工＋v6 未完成主表（A-16.0~A-16.7／B-16）＋驗收基準＋紅線紀律** |
| [03A_項目現況_TaskA.md](03A_項目現況_TaskA.md) | Task A（無 GPU）：v6 待辦項目＋環境＋build 陷阱 |
| [03B_項目現況_TaskB.md](03B_項目現況_TaskB.md) | Task B（GPU）：B-16 待辦＋環境＋約束 |
| [RUNBOOK_B_v6.md](RUNBOOK_B_v6.md) | **Task B 復驗手冊（B-16，零判斷）**：build／載入／e2e／t/s／oracle／命中分佈／收尾 |

## 目錄結構

```
推理工具/
├── 01/02/03A/03B/README.md            # v6 架構 / 規劃 / 現況×2 / 索引
├── RUNBOOK_B_v6.md                    # B-16 零判斷手冊
└── engine/                            # **唯一實現（C++，q4e 樹）**
    ├── common/                        # q4e 核心（專家快取／預取／ngram；v6 於此演進）
    ├── src/                           # qwen4exp MoE 路徑（v6 graph 手術目標）
    ├── tools/
    │   ├── q4e-e2e/                   # C++ e2e 客戶端（G0–G6）
    │   └── q4e-bench/                 # C++ bench 客戶端（ON/OFF＋JSON）
    ├── tests/                         # test-q4e-*（單元＋向量凍結）
    ├── build-msvc/                    # A 側 CPU build（bin/*.exe）
    └── build-cuda/                    # B 側 CUDA build（bin/q4e-server.exe）
```

## build（A 端，CPU）

```powershell
# 環境：vcvars64（MSVC）＋ CMake ＋ ninja（PATH 上的 MinGW 會搶編譯器，必須包起來）
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cmake -S engine -B engine/build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CPU_REPACK=OFF -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON && cmake --build engine/build-msvc'
# 測試：engine\build-msvc\bin\test-q4e-*.exe
```

## v6 主門（詳見 [02 §三](02_實作項目工作規劃.md)）

1. **t/s**：ON 暖 t/s **≥ 8**（理想 L1 全命中 ≥ 40）。
2. **VRAM-fit**：`N=5000` 載入成功、VRAM ≤ 24 GB。
3. **Oracle 對答案**：48 層全 GPU＋全局池＋Gather 逐層輸出＝oracle。
4. **回歸門**：e2e 5 閘綠＋冷載入 ≤~250 s＋收尾 VRAM 歸零。

git：main；v6 起所有狀態以 01/02/03A/03B 為準。
