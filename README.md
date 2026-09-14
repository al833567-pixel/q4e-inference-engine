# 推理工具 — Qwen3.8-Flash-Next（qwen4exp）推理（v6 主線）

> **現況**：**v6 A 側已全線交付**（A-16.0 設計拍板＋A-16.1a~A-16.7b 13 個原子步＋A-16.8 交付包，16 測試全綠 CPU；[01_架構系統資料.md](01_架構系統資料.md)：全局動態池＋VRAM 內 Gather＋H2D 管線＋100k KV 分頁）。**Task B 之 B-16 前置條件全部滿足**，待 GPU 釋放即啟動真機復驗。
> **歷史版本**：v4.1/v5 **已全面廢棄**，本 repo 不保留其歷史程式碼與除錯紀錄（證據/報告/舊手冊已清除；git log 可查）。
> **策略**：**C++ 唯一**（`engine/` q4e 樹）；交付樹零 .py/.ps1（上游 `engine/` 開發工具除外）。
> **紅線**：48 層 expert **全 GPU 算**；L1 **全局池**＋VRAM 內 Gather；**禁用 Per-Layer Pool**；B 端零判斷；build 必包 `vcvars64.bat`＋`-DBUILD_SHARED_LIBS=OFF -DGGML_CPU_REPACK=OFF`。
> **測試模型**：`D:\AI_Model\Qwen3.8-Flash-Next\`（4 片 `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf`，≈103.7 GiB）。

## 文件

| 文件 | 說明 |
|---|---|
| [01_架構系統資料.md](01_架構系統資料.md) | **v6 架構**：工具定位、模型事實、三層專家快取、記憶體佈局、執行參數 |
| [02_實作項目工作規劃.md](02_實作項目工作規劃.md) | **分工＋v6 主表（A-16.0~A-16.8 ✅／B-16 待啟動）＋驗收基準＋紅線紀律** |
| [03A_項目現況_TaskA.md](03A_項目現況_TaskA.md) | Task A（無 GPU）：**v6 交付現況（16 測試全綠）＋新元件＋環境＋build 陷阱** |
| [03B_項目現況_TaskB.md](03B_項目現況_TaskB.md) | Task B（GPU）：**B-16 前置已滿足（A 側已交付）＋門＋環境＋約束** |
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

## v6 交付狀態（A 側完成，待 B-16）

> **主線一句話**：100k 主線＝L1 全局動態池＋VRAM 內 Gather（staging 雙緩衝）＋H2D 提升管線＋100k KV Cache 分頁 Offload；48 層全 GPU 算。

- **A 側已交付**：A-16.0 設計拍板（`f5b15e9`）＋A-16.1a~A-16.7b 13 個原子步（`d467f05`…`9ccded1`）＋A-16.8 交付包（`f4aaecd`/`321c1f8`）；**16 測試全綠（CPU）**＝13 項 v6 新＋3 項 v5 凍結回歸（詳見 [03A](03A_項目現況_TaskA.md)）。
- **範圍說明**：A-16.5b-2（PagedAttention 深度接線）併入 A-16.6 範圍處理——A 側 16.5 交付＝分頁管理器＋bridge＋server 接線（C++ 測試全綠，達 02 接受準）；**CUDA attention 路徑與 GPU 驗證屬 B 側（RUNBOOK B-16）**。

### v6 CLI 參數（6 項）
| 參數 | 預設 | 說明 |
|---|---|---|
| `--q4e-moe-vram` | 5000 | **v6 重定義**：L1 全局 VRAM pool 駐留 routed 專家總數（0 = pool off） |
| `--q4e-moe-ram` | 4000 | L2 Pinned RAM 駐留專家數（**100k 極限配置顯式傳 3000**，見 RUNBOOK） |
| `--q4e-staging-mb` | 128 | Staging Buffer（A/B 雙緩衝）總大小 MB |
| `--q4e-layer-floor` | 10 | Layer-Aware LRU 每層保底 expert 數（0 = 純全局 LRU） |
| `--q4e-ple-mmap` | on\|off | PLE/ngram 保持 SSD mmap（off 僅警告；生產恒 on 為紅線） |
| `--q4e-kv-ram` | 6000 | KV Cache L2 Pinned RAM 預算 MB（100k 分頁 offload） |

### 快速驗證（A 側 CPU）
```powershell
engine\build-msvc\bin\q4e-e2e.exe --v6 --v6-kv100k    # e2e v6 五門（含 100k 滑窗）
engine\build-msvc\bin\q4e-bench.exe --v6 --v6-kv100k  # bench v6（五元件 throughput＋100k 壓力）
# server：/q4e/stats 回 v6 群組（七元件快照＋KV 分頁統計）；真機步驟見 RUNBOOK_B_v6.md
```

**狀態**：A 側完成（A-16.0~A-16.8 全交付，16 commits `f5b15e9`…`321c1f8`）→ **待 B-16 真機復驗**（RUNBOOK_B_v6.md 已對齊 02 正典）。

## v6 主門（詳見 [02 §三](02_實作項目工作規劃.md)）

1. **t/s**：ON 暖 t/s **≥ 8**（理想 L1 全命中 ≥ 40）。
2. **VRAM-fit**：`N=5000` 載入成功、VRAM ≤ 24 GB。
3. **Oracle 對答案**：48 層全 GPU＋全局池＋Gather 逐層輸出＝oracle。
4. **回歸門**：e2e 5 閘綠＋冷載入 ≤~250 s＋收尾 VRAM 歸零。

git：main；v6 起所有狀態以 01/02/03A/03B 為準。
