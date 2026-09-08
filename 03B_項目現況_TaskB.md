# 推理工具 — 項目現況 · Task B（有 GPU，較笨 AI）

> 更新：2026-09-08（**本檔只留未完成項目**；已完成工作→[已完成工作.md](已完成工作.md)；commit 全錄→`git log --oneline`）
> 姊妹檔：[03A_項目現況_TaskA.md](03A_項目現況_TaskA.md)
> 關聯：[02_實作項目工作規劃.md](02_實作項目工作規劃.md)；[RUNBOOK_B_v3.md](RUNBOOK_B_v3.md)

## 〇、Task B 的角色

**有 GPU 可用、較笨 AI**：照 RUNBOOK **零判斷執行**（每步＝指令＋預期輸出＋備案）：CUDA build＋真機載入＋oracle 對答案＋驗收＋§9 回報。**失敗回報格式＝錯誤原文＋證據，不猜。**

## 一、未完成項目

| 項 | 名稱 | 狀態 | 內容／門 |
|---|---|---|---|
| **B-8**（v4.1） | **復驗（A-12 交付後）**：照 [RUNBOOK_B_v3.md](RUNBOOK_B_v3.md) 零判斷 | **⏳ 就緒（A-14 全 C++ 化全完成（含 A-14.4 ngram live＋A-14.5 清理）；B 側動工前逐項對 RUNBOOK「A 側已驗證」表）** | **重驗 v4 三門**＋回歸門：(1) **B-4 硬線**：暖 t/s **ON ≥ OFF**（A-9 已移除 sync D2H 根因）；(2) **#3′**：ON ≥ OFF 且 **ON ≥ 1.95 t/s**（A-11 拍板線）；(3) **#4 重疊**：`prefetch_overlap_ratio` **≥80%**（A-14.1 提供欄位；定義＝觸碰前已完成預取的 touches 佔比）＋`prefetch_io_issued>0`；**#5 N-gram＝A-14.4 ✓ 可量**（`5b09765`／`c92606e`；CPU 預驗 PASS max drop +0.22%；B 側 GPU 量）；回歸門＝冷載入≤~250 s＋VRAM 歸零＋oracle 對答案＋e2e 5 閘。**工具**：`q4e-bench`（ON/OFF t/s＋JSON）＋`q4e-e2e`（取代 curl 手動步）。**設定**：ON＝q4e `--n-gpu-layers all --q4e-moe-vram 10`（K=10 預設、**K 可設定**、B-8 可量 12/13 找上限）；OFF＝oracle（上游原生旗標）`-ngl all -ncmoe 38`（B-8-0 先驗 oracle 有 `-ncmoe`；若無→q4e `--no-q4e-prefetch` 同 placement）。**FAIL＝停下回報原文，不調參。** |

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
| **GPU** | ✓ | RTX 3090 **24 GB**；舊配置 ngl=10 實測 **16,724 MiB**；新配置（ngl=all+K=10）預計 **≈20.6 GB**（B-8-2 實測）；收尾歸零 |
| **CUDA toolkit（nvcc）** | ✓ | **v13.3**（nvcc V13.3.73；driver 610.47） |
| **CPU / RAM / SSD** | ✓ | i7-9700 8C/8T；RAM ~20 GB；SATA 0.31–0.44 GB/s |
| **GGUF** | ✓ | `D:\AI_Model\Qwen3.8-Flash-Next\`（4 片 111,334,654,784 B ≈103.69 GiB Q4_K_XL） |
| **oracle** | ✓ | `D:\llama\llama-server.exe`（fork build 10685；**僅二進位**） |
| **MSVC** | ✓ | VS 2022 BuildTools（vcvars64.bat）—CUDA build 用 |
| **harness 注意** | ⚠ | harness 自身的模型服務＝`D:\llama\llama-server.exe` 常駐；**殘留檢查只配對 `q4e-server`，絕不殺 `llama-server`** |

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
