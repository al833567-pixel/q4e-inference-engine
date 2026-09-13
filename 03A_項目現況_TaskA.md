# 推理工具 — 項目現況 · Task A (v6 100k 主線)

## 〇、Task A 角色
無 GPU 可用：設計＋寫碼（C++）＋CPU build＋測試＋交付包。判斷權全在 A 側。

## 一、當前狀態
v6 架構（全局動態池 + VRAM Gather + PLE/ngram SSD mmap + 100k KV Cache 分頁 Offload）已於 `01_架構系統資料.md` 拍板定案。
歷史版本已全面廢棄。
下一步：啟動 **A-16.0 (v6 設計拍板＋參數語義)**。

## 二、v6 待辦項目
| 項 | 名稱 | 狀態 | 內容／驗收 |
|---|---|---|---|
| A-16.0 | v6 設計拍板＋參數語義 | ⏳ | 重定義 `--q4e-moe-vram N`；新增 `--q4e-staging-mb`、`--q4e-layer-floor`、`--q4e-ple-mmap`、`--cache-type-k q8_0`、`--q4e-kv-ram`。 |
| A-16.1 | 全局動態池管理器 | ⏳ | Global LRU＋Layer-Aware Floor (防層間擠壓)。 |
| A-16.2 | VRAM 內 Gather 設計 | ⏳ | Staging Buffer 雙緩衝 (A/B) 管理邏輯＋CUDA Kernel 介面定義。 |
| A-16.3 | H2D 提升管線 | ⏳ | L2 Pinned RAM＋SSD→L2 fill＋非同步 H2D。 |
| A-16.4 | Graph 手術 (最高風險) | ⏳ | 48 層全 GPU＋動態 slot 映射＋promotion op。 |
| A-16.5 | **100k KV Cache 分頁 Offload** | ⏳ | PagedAttention 接線＋KV L2(RAM)/L3(SSD) 分頁管理邏輯。 |
| A-16.6 | Server 接線＋stats | ⏳ | Expert L1/L2/L3 命中＋KV 分頁統計。 |
| A-16.7 | e2e/bench v6 | ⏳ | 客戶端工具更新（含 100k 壓力測試）。 |
| A-16.8 | 交付包 v6 | ⏳ | 文件與腳本落地。 |

## 三、環境與約束
- CPU/RAM: i7-9700 / ~20 GB RAM / SATA SSD。
- GPU: 無 (A 側)。
- 架構紅線：48 層全 GPU 算；L1 全局池＋VRAM 內 Gather；PLE 必須 SSD mmap；100k KV 必須量化＋分頁。
- Build 陷阱：`GGML_CPU_REPACK=OFF`；`BUILD_SHARED_LIBS=OFF`；必包 `vcvars64.bat`。