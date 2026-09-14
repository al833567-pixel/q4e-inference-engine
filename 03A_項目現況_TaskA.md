# 推理工具 — 項目現況 · Task A (v6 100k 主線)

## 〇、Task A 角色
無 GPU 可用：設計＋寫碼（C++）＋CPU build＋測試＋交付包。判斷權全在 A 側。

## 一、當前狀態
**v6 A 側完成**：A-16.0 設計拍板＋A-16.1a~A-16.7b 13 個原子步交付，**16 測試全綠（CPU）**（v6 架構：全局動態池 + VRAM Gather + PLE/ngram SSD mmap + 100k KV Cache 分頁 Offload，已於 `01_架構系統資料.md` 拍板定案並落地）。
歷史版本已全面廢棄。
下一步：**A-16.8 交付包**（8a 文件對齊完成／8b RUNBOOK_B_v6 編號對齊待做）→ 之後 **B-16 真機復驗（Task B）**。

## 二、v6 交付項目 (全完成)
| 項 | 名稱 | 狀態 | 內容／驗收／commit |
|---|---|---|---|
| A-16.0 | v6 設計拍板＋參數語義 | ✅ `f5b15e9` | 重定義 `--q4e-moe-vram N`；新增 `--q4e-staging-mb`、`--q4e-layer-floor`、`--q4e-ple-mmap`、`--cache-type-k q8_0`、`--q4e-kv-ram`。design.md＋test-arg-parser 全綠。 |
| A-16.1 | 全局動態池管理器 | ✅ `d467f05`/`91abfb0` | Global LRU＋Layer-Aware Floor (防層間擠壓)。188 斷言＋接入 common 庫。 |
| A-16.2 | VRAM 內 Gather 設計 | ✅ `2e597b5`/`ecfb070` | Staging Buffer 雙緩衝 (A/B) 管理邏輯 (118)＋CUDA Kernel 介面定義 (34)。 |
| A-16.3 | H2D 提升管線 | ✅ `7de45e0` | L2 Pinned RAM＋SSD→L2 fill＋非同步 H2D (118，非同步 job queue＋L2 LRU 保護)。 |
| A-16.4 | Graph 手術 (最高風險) | ✅ `5f19be6`/`d75f4c4` | 48 層全 GPU＋動態 slot 映射 (78)＋promotion op (427，含 oracle 覆蓋)。 |
| A-16.5 | **100k KV Cache 分頁 Offload** | ✅ `6207f7c`/`abc7afc` | 分頁管理器 (947862，含 100k oracle 滑窗)＋KV bridge (40)。**16.5b-2 深度接線併入 A-16.6**（CUDA 路徑屬 B-16）。 |
| A-16.6 | Server 接線＋stats | ✅ `a891982`/`9445169` | Expert L1/L2/L3 命中＋KV 分頁統計 (139/50)，`/q4e/stats` v6 群組上線。 |
| A-16.7 | e2e/bench v6 | ✅ `c82fbc1`/`9ccded1` | 客戶端工具更新（bench 30＋e2e 五門 28，含 `--v6-kv100k` 100k 壓力）。 |
| A-16.8 | 交付包 v6 | 🔄 | 8a 文件對齊完成（本步）／8b RUNBOOK_B_v6 編號對齊待做。 |

## 三、現況總表 (16 測試全綠，CPU)
| 測試 | 斷言數 | 備註 |
|---|---|---|
| test-arg-parser | — | v6 6 參數解析＋驗證 |
| test-q4e-global-pool | 188 | 全局池核心 |
| test-q4e-staging | 118 | 雙緩衝管理 |
| test-q4e-gather | 34 | kernel 介面 args build/validate |
| test-q4e-h2d-pipeline | 118 | 非同步管線 |
| test-q4e-graph-map | 78 | 動態 slot 映射 |
| test-q4e-promotion | 427 | **含 oracle 覆蓋** |
| test-q4e-kv-paging | 947862 | **含 100k oracle 滑窗** |
| test-q4e-kv-bridge | 40 | params→paging 組裝 |
| test-q4e-stats-v6 | 139 | 七元件快照＋JSON |
| test-q4e-server-v6 | 50 | server 接線 |
| test-q4e-bench-v6 | 30 | 五元件 throughput＋100k 壓力 |
| test-q4e-e2e-v6 | 28 | **五門**（池組裝／覆蓋路徑 oracle／管線壓力／KV 100k 滑窗／stats-JSON） |
| test-q4e-expert-cache（v5 凍結） | 159 | 回歸 |
| test-q4e-expert-warm（v5 凍結） | 195 | 回歸 |
| test-q4e-ngram（v5 凍結） | 2989 | 回歸 |

**新元件（engine/common/）**：`q4e-global-pool`、`q4e-staging`、`q4e-gather`（介面）、`q4e-h2d-pipeline`、`q4e-graph-map`、`q4e-promotion`、`q4e-kv-paging`、`q4e-kv-bridge`、`q4e-stats-v6`、`q4e-server-v6`；工具：`q4e-bench --v6/--v6-kv100k`、`q4e-e2e --v6/--v6-kv100k`；server：`/q4e/stats` 回 v6 群組。

**範圍說明（16.5b-2）**：A-16.5b-2（PagedAttention 深度接線）併入 A-16.6 範圍處理——A 側 16.5 交付＝分頁管理器＋bridge＋server 接線（C++ 測試全綠，達 02 接受準）；**CUDA attention 路徑與 GPU 驗證屬 B 側（RUNBOOK B-16）**。

## 四、環境與約束
- CPU/RAM: i7-9700 / ~20 GB RAM / SATA SSD。
- GPU: 無 (A 側)。
- 架構紅線：48 層全 GPU 算；L1 全局池＋VRAM 內 Gather；PLE 必須 SSD mmap；100k KV 必須量化＋分頁。
- Build 陷阱：`GGML_CPU_REPACK=OFF`；`BUILD_SHARED_LIBS=OFF`；必包 `vcvars64.bat`。
- **沙箱 build 附註**：本機 DSH 沙箱禁 ninja 直跑（pipe-spawn 封鎖＋CJK stat）；交付 build 用 `ninja -t commands`＋純 ASCII .bat（%REPO% 注入）。B 側正常環境不受限（ninja 可直跑）。