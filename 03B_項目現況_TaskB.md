# 推理工具 — 項目現況 · Task B (v6 主線)

## 〇、Task B 角色
有 GPU 可用：照 RUNBOOK 零判斷執行（CUDA build＋真機載入＋oracle 對答案＋驗收）。失敗回報原文，不猜。

## 一、當前狀態
等待 Task A 完成 v6 核心實作 (A-16.0 ~ A-16.7) 並交付構建包。
歷史版本 (v4.1/v5) 之除錯紀錄已全面清除。
下一步：收到 v6 交付包後，啟動 **B-16 (v6 真機復驗)**。

## 二、v6 待辦項目
| 項 | 名稱 | 狀態 | 內容／門 |
|---|---|---|---|
| B-16 | v6 真機復驗 | ⏳ | 照 RUNBOOK 零判斷。主門：ON 暖 t/s ≥ 8 (理想 40+)＋L1/L2/L3 命中分佈＋Oracle 對答案＋VRAM-fit (≤24GB)＋回歸門。 |

## 三、Task B 環境
| 項目 | 狀態 | 說明 |
|---|---|---|
| GPU | ✓ | RTX 3090 24 GB |
| CUDA | ✓ | v13.3 (nvcc V13.3.73) |
| CPU/RAM/SSD | ✓ | i7-9700 / ~20 GB / SATA 0.31–0.44 GB/s |
| GGUF | ✓ | 4 片 ≈103.7 GB (D:) |

## 四、約束
- 零判斷：照 RUNBOOK 執行；失敗＝錯誤原文＋證據。
- 並行紅線：永不過兩個模型實例；絕不殺 harness 之 `llama-server`。
- Build 旗標：`-DGGML_CPU_REPACK=OFF`＋`-DBUILD_SHARED_LIBS=OFF`；必包 `vcvars64.bat`。