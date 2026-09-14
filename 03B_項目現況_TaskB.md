# 推理工具 — 項目現況 · Task B (v6 100k 主線)

## 〇、Task B 角色
有 GPU 可用：照 RUNBOOK 零判斷執行（CUDA build＋真機載入＋oracle 對答案＋驗收）。失敗回報原文，不猜。

## 一、當前狀態
**v6 A 側已交付**：A-16.0 設計拍板＋A-16.1a~A-16.7b 13 個原子步（14 commits `f5b15e9`~`9ccded1`），16 測試全綠（CPU）。
歷史版本之除錯紀錄已全面清除。
**B-16 前置條件已滿足**：A-16.8 交付包已全完（`f4aaecd`/`321c1f8`）。依 RUNBOOK_B_v6.md 啟動 **B-16 (v6 真機復驗)**——待 GPU 釋放（harness 模型服務現佔 23.6/24 GB VRAM，B-16-0 要求 VRAM 空且絕不殺該服務）。

## 二、v6 待辦項目
| 項 | 名稱 | 狀態 | 內容／門 |
|---|---|---|---|
| B-16 | v6 真機復驗 (含 100k) | 🚦 前置已滿足 | 照 RUNBOOK 零判斷。門：**VRAM-fit N=5000 ≤ 24GB**＋**100k ctx q8_0 載入不 OOM**＋**ON 暖 t/s ≥ 8 (理想 40+)**＋**Oracle 對答案**＋**e2e 5 門綠**＋冷載入 ≤~250 s＋收尾 VRAM 歸零。 |

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
- **沙箱限制不適用**：A 側 DSH 沙箱之 build 限制（ninja 禁直跑）不適用於 B 側正常環境；B 側 ninja/build 正常直跑。