# 推理工具 — 項目現況 · Task B (v6 100k 主線)

## 〇、Task B 角色
有 GPU 可用：照 RUNBOOK 零判斷執行（CUDA build＋真機載入＋oracle 對答案＋驗收）。失敗回報原文，不猜。

## 一、當前狀態
**v6 A 側已交付**：A-16.0 設計拍板＋A-16.1a~A-16.7b 13 個原子步（14 commits `f5b15e9`~`9ccded1`），17 測試全綠（CPU）。
歷史版本之除錯紀錄已全面清除。
**B-16 前置條件已滿足**：A-16.8 交付包已全完（`f4aaecd`/`321c1f8`）。依 RUNBOOK_B_v6.md 啟動 **B-16 (v6 真機復驗)**。
**B-16-0 ✓ 全綠**（2026-07 本 session）：VRAM 0 MiB、nvcc V13.3.73、GGUF 4 片 111,334,654,784 B、無殘留 server、HEAD `1930478` tree clean。
**B-16-1 ✓ build 完成（2026-07-15 22:51，job `pwsh-6` exit 0）**：484 步全過，`b16.done` 內容 `BUILD_OK`（此為成功標記，.bat 末步寫入）。產物 `build-cuda\bin\q4e-server.exe` 44,102,656 B @ 22:51:43；冒煙 `--version` exit 0 → `version: 0.3.0-dev (build 80, commit 1930478)`（commit 與 B-16-0 HEAD 一致）、`built with MSVC 19.44.35228.0 for Windows AMD64`。全程僅 C4819 字碼頁警告（CP950 顯示噪音）＋1 良性 `#177-D`（mmid.cu `neu_padded` unused）。下一步＝**B-16-2 真機載入**（VRAM-fit N=5000 ≤ 24 GB）。
**B-16-1 ⏳ build 進行中**（DSH job `pwsh-3`，484 步 .bat）：runbook 直跑命令兩度失敗（PATH 上 MSYS2 mingw64 ninja 以 ANSI 重編碼 CJK 路徑 → `FindFirstFileExA(../../../../AI研究/推理工具/engine/common): ...syntax is incorrect`；備案刪 `.ninja_log` 重試同錯——該檔本就不存在）。另證：**native MSVC ninja 亦無法直跑此樹**（`-n` dry-run → `mkdir(D:/Nova2_home/AI??s): Illegal byte sequence`＋source missing）。採 A 側同機已證配方：native ninja `-t commands q4e-server`（純文字提取，484 步）→ 純 ASCII .bat（CJK 前綴 2106 處以 `%REPO%` env 替換）→ `cmd /c "%BAT%"` 背景 job。手補 CMake 4.4.2 漏生成之 `CMakeFiles\llama.rsp`(183 obj)/`ggml-cuda.rsp`(142 obj)（自 build.ninja lib edges 提取）。
**三環境根因已修（job 鏈 `pwsh-3`→`pwsh-4`→`pwsh-5`→`pwsh-6`，皆環境層級、無 CMake 參數變更）**：
(a) ccache 4.13.6 以預設 `CCACHE_DIR=C:\Users\al833\AppData\Local\ccache` 靜默掛死（零輸出、無 cl 子程；對照實驗：同一 cache 全量複製放 D: 即 ~2 s 編過）→ `CCACHE_DIR` 改 D: 側 A 側暖備份 `v6_work\b16\diag\ccache_copy`（7626 項）。
(b) harness pwsh 環境無 INCLUDE/LIB/LIBPATH → STEP=1 `C1034: inttypes.h` → .bat 於 `setlocal` 後加 `call vcvars64.bat`（與 runbook 自身 build 命令之首命令同一動作，非調參）。
(c) PATH 上唯一 gzip＝MSYS2 版，沙箱下 `*** fatal error - couldn't create signal pipe, Win32 error 5`（Git for Windows 版同死，exit -1073741502）→ STEP=3 gzip 失敗。自建 native shim `v6_work\b16\bin\gzip.exe`（C `wmain`→`CreateProcessW` 直啟 powershell 5.1 `GzipStream`；`_wsystem`/cmd 命令列在 CJK 下必敗，故不走 cmd）。ui-assets.cmake 之 `find_program(gzip)` 於 `-P` script 執行期自 PATH 解析 → .bat 將 `bin\` 前置 PATH（環境修，非參數修）。shim 全過 findprobe：find_program 命中、`-c` exit 0、magic `1F 8B`、roundtrip 位元一致。step 3 之 HF 下載 SSL 失敗（b80＋latest）＝設計內 fallback（embedding stale assets），非致命。

## 二、v6 待辦項目
| 項 | 名稱 | 狀態 | 內容／門 |
|---|---|---|---|
| B-16 | v6 真機復驗 (含 100k) | B-16-0 ✓／B-16-1 ✓（22:51 `pwsh-6` exit 0，`BUILD_OK`，`bin\q4e-server.exe` 44,102,656 B，冒煙 `build 80 commit 1930478`）／**B-16-2 ✗ 停在 VRAM-fit 門**（22:57 job `pwsh-7`，runbook 原參數 N=5000）：server 預檢 `common_q4e_moe_vram_check` 拒絕載入——`CUDA0 needs 78431 MiB, only 23222 MiB free`→exit。依備案回報原文、**不降 N**。runbook 預期 N=5000≈15.4 GiB 與預檢 78431 MiB 相差甚遠，兩者原文並列呈報。
**A 側已裁定＋修正（A-16.9，2026-07-15）**：root cause＝v6 重定義只改 parser/help，load path 仍走 v4.1「K=層數」buft 語義（`N=5000 ≥ n_layers=48` → 零 override）→ 24576 experts 全量上 GPU（≈75 GB）被 fit 門正確拒載（78431 MiB 含 attn/KV/compute，與「全量 expert 上 GPU」的實測一致）。修正＝`common_q4e_moe_vram_apply` 經新 `llm_q4e_moe_vram_keep_layers()`（N→ceil(N/512) 整層：N=5000→10 層=5120 experts≈15.7 GB≥N 預算；N=0→全 CPU）再插 override；fit 訊息 k→N；test-arg-parser +8 斷言。**下一動＝B-16-1 重編（B 側，同 run_b16 配方）→ B-16-2 依原參數重跑**（不降 N）。 | 照 RUNBOOK 零判斷。門：**VRAM-fit N=5000 ≤ 24GB**＋**100k ctx q8_0 載入不 OOM**＋**ON 暖 t/s ≥ 8 (理想 40+)**＋**Oracle 對答案**＋**e2e 5 門綠**＋冷載入 ≤~250 s＋收尾 VRAM 歸零。 |

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
- **CJK 路徑 build 陷阱（本 session 實證）**：本機與 A 側同環境同 PATH——ninja 作為 executor 直跑此 CJK 樹必敗（MSYS2 ninja：ANSI 重編碼 FindFirstFileExA 錯；native MSVC ninja：mkdir Illegal byte sequence）。`-t commands` 純文字提取可行。故 build 一律走 A 側已證配方（`.bat`＋`%REPO%` env＋背景 job；detached 樹會被沙箱收走，不可用 `Start-Process -Wait`）。
- **環境層級修正三條（本 session 實證，皆不動 build 參數）**：(a) `CCACHE_DIR` 必須在 D:（C: AppData 路徑 ccache 靜默掛死）；(b) .bat 必含 `vcvars64.bat` call（harness pwsh 無 INCLUDE/LIB）；(c) PATH 必前置 `v6_work\b16\bin`（native gzip shim；系統 gzip 皆為 cygwin 系、沙箱下死，且 CJK 命令列不可經 cmd/`_wsystem`，需 CreateProcessW 直啟）。