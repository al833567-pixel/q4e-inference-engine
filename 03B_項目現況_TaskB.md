# 推理工具 — 項目現況 · Task B（有 GPU，較笨 AI）

> 更新：2026-09-02（v4：C++ 唯一實現；**Task A 全數完成（A-0～A-6）→ Task B = READY / 待執行**；零判斷手冊＝[RUNBOOK_B_v2.md](RUNBOOK_B_v2.md)）
> 姊妹檔：[03A_項目現況_TaskA.md](03A_項目現況_TaskA.md)
> 關聯：[02_實作項目工作規劃.md](02_實作項目工作規劃.md)（Task B 項目＝§五）

## 〇、Task B 的角色

**有 GPU 可用、較笨 AI**：照 RUNBOOK **零判斷執行**（每步＝指令＋預期輸出＋備案）：CUDA build＋真機載入＋oracle 對答案＋驗收＋§9 回報。
**v4 不變**：B 端工作全部照 RUNBOOK_B v2 執行；被測對象＝**C++ 產品**（`engine/` q4e 樹）。
**失敗回報格式＝錯誤原文＋證據，不猜。**

## 一、Task B 環境（已實測）

| 項目 | 狀態 | 說明 |
|---|---|---|
| **GPU** | ✓ | RTX 3090 **24 GB**；ngl=10 跑動佔 **16,750 MiB**；收尾歸零（v1 實測） |
| **CUDA toolkit（nvcc）** | **待確認** | B-0 前置；A 端 PATH 見 **v13.3**；fork 為 CUDA 12 系（cublas64_12）→ B-0 對齊版本 |
| **CPU / RAM / SSD** | ✓ | i7-9700 8C/8T；RAM ~20 GB；SATA 0.31–0.44 GB/s |
| **GGUF** | ✓ | `D:\AI_Model\Qwen3.8-Flash-Next\`（4 片 ≈103.7 GB Q4_K_XL） |
| **oracle** | ✓ | `D:\llama\llama-server.exe`（fork build 10685；**僅二進位**） |
| **MSVC** | ✓ | VS 2022 BuildTools（vcvars64.bat）—B-1 CUDA build 用 |
| **Python** | ◐ | `py -3.14 -X utf8`（**僅** legacy_v1 方法學工具＋repo 外 t/s 估算；v3 合約層已移除，B 端不依賴） |

## 二、Task B 已驗證基準（v1 證據；v3/v4 驗收對照用）

| 數字 | 值 | 來源（`legacy_v1/evidence/`） |
|---|---|---|
| 冷載入（ngl=10） | **226.4 s**，VRAM 16,750/24,576 MiB | `smoke_b1.log` |
| 暖 t/s（無預取，M=512 上界列） | **1.95**（PASS ≥0.28） | `bench_noprefetch.log`/`bench_check.log` |
| 冷 t/s | 0.33 | bench |
| fake_router 無局部性（M=512 均勻） | 0.10–0.32 t/s＝SATA 頻寬牆（**非快取失敗**） | `wire_cache_live.log`/`bench_prefetch.log` |
| 快取命中（301 tokens） | vram 9.1% / ram **90.6%** / ssd **0.3%** | `wire_cache_live.log` |
| N-gram 預取 | **淨加速**（交錯 4 組 drop −7.4%…−10.2%；首次 34.19% FAIL＝ramp 假象） | `ngram_accept.log`/`ngram_diag.log` |
| ramp law | 新 phase 前 ~300 tokens 由 0.5→2.4 t/s 爬升 | `ngram_diag.log` |
| 旁證：真實路由有局部性 | 每 token 讀 ~1.17 GB 權重仍達 1.95 t/s→工作集住進 page cache→**真實 M 遠小於 512** | `legacy_v1/B9_report.md` §B-5 |

## 三、Task B 資產（v4）

| 資產 | 位置 | 角色 |
|---|---|---|
| **C++ 產品**（q4e 樹） | `engine/` | **B-1～B-7 的被測對象**（CUDA build＋真機） |
| oracle（fork 二進位） | `D:\llama\` | B-2 輸出對答案（token 序列）；**只對答案，代碼不進產品** |
| v1 證據（36 檔） | `legacy_v1/evidence/` | 基準數＋方法學實證 |
| v1 驗收報告 | `legacy_v1/B9_report.md` | §9 回報格式範本 |
| v1 工具（封存） | `legacy_v1/tools/` | 方法學參考（交錯 A/B、tail 窗、ExpertStore 佈局） |
| e2e body | `tools/b5_body.json` | B-6 e2e 請求體（JSON，保留） |
| t/s 驗收線 | [../冷熱架構與t_s估算/estimate_tps.py](../冷熱架構與t_s估算/estimate_tps.py) | B-5：`estimate_tps(M_real)`×0.7 |

## 四、Task B 項目（v4 不變）

| 項目 | 名稱 | 狀態 | 說明 |
|---|---|---|---|
| **B-0** | 前置：VRAM 空出＋CUDA 13.3（nvcc）＋GGUF 4 片＋無並行實例 | **READY / 待執行** | 前置清單逐項 ✓ |
| **B-1** | q4e **CUDA build**（vcvars64＋`GGML_CUDA=ON`＋`BUILD_SHARED_LIBS=OFF`＋`GGML_CPU_REPACK=OFF`） | **READY / 待執行** | `engine\build-cuda\bin\q4e-server.exe` |
| **B-2** | 真 GGUF 載入（**ngl=10**）＋`/health` 200（≤~250 s）＋VRAM 占用 | **READY / 待執行** | v1 參考 226.4 s／16,750 MiB |
| **B-3** | `/v1` e2e 5 閘（200／≥1 token／401／SSE）＋`/q4e/stats` 活數據 | **READY / 待執行** | 同 A-5 閘（CPU e2e 已綠） |
| **B-4** | 快取驗 **開 vs 關硬線**：暖 t/s（ON）≥（OFF，oracle 無 hook 基線） | **READY / 待執行** | 02 §八硬線；ON<OFF→回報不調參 |
| **B-5** | **oracle 對答案**（vs `D:\llama\llama-server.exe`，同 prompt＋temp=0） | **READY / 待執行** | 一致或可解釋差異；diff 原文回報 |
| **B-6** | 性能基準（02 §六）：暖 t/s ≥ `estimate_tps(M_real)`×0.7＋重疊 ≥80%＋N-gram 掉幅 <1% | **READY / 待執行** | 基準參照 1.95 t/s 暖；不及→profile＋回報 |
| **B-7** | 收尾：VRAM 歸零＋PLE 28.8 GB 留 SSD＋程序退出 | **READY / 待執行** | 未歸零→回報原文 |

> **Task A 全數完成（A-0～A-6）→ B 端照 [RUNBOOK_B_v2.md](RUNBOOK_B_v2.md) 依 B-0→B-7 零判斷執行**；每步＝指令＋預期輸出＋備案。驗收＝02 §六 M-C 8 項。

## 五、Task B 約束

| 約束 | 決定 |
|---|---|
| **ngl=10 固定** | **使用者紅線**（永不 auto/999/48） |
| **build 旗標紅線** | CMake 必帶 `-DGGML_CPU_REPACK=OFF`＋`-DBUILD_SHARED_LIBS=OFF`；build 環境＝`vcvars64`＋cl（PATH 上的 MinGW 會搶編譯器） |
| **並行** | 永不過兩個模型實例同時載入；先 kill server 再 in-process 載 |
| **零判斷** | 照 RUNBOOK_B v2 執行；每步＝指令＋預期輸出＋備案 |
| **失敗回報** | **錯誤原文＋證據**（不猜、不繞） |
| **量測紀律** | temp=0、固定 prime prompt、warmup 32；A/B 交錯（A B A' B'）＋tail 窗（ramp law） |
| **日誌** | PS Tee 輸出＝UTF-16LE（讀時 BOM sniff）；證據全保留 |
| **console** | legacy 工具用 `py -3.14 -X utf8`（cp950 對策）；產品本身＝C++ 二進位 |

## 六、現況總結

- **v1 B-0～B-6：全數完成並驗收**（M-B 達標；§9 回報＋36 證據在 `legacy_v1/`）。
- **v4：Task A 全數完成（A-0～A-6）→ Task B = READY / 待執行**。照 [RUNBOOK_B_v2.md](RUNBOOK_B_v2.md) 依 **B-0→B-7** 零判斷執行（每步＝指令＋預期輸出＋備案）。
- **驗收**＝02 §六 M-C 8 項（ngl=10 紅線、`/v1` e2e 綠、真實 M 可測、暖 t/s×0.7、重疊 ≥80%、N-gram <1%、命中報告、A 側測試全綠、冷載入 ≤~250 s＋VRAM 歸零）。
- **紅線**：ngl=10 固定（不 999/48/auto）；oracle `D:\llama` 只對答案、代碼不進產品；不為乾淨砍功能。
- 卡住規則：任一门失敗→停下回報（步驟、錯誤原文、已試事項），不猜。
