# RUNBOOK_B v3 — Task B（GPU 側）v4.1 復驗手冊（B-8，零判斷）

> **你是 Task B（有 GPU、較笨的 AI）**。v4 輪（B-0～B-7）已執行完畢（2026-09-06，[B9_report_v4.md](B9_report_v4.md)）：M-C 8 項＝5 PASS／1 FAIL（#3）／2 本 build 不可量（#4/#5）；B-4 硬線 FAIL（ON<OFF −11.3%）。
> **本輪（B-8）目的＝重驗 v4 三門＋回歸門**：① B-8-4 硬線 ON≥OFF（v4 FAIL）；② B-8-6 暖 t/s vs A-11 拍板線＋預取重疊 ≥80%（v4 不可量）；③ N-gram 掉幅（待 A-13，本輪仍「不可量」，照實回報）；回歸門＝e2e／oracle 對答案／冷載入／收尾 VRAM 歸零。
> **你的工作只有：B-8-0 → B-8-7 依序執行，對照預期輸出；任一步失敗→停下回報。** 不改 `engine/` 任何 source/test、不自行調參、不猜、不編造數字。
> 技術基底同 v2：Windows PowerShell（每步指令從 repo 根 `D:\Nova2_home\AI研究\推理工具\` 執行）；MSVC（vcvars64）＋CMake 4.4.2＋Ninja 1.13.2＋CUDA 13.3（nvcc）；RTX 3090 24 GB。
> 被測對象＝C++ 產品 `q4e-server`（`engine/`，v4.1＝A-9/A-10 之後的 build）；oracle＝`D:\llama\llama-server.exe`（fork build 10685，**只對答案，代碼不進產品**）。

## A 側已驗證（交付包 v4.1 內容；**B-8 動工前逐項對「✓」，任一未勾→停下回報 A**）

| 項 | 狀態 | 證據/commit |
|---|---|---|
| A-9 熱路徑開銷移除（`cb_eval` 不再對 device-resident ids 做 sync D2H；只簿記 CPU 層；stats 加 touched 層數） | **待 A 填** | **待 A 填** |
| A-11 驗收線拍板（M-C #3 值＋口徑） | **✓ 已拍板（2026-09-06）：Option A＝#3′ 暖 t/s（ON）≥ OFF 且 ≥ 1.95 t/s** | 02 §六 v4.1 修正 |
| A-10 live 預取接線（上一 token 選擇→`prefetch()`；real backend＝GGUF 專家切片 page-warm；`--q4e-prefetch off` 旗標） | **待 A 填** | **待 A 填** |
| C++ 測試全綠（test-q4e-expert-cache／test-q4e-ngram） | **待 A 填** | **待 A 填** |
| CPU e2e 全綠＋`prefetch_issued>0` 證據 | **待 A 填** | **待 A 填** |
| 設定紅線不變 | [config.yaml](config.yaml) `ngl: 10` | 同 v4 |

> 本表由 A-12 填完後 B 端才可動工；B 端只核對「全部 ✓＋有 commit/證據」，不做技術判斷。

## B-8-0 前置確認

**指令**
```powershell
nvidia-smi
nvcc --version
Get-ChildItem "D:\AI_Model\Qwen3.8-Flash-Next\*.gguf"
Get-Process | Where-Object { $_.ProcessName -match 'q4e-server|llama-server' }
# 交付包版本核對（須比 v4 的 1144ddd 新）
engine\build-cuda\bin\q4e-server.exe --version
```
**預期輸出**：GPU＝RTX 3090 且 VRAM 空（末條程序指令回傳空）；`nvcc` V13.3.x；4 片 `.gguf` 共 ~103.7 GB；`--version` commit **新於 v4（1144ddd）**。
**備案**：任一不符 → **停下回報**（原文），不猜。

## B-8-1 q4e CUDA build（v4.1 source）

**指令**（repo 根；vcvars64 包起來）
```powershell
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cmake -S engine -B engine/build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DGGML_CPU_REPACK=OFF && cmake --build engine/build-cuda --target q4e-server'
Test-Path engine\build-cuda\bin\q4e-server.exe
engine\build-cuda\bin\q4e-server.exe --version
```
**預期輸出**：build 結束無錯誤；`Test-Path` → **True**；`--version` commit＝A 側交付版。
**備案**：
- 若 incremental build 出現 `ninja: error: FindFirstFileExA(...): ... syntax is incorrect`（CJK 路徑增量陷阱，v4 B-1 已見，見 `evidence_b1/root_cause_b1.md`）→ **刪 `engine\build-cuda\.ninja_log` 後重試同一指令一次**；仍失敗→回報原文。
- 其他 CUDA/arch 錯誤 → **回報原文**（錯誤全文＋build 日誌尾段），不改 CMake 參數。

## B-8-2 真機載入（ngl=10）

**指令**（同 v4 B-2）
```powershell
Start-Process -FilePath "engine\build-cuda\bin\q4e-server.exe" -ArgumentList @(
  '--model','D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
  '--host','127.0.0.1','--port','8124','--api-key','bkey',
  '--n-gpu-layers','10','--ctx-size','2048','--parallel','1'
) -RedirectStandardOutput 'b8_2_server.log' -RedirectStandardError 'b8_2_server.err.log' -NoNewWindow

$deadline = (Get-Date).AddSeconds(300)
while ((Get-Date) -lt $deadline) {
  if ((curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8124/health) -eq "200") { break }
  Start-Sleep -Seconds 15
}
curl.exe -s -o NUL -w "health=%{http_code}\n" http://127.0.0.1:8124/health
nvidia-smi
```
**預期輸出**：`health=200`（**≤~250 s**，v4 實測 ~206 s）；VRAM 占用 **≈16,7xx MiB**（ngl=10）；log 有 model loaded／listening。
**備案**：超時/失敗 → **回報原文**（`b8_2_server.err.log`＋`nvidia-smi`），不自動降 ngl。

## B-8-3 `/v1` e2e 5 閘＋`/q4e/stats`（回歸門）

**指令**（同 v4 B-3；載荷改 `@file` 發送——harness 會剝單引號 JSON 雙引號，v4 B-3 已證實）
```powershell
# 載荷檔（v4 已存在：body_b.json=64 token 問、body_c.json=無 key、body_d.json=SSE；直接複用）
curl.exe -s -o NUL -w "%{http_code}\n" http://127.0.0.1:8124/health
curl.exe -s -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@body_b.json"
curl.exe -s -o NUL -w "%{http_code}\n" -X POST http://127.0.0.1:8124/v1/chat/completions -H "Content-Type: application/json" -d "@body_c.json"
curl.exe -s -N -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@body_d.json"
curl.exe -s http://127.0.0.1:8124/q4e/stats -H "Authorization: Bearer bkey"
```
**預期輸出**：(a) 200；(b) 200＋`usage.completion_tokens≥1`（token 可能在 `reasoning_content`，正常）；(c) 401；(d) 多行 `data:`＋`data: [DONE]`；(e) 200＋JSON 含 `hits_*`、`hit_rate_*`、**`prefetch.issued`（A-10 後應為數字欄位）**、touched 層數欄位。
**備案**：任一閘不綠 → **回報原文**（該閘完整回應體＋status＋log 尾段）。

## B-8-4 快取開 vs 關硬線（**v4 FAIL 重驗：期望 ON ≥ OFF**）

> 量測紀律同 v4：temp=0、固定 prompt、**2×256 tokens**（第 1 跑＝warmup、第 2 跑＝計數）；t/s＝`usage.completion_tokens ÷ time_total`。
> 並行紅線：ON/OFF 串行；kill→VRAM 歸零確認→再起 oracle。

**指令**（repo 根）
```powershell
# 載荷：256 token essay 問（v4 已存在 body_on256.json；64 token 對答案問＝body_b5_64.json）
# ── ON 相（q4e-server 8124，B-8-2 已起、已暖）──
curl.exe -s -o NUL -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@body_on256.json"
curl.exe -s -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@body_on256.json" -w "\nON  time_total=%{time_total}\n" -o b8_4_on.json
curl.exe -s http://127.0.0.1:8124/q4e/stats -H "Authorization: Bearer bkey" -o b8_6_stats.json
curl.exe -s -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@body_b5_64.json" -o b8_5_q4e.json

# ── 切換 ──
Stop-Process -Name q4e-server -Force
Start-Sleep -Seconds 5
nvidia-smi    # 須 VRAM 歸零再繼續

# ── OFF 相（oracle 8125，無 q4e hook）──
Start-Process -FilePath "D:\llama\llama-server.exe" -ArgumentList @(
  '-m','D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
  '--host','127.0.0.1','--port','8125','-ngl','10','--ctx-size','2048'
) -RedirectStandardOutput 'b8_4_off.log' -RedirectStandardError 'b8_4_off.err.log' -NoNewWindow
$deadline = (Get-Date).AddSeconds(300)
while ((Get-Date) -lt $deadline) {
  if ((curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8125/health) -eq "200") { break }
  Start-Sleep -Seconds 15
}
curl.exe -s -o NUL -w "off_health=%{http_code}\n" http://127.0.0.1:8125/health

curl.exe -s -o NUL -X POST http://127.0.0.1:8125/v1/chat/completions -H "Content-Type: application/json" -d "@body_on256.json"
curl.exe -s -X POST http://127.0.0.1:8125/v1/chat/completions -H "Content-Type: application/json" -d "@body_on256.json" -w "\nOFF time_total=%{time_total}\n" -o b8_4_off_resp.json
curl.exe -s -X POST http://127.0.0.1:8125/v1/chat/completions -H "Content-Type: application/json" -d "@body_b5_64.json" -o b8_5_oracle.json

Stop-Process -Name llama-server -Force
```
**預期輸出（v4.1 門）**：暖 t/s（ON）≥ 暖 t/s（OFF）——**硬線**（v4 為 ON 1.5692 < OFF 1.7700；A-9 移除 sync D2H 後期望翻綠）；`b8_6_stats.json`、`b8_5_q4e.json`、`b8_5_oracle.json` 產出。
**備案**：ON < OFF → **回報**（兩端 t/s、completion_tokens、time_total、雙方 log 尾段 30 行＋`b8_6_stats.json` 全欄位），不自行調參、不調預取參數。

## B-8-5 oracle 對答案（回歸門）

**指令**
```powershell
Get-Content b8_5_q4e.json -Raw -Encoding UTF8
Get-Content b8_5_oracle.json -Raw -Encoding UTF8
```
**預期輸出**：兩端 `choices[0].message`（`content`＋`reasoning_content`）一致，或差異可解釋（kernel/版本差 b10729 基 vs 10685＋prompt-cache 狀態差；v4 先例＝content 位元相同、reasoning 差 1 token）。
**備案**：不一致且無法解釋 → **回報原文**（兩份 JSON 全文），不猜。

## B-8-6 性能基準（v4.1 門：t/s＋重疊；N-gram 待 A-13）

**指令**
```powershell
# 1) A-11 拍板線（見「A 側已驗證」表＋02 §六 v4.1 修正）
# 2) M_real（A-9 後口徑：touched 層數 × tokens；stats 有 touched 層數欄位）
Get-Content b8_6_stats.json -Raw -Encoding UTF8
# 3) estimate_tps 表（對照用；v4.1 拍板線若＝Option A，此表僅記錄在案）
py -3.14 -X utf8 "D:\Nova2_home\AI研究\冷熱架構與t_s估算\estimate_tps.py"
```
**預期輸出（02 §六 v4.1）**：
1. **暖 t/s（ON）≥ A-11 拍板線**（Option A 時＝**≥ OFF 且 ≥ 1.95 t/s**；以 B-8-4 計數跑 t/s 為準）；
2. **預取重疊 ≥80%**：`b8_6_stats.json` 之 `prefetch.overlap_ratio`（A-10 定義：預取完成早於專家被使用之比例）；
3. N-gram 掉幅 <1%：**A-13 未接線→本 build 不可量**→照實回報（`prefetch` 全欄位原文），**不編造**。
**備案**：任一门不及 → **回報原文**（t/s、M_real、touches/hits/occ/prefetch 全欄位＋log 尾段），不猜、不 profile 代碼（profile 屬 A 端權限）。

## B-8-7 收尾（回歸門）

**指令**
```powershell
Get-Process | Where-Object { $_.ProcessName -match 'q4e-server|llama-server' }
nvidia-smi
Get-ChildItem "D:\AI_Model\Qwen3.8-Flash-Next\*.gguf"
```
**預期輸出**：無殘留程序（首條空）；VRAM **歸零**；GGUF 4 片在 D: SSD（合計 111,334,654,784 B）。
**備案**：未歸零/有殘留 → **回報原文**（`nvidia-smi`＋`Get-Process`），不自行處理其他程序。

## 驗收基準（v4.1；02 §六＋v4.1 修正）

1. q4e-server 載真 GGUF（**ngl=10**）＋`/v1` e2e 綠（200/401/SSE）。
2. **真實 M 可測**（核心看得到 router；口徑＝touched 層）。
3. **暖 t/s ≥ A-11 拍板線**（Option A：≥ OFF 且 ≥ 1.95 t/s；Option B 值＝拍板後填入）。
4. 預取重疊度（PCIe 段）≥80%（`prefetch.overlap_ratio`；A-10 後可量）。
5. N-gram 預取 t/s 掉幅 <1%（**待 A-13；本輪照實回報「不可量」**）。
6. 快取命中報告（vram/ram/ssd）＋I/O bytes＋真實 M（口徑注意：v4 起＝**CPU 層簿記**，GPU 層不計 touches）。
7. A 側 C++ 測試全綠（A-9/A-10 交付證據）。
8. 冷載入 ≤ ~250 s；收尾 VRAM 歸零。

## 紅線

**ngl=10 固定（不 999/48/auto）；不砍功能；不改 `engine/` source/test；不改快取/預取參數；永不過兩模型實例並行；oracle 只對答案。**

## 卡住規則

**任一步失敗→停**：回報「步號＋錯誤原文＋已試事項＋相關 log/證據」，不猜、不繞、不自行調參、不編造數字。
