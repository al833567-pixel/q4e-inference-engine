# RUNBOOK_B v3 — Task B（GPU 側）v4.1 復驗手冊（B-8，零判斷）

> **你是 Task B（有 GPU、較笨的 AI）**。本輪（B-8）目的＝重驗三門＋回歸門：① B-8-4 硬線 **ON ≥ OFF**；② B-8-6 暖 t/s（**ON ≥ OFF 且 ≥ 1.95 t/s**，A-11 拍板線）＋預取重疊 **≥80%**（`prefetch_overlap_ratio`）；③ N-gram 掉幅（**A-14.4 ✓ 可量**；B 側 GPU 量，工具 `q4e-ngram accept`）；回歸門＝e2e／oracle 對答案／冷載入 ≤~250 s／收尾 VRAM 歸零。
> **你的工作只有：B-8-0 → B-8-7 依序執行，對照預期輸出；任一步失敗→停下回報。** 不改 `engine/` 任何 source/test、不自行調參、不猜、不編造數字。
> 環境：Windows PowerShell（每步指令從 repo 根 `D:\Nova2_home\AI研究\推理工具\` 執行）；MSVC（vcvars64）＋CMake＋Ninja＋CUDA 13.3（nvcc）；RTX 3090 24 GB。
> 被測對象＝C++ 產品 `q4e-server`（`engine/`，A-9＋A-14.1 之後的 build）；oracle＝`D:\llama\llama-server.exe`（fork build 10685，**只對答案，代碼不進產品**）。
> **工具（A-12 交付包）**：若交付包內含 `q4e-e2e.exe`（A-14.2）／`q4e-bench.exe`（A-14.3），B-8-3／B-8-4 優先用它們跑（本手冊對應步會註明）；以下 curl 步驟＝備援。

## A 側已驗證（交付包 v4.1 內容；**B-8 動工前逐項對「✓」，任一未勾→停下回報 A**）

| 項 | 狀態 | 證據/commit |
|---|---|---|
| A-9 熱路徑開銷移除（`cb_eval` 不再對 device-resident ids 做 sync D2H；只簿記 CPU 層；stats 加 touched 層數） | **✓（`8413022`）** | 測試 98/98＋2989/2989；CPU e2e `A9_E2E_ALL_PASS` |
| A-11 驗收線拍板（M-C #3） | **✓（2026-09-06）：Option A＝#3′ 暖 t/s（ON）≥ OFF 且 ≥ 1.95 t/s** | 02 §六 |
| A-14.1 live 預取接線＋real backend（`q4e-expert-warm`；stats 加 `prefetch_io_issued`／`prefetch_io_bytes`／`prefetch_overlap_ratio`；`--q4e-prefetch on/off` 旗標） | **✓（commits `243dcac`／`0c204f7`／`43d06c8`／`d0aba15`／`9a9472b`）** | `evidence_a141/stats.json`（`prefetch_io_issued=13019>0`、`prefetch_overlap_ratio=0.6105`、`prefetch_overlap_hits=24144`、touches=39550、48 層）；旗標 `--q4e-prefetch on/off`＋`--no-q4e-prefetch`（OFF fallback 可用） |
| A-14.2 `q4e-e2e`／A-14.3 `q4e-bench`（C++ 客戶端，交付包內） | **✓（`19eaae3`／`471abae`）** | `evidence_a141/`（q4e-e2e `Q4E_E2E_ALL_PASS`）＋`evidence_a143/bench_report.json`（q4e-bench ON/OFF＋JSON；CPU smoke 的 perf verdict FAIL＝預期，perf 門在 B 側） |
| A-14.4 ngram live 接線＋`q4e-ngram` 工具（diag 4 階段／accept P1–P4；解鎖 M-C #5 可量） | **✓（`5b09765`／`c92606e`）** | `evidence_a144/`；CPU accept **verdict PASS**（max drop +0.22% < 1%）、diag A/B/C/D green；B 側 GPU 終驗 |
| C++ 測試全綠（test-q4e-expert-cache／test-q4e-ngram） | **✓（`9a9472b` 前 S5 全量 ctest）** | test-q4e-expert-cache 98/98、test-q4e-expert-warm 59/59、test-q4e-ngram 2989/2989、test-arg-parser OK；20 個上游 FAIL＝CJK 路徑環境問題（ASCII 相對路徑 20/20 PASS）；`S5-ctest.log`／`S5-failures.txt`（暫存 log 已清除、歸檔 git `bcaf1c9`） |
| CPU e2e 全綠＋`prefetch_io_issued>0` 證據 | **✓（`9a9472b`）** | `evidence_a141/stats.json`（`prefetch_io_issued=13019`）＋`Q4E_E2E_ALL_PASS` |
| 設定紅線：`--n-gpu-layers all`＋`--q4e-moe-vram 10`（K=10、K 可設定 0–999、VRAM 裝不下跳錯；舊 `-ncmoe/-cmoe` 已連 core 移除） | 新（A-14.0） | 01 §六／03B §三 |

> 本表由 A-12 填完後 B 端才可動工；B 端只核對「全部 ✓＋有 commit/證據」，不做技術判斷。

## B-8-0 前置確認

**指令**
```powershell
nvidia-smi
nvcc --version
Get-ChildItem "D:\AI_Model\Qwen3.8-Flash-Next\*.gguf"
Get-Process | Where-Object { $_.ProcessName -match 'q4e-server' }
# 交付包版本核對（須比 1144ddd 新）
engine\build-cuda\bin\q4e-server.exe --version
# 驗 oracle 的 -ncmoe（OFF 相旗標；若無→B-8-4 OFF 相改用 q4e --no-q4e-prefetch）
D:\llama\llama-server.exe --help 2>&1 | Select-String -Pattern "ncmoe|n-cpu-moe"
```
**預期輸出**：GPU＝RTX 3090 且 VRAM 空；`nvcc` V13.3.x；4 片 `.gguf` 共 ~103.7 GB；`q4e-server` 無殘留；`--version` commit **新於 `1144ddd`**；oracle help **有** `-ncmoe/--n-cpu-moe`（**若無→記入回報**：B-8-4 OFF 相改 q4e `--no-q4e-prefetch` 同 placement）。
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
- 若 incremental build 出現 `ninja: error: FindFirstFileExA(...): ... syntax is incorrect`（CJK 路徑增量陷阱，前輪 B-1 已見；根因見 git commit `24f11f7` 之 `evidence_b1/root_cause_b1.md`）→ **刪 `engine\build-cuda\.ninja_log` 後重試同一指令一次**；仍失敗→回報原文。
- 其他 CUDA/arch 錯誤 → **回報原文**（錯誤全文＋build 日誌尾段），不改 CMake 參數。

## B-8-2 真機載入（ngl=all＋K=10）

**指令**
```powershell
$q4e = Start-Process -FilePath "engine\build-cuda\bin\q4e-server.exe" -ArgumentList @(
  '--model','D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
  '--host','127.0.0.1','--port','8124','--api-key','bkey',
  '--n-gpu-layers','all','--q4e-moe-vram','10','--ctx-size','2048','--parallel','1'
) -RedirectStandardOutput 'b8_2_server.log' -RedirectStandardError 'b8_2_server.err.log' -NoNewWindow -PassThru
$deadline = (Get-Date).AddSeconds(300)
while ((Get-Date) -lt $deadline) {
  if ((curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8124/health) -eq "200") { break }
  Start-Sleep -Seconds 15
}
curl.exe -s -o NUL -w "health=%{http_code}\n" http://127.0.0.1:8124/health
nvidia-smi
```
**預期輸出**：`health=200`（**≤~250 s**；前輪實測 ~206 s）；VRAM 占用 **≈20,0xx–21,0xx MiB**（ngl=all＋K=10；K 過大裝不下→**清晰 VRAM 錯誤**、非 crash）；log 有 model loaded／listening。
**備案**：超時/失敗 → **回報原文**（`b8_2_server.err.log`＋`nvidia-smi`），不自動降 K。

## B-8-3 `/v1` e2e 5 閘＋`/q4e/stats`（回歸門）

> 優先用交付包 `q4e-e2e.exe`（A-14.2，G0–G6 全門自動）；以下 curl＝備援。

**指令**（curl 備援；body 直接 here-string 寫入暫存檔再 `@file` 發送——harness 會剝單引號 JSON 雙引號，前輪已證實）
```powershell
@'
{"messages":[{"role":"user","content":"What is a cache? Explain in one sentence."}],"max_tokens":64,"temperature":0,"stream":false}
'@ | Set-Content -Encoding UTF8 b8_body_64.json
@'
{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}
'@ | Set-Content -Encoding UTF8 b8_body_401.json
@'
{"messages":[{"role":"user","content":"What is a cache? Explain in one sentence."}],"max_tokens":64,"temperature":0,"stream":true}
'@ | Set-Content -Encoding UTF8 b8_body_sse.json
@'
{"messages":[{"role":"user","content":"Write an essay about caching in computer systems."}],"max_tokens":256,"temperature":0,"stream":false}
'@ | Set-Content -Encoding UTF8 b8_body_256.json

curl.exe -s -o NUL -w "%{http_code}\n" http://127.0.0.1:8124/health
curl.exe -s -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@b8_body_64.json"
curl.exe -s -o NUL -w "%{http_code}\n" -X POST http://127.0.0.1:8124/v1/chat/completions -H "Content-Type: application/json" -d "@b8_body_401.json"
curl.exe -s -N -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@b8_body_sse.json"
curl.exe -s http://127.0.0.1:8124/q4e/stats -H "Authorization: Bearer bkey"
```
**預期輸出**：(a) 200；(b) 200＋`usage.completion_tokens≥1`（token 可能在 `reasoning_content`，正常）；(c) 401；(d) 多行 `data:`＋`data: [DONE]`；(e) 200＋JSON 含 `hits_*`、`hit_rate_*`、**`prefetch_io_issued`（A-14.1 後為數字，e2e 後應 >0）**、`prefetch_overlap_ratio`、touched 層數欄位。
**備案**：任一閘不綠 → **回報原文**（該閘完整回應體＋status＋log 尾段）。

## B-8-4 快取開 vs 關硬線（**期望 ON ≥ OFF**）

> 量測紀律：temp=0、固定 prompt、**2×256 tokens**（第 1 跑＝warmup、第 2 跑＝計數）；t/s＝`usage.completion_tokens ÷ time_total`。
> 並行紅線：ON/OFF 串行；kill→VRAM 歸零確認→再起 oracle。**oracle 只以 PID kill（`Stop-Process -Id`），絕不用 `Stop-Process -Name llama-server`**（harness 的模型服務是同一二進位，按名稱 kill 會誤殺）。

**指令**（repo 根）
```powershell
# ── ON 相（q4e-server 8124，B-8-2 已起、已暖）──
curl.exe -s -o NUL -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@b8_body_256.json"
curl.exe -s -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@b8_body_256.json" -w "\nON  time_total=%{time_total}\n" -o b8_4_on.json
curl.exe -s http://127.0.0.1:8124/q4e/stats -H "Authorization: Bearer bkey" -o b8_6_stats.json
curl.exe -s -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@b8_body_64.json" -o b8_5_q4e.json

# ── 切換（只 kill 本輪起的 q4e PID）──
Stop-Process -Id $q4e.Id -Force
Start-Sleep -Seconds 5
nvidia-smi    # 須 VRAM 歸零再繼續

# ── OFF 相（oracle 8125，無 q4e hook；PassThru 拿 PID）──
# OFF＝oracle（上游 b10685，有自己原生 -ncmoe；-ncmoe 38＝38 層 MoE→CPU＝等同 K=10）
# 若 B-8-0 顯示 oracle 無 -ncmoe→OFF 相改用 q4e `--no-q4e-prefetch`（同 placement、無預取）
$oracle = Start-Process -FilePath "D:\llama\llama-server.exe" -ArgumentList @(
  '-m','D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
  '--host','127.0.0.1','--port','8125','-ngl','all','-ncmoe','38','--ctx-size','2048'
) -RedirectStandardOutput 'b8_4_off.log' -RedirectStandardError 'b8_4_off.err.log' -NoNewWindow -PassThru
$deadline = (Get-Date).AddSeconds(300)
while ((Get-Date) -lt $deadline) {
  if ((curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8125/health) -eq "200") { break }
  Start-Sleep -Seconds 15
}
curl.exe -s -o NUL -w "off_health=%{http_code}\n" http://127.0.0.1:8125/health

curl.exe -s -o NUL -X POST http://127.0.0.1:8125/v1/chat/completions -H "Content-Type: application/json" -d "@b8_body_256.json"
curl.exe -s -X POST http://127.0.0.1:8125/v1/chat/completions -H "Content-Type: application/json" -d "@b8_body_256.json" -w "\nOFF time_total=%{time_total}\n" -o b8_4_off_resp.json
curl.exe -s -X POST http://127.0.0.1:8125/v1/chat/completions -H "Content-Type: application/json" -d "@b8_body_64.json" -o b8_5_oracle.json

# ── 只 kill 本輪 oracle PID ──
Stop-Process -Id $oracle.Id -Force
```
**預期輸出（v4.1 門）**：暖 t/s（ON）≥ 暖 t/s（OFF）——**硬線**（前輪 ON 1.5692 < OFF 1.7700；A-9 移除 sync D2H＋A-14.1 real backend 後期望翻綠）；`b8_6_stats.json`、`b8_5_q4e.json`、`b8_5_oracle.json` 產出。
**備案**：ON < OFF → **回報**（兩端 t/s、completion_tokens、time_total、雙方 log 尾段 30 行＋`b8_6_stats.json` 全欄位），不自行調參、不調預取參數。

## B-8-5 oracle 對答案（回歸門）

**指令**
```powershell
Get-Content b8_5_q4e.json -Raw -Encoding UTF8
Get-Content b8_5_oracle.json -Raw -Encoding UTF8
```
**預期輸出**：兩端 `choices[0].message`（`content`＋`reasoning_content`）一致，或差異可解釋（kernel/版本差 b10729 基 vs 10685＋prompt-cache 狀態差；前輪先例＝content 位元相同、reasoning 差 1 token）。
**備案**：不一致且無法解釋 → **回報原文**（兩份 JSON 全文），不猜。

## B-8-6 性能基準（門：t/s＋重疊；N-gram 掉幅 A-14.4 ✓ 可量）

> 優先用交付包 `q4e-bench.exe`（A-14.3，ON/OFF t/s＋重疊＋JSON 一鍵）；以下＝讀取 B-8-4 產物的備援。

**指令**
```powershell
Get-Content b8_6_stats.json -Raw -Encoding UTF8
```
**預期輸出（門）**：
1. **暖 t/s（ON）≥ OFF 且 ≥ 1.95 t/s**（A-11 拍板線；以 B-8-4 計數跑 t/s 為準）；
2. **預取重疊 ≥80%**：`b8_6_stats.json` 之 `prefetch_overlap_ratio`（A-14.1 定義：觸碰前已完成預取的 touches 佔比）＋`prefetch_io_issued>0`；
3. N-gram 掉幅 <1%：**A-14.4 ✓ 可量**（`q4e-ngram accept` 工具已入包；CPU 預驗 PASS max drop +0.22%）→B 側 GPU 量（`prefetch` 全欄位原文），**不編造**。
**備案**：任一门不及 → **回報原文**（t/s、touches/hits/occ/prefetch 全欄位＋log 尾段），不猜、不 profile 代碼（profile 屬 A 端權限）。

## B-8-7 收尾（回歸門）

**指令**
```powershell
Get-Process | Where-Object { $_.ProcessName -match 'q4e-server' }
nvidia-smi
Get-ChildItem "D:\AI_Model\Qwen3.8-Flash-Next\*.gguf"
```
**預期輸出**：無 `q4e-server` 殘留（首條空）；VRAM **歸零**；GGUF 4 片在 D: SSD（合計 111,334,654,784 B）。
**備案**：未歸零/有殘留 → **回報原文**（`nvidia-smi`＋`Get-Process`），不自行處理其他程序（**絕不 kill `llama-server`**——harness 模型服務）。

## 驗收基準（v4.1）

1. q4e-server 載真 GGUF（**ngl=all＋K=10**）＋`/v1` e2e 綠（200/401/SSE）。
2. **真實 M 可測**（核心看得到 router；口徑＝touched 層）。
3. **暖 t/s ≥ 拍板線**（ON ≥ OFF 且 ≥ 1.95 t/s）。
4. 預取重疊度（PCIe 段）≥80%（`prefetch_overlap_ratio`）。
5. N-gram 預取 t/s 掉幅 <1%（**A-14.4 ✓ 可量**；B 側 GPU 量；工具 `q4e-ngram accept`）。
6. 快取命中報告（vram/ram/ssd）＋I/O bytes＋真實 M（口徑＝**CPU 層簿記**，GPU 層不計 touches）。
7. A 側 C++ 測試全綠（交付證據）。
8. 冷載入 ≤ ~250 s；收尾 VRAM 歸零。

## 紅線

**設定＝`--n-gpu-layers all --q4e-moe-vram 10`（K=10；K 可設定 0–999、VRAM 裝不下跳錯）；不砍功能；不改 `engine/` source/test；不改快取/預取/K 參數；永不過兩模型實例並行；oracle 只對答案；kill 只按 PID、絕不 `Stop-Process -Name llama-server`。**

## 卡住規則

**任一步失敗→停**：回報「步號＋錯誤原文＋已試事項＋相關 log/證據」，不猜、不繞、不自行調參、不編造數字。
