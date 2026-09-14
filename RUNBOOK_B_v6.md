# RUNBOOK_B v6 — Task B（GPU 側）v6 真機復驗手冊（B-16，零判斷）

> **你是 Task B（有 GPU、照本手冊零判斷執行）**。本輪（B-16）目的＝v6 三層專家快取真機驗收：**① ON 暖 t/s ≥ 8（理想 ≥ 40）；② VRAM-fit（N=5000 載入成功、VRAM ≤ 24 GB）；③ 100k q8_0 載入無 OOM（KV 分頁 L1 64k/L2 32k pinned/L3 SSD）；④ Oracle 對答案；⑤ 回歸門（e2e 5 閘＋冷載入 ≤~250 s＋收尾 VRAM 歸零）**。
> **你的工作只有：B-16-0 → B-16-8 依序執行，對照預期輸出；任一步失敗→停下回報。** 不改 `engine/` 任何 source/test、不自行調參、不猜、不編造數字。
> 環境：Windows PowerShell（每步指令從 repo 根 `D:\Nova2_home\AI研究\推理工具\` 執行）；MSVC（vcvars64）＋CMake＋Ninja＋CUDA 13.3（nvcc）；RTX 3090 24 GB；~20 GB RAM；SATA SSD。
> 被測對象＝C++ 產品 `q4e-server`（`engine/`，**Task A v6 交付版**）；oracle＝`D:\llama\llama-server.exe`（**只對答案，代碼不進產品**）。
> 測試模型：`D:\AI_Model\Qwen3.8-Flash-Next\`（4 片 Q4_K_XL，≈103.7 GiB）。
> 工具（v6 交付包）：`q4e-e2e.exe`／`q4e-bench.exe` 優先使用；本手冊 curl 步驟＝備援。

### 編號對照表（一次性；舊編號→02 正典，本文件正文已全部改用 02 正典）

| 舊編號（本文件舊版） | 02 正典（現行） |
|---|---|
| A-16.0 v6 設計拍板＋參數語義 | A-16.0（不變） |
| A-16.1 全局動態池 | A-16.1（不變） |
| A-16.2 VRAM 內 Gather | A-16.2（不變） |
| A-16.3 H2D 管線 | A-16.3（不變） |
| A-16.4 Graph 手術 | A-16.4（不變） |
| A-16.5（原：Server 接線＋`/q4e/stats` v6） | **A-16.6** |
| A-16.6（原：q4e-bench／q4e-e2e v6） | **A-16.7** |
| A-16.7（原：交付包 v6） | **A-16.8** |
| —（舊版無） | **A-16.5**（100k KV 分頁，新增） |

## 前置（Task A v6 交付確認；**B-16 動工前逐項核對，任一未勾→停下回報 A**）

| 項 | 狀態 |
|---|---|
| A-16.0 v6 設計拍板＋參數語義（`--q4e-moe-vram N` 全局池 expert 數／`--q4e-moe-ram R`／`--q4e-staging-mb S`／`--q4e-layer-floor F`／`--q4e-ple-mmap on|off`（預設 on）／`--q4e-kv-ram` 100k） | 須 ✓ |
| A-16.1 全局動態池管理器（Global LRU＋Layer-Aware Floor＋M 計數）C++ 測試全綠 | 須 ✓ |
| A-16.2 VRAM 內 Gather（Staging Buffer 雙緩衝＋CUDA Kernel 介面）C++ 測試全綠 | 須 ✓ |
| A-16.3 H2D 提升管線（L2 Pinned RAM＋SSD→L2 fill＋非同步 H2D）C++ 測試全綠 | 須 ✓ |
| A-16.4 Graph 手術（48 層全 GPU＋動態 slot 映射＋promotion op）Oracle 對答案＋C++ 測試 | 須 ✓ |
| A-16.5 100k KV 分頁（page=32 tokens＋L1/L2/L3 tier＋LRU promote/demote＋100k oracle 滑窗）C++ 測試全綠 | 須 ✓ |
| A-16.6 Server 接線＋`/q4e/stats` v6（七群組：pool/staging/pipe/map/promo/kv/meta） | 須 ✓ |
| A-16.7 q4e-bench／q4e-e2e v6（含 `--v6`／`--v6-kv100k`） | 須 ✓ |
| A-16.8 交付包 v6（README/01/02/03A/03B 全落地） | 須 ✓ |

> 本表由 Task A 填完後 B 端才可動工；B 端只核對「全部 ✓」，不做技術判斷。

## B-16-0 前置確認

**指令**
```powershell
nvidia-smi
nvcc --version
Get-ChildItem "D:\AI_Model\Qwen3.8-Flash-Next\*.gguf"
Get-Process | Where-Object { $_.ProcessName -match 'q4e-server' }
# 交付包版本核對（v6 交付版 commit）
engine\build-cuda\bin\q4e-server.exe --version
```
**預期輸出**：GPU＝RTX 3090 且 VRAM 空；`nvcc` V13.3.x；4 片 `.gguf` 共 ≈103.7 GiB（111,334,654,784 B）；`q4e-server` 無殘留；`--version`＝v6 交付版 commit。
**備案**：任一不符 → **停下回報**（原文），不猜。

## B-16-1 q4e CUDA build（v6 source）

**指令**（repo 根；vcvars64 包起來）
```powershell
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cmake -S engine -B engine/build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DGGML_CPU_REPACK=OFF && cmake --build engine/build-cuda --target q4e-server'
Test-Path engine\build-cuda\bin\q4e-server.exe
engine\build-cuda\bin\q4e-server.exe --version
```
**預期輸出**：build 結束無錯誤；`Test-Path` → **True**；`--version`＝A 側交付版。
**備案**：
- 若 incremental build 出現 `ninja: error: FindFirstFileExA(...): ... syntax is incorrect`（CJK 路徑增量陷阱）→ **刪 `engine\build-cuda\.ninja_log` 後重試同一指令一次**；仍失敗→回報原文。
- 其他 CUDA/arch 錯誤 → **回報原文**（錯誤全文＋build 日誌尾段），不改 CMake 參數。

## B-16-2 真機載入（v6 預設參數）

**指令**
```powershell
$q4e = Start-Process -FilePath "engine\build-cuda\bin\q4e-server.exe" -ArgumentList @(
  '--model','D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
  '--host','127.0.0.1','--port','8124','--api-key','bkey',
  '--n-gpu-layers','all','--q4e-moe-vram','5000','--q4e-moe-ram','4000',
  '--q4e-staging-mb','128','--q4e-layer-floor','10',
  '--ctx-size','2048','--parallel','1'
) -RedirectStandardOutput 'b16_2_server.log' -RedirectStandardError 'b16_2_server.err.log' -NoNewWindow -PassThru
$deadline = (Get-Date).AddSeconds(300)
while ((Get-Date) -lt $deadline) {
  if ((curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8124/health) -eq "200") { break }
  Start-Sleep -Seconds 15
}
curl.exe -s -o NUL -w "health=%{http_code}\n" http://127.0.0.1:8124/health
nvidia-smi
```
**預期輸出**：`health=200`（**≤~250 s**）；VRAM 占用 **≤ 24 GB（≈22,0xx–24,0xx MiB 為正常區間：Attn/Router/Shared＋全局池 N=5000≈15.4 GiB＋Staging 128 MB＋KV）**；log 有 model loaded／listening。**（VRAM-fit 門：裝不下＝清晰 VRAM 錯誤，非 crash）**
**備案**：超時/失敗 → **回報原文**（`b16_2_server.err.log`＋`nvidia-smi`），不自動降 N。

## B-16-3 `/v1` e2e 5 閘＋`/q4e/stats` v6（回歸門）

> 優先用交付包 `q4e-e2e.exe`（G0–G6 全門自動）；以下 curl＝備援。

**指令**（curl 備援；body 直接 here-string 寫入暫存檔再 `@file` 發送——harness 會剝單引號 JSON 雙引號）
```powershell
@'
{"messages":[{"role":"user","content":"What is a cache? Explain in one sentence."}],"max_tokens":64,"temperature":0,"stream":false}
'@ | Set-Content -Encoding UTF8 b16_body_64.json
@'
{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}
'@ | Set-Content -Encoding UTF8 b16_body_401.json
@'
{"messages":[{"role":"user","content":"What is a cache? Explain in one sentence."}],"max_tokens":64,"temperature":0,"stream":true}
'@ | Set-Content -Encoding UTF8 b16_body_sse.json
@'
{"messages":[{"role":"user","content":"Write an essay about caching in computer systems."}],"max_tokens":256,"temperature":0,"stream":false}
'@ | Set-Content -Encoding UTF8 b16_body_256.json

curl.exe -s -o NUL -w "%{http_code}\n" http://127.0.0.1:8124/health
curl.exe -s -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@b16_body_64.json"
curl.exe -s -o NUL -w "%{http_code}\n" -X POST http://127.0.0.1:8124/v1/chat/completions -H "Content-Type: application/json" -d "@b16_body_401.json"
curl.exe -s -N -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@b16_body_sse.json"
curl.exe -s http://127.0.0.1:8124/q4e/stats -H "Authorization: Bearer bkey"
```
**預期輸出**：(a) 200；(b) 200＋`usage.completion_tokens≥1`（token 可能在 `reasoning_content`，正常）；(c) 401；(d) 多行 `data:`＋`data: [DONE]`；(e) 200＋JSON 含 **v6 七群組：pool/staging/pipe/map/promo/kv/meta（內含 L1/L2/L3 命中分佈＋Gather 統計＋真實 M（expert 計數））**（欄位名以 A-16.8 交付文件為準）。
**備案**：任一閘不綠 → **回報原文**（該閘完整回應體＋status＋log 尾段）。

## B-16-4 性能基準（門：**ON 暖 t/s ≥ 8**，理想 ≥ 40）

> 量測紀律：temp=0、固定 prompt、**2×256 tokens**（第 1 跑＝warmup、第 2 跑＝計數）；t/s＝`usage.completion_tokens ÷ time_total`。
> 並行紅線：串行量測；永不過兩個模型實例並行。

**指令**
```powershell
curl.exe -s -o NUL -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@b16_body_256.json"
curl.exe -s -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@b16_body_256.json" -w "\nON  time_total=%{time_total}\n" -o b16_4_on.json
curl.exe -s http://127.0.0.1:8124/q4e/stats -H "Authorization: Bearer bkey" -o b16_6_stats.json
curl.exe -s -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d "@b16_body_64.json" -o b16_5_q4e.json
```
**預期輸出**：計數跑 **暖 t/s ≥ 8**（理想：L1 全命中時 ≥ 40；L1 hit 越高越快）；`b16_4_on.json`／`b16_6_stats.json` 產出。
**備案**：t/s < 8 → **回報**（t/s、completion_tokens、time_total、`b16_6_stats.json` 全欄位＋log 尾段 30 行），不自行調參、不調快取參數。

## B-16-5 Oracle 對答案（回歸門）

**指令**（只 kill 本輪起的 q4e PID→VRAM 歸零→再起 oracle）
```powershell
Stop-Process -Id $q4e.Id -Force
Start-Sleep -Seconds 5
nvidia-smi    # 須 VRAM 歸零再繼續
$oracle = Start-Process -FilePath "D:\llama\llama-server.exe" -ArgumentList @(
  '-m','D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
  '--host','127.0.0.1','--port','8125','-ngl','all','--ctx-size','2048'
) -RedirectStandardOutput 'b16_5_oracle.log' -RedirectStandardError 'b16_5_oracle.err.log' -NoNewWindow -PassThru
$deadline = (Get-Date).AddSeconds(300)
while ((Get-Date) -lt $deadline) {
  if ((curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8125/health) -eq "200") { break }
  Start-Sleep -Seconds 15
}
curl.exe -s -o NUL -w "off_health=%{http_code}\n" http://127.0.0.1:8125/health
curl.exe -s -X POST http://127.0.0.1:8125/v1/chat/completions -H "Content-Type: application/json" -d "@b16_body_64.json" -o b16_5_oracle.json
Stop-Process -Id $oracle.Id -Force
```
**預期輸出**：q4e（B-16-4 之 `b16_5_q4e.json`）與 oracle（`b16_5_oracle.json`）之 `choices[0].message`（`content`＋`reasoning_content`）**一致**（v6＝48 層全 GPU＋全局池＋Gather，與 oracle 同 `-ngl all` 佈局；允許可解釋的 kernel/版本差）。
**備案**：不一致且無法解釋 → **回報原文**（兩份 JSON 全文），不猜。

## B-16-6 命中分佈＋Gather 統計（v6 核心觀測）

**指令**
```powershell
Get-Content b16_6_stats.json -Raw -Encoding UTF8
```
**預期輸出（觀測門）**：
1. **L1 命中分佈存在且合理**（48 層共享全局池 N=5000；Layer-Aware Floor F=10 生效＝每層 ≥10 槽不被擠出）；
2. **Gather 統計有值**（Staging Buffer 雙緩衝複製事件計數；e2e 後 >0）；
3. 真實 M 計數（touched expert 口徑）與 48 層 router 選擇一致（每 token 每層 10 routed）。
> 此步為**觀測**（非 t/s 門）：數值異常→回報原文，由 A 側裁決。

## B-16-7 100k KV 分頁驗證（v6 新門）

> KV 分頁對齊 A-16.5 交付（`q4e-kv-paging`＋`q4e-kv-bridge`）：page＝32 tokens；tier＝L1 64k（VRAM）／L2 32k（pinned RAM）／L3（SSD）；100k 極限配置＝`--ctx-size 100000 --cache-type-k q8_0 --q4e-kv-ram 6000 --q4e-moe-ram 3000`（**本步唯一允許的配置變更，屬固定配置，非自行調參**）。
> 前置：B-16-5 已 kill 2048 server（`$q4e`），VRAM 已歸零。

**快速驗證指令（優先，交付包工具）**
```powershell
engine\build-cuda\bin\q4e-e2e.exe --v6 --v6-kv100k
engine\build-cuda\bin\q4e-bench.exe --v6
```
**備援（手動 100k 載入）**
```powershell
nvidia-smi    # 須 VRAM 歸零再繼續
$q4e100k = Start-Process -FilePath "engine\build-cuda\bin\q4e-server.exe" -ArgumentList @(
  '--model','D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
  '--host','127.0.0.1','--port','8126','--api-key','bkey',
  '--n-gpu-layers','all','--q4e-moe-vram','5000','--q4e-moe-ram','3000',
  '--q4e-staging-mb','128','--q4e-layer-floor','10','--q4e-ple-mmap','on',
  '--ctx-size','100000','--cache-type-k','q8_0','--q4e-kv-ram','6000','--parallel','1'
) -RedirectStandardOutput 'b16_7_100k.log' -RedirectStandardError 'b16_7_100k.err.log' -NoNewWindow -PassThru
$deadline = (Get-Date).AddSeconds(600)
while ((Get-Date) -lt $deadline) {
  if ((curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8126/health) -eq "200") { break }
  Start-Sleep -Seconds 15
}
curl.exe -s -o NUL -w "health100k=%{http_code}\n" http://127.0.0.1:8126/health
curl.exe -s http://127.0.0.1:8126/q4e/stats -H "Authorization: Bearer bkey" -o b16_7_stats100k.json
nvidia-smi
Stop-Process -Id $q4e100k.Id -Force
```
**預期輸出**：載入成功（`health100k=200`）**無 OOM**、VRAM 不爆（≤ 24 GB）；`b16_7_stats100k.json` 含 v6 七群組 **pool/staging/pipe/map/promo/kv/meta**，`kv` 群組見分頁運作（L1 64k/L2 32k pinned/L3 SSD，page=32 tokens）。
**備案**：OOM／載入失敗→**停下回報原文**（`b16_7_100k.err.log`＋`nvidia-smi`＋stats JSON 原文），不自動降參、不重試。

## B-16-8 收尾（回歸門）

**指令**
```powershell
Get-Process | Where-Object { $_.ProcessName -match 'q4e-server|llama-server' }
nvidia-smi
Get-ChildItem "D:\AI_Model\Qwen3.8-Flash-Next\*.gguf"
```
**預期輸出**：無 `q4e-server` 殘留；VRAM **歸零**；GGUF 4 片在 D: SSD（合計 111,334,654,784 B）。
**備案**：未歸零/有殘留 → **回報原文**（`nvidia-smi`＋`Get-Process`），不自行處理其他程序（**絕不 `Stop-Process -Name llama-server`**——harness 模型服務；kill 只按 PID）。

## 驗收基準（v6）

1. **t/s**：ON 暖 t/s **≥ 8**（理想 L1 全命中 ≥ 40；依賴 L2 PCIe 補齊）。
2. **100k**：`--ctx-size 100000 --cache-type-k q8_0` 載入成功**無 OOM**、VRAM 不爆、KV 分頁正常運作（B-16-7）。
3. **VRAM-fit**：`N=5000` 載入成功、VRAM ≤ 24 GB。
4. **Oracle 對答案**：48 層全 GPU＋全局池＋Gather 逐層輸出＝oracle。
5. **回歸門**：e2e 5 閘綠＋冷載入 ≤~250 s＋收尾 VRAM 歸零。
6. **觀測**：L1/L2/L3 命中分佈＋Gather 統計＋真實 M（B-16-6）；stats 七群組（pool/staging/pipe/map/promo/kv/meta）齊全。
> **A 側前置補註**：CPU 測試矩陣 16 項已全綠（引 `03A_項目現況_TaskA.md` §三，含 promotion oracle 覆蓋＋100k KV 滑窗 oracle）；本冊僅做真機驗證，不重跑 CPU 測試。

## 紅線

**設定＝`--n-gpu-layers all --q4e-moe-vram 5000 --q4e-moe-ram 4000 --q4e-staging-mb 128 --q4e-layer-floor 10`（100k 步 B-16-7 用固定極限配置 `--ctx-size 100000 --cache-type-k q8_0 --q4e-kv-ram 6000 --q4e-moe-ram 3000`，唯一允許的配置變更）；不砍功能；不改 `engine/` source/test；不改快取/Gather/池參數；永不過兩模型實例並行；oracle 只對答案；kill 只按 PID、絕不 `Stop-Process -Name llama-server`。**

## 卡住規則

**任一步失敗→停**：回報「步號＋錯誤原文＋已試事項＋相關 log/證據」，不猜、不繞、不自行調參、不編造數字。
