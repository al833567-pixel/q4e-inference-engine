# RUNBOOK_B v2 — Task B（GPU 側）執行手冊（零判斷）

> **你是 Task B（有 GPU、較笨的 AI）**。Task A 已完成設計＋寫碼（C++ `engine/` q4e 樹）＋CPU build＋C++ 測試＋CPU e2e，見下方「A 側已驗證」。
> **你的工作只有：B-0 → B-7 依序執行，對照預期輸出；任一步失敗→停下回報。** 不改 `engine/` 任何 source/test、不自行調參、不猜。
> 技術基底：Windows PowerShell（每步指令從 repo 根 `D:\Nova2_home\AI研究\推理工具\` 執行）；MSVC（vcvars64）＋CMake 4.4.2＋Ninja 1.13.2＋CUDA 13.3（nvcc）；RTX 3090 24 GB。
> 被測對象＝C++ 產品 `q4e-server`（`engine/`）；oracle＝`D:\llama\llama-server.exe`（fork build 10685，**只對答案，代碼不進產品**）。

## A 側已驗證（交付包內容，B 端不用重做）

| 項 | 狀態 |
|---|---|
| CPU build | ✓ 綠（MSVC 14.44 static Release；`engine\build-msvc\bin\q4e-server.exe` 可執行） |
| C++ 測試 | ✓ **test-q4e-ngram 2957 checks**＋**test-q4e-expert-cache 82 checks** 全綠（oracle 向量凍結於測試內） |
| CPU e2e（真 GGUF 103.7 GB） | ✓ `/health` 200／≥1 token（16）／401／SSE `data: [DONE]`／`/q4e/stats` 活數據（`cb_eval` hook 真觸發：touches=10720、io_bytes_estimate=79 GB）——證據 [evidence_A5_cpu_e2e.md](evidence_A5_cpu_e2e.md) |
| A-5 commit | `5d98d54`（expert-cache `cb_eval` hook 接線＋`/q4e/stats` 端點） |
| 設定紅線 | [config.yaml](config.yaml)：`ngl: 10`、GGUF 4 片、模組／端點指標 |

## B-0 前置確認

**指令**
```powershell
nvidia-smi
nvcc --version
Get-ChildItem "D:\AI_Model\Qwen3.8-Flash-Next\*.gguf"
Get-Process | Where-Object { $_.ProcessName -match 'q4e-server|llama-server' }
```
**預期輸出**：GPU＝RTX 3090 **≥24 GB** 且 VRAM 空（無殘留 server 實例，末條指令回傳空）；`nvcc`＝**release 13.3**（本機實測 V13.3.73）；4 片 `.gguf`（`...-00001-of-00004.gguf`…`00004`）共 **~103.7 GB**。
**備案**：toolkit 版本不是 13.3 → **回報 A**（裝對應 toolkit 或 A 調 build 參數），不猜。VRAM 未空→先請使用者關掉占用程序再報。

## B-1 CUDA build

**指令**（repo 根；vcvars64 包起來，PATH 上的 MinGW 會搶編譯器）
```powershell
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cmake -S engine -B engine/build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DGGML_CPU_REPACK=OFF && cmake --build engine/build-cuda --target q4e-server'
Test-Path engine\build-cuda\bin\q4e-server.exe
```
**預期輸出**：build 結束無錯誤（exit 0）；`Test-Path` → **True**（`engine\build-cuda\bin\q4e-server.exe` 產生）。
**備案**：CUDA/arch 編譯錯誤 → **回報原文**（錯誤訊息全文＋build 日誌尾段），不自行改 CMake 參數。

## B-2 真機載入（GPU，ngl=10）

**指令**（repo 根；背景執行＋留 log）
```powershell
Start-Process -FilePath "engine\build-cuda\bin\q4e-server.exe" -ArgumentList @(
  '--model','D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
  '--host','127.0.0.1','--port','8124','--api-key','bkey',
  '--n-gpu-layers','10','--ctx-size','2048','--parallel','1'
) -RedirectStandardOutput 'b2_server.log' -RedirectStandardError 'b2_server.err.log' -NoNewWindow

# 輪詢就緒（冷載入預算 300 s；v1 參考 226.4 s）
$deadline = (Get-Date).AddSeconds(300)
while ((Get-Date) -lt $deadline) {
  if ((curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8124/health) -eq "200") { break }
  Start-Sleep -Seconds 15
}
curl.exe -s -o NUL -w "health=%{http_code}\n" http://127.0.0.1:8124/health
nvidia-smi
```
**預期輸出**：`health=200`（**≤~250 s** 內，v1 參考 226.4 s）；`nvidia-smi` 顯示 VRAM 占用（ngl=10，v1 實測約 **16,750 MiB**＋KV/緩衝 < 24 GB）；`b2_server.log` 有 model loaded／server listening 行。
**備案**：300 s 未就緒或載入失敗 → **回報原文**（`b2_server.err.log`＋`nvidia-smi` 輸出），不猜、不自動降 ngl。

## B-3 /v1 e2e＋/q4e/stats（GPU，5 閘同 A-5）

**指令**（repo 根；prompt 用 ASCII 避開 cp950 編碼坑）
```powershell
# (a) 健康
curl.exe -s -o NUL -w "%{http_code}\n" http://127.0.0.1:8124/health

# (b) ≥1 token（非串流）
curl.exe -s -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"What is a cache? Explain in one sentence."}],"max_tokens":64,"temperature":0,"stream":false}'

# (c) 無 key → 401
curl.exe -s -o NUL -w "%{http_code}\n" -X POST http://127.0.0.1:8124/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}'

# (d) SSE 串流
curl.exe -s -N -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"What is a cache? Explain in one sentence."}],"max_tokens":64,"temperature":0,"stream":true}'

# (e) q4e 統計
curl.exe -s http://127.0.0.1:8124/q4e/stats -H "Authorization: Bearer bkey"
```
**預期輸出**：(a) **200**；(b) **200**＋`usage.completion_tokens ≥ 1`（⚠️ 推理模型：token 可能全在 `choices[0].message.reasoning_content`、`content` 空字串是**正常**——閘是 `completion_tokens≥1`，不是 content 非空）；(c) **401**；(d) 多行 `data: {...}` SSE＋末行 **`data: [DONE]`**；(e) **200**＋JSON 含 `hits_vram/hits_ram/hits_ssd`、`hit_rate_*`、`io_bytes_estimate`，且 `touches` 非零（＝`cb_eval` hook 真觸發）。
**備案**：任一閘不綠 → **回報原文**（該閘完整回應體＋status code＋`b2_server.log` 尾段），不猜。

## B-4 快取驗（開 vs 關，硬線：開 ≥ 關）

> **現況說明（事實，非判斷）**：A-5 接線＝`cb_eval` **狀態機 hook**（`server-context.cpp` 恆常接上，**無 OFF 旗標**；live backend＝no-op，只統計、不做真實 PCIe 搬移，A-5 證據 `prefetch_issued=0`）。故「關快取」基線＝**oracle 二進位**（`D:\llama`，無 q4e hook）——本硬線驗的是「hook 不回歸 MoE 熱路徑」。
> 量測紀律（03B §五）：temp=0、固定 prompt、**2×256 tokens**（第 1 跑＝warmup，第 2 跑＝計數）；t/s＝`usage.completion_tokens ÷ time_total`。
> **並行紅線（03B §五）**：永不過兩個模型實例同時載入 → **ON／OFF 串行**：先量 ON（B-2 的 q4e）→ kill＋VRAM 歸零確認 → 再起 oracle 量 OFF。順帶抓 B-5 對答案與 B-6 stats（各 server 只載一次，省冷載入）。

**指令**（repo 根）
```powershell
# ── ON 相（q4e-server 8124，B-2 已起、已暖）──
# ON warmup（不計數）
curl.exe -s -o NUL -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Write an essay about caching in computer systems."}],"max_tokens":256,"temperature":0,"stream":false}'
# ON 計數跑（t/s_ON = completion_tokens ÷ time_total）
curl.exe -s -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Write an essay about caching in computer systems."}],"max_tokens":256,"temperature":0,"stream":false}' -w "\nON  time_total=%{time_total}\n"
# 順帶抓 B-6 stats＋B-5 對答案（q4e 還在跑）
curl.exe -s http://127.0.0.1:8124/q4e/stats -H "Authorization: Bearer bkey" -o b6_stats.json
$p5 = '{"messages":[{"role":"user","content":"What is a cache? Explain in one sentence."}],"max_tokens":64,"temperature":0,"stream":false}'
curl.exe -s -X POST http://127.0.0.1:8124/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d $p5 -o b5_q4e.json

# ── 切換（先 kill q4e，VRAM 歸零後再起 oracle）──
Stop-Process -Name q4e-server -Force
Start-Sleep -Seconds 5
nvidia-smi    # 須 VRAM 歸零再繼續

# ── OFF 相（oracle 8125，無 q4e hook）──
# 注：oracle 不需 auth（無 key 時忽略 Authorization header，故不帶 --api-key 旗標降低未知旗標風險）
Start-Process -FilePath "D:\llama\llama-server.exe" -ArgumentList @(
  '-m','D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
  '--host','127.0.0.1','--port','8125',
  '-ngl','10','--ctx-size','2048'
) -RedirectStandardOutput 'b4_off.log' -RedirectStandardError 'b4_off.err.log' -NoNewWindow

# 輪詢 OFF 就緒（同 B-2，port 8125／300 s）
$deadline = (Get-Date).AddSeconds(300)
while ((Get-Date) -lt $deadline) {
  if ((curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8125/health) -eq "200") { break }
  Start-Sleep -Seconds 15
}
curl.exe -s -o NUL -w "off_health=%{http_code}\n" http://127.0.0.1:8125/health

# OFF warmup（不計數）
curl.exe -s -o NUL -X POST http://127.0.0.1:8125/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Write an essay about caching in computer systems."}],"max_tokens":256,"temperature":0,"stream":false}'
# OFF 計數跑（t/s_OFF = completion_tokens ÷ time_total）
curl.exe -s -X POST http://127.0.0.1:8125/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Write an essay about caching in computer systems."}],"max_tokens":256,"temperature":0,"stream":false}' -w "\nOFF time_total=%{time_total}\n"
# 順帶抓 B-5 對答案
curl.exe -s -X POST http://127.0.0.1:8125/v1/chat/completions -H "Authorization: Bearer bkey" -H "Content-Type: application/json" -d $p5 -o b5_oracle.json

# ── 收尾 OFF ──
Stop-Process -Name llama-server -Force
```
**預期輸出**：暖 t/s（ON，q4e hook 開）≥ 暖 t/s（OFF，oracle 無 hook）——**硬線**（02 §八：快取改 MoE 熱路徑→回歸→先驗無快取對齊再驗快取）；`b6_stats.json`、`b5_q4e.json`、`b5_oracle.json` 三檔產出。
**備案**：ON < OFF → **回報**（兩端 t/s、completion_tokens、time_total、雙方 log 尾段），**不自行調參**。oracle 為 build 10685、q4e 為 b10729：若基線差距大到無法比對（版本差），回報事實由 A 判定。

## B-5 oracle 對答案

**指令**（repo 根；比對 B-4 抓下的兩份 JSON）
```powershell
Get-Content b5_q4e.json -Raw -Encoding UTF8
Get-Content b5_oracle.json -Raw -Encoding UTF8
# 比對兩份 choices[0].message 的 reasoning_content＋content
```
**預期輸出**：兩端 `choices[0].message`（`reasoning_content`＋`content`）一致，或差異可解釋（kernel/版本差：10685 vs b10729）。
**備案**：不一致且無法解釋 → **回報原文**（兩份 JSON 全文），不猜。

## B-6 性能基準（02 §六）

**指令**（repo 根；warm t/s 用 B-4 的計數跑，stats 用 `b6_stats.json`）
```powershell
# 1) estimate_tps 基準表（M → t/s；M=20/44/90/142/164/256/512 列）
py -3.14 -X utf8 "D:\Nova2_home\AI研究\冷熱架構與t_s估算\estimate_tps.py"

# 2) M_real 代理（本 build /q4e/stats 只公開 touches 總數）：
#    M_real_lower = touches ÷ (48 層 × B-4 計數跑 completion_tokens)
Get-Content b6_stats.json -Raw -Encoding UTF8

# 3) 驗收線：estimate_tps 表中 ≥ M_real_lower 的最小 M 列 × 0.7
#    （top-10 路由下 M_real_lower ≈ 10–11 → 取 M=20 列）
```
**預期輸出**（02 §六 三行）：
1. 暖 t/s ≥ `estimate_tps(M)`×0.7（基準參照：無快取暖 1.95 t/s；v1 實測 1.95＝M=512 上界列仍達標）；
2. 預取重疊度（PCIe 段）≥80%；
3. N-gram 預取 t/s 掉幅 <1%（交錯 A/B 方法學）。

> **現況說明（事實，非判斷）**：A-5 接線為狀態機（no-op backend）——若 `b6_stats.json` 的 `prefetch.issued`＝0（A-5 證據即為 0），則第 2/3 行**本 build 無 live 預取可量**→**回報 A 現況**（A 決定是否先補 live 接線再驗），**不編造重疊數字**。
**備案**：任一線不及 → profile MoE 分發/kernel；**回報原文**（t/s、M_real_lower、touches/hits/occ/prefetch 全欄位），不猜。

## B-7 收尾

**指令**（repo 根；B-4 已 kill 兩端，此步＝驗證）
```powershell
Get-Process | Where-Object { $_.ProcessName -match 'q4e-server|llama-server' }
nvidia-smi
Get-ChildItem "D:\AI_Model\Qwen3.8-Flash-Next\*.gguf"   # PLE 表（28.8 GB）留 SSD
```
**預期輸出**：無殘留 q4e-server／llama-server 程序（首條回傳空）；VRAM **歸零**（`nvidia-smi` 顯示無占用）；PLE 表（`per_layer_token_embd`）留 **D: SSD**（離載，不佔 VRAM；檔案 handle 隨程序退出釋放）。
**備案**：VRAM 未歸零或程序未退 → **回報原文**（`nvidia-smi`＋`Get-Process` 輸出），不自行處理其他程序。

---

## 驗收基準（02 §六，M-C 8 項）

1. q4e-server 載真 GGUF（**ngl=10**，VRAM≈16,750 MiB）＋`/v1` e2e 綠（200/401/SSE）。
2. **真實 M 可測**（核心看得到 router）→ 解掉 v1「M 引擎外不可測」gap。
3. 暖 t/s ≥ `estimate_tps(M_real)` × 0.7（70% 規則；基準參照：無快取暖 1.95 t/s）。
4. 預取重疊度（PCIe 段）≥80%（v1 外部黑箱下物理不可達；核心 hook＋真實 router 局部性下才可驗）。
5. N-gram 預取 t/s 掉幅 <1%（交錯 A/B 方法學）。
6. 快取命中報告（vram/ram/ssd）＋I/O bytes＋真實 M。
7. A 側 C++ 測試全綠（A-3 82 checks＋A-4 2957 checks＋A-5 e2e）。
8. 冷載入 ≤ ~250 s（參考 226.4 s）；收尾 VRAM 歸零。

## 紅線

**Qwen3.8-Flash-Next 完整載入使用＝ngl=10＋推理（真實 M）＋`/v1`＋專家快取＋PLE 離載；不為乾淨砍功能。** B 端：不改 ngl（不 999/48/auto）、不改快取參數、不砍功能、不改 `engine/` source/test。

## 卡住規則

**任一步失敗→停**：回報「步號＋錯誤原文＋已試事項＋相關 log/證據」，不猜、不繞、不自行調參。
