# RUNBOOK_B — Track B 操作手冊（略笨 AI 照此做，零判斷）

> **你是 Track B（有 GPU 的 AI）**。聰明 AI（Track A）已把「介面、邏輯、config、測試」都做完。
> **你的工作只有：照下面 1→10 逐步跑、看預期輸出、卡住就停下回報。** 不要改介面簽章（`interfaces.py`）、不要重寫快取邏輯（`cache.py`）。
> 技術基底：Python 3.14（`py -3.14`）+ `llama-cpp-python` + FastAPI。只有 GGUF。
> 所有路徑以本專案資料夾 `D:\Nova2_home\AI研究\推理工具\` 為基準。

---

## 0. 前置（必做）

| 步驟 | 指令 | 預期 |
|---|---|---|
| 0.1 確認 GPU 已釋放 | `nvidia-smi` | 有空 VRAM（**先關 `llama-server.exe`**，它佔 ~24 GB） |
| 0.2 確認 GGUF 在位 | `dir D:\AI_Model\Qwen3.8-Flash-Next` | 4 片 `.gguf`（00001…00004），共 ~103.7 GB |
| 0.3 確認 Python | `py -3.14 --version` | `Python 3.14.x` |
| 0.4 確認相依已裝 | `py -3.14 -c "import llama_cpp, fastapi"` | 不報錯 |

> 若 0.4 報 `llama_cpp` 找不到 → 跳到 **§備案 A**（llama-cpp-python 沒裝成功）。

---

## 1. 載 GGUF + 冒煙（B-1）

- 指令：`py -3.14 tools\smoke_load.py`
- 做什麼：載 4 片 GGUF（`-ngl` 設 VRAM 層）→ 固定 prompt → 產 ≥1 token。
- **預期**：印出「loaded, n_params=…, first_token=…」+ 1 個 token；VRAM 佔用落在 config 預算內。
- **失敗處理**：
  - `out of memory` / OOM → 把 `config.yaml` 的 `engine.ngl` 調小（少放幾層上 GPU），重跑。
  - 找不到 GGUF → 檢查 `config.yaml` 的 `engine.gguf_dir`。
  - `llama_cpp` import 失敗 → **§備案 A**。

## 2. 起 API server（B-4）

- 指令：`py -3.14 -m uvicorn server:app --host 127.0.0.1 --port 8080`
- **預期**：uvicorn 啟動、`GET /health` 回 `{"status":"ok"}`。
- 注意：server 用 `server.py`（A 已寫好）；推理接 `InferenceEngine`（你實作 `MyEngine`）。

## 3. 驗 `/v1` + API key（B-5）

- 先設 key：`$env:DSH_API_KEY="sk-test123"`（或改 `config.yaml` 的 `server.api_keys`）。
- 指令：
  ```
  curl -H "Authorization: Bearer sk-test123" -X POST http://127.0.0.1:8080/v1/chat/completions -d @"req.json"
  ```
- **預期**：回 OpenAI 相容 JSON（`choices[0].message.content` 有文字）。
- **無 key 應回 401**：`curl -X POST http://127.0.0.1:8080/v1/chat/completions -d @"req.json"`（不帶 Authorization）→ `401`。
- `req.json` 範例（A 提供）：
  ```json
  { "model": "qwen3.8-flash-next",
    "messages": [{"role":"user","content":"你好"}],
    "max_tokens": 16, "stream": false }
  ```

## 4. 接兩層快取（B-2）

- 指令：`py -3.14 tools\wire_cache.py`
- 做什麼：把 A 的 `CacheManager`（`cache.py`，LRU 邏輯）的 `place()/evict()` 接你的真實 placement（`-ngl` offload / pinned H2D）。
- **預期**：印出「每層命中分佈 {vram,ram,ssd}」+ t/s 落在 M 對應列 70%+（對照 [../冷熱架構與t_s估算/estimate_tps.py](../冷熱架構與t_s估算/estimate_tps.py)）。
- **失敗處理**：命中分佈全落 ssd → `ram_capacity_per_layer` 太小或 placement 沒生效；回報聰明 AI。

## 5. N-gram offload（B-3）

- 指令：`py -3.14 tools\ngram_offload.py`
- 做什麼：把 N-gram 表（28.8 GB）離到 host/SSD + 非同步預取（A 的 `ngram.py` 解析已寫好，你覆寫 `offload()/prefetch()`）。
- **預期**：N-gram 住 host/SSD；t/s 相對「N-gram 住 RAM」掉 **<1%**。
- **失敗處理**：解碼值不對 → 回報聰明 AI（A-3 的 4.5-bit 解碼）。

## 6. 投機預取 + 重疊（B-4）

- 指令：`py -3.14 tools\bench_tps.py --prefetch`
- **預期**：重疊度 ≥80%（PCIe 段）；t/s 較「串聯」提升。

## 7. 基準 + 驗收（B-5）

- 指令：`py -3.14 tools\bench_tps.py`
- **預期**：量出 **M（每層不同專家數）** + t/s，對照 estimate_tps：
  | M/層 | 目標 t/s |
  |---|---|
  | ≤20 | ~95 |
  | ~44 | ~15 |
  | ~90 | ~11 |
  | ≤162 | **7–15（PCIe）** |
  | >162 | 0.4–0.8（SSD） |
- **驗收**：t/s ≥ M 對應列的 70%。

## 8. 整合冒煙（B-6）

- 指令：`py -3.14 -m pytest tests/ -q`（GPU-free 測試）+ 端到端 curl 一次 `/v1`。
- **預期**：pytest 全綠 + `/v1` 回正常 JSON。

## 9. 回報（必做）

回報聰明 AI（Track A）：
1. 每步（1–8）的**輸出**（特別是 t/s、命中分佈、M）。
2. **任何錯誤訊息原文**（別 paraphrase）。
3. `nvidia-smi` 的 VRAM 佔用。
4. 若卡住：卡在第幾步、錯誤訊息、你試過什麼。

---

## 備案 A — llama-cpp-python 沒裝成功（無 MSVC / 無 3.14 wheel）

1. **試預編 wheel**：`py -3.14 -m pip install llama-cpp-python --prefer-binary`；或找 3.12/3.11 的 wheel 裝到對應 venv。
2. **改用現有 `D:\llama\llama-server.exe` 跑載模型+推理**（它已能載 GGUF + 出 /v1）：
   - 起 server：`D:\llama\llama-server.exe -m D:\AI_Model\Qwen3.8-Flash-Next\...gguf -ngl 48 --port 8080`
   - 你的 Python（Track A 的 `server.py`/`cache.py`/`ngram.py`）改**外接** `http://127.0.0.1:8080/v1`（API 層 + 快取邏輯 + N-gram 離載仍由 Python 做）。
3. **回報聰明 AI** 哪條路走通，再決定要不要繼續 `llama-cpp-python`。

## 備案 B — VRAM 不夠（OOM）

1. 調小 `config.yaml` 的 `engine.ngl`（少放層上 GPU）。
2. 或把更多專家放 RAM（`ram_capacity_per_layer` 調大，吃 PCIe）。
3. 仍 OOM → 回報聰明 AI，評估降 `max_ctx` 或改量化。

## 約束（別越界）

- ❌ 不改 `interfaces.py` 的簽章。
- ❌ 不重寫 `cache.py`（LRU 邏輯）/ `ngram.py`（解析）。
- ✅ 你只實作 `InferenceEngine`、覆寫 `CacheManager.place()/evict()`、`NgramTable.offload()/prefetch()`，照 `interfaces.py` 簽章填。
- ✅ 卡住就**停下回報**，別自己猜。
