# A-5 CPU e2e 證據（q4e-server，真 GGUF）

> 產生：2026-07-18。目標：證 A-5 的 `cb_eval` hook + `/q4e/stats` 端點在**真模型 CPU 推理**下生效。
> 方法：`q4e-server.exe` 載真 GGUF（CPU，`--n-gpu-layers 0`），對 `/v1` 端點 + `/q4e/stats` 做 e2e。

## 命令
```
engine\build-msvc\bin\q4e-server.exe
  --model "D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
  --host 127.0.0.1 --port 8123 --api-key testkey123 --n-gpu-layers 0 --ctx-size 2048 --parallel 1
```
- 模型：177B params，103.69 GB，Q4_K-M，4 分片（自動全載）；CPU mmap＋page-file（31.8 GB RAM）。
- 就緒：`/health` 503→503→**200**（~1-2 min，權重逐頁載入）。

## 閘（全綠）
| 閘 | 請求 | 結果 |
|---|---|---|
| (a) 健康 | `GET /health` | **200** `{"status":"ok"}` |
| (b) ≥1 token | `POST /v1/chat/completions`（Bearer key，max_tokens 16） | **200**，`completion_tokens:16`（prompt 59 / total 75） |
| (c) 401 | `POST /v1/chat/completions`（無 key） | **401** `{"error":{"message":"Invalid API Key","type":"authentication_error","code":401}}` |
| (d) SSE | `POST /v1/chat/completions`（key，`stream:true`） | **200**，19 筆 `data:` 行 + 末行 `data: [DONE]` |
| (e) q4e 統計 | `GET /q4e/stats`（key） | **200**，見下 JSON |

### (e) `/q4e/stats` 回傳（真實，累計 b/c/d 之後）
```json
{"layers":48,"touches":10720,"hits_vram":256,"hits_ram":398,"hits_ssd":10066,
 "hit_rate_vram":0.0239,"hit_rate_ram":0.0371,"hit_rate_ssd":0.939,
 "prefetch_issued":0,"occ_vram_avg":20.0,"occ_ram_avg":69.1,
 "placement_events":25760,"io_bytes_estimate":79134720000}
```
→ **`cb_eval` hook 在真推理中確實觸發**（`touches`/`hits`/`placement_events` 非零；CPU 下 ssd 層命中高＝page-file 後端），證明「真實 M 可測」（02 §六 2）＋「快取命中報告 vram/ram/ssd＋I/O」（§六 6）。

## 誠實註記（非缺陷）
- (b) `choices[0].message.content` 為**空字串**：本模型是**推理型**（Qwen3.8-Flash-Next），16 token 全花在 `reasoning_content`（SSE (d) 可見 `delta.reasoning_content` 逐字串流），`finish_reason:"length"`（撞到 max_tokens 上限）。**`completion_tokens:16`＝已產生 ≥1 token**，e2e 實質達成；`content` 空是「推理模型＋max_tokens=16」的產物，非 server 缺陷（調大 max_tokens 即有正式 content）。

## 結論
A-5 達成：**CPU e2e 全綠**（真 GGUF + ≥1 token + 200/401/SSE + `/q4e/stats` 真數據）。`q4e-server`（含 `cb_eval` hook + q4e 統計端點）已建置＋驗證。
