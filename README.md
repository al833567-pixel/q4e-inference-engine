# 推理工具 — Qwen3.8-Flash-Next MoE 推理（兩層快取 + /v1 + API key）

> 一個「類 llama.cpp」的推理工具：載 **Qwen3.8-Flash-Next GGUF**、實作**兩層快取**（VRAM 2×激活 / RAM N×一輪未命中 / SSD 冷 + N-gram 離載）、對外 **OpenAI 相容 `/v1` + API key**。
> 技術基底：**Python 3.14 + `llama-cpp-python`（llama.cpp 當函式庫，不改 C++）+ FastAPI**。只有 GGUF。
> **兩 AI 分工**：聰明 AI（Track A，GPU 不可用）做設計/CPU 側/介面/手冊；略笨 AI（Track B，GPU 可用）照 `RUNBOOK_B.md` 做 GPU 側。

## 文件

| 文件 | 說明 |
|---|---|
| [01_架構系統資料.md](01_架構系統資料.md) | 架構、技術決策、記憶體佈局、/v1 設計、效能目標 |
| [02_實作項目工作規劃.md](02_實作項目工作規劃.md) | **兩 AI 分工**：Track A（A-0～A-5）/ Track B（B-1～B-6）+ 介面合約 |
| [03_項目現況.md](03_項目現況.md) | 環境、各項目前況、已確認約束、缺口 |
| [RUNBOOK_B.md](RUNBOOK_B.md) | **Track B 操作手冊**（略笨 AI 照此做，零判斷 + 備案） |
| [interfaces.py](interfaces.py) | **介面合約**（`InferenceEngine`/`CacheManager`/`NgramTable`，A 定義簽章 → B 實作） |
| [config.yaml](config.yaml) | 設定（server/engine/cache/ngram） |
| [requirements.txt](requirements.txt) | 相依 |

## 兩層快取（CPU 式，不訓練模型）

| 層 | 內容 | 大小 | 速度 |
|---|---|---|---|
| L1 VRAM | 熱專家 2×激活 + 非專家權重 | ~8–15 GB | 677 GB/s |
| L2 RAM | N×一輪未命中（pinned） | ≤18 GB | PCIe 12 GB/s |
| L3 SSD | 冷專家 + N-gram 表（28.8 GB） | ~100 GB | 0.44 GB/s |

判斷機制＝CPU/OS 現成：**LRU 取代 + 需求分頁 + 確定性預取（上一 token 選擇）+ 快取過濾（只預取進 RAM）+ 雙緩衝**。**不訓練模型**。

## 快速啟動（Track A stub 模式，無 GPU 現在就能跑）

```powershell
py -3.14 -m pip install -r requirements.txt
$env:DSH_API_KEY="sk-dev"
py -3.14 -m uvicorn server:app --port 8080     # 預設 engine=stub，不載模型
curl -H "Authorization: Bearer sk-dev" -X POST http://127.0.0.1:8080/v1/chat/completions -d @"req.json"
py -3.14 -m pytest tests/ -q                    # 47 項全綠（含真 GGUF 解析）
py -3.14 tools\bench_ngram.py --n 1000          # N-gram 表 SSD 隨機讀（CPU-only）
py -3.14 tools\wire_cache.py --fake             # 兩層快取 LRU 模擬（CPU-only）
py -3.14 tools\bench_tps.py --plan              # t/s 規劃表 + 驗收行
```

## 快速啟動（Track B 有 GPU 時）

```powershell
# 1. 裝相依（Windows/無 MSVC 可能需預編 wheel，見 RUNBOOK_B §備案 A）
py -3.14 -m pip install -r requirements.txt
# 2. 載 GGUF 冒煙
py -3.14 tools\smoke_load.py
# 3. 起 server（config.yaml 改 server.engine: llama）
py -3.14 -m uvicorn server:app --port 8080
# 4. 呼叫（設 DSH_API_KEY）
curl -H "Authorization: Bearer <key>" -X POST http://127.0.0.1:8080/v1/chat/completions -d @"req.json"
# 5. 兩層快取 / N-gram offload / t/s 驗收
py -3.14 tools\wire_cache.py --live
py -3.14 tools\ngram_offload.py --live
py -3.14 tools\bench_tps.py --measure
```

> 完整步驟、預期輸出、備案見 [RUNBOOK_B.md](RUNBOOK_B.md)。

## 狀態（M-A 已達）

- **Track A（聰明 AI，GPU-free）— 完成**：
  - A-0 規劃/介面合約/config/RUNBOOK_B ✓
  - A-1 `server.py`（`/v1` + API key + 串流 SSE + stub/llama 引擎）✓ 實機 curl 驗證（200/401/SSE/[DONE]）
  - A-2 `cache.py`（LRU + 需求分頁 + 確定性預取 + 快取過濾 + 命中統計，B 只覆寫 `place()/evict()`）✓
  - A-3 `ngram.py`（真 GGUF 解析：160×320,001,536、**4.5 bit/值實測確認**、20,000,096 entries）+ `tools/bench_ngram.py` ✓（1000 隨機讀：p50 ≈1.0 ms）
  - A-4 B 工具 scaffolds（`smoke_load`/`wire_cache`/`ngram_offload`/`bench_tps`，CPU 路徑可跑、GPU 路徑標「B 填」）✓
  - A-5 `pytest tests/ -q` **47 項全綠** ✓
- **Track B（略笨 AI，GPU）**：M-A 已達，可照 RUNBOOK_B 開始（前置：釋放 VRAM）。
