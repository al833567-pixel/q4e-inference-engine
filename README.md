# 推理工具 — Qwen3.8-Flash-Next（qwen4exp）推理

> **v2（2026-09-01）**：自有引擎 **`q4e`** ＝ clone llama.cpp 最新 source → 改名 → **改核心**（專家級動態快取＋PLE 離載）→ 砍到單模型（只留 qwen4exp）。
> Python 端只保留**合約參考＋spec 測試**：/v1 API 合約（server.py+auth.py+tests）、快取狀態機（cache.py）、PLE 合約（ngram.py）。
> 生產引擎＝C++ `q4e-server`。v1（llama-cpp-python wheel）路線已退役 → [`legacy_v1/`](legacy_v1/)。

## 文件

| 文件 | 說明 |
|---|---|
| [01_架構系統資料.md](01_架構系統資料.md) | 架構、記憶體佈局、/v1 設計、效能目標（仍有效） |
| [02_實作項目工作規劃.md](02_實作項目工作規劃.md) | **v2 工作規劃**：路線決策＋P0 可行性門→P1 快取核心→P2 server+驗收→P3 收口 |
| [03_項目現況.md](03_項目現況.md) | 環境、已驗證基準數字、項目現況 |
| [legacy_v1/](legacy_v1/) | v1（A/B 兩軌）RUNBOOK＋驗收報告＋工具＋36 份證據（封存） |

## 目錄結構

```
推理工具/
├── 01/02/03 .md                  # 架構 / v2 規劃 / 現況
├── cache.py + tests/test_cache.py    # **v2 核心快取 spec**（LRU 狀態機，14 測試向量）
├── ngram.py + tests/test_ngram.py    # **PLE 合約 spec**（FNV hash / 4.5-bit 解碼）
├── server.py / auth.py + tests/      # **/v1 API 合約 oracle**（OpenAI 相容子集＋401＋SSE）
├── config.py / config.yaml / interfaces.py  # 設定＋合約層
├── tools/
│   ├── b5_body.json        # e2e 請求體（v2 P2.2 用）
│   └── bench_ngram.py      # PLE 表 SSD 隨機讀 bench（CPU-only）
└── legacy_v1/              # v1 路線封存（tools / evidence / RUNBOOK / B9 報告）
```

## 快速啟動（合約參考，無 GPU 即可）

```powershell
py -3.14 -m pip install -r requirements.txt
$env:DSH_API_KEY="sk-dev"
py -3.14 -m uvicorn server:app --port 8080     # StubEngine（確定性、不載模型）
curl -H "Authorization: Bearer sk-dev" -X POST http://127.0.0.1:8080/v1/chat/completions -d @"tools\b5_body.json"
py -3.14 -m pytest tests/ -q                    # 47 項全綠（含真 GGUF 解析）
py -3.14 tools\bench_ngram.py --n 1000          # PLE 表 SSD 隨機讀
```

## 狀態

- **v1（A/B 兩軌）：全數完成並驗收**（M-A、M-B）；證據＝`legacy_v1/evidence/`（36 檔）＋`legacy_v1/B9_report.md`
- **v2：未動工**；工具鏈已確認（VS 2022 BuildTools 在位）；下一步＝P0.2（clone 最新 llama.cpp→改名 `q4e`→base build）
- 紅線：ngl=10 固定；永不過兩個模型實例並行；單模型（qwen4exp）
