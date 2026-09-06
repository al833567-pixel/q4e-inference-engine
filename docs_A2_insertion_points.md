# qwen4exp 注入點（q4e 核心 hook 位置）

> 用途：q4e 核心（專家快取／PLE 離載／A-14.1 real backend）掛進 qwen4exp 引擎的**精確位置**＋**現有機制**（不新造輪子）。
> 證據＝本 repo `engine/`（llama.cpp b10729 fork）行號；實測常數＝`legacy_v1/tools/live_common.py`。

## 一、MoE 分派路徑（touch() 掛點）

### 1. 圖建構（graph build）
- `src/models/qwen4exp.cpp:912` `build_layer_ffn` → 調 `build_moe_ffn`（:916）。
- 本 GGUF **沒有** merged `ffn_gate_up_exps` 張量（`llama-model.cpp:3143` TENSOR_NOT_REQUIRED→nullptr）
  → graph 走**分離路徑**：3 個 `MUL_MAT_ID`（`ffn_up_exps` / `ffn_gate_exps` / `ffn_down_exps`）。
- `src/llama-graph.cpp:1894/1938` `build_moe_ffn`：
  1. `logits = mm(gate_inp, cur)`（:1970）— router 輸入投影（`blk.N.ffn_gate_inp.weight` [n_embd,512]）
  2. `probs = softmax(logits)`（:1988，qwen4exp 用 SOFTMAX，qwen4exp.cpp:925）
  3. **`selected_experts = ggml_argsort_top_k(probs, n_expert_used=10)`**（:2054）→ shape `[10, n_tokens]` ← **真實 router 選擇**
  4. `weights = get_rows(probs, selected_experts)`（:2068）
  5. 3× `MUL_MAT_ID` 專家矩陣乘（:2114 起）

### 2. 執行時讀取（A-3 hook，已實作於 `common/q4e-expert-cache.cpp`）
- **機制**：`llama_context_params.cb_eval`（`include/llama.h:386`；型別
  `ggml_backend_sched_eval_callback = bool(ggml_tensor*, bool ask, void*)`，ggml-backend.h）。
  ask 協議：`ask=true` 表態對該 op 有興趣 → `ask=false` 時 op 輸出已算完、可讀值。
- **先例**：`tools/imatrix/imatrix.cpp:234` `collect_imatrix` — 同樣讀 `MUL_MAT_ID` 的
  `t->src[2]`（ids `[n_expert_used, n_tokens]`），用 `ggml_backend_tensor_get` 取回 host。
- **q4e hook**（`q4e_expert_cache_cb_eval`）：
  - filter：`op == MUL_MAT_ID && strstr(src0->name, "up")` → 每 MoE 層**恰好 1 個 op**
    （`blk.N.ffn_up_exps.weight`；`down`/`gate` 不匹配；shexp 是常規 MUL_MAT，不會匹配）。
  - layer id：`q4e_expert_layer_from_name(src0->name)`（`"blk.N."` 解析；單測覆蓋）。
  - 讀法：column-major，token t 的專家 k ＝ `ids_data[k * n_tokens + t]`；每 token 調
    `cache->touch(il, ids, 10)`。
- **註冊點**（A-5）：server 建立 context 時 `params.cb_eval = q4e_expert_cache_cb_eval;
  params.cb_eval_user_data = &cache;`（參照 `common/debug.cpp:33`、`tools/imatrix/imatrix.cpp:1154`）。

### 3. 專家佈局（實測，v1 `live_common.py`）
| 張量 | dims [n_embd, n_ff, n_expert] | 位元 | stride/專家 |
|---|---|---|---|
| `blk.N.ffn_gate_exps.weight` | [2560, 640, 512] | 4.5 bit | 921,600 B |
| `blk.N.ffn_up_exps.weight` | [2560, 640, 512] | 4.5 bit | 921,600 B |
| `blk.N.ffn_down_exps.weight` | [640, 2560, 512] | 6 bit | 1,228,800 B |
- 一個 routed 專家＝3 slices＝**3,072,000 B**（≈3.1 MB，與 01 檔一致）；expert ＝ ggml 最外維（e 佔 `[base + e*stride, +stride)`）。
- shared 專家：`blk.N.ffn_{gate,up,down}_shexp.weight`（獨立張量；graph 常規 `MUL_MAT`＋sigmoid gate，qwen4exp.cpp:933-951）。
- **v1 place/evict 語義**（CPU 層）：hot tier（vram/ram）物理上＝「讀進 OS page cache」→ mmap slice read
  （與引擎自帶 mmap 共享同一 page cache，預取同時暖引擎）；evict＝no-op（Windows 無 drop_caches）。
  GPU 層（L < ngl）整層住 VRAM＝零 I/O。

## 二、PLE 路徑（A-4 掛點）

### 1. 引擎端（upstream qwen4exp 原生）
- KV：`q4e.ple_layers=[1]`、`ple_ngram_size=3`、`ple_heads_per_ngram=8`、`ple_conv_kernel=8`、
  `ple_eos_token_id=248044` → **16 heads**（8 bigram + 8 trigram）；`n_embd_per_layer=160`。
- 表：`per_layer_token_embd.weight` dims **[160, 320,001,536]**＝28.8 GB、**恰好 4.5 bit/值**（18 B/32 值）；
  `qwen4exp.cpp:139` 以 **`TENSOR_READ_LAZY`** 建立＝**已在盤上（mmap），不進 RAM**。
- hash（`qwen4exp.cpp:962-964` 註＋host 側實作）：
  `mixed_n = (t[p]*m[0]) ^ (t[p-1]*m[1]) ^ (t[p-n+1]*m[n-1])`；`row = mixed_n % vocab[h] + offset[h]`；
  每 head 的 `offset`/`vocab` 來自 GGUF `LLM_KV_PLE_HEAD_OFFSETS/HEAD_VOCAB_SIZES`；**EOS 重置窗口**。
  （ggml 沒有 int64/xor → 引擎在 host 側算 row id，再 `ggml_get_rows` 取行。）
- 每 token 取 16 行 × 160 值 × 4.5 bit ＝ **1,440 B**（≈1.4 KB）。

### 2. A 側合約（`engine/tests/test-q4e-ngram.cpp`＝FNV/RNG/解碼向量凍結；原 ngram.py/test_ngram.py 已於 v4 移除）
- 解碼：`d = f16 LE (2B)`＋`qs[16]`（byte j 低 nibble＝元素 2j、高 nibble＝2j+1）；`x[i] = (q[i]-8)*d`。
- row 長＝5 blocks × 18 B＝**90 B**；entry＝16 subrows；n_entries＝n_rows/16。
- A 側測試 hash＝FNV-1a64(head salt, ids) % n_entries × 16 + head（確定性、可重放；**測試用**，
  真引擎 row id 以上游 metadata hash 為準——兩者可並存：合約測邏輯，真檔測 I/O）。
- `offload(path)`/`prefetch(ids)`＝B 覆寫點（真實 host/SSD＋非同步預取）；A 的 fake 版只記路徑＋同步讀。
- bench：`bench_random_reads(n, seed)` → `{n, bytes_per_read, mean/p50/p99/min/max_ms, throughput_mbps, mode}`；
  `read_sequence(n, seed)` 可重放。

### 3. A-4 引擎端掛點（C++ 離載＋預取）
- row id 可**提前**在 host 算出（hash 純 token 函數）→ 下一 token 的 16 行可與當前計算重疊讀取（mmap readahead）。
- 掛法：eval 後由 server loop 對「已生成 token 序列」預取下一 token 的 16 行（與 A-3 同路徑；
  或 A-5 用 cb_eval 之後的 host 側計算——**A-4 先做 CPU 核心＋bench，引擎掛法在 A-5 驗證**）。

## 三、Server 端（A-5）

| 項目 | 狀態 | 位置 |
|---|---|---|
| `/v1/chat/completions` | 上游有 | `server-http.cpp` |
| `/v1/models`、`/v1/health` | 上游有 | `server-http.cpp:33-37` |
| Bearer auth | 上游有（`--api-key`；`Authorization`/`X-Api-Key`） | `server-http.cpp:206-231` |
| `/metrics` | 上游＝**proxy 轉發**（非 Prometheus 統計） | `server-models.cpp:1900`（`proxy_get`） |
| q4e 統計端點 | **需新增**（M/hits/I-O；Prometheus 文字格式） | A-5 新增 route |
| cb_eval 註冊 | 上游無（imatrix 是 tool 層用法） | A-5 在 context 參數設定 |

## 四、build/測試紀律（本次實測新增）
- **cmd.exe ANSI 轉換會把中文路徑吃掉**（`cmd /c "D:\...\AI研究\..."` → 路徑碎裂、命令找不到）；
  **解法**：vcvars 用純 ASCII 命令導入（`cmd /c '"...vcvars64.bat" >nul && set'` → pwsh 注入環境變數），
  `ninja -C <中文路徑>` **直接由 PowerShell 執行**（Unicode 安全）→ reconfigure＋build 成功。
- reconfigure（改 CMakeLists）本次**未**撞 ninja 路徑編碼錯（上次是改參數＋cmd 層問題重疊）；
  若再撞＝新目錄重建（舊規則仍有效）。
- C++ 單元測試：`tests/test-<name>.cpp`（自帶 `main()`、CHECK 宏）＋`llama_build_and_test(...)`
  掛 `llama-common`（`tests/CMakeLists.txt:154` 區塊，Win32 靜態 build 啟用）。
