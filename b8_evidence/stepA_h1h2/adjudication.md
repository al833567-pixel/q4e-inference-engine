# B-8-2 H1/H2 裁決（A 側）— 2026-09-10

> 交接來源：[03B §一 B-8-2 空 GPU 重跑證據](../../03B_項目現況_TaskB.md) 之「兩假說（H1 override 未生效 / H2 估算錯）」。
> 本檔＝A 側交付三件：(a) required 組成之 per-buft breakdown；(b) H1→修復＋重構＋新 exe；(c) 新預期 VRAM 數字（B 側驗證線）。

## 結論：H1 成立

build ≤39（`0ccc16c`、build 39）的 `--q4e-moe-vram K` **override 完全未生效（loader 套用到 0 筆）**，故 48 層 routed experts 全留 CUDA → VRAM-fit 實測 **78,431 / 78,768 MiB**（B 側 `no_alloc` 確定性測量、與 GPU 占用無關）＝預期 ≈20.6 GB 之 3.8×。

**根因（source 證據，非猜）**：
1. `common_params_parse_ex`（`engine/common/arg.cpp:945-949`）在解析後**無條件**把 `params.tensor_buft_overrides` pad 到 `llama_max_tensor_buft_overrides()`＝4096 筆 `{nullptr,nullptr}`（為 `llama_params_fit` 的 buffer 界線；`fit.cpp` 會對 <4096 的 buffer throw）。
2. 其後 `common_q4e_moe_vram_apply`（`common.cpp:1311`，build ≤39 呼叫 `llm_add_q4e_moe_vram_overrides`）把 38 筆 pattern **append 在 4096 個 null 之後**。
3. loader 端 `llama_model_loader::create_tensor`（`llama-model-loader.cpp:1109-1383`）以 **null-terminated loop** 消費 overrides：第 0 筆即 null → **0 overrides 套用** → 全部 experts 留 CUDA。
4. `-ot`（上游）不受影響（pattern 在 pad 之前 push）。Release 下 `GGML_ASSERT` 編譯掉 → 不崩、靜默 no-op（build 39 實機即此形態）。
5. **regex 本身無誤**：`s4_regex_probe/`（MSVC cl 19.44.35228 與 MinGW g++ 16.1.0 雙引擎實測 `A1=1 A3=0 B=1 C=1 D=1`，`(ch|)` 交替正常；A2=0 為 probe 索引錯配、非 regex 問題）→ 排除「pattern 名稱不匹配」假說。

## (b) 修復（已交付）

| 檔 | 變更 |
|---|---|
| `engine/common/common.h` | 新增 `llm_insert_q4e_moe_vram_overrides(n_layers, k, ovs)`：把 38 筆 pattern **插入第一個 null 之前**（保留 4096 pad 不變式、fit 相容；K≥n_layers 時 no-op） |
| `engine/common/common.cpp:1311` | 改呼叫 insert 版（取代 append 版） |
| `engine/tests/test-arg-parser.cpp` | 新增「override insertion into padded buffer」hard-check 區塊：先確認 pad 狀態（`front().pattern==nullptr`、`size()==4096`），insert 後 slot 0..37 非 null＋pattern 字串逐筆正確、slot 38 null；`K=999`（≥n_layers）邊界不變 |

**交付件**：`engine\build-cuda\bin\q4e-server.exe`＝`version: 0.3.0-dev (build 42, commit 207fe88)`（build 樹＝`D:\q4e_build` ASCII 鏡像，CJK 路徑 ninja 陷阱之對策）。`test-arg-parser` **exit 0、all tests OK**（含新區塊；2026-09-10 本 session 複驗，log＝`s5_fix_testargparser.log`）。

## (a) per-buft breakdown（K=10、ngl=all、ctx 2048、parallel 1；build 42 實測帳目）

| Buft | 內容 | 大小 |
|---|---|---|
| **CUDA0（GPU）** | blk.**38–47** routed experts（10 層×gate/up/down≈1,550 MiB/層）15,500.00 ＋ shared experts（48 層）239.53 ＋ attn/SSM/output/其餘 ≈4,342（G5 4,977.53 − token_embd ≈635.6） | **20,106.45 MiB** |
| CPU_Mapped（lazy host/SSD）#1 | blk.**0–37** routed experts（子集一） | 42,270.20 MiB |
| CPU_Mapped（lazy host/SSD）#2 | blk.**0–37** routed experts（子集二） | 19,431.35 MiB |
| CPU_Mapped（lazy host/SSD）#3 | PLE `per_layer_token_embd`（layer 1） | 27,465.95 MiB |
| （CPU buft） | `token_embd.weight`（q8_0，「and 114 others」同組） | ≈635.6 MiB |
| ctx/compute | KV（ctx 2048）＋compute buffers | ≈1,208.6 MiB |
| **required（fit 投影）** | | **21,315 MiB** |

**對帳**：20,106.45（CUDA0 實測）≈ 78,056（S3：全 experts on GPU）− 57,950（S2 inventory G1＝層 0–37）＝ **20,106** → 吻合（Δ<0.2%）。38 層 experts（61,701.55 MiB）與 PLE（27,465.95 MiB）在 lazy host/SSD＝v4.1 設計語意（K＝GPU 駐留層數，其餘 host 算＋live prefetch 暖頁）。

## (c) 新預期 VRAM（B 側驗證線）

**21,315 MiB**（`common_params_fit_impl` 自身投影：model 20,106.45＋ctx/compute ≈1,208.6；B-8-2 原文 ctx 2048/parallel 1）。

B 側重跑 B-8-2 原文（build 42、ngl=all＋K=10）之**預期形態**：
1. VRAM-fit check 應報 **projected ≈21,3xx MiB ≤ free**（空 GPU 下 free＝WDDM 22,884–23,141）→ **通過**（不再是 build 39 的 `needs 78,7xx MiB`）。
2. `health=200 ≤ ~250 s`（lazy 載入快；load_tensors 實測 ~3 s）。
3. nvidia-smi 實占 **≈20,1xx–21,3xx MiB**（model 20,106＋ctx/compute）。
4. log（`-lv 10`）應見 **152 行**（38 層×4…＝38×3 tensor 每 load；3 次 load 共 ~342 行）`buffer type overridden to CUDA_Host`＋per-buft 表如上方 (a)。

## A 側範圍聲明（重要）

本 session 之 `s6_diag` 診斷（RUN1/RUN2）以 `--n-gpu-layers all` **實際觸及 GPU**——**超出 A 側「無 GPU」範圍**（GPU 實機驗收＝Task B）。故本裁決**只採 CPU 側證據**：
- fit `no_alloc` 投影（21,315 MiB）＝純 CPU 計算；
- loader placement 帳目（buffer 表、override 行數）＝載入規劃階段、CPU 側產出；
- RUN1 在 load 完成後 hang（health 503→12:43:07 程序消失；12:43:22 GPU device 消失「No devices were found」）——與 llama-server 占 23.7 GB 之上 WDDM 超分（20.1 GB 虛擬配額＋page-in）一致，屬**環境問題**，不作 GPU 驗收結論。

**GPU 實機驗收（health/VRAM/t/s）＝Task B 於空 GPU 重跑 B-8-2 原文判定。**

## B 側環境警示（2026-09-10 實測）

1. **llama-server 已回來**：PID 13392（`D:\llama\llama-server.exe`）、nvidia-smi used **23,680/24,576 MiB** → **B-8-2 重跑前仍需用戶釋放 GPU**（B 側紅線：絕不殺 `llama-server`）。
2. **機器 GPU 驅動長期不穩**：Windows WER 存有 2026-06-16…09-08 多筆 kernel WATCHDOG LiveKernelEvent dump＋9/10 09:00 BSOD（bugcheck **0x133 DPC_WATCHDOG_VIOLATION**，minidump `C:\WINDOWS\Minidump\091026-42859-01.dmp`）；9/10 12:43 GPU device 一度消失。**若 B 側重跑期間機器崩潰/藍屏＝環境問題，照紀律回報原文即可、非 engine 責任。**
3. **冷生成首 token 會慢＝預期行為**：38 層 experts（≈61.7 GB）在 lazy host/SSD，首次生成 page-in 約 150–200 s（SATA 0.31–0.44 GB/s）；health 門與暖測紀律不變。

## 證據索引（本目錄）

| 目錄/檔 | 內容 |
|---|---|
| `s2_inventory/` | tensor_inventory.csv（1,224 張）＋G1–G5 分組（G1＝層 0–37 experts 57,950.00 MiB/114 張） |
| `s3_diag/` | build 39 K=10/K=0 差分（override 0 行＝H1 實機證據）＋gpu_state |
| `s4_regex_probe/` | regex 雙引擎 probe（MSVC/MinGW）＋SUMMARY |
| `s5_fix_testargparser.log` | build 42 test-arg-parser 全綠（含新 padded-buffer insert 測試） |
| `s6_diag/` | build 42 K=10/K=0 實跑 log（`-lv 10`）：342/144 override 行、fit 21,315、buffer 表（A 側已聲明超出範圍、僅取其 CPU 側帳目） |
