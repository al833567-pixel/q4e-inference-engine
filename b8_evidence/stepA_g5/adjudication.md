# B-8-3 G5 裁決（A 側）— 2026-09-12

> 交接來源：[03B §一 B-8-3 G5 FAIL](../../03B_項目現況_TaskB.md) 之「問 A 側（待裁決）」。
> 本檔＝A 側交付兩件：(a) `layers_touched=38` 之裁決＋source 證據；(b) placement-aware G5 gate 修復＋新 exe。

## 結論：`layers_touched=38` 於 ngl=all＋K=10 下為**預期值**，非回歸

G5 舊 gate 寫死 `layers_touched == layers`（＝48），以 A-14.1 CPU 基線（ngl=0）校準；但 GPU placement 下 10 層專家 VRAM-resident、**不參與 CPU 簿記** → 48−10＝38 才正確。

**根因（source 證據，非猜）**：
1. `q4e_expert_cache_cb_eval`（`engine/common/q4e-expert-cache.cpp:499-504`）：`ids->buffer` 非 host（即該層在 device）→ `note_hook_skip()` 回 true、**不呼叫 `touch()`**；只有 host-resident（CPU）層才 `touch()`。
2. `touch()`（:94-99）是唯一標記 `layer_touched_` 之處 → **`layers_touched`＝被觸碰的 CPU 側 MoE 層數**，VRAM-resident 層恒不計入（無論是否被觸碰）。
3. `hook_skips_weights_device`（:502 → stats :197）＝device-resident 層被跳過之觸碰計數 → **>0 ⇔ 該 run 存在 VRAM-resident 層**（G3 已保證 ≥1 token 走完整 48 層 decoder，device 層必然被評估）。
4. B-8-3 實測帳目自洽：`layers_touched=38`＋`hook_skips_weights_device=650`＋`q4e_moe_vram=10`；A-14.1 CPU 基線（同 K=10、ngl=0）＝`layers_touched=48`＋`hook_skips_weights_device=0`。
5. `stats.json` 之 `q4e_moe_vram`＝`params.q4e_moe_vram`（設定值，`server-context.cpp:4722`），非實測 placement。

## (b) 修復（已交付）

**改法**：`engine/tools/q4e-e2e/main.cpp` G5 段（+17/−4 行；未動 server、未動 test、未動其他 gate）：

```cpp
const long long q4e_moe_vram = num("q4e_moe_vram");
const long long hook_skips_device = num("hook_skips_weights_device");
...
// touched books host-resident MoE layers only; VRAM-resident layers are
// skipped (hook_skips_weights_device). Expected touched = layers minus
// the device-resident MoE layers (K when placement active, else 0).
const long long expected_touched =
    (hook_skips_device > 0)
        ? ((q4e_moe_vram >= layers) ? 0 : (layers - q4e_moe_vram))
        : layers;
if (layers_touched != expected_touched) {
    bad.push_back("layers_touched=" + ... + " (want expected=" + ... );
}
```

**對交接建議（`03B` 所附 snippet）的兩處偏離（A 側裁決）**：
1. **placement 偵測改用 `hook_skips_weights_device > 0`**，非只看 `q4e_moe_vram`：A-14.1 CPU 基線 stats 為 `q4e_moe_vram=10`＋`layers_touched=48`（ngl=0、K 未生效）——純 K 判斷式會預期 38→**誤殺 CPU 基線（工具校準參考）**；`hook_skips==0` ⇔ 無 device 層 ⇔ 預期 48，兩基線同時成立。
2. **補 `K >= layers` 分支（預期 0）**：B-9 v5 設定 K=5000≥48（全 experts VRAM）時 `layers_touched` 必為 0；純 K 判斷式（`K<layers` 才減）會預期 48→**B-9 必踩同一 gate**。
   註：B-9 v5 之 G5 其餘子門（`touches>0`／`prefetch_io_issued>0`）係 v4.1（K<layers）語意，v5 全 VRAM 配置需 A-15.6「q4e-bench / q4e-e2e v5」重定——屬既有 roadmap、非本輪範圍。

**交付件**：
- `engine\build-msvc\bin\q4e-e2e.exe`（B-8-3 所用之 client binary；純 HTTP client、無 CUDA 依賴）
- SHA-256：`EDD6E0BC163C49053D1437666D63B9EC9DA3714894C0E2F8235FBD3BFEF0610F`
- 367,104 B、build 2026-09-12 14:28（vcvars64＋MSVC 19.44.35207＋Ninja、`--target q4e-e2e` EXIT=0）
- `q4e-server.exe`（build 42、`207fe88`）**未動**；B-8-2 之 VRAM-fit 通過記錄不受影響。
- 源碼改動未 commit（待用戶裁決是否 commit；commit 後 B 側引用 hash）。

## 驗證（A 側、離線）

gate 規則對兩份已歸檔 stats 之實算（同一判斷式）：

| run | layers | K | hook_skips | layers_touched | expected | 結果 |
|---|---|---|---|---|---|---|
| B-8-3 GPU（ngl=all、K=10） | 48 | 10 | 650 | 38 | 38 | **G5-PASS**（舊 gate＝FAIL、即 B 側所見） |
| A-14.1 CPU（ngl=0、K=10） | 48 | 10 | 0 | 48 | 48 | **G5-PASS**（舊 gate 亦 PASS；新 gate 不回歸） |
| B-9 假設（K=5000、全 VRAM） | 48 | 5000 | >0 | 0 | 0 | G5-PASS（新 gate 相容；純 K snippet 會誤判 FAIL） |

新 exe `--help` exit 0（本機實跑）。**真機 5 閘＋G6＝B 側重跑判定。**

## B 側下一步（零判斷）

1. 重跑 **B-8-3 原文**（RUNBOOK 同指令；`q4e-e2e` 用 `engine\build-msvc\bin\q4e-e2e.exe` 新件）→ 預期 `G0–G6 全 PASS`＋`Q4E_E2E_ALL_PASS`（G5 行應印 `layers_touched=38/48 (expected 38)`）。
2. G5 PASS → 續 **B-8-4～B-8-7**（t/s ON/OFF、overlap、ngram、oracle＋回歸門）。
3. 任一 gate 仍 FAIL → 照備案**回報原文**（工具 log-dir 證據＋stats 原文），不猜。
