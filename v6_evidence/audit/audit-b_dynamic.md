# Audit-B: v6 Task A 動態審計（build + 測試 + 工具實跑）

- 審計者：TaskA（總指揮家）直接執行（build/測試類工作，子代理在沙箱內無法完成 cl/link 執行）
- repo：`D:\Nova2_home\AI研究\推理工具`
- 方法：完整重編譯（MSVC Ninja 591 步）＋ 16 個 q4e 測試實跑斷言矩陣 ＋ q4e-e2e / q4e-bench `--v6 --v6-kv100k` 實跑
- 結論：**build ✓（590/591，1 步環境性跳過）；測試 16/16 ✓（2 個缺陷已修復後全綠）；工具 2/2 exit 0 ✓**

## 1. Build（engine/build-msvc）

| 項 | 結果 | 證據 |
|---|---|---|
| 完整重編譯 | ✓ 590/591 | `run_build2/3/4.bat` 分段執行；marker `build9.done` = `BUILD_OK tail=591`；16 個 test exe + q4e-server/e2e/bench 全新生（9/14 11:58–11:59） |
| step 539 ui-assets | ⚠ 環境性失敗（非產品缺陷） | HF 下載 SSL 被封鎖（沙箱無網路）＋ MSYS2 gzip signal pipe 被封鎖（Win32 error 5）；該步僅產生 UI 前端資產，不影響任何 q4e 產品/測試 binary；其餘 590 步全綠 |
| 構建配方（環境限制下的可行路徑） | 已驗證 | PATH 上的 ninja 為 MSYS2 mingw64 build，會把 UTF-8 CJK 路徑（`AI研究`/`推理工具`）以 ANSI 重新編碼 → `GetFileAttributesEx ... syntax is incorrect`。正解：MSVC 原生 ninja（VS BuildTools CommonExtensions）只用於 `-t commands` 抽取；實際執行用純 ASCII .bat（CJK 路徑以 `%REPO%` 環境變數傳遞）＋ DSH 背景 job `cmd /c "%BAT%"`。ninja 直接執行、Start-Process -Wait、per-step log redirect 在本沙箱均失敗（pipe spawn 卡死 / Access denied / 檔案佔用），已記錄避免重蹈 |

## 2. 測試斷言矩陣（16/16 ✓）

| 測試 | checks | 結果 |
|---|---|---|
| test-q4e-global-pool | 188 | ✓ pass |
| test-q4e-staging | 118 | ✓ pass |
| test-q4e-gather | 34 | ✓ pass |
| test-q4e-h2d-pipeline | 118 | ✓ pass |
| test-q4e-graph-map | 78 | ✓ pass |
| test-q4e-promotion | 427 | ✓ pass |
| test-q4e-kv-paging | 947862 | ✓ pass |
| test-q4e-kv-bridge | 40 | ✓ pass |
| test-q4e-stats-v6 | 139 | ✓ pass |
| test-q4e-server-v6 | 50（0 failures） | ✓ pass |
| test-q4e-bench-v6 | 30 | ✓ pass |
| test-q4e-e2e-v6 | 28 | ✓ pass |
| test-q4e-expert-cache | 159 | ✓ pass |
| test-q4e-expert-warm | 195 | ✓ pass（real-gguf 段以 `Q4E_WARM_REAL_DIR` 未設而 skip，屬設計） |
| test-q4e-ngram | 2989 | ✓ pass |
| test-q4e-moe-pool | **4501** | ✓ pass（**修復前崩潰**，見 F-1/F-2） |

## 3. 工具實跑（2/2 ✓）

| 工具 | 指令 | 結果 |
|---|---|---|
| q4e-e2e | `q4e-e2e.exe --v6 --v6-kv100k` | exit 0；gate 1–5（pool_assembly / coverage_path / pipeline_stress / kv_sliding_window / stats_json）全 ok，`all_ok=true` |
| q4e-bench | `q4e-bench.exe --v6 --v6-kv100k` | exit 0；v6 sections 全 ok，`ok=true` |

## 4. 發現與修復（F-1 / F-2，均已閉合）

### F-1：test-q4e-moe-pool 崩潰（`GGML_ASSERT(sched) failed`，exit 0xC0000409）
- **症狀**：前 3 個靜態 graph case 全過後，`test_dynamic_oracle_sequence` 在 `ggml_backend_sched_reset(env.sched)` 崩潰——`env.sched` 恆為 nullptr。
- **根因**：commit `19b4082`（A-15.2b）在 `dyn_env` 加入 `cpu`/`sched` 欄位、析構釋放、reset/compute 呼叫，**但從未加入建立呼叫**（無 `ggml_backend_init_by_type`/`ggml_backend_sched_new`）。該 commit 後此測試從未跑過（A-15.2a 的「1841 checks green」早於 sched 重構）。
- **性質**：測試缺陷（缺 setup），非產品缺陷。
- **修復**：`engine/tests/test-q4e-moe-pool.cpp` 增 `dyn_env::sched_setup()`（與靜態 case 相同模式）＋ `test_dynamic_oracle_sequence` 中呼叫。

### F-2：pool 統計 `m_sum` 少計（6 vs 期望 15）
- **症狀**：F-1 修復後測試可跑，5 個 dyn round 的 pool-vs-full 全 bit-exact（max_abs_err = 0），但 `ps.m_sum == 15` 失敗（實際 6）。
- **根因**：`q4e_pool_dynamic::access`（`q4e-expert-warm.cpp`）的 **miss 路徑刻意不呼叫 `pool_.access`**（A-15.3 caliber：pool 層級狀態在 job done 才翻轉），但 pool 的 per-round distinct count（M）只由 `pool_.access` 在 round 期間累積 → 被 promote 的 expert 永不進 `round_ids` → `m_sum` 只數到 L1 hit。`m_sum` 是 v6 stats（`/q4e/stats` 的 `pool.m_sum`）的對外數值，少計即產品缺陷。
- **性質**：產品缺陷（A-15.2b 動態層與 pool 統計語義「sum of per-round distinct expert count」矛盾）。
- **修復**（3 檔，最小改動）：
  - `engine/common/q4e-expert-cache.h/.cpp`：新增 `q4e_expert_pool::note_round(layer, expert)`——round 期間僅記 `round_ids`，不改 tier 狀態。
  - `engine/common/q4e-expert-warm.cpp`：`q4e_pool_dynamic::access` miss 路徑呼叫 `pool_.note_round(...)`。
  - 其他 m_sum 斷言（expert-cache/global-pool/stats-v6）直接走 `pool_.access`，不受影響；重跑 16 測試全綠驗證。
- **驗證**：`test-q4e-moe-pool` 4501 checks 全過（含 `ps.m_sum==15`、`ps.m_max==3`、dyn 5 rounds max_abs_err=0）。

## 5. 審計範圍外（記錄，不處理）
- B-16（GPU 實測）：本機 RTX 3090 24 GB 被 harness `llama-server`（PID 13804, port 8080, 23.6 GB VRAM）佔用；依約定不殺該進程，B-16 待 GPU 釋放後由 RUNBOOK_B 執行。
- ui-assets 步驟的真實資產生成：需網路（HF）環境。
- test-q4e-expert-warm 的 real-gguf 段：需真實 GGUF 檔案（`Q4E_WARM_REAL_DIR`）。
