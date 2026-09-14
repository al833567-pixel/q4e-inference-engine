# Audit-A: v6 Task A 靜態一致性審計

- 審計者：Audit-A 子代理（只讀＋本檔）
- repo：`D:\Nova2_home\AI研究\推理工具`（HEAD = `06153a6`）
- 方法：git log/show、Select-String 行號核對、文件抽查（未執行 build/測試）
- 結論：**8 項 ✓，1 項 ✗（第 2 項 q4e-gather.cpp 缺失）；第 7 項含 1 處文件不一致（README 過期）**

## 審計表

| 項 | ✓/✗ | 證據（行號/commit） | 說明 |
|---|---|---|---|
| 1. commit 序（17 個） | ✓ | `git log --oneline -25`：`06153a6`/`321c1f8`/`f4aaecd`/`9ccded1`/`c82fbc1`/`9445169`/`a891982`/`abc7afc`/`6207f7c`/`d75f4c4`/`5f19be6`/`7de45e0`/`ecfb070`/`2e597b5`/`91abfb0`/`d467f05`/`f5b15e9` 全在 | 17/17 存在。註：`06153a6`（收尾對齊）實為 HEAD，新於 `321c1f8`，與清單「新→舊」順序略有出入，但存在性滿足；02 文件「16 commits f5b15e9~321c1f8」自洽（16 個） |
| 2. 元件檔存在（10 組） | ✗ | `engine/common/` 列檔：9 組 .h+.cpp 齊備（非空）；**`q4e-gather.cpp` 不存在**（僅 `q4e-gather.h` 5117 B） | gather 為 header-only 介面（commit `ecfb070`「CUDA Kernel 介面定義」、CMake 僅註冊 `q4e-gather.h`、03A 標「介面」），可能為設計意圖，但與審計清單「10 組 .h+.cpp」不符，記 ✗ 待裁決 |
| 3. CMake 註冊 | ✓ | `engine/common/CMakeLists.txt` L99–122 全 10 組（gather 僅 .h）；`engine/tests/CMakeLists.txt` L159–176＋L309 | 16/16 測試 target 全註冊：arg-parser、expert-cache、expert-warm、ngram、global-pool、staging、gather、h2d-pipeline、graph-map、promotion、kv-paging、kv-bridge、stats-v6、server-v6、bench-v6、e2e-v6（另有多餘 test-q4e-moe-pool，不影響） |
| 4. 6 v6 參數 | ✓ | `common.h` L480–489：`q4e_moe_vram=5000`（L480 註「total routed experts resident」）、`q4e_moe_ram=4000`、`q4e_staging_mb=128`、`q4e_layer_floor=10`、`q4e_ple_mmap=true`、`q4e_kv_ram=6000`；`arg.cpp` L2782–2912 全解析；`arg.cpp` L2895 `LOG_WRN`（ple-mmap off） | 預設值、語義、on\|off 解析、off 警告全符合 |
| 5. server 接線 | ✓ | `tools/server/server.cpp` L14 `#include "q4e-server-v6.h"`、L252 `ctx_http.get("/q4e/stats", ...)`、L259 `q4e_server_v6_state(cfg)`、L262–264 `q4e_server_v6_http/body`；`q4e-server-v6.cpp` L67 `q4e_stats_v6_json(q4e_stats_v6_collect(...))` | route＋q4e_server_v6＋q4e_stats_v6 引用全存在 |
| 6. 工具 --v6 | ✓ | `tools/q4e-bench/main.cpp` L54–55、L750–753（`--v6`/`--v6-kv100k`）；`tools/q4e-e2e/main.cpp` L44–45、L522–525 | 兩工具存在且 main 支援 `--v6`/`--v6-kv100k` |
| 7. 文件一致性 | ✗ | 測試矩陣：01 L77、03A L27–42 與實際 target 一致✓；參數預設：README L54–59、01 L54–61 與 common.h 一致✓；commit 引用（01 L68–75、02 L6、L12–20）與 git 一致✓；RUNBOOK 編號表（L12–22）與 02 主表（A-16.5=KV 分頁/A-16.6=server/A-16.7=bench-e2e/A-16.8=交付包）一致✓。**但 README 過期**：README L3/L14/L68 仍稱「A-16.8 交付包進行中」「8b RUNBOOK 編號對齊待做」，而 `321c1f8`（A-16.8b）已存在、`06153a6` 已把 02/03A/03B 更新為 ✅（`git show --stat 06153a6` 僅改 02/03A/03B，README 未更新） | README 與 02/03A/03B 狀態矛盾（README 稱 8b 待做，02/03A 稱 A-16.8 ✅）；其餘文件一致 |
| 8. 殘留 | ✓ | `v6_work/` 不存在；repo 根無 `b16_*`；`git status --short` 空；tracked 無 `.ninja_lock`/`llama.rsp` | 全乾淨 |
| 9. v5 凍結未被破壞 | ✓ | 三組 v5 檔存在（expert-cache 15545/7838 B、expert-warm 40941/14138 B、ngram 15764/3862 B）；`git log -3 -- q4e-expert-cache.cpp`：`19b4082`/`ef02cd5`/`8413022` 全早於 `f5b15e9`（A-16.0）；expert-warm/ngram 最近改動 `19b4082`/`0f9fc74` 同早於 A-16.0 | v5 核心未被 v6 改動 |

## ✗ 項清單（只記錄，不修）

1. **項 2**：`engine/common/q4e-gather.cpp` 缺失（header-only；CMake/文件均以「介面」描述，可能為設計意圖，需 A 側裁決是否補 .cpp 或修正清單預期）。
2. **項 7（部分）**：README.md L3/L14/L68 過期——仍稱 A-16.8 進行中、8b 待做，與 `321c1f8`/`06153a6` 後 02/03A/03B 的「A-16.8 ✅」矛盾；`06153a6` 收尾對齊漏掉 README。
   - **後續（審計提交後）**：README 已於 commit `6c0f150`（Audit-A 修正 F-1）更新為「A-16.0~A-16.8 全交付」，項 7 文件不一致已閉合；本審計保留原始發現作為證據。
