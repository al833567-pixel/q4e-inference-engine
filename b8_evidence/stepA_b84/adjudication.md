# A 側裁決＋修復 — B-8-4 硬線 FAIL（ON 1.6366 < OFF 2.3002 t/s）

> TaskA 裁決文件。B 側回報（`03B_項目現況_TaskB.md` B-8 段＋`b8_evidence\step4_rerun_g5fix\b8_run5\`）為輸入數據；
> 本文＝根因診斷＋修復設計＋A 側驗證＋B 側重跑指示。原始數字與證據路徑逐條列，不做無出處推斷。

## 一、B 側回報（輸入，原文數字）

| 項 | 結果 | 原始數字 |
|---|---|---|
| B-8-3 `/v1` e2e 5 閘 | ✅ PASS | G0–G6 全綠（G5 已修）；`Q4E_E2E_ALL_PASS` |
| B-8-4 快取 ON vs OFF | ⛔ **硬線 FAIL** | ON **1.6366** t/s（156.418471 s / 256 tok，PID 7192，port 8124，VRAM 21,606 MiB）< OFF **2.3002** t/s（111.294254 s / 256 tok，PID 12984，oracle `-ngl all -ncmoe 38`，port 8125，VRAM 21,952 MiB）；log 尾 ON `eval time = 154358.33 ms / 256 (1.65 t/s)`，OFF `108765.54 ms / 256 (2.34 t/s)` |
| B-8-5 oracle 對答案 | ✅ PASS | 兩端 52 tok、finish_reason=stop、`message.content`＋`reasoning_content` 位元相同 |
| B-8-6 門 1（ON≥OFF 且 ≥1.95） | ⛔ FAIL | 同 B-8-4 |
| B-8-6 門 2（overlap≥80%） | ✅ PASS | `prefetch_overlap_ratio=0.9299`、`prefetch_io_issued=10091>0` |
| B-8-6 門 3（ngram 掉幅<1%） | 未量 | 待 A 側裁決是否量 |
| B-8-7 收尾 | ✅ PASS | 無殘留、VRAM 0、GGUF 4 片 111,334,654,784 B |

ON 相 stats（`b8_run5\b8_6_stats.json`，716 B）：`touches=194000`、`hits_vram=88920`、`hits_ram=81470`、`hits_ssd=22090`、`prefetch_io_issued=10091`、`prefetch_io_bytes=31,585,075,200`（31.6 GB）、`prefetch_io_completed=10091`、`prefetch_io_dropped=0`、`prefetch_events_dropped=814`、`prefetch_overlap_ratio=0.9299`、`layers_touched=38`、`q4e_moe_vram=10`、`hook_skips_weights_device=5160`。

## 二、根因診斷（source＋證據鏈）

### 2.1 代碼事實（v4.1 live prefetch backend，`engine/common/q4e-expert-warm.cpp`）

- worker `drain_once`（每 ids 事件）：對**前一個 token** 在同一層的每個被選中 expert 發 `warm_hint`（`do_warm_one`＝把該 expert 的 3 片 slice 從 mmap view `memcpy` 進 staging，把頁拉進 OS page cache；單 expert ≈ 3.13 MB）。
- `warm_done_` 為**一次性永久集合**：一個 (layer, expert) 只会被 warm 一次，之後永不重 warm（`do_warm_one` 見 `warm_done_` 即跳過）。
- 結果：整個 run 的 warm I/O ＝ **distinct (layer, expert) 集合各一次**。B-8-4 ON 實測 `io_issued=10091`、`io_bytes=31.6 GB`、`io_dropped=0` —— 與「distinct 全量各 warm 一次」完全吻合（10091 × 3.13 MB ≈ 31.6 GB）。

### 2.2 工作集 vs page cache（數量級）

- distinct 工作集 10,091 expert ≈ **31.6 GB** ＞ 本機 RAM ~20 GB 的可用 page cache（~14 GB；03B 環境調查）。
- 其中**一次性（one-shot）expert**：只被路由過 1 次、之後不再出現。對 compute 而言，這些頁被讀一次即棄（OFF 同樣只讀一次）；對 warm 而言，每次 warm 都是**純浪費的冷盤 I/O**（3.13 MB/次，且拉進 cache 即被後續讀取擠出）。
- warm 的 31.6 GB 流量與 compute 自己的讀取**共享同一顆 SATA SSD（~400 MB/s）**：(a) 搶帶寬，(b) 把 hot core 頁擠出 page cache，使 compute 熱路徑重讀盤。ON 總盤 I/O ≫ OFF → 關鍵路徑（compute）飢荒 → ON < OFF。

### 2.3 佐證（獨立數據點）

1. **A-14.3 CPU 同二進位 A/B**（`evidence_a143/bench_report.json`）：ON（prefetch on）0.845 t/s vs OFF（prefetch off）1.682 t/s —— warm thread 本身在 CPU 上 −50%，與 GPU 同方向。
2. **A-14.1 CPU e2e**（`evidence_a141/stats.json`）：ON 0.5637 t/s、`io_issued=13019`、`io_bytes=40.8 GB`、`io_dropped=1447`、`events_dropped=2734` —— worker 被 warm 流量壓垮（drops 大）。
3. **v1 基線（無 live warm backend）**：GPU ON 1.5692 / OFF 1.7700（−11%，A-9 歸因 sync D2H，已修）—— 無 warm thread 時 ON≈OFF。
4. **OFF 已吃滿盤**：OFF 111.3 s / ~40 GB ≈ 367 MB/s ≈ SATA 序讀上限 400 MB/s 的 ~92% → OFF 已盤界；ON 要在其之上贏，只能**減少總盤讀或把盤等待移出關鍵路徑**，絕不能加盤流量。
5. **ON log 形態**（`b8_run5\b8_4_on_server.err.log`）：tg_3s 0.35–5.23 振盪、ramp 慢（task259 eval 154,358 ms/256 vs OFF 108,766 ms/256）；task516（B-8-5，20 新 token 新路由）ON 1535 ms/tok vs OFF 477 ms/tok（3.2×）—— 新路由（=新 expert 冷讀）是 ON 的重災區，與「one-shot warm 浪費＋擠出 hot 頁」一致。
6. **overlap 0.93 不是實證**：`prefetch_overlap_ratio` 為邏輯計數（觸碰時該 expert 曾 warm 過即計 hit），非牆鐘重疊；0.93 只證明「warm 都做完了一次」，不證明「對 compute 有幫助」。

### 2.4 結論

**v4.1 的 warm 策略是淨負向**：它把整個 distinct 工作集（含一次性 tail）各冷讀一次（31.6–40.8 GB），其中一次性部分對 compute 零幫助（OFF 也只讀一次），卻搶走共享 SATA 帶寬並擠出 hot core 頁。B-8-4 硬線 FAIL 是該策略的必然結果，非 placement／kernel／環境問題（B-8-3/5/7 全綠、雙端 placement 相同）。

## 三、修復（B-8-4-F1：min-freq warm gate，opt-in）

### 3.1 設計

- **規則**：prev-token reuse warm 只 warm「已被選中 ≥ N 次」的 expert（N＝`--q4e-warm-min-freq`，**預設 1＝舊行為（凍結契約不變）**；N=2＝跳過一次性 expert）。
- **理由**：一次性 expert 的頁 compute 只讀一次（與 OFF 相同）——warm 它只增加 I/O，永遠不會藏住任何延遲；被選中 ≥2 次的 expert 其頁會被重用，warm 才有「保持在 page cache」的價值。
- **機制**：worker 在 `drain_once` 累計 per-(layer, expert) 選中計數 `freq_`；prev-warm 迴圈對未達 N 的 expert 跳過並計 `prefetch_warm_gated`。熱路徑（`enqueue_host_ids`）零新增開銷；`warm_hint` 公開 API 不設 gate（直接請求照舊，`test_real_gguf` 契約不變）；ngram（PLE 行）路徑 untouched。
- **stats**：`/q4e/stats` 新增 `prefetch_warm_gated`、`prefetch_min_freq`（additive；既有欄位與語義不變 → B-8-3 與 q4e-e2e G0–G6 不受影響）。
- **旗標**：`--q4e-warm-min-freq N`（1–1000000，預設 1；env `Q4E_WARM_MIN_FREQ`）。

### 3.2 變更清單（repo 工作樹，未 commit；與 A-15.2b WIP 正交）

| 檔 | 變更 |
|---|---|
| `engine/common/q4e-expert-warm.h` | `set_min_freq/min_freq/warm_gated`＋`freq_`/`min_freq_`/`warm_gated_` 成員＋`warm_gate` |
| `engine/common/q4e-expert-warm.cpp` | `drain_once`：ids 計數＋prev-warm gate；`warm_gate`/`set_min_freq`/`warm_gated` 實作 |
| `engine/common/common.h` | `common_params.q4e_warm_min_freq = 1` |
| `engine/common/arg.cpp` | `--q4e-warm-min-freq N` 旗標 |
| `engine/tools/server/server-context.cpp` | 接線（`set_min_freq` 於 `start()` 前）＋stats 2 欄 |
| `engine/tools/q4e-bench/main.cpp` | `--warm-min-freq` 透傳至 ON 相 server 參數 |
| `engine/tests/test-q4e-expert-warm.cpp` | **新增** `test_freq_gate` 節（additive；既有凍結節 untouched，min_freq=1 預設下行為逐位元不變） |

### 3.3 門相容性

- **B-8-4 硬線（ON≥OFF）**：warm 流量從 31.6 GB 降至「≥2 次 expert 集合」（B-8-4 數據下＝distinct − one-shot；量級 10–20 GB → 數 GB，以 `prefetch_warm_gated` 實測為準），一次性浪費歸零，hot 頁不再被 warm 流量擠出 → ON 不再被自己的 warm thread 拖垮。
- **B-8-6 門 2（overlap≥80%）**：語義＝「觸碰時該 expert 曾被 warm 且未 inflight」。core expert（≥2 次）早期進入 `warm_done_` 後其觸碰皆計 hit；one-shot 觸碰不計。預設 min_freq=1 時該指標與 B-8-4 實測 0.93 相同；min_freq=2 時預估值＝core 觸碰佔比（B-8-4 數據下 ~85–95%），≥80% 應守得住。若 GPU 實測 <80%，該數值本身即診斷數據（core/one-shot 分佈），回報 A 側。
- **B-8-6 門 3（ngram 掉幅<1%）**：ngram 路徑 untouched（獨立迴圈、90 B/行、計數器獨立）。**裁決：維持 RUNBOOK 原定義＝B 側 GPU 量（`q4e-ngram accept` 原文回報）**；A-14.4 CPU accept PASS（max drop +0.22%）為預驗證據，不替代 GPU 量。

## 四、A 側驗證（本機 CPU）

1. **build 47**（D:\q4e_build ASCII 鏡像，MSVC/Ninja，mirror HEAD `5efd937`）：`test-q4e-expert-warm`＋`q4e-server`＋`q4e-bench` 全綠。
2. **單元測試全綠**：test-q4e-expert-warm **195 checks**（含新 `test_freq_gate`；凍結節逐位元不變）＋test-q4e-expert-cache **159**＋test-arg-parser **all OK**（exit 0）。
3. **CPU A/B run 1**（`cpu_ab/`、06:17–07:56、harness 模型服務未回來）：
   - **ON（min_freq=2）＝0.4916 t/s**（518,673.9 ms/256；`predicted_per_second=0.4916`）；stats：`prefetch_warm_gated=14,328`（gate 生效）、`prefetch_min_freq=2`、`prefetch_io_issued=12,743`、`io_bytes=39.96 GB`、`io_dropped=0`、`events_dropped=110`、`prefetch_overlap_ratio=0.8825`（≥0.80 ✓）、`layers_touched=48`、`touches=256,430`、`hits_ssd=33,748`（13.2%）、`ngram_rows_warmed=3,928`。
   - **OFF 相未完成**：被 **9/12 07:30 PM BSOD**（`091226-77640-01.dmp`、`LastBoot=19:29`）直接終止（OFF server 06:42:09 啟動、無 `off_*.json`/`bench_report.json`、job 消失）——**機器崩潰、非代碼失敗**；harness 模型服務（Qwen3.8-27B）崩潰後 07:56 重啟（PID 15320、`--no-mmap` 佔 RAM）。OFF 對照引用 A-14.3 淨機實測 **1.682 t/s**（9/8、同機同模型同 ngl=0）。
4. **CPU A/B run 2**（`cpu_ab2/`、08:16 起）：ON 相 server 啟動、health PASS，warmup 256 tok 於 900 s read-timeout 內**無回應**（<0.285 t/s）→ bench 判 FAIL、自清。事件日誌無 q4e-server crash（唯 llama-server BEX64 0x c0000409 @07:51＝harness 服務自身重啟事件）→ **環境性（BSOD 後冷態＋harness 服務佔 RAM），非代碼回歸**。
5. **CPU 驗證最終狀態（補上 OFF 後）**：
   - **機器事件鏈（硬證據）**：9/12 **07:30 PM BSOD**（`C:\Windows\Minidump\091226-77640-01.dmp`、`LastBoot=09/12 19:29`）＝終止 run 1 OFF 相；前例 9/10 09:00 BSOD（0x133、03B 已記）→ **本機慢性不穩（2 次 BSOD/2 日）**。
   - **OFF 補測完成**（`cpu_ab3_off`、重啟後、harness 服務在跑）：**OFF＝0.882 t/s**（290.291 s/256、`prefetch=off`、touches=257,530）——OFF 相（無 warm thread）即便在 RAM 受限下仍跑通；**ON 相（含 warm thread）在同等條件下 900 s 超時（run 2、<0.285 t/s）＝warm thread 把 ON 推過超時邊界，OFF 無此問題**。
   - **CPU A/B 全數據（每組 ON<OFF）**：

     | 相 | t/s | 條件 |
     |---|---|---|
     | OFF（q4e 關） | **0.882** | 9/12 重啟後、harness 在跑 |
     | ON（gate min_freq=2） | **<0.285**（超時） | 9/12 重啟後（run 2） |
     | ON（gate min_freq=2） | **0.4916** | 9/12 重啟前淨態（run 1） |
     | OFF（q4e 關） | **1.682** | A-14.3（9/8 淨機） |
     | ON（min_freq=1） | 0.845 | A-14.3（9/8 淨機） |
     | ON（min_freq=1） | 0.5637 | A-14.1（9/8） |

   - **A 側 CPU 結論（誠實版、比先前更強）**：gate 生效實證＝stats（`warm_gated=14,328`）＋195 checks 全綠；但**每一組 ON 皆 < OFF（含 gate 後）**，且 warm thread 在 RAM 受限下直接把 ON 推到超時（OFF 無此問題）→ **min-freq gate 單獨用不足以翻轉 ON<OFF；v4.1 warm backend（事後 re-warm）在本硬體（工作集≫RAM）為淨負向**。CPU 為弱儀器（§4.1）但方向一致；**B 側 GPU 仍為交付門，但 CPU 證據已強烈暗示 gate 單獨不夠→v2 備案（§4.2）大概率需要**。

### 4.1 CPU 量測儀器性說明（重要，裁決依據）

- **CPU t/s 受機器狀態主導**：同 ON 設定跨 run 散佈 0.5637（A-14.1）↔0.845（A-14.3）＝±30% 以上雜訊；run 1 ON 0.4916 與兩者皆不可直接比。
- **min_freq 1→2 在此模型上只省 ~2% warm I/O**（distinct 13,019→12,743；`touches/distinct ≈ 20`＝工作集 ~98% freq≥2、one-shot tail 小）→ gate 是**必要但二階**的改善，不是 CPU 上 ON<OFF 的主因解。
- **v4.1 warm 的本質（refined 根因）**：prev-token warm 是**事後 re-warm**（compute 已讀過該頁、page cache 已含；warm 是重複觸碰＋被逐出後的重新冷讀），非真正 lookahead；工作集 31–40 GB ≫ 可用 RAM（31.9 GB 總量 − harness 服務 − OS）→ 持續逐出＋warm 與 compute 搶同一顆 SATA→**淨負向**。
- **結論（補 OFF 後更新）**：gate＝正確且必要（消滅可證浪費、凍結契約不變、stats 可觀測）；但**補上 OFF 0.882 後，CPU 全數據每組 ON<OFF（含 gate 後）→ gate 單獨大概率不足以翻轉 ON<OFF**。B 側 GPU 仍為交付門（最終裁決），但 CPU 證據已強烈指向**需要 §4.2 的 v2 備案**（warm 預設停用／I/O rate cap／真 lookahead）。

### 4.2 v2 備案（若 GPU 仍 ON<OFF）

(a) host 路徑 warm 預設停用（ON≈OFF、零回歸；交付語意＝「不輸 OFF」）；(b) warm I/O budget／rate cap（限流、保 compute 帶寬）；(c) 真 expert-level lookahead（ngram→expert 接線；v5 A-15.x pool/H2D 基礎已備）＝正解但工程量大。**依 GPU 數據擇一，不並行堆疊。**

## 五、B 側重跑指示（零判斷）

### 5.0 階段目標（本次新增、優先於「擇 v2 備案」）

**本階段唯一目標：B-8 完整跑完 → 確定「命中率」→ 才據此判斷 V5 是否有做的必要。**（現行 build 47／min-freq gate 不改，照跑。）

- **命中率定義（B 側須原文回報、取自 ON 相 `b8_6_stats.json`）**：
  1. **快層命中率**＝`(hits_vram + hits_ram) / touches`（expert 由 VRAM/RAM 快層服務的比例；**越高＝v4.1 cache 越有效**）；
  2. **SSD miss 率**＝`hits_ssd / touches`（由 host 磁盤重讀的比例；**越高＝cache 越 ineffective、host I/O 越重**）；
  3. **overlap_ratio**（門 2）＝`prefetch_overlap_ratio`；
  4. 輔以 ON/OFF t/s（門 1）。
- **V5 決策邏輯（命中率 + ON/OFF 共同判定，非單一指標）**：
  | 快層命中率 | ON vs OFF | 解讀 | V5 必要性 |
  |---|---|---|---|
  | **高（≥~85%）** | **ON < OFF** | cache 有效、但 host I/O／warm 競奪仍淨負向 → v4.1 路徑在本硬體有天花板 | **建議做 V5**（全 GPU MoE，根除 host I/O） |
  | **低（<~80%）** | ON < OFF | cache 失效（expert 持續被逐出、重讀）→ host cache 路徑在此硬體不成立 | **建議做 V5**（全 GPU，繞過 host cache） |
  | **高（≥~85%）** | **ON ≥ OFF** | v4.1 已生效且不輸 OFF | **V5 暫不急需**（v4.1 內再調優即可） |
- **紀律**：命中率＋ON/OFF 兩組數都到手前，**不下 V5 結論**；A 側不預設 V5 做／不做，等 B 側 GPU 數據。

**前置（人類）**：① GPU 現被 harness `llama-server`（Qwen3.8-27B）佔 **23,608/24,576 MiB** → B 側 GPU 實跑前需釋放（同 03B「待用戶處理」紀律；A 側絕不 kill）；② **機器風險＝2 次 BSOD（9/10 0x133、9/12 07:30）＋GPU driver 慢性 WATCHDOG**（03B 已記）→ 建議選低負載時段跑、每步落盤證據、中斷即原文回報。

1. **B-8-1 重build**（RUNBOOK 原指令）→ 新 `q4e-server.exe`；記 `--version`（build 47）＋SHA-256。
2. **B-8-2 載入**：RUNBOOK 原參數＋**`--q4e-warm-min-freq 2`**（唯一新增旗標；health 紀律不變）。
3. **B-8-3**：原樣（q4e-e2e；stats 新增 2 欄為 additive，不影響 G0–G6）。
4. **B-8-4**：原樣（ON≥OFF 硬線；2×256 tokens 紀律）。
5. **B-8-5**：原樣。
6. **B-8-6**：門 1＋門 2 原樣；門 3＝`q4e-ngram accept` GPU 量（原文回報）。
7. **B-8-7**：原樣。
8. 證據必含：新 exe SHA-256、`b8_6_stats.json` 全欄位（含 `prefetch_warm_gated`（應 >0，證明 gate 生效）＋`prefetch_min_freq=2`）、ON/OFF t/s＋log 尾段、**命中率三數（§5.0：快層命中率 `(hits_vram+hits_ram)/touches`、SSD miss 率 `hits_ssd/touches`、`prefetch_overlap_ratio`）＝V5 決策依據**。
9. **ON 仍 < OFF 時**：原文回報（不調參）＋**加跑 1 組控制**＝B-8-4 同紀律（2×256 tokens）、唯一差異 `--q4e-warm-min-freq 1`（同 build 47、warm-all 基線）→ 隔離 gate 在 GPU 上的淨效果（ON(gate) vs ON(leg) vs OFF 三數）。A 側**先依 §5.0 命中率判 V5 必要性**；若 V5 暫不做，再擇 v2 備案（§4.2），不並行堆疊。

## 六、紅線紀律（不變）

placement `--n-gpu-layers all --q4e-moe-vram 10`；不並行雙模型實例；kill 只按 PID；**絕不 kill `llama-server`**；B 側不改 engine source；commit 僅在人類明確指示後。
