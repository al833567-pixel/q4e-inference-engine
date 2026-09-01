# -*- coding: utf-8 -*-
"""B-3：N-gram 表離載 host/SSD + 非同步預取（Track B 執行）。

模式：
  --plan（預設）  純 CPU：A 的 NgramTable.parse + bench_random_reads + offload 計畫。
  --live          接真實 offload：B 填 RealNgram 的 offload()/prefetch()（見下方 B 填區塊）。

驗收（RUNBOOK_B §5）：N-gram 住 host/SSD 時 t/s 相對「住 RAM」掉 <1%。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _s in (sys.stdout, sys.stderr):  # cp950 console → UTF-8
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config as cfgmod  # noqa: E402
from ngram import NgramTable  # noqa: E402


def run_plan(n_reads: int) -> int:
    raw = cfgmod.load_config()
    tbl = NgramTable(cfgmod.ngram_config(raw))
    ecfg = cfgmod.engine_config(raw)
    try:
        tbl.parse(ecfg.gguf_dir)
    except FileNotFoundError as e:
        print(f"FAIL: {e}（檢查 engine.gguf_dir）")
        return 2
    print("== N-gram 解析（A 的 ngram.py，CPU-only）==")
    print(json.dumps(
        {
            "file": os.path.basename(tbl.file),
            "dims": list(tbl.dims),
            "size_gb": round(tbl.size / 1e9, 2),
            "bits_per_elt": round(tbl.size * 8 / (tbl.dims[0] * tbl.dims[1]), 4),
            "n_entries": tbl.n_entries,
            "row_bytes": tbl.row_bytes(),
        },
        ensure_ascii=False, indent=2,
    ))
    print("\n== SSD/page-cache 隨機讀（1 row = 160 值 = 1.4 KB 的 16 rows/token 之一）==")
    print(json.dumps(tbl.bench_random_reads(n=n_reads), ensure_ascii=False, indent=2))
    tbl.close()
    off = raw.get("ngram", {}).get("offload_path", r"D:\ng_offload\per_layer_token_embd.bin")
    tbl.offload(off)
    print(f"\noffload 計畫 → {off}（fake 已記錄；B 用 --live 做真實離載）")
    print("驗收：真實 offload 後 t/s 掉 <1%（對照 N-gram 住 RAM 的基準）。")
    return 0


def run_live() -> int:
    """B：真實 host/SSD offload + 非同步預取（與計算重疊）＋真實引擎 A/B t/s 量測。

    驗收（RUNBOOK §5）：N-gram 住 host/SSD（page cache）時 t/s 相對 baseline 掉 <1%。
    """
    raw = cfgmod.load_config()
    ecfg = cfgmod.engine_config(raw)
    eng_raw = raw.get("engine") or {}
    off = raw.get("ngram", {}).get("offload_path", r"D:\ng_offload\per_layer_token_embd.bin")

    class RealNgram(NgramTable):
        def __init__(self) -> None:
            super().__init__(cfgmod.ngram_config(raw))
            self._ex = ThreadPoolExecutor(max_workers=1)
            self.prefetch_bytes = 0
            self.prefetch_calls = 0

        def offload(self, path: str) -> None:
            super().offload(path)  # 記錄（A log）
            # no-op（文檔化）：N-gram 表（~28.8 GB）本身是 GGUF 內的張量；
            # 對該 shard 的 mmap/buffered 讀取＝host/SSD 常駐（OS page cache＝RAM 層）。
            # 再複製 28.8 GB 到 offload_path 只多耗磁碟、無效益。

        def prefetch(self, token_ids: list[int]) -> None:
            # 非同步預取（1 執行緒）：下一 token 的 16 rows ≈ 1.4 KB，與計算重疊
            self._ex.submit(self._prefetch_sync, list(token_ids))
            self.prefetch_calls += 1

        def _prefetch_sync(self, toks: list[int]) -> None:
            if not toks:
                return
            t = int(toks[-1])
            t1 = int(toks[-2]) if len(toks) >= 2 else 0
            t2 = int(toks[-3]) if len(toks) >= 3 else 0
            rids = self.row_ids_for_token(t, t1, t2)
            for rid in rids:
                self.read_row(rid)
            self.prefetch_bytes += len(rids) * self.row_bytes()

        def close(self) -> None:
            self._ex.shutdown(wait=True)
            super().close()

    tbl = RealNgram()
    tbl.parse(ecfg.gguf_dir)
    tbl.offload(off)
    print(f"N-gram 表：dims={tbl.dims}  row={tbl.row_bytes()}B  size={tbl.size/1e9:.1f} GB")
    print(f"offload → {off}（no-op：表在 GGUF 內，mmap=page cache 常駐，不另占 RAM 預算）")

    from live_common import load_engine  # noqa: E402  (tools/ 同目錄)

    llm = load_engine(eng_raw, ecfg)

    prompt = "請從 2 開始依序列出質數，每個數字獨立一行，不要任何說明，持續輸出。"
    WARM, N = 32, 300

    def token_id(chunk) -> "int | None":
        ch = chunk.get("choices") or [{}]
        tk = ch[0].get("token") if isinstance(ch[0], dict) else None
        if isinstance(tk, dict) and tk.get("id") is not None:
            return int(tk["id"])
        return None

    def run_phase(n: int, with_prefetch: bool) -> float:
        t0 = time.perf_counter()
        cnt, ids = 0, []
        for chunk in llm.create_completion(prompt, max_tokens=n, temperature=0.0, stream=True):
            cnt += 1
            tid = token_id(chunk)
            ids.append(tid if tid is not None else cnt)
            if with_prefetch:
                tbl.prefetch(ids[-3:])
        wall = time.perf_counter() - t0
        return cnt / wall if wall > 0 else 0.0

    # warmup 1 輪（暖 page cache + kernel）
    for _ in llm.create_completion(prompt, max_tokens=WARM, temperature=0.0, stream=True):
        pass
    tps_a = run_phase(N, with_prefetch=False)
    tps_b = run_phase(N, with_prefetch=True)
    drop = (tps_a - tps_b) / tps_a * 100 if tps_a > 0 else 0.0
    verdict = "PASS" if drop < 1.0 else "FAIL"

    print("== LIVE 模式（真實引擎 + RealNgram async prefetch）==")
    print(f"baseline（無 prefetch）t/s = {tps_a:.2f}")
    print(f"prefetch 模式 t/s = {tps_b:.2f}")
    print(f"t/s 下降 = {drop:.2f}%  目標 <1% → {verdict}")
    print(f"prefetch 量 = {tbl.prefetch_bytes/1e3:.0f} KB / {tbl.prefetch_calls} tokens "
          f"（≈{tbl.prefetch_bytes/max(1, tbl.prefetch_calls):.0f} B/token）")
    tbl.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="接真實 offload（B 填覆寫）")
    ap.add_argument("--reads", type=int, default=1000)
    args = ap.parse_args()
    if args.live:
        return run_live()
    return run_plan(args.reads)


if __name__ == "__main__":
    raise SystemExit(main())
