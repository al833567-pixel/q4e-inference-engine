# -*- coding: utf-8 -*-
"""B：N-gram offload/prefetch 驗收量測（RUNBOOK_B §5，t/s 下降 <1%）。

設計（抵消 phase 位置的 page-cache 漂移）：
  warmup 300 → P1 none → P2 full → P3 none → P4 full（各 300 tokens，temp=0 固定 prompt）
判讀：
  drop_cum  = (none_cum  − full_cum)  / none_cum
  drop_tail = (none_tail − full_tail) / none_tail   （tail = 每階段最後 150 tokens）
PASS 條件：drop_cum < 1% 且 drop_tail < 1%（兩組對照 P1/P2 與 P3/P4 都看）。
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config as cfgmod  # noqa: E402
from ngram import NgramTable  # noqa: E402
from live_common import load_engine  # noqa: E402

PROMPT = "請從 2 開始依序列出質數，每個數字獨立一行，不要任何說明，持續輸出。"
WARM, N, TAIL = 300, 300, 150


def token_id(chunk):
    ch = chunk.get("choices") or [{}]
    tk = ch[0].get("token") if isinstance(ch[0], dict) else None
    if isinstance(tk, dict) and tk.get("id") is not None:
        return int(tk["id"])
    return None


def run_phase(llm, tbl, mode: str) -> tuple[float, float, list[float]]:
    """回 (cum t/s, tail t/s, per-token 抵達時間戳)。"""
    ex = ThreadPoolExecutor(max_workers=1) if mode == "full" else None
    stamps: list[float] = []
    t0 = time.perf_counter()
    ids: list[int] = []
    for chunk in llm.create_completion(PROMPT, max_tokens=N, temperature=0.0, stream=True):
        stamps.append(time.perf_counter())
        tid = token_id(chunk)
        ids.append(tid if tid is not None else len(ids) + 1)
        if mode == "full":
            tbl.prefetch(ids[-3:])
    wall = time.perf_counter() - t0
    if ex:
        ex.shutdown(wait=True)
    cum = len(stamps) / wall if wall > 0 else 0.0
    if len(stamps) > TAIL:
        span = stamps[-1] - stamps[-TAIL - 1]
        tail = TAIL / span if span > 0 else 0.0
    else:
        tail = cum
    return cum, tail, stamps


def main() -> int:
    raw = cfgmod.load_config()
    ecfg = cfgmod.engine_config(raw)
    eng_raw = raw.get("engine") or {}

    class RealNgram(NgramTable):
        def __init__(self) -> None:
            super().__init__(cfgmod.ngram_config(raw))
            self._ex = ThreadPoolExecutor(max_workers=1)
            self.prefetch_bytes = 0
            self.prefetch_calls = 0

        def prefetch(self, token_ids: list[int]) -> None:
            self._ex.submit(self._sync, list(token_ids))
            self.prefetch_calls += 1

        def _sync(self, toks: list[int]) -> None:
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
    off = raw.get("ngram", {}).get("offload_path", r"D:\ng_offload\per_layer_token_embd.bin")
    tbl.offload(off)
    print(f"N-gram 表：dims={tbl.dims}  row={tbl.row_bytes()}B  offload→{off}（no-op：mmap=page cache 常駐）")

    llm = load_engine(eng_raw, ecfg)

    for _ in llm.create_completion(PROMPT, max_tokens=WARM, temperature=0.0, stream=True):
        pass
    print(f"warmup {WARM} tokens done")

    res = {}
    for name, mode in (("P1", "none"), ("P2", "full"), ("P3", "none"), ("P4", "full")):
        cum, tail, _ = run_phase(llm, tbl, mode)
        res[name] = (cum, tail)
        print(f"{name} ({mode:4})  cum={cum:.2f} t/s  tail150={tail:.2f} t/s", flush=True)
    tbl.close()

    (c1, t1), (c2, t2) = res["P1"], res["P2"]
    (c3, t3), (c4, t4) = res["P3"], res["P4"]
    drop_cum1 = (c1 - c2) / c1 * 100 if c1 else 0.0
    drop_cum2 = (c3 - c4) / c3 * 100 if c3 else 0.0
    drop_tail1 = (t1 - t2) / t1 * 100 if t1 else 0.0
    drop_tail2 = (t3 - t4) / t3 * 100 if t3 else 0.0

    print("== N-gram offload/prefetch 驗收（RUNBOOK_B §5）==")
    print(f"對照 1（P1 none vs P2 full）:  cum {c1:.2f} → {c2:.2f}  drop={drop_cum1:+.2f}%   tail {t1:.2f} → {t2:.2f}  drop={drop_tail1:+.2f}%")
    print(f"對照 2（P3 none vs P4 full）:  cum {c3:.2f} → {c4:.2f}  drop={drop_cum2:+.2f}%   tail {t3:.2f} → {t4:.2f}  drop={drop_tail2:+.2f}%")
    verdict = "PASS" if max(drop_cum1, drop_cum2, drop_tail1, drop_tail2) < 1.0 else "FAIL"
    print(f"t/s 下降 = max({drop_cum1:.2f}%, {drop_cum2:.2f}%, {drop_tail1:.2f}%, {drop_tail2:.2f}%)  目標 <1% → {verdict}")
    print(f"prefetch 量 = {tbl.prefetch_bytes/1e3:.0f} KB / {tbl.prefetch_calls} tokens "
          f"（≈{tbl.prefetch_bytes/max(1, tbl.prefetch_calls):.0f} B/token）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
