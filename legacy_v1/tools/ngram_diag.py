# -*- coding: utf-8 -*-
"""B-diag：ngram prefetch 34% 掉落的歸因實驗（一次載入，4 階段對照）。

A: 無 prefetch（baseline）
B: submit-only shim（executor + submit，不讀行）→ 隔離 executor/GIL 成本
C: 完整 prefetch（16 rows/token）→ 隔離讀取成本
D: 完整 prefetch 再跑 → 穩定性

每階段 300 tokens、temp=0、固定 prompt；印 rolling t/s（每 50 tokens）。
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
WARM, N = 32, 300


def token_id(chunk):
    ch = chunk.get("choices") or [{}]
    tk = ch[0].get("token") if isinstance(ch[0], dict) else None
    if isinstance(tk, dict) and tk.get("id") is not None:
        return int(tk["id"])
    return None


def run_phase(llm, tbl, mode: str) -> float:
    """mode: none | shim | full。回 t/s。"""
    from concurrent.futures import ThreadPoolExecutor

    ex = ThreadPoolExecutor(max_workers=1) if mode != "none" else None
    t0 = time.perf_counter()
    last = t0
    cnt = 0
    ids: list[int] = []
    for chunk in llm.create_completion(PROMPT, max_tokens=N, temperature=0.0, stream=True):
        cnt += 1
        tid = token_id(chunk)
        ids.append(tid if tid is not None else cnt)
        if mode == "shim":
            ex.submit(lambda: None)
        elif mode == "full":
            tbl.prefetch(ids[-3:])
        if cnt % 50 == 0:
            now = time.perf_counter()
            print(f"    [{mode}] tok {cnt}  rolling {50/(now-last):.2f} t/s  cum {cnt/(now-t0):.2f}")
            last = now
    wall = time.perf_counter() - t0
    if ex:
        ex.shutdown(wait=True)
    return cnt / wall if wall > 0 else 0.0


def main() -> int:
    raw = cfgmod.load_config()
    ecfg = cfgmod.engine_config(raw)
    eng_raw = raw.get("engine") or {}

    class RealNgram(NgramTable):
        def __init__(self) -> None:
            super().__init__(cfgmod.ngram_config(raw))
            self._ex = ThreadPoolExecutor(max_workers=1)

        def prefetch(self, token_ids: list[int]) -> None:
            self._ex.submit(self._sync, list(token_ids))

        def _sync(self, toks: list[int]) -> None:
            t = int(toks[-1])
            t1 = int(toks[-2]) if len(toks) >= 2 else 0
            t2 = int(toks[-3]) if len(toks) >= 3 else 0
            for rid in self.row_ids_for_token(t, t1, t2):
                self.read_row(rid)

        def close(self) -> None:
            self._ex.shutdown(wait=True)
            super().close()

    tbl = RealNgram()
    tbl.parse(ecfg.gguf_dir)
    llm = load_engine(eng_raw, ecfg)

    for _ in llm.create_completion(PROMPT, max_tokens=WARM, temperature=0.0, stream=True):
        pass

    a = run_phase(llm, tbl, "none")
    b = run_phase(llm, tbl, "shim")
    c = run_phase(llm, tbl, "full")
    d = run_phase(llm, tbl, "full")
    tbl.close()

    print(f"\n== diag 結果（每階段 {N} tokens）==")
    print(f"A none  = {a:.2f} t/s")
    print(f"B shim  = {b:.2f} t/s  （vs A: {(b-a)/a*100:+.1f}%）")
    print(f"C full  = {c:.2f} t/s  （vs A: {(c-a)/a*100:+.1f}%）")
    print(f"D full2 = {d:.2f} t/s  （vs A: {(d-a)/a*100:+.1f}%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
