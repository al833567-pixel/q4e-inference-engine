# -*- coding: utf-8 -*-
"""B-5/B-6：t/s 基準 + 驗收（Track B 執行）。

模式：
  --plan（預設）  純 CPU：印兩層快取規劃表（import two_tier_plan）+ 驗收行。
  --measure       接真實引擎量 M（每層不同專家數）與 t/s（B 填掛鉤）。
  --prefetch      --measure 的投機預取對照（重疊度 ≥80%，RUNBOOK_B §6）。

驗收規則（RUNBOOK_B §7）：t/s ≥ M 對應列目標的 70%。
  | M/層 | 目標 t/s |
  | ≤20  | ~95      |
  | ~44  | ~15      |
  | ~90  | ~11      |
  | ≤162 | 7–15（PCIe）|
  | >162 | 0.4–0.8（SSD）|
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "冷熱架構與t_s估算"))

for _s in (sys.stdout, sys.stderr):  # cp950 console 編不了 ≥/≤ → 統一 UTF-8
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def acceptance_row(m: float) -> tuple[str, float, float]:
    """回 (bucket, target_lo, target_hi)。"""
    if m <= 20:
        return ("≤20", 95.0, 95.0)
    if m <= 44:
        return ("~44", 15.0, 15.0)
    if m <= 90:
        return ("~90", 11.0, 11.0)
    if m <= 162:
        return ("≤162", 7.0, 15.0)
    return (">162", 0.4, 0.8)


def verdict(m: float, tps: float) -> str:
    bucket, lo, hi = acceptance_row(m)
    target = lo  # 以保守端（列低值）×70% 為及格線
    if tps >= target * 0.7:
        return f"PASS（M={m:.0f} → {bucket}，t/s={tps:.2f} ≥ {target*0.7:.2f}）"
    return f"FAIL（M={m:.0f} → {bucket}，t/s={tps:.2f} < {target*0.7:.2f}；回報聰明 AI）"


def run_plan() -> int:
    try:
        from estimate_tps import two_tier_plan
    except ImportError as e:
        print(f"FAIL: 無法 import estimate_tps.two_tier_plan：{e}")
        return 2
    two_tier_plan(ram_budget_gb=20.0, tokens_per_round=1000)
    print("\n[驗收行（B 量出 M 與 t/s 後對照）]")
    for m in (20, 44, 90, 162, 256, 512):
        bucket, lo, hi = acceptance_row(m)
        tgt = f"{lo:g}" if lo == hi else f"{lo:g}–{hi:g}"
        print(f"  M/層 {m:>4} → {bucket:>5}  目標 {tgt:>7} t/s  及格線 {lo*0.7:g}")
    return 0


def run_measure(prefetch: bool) -> int:
    """B：真實引擎 decode t/s（固定 workload ≥1000 tokens、temp=0、暖機 1 輪）。

    M（每層不同 routed 專家數）引擎外不可得（wheel 不暴露 per-token 路由決策）
    → 驗收用上界 M=512（保守列）；另報 fake_router 的 M_fake 作參考（標註為替代）。
    --prefetch：async 專家 slice 預取（fake_router 下一 token 決策集），
    重疊度＝前一 batch 在下一 chunk 抵達前完成的比例。
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    import config as cfgmod  # noqa: E402
    from wire_cache import fake_router  # noqa: E402  確定性替代（文檔化）

    raw = cfgmod.load_config()
    ecfg = cfgmod.engine_config(raw)
    eng_raw = raw.get("engine") or {}
    n_layers, active = 48, 10

    from live_common import ExpertStore, load_engine  # noqa: E402  (tools/ 同目錄)

    llm = load_engine(eng_raw, ecfg)
    store = ExpertStore(ecfg.gguf_dir, 512) if prefetch else None

    ngl = int(eng_raw.get("ngl") or ecfg.ngl or 10)

    def _batch(st: ExpertStore, t: int) -> None:
        for L in range(ngl, n_layers):  # GPU 層（L<ngl）常駐 VRAM → 零 I/O
            for e in fake_router(L, t, active):
                st.read_expert(L, e)

    prompt = "請從 2 開始依序列出質數，每個數字獨立一行，不要任何說明，持續輸出。"
    WARM, N = 32, 1000

    # 暖機 1 輪（暖 page cache + kernel）
    for _ in llm.create_completion(prompt, max_tokens=WARM, temperature=0.0, stream=True):
        pass

    ex = ThreadPoolExecutor(max_workers=1) if prefetch else None
    touched: set[tuple[int, int]] = set()
    t0 = time.perf_counter()
    n_tok = 0
    prev_fut = None
    on_time = checked = 0
    for chunk in llm.create_completion(prompt, max_tokens=N, temperature=0.0, stream=True):
        n_tok += 1
        for L in range(n_layers):  # 本 token 的「決策集」（fake_router，主執行緒記 M_fake）
            for e in fake_router(L, n_tok, active):
                touched.add((L, e))
        if prefetch:
            if prev_fut is not None:
                if prev_fut.done():
                    on_time += 1
                checked += 1
            prev_fut = ex.submit(_batch, store, n_tok + 1)
    wall = time.perf_counter() - t0
    tps = n_tok / wall if wall > 0 else 0.0
    if ex is not None:
        # 收尾：取消排隊中的 batch（非 daemon 執行緒，不 cancel 會卡死 interpreter exit）
        ex.shutdown(wait=False, cancel_futures=True)
    if store is not None:
        store.close()

    per_layer = [sum(1 for (L, e) in touched if L == l) for l in range(n_layers)]
    m_fake = sum(per_layer) / n_layers

    print("== MEASURE 模式（真實引擎 decode）==")
    print(f"workload = {N} tokens（prime 串流、temp=0、暖機 {WARM} tokens）")
    print(f"t/s = {tps:.2f}")
    if prefetch:
        overlap = on_time / checked if checked else 0.0
        print(f"prefetch 重疊度 = {overlap:.1%}（{on_time}/{checked} batches 在下一 chunk 前完成）"
              f"  目標 ≥80%")
        print(f"I/O 總量 = {store.bytes_read/1e9:.2f} GB")
    print(f"M_fake = {m_fake:.0f}（fake_router 替代，僅參考；真實 M 引擎外不可測）")
    M = 512.0  # 驗收用上界（保守列）
    print(f"M={M:.0f}  tps={tps:.2f}  prefetch={prefetch}")
    print(verdict(M, tps))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="印規劃表（純 CPU，預設）")
    mode.add_argument("--measure", action="store_true", help="接真實引擎量 M/t/s（B 填）")
    ap.add_argument("--prefetch", action="store_true", help="--measure 的投機預取對照")
    ap.add_argument("--check", type=float, nargs=2, metavar=("M", "TPS"), help="直接驗收：--check M tps")
    args = ap.parse_args()
    if args.check:
        print(verdict(args.check[0], args.check[1]))
        return 0
    if args.measure:
        return run_measure(args.prefetch)
    return run_plan()


if __name__ == "__main__":
    raise SystemExit(main())
