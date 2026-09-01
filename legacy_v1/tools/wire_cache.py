# -*- coding: utf-8 -*-
"""B-2：兩層快取接線（Track B 執行）。

模式：
  --fake（預設）  純 CPU：A 的 TieredCache + 確定性 fake router，驗證 LRU/命中分佈。
  --live          接真實引擎：B 填 RealEngine 的 router 掛鉤（見下方 B 填區塊）。

驗收對照：../冷熱架構與t_s估算/estimate_tps.py 的 two_tier_plan（M 對應列 70%+）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _s in (sys.stdout, sys.stderr):  # cp950 console → UTF-8
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config as cfgmod  # noqa: E402
import interfaces  # noqa: E402
from cache import SHARED_EXPERT_ID, TieredCache  # noqa: E402


def fake_router(layer: int, token: int, active: int) -> list[int]:
    """確定性 fake：每層每 token 選 active 個 routed 專家（B 換成真實 router 決策）。"""
    return sorted(((layer * 31 + token * 7 + i * 13) % 512 for i in range(active)))


def run_fake(tokens: int, active: int) -> int:
    raw = cfgmod.load_config()
    ccfg = cfgmod.cache_config(raw)
    active = min(active, ccfg.n_experts_per_layer)
    cache = TieredCache(ccfg)
    for t in range(tokens):
        for l in range(ccfg.n_layers):
            ids = fake_router(l, t, active)
            cache.touch(l, ids + [SHARED_EXPERT_ID])
            # 確定性預取：「上一 token 的選擇」投機進 RAM
            cache.prefetch(l, fake_router(l, t + 1, active))
    s = cache.stats()
    print("== fake 模式（純 CPU；B 改 --live 接真實 placement）==")
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print(f"\ntouches={s['touches']}  hit_rate={s['hit_rate']}")
    print("驗收：hit_rate.ssd 應趨近 0（暖機後）；vram+ram 覆蓋熱集合。")
    return 0


def run_live(tokens: int) -> int:
    """B：RealCache（真實 slice I/O）＋真實引擎 decode，per-token touch/prefetch。

    真實性邊界（誠實標註）：
    - wheel 不暴露 per-token 專家路由決策 → expert ids 用 fake_router（確定性、
      文檔化替代）；I/O 本身是真實的（GGUF 內專家權重 slice，與引擎共享 page cache）。
    - place(to_tier)：GPU 層（L<ngl）零 I/O（整層住 VRAM）；CPU 層＝mmap slice read。
    - evict()：Windows 無 drop_caches → no-op（OS 管理 page cache）。
    """
    raw = cfgmod.load_config()
    ccfg = cfgmod.cache_config(raw)
    ecfg = cfgmod.engine_config(raw)
    eng_raw = raw.get("engine") or {}
    ngl = int(eng_raw.get("ngl") or ecfg.ngl or 10)
    active = min(10, ccfg.n_experts_per_layer)

    from live_common import ExpertStore, load_engine  # noqa: E402  (tools/ 同目錄)

    store = ExpertStore(ecfg.gguf_dir, ccfg.n_experts_per_layer)

    class RealCache(TieredCache):
        def __init__(self):
            super().__init__(ccfg)
            self.placements = {"vram": 0, "ram": 0}

        def place(self, layer: int, expert: int, to_tier: str) -> None:
            super().place(layer, expert, to_tier)  # A 的 placement log
            self.placements[to_tier] += 1
            if layer < ngl:
                return  # GPU 常駐層：整層（含全部 512 專家）已在 VRAM，零 I/O
            # CPU 層：hot tier（此處 to_tier 恆為 "vram"＝引擎實際讀取層）物理上是
            # 「讀進 page cache」→ 真實 mmap slice read
            if expert == SHARED_EXPERT_ID:
                store.read_shared(layer)
            else:
                store.read_expert(layer, expert)

        def prefetch(self, layer: int, hint_ids: list[int]) -> None:
            before = set(self._prefetched[layer])
            super().prefetch(layer, hint_ids)
            # 本 call 新 staged（_prefetched 增量）→ 真實 SSD→RAM slice read
            # （每個專家只在其 staged 當下讀一次；evict 不重讀）
            for e in self._prefetched[layer]:
                if e not in before and layer >= ngl:
                    store.read_expert(layer, e)

        def evict(self, layer: int, expert: int, to_tier: str) -> None:
            super().evict(layer, expert, to_tier)
            # no-op：Windows page cache 由 OS 管理（無 drop_caches）

    cache = RealCache()
    llm = load_engine(eng_raw, ecfg)

    prompt = "請從 2 開始依序列出質數，每個數字獨立一行，不要任何說明，持續輸出。"
    t0 = time.perf_counter()
    n_tok = 0
    per_layer: dict[int, dict[str, int]] = {l: {"vram": 0, "ram": 0, "ssd": 0} for l in range(ccfg.n_layers)}
    for chunk in llm.create_completion(prompt, max_tokens=tokens, temperature=0.0, stream=True):
        n_tok += 1
        for l in range(ccfg.n_layers):
            ids = fake_router(l, n_tok, active)
            b = cache.touch(l, ids + [SHARED_EXPERT_ID])
            for k in ("vram", "ram", "ssd"):
                per_layer[l][k] += b[k]
            cache.prefetch(l, fake_router(l, n_tok + 1, active))
    wall = time.perf_counter() - t0
    tps = n_tok / wall if wall > 0 else 0.0
    s = cache.stats()

    print(f"== LIVE 模式（真實引擎 + RealCache 真實 slice I/O）tokens={n_tok} ==")
    print("註：expert ids 為 fake_router 確定性替代（wheel 不暴露路由決策）；"
          "I/O 為真實 GGUF slice（與引擎共享 page cache）。")
    print(f"t/s（含快取 I/O）= {tps:.2f}")
    print(f"I/O 總量 = {store.bytes_read/1e9:.2f} GB  placements={cache.placements}")
    print(f"touch 命中：{s['hits']}  hit_rate={s['hit_rate']}")
    print(f"prefetch：{s['prefetch']}")
    print("\n== 每層命中分佈 {vram, ram, ssd} ==")
    agg = {"vram": 0, "ram": 0, "ssd": 0}
    for l in range(ccfg.n_layers):
        d = per_layer[l]
        for k in d:
            agg[k] += d[k]
        mark = "GPU" if l < ngl else "CPU"
        print(f"  L{l:>2} [{mark}]  vram={d['vram']:>6}  ram={d['ram']:>6}  ssd={d['ssd']:>6}")
    tot = sum(agg.values()) or 1
    print(f"  Σ      vram={agg['vram']:>6}  ram={agg['ram']:>6}  ssd={agg['ssd']:>6}"
          f"  (vram {agg['vram']/tot:.1%} / ram {agg['ram']/tot:.1%} / ssd {agg['ssd']/tot:.1%})")
    # M（fake router 下每層 touched 的不同專家數；引擎真實 M 外測不可得 → 驗收用上界 512）
    store.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="接真實引擎（B 填掛鉤）")
    ap.add_argument("--fake", action="store_true", help="純 CPU fake router（預設）")
    ap.add_argument("--tokens", type=int, default=1000)
    ap.add_argument("--active", type=int, default=10)
    args = ap.parse_args()
    if args.live:
        return run_live(args.tokens)
    return run_fake(args.tokens, args.active)


if __name__ == "__main__":
    raise SystemExit(main())
