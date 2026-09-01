# -*- coding: utf-8 -*-
"""A-2：兩層快取 LRU 邏輯單元測試（純 CPU，fake 專家 id）。"""
import random

import interfaces
from cache import SHARED_EXPERT_ID, TieredCache


def make_cfg(vram=3, ram=5, n_layers=2, has_shared=False, n_experts=512):
    return interfaces.CacheConfig(
        n_layers=n_layers,
        n_experts_per_layer=n_experts,
        vram_capacity_per_layer=vram,
        ram_capacity_per_layer=ram,
        has_shared=has_shared,
    )


def test_first_touch_is_ssd_then_vram_hit():
    c = TieredCache(make_cfg(vram=3, ram=5))
    b1 = c.touch(0, [1, 2, 3])
    assert b1 == {"vram": 0, "ram": 0, "ssd": 3}  # 首次都從 ssd 載入
    b2 = c.touch(0, [1])
    assert b2 == {"vram": 1, "ram": 0, "ssd": 0}  # 已在 vram


def test_vram_lru_evicts_to_ram():
    c = TieredCache(make_cfg(vram=2, ram=4))
    c.touch(0, [1])          # vram: [1]
    c.touch(0, [2])          # vram: [1,2]
    c.touch(0, [3])          # 滿 → 逐出 1 → ram；vram: [2,3]
    assert c.tier_of(0, 1) == "ram"
    assert c.tier_of(0, 2) == "vram"
    assert c.tier_of(0, 3) == "vram"
    b = c.touch(0, [1])      # ram 命中 → 提升 vram → 逐出 2（LRU）
    assert b == {"vram": 0, "ram": 1, "ssd": 0}
    assert c.tier_of(0, 1) == "vram"
    assert c.tier_of(0, 2) == "ram"


def test_ram_lru_evicts_to_ssd():
    c = TieredCache(make_cfg(vram=1, ram=2))
    c.touch(0, [1])          # vram: [1]
    c.touch(0, [2])          # 逐出 1→ram；vram: [2], ram: [1]
    c.touch(0, [3])          # 逐出 2→ram[1,2]；vram: [3]
    c.touch(0, [4])          # 逐出 3→ram 滿 → 逐出 1→ssd；ram: [2,3], vram: [4]
    assert c.tier_of(0, 1) == "ssd"
    assert c.tier_of(0, 2) == "ram"
    assert c.tier_of(0, 3) == "ram"
    assert c.tier_of(0, 4) == "vram"
    # 1 重新觸發 → ssd 命中
    b = c.touch(0, [1])
    assert b["ssd"] == 1


def test_lru_order_respects_reuse():
    c = TieredCache(make_cfg(vram=2, ram=4))
    c.touch(0, [1, 2])       # vram LRU→MRU: 1, 2
    c.touch(0, [1])          # 1 變 MRU: 2, 1
    c.touch(0, [3])          # 應逐出 2（最久未用），不是 1
    assert c.tier_of(0, 2) == "ram"
    assert c.tier_of(0, 1) == "vram"
    assert c.tier_of(0, 3) == "vram"


def test_prefetch_goes_only_to_ram():
    c = TieredCache(make_cfg(vram=2, ram=4))
    c.prefetch(0, [10, 11])
    assert c.tier_of(0, 10) == "ram"
    assert c.tier_of(0, 11) == "ram"
    assert c.stats()["prefetch"]["issued"] == 2


def test_prefetch_hit_counted_on_promotion():
    c = TieredCache(make_cfg(vram=2, ram=4))
    c.prefetch(0, [10])
    b = c.touch(0, [10])     # ram 命中（且是預取過）
    assert b["ram"] == 1
    assert c.stats()["prefetch"]["prefetch_hits"] == 1


def test_prefetch_skipped_when_hot():
    c = TieredCache(make_cfg(vram=2, ram=4))
    c.touch(0, [5])
    c.prefetch(0, [5])
    s = c.stats()["prefetch"]
    assert s["skipped_in_vram"] == 1
    assert s["issued"] == 0
    c.prefetch(0, [6])       # 進 ram
    c.prefetch(0, [6])       # already staged
    assert c.stats()["prefetch"]["already_staged"] == 1


def test_shared_resident_and_pinned():
    c = TieredCache(make_cfg(vram=1, ram=1, has_shared=True))
    b = c.touch(0, [SHARED_EXPERT_ID])
    assert b["vram"] == 1
    # 大量逐出後 shared 仍常駐
    for i in range(50):
        c.touch(0, [i])
        c.prefetch(0, [100 + i])
    assert c.tier_of(0, SHARED_EXPERT_ID) == "vram"
    b = c.touch(0, [SHARED_EXPERT_ID])
    assert b["vram"] == 1


def test_deterministic_replay():
    def run():
        c = TieredCache(make_cfg(vram=4, ram=8, n_layers=3))
        rng = random.Random(42)
        for t in range(300):
            for l in range(3):
                ids = sorted(rng.sample(range(64), 6))
                c.touch(l, ids)
                c.prefetch(l, sorted(rng.sample(range(64), 4)))
        return c.stats()

    a, b = run(), run()
    assert a == b


def _run_workload(hot: int, tokens: int = 1000, n_layers: int = 48):
    c = TieredCache(interfaces.CacheConfig(
        n_layers=n_layers,
        n_experts_per_layer=512,
        vram_capacity_per_layer=20,
        ram_capacity_per_layer=142,
        has_shared=True,
    ))
    for t in range(tokens):
        for l in range(n_layers):
            ids = [(l * 31 + t * 7 + i * 13) % hot for i in range(10)]
            c.touch(l, ids + [SHARED_EXPERT_ID])
    return c.stats()


def test_workload_hot_set_fits_vram():
    """熱集合 15/層 ≤ vram 20 → 暖機後幾乎全 vram 命中。"""
    s = _run_workload(hot=15)
    assert s["touches"] == 1000 * 48 * 11
    assert abs(s["hit_rate"]["vram"] + s["hit_rate"]["ram"] + s["hit_rate"]["ssd"] - 1.0) < 1e-6
    assert s["hit_rate"]["ssd"] < 0.05     # 暖機（每層 15 個新專家）後不再碰 ssd
    assert s["hit_rate"]["vram"] > 0.9


def test_workload_needs_ram_tier():
    """熱集合 30/層 > vram 20、≤ vram20+ram142 → 由 RAM 層兜住，ssd ≈ 0。"""
    s = _run_workload(hot=30)
    assert s["hit_rate"]["ssd"] < 0.05
    assert s["hit_rate"]["vram"] + s["hit_rate"]["ram"] > 0.95
    assert s["hit_rate"]["ram"] > 0.4     # 超出 vram 的部分落在 RAM


def test_stats_and_placement_log():
    c = TieredCache(make_cfg(vram=2, ram=3))
    c.touch(0, [1, 2, 3, 4])
    c.prefetch(0, [9])
    s = c.stats()
    for k in ("layers", "touches", "hits", "hit_rate", "prefetch", "occupancy", "placement_events"):
        assert k in s
    assert s["touches"] == 4
    assert s["hits"] == {"vram": 0, "ram": 0, "ssd": 4}
    assert s["placement_events"] > 0
    for layer, expert, to in c._placement_log:
        assert 0 <= layer < 2
        assert to in ("vram", "ram", "ssd")


def test_range_checks():
    c = TieredCache(make_cfg())
    import pytest

    with pytest.raises(ValueError):
        c.touch(99, [1])
    with pytest.raises(ValueError):
        c.touch(0, [99999])
    with pytest.raises(ValueError):
        c.prefetch(99, [1])


def test_reset():
    c = TieredCache(make_cfg())
    c.touch(0, [1, 2])
    c.prefetch(0, [3])
    c.reset()
    s = c.stats()
    assert s["touches"] == 0
    assert s["hits"] == {"vram": 0, "ram": 0, "ssd": 0}
    assert c.tier_of(0, 1) == "ssd"


def test_subclass_override_place_evict():
    """B 的用法：繼承 TieredCache，只覆寫 place()/evict()。"""
    calls = []

    class RealCache(TieredCache):
        def place(self, layer, expert, to_tier):
            calls.append(("place", layer, expert, to_tier))
            super().place(layer, expert, to_tier)

        def evict(self, layer, expert, to_tier):
            calls.append(("evict", layer, expert, to_tier))
            super().evict(layer, expert, to_tier)

    c = RealCache(make_cfg(vram=2, ram=2))
    c.touch(0, [1, 2, 3])
    kinds = {x[0] for x in calls}
    assert "place" in kinds and "evict" in kinds
