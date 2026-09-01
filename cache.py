# -*- coding: utf-8 -*-
"""兩層快取決策邏輯（A-2）：LRU 取代 + 需求分頁 + 確定性預取 + 命中統計。

純 Python、不碰 GPU/模型：專家 id 是抽象 int（0..n_experts_per_layer-1）。
B（Track B）繼承 `TieredCache`，**只覆寫 place()/evict()** 接真實 placement
（pinned H2D / D2H / SSD），簽章不改（見 interfaces.CacheManager）。

層級（interfaces.TIERS）：
  - vram (L1)：每層容量 `vram_capacity_per_layer` 個 routed 專家；
               若 `has_shared`，shared 專家（SHARED_EXPERT_ID=-1）常駐、不占容量、永不逐出。
  - ram  (L2)：每層容量 `ram_capacity_per_layer` 個專家。
  - ssd  (L3)：其餘全部（不追蹤狀態，視為無限）。

確定性規則：
  - touch（需求）：
      vram 命中 → 標 MRU；
      ram  命中 → 提升到 vram（計 ram 命中；若曾被 prefetch 過 → prefetch_hits+1）；
      ssd  未中 → 直接載入 vram（計 ssd 命中）。
      vram 滿 → 逐出**最久未用**到 ram（vram→ram 只降一級，不直落 ssd）。
  - prefetch（投機，確定性：上一 token 該層實際選擇）：
      **只載入 ram**（快取過濾器：永不進 vram，保護熱層）；
      已在 vram/ram → no-op（計 skipped_in_vram / already_staged）；
      ram 滿 → 逐出 LRU 到 ssd。
  - 全程無隨機性：同輸入必同狀態（可重放、可除錯）。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import interfaces

# shared 專家 id（0..511 是 routed；-1 代表每層的 shared 專家）
SHARED_EXPERT_ID = -1


class TieredCache(interfaces.CacheManager):
    def __init__(self, cfg: interfaces.CacheConfig | None = None) -> None:
        super().__init__(cfg or interfaces.CacheConfig())
        n = self.cfg.n_layers
        self._vram: list[OrderedDict] = [OrderedDict() for _ in range(n)]
        self._ram: list[OrderedDict] = [OrderedDict() for _ in range(n)]
        self._prefetched: list[set] = [set() for _ in range(n)]
        # 統計
        self._touches = 0
        self._hits = {"vram": 0, "ram": 0, "ssd": 0}
        self._pref = {
            "issued": 0,
            "skipped_in_vram": 0,
            "already_staged": 0,
            "evicted_from_ram": 0,
            "prefetch_hits": 0,
        }
        self._placement_log: list[tuple[int, int, str]] = []  # (layer, expert, "→tier")

    # ------------------------------------------------------------------ #
    # 內部搬運（fake placement；B 覆寫 place/evict 接真實 H2D/D2H）
    # ------------------------------------------------------------------ #
    def _vram_insert(self, layer: int, expert: int) -> None:
        v = self._vram[layer]
        if expert in v:
            v.move_to_end(expert)
            return
        if len(v) >= self.cfg.vram_capacity_per_layer:
            old, _ = v.popitem(last=False)  # LRU
            self._ram_insert(layer, old)
            self.evict(layer, old, "ram")
        v[expert] = True
        self.place(layer, expert, "vram")

    def _ram_insert(self, layer: int, expert: int) -> None:
        r = self._ram[layer]
        if expert in r:
            r.move_to_end(expert)
            return
        if len(r) >= self.cfg.ram_capacity_per_layer:
            old, _ = r.popitem(last=False)  # LRU（popitem 已移除）
            self._prefetched[layer].discard(old)
            self._pref["evicted_from_ram"] += 1
            self.evict(layer, old, "ssd")
        r[expert] = True

    # ------------------------------------------------------------------ #
    # 合約方法
    # ------------------------------------------------------------------ #
    def touch(self, layer: int, expert_ids: list[int]) -> dict[str, int]:
        """一個 token 在第 layer 層用到 expert_ids → 回 {vram, ram, ssd} 命中統計。"""
        if not (0 <= layer < self.cfg.n_layers):
            raise ValueError(f"layer {layer} out of range [0,{self.cfg.n_layers})")
        b = {"vram": 0, "ram": 0, "ssd": 0}
        v, r = self._vram[layer], self._ram[layer]
        for e in expert_ids:
            self._touches += 1
            if e == SHARED_EXPERT_ID:
                if self.cfg.has_shared:
                    b["vram"] += 1
                else:
                    b["ssd"] += 1
                    self._vram_insert(layer, e)
                continue
            if not (0 <= e < self.cfg.n_experts_per_layer):
                raise ValueError(f"expert id {e} out of range [0,{self.cfg.n_experts_per_layer})")
            if e in v:
                b["vram"] += 1
                v.move_to_end(e)
                self._prefetched[layer].discard(e)
            elif e in r:
                b["ram"] += 1
                if e in self._prefetched[layer]:
                    self._pref["prefetch_hits"] += 1
                    self._prefetched[layer].discard(e)
                del r[e]
                self._vram_insert(layer, e)  # 提升（可能逐出 vram LRU → ram）
            else:
                b["ssd"] += 1
                self._vram_insert(layer, e)
        for k in ("vram", "ram", "ssd"):
            self._hits[k] += b[k]
        return b

    def prefetch(self, layer: int, hint_ids: list[int]) -> None:
        """投機預取：只進 ram（快取過濾），永不進 vram。可失敗、可被 LRU 逐出。"""
        if not (0 <= layer < self.cfg.n_layers):
            raise ValueError(f"layer {layer} out of range [0,{self.cfg.n_layers})")
        v, r = self._vram[layer], self._ram[layer]
        for e in hint_ids:
            if e == SHARED_EXPERT_ID or e in v:
                self._pref["skipped_in_vram"] += 1
                continue
            if e in r:
                self._pref["already_staged"] += 1
                continue
            self._ram_insert(layer, e)
            self._prefetched[layer].add(e)
            self._pref["issued"] += 1

    def stats(self) -> dict[str, Any]:
        t = self._touches
        n = self.cfg.n_layers

        def rate(x: int) -> float:
            return round(x / t, 4) if t else 0.0

        def occ(tiers: list[OrderedDict]) -> dict[str, float]:
            lens = [len(x) for x in tiers]
            return {
                "avg": round(sum(lens) / n, 1) if n else 0.0,
                "min": min(lens) if lens else 0,
                "max": max(lens) if lens else 0,
            }

        return {
            "layers": n,
            "touches": t,
            "hits": dict(self._hits),
            "hit_rate": {k: rate(x) for k, x in self._hits.items()},
            "prefetch": dict(self._pref),
            "occupancy": {"vram": occ(self._vram), "ram": occ(self._ram)},
            "placement_events": len(self._placement_log),
        }

    # ------------------------------------------------------------------ #
    # B 覆寫點（fake 版只記 log；B 做真實 pinned H2D / D2H / SSD 寫入）
    # ------------------------------------------------------------------ #
    def place(self, layer: int, expert: int, to_tier: str) -> None:
        """把某專家搬到某層。A 的 fake：只記錄；B 覆寫做真實載入（SSD→RAM→VRAM）。"""
        self._placement_log.append((layer, expert, to_tier))

    def evict(self, layer: int, expert: int, to_tier: str) -> None:
        """把某專家逐出到某層。A 的 fake：只記錄；B 覆寫做真實 D2H / 寫 SSD。"""
        self._placement_log.append((layer, expert, to_tier))

    # ------------------------------------------------------------------ #
    # 除錯/測試輔助
    # ------------------------------------------------------------------ #
    def tier_of(self, layer: int, expert: int) -> str:
        if expert == SHARED_EXPERT_ID and self.cfg.has_shared:
            return "vram"
        if expert in self._vram[layer]:
            return "vram"
        if expert in self._ram[layer]:
            return "ram"
        return "ssd"

    def reset(self) -> None:
        n = self.cfg.n_layers
        self._vram = [OrderedDict() for _ in range(n)]
        self._ram = [OrderedDict() for _ in range(n)]
        self._prefetched = [set() for _ in range(n)]
        self._touches = 0
        self._hits = {"vram": 0, "ram": 0, "ssd": 0}
        self._pref = {
            "issued": 0,
            "skipped_in_vram": 0,
            "already_staged": 0,
            "evicted_from_ram": 0,
            "prefetch_hits": 0,
        }
        self._placement_log = []
