# -*- coding: utf-8 -*-
"""B probe：GGUF 內 MoE 專家張量命名/大小（header-only，不碰 VRAM）。"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ngram import read_tensor_index  # noqa: E402

DIR = r"D:\AI_Model\Qwen3.8-Flash-Next"
idx = read_tensor_index(DIR)
print(f"total tensors: {len(idx)}")

pat = re.compile(r"^(\d+)(?:\.|$)")
prefixes: dict[str, int] = {}
for t in idx:
    m = re.match(r"^(blk\.\d+\.)", t["name"])
    pfx = m.group(1) if m else t["name"][:t["name"].find(".")] + ".*"
    key = re.sub(r"\.\d+\.", ".<id>.", pfx) if m else pfx
    prefixes[key] = prefixes.get(key, 0) + 1

print("\n== name patterns (count) ==")
for k in sorted(prefixes, key=lambda x: (-prefixes[x], x))[:40]:
    print(f"  {k}: {prefixes[k]}")

# sample one expert's gate/up/down in a CPU layer (layer 20)
print("\n== sample: blk.20 expert 5 ==")
n = 0
for t in idx:
    if t["name"].startswith("blk.20.") and ".5." in t["name"] + ".":
        print(f"  {t['name'][:90]:90} file={os.path.basename(t['file'])} off={t['offset']:,} size={t['size']:,}B gtype={t['gtype']} dims={t['dims']}")
        n += 1
        if n >= 12:
            break

# total expert bytes for one layer (layer 20, routed only)
tot = sum(t["size"] for t in idx if t["name"].startswith("blk.20.") and "ffn" in t["name"])
print(f"\nblk.20 ffn* total: {tot/1e6:.1f} MB  (≈ {tot/1e6/512:.2f} MB/expert if /512)")
