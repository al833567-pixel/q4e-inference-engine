"""Summarize a (split) GGUF: arch, key hparams, and per-layer expert/dense byte totals."""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"D:\AI model\engines\Maya\tools")
from gguf_reader import GGUFFile  # noqa: E402

first = Path(sys.argv[1])
shards = sorted(first.parent.glob(re.sub(r"-0+1-of-", "-*-of-", first.name)))
meta, tensors = {}, []
for s in shards:
    g = GGUFFile(s)
    if not meta:
        meta = g.metadata
    tensors += g.tensors
arch = meta.get("general.architecture")
print("arch:", arch, "| shards:", len(shards))
for k, v in sorted(meta.items()):
    if k.startswith(arch + ".") and not isinstance(v, (list, tuple)):
        print(f"  {k} = {v}")
cls = defaultdict(int)
types = defaultdict(set)
layers = set()
for t in tensors:
    n = t.name
    m = re.match(r"blk\.(\d+)\.", n)
    if m:
        layers.add(int(m.group(1)))
    if "_exps" in n:
        key = "routed experts"
    elif "shexp" in n:
        key = "shared expert"
    elif "per_layer" in n or "ple" in n or "ngram" in n or "engram" in n:
        key = "PLE/ngram/engram"
    elif n.startswith(("token_embd", "output")):
        key = "embed/output"
    elif "attn" in n or "ssm" in n or "kda" in n or "dsa" in n or "indexer" in n:
        key = "attention/mixer"
    else:
        key = "other"
    cls[key] += t.expected_bytes() or 0
    types[key].add(t.type_name)
print("layers (blk.N):", len(layers))
for k, v in sorted(cls.items(), key=lambda x: -x[1]):
    print(f"  {k:18s} {v / 2**30:8.2f} GiB  types={sorted(types[k])}")
print(f"  total              {sum(cls.values()) / 2**30:8.2f} GiB")
