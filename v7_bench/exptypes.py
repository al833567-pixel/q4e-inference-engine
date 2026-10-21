"""Per expert tensor kind: shape and quant types across layers (to pick a legal sub-expert block size)."""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, r"D:\AI model\engines\Maya\tools")
from gguf_reader import GGUFFile  # noqa: E402

first = Path(sys.argv[1])
shards = sorted(first.parent.glob(re.sub(r"-0+1-of-", "-*-of-", first.name)))
kinds = defaultdict(Counter)
shapes = {}
nbytes = defaultdict(int)
for s in shards:
    for t in GGUFFile(s).tensors:
        m = re.match(r"blk\.(\d+)\.(ffn_\w+_exps)\.weight", t.name)
        if m:
            kinds[m.group(2)][t.type_name] += 1
            shapes[m.group(2)] = t.shape
            nbytes[m.group(2)] += t.expected_bytes() or 0
for k in sorted(kinds):
    n_exp = shapes[k][2]
    print(f"{k:16s} shape={shapes[k]} types={dict(kinds[k])} per-expert avg={nbytes[k] / sum(kinds[k].values()) / n_exp / 2**20:.3f} MiB")
