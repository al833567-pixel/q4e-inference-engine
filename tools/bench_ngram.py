# -*- coding: utf-8 -*-
"""A-3：N-gram 表 SSD 隨機讀 bench（CPU-only，不需 GPU）。

用法：
  py -3.14 tools\\bench_ngram.py [--dir D:\\AI_Model\\Qwen3.8-Flash-Next] [--n 1000] [--seed 0]

輸出：解析資訊（dims/gtype/4.5-bit 驗證）+ 隨機讀延遲統計（mean/p50/p99/吞吐）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfgmod  # noqa: E402
from ngram import BITS_PER_ELT, NgramTable  # noqa: E402


def main() -> int:
    raw = cfgmod.load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=(raw.get("engine") or {}).get("gguf_dir"))
    ap.add_argument("--n", type=int, default=int((raw.get("ngram") or {}).get("bench_reads", 1000)))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.dir:
        print("error: no --dir and config.yaml has no engine.gguf_dir", file=sys.stderr)
        return 2

    tbl = NgramTable(cfgmod.ngram_config(raw))
    tbl.parse(args.dir)
    print("== parse ==")
    print(f"file        : {os.path.basename(tbl.file)}")
    print(f"tensor      : per_layer_token_embd.weight")
    print(f"dims        : {tbl.dims[0]} x {tbl.dims[1]:,} (row_len x n_rows)")
    print(f"size        : {tbl.size:,} bytes  ({tbl.size/1e9:.2f} GB)")
    print(f"bits/elt    : {tbl.size*8/(tbl.dims[0]*tbl.dims[1]):.4f}  (expect {BITS_PER_ELT})")
    print(f"n_entries   : {tbl.n_entries:,}  (= n_rows/16)")
    print(f"row_bytes   : {tbl.row_bytes()} B (160 vals x 4.5 bit)")

    print(f"\n== bench_random_reads(n={args.n}, seed={args.seed}) ==")
    r = tbl.bench_random_reads(n=args.n, seed=args.seed)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    tbl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
