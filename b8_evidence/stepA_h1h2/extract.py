#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract.py — parse ONLY the GGUF header (KV metadata + tensor entries) of
shard 1 of Qwen3.8-Flash-Next-UD-Q4_K_XL and dump all tensor names.

- opens the input file in binary read mode and reads sequentially
- stops immediately after the last tensor entry (NO tensor data is read)
- sanity checks: magic == b"GGUF"; tensor_count <= 1_000_000; total bytes
  consumed capped at 64 MB

NOTE ON FORMAT DEVIATION:
  The task description (and the stock GGUF spec) says strings are
  u32-byte-length + bytes padded so (4 + length) % 8 == 0.
  This file does NOT follow that: hexdump verification shows every
  string field (KV keys, string values, array string elements) uses a
  u64 (8-byte) length prefix with NO padding. All 67 KV entries parse
  cleanly with that rule and land exactly on the next field, while the
  u32+padding rule misaligns on the very first key. This script parses
  the ACTUAL file format (u64 length, no padding); everything else
  (counts, value types, array layout, tensor-entry layout) is standard.
"""
import os
import struct
import sys

INPUT = r"D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
OUTDIR = r"D:\Nova2_home\AI研究\推理工具\b8_evidence\stepA_h1h2"

MAX_TENSORS = 1_000_000
BYTE_CAP = 64 * 1024 * 1024

# GGUF scalar value types -> byte size
# 0=uint8 1=int8 2=uint16 3=int16 4=uint32 5=int32 6=float32 7=bool
# 8=string 9=array 10=uint64 11=int64 12=float64
SCALAR_SIZES = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
}


class Reader:
    """Sequential binary reader with a hard byte cap."""

    def __init__(self, f, cap):
        self.f = f
        self.cap = cap
        self.consumed = 0

    def read(self, n):
        if self.consumed + n > self.cap:
            raise RuntimeError(
                "sanity cap exceeded: %d bytes needed > %d byte cap"
                % (self.consumed + n, self.cap)
            )
        data = self.f.read(n)
        if data is None or len(data) < n:
            got = len(data) if data is not None else 0
            raise RuntimeError(
                "unexpected EOF: wanted %d bytes, got %d (consumed so far: %d)"
                % (n, got, self.consumed)
            )
        self.consumed += n
        return data

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]


def read_padded_str(r):
    """String in this file's format: u64 byte-length + UTF-8 bytes, no padding."""
    length = r.u64()
    data = r.read(length)
    return data.decode("utf-8", errors="replace")


def skip_value(r, t):
    """Consume one GGUF value of type t (array values skipped recursively)."""
    if t in SCALAR_SIZES:
        r.read(SCALAR_SIZES[t])
    elif t == 8:  # string
        read_padded_str(r)
    elif t == 9:  # array
        elem = r.u32()
        n = r.u64()
        if n > MAX_TENSORS:
            raise RuntimeError("sanity: array element count %d > %d" % (n, MAX_TENSORS))
        for _ in range(n):
            skip_value(r, elem)
    else:
        raise RuntimeError("unknown GGUF value type %d" % t)


def main():
    with open(INPUT, "rb") as f:
        r = Reader(f, BYTE_CAP)

        magic = r.read(4)
        if magic != b"GGUF":
            raise RuntimeError("bad magic: %r (expected b'GGUF')" % (magic,))

        version = r.u32()
        tensor_count = r.u64()
        kv_count = r.u64()
        if tensor_count > MAX_TENSORS:
            raise RuntimeError(
                "sanity: tensor_count %d > %d" % (tensor_count, MAX_TENSORS)
            )

        # ---- KV metadata: keep keys, skip values safely ----
        kv_keys = []
        for _ in range(kv_count):
            kv_keys.append(read_padded_str(r))
            skip_value(r, r.u32())

        # ---- tensor entries ----
        names = []
        for _ in range(tensor_count):
            names.append(read_padded_str(r))
            n_dims = r.u32()
            for _ in range(n_dims):
                r.u64()      # dim values
            r.u32()          # ggml_type
            r.u64()          # data_offset
        # STOP here — no tensor data is read at all

    bytes_consumed = r.consumed

    os.makedirs(OUTDIR, exist_ok=True)
    names_path = os.path.join(OUTDIR, "tensor_names.txt")
    with open(names_path, "w", encoding="utf-8", newline="\n") as f:
        for name in names:
            f.write(name)
            f.write("\n")

    expert_matches = [n for n in names if "expert" in n][:15]  # covers "expert" and "experts"
    ffn_matches = [n for n in names if "ffn" in n][:5]

    summary = []
    summary.append("file: " + INPUT)
    summary.append("version: " + str(version))
    summary.append("tensor_count: " + str(tensor_count))
    summary.append("kv_count: " + str(kv_count))
    summary.append("total_bytes_consumed: " + str(bytes_consumed))
    summary.append("note: this file's strings use u64 length prefixes (no padding); "
                   "tensor_count is 0, so the header contains no tensor entries")
    summary.append("")
    summary.append("first 20 tensor names:")
    if names:
        for n in names[:20]:
            summary.append(n)
    else:
        summary.append("(none - tensor_count is 0)")
    summary.append("")
    summary.append("last 5 tensor names:")
    if names:
        for n in names[-5:]:
            summary.append(n)
    else:
        summary.append("(none - tensor_count is 0)")
    summary.append("")
    summary.append("KV metadata keys (%d):" % len(kv_keys))
    for k in kv_keys:
        summary.append(k)

    summary_path = os.path.join(OUTDIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(summary))
        f.write("\n")

    # ---- console report ----
    print("version:", version)
    print("tensor_count:", tensor_count)
    print("kv_count:", kv_count)
    print("bytes_consumed:", bytes_consumed)
    print("tensor_names_lines:", len(names))
    print("--- first 15 tensor names ---")
    for n in names[:15]:
        print(n)
    if not names:
        print("(none - tensor_count is 0)")
    print("--- first 15 names containing 'expert' (covers 'experts') ---")
    for n in expert_matches:
        print(n)
    if not expert_matches:
        print("(none)")
    print("--- first 5 names containing 'ffn' ---")
    for n in ffn_matches:
        print(n)
    if not ffn_matches:
        print("(none)")
    print("wrote:", names_path)
    print("wrote:", summary_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR: %s" % (e,), file=sys.stderr)
        sys.exit(1)
