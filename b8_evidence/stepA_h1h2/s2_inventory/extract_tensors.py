#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_tensors.py — bounded GGUF header inventory for shards 2-4 of
Qwen3.8-Flash-Next-UD-Q4_K_XL (4-shard GGUF).

Reads ONLY the first 24 MB of each input file (hard per-file cap, bounded
reads via .read(n) — no mmap, no full-file reads). Parses the GGUF header
with u64 little-endian string length prefixes (primary convention;
fallback: standard u32 length + 8-byte padding) and reports which one
parsed cleanly.

Outputs (all inside OUTDIR only):
  tensor_inventory.csv   — shard, name, offset, size_bytes (all tensors)
  breakdown_calc.md      — group table, VRAM predictions, per-layer experts,
                           10 largest tensors, per-shard summary
  hparams_from_dims.txt  — layer-0 tensor names + dims
  kv_metadata.txt        — KV metadata dump (truncated arrays)
Prints a machine-readable summary block to stdout.
"""
import csv
import os
import re
import struct

BASE = r"D:\AI_Model\Qwen3.8-Flash-Next"
SHARDS = [
    ("00002", os.path.join(BASE, "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf")),
    ("00003", os.path.join(BASE, "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf")),
    ("00004", os.path.join(BASE, "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf")),
]
OUTDIR = r"D:\Nova2_home\AI研究\推理工具\b8_evidence\stepA_h1h2\s2_inventory"
CAP = 24 * 1024 * 1024  # hard read cap per file (24 MB)

# GGUF value types: 0=uint8 1=int8 2=uint16 3=int16 4=uint32 5=int32
#                   6=float32 7=bool 8=string 9=array 10=uint64 11=int64 12=float64
SCALAR = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
          5: ("i", 4), 6: ("f", 4), 7: ("b", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}

R_EXPERT = re.compile(r"blk\.(\d+)\.ffn_(up|down|gate|gate_up)_(ch|)exps")


class BoundedReader:
    """Binary reader with a hard byte cap (never reads past `cap` total)."""

    def __init__(self, path, cap):
        self.path = path
        self.cap = cap
        self.pos = 0
        self.file_size = os.path.getsize(path)
        self.f = open(path, "rb")

    def read(self, n):
        if n == 0:
            return b""
        if self.pos + n > self.cap:
            raise EOFError(
                f"24MB cap exceeded at pos {self.pos} (+{n}) for {os.path.basename(self.path)}"
            )
        data = self.f.read(n)
        if len(data) != n:
            raise EOFError(f"unexpected EOF at pos {self.pos} (+{n})")
        self.pos += n
        return data

    def _u(self, fmt, n):
        return struct.unpack(fmt, self.read(n))[0]

    def u8(self):
        return self._u("<B", 1)

    def u16(self):
        return self._u("<H", 2)

    def u32(self):
        return self._u("<I", 4)

    def u64(self):
        return self._u("<Q", 8)

    def i8(self):
        return self._u("<b", 1)

    def i16(self):
        return self._u("<h", 2)

    def i32(self):
        return self._u("<i", 4)

    def i64(self):
        return self._u("<q", 8)

    def f32(self):
        return self._u("<f", 4)

    def f64(self):
        return self._u("<d", 8)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def build_parser(br, convention):
    if convention == "u64":

        def read_str():
            n = br.u64()
            if n > 16 * 1024 * 1024:
                raise ValueError(f"implausible string length {n}")
            return br.read(n)
    else:  # u32 length + pad (4+len) up to 8-byte boundary

        def read_str():
            n = br.u32()
            if n > 16 * 1024 * 1024:
                raise ValueError(f"implausible string length {n}")
            data = br.read(n)
            pad = (4 + n) % 8
            if pad:
                br.read(8 - pad)
            return data

    def scalar_value(vt):
        fmt, n = SCALAR[vt]
        return struct.unpack(fmt, br.read(n))[0]

    def read_value(vt):
        if vt in SCALAR:
            return scalar_value(vt)
        if vt == 8:
            return read_str().decode("utf-8", "replace")
        if vt == 9:
            et = br.u32()
            n = br.u64()
            if n > 5_000_000:
                raise ValueError(f"implausible array length {n}")
            if et == 8:
                return [read_str().decode("utf-8", "replace") for _ in range(n)]
            if et in SCALAR:
                fmt, sz = SCALAR[et]
                return list(struct.unpack(f"<{n}{fmt}", br.read(sz * n)))
            raise ValueError(f"unknown array element type {et}")
        raise ValueError(f"unknown KV value type {vt}")

    def parse():
        magic = br.read(4)
        if magic != b"GGUF":
            raise ValueError(f"bad magic {magic!r}")
        version = br.u32()
        tensor_count = br.u64()
        kv_count = br.u64()
        if not (1 <= tensor_count < 100000):
            raise ValueError(f"implausible tensor_count {tensor_count}")
        if kv_count > 100000:
            raise ValueError(f"implausible kv_count {kv_count}")
        kv = {}
        for _ in range(kv_count):
            key = read_str().decode("utf-8", "replace")
            if (not key or len(key) > 256
                    or any(ord(c) > 127 or c in "\r\n" for c in key)):
                raise ValueError(f"implausible KV key {key!r}")
            vt = br.u32()
            if vt > 12:
                raise ValueError(f"unknown KV value type {vt} (key={key!r})")
            kv[key] = read_value(vt)
        tensors = []
        for _ in range(tensor_count):
            name = read_str().decode("utf-8", "replace")
            if not name or len(name) > 256 or any(ord(c) > 127 for c in name):
                raise ValueError(f"implausible tensor name {name!r}")
            nd = br.u32()
            if not (1 <= nd <= 16):
                raise ValueError(f"implausible n_dims {nd} for {name!r}")
            dims = tuple(br.u64() for _ in range(nd))
            ggml_type = br.u32()
            if ggml_type > 40:
                raise ValueError(f"implausible ggml_type {ggml_type} for {name!r}")
            off = br.u64()
            tensors.append((name, dims, ggml_type, off))
        data_start = br.pos
        if data_start > CAP:
            raise ValueError(f"header size {data_start} exceeds 24MB cap")
        return version, kv, tensors, data_start

    return parse


def parse_shard(label, path, errors):
    last_err = None
    for conv in ("u64", "u32pad"):
        br = BoundedReader(path, CAP)
        try:
            parse = build_parser(br, conv)
            version, kv, tensors, data_start = parse()
            return {
                "label": label,
                "path": path,
                "file_size": br.file_size,
                "convention": conv,
                "version": version,
                "kv": kv,
                "tensors": tensors,
                "data_start": data_start,
                "header_bytes": br.pos,
            }
        except Exception as e:
            last_err = f"[{label}/{conv}] {type(e).__name__}: {e}"
            errors.append(last_err)
        finally:
            br.close()
    raise RuntimeError(f"{path}: failed under both conventions; last: {last_err}")


def compute_sizes(shard, errors):
    file_size = shard["file_size"]
    data_start = shard["data_start"]
    ts = sorted(shard["tensors"], key=lambda t: t[3])
    offs = [t[3] for t in ts]
    first_off = offs[0]
    sizes = [offs[i + 1] - offs[i] for i in range(len(offs) - 1)]
    if first_off == 0:
        # offsets relative to data start (first tensor at 0): spec formula
        last_sz = file_size - data_start - offs[-1]
        residual = file_size - data_start - (sum(sizes) + last_sz)
        mode = "relative(first=0)"
    else:
        # offsets appear absolute (byte position in file)
        last_sz = file_size - offs[-1]
        residual = file_size - first_off - (sum(sizes) + last_sz)
        residual_spec = file_size - data_start - (sum(sizes) + last_sz)
        mode = f"absolute(first={first_off}, data_start={data_start}); residual_spec_formula={residual_spec}"
        errors.append(f"[{shard['label']}] WARNING first tensor offset {first_off} != 0; used absolute interpretation")
    sizes.append(last_sz)
    shard["tensors_sorted"] = list(zip(ts, sizes))
    shard["sum_sizes"] = sum(sizes)
    shard["residual"] = residual
    shard["offset_mode"] = mode
    return shard


def group_of(name):
    m = R_EXPERT.search(name)
    if m:
        layer = int(m.group(1))
        if 0 <= layer <= 37:
            return "G1"
        if 38 <= layer <= 47:
            return "G2"
    if "shexp" in name:
        return "G3"
    if "ple" in name or "per_layer" in name:
        return "G4"
    return "G5"


def fmt(v, nd=2):
    return f"{v:.{nd}f}"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    errors = []
    shards = []
    for label, path in SHARDS:
        shards.append(parse_shard(label, path, errors))
        shards[-1] = compute_sizes(shards[-1], errors)

    # ---------- tensor_inventory.csv ----------
    csv_path = os.path.join(OUTDIR, "tensor_inventory.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shard", "name", "offset", "size_bytes"])
        for sh in shards:
            for (name, dims, gtype, off), sz in sh["tensors_sorted"]:
                w.writerow([sh["label"], name, off, sz])

    # ---------- grouping ----------
    grand_total = sum(sh["sum_sizes"] for sh in shards)
    grand_tensors = sum(len(sh["tensors"]) for sh in shards)

    groups = {g: {"count": 0, "bytes": 0} for g in ("G1", "G2", "G3", "G4", "G5")}
    per_layer_expert = {}
    all_tensors = []  # (name, size, shard_label, dims)
    for sh in shards:
        for (name, dims, gtype, off), sz in sh["tensors_sorted"]:
            g = group_of(name)
            groups[g]["count"] += 1
            groups[g]["bytes"] += sz
            m = R_EXPERT.search(name)
            if m:
                layer = int(m.group(1))
                if g in ("G1", "G2"):
                    per_layer_expert[layer] = per_layer_expert.get(layer, 0) + sz
            all_tensors.append((name, sz, sh["label"], dims, g))

    MiB = 1024.0 ** 2
    GiB = 1024.0 ** 3
    pred_k10 = grand_total - groups["G1"]["bytes"]
    pred_k10_ple = grand_total - groups["G1"]["bytes"] - groups["G4"]["bytes"]
    top10 = sorted(all_tensors, key=lambda t: -t[1])[:10]

    conv_used = {sh["label"]: sh["convention"] for sh in shards}

    # ---------- breakdown_calc.md ----------
    md = []
    md.append("# GGUF shard 2-4 tensor inventory — breakdown (stepA_h1h2 / s2_inventory)\n")
    md.append(f"- Files: 3 shards of `{os.path.basename(SHARDS[0][0])}` (shards 2, 3, 4 of 4)")
    md.append(f"- String-length convention used: " +
              "; ".join(f"shard {k} -> `{v}`" for k, v in conv_used.items()))
    md.append(f"- Total tensors: **{grand_tensors}** (expected 1224)")
    md.append("")
    md.append("## Per-shard summary\n")
    md.append("| shard | convention | tensor_count | header_bytes | first_data_offset | offset_mode | sum_tensor_bytes | file_size | residual |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for sh in shards:
        first_off = sh["tensors_sorted"][0][0][3]
        md.append(f"| {sh['label']} | {sh['convention']} | {len(sh['tensors'])} | {sh['header_bytes']} | "
                  f"{first_off} | {sh['offset_mode']} | {sh['sum_sizes']} | {sh['file_size']} | {sh['residual']} |")
    md.append("")
    md.append(f"Grand total tensor bytes: **{grand_total}** = {fmt(grand_total / MiB)} MiB = {fmt(grand_total / GiB, 4)} GiB\n")
    md.append("## Group table (regex R = `^blk\\.\\d+\\.ffn_(up|down|gate|gate_up)_(ch|)exps`, SEARCH)\n")
    md.append("| group | rule | tensors | bytes | MiB |")
    md.append("|---|---|---|---|---|")
    rules = {
        "G1": "R matches, layer 0..37",
        "G2": "R matches, layer 38..47",
        "G3": "name contains 'shexp'",
        "G4": "name contains 'ple' or 'per_layer'",
        "G5": "everything else",
    }
    for g in ("G1", "G2", "G3", "G4", "G5"):
        b = groups[g]["bytes"]
        md.append(f"| {g} | {rules[g]} | {groups[g]['count']} | {b} | {fmt(b / MiB)} |")
    md.append("")
    md.append("## VRAM predictions (K=10, ngl=all)\n")
    md.append(f"- **PREDICTED_VRAM_K10** = grand_total - G1 = {grand_total} - {groups['G1']['bytes']} = "
              f"**{pred_k10}** bytes = **{fmt(pred_k10 / MiB)} MiB** = **{fmt(pred_k10 / GiB, 4)} GiB**")
    md.append(f"- **PREDICTED_VRAM_K10_PLE_CPU** = grand_total - G1 - G4 = {grand_total} - {groups['G1']['bytes']} - {groups['G4']['bytes']} = "
              f"**{pred_k10_ple}** bytes = **{fmt(pred_k10_ple / MiB)} MiB** = **{fmt(pred_k10_ple / GiB, 4)} GiB**")
    md.append("")
    md.append("## Per-layer routed-expert bytes (sum of G1/G2 tensors of that layer)\n")
    md.append("| layer | expert tensors | bytes | MiB |")
    md.append("|---|---|---|---|")
    for layer in (0, 1, 37, 38, 47):
        cnt = sum(1 for t in all_tensors if R_EXPERT.search(t[0]) and int(R_EXPERT.search(t[0]).group(1)) == layer)
        b = per_layer_expert.get(layer, 0)
        md.append(f"| {layer} | {cnt} | {b} | {fmt(b / MiB)} |")
    md.append("")
    md.append("## 10 largest single tensors\n")
    md.append("| # | name | shard | MiB |")
    md.append("|---|---|---|---|")
    for i, (name, sz, lab, dims, g) in enumerate(top10, 1):
        md.append(f"| {i} | {name} | {lab} | {fmt(sz / MiB)} |")
    md.append("")
    if errors:
        md.append("## Errors / warnings\n")
        for e in errors:
            md.append(f"- {e}")
        md.append("")
    with open(os.path.join(OUTDIR, "breakdown_calc.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # ---------- hparams_from_dims.txt ----------
    layer0 = sorted([t for t in all_tensors if t[0].startswith("blk.0.")], key=lambda t: t[0])
    hp = ["# Layer-0 tensors: name + dims (Qwen3.8-Flash-Next, UD-Q4_K_XL)",
          "# (read n_embd / n_head / head_dim off of attn_q/attn_k/attn_v/attn_o row shapes)", ""]
    for name, sz, lab, dims, g in layer0:
        hp.append(f"{name}  dims={list(dims)}  size={sz} bytes ({fmt(sz / MiB)} MiB)  [{g}]")
    with open(os.path.join(OUTDIR, "hparams_from_dims.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(hp) + "\n")

    # ---------- kv_metadata.txt ----------
    kv_lines = []
    for sh in shards:
        kv_lines.append(f"=== shard {sh['label']} (kv_count={len(sh['kv'])}, version={sh['version']}) ===")
        for k in sorted(sh["kv"]):
            v = sh["kv"][k]
            if isinstance(v, list):
                vs = ", ".join(repr(x) for x in v[:8])
                if len(v) > 8:
                    vs += f", ... ({len(v)} total)"
                v = "[" + vs + "]"
            kv_lines.append(f"{k} = {v}")
        kv_lines.append("")
    with open(os.path.join(OUTDIR, "kv_metadata.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(kv_lines))

    # ---------- stdout summary ----------
    print("=== SUMMARY ===")
    print(f"convention: " + ", ".join(f"{k}={v}" for k, v in conv_used.items()))
    for sh in shards:
        first_off = sh["tensors_sorted"][0][0][3]
        print(f"shard {sh['label']}: tensor_count={len(sh['tensors'])} header_bytes={sh['header_bytes']} "
              f"first_data_offset={first_off} sum_sizes={sh['sum_sizes']} file_size={sh['file_size']} "
              f"residual={sh['residual']}")
    print(f"grand_tensors={grand_tensors}")
    print(f"grand_bytes={grand_total} grand_MiB={fmt(grand_total / MiB)} grand_GiB={fmt(grand_total / GiB, 4)}")
    for g in ("G1", "G2", "G3", "G4", "G5"):
        print(f"{g}: count={groups[g]['count']} bytes={groups[g]['bytes']} MiB={fmt(groups[g]['bytes'] / MiB)}")
    print(f"PREDICTED_VRAM_K10={pred_k10} bytes MiB={fmt(pred_k10 / MiB)} GiB={fmt(pred_k10 / GiB, 4)}")
    print(f"PREDICTED_VRAM_K10_PLE_CPU={pred_k10_ple} bytes MiB={fmt(pred_k10_ple / MiB)} GiB={fmt(pred_k10_ple / GiB, 4)}")
    for layer in (0, 1, 37, 38, 47):
        b = per_layer_expert.get(layer, 0)
        print(f"layer {layer} expert bytes={b} MiB={fmt(b / MiB)}")
    print("--- top10 ---")
    for i, (name, sz, lab, dims, g) in enumerate(top10, 1):
        print(f"{i}. {name} [{lab}] {fmt(sz / MiB)} MiB")
    if errors:
        print("--- errors ---")
        for e in errors:
            print(e)
    print("=== END ===")


if __name__ == "__main__":
    main()
