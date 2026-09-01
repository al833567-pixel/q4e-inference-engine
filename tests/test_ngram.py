# -*- coding: utf-8 -*-
"""A-3：N-gram 解析 + 4.5-bit 解碼 + 隨機讀 bench（純 CPU）。"""
import math
import os
import struct

import pytest

from ngram import (
    BLOCK_BYTES,
    NgramTable,
    decode_block_45,
    read_tensor_index,
)

REAL_DIR = r"D:\AI_Model\Qwen3.8-Flash-Next"


# --------------------------------------------------------------------------- #
# 解碼（手工參考值）
# --------------------------------------------------------------------------- #
def test_decode_block_reference():
    d = struct.pack("<e", 2.0)  # 0x4000
    # byte j 的低 nibble＝元素 2j、高 nibble＝元素 2j+1
    block = d + bytes([0x00, 0xFF, 0x53] + [0x00] * 13)
    v = decode_block_45(block)
    assert len(v) == 32
    assert v[0] == (0 - 8) * 2.0        # byte0 低 nibble=0
    assert v[1] == (0 - 8) * 2.0        # byte0 高 nibble=0
    assert v[2] == (15 - 8) * 2.0       # byte1=0xFF 低
    assert v[3] == (15 - 8) * 2.0       # byte1=0xFF 高
    assert v[4] == (3 - 8) * 2.0        # 0x53 低 nibble=3
    assert v[5] == (5 - 8) * 2.0        # 0x53 高 nibble=5
    assert v[6:] == [(-8.0) * 2.0] * 26


def test_decode_block_d1_range():
    d = struct.pack("<e", 1.0)
    block = d + b"\x00" * 16
    v = decode_block_45(block)
    assert all(x == -8.0 for x in v)
    block2 = d + b"\xFF" * 16
    v2 = decode_block_45(block2)
    assert all(x == 7.0 for x in v2)  # (15-8)*1


def test_decode_block_short_raises():
    with pytest.raises(ValueError):
        decode_block_45(b"\x00" * (BLOCK_BYTES - 1))


# --------------------------------------------------------------------------- #
# 合成 GGUF（驗證 parse/lookup 全流程，不碰 28.8 GB 真檔）
# --------------------------------------------------------------------------- #
def write_fake_gguf(path: str, n_rows: int = 128, row_len: int = 160) -> None:
    """寫一個最小 GGUF：1 個 per_layer_token_embd.weight，全部 row 填 d=1.0、quants=0。"""
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", 1)  # tensor_count
    out += struct.pack("<Q", 2)  # kv_count
    # kv pairs（本專案 parser 的值型別：8=string）
    for key, val in (("general.architecture", "qwen3_8_next"), ("general.name", "fake-45bit")):
        kb = key.encode()
        vb = val.encode()
        out += struct.pack("<Q", len(kb)) + kb
        out += struct.pack("<I", 8)
        out += struct.pack("<Q", len(vb)) + vb
    # tensor entry
    name = b"per_layer_token_embd.weight"
    out += struct.pack("<Q", len(name)) + name
    out += struct.pack("<I", 2)
    out += struct.pack("<Q", row_len)
    out += struct.pack("<Q", n_rows)
    out += struct.pack("<I", 20)  # gtype（與真檔一致；parser 只透傳）
    offset = len(out) + 8  # 資料起點＝offset 欄位本身之後
    out += struct.pack("<Q", offset)
    # 資料體：每 row 5 blocks × 18 B，d=1.0 (0x3C00)、quants=0 → 每值 (0-8)*1 = -8
    block = b"\x00\x3C" + b"\x00" * 16
    row = block * (row_len // 32)
    out += row * n_rows
    with open(path, "wb") as f:
        f.write(out)


@pytest.fixture()
def fake_gguf(tmp_path):
    p = os.path.join(str(tmp_path), "fake.gguf")
    write_fake_gguf(p)
    return p


def test_parse_synthetic(fake_gguf):
    t = NgramTable()
    t.parse(fake_gguf)
    assert t.dims == (160, 128)
    assert t.n_entries == 8  # 128 rows / 16
    assert t.row_bytes() == 90
    assert t.size == 128 * 90
    assert t.gtype == 20


def test_row_ids_deterministic_and_in_range(fake_gguf):
    t = NgramTable()
    t.parse(fake_gguf)
    r1 = t.row_ids_for_token(30, 20, 10)
    r2 = t.row_ids_for_token(30, 20, 10)
    assert r1 == r2
    assert len(r1) == 16
    assert all(0 <= x < t.n_rows for x in r1)
    # 不同 token → (幾乎必然)不同 row 集合
    r3 = t.row_ids_for_token(99, 20, 10)
    assert r3 != r1


def test_read_row_values(fake_gguf):
    t = NgramTable()
    t.parse(fake_gguf)
    row = t.read_row(0)
    assert len(row) == 160
    assert all(x == -8.0 for x in row)


def test_lookup_16_rows(fake_gguf):
    t = NgramTable()
    t.parse(fake_gguf)
    vals = t.lookup([10, 20, 30])
    assert len(vals) == 16 * 160
    assert all(x == -8.0 for x in vals)


def test_lookup_needs_history_padding(fake_gguf):
    t = NgramTable()
    t.parse(fake_gguf)
    vals = t.lookup([42])  # 只有 1 token → 前 2 個用 0 padding
    assert len(vals) == 16 * 160


def test_lookup_empty_raises(fake_gguf):
    t = NgramTable()
    t.parse(fake_gguf)
    with pytest.raises(ValueError):
        t.lookup([])


def test_bench_synthetic(fake_gguf):
    t = NgramTable()
    t.parse(fake_gguf)
    r = t.bench_random_reads(n=30, seed=1)
    for k in ("n", "bytes_per_read", "mean_ms", "p50_ms", "p99_ms", "min_ms", "max_ms", "throughput_mbps", "mode"):
        assert k in r
    assert r["bytes_per_read"] == 90
    assert r["p99_ms"] >= r["p50_ms"] >= r["min_ms"]
    # wall-time 量測本質非確定 → 只驗 sanity，不斷言位元相等
    assert r["mean_ms"] > 0
    # 可重放（seed 相同 → row id 序列相同；確定性在序列，不在延遲）
    s1 = t.read_sequence(n=30, seed=1)
    s2 = t.read_sequence(n=30, seed=1)
    assert s1 == s2
    assert len(s1) == 30
    assert all(0 <= x < t.n_rows for x in s1)


def test_parse_missing_tensor(tmp_path):
    p = os.path.join(str(tmp_path), "empty.gguf")
    out = bytearray(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0))
    with open(p, "wb") as f:
        f.write(out)
    with pytest.raises(FileNotFoundError):
        NgramTable().parse(p)


def test_fake_offload_records_path(fake_gguf):
    t = NgramTable()
    t.parse(fake_gguf)
    t.offload(r"D:\ng_offload")
    assert t.offload_path == r"D:\ng_offload"
    # fake prefetch 同步讀不拋錯
    t.prefetch([1, 2, 3])


# --------------------------------------------------------------------------- #
# 真檔（D:\AI_Model\Qwen3.8-Flash-Next）— 環境沒有則 skip
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir(REAL_DIR), reason="real GGUF dir not present")
def test_parse_real_gguf():
    t = NgramTable()
    t.parse(REAL_DIR)
    assert t.dims == (160, 320001536)
    assert t.size == 28800138240          # 51,200,245,760 值 × 4.5 bit
    assert t.size * 8 / (t.dims[0] * t.dims[1]) == pytest.approx(4.5, abs=1e-6)
    assert t.n_entries == 20000096
    assert t.row_bytes() == 90
    row = t.read_row(0)
    assert len(row) == 160
    assert all(math.isfinite(x) for x in row)
    r = t.bench_random_reads(n=32, seed=0)
    assert r["n"] == 32 and r["mean_ms"] >= 0
    t.close()


@pytest.mark.skipif(not os.path.isdir(REAL_DIR), reason="real GGUF dir not present")
def test_real_tensor_index():
    idx = read_tensor_index(REAL_DIR)
    names = {e["name"] for e in idx}
    assert "per_layer_token_embd.weight" in names
    assert "blk.1.ffn_gate_inp.weight" in names
    assert len(idx) >= 1000
