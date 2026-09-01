# -*- coding: utf-8 -*-
"""N-gram 表（per_layer_token_embd）解析 + 4.5-bit 解碼 + SSD 隨機讀 bench（A-3，CPU-only）。

實測事實（tensor_audit.py + 本專案 probe，2026-07）：
- 張量：`per_layer_token_embd.weight`
  dims [160, 320,001,536]＝每 row 160 值 × 320,001,536 rows
  51,200,245,760 值、28,800,138,240 bytes → **恰好 4.5 bit/值 ＝ 18 bytes/32 值**
- row ＝ 5 blocks × 18 B ＝ 90 B；每 token 查 16 rows ≈ 1,440 B（1.4 KB，與 01 檔一致）
- 20,000,096 entries × 16 subrows（官方名額 20M entries，padding 到 20,000,096）
- 16 row IDs ＝ 8×bigram head (x_{t-1}, x_t) + 8×trigram head (x_{t-2}, x_{t-1}, x_t)
  @ layer 2（確定性 hash，位置可提前算出 → 可 host offload + 非同步預取）

4.5-bit block 佈局（Q4_0 系，18 B/32 值，與實測位元組數精確吻合）：
  d = f16 (2 B)；qs[16] 每 byte 裝 2 個 4-bit 量化值（低 nibble＝偶數索引）
  x[i] = (q[i] - 8) * d

v2：q4e 核心（C++）依此合約實作 `offload()` / `prefetch()` 接真實 host/SSD
offload + 非同步預取（簽章見 interfaces.NgramTable；本檔 fake 版只記路徑/同步讀，供合約測試）。
"""
from __future__ import annotations

import os
import random
import struct
import time
from typing import Any, Optional

import numpy as np

import interfaces

NG_TENSOR = "per_layer_token_embd.weight"
BLOCK_ELEMS = 32
BLOCK_BYTES = 18
BITS_PER_ELT = 4.5
N_HEADS = 16          # 8 bigram + 8 trigram
ROWS_PER_ENTRY = 16   # 每 entry 16 個 subrows（bigram 0-7 / trigram 8-15）

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x00000100000001B3


# --------------------------------------------------------------------------- #
# 解碼
# --------------------------------------------------------------------------- #
def decode_block_45(block: bytes | bytearray) -> list[float]:
    """解碼一個 18-byte block（32 值，4.5 bit/值）→ list[float]。

    佈局：d=f16 LE (2B) + qs[16]（byte j 的低 nibble＝元素 2j、高 nibble＝元素 2j+1）。
    x[i] = (q[i] - 8) * d
    """
    if len(block) < BLOCK_BYTES:
        raise ValueError(f"block too short: {len(block)} < {BLOCK_BYTES}")
    d = struct.unpack_from("<e", block, 0)[0]
    qs = np.frombuffer(bytes(block[2 : 2 + 16]), dtype=np.uint8)
    q = np.empty(32, dtype=np.uint8)
    q[0::2] = qs & 0xF
    q[1::2] = (qs >> 4) & 0xF
    return ((q.astype(np.float64) - 8.0) * d).tolist()


# --------------------------------------------------------------------------- #
# GGUF header reader（只讀 header，不碰資料體；已對本機 4 片實測驗證）
# --------------------------------------------------------------------------- #
def _skip_metadata(buf: bytes, p: int, kvc: int) -> int:
    for _ in range(kvc):
        l = struct.unpack_from("<Q", buf, p)[0]
        p += 8 + l  # key 長 + key bytes
        vt = struct.unpack_from("<I", buf, p)[0]
        p += 4
        if vt == 8:  # string（blob）
            l2 = struct.unpack_from("<Q", buf, p)[0]
            p += 8 + l2
        elif vt == 9:  # array
            et = struct.unpack_from("<I", buf, p)[0]
            p += 4
            cnt = struct.unpack_from("<Q", buf, p)[0]
            p += 8
            sizes = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}
            if et == 8:
                for _ in range(cnt):
                    l3 = struct.unpack_from("<Q", buf, p)[0]
                    p += 8 + l3
            elif et in sizes:
                p += sizes[et] * cnt
            else:
                raise ValueError(f"unsupported array element type {et}")
        else:
            sizes = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}
            if vt not in sizes:
                raise ValueError(f"unsupported value type {vt}")
            p += sizes[vt]
    return p


def read_tensor_index(gguf_path: str, header_cap: int = 64 * 1024 * 1024) -> list[dict]:
    """讀 GGUF header 的張量表（名稱/dims/gtype/offset/size）。

    gguf_path 可以是單一 .gguf 檔或目錄（掃全部 *.gguf）。
    回 list[dict(file, name, dims, gtype, offset, size)]。
    """
    if os.path.isdir(gguf_path):
        files = sorted(f for f in os.listdir(gguf_path) if f.endswith(".gguf"))
        if not files:
            raise FileNotFoundError(f"no .gguf in {gguf_path}")
        paths = [os.path.join(gguf_path, f) for f in files]
    else:
        paths = [gguf_path]

    out: list[dict] = []
    for fn in paths:
        size = os.path.getsize(fn)
        with open(fn, "rb") as f:
            buf = f.read(min(size, header_cap))
        if buf[:4] != b"GGUF":
            raise ValueError(f"{fn}: not a GGUF file (bad magic)")
        p = 8
        tc = struct.unpack_from("<Q", buf, p)[0]
        p += 8
        kvc = struct.unpack_from("<Q", buf, p)[0]
        p += 8
        p = _skip_metadata(buf, p, kvc)
        entries = []
        for _ in range(tc):
            l = struct.unpack_from("<Q", buf, p)[0]
            name = buf[p + 8 : p + 8 + l].decode("utf-8")
            p += 8 + l
            nd = struct.unpack_from("<I", buf, p)[0]
            p += 4
            dims = [struct.unpack_from("<Q", buf, p + 8 * j)[0] for j in range(nd)]
            p += 8 * nd
            gtype = struct.unpack_from("<I", buf, p)[0]
            p += 4
            offset = struct.unpack_from("<Q", buf, p)[0]
            p += 8
            entries.append((name, list(dims), int(gtype), int(offset)))
        # size：下一張量 offset 之差；最後一張量到 EOF（標準 GGUF 假設）
        for i, (name, dims, gtype, offset) in enumerate(entries):
            sz = entries[i + 1][3] - offset if i + 1 < len(entries) else size - offset
            out.append(dict(file=fn, name=name, dims=dims, gtype=gtype, offset=offset, size=int(sz)))
    return out


# --------------------------------------------------------------------------- #
# NgramTable（interfaces.NgramTable 的 A 側實作）
# --------------------------------------------------------------------------- #
class NgramTable(interfaces.NgramTable):
    """解析（CPU-only）＋確定性 hash＋16-row lookup＋隨機讀 bench。

    B 覆寫 `offload()` / `prefetch()` 接真實 host/SSD offload + 非同步預取。
    """

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.file: Optional[str] = None
        self.offset = 0
        self.dims: Optional[tuple[int, int]] = None  # (row_len, n_rows)
        self.gtype: Optional[int] = None
        self.size = 0
        self.n_entries = 0
        self.offload_path: Optional[str] = None
        self._fh: Optional[Any] = None

    # -- shape ------------------------------------------------------------- #
    @property
    def row_len(self) -> Optional[int]:
        return self.dims[0] if self.dims else None

    @property
    def n_rows(self) -> Optional[int]:
        return self.dims[1] if self.dims else None

    def row_bytes(self) -> int:
        if self.row_len is None:
            raise RuntimeError("call parse() first")
        if self.row_len % BLOCK_ELEMS:
            raise ValueError(f"row_len {self.row_len} not multiple of {BLOCK_ELEMS}")
        return self.row_len // BLOCK_ELEMS * BLOCK_BYTES

    def parse(self, gguf_path: str) -> None:
        """在 gguf_path（檔或目錄）找 N-gram 張量並驗證 4.5-bit 佈局。"""
        for t in read_tensor_index(gguf_path):
            if t["name"] != NG_TENSOR:
                continue
            if self.dims is not None:
                raise RuntimeError(f"multiple {NG_TENSOR} tensors found")
            self.file = t["file"]
            self.offset = t["offset"]
            self.dims = (int(t["dims"][0]), int(t["dims"][1]))
            self.gtype = t["gtype"]
            self.size = t["size"]
            break
        if self.dims is None:
            raise FileNotFoundError(f"{NG_TENSOR} not found under {gguf_path}")
        row_len, n_rows = self.dims
        if row_len % BLOCK_ELEMS:
            raise ValueError(f"row_len {row_len} not multiple of {BLOCK_ELEMS}")
        expected = n_rows * row_len // BLOCK_ELEMS * BLOCK_BYTES
        if abs(expected - self.size) > 2:
            raise ValueError(f"tensor size {self.size} != 4.5-bit expected {expected}")
        if n_rows % ROWS_PER_ENTRY:
            raise ValueError(f"n_rows {n_rows} not multiple of {ROWS_PER_ENTRY}")
        self.n_entries = n_rows // ROWS_PER_ENTRY

    # -- hash（A 定義合約：FNV-1a 64 + per-head salt；確定性、可重放）-------- #
    @staticmethod
    def _fnv1a64(b: bytes) -> int:
        h = _FNV_OFFSET
        for x in b:
            h ^= x
            h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        return h

    def _hash(self, head: int, ids: tuple[int, ...]) -> int:
        raw = struct.pack("<I", head) + b"".join(struct.pack("<Q", i) for i in ids)
        return self._fnv1a64(raw) % self.n_entries

    def row_ids_for_token(self, t: int, t_prev1: int, t_prev2: int) -> list[int]:
        """回 16 個 row id：head 0-7＝bigram (t_prev1, t)，head 8-15＝trigram (t_prev2, t_prev1, t)。"""
        rids: list[int] = []
        for h in range(8):
            rids.append(self._hash(h, (t_prev1, t)) * ROWS_PER_ENTRY + h)
        for h in range(8, 16):
            rids.append(self._hash(h, (t_prev2, t_prev1, t)) * ROWS_PER_ENTRY + (h - 8))
        return rids

    # -- 讀取 / 解碼 --------------------------------------------------------- #
    def _ensure_fh(self):
        if self._fh is None:
            if self.file is None:
                raise RuntimeError("call parse() first")
            self._fh = open(self.file, "rb")
        return self._fh

    def read_row(self, row_id: int) -> list[float]:
        """讀 1 個 row（160 值）並解碼 → list[float]（160 個）。"""
        if self.file is None:
            raise RuntimeError("call parse() first")
        if not (0 <= row_id < self.n_rows):
            raise IndexError(f"row_id {row_id} out of range [0,{self.n_rows})")
        rb = self.row_bytes()
        fh = self._ensure_fh()
        fh.seek(self.offset + row_id * rb)
        data = fh.read(rb)
        if len(data) != rb:
            raise EOFError(f"short read at row {row_id}")
        n_blk = rb // BLOCK_BYTES
        return [v for blk in range(n_blk) for v in decode_block_45(data[blk * BLOCK_BYTES : (blk + 1) * BLOCK_BYTES])]

    def lookup(self, token_ids: list[int]) -> list[float]:
        """layer 2 查 16 rows（8 bigram + 8 trigram）→ 16×160＝2560 floats（E_t 拼接）。"""
        if not token_ids:
            raise ValueError("token_ids is empty")
        if self.file is None:
            raise RuntimeError("call parse() first")
        t = int(token_ids[-1])
        t1 = int(token_ids[-2]) if len(token_ids) >= 2 else 0
        t2 = int(token_ids[-3]) if len(token_ids) >= 3 else 0
        out: list[float] = []
        for rid in self.row_ids_for_token(t, t1, t2):
            out.extend(self.read_row(rid))
        return out

    # -- bench（CPU-only；量 SSD/page-cache 隨機讀延遲）---------------------- #
    def bench_random_reads(self, n: int = 1000, seed: int = 0, warmup: int = 10) -> dict[str, float]:
        """隨機讀 n 次（每次 1 row＝160 值），量延遲與吞吐。"""
        if self.file is None:
            raise RuntimeError("call parse() first")
        if n <= 0:
            raise ValueError("n must be > 0")
        rng = random.Random(seed)
        rb = self.row_bytes()
        fh = open(self.file, "rb")

        def one(rid: int) -> bytes:
            fh.seek(self.offset + rid * rb)
            return fh.read(rb)

        try:
            for _ in range(warmup):
                one(rng.randrange(self.n_rows))
            lat: list[float] = []
            for _ in range(n):
                rid = rng.randrange(self.n_rows)
                t0 = time.perf_counter()
                data = one(rid)
                lat.append((time.perf_counter() - t0) * 1000.0)
                if len(data) != rb:
                    raise EOFError("short read during bench")
        finally:
            fh.close()
        lat.sort()
        total_ms = sum(lat)
        return {
            "n": n,
            "bytes_per_read": rb,
            "mean_ms": round(total_ms / n, 4),
            "p50_ms": round(lat[n // 2], 4),
            "p99_ms": round(lat[min(n - 1, int(n * 0.99))], 4),
            "min_ms": round(lat[0], 4),
            "max_ms": round(lat[-1], 4),
            "throughput_mbps": round((n * rb / 1e6) / (total_ms / 1000.0), 2) if total_ms > 0 else 0.0,
            "mode": "buffered (page-cache 可能命中；B-3 用真實 offload 再量)",
        }

    def read_sequence(self, n: int = 1000, seed: int = 0, warmup: int = 10) -> list[int]:
        """重放 bench_random_reads(seed) 將讀取的 n 個 row id（不量測）— 確定性：同 seed → 同序列。"""
        if self.file is None:
            raise RuntimeError("call parse() first")
        if n <= 0:
            raise ValueError("n must be > 0")
        rng = random.Random(seed)
        for _ in range(warmup):
            rng.randrange(self.n_rows)
        return [rng.randrange(self.n_rows) for _ in range(n)]

    # -- B 覆寫點 ------------------------------------------------------------- #
    def offload(self, path: str) -> None:
        """fake：只記錄 offload 目標。B 覆寫做真實 host/SSD 離載（mmap/NO_BUFFERING）。"""
        self.offload_path = path

    def prefetch(self, token_ids: list[int]) -> None:
        """fake：同步讀（保證 page-cache 熱）。B 覆寫做真實非同步預取（與計算重疊）。"""
        if not token_ids:
            return
        t = int(token_ids[-1])
        t1 = int(token_ids[-2]) if len(token_ids) >= 2 else 0
        t2 = int(token_ids[-3]) if len(token_ids) >= 3 else 0
        for rid in self.row_ids_for_token(t, t1, t2):
            self.read_row(rid)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
