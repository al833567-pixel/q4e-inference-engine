# -*- coding: utf-8 -*-
"""B 共用層：真實引擎載入（singleton）＋專家權重 slice mmap I/O。

三個 --live 工具共用：wire_cache / ngram_offload / bench_tps --measure。

實測事實（本 session，probe_expert_tensors.py + A1/SMOKE）：
- qwen4exp fused MoE（expert-major，ggml 最外維＝expert）：
    blk.L.ffn_gate_exps.weight  dims [2560,640,512]  4.5 bit/elt → stride = size/512 = 921,600 B
    blk.L.ffn_up_exps.weight    dims [2560,640,512]  4.5 bit/elt → 921,600 B
    blk.L.ffn_down_exps.weight  dims [640,2560,512]  6   bit/elt → 1,228,800 B
    → 一個 routed 專家 ≈ 3.07 MB（與 01 檔的 3.1 MB 一致）
  expert e 佔 [base + e*stride, +stride) 連續區。
- shared 專家：blk.L.ffn_{gate,up,down}_shexp.weight（獨立張量，整張量＝一個 slice）。
- wheel 不暴露 per-token 專家路由決策（router_experts 拿不到）→ 工具用確定性
  fake_router（tools/wire_cache.py）作為文檔化替代，輸出標註。
- M（每層不同專家數）引擎外不可測 → bench 用上界 M=512（保守列）。
- place 語義：
    GPU 層（L < ngl）整層已住 VRAM（fork 整層載入）→ 零 I/O；
    CPU 層 hot tier（vram 或 ram）物理上都是「讀進 page cache」→ mmap slice read
    （與引擎自己的 mmap 共享同一 OS page cache，預取同時暖引擎的讀）。
  evict 語義：Windows 無 drop_caches → no-op（page cache 由 OS 管理，文檔化）。
"""
from __future__ import annotations

import mmap
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROLE_KEYS = ("gate", "up", "down")
_EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
_SHEXP_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_shexp\.weight$")


def first_gguf(d: str) -> str:
    """Llama() 不解析目錄 → 傳第一片 .gguf，其餘分片 llama.cpp 自發現。"""
    if os.path.isfile(d):
        return d
    cands = sorted(f for f in os.listdir(d) if f.endswith(".gguf"))
    if not cands:
        raise FileNotFoundError(f"{d} 內沒有 .gguf")
    return os.path.join(d, cands[0])


_engine = None


def load_engine(eng_raw: dict, ecfg) -> "Llama":
    """Singleton Llama。eng_raw = config 的 engine 節（含 lib_path）。"""
    global _engine
    if _engine is None:
        lib_dir = str(eng_raw.get("lib_path") or r"D:\llama")
        # import 前設定 backend（wheel 自帶 llama.cpp 載不了 qwen4exp；
        # fork 80B params 根因見 fork_compat.py）
        os.environ.setdefault("LLAMA_CPP_LIB_PATH", lib_dir)
        os.chdir(lib_dir)
        import fork_compat  # noqa: E402

        fork_compat.install()
        from llama_cpp import Llama  # noqa: E402

        ngl = ecfg.ngl
        ngl = 999 if str(ngl).lower() == "auto" else int(ngl)
        kwargs = dict(
            model_path=first_gguf(ecfg.gguf_dir),
            n_ctx=ecfg.max_ctx,
            n_batch=ecfg.n_batch,
            n_gpu_layers=ngl,
            verbose=False,
        )
        if ecfg.n_threads:
            kwargs["n_threads"] = ecfg.n_threads
        _engine = Llama(**kwargs)
    return _engine


class ExpertStore:
    """mmap 全部 shard（唯讀）；read_expert/read_shared＝真實 SSD→RAM 頁讀。

    與引擎共享 OS page cache：首次＝磁碟 I/O，之後＝RAM 速。
    bytes_read 累計（報告用）。
    """

    def __init__(self, gguf_dir: str, n_experts: int = 512) -> None:
        from ngram import read_tensor_index  # 本地已 sys.path 插入

        self.n_experts = n_experts
        self._mm: dict[str, mmap.mmap] = {}
        self.base: dict[int, dict[str, tuple[str, int, int]]] = {}   # L → role → (file, off, stride)
        self.shared: dict[int, list[tuple[str, int, int]]] = {}      # L → [(file, off, size)]
        for t in read_tensor_index(gguf_dir):
            m = _EXPS_RE.match(t["name"])
            if m:
                L = int(m.group(1))
                self.base.setdefault(L, {})[m.group(2)] = (t["file"], t["offset"], t["size"] // n_experts)
                continue
            m = _SHEXP_RE.match(t["name"])
            if m:
                self.shared.setdefault(int(m.group(1)), []).append((t["file"], t["offset"], t["size"]))
        self.n_layers = len(self.base)
        for L in range(self.n_layers):
            got = set(self.base.get(L, {}))
            if got != set(ROLE_KEYS):
                raise RuntimeError(f"layer {L} exps 缺角色：缺 {set(ROLE_KEYS) - got}")
        self.bytes_read = 0
        self._fh: dict[str, object] = {}  # 防 GC 關掉底層句柄（mmap 3.14 不可掛屬性）

    def _m(self, path: str) -> mmap.mmap:
        m = self._mm.get(path)
        if m is None:
            fh = open(path, "rb")
            m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._fh[path] = fh
            self._mm[path] = m
        return m

    def read_expert(self, layer: int, expert: int) -> int:
        """讀一個 routed 專家的 3 個 slice（gate/up/down）；回總 bytes。"""
        tot = 0
        for role in ROLE_KEYS:
            path, off, stride = self.base[layer][role]
            m = self._m(path)
            m.seek(off + expert * stride)
            m.read(stride)
            tot += stride
        self.bytes_read += tot
        return tot

    def read_shared(self, layer: int) -> int:
        tot = 0
        for path, off, size in self.shared.get(layer, ()):
            m = self._m(path)
            m.seek(off)
            m.read(size)
            tot += size
        self.bytes_read += tot
        return tot

    def expert_bytes(self, layer: int) -> int:
        return sum(s for _, _, s in self.base[layer].values())

    def close(self) -> None:
        for m in self._mm.values():
            m.close()
        for fh in self._fh.values():
            fh.close()
        self._mm = {}
        self._fh = {}
