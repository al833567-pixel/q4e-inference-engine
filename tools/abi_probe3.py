# -*- coding: utf-8 -*-
r"""B 診斷 3：量 fork 的 llama_context_params 大小（wheel 72B 模型參數已被證實過小 8B）。"""
from __future__ import annotations

import ctypes
import os

os.environ.setdefault("LLAMA_CPP_LIB_PATH", r"D:\llama")

import llama_cpp  # noqa: E402
import llama_cpp.llama_cpp as mod  # noqa: E402

lib = mod._lib

wheel_ctx_size = ctypes.sizeof(mod.llama_context_params)
print("wheel llama_context_params size =", wheel_ctx_size)

f_default = lib.llama_context_default_params
f_default.argtypes = []


def dump(size: int) -> bytes:
    class Raw(ctypes.Structure):
        _fields_ = [("b", ctypes.c_ubyte * size)]
    f_default.restype = Raw
    r = f_default()
    return bytes(r.b)


base = wheel_ctx_size
for extra in (0, 8, 16, 24, 32, 40):
    size = base + extra
    a = dump(size)
    b = dump(size)
    same = "SAME" if a == b else "DIFF"
    print(f"--- size={size} two-calls={same}")
    print("  tail:", a[base - 8:].hex())
