# -*- coding: utf-8 -*-
r"""B 診斷 2：量 fork 的 llama_model_params 真實大小與 64 位元組以後的欄位。

方法：llama_model_default_params() 傳回 fork 自己的結構體。
用「比 wheel 72B 更大的 restype」讀回，每次呼叫 2 次比對：
  - fork 寫入的位元組：兩次呼叫穩定
  - fork 大小之外的位元組：堆積殘留，可能變動
"""
from __future__ import annotations

import ctypes
import os

os.environ.setdefault("LLAMA_CPP_LIB_PATH", r"D:\llama")

import llama_cpp.llama_cpp as mod  # noqa: E402

lib = mod._lib
f_default = lib.llama_model_default_params
f_default.argtypes = []


def dump(size: int) -> bytes:
    class Raw(ctypes.Structure):
        _fields_ = [("b", ctypes.c_ubyte * size)]
    f_default.restype = Raw
    r = f_default()
    return bytes(r.b)


sizes = [72, 80, 88, 96, 104, 112]
for size in sizes:
    a = dump(size)
    b = dump(size)
    same = "SAME" if a == b else "DIFF"
    print(f"--- size={size} two-calls={same}")
    # print bytes 56..end, two hex rows
    print("  56..:", a[56:].hex())

# interpret the 64..80 region of the stable 80-byte read
r80 = dump(80)
print("\n== interpret 64..80 as fork fields (candidate layouts) ==")
print("i32@64 =", ctypes.c_int32.from_buffer_copy(r80[64:68]).value,
      " i32@68 =", ctypes.c_int32.from_buffer_copy(r80[68:72]).value)
print("bools@72..79 =", r80[72:80].hex())
print("bools@64..71 (wheel layout) =", r80[64:72].hex())
