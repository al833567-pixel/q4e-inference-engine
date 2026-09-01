# -*- coding: utf-8 -*-
r"""B 診斷：比對 wheel(llama-cpp-python 0.3.35) 的 llama_model_params 佈局 vs D:\llama build 10685 DLL 實際期望。

做法：
  1) 呼叫 DLL 的 llama_model_default_params()（C 端用「自己的佈局」填預設值），dump 原始 bytes。
  2) 依 wheel 佈局解讀各欄位，看是否符合預期預設值。
     若 C 端預設值（n_gpu_layers=-1、load_mode=MMAP、bools=0...）出現在「錯的 offset」→ 佈局漂移。
"""
from __future__ import annotations

import ctypes
import os
import struct
import sys

sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("LLAMA_CPP_LIB_PATH", r"D:\llama")
import llama_cpp  # noqa: E402  (pure-ctypes bindings, loads D:\llama\llama.dll)
import llama_cpp.llama_cpp as _mod  # noqa: E402  (submodule holds _lib)

lib = _mod._lib  # the loaded llama library handle


def build_number() -> int:
    fn = getattr(lib, "llama_build_number", None)
    if fn is None:
        return -1
    try:
        fn.restype = ctypes.c_int
        return fn()
    except Exception as e:  # pragma: no cover
        print("llama_build_number err:", e)
        return -1


def dump_default_params() -> None:
    # 用 wheel 自己的 restype（by-value struct return 的正確 ABI 協議）
    obj = _mod.llama_model_default_params()
    sz = ctypes.sizeof(obj)
    raw = (ctypes.c_char * max(sz, 96)).from_address(ctypes.addressof(obj))
    raw = bytes(raw[: min(sz, 96)])
    print(f"wheel sizeof(llama_model_params) = {sz}")
    print("raw bytes:")
    for i in range(0, len(raw), 8):
        chunk = raw[i : i + 8]
        hexs = " ".join(f"{b:02x}" for b in chunk)
        as_i32 = struct.unpack_from("<ii", chunk)
        as_u64 = struct.unpack_from("<Q", chunk)[0]
        print(f"  off {i:3d}: {hexs}  | i32={as_i32} | u64={as_u64:#018x}")

    # wheel 佈局解讀
    print("\nwheel-layout decode (offset: field = value):")
    fields = [
        (0, "devices ptr"),
        (8, "tensor_buft_overrides ptr"),
        (16, "n_gpu_layers i32"),
        (20, "split_mode i32"),
        (24, "load_mode i32"),
        (28, "main_gpu i32"),
        (32, "tensor_split ptr"),
        (40, "progress_callback ptr"),
        (48, "progress_callback_user_data ptr"),
        (56, "kv_overrides ptr"),
        (64, "vocab_only bool"),
        (65, "check_tensors bool"),
        (66, "use_extra_bufts bool"),
        (67, "no_host bool"),
        (68, "no_alloc bool"),
        (69, "load_mtp bool"),
    ]
    for off, name in fields:
        if name.endswith("ptr"):
            v = struct.unpack_from("<Q", raw, off)[0]
            print(f"  {off:3d} {name:36s} = {v:#x}")
        elif "i32" in name:
            print(f"  {off:3d} {name:36s} = {struct.unpack_from('<i', raw, off)[0]}")
        else:
            print(f"  {off:3d} {name:36s} = {raw[off]}")


if __name__ == "__main__":
    print(f"llama_build_number = {build_number()}")
    dump_default_params()
