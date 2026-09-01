# -*- coding: utf-8 -*-
r"""B 診斷：二分 wheel 覆寫欄位 vs DLL 預設，找出觸發 llama-hparams.cpp:55 fatal 的欄位。

用法：py -3.14 tools\params_bisect.py <T0_wheel|T1_nonly|T2_mmap|T3_split|T4_main>
每個測試獨立 process（fatal → abort 會殺 process）。
直接呼叫 C 層 llama_model_load_from_file（與 wheel internals 相同），
params 以 DLL 自己的 llama_model_default_params() 為基底，只改該測試指定的欄位。
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

TEST = sys.argv[1] if len(sys.argv) > 1 else "T1_nonly"
TOOLS = r"D:\Nova2_home\AI研究\推理工具\tools"
LOG_PATH = os.path.join(TOOLS, f"bisect_{TEST}.log")
MODEL = r"D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"

_logf = open(LOG_PATH, "w", encoding="utf-8", buffering=1)


def _log(msg: str) -> None:
    _logf.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    _logf.flush()


os.environ.setdefault("LLAMA_CPP_LIB_PATH", r"D:\llama")
os.chdir(r"D:\llama")

import llama_cpp  # noqa: E402
import llama_cpp.llama_cpp as mod  # noqa: E402

lib = mod._lib

_GGML_LOG = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_cb = _GGML_LOG(lambda lvl, text, ud: (_logf.write(f"[{lvl}] {text.decode('utf-8','replace')}\n"), _logf.flush())[1] if text else None)
llama_cpp.llama_log_set(_cb, None)

params = mod.llama_model_default_params()
_log(f"== test {TEST}: defaults ngl={params.n_gpu_layers} split={params.split_mode} load={params.load_mode} main_gpu={params.main_gpu}")

if TEST == "T0_wheel":  # wheel 的精確行為
    params.n_gpu_layers = 10
    params.split_mode = 0
    params.main_gpu = 0
    params.load_mode = 1  # LLAMA_LOAD_MODE_MMAP
    params.vocab_only = False
elif TEST == "T1_nonly":
    params.n_gpu_layers = 10
elif TEST == "T2_mmap":
    params.n_gpu_layers = 10
    params.load_mode = 1
elif TEST == "T3_split":
    params.n_gpu_layers = 10
    params.split_mode = 0
elif TEST == "T4_main":
    params.n_gpu_layers = 10
    params.main_gpu = 0
else:
    raise SystemExit(f"unknown test {TEST}")

_log(f"== test {TEST}: mutated ngl={params.n_gpu_layers} split={params.split_mode} load={params.load_mode} main_gpu={params.main_gpu}")

llama_cpp.llama_backend_init()
_log("== backend_init done")

f_load = lib.llama_model_load_from_file
f_load.argtypes = [ctypes.c_char_p, mod.llama_model_params]
f_load.restype = ctypes.c_void_p

_log("== llama_model_load_from_file start")
model = f_load(MODEL.encode("utf-8"), params)
if model:
    _log(f"== LOAD_OK {TEST}")
    f_free = lib.llama_model_free
    f_free.argtypes = [ctypes.c_void_p]
    f_free.restype = None
    f_free(model)
else:
    _log(f"== LOAD_NULL {TEST}")
_log(f"== done {TEST}")
