# -*- coding: utf-8 -*-
r"""B 診斷 4：用 fork 自己的 80B llama_model_params（C 預設 + 指定欄位）載入模型。

用法：py -3.14 tools\params80_load.py <A1_ngl|A2_server>
  A1_ngl   : fork 預設 + n_gpu_layers=10
  A2_server: fork 預設 + n_gpu_layers=10 + main_gpu=0（完全模擬 llama-server 的參數）
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

TEST = sys.argv[1] if len(sys.argv) > 1 else "A1_ngl"
TOOLS = r"D:\Nova2_home\AI研究\推理工具\tools"
LOG_PATH = os.path.join(TOOLS, f"params80_{TEST}.log")
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

# fork 的 80B 結構體：以 fork 自己的 C 預設為基底
class ForkModelParams(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_ubyte * 80)]


f_default = lib.llama_model_default_params
f_default.argtypes = []
f_default.restype = ForkModelParams
p = f_default()
p.raw[16:20] = (10).to_bytes(4, "little")  # n_gpu_layers = 10
if TEST == "A2_server":
    p.raw[28:32] = (0).to_bytes(4, "little")  # main_gpu = 0
_log(f"== test {TEST}: 80B params, ngl=10, main_gpu={int.from_bytes(p.raw[28:32],'little')}, "
     f"split={int.from_bytes(p.raw[20:24],'little')}, load={int.from_bytes(p.raw[24:28],'little')}, bools@72..79={bytes(p.raw[72:80]).hex()}")

llama_cpp.llama_backend_init()
_log("== backend_init done")

f_load = lib.llama_model_load_from_file
f_load.argtypes = [ctypes.c_char_p, ForkModelParams]
f_load.restype = ctypes.c_void_p

_log("== llama_model_load_from_file start")
model = f_load(MODEL.encode("utf-8"), p)
if model:
    _log(f"== LOAD_OK {TEST}")
    # 驗證：取得模型資訊
    f_nlayers = lib.llama_model_n_layers
    f_nlayers.argtypes = [ctypes.c_void_p]
    f_nlayers.restype = ctypes.c_int
    _log(f"== n_layers={f_nlayers(model)}")
    f_free = lib.llama_model_free
    f_free.argtypes = [ctypes.c_void_p]
    f_free.restype = None
    f_free(model)
else:
    _log(f"== LOAD_NULL {TEST}")
_log(f"== done {TEST}")
