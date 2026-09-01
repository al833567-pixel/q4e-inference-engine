# -*- coding: utf-8 -*-
r"""B 診斷：抓取 llama-hparams.cpp:55 fatal error 的完整訊息。

原理：abort() 會丟掉 C stderr buffer（訊息遺失）。改用 llama_log_set 註冊 Python 回調，
C 端每行 log 同步進入 Python（先 flush 到檔）再 abort → 訊息必被保留。

重現先前失敗條件：CWD=D:\llama（載入 CUDA+RPC+haswell 全部後端）、ngl=10。
"""
from __future__ import annotations

import os
import sys
import time

LOG_PATH = r"D:\Nova2_home\AI研究\推理工具\tools\fatal_probe.log"

_logf = open(LOG_PATH, "w", encoding="utf-8", buffering=1)  # 行緩衝 + 顯式 flush


def _ts() -> str:
    return time.strftime("%H:%M:%S")


os.environ.setdefault("LLAMA_CPP_LIB_PATH", r"D:\llama")
os.chdir(r"D:\llama")  # 預設後端掃描 → CUDA + RPC + haswell（與先前失敗一致）

import ctypes  # noqa: E402
import llama_cpp  # noqa: E402

# ggml_log_callback: void (*)(int level, const char * text, void * user_data)
_GGML_LOG = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_cb = _GGML_LOG(lambda lvl, text, ud: (_logf.write(f"[{lvl}] {text.decode('utf-8', 'replace')}"), _logf.flush())[1] if text else None)
llama_cpp.llama_log_set(_cb, None)
_logf.write(f"== {_ts()} log callback registered, build check:\n")
try:
    import llama_cpp.llama_cpp as _mod
    _fn = getattr(_mod._lib, "llama_build_number", None)
    if _fn is not None:
        _fn.restype = ctypes.c_int
        _logf.write(f"llama_build_number = {_fn()}\n")
except Exception as e:  # pragma: no cover
    _logf.write(f"build number probe failed: {e}\n")
_logf.flush()

MODEL = r"D:\AI_Model\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"

_logf.write(f"== {_ts()} Llama() start ngl=10 ctx=2048 batch=512\n")
_logf.flush()
try:
    llm = llama_cpp.Llama(
        MODEL,
        n_gpu_layers=10,
        n_ctx=2048,
        n_batch=512,
        verbose=True,
    )
    _logf.write(f"== {_ts()} LOAD_OK\n")
    _logf.flush()
    print("LOAD_OK")
except Exception as e:
    _logf.write(f"== {_ts()} EXC {type(e).__name__}: {e}\n")
    _logf.flush()
    raise
_logf.write(f"== {_ts()} done\n")
