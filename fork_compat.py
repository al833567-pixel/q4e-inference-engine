# -*- coding: utf-8 -*-
"""B 實作：llama-cpp-python(0.3.35, 72B) ↔ fork llama.dll(80B) llama_model_params 相容層。

根因（已證實，見 tools/abi_probe2.py + params80_load.py A1_ngl LOAD_OK）：
  - wheel 的 llama_model_params 是 72 位元組；fork（build 10685）是 80 位元組。
  - fork 在 bool 群之前多了 2 個 i32 欄位（offset 64..71，C 預設 0，
    推測為 tensor_read_lazy 之類 fork 專屬參數）。
  - wheel 傳 72B → fork 讀自己的 80B → bool 群（vocab_only 等）偏移到 OOB 堆積位元組
    （實測 vocab_only=199, no_alloc=254 垃圾值）→ llama-hparams.cpp:55 fatal（空訊息，
    stderr 直寫 + abort，繞過 llama_log_set 回呼）。
  - 用 fork 自己的 80B C 預設 + n_gpu_layers=10 → LOAD_OK（A1_ngl，224s）。

用法：
    os.environ.setdefault("LLAMA_CPP_LIB_PATH", r"D:\\llama")
    os.chdir(r"D:\\llama")            # backend 預設掃描 CWD（CUDA/RPC/haswell）
    import fork_compat
    fork_compat.install()
    from llama_cpp import Llama        # 之後照常用 wheel 高層 API

install() 包裝 llama_cpp.llama_model_load_from_file：
把 wheel 的 72B 實例轉成 fork 的 80B（前 64B 同佈局直拷；64..71 = fork C 預設；
bool 群 64..69 → 72..77），device 策略強制用 A1 已證實的 C 預設
（split_mode=1, load_mode=auto, main_gpu=1），n_gpu_layers 保留 wheel 值。
不改 interfaces.py；不碰 wheel 原始碼；可重入（第二次 install 是 no-op）。
"""
from __future__ import annotations

import ctypes

_installed = False
_FORK_SIZE = 80


def install() -> None:
    global _installed
    if _installed:
        return
    import llama_cpp  # env 必須已由呼叫者設定好
    import llama_cpp.llama_cpp as mod

    lib = mod._lib

    class ForkModelParams(ctypes.Structure):
        _fields_ = [("raw", ctypes.c_ubyte * _FORK_SIZE)]

    # fork 自己的 C 預設（80B）——與 llama-server 使用同一組預設值。
    # 注意：lib.llama_model_default_params 與 wheel 模組屬性是同一個函式物件，
    # 改 restype 會污染 wheel 的 Llama()（實測：wheel 拿到 80B 實例）。讀完立刻還原。
    f_def = lib.llama_model_default_params
    _prev_def_restype = f_def.restype
    f_def.argtypes = []
    f_def.restype = ForkModelParams
    _fork_defaults = bytes(f_def().raw)
    f_def.restype = getattr(mod, "llama_model_params", _prev_def_restype)  # 還原 wheel 72B 類別

    _f80 = lib.llama_model_load_from_file
    _orig = mod.llama_model_load_from_file
    _f80.argtypes = [ctypes.c_char_p, ForkModelParams]
    _f80.restype = _orig.restype

    def _load80(path: bytes, params72) -> object:
        b72 = bytes(params72)
        if len(b72) != 72:
            raise ValueError(f"expected wheel 72B params, got {len(b72)}")
        p = ForkModelParams()
        r = p.raw
        r[0:64] = b72[0:64]  # 相同前綴：ptrs、n_gpu_layers、split、load、main_gpu、callbacks、kv_overrides
        r[64:72] = _fork_defaults[64:72]  # fork 專屬 i32 群（C 預設，含 tensor_read_lazy 類欄位）
        r[72:78] = b72[64:70]  # bool 群移位：vocab_only/check_tensors/use_extra_bufts/no_host/no_alloc/load_mtp
        r[78:80] = b"\x00\x00"  # padding
        # device 策略：強制 A1 已證實的 C 預設（wheel 的 0/0/MMAP 未在 fork 驗證）
        r[20:24] = _fork_defaults[20:24]  # split_mode = 1
        r[24:28] = _fork_defaults[24:28]  # load_mode = -1 (auto → mmap)
        r[28:32] = _fork_defaults[28:32]  # main_gpu = 1
        return _f80(path, p)

    mod.llama_model_load_from_file = _load80
    _installed = True
