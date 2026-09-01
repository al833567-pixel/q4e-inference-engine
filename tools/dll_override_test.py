# -*- coding: utf-8 -*-
"""一次性測試：in-process 載入 + D:\llama DLL 覆寫（LLAMA_CPP_LIB_PATH）。

wheel（llama-cpp-python 0.3.35）的純 ctypes 綁定在 import 時讀
`LLAMA_CPP_LIB_PATH`，改載 D:\llama 的 llama.dll（支援 qwen4exp）。
ABI 相容性未驗證 —— 本腳本就是驗證：載入 + n_vocab + 16 token 生成。

跑法（先確認 VRAM 已釋放）：
  $env:LLAMA_CPP_LIB_PATH='D:\llama'
  py -3.14 -X utf8 tools\dll_override_test.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _s in (sys.stdout, sys.stderr):  # cp950 console → UTF-8
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

assert os.environ.get("LLAMA_CPP_LIB_PATH") == r"D:\llama", (
    "先設 $env:LLAMA_CPP_LIB_PATH='D:\\llama' 再跑"
)

import config as cfgmod  # noqa: E402


def first_gguf(d: str) -> str:
    if os.path.isfile(d):
        return d
    fs = sorted(f for f in os.listdir(d) if f.endswith(".gguf"))
    if not fs:
        raise FileNotFoundError(f"no .gguf in {d}")
    return os.path.join(d, fs[0])


def main() -> int:
    raw = cfgmod.load_config()
    ecfg = cfgmod.engine_config(raw)
    mp = first_gguf(ecfg.gguf_dir)
    print(f"model_path={mp}")
    print(f"ngl={ecfg.ngl} ctx={ecfg.max_ctx} batch={ecfg.n_batch}")

    from llama_cpp import Llama  # import 後才載 DLL

    t0 = time.time()
    llm = Llama(
        model_path=mp,
        n_gpu_layers=int(ecfg.ngl),
        n_ctx=ecfg.max_ctx,
        n_batch=ecfg.n_batch,
        verbose=True,
    )
    dt = time.time() - t0
    nv = llm.n_vocab()
    print(f"LOAD_OK seconds={dt:.1f} n_vocab={nv}")

    t0 = time.time()
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": "你好"}],
        max_tokens=16,
        temperature=0.0,
    )
    dt = time.time() - t0
    ch = out["choices"][0]["message"]["content"]
    print(f"GEN_OK seconds={dt:.2f} tokens={out['usage']['completion_tokens']}")
    print(f"content={ch!r}")
    print("SMOKE_OK_DLL_OVERRIDE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
