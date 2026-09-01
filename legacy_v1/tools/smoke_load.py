# -*- coding: utf-8 -*-
"""B-1：載 GGUF + 冒煙（Track B 執行；Track A 環境跑會預期失敗，只驗語法）。

用法：py -3.14 tools\\smoke_load.py [--prompt 你好]

預期成功輸出（RUNBOOK_B §1）：
  loaded, n_params=..., first_token=...
失敗處理：見 RUNBOOK_B §1 與 §備案 A/B。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _s in (sys.stdout, sys.stderr):  # cp950 console → UTF-8
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config as cfgmod  # noqa: E402
import interfaces  # noqa: E402


def first_gguf(d: str) -> str:
    """B-1 修：llama-cpp-python 0.3.x 的 Llama() 不解析目錄（gguf_init_from_file
    把目錄當檔開 → Permission denied）。要傳「第一片 .gguf」路徑，
    llama.cpp 自行發現其餘分片（log: additional 3 GGUFs metadata loaded）。"""
    if os.path.isfile(d):
        return d
    cands = sorted(f for f in os.listdir(d) if f.endswith(".gguf"))
    if not cands:
        raise FileNotFoundError(f"{d} 內沒有 .gguf")
    return os.path.join(d, cands[0])


def main() -> int:
    raw = cfgmod.load_config()
    ecfg = cfgmod.engine_config(raw)
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="用一句話解釋什麼是快取。")
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--ngl", default=ecfg.ngl, help="覆蓋 config engine.ngl（auto=999）")
    args = ap.parse_args()

    if not os.path.isdir(ecfg.gguf_dir):
        print(f"FAIL: GGUF 目錄不存在：{ecfg.gguf_dir}（檢查 config.yaml engine.gguf_dir）")
        return 2

    # B：import llama_cpp 之前設定 backend 路徑（wheel 自帶 llama.cpp 載不了 qwen4exp，
    # 必須指到 fork D:\llama\llama.dll）；fork_compat 把 wheel 的 72B model params
    # 轉成 fork 的 80B（根因見 fork_compat.py docstring 與 tools/abi_probe2.py）。
    lib_dir = str((raw.get("engine") or {}).get("lib_path") or r"D:\llama")
    os.environ.setdefault("LLAMA_CPP_LIB_PATH", lib_dir)
    os.chdir(lib_dir)  # backend 預設掃描 CWD（CUDA/RPC/haswell）
    try:
        import fork_compat  # noqa: E402  （B：72B→80B params 相容層；install() 才 import llama_cpp）
        fork_compat.install()
        from llama_cpp import Llama  # noqa: E402
    except ImportError as e:
        print(f"FAIL: llama_cpp import 失敗：{e}\n→ RUNBOOK_B §備案 A")
        return 3

    ngl = 999 if args.ngl in ("auto", None) else int(args.ngl)
    t0 = time.perf_counter()
    try:
        llm = Llama(
            model_path=first_gguf(ecfg.gguf_dir),
            n_ctx=ecfg.max_ctx,
            n_batch=ecfg.n_batch,
            n_gpu_layers=ngl,
            verbose=False,
        )
    except Exception as e:
        msg = str(e)
        print(f"FAIL: GGUF 載入失敗：{msg}")
        if "out of memory" in msg.lower() or "oom" in msg.lower():
            print("→ OOM：調小 config.yaml engine.ngl 重跑（RUNBOOK_B §備案 B）")
        return 4

    n_vocab = llm.n_vocab()  # 佔位：llama-cpp-python 不一定暴露 n_params；B 可換 model.size() 等
    t_load = time.perf_counter() - t0
    print(f"loaded in {t_load:.1f}s (ngl={ngl})  n_vocab={n_vocab}")

    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        temperature=0.0,
        stream=True,
    )
    first = None
    for chunk in out:
        choice = (chunk.get("choices") or [{}])[0]
        tok = (choice.get("delta") or {}).get("content")
        if tok:
            if first is None:
                first = tok
                print(f"first_token={tok!r}")
    if first is None:
        print("FAIL: 0 tokens（模型未出 token）")
        return 5
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
