# -*- coding: utf-8 -*-
"""config.yaml 載入器（A-4）。

把 YAML 映射到 interfaces.py 的資料類別。所有模組（server/cache/ngram/tools）
都經這裡讀設定，B 不需要自己 parse YAML。
"""
from __future__ import annotations

import os

import yaml

import interfaces

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_config(path: str | None) -> str | None:
    if path:
        return path
    p = os.path.join(_HERE, "config.yaml")
    return p if os.path.exists(p) else None


def load_config(path: str | None = None) -> dict:
    """讀 config.yaml；找不到回 {}（用 defaults）。"""
    p = _find_config(path)
    if p is None:
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def server_config(raw: dict) -> interfaces.ServerConfig:
    d = raw.get("server") or {}
    return interfaces.ServerConfig(
        host=d.get("host", "127.0.0.1"),
        port=int(d.get("port", 8080)),
        api_keys=[k for k in (d.get("api_keys") or []) if isinstance(k, str)],
        model_name=d.get("model_name", "qwen3.8-flash-next"),
        stream_default=bool(d.get("stream_default", False)),
        engine=d.get("engine", "stub"),
    )


def engine_config(raw: dict) -> interfaces.EngineConfig:
    d = raw.get("engine") or {}
    e = interfaces.EngineConfig()
    return interfaces.EngineConfig(
        gguf_dir=d.get("gguf_dir", e.gguf_dir),
        ngl=d.get("ngl", "auto"),
        max_ctx=int(d.get("max_ctx", e.max_ctx)),
        n_batch=int(d.get("n_batch", e.n_batch)),
        n_threads=int(d.get("n_threads", e.n_threads)),
    )


def cache_config(raw: dict) -> interfaces.CacheConfig:
    d = raw.get("cache") or {}
    c = interfaces.CacheConfig()
    return interfaces.CacheConfig(
        n_layers=int(d.get("n_layers", c.n_layers)),
        n_experts_per_layer=int(d.get("n_experts_per_layer", c.n_experts_per_layer)),
        active_routed=int(d.get("active_routed", c.active_routed)),
        has_shared=bool(d.get("has_shared", c.has_shared)),
        vram_capacity_per_layer=int(d.get("vram_capacity_per_layer", c.vram_capacity_per_layer)),
        ram_capacity_per_layer=int(d.get("ram_capacity_per_layer", c.ram_capacity_per_layer)),
        ram_mult=int(d.get("ram_mult", c.ram_mult)),
        ssd_path=d.get("ssd_path", c.ssd_path),
        ng_on_ssd=bool(d.get("ng_on_ssd", c.ng_on_ssd)),
        prefetch_into=d.get("prefetch_into", c.prefetch_into),
    )


def ngram_config(raw: dict) -> dict:
    return raw.get("ngram") or {}
