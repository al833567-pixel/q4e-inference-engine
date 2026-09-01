# -*- coding: utf-8 -*-
"""`/v1` API server（A-1）：OpenAI 相容子集 + API key + 串流/非串流。

端點：
  GET  /health                免驗 key
  GET  /v1/models             列模型
  POST /v1/chat/completions   stream=true → SSE；false → JSON
  POST /v1/completions        純文字補全

引擎（BaseEngine）：
  - StubEngine  預設，**不載模型、不需 GPU**（A 的 curl/測試用；回應確定性、可重放）。
  - LlamaEngine llama-cpp-python 適配器（Track B 用）；**lazy load**：第一次請求才載 GGUF。
    B 設 `config.yaml server.engine: llama` 或 env `DSH_ENGINE=llama`。

啟動：
  py -3.14 -m uvicorn server:app --host 127.0.0.1 --port 8080
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Iterator, Optional

from fastapi import Depends, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import auth
import config as cfgmod
import interfaces

# --------------------------------------------------------------------------- #
# stub tokenizer（確定性，僅供 stub 引擎與測試數 token）
# --------------------------------------------------------------------------- #
def stub_tokenize(text: str) -> list[str]:
    """CJK 1 字＝1 token；連續 ASCII 字母數字＝1 token；其他符號分隔。"""
    toks: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ord(ch) > 0x2E80:  # CJK / 全形等
            if buf:
                toks.append("".join(buf))
                buf = []
            toks.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                toks.append("".join(buf))
                buf = []
    if buf:
        toks.append("".join(buf))
    return toks


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
class BaseEngine:
    """server 需要的最小引擎介面（比 interfaces.InferenceEngine 更薄）。"""

    kind = "base"
    model_name = "qwen3.8-flash-next"

    def is_loaded(self) -> bool:
        return False

    def stream_chat(self, messages: list[dict], max_tokens: int, temperature: float) -> Iterator[str]:
        raise NotImplementedError

    def stream_complete(self, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
        raise NotImplementedError


class StubEngine(BaseEngine):
    """確定性 stub：不載模型。回應 = 固定模板 + 回顯最後一則 user 訊息。"""

    kind = "stub"

    def __init__(self, model_name: str = "qwen3.8-flash-next") -> None:
        self.model_name = model_name

    def is_loaded(self) -> bool:
        return True

    def _chat_reply(self, messages: list[dict]) -> str:
        last = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last = m.get("content") or ""
                break
        return f"（stub）已收到你的訊息（{len(last)} 字元）：「{last[:64]}」"

    def _complete_reply(self, prompt: str) -> str:
        return f"（stub）補全提示（{len(prompt)} 字元）：「{prompt[:64]}」"

    def stream_chat(self, messages, max_tokens, temperature) -> Iterator[str]:
        text = self._chat_reply(messages)
        toks = stub_tokenize(text)
        if max_tokens:
            toks = toks[:max_tokens]
        for i in range(0, len(toks), 2):
            yield "".join(toks[i : i + 2])

    def stream_complete(self, prompt, max_tokens, temperature) -> Iterator[str]:
        text = self._complete_reply(prompt)
        toks = stub_tokenize(text)
        if max_tokens:
            toks = toks[:max_tokens]
        for i in range(0, len(toks), 2):
            yield "".join(toks[i : i + 2])


def _resolve_ngl(ngl: int | str) -> int:
    """`auto` → 999（llama.cpp 慣例：能放幾層放幾層）；B 建議改具體數字。"""
    if isinstance(ngl, int):
        return ngl
    return 999


class LlamaEngine(BaseEngine):
    """llama-cpp-python 適配器（Track B）。lazy load：首次請求才載 GGUF（thread-safe）。

    B 可直接用；若要 router_experts/快取接線，改實作 interfaces.InferenceEngine
    並在 tools/wire_cache.py 的 RealEngine 掛上（見 RUNBOOK_B §4）。
    """

    kind = "llama"

    def __init__(
        self,
        cfg: interfaces.EngineConfig,
        model_name: str = "qwen3.8-flash-next",
        lib_path: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.model_name = model_name
        self.lib_path = lib_path or r"D:\llama"
        self._llm = None
        self._lock = threading.Lock()
        self.last_load_error: str | None = None

    def is_loaded(self) -> bool:
        return self._llm is not None

    def _first_shard(self) -> str:
        """llama-cpp-python 的 Llama() 不解析目錄 → 傳第一片 .gguf，其餘分片由 llama.cpp 自發現。"""
        d = self.cfg.gguf_dir
        if os.path.isfile(d):
            return d
        cands = sorted(f for f in os.listdir(d) if f.endswith(".gguf"))
        if not cands:
            raise FileNotFoundError(f"{d} 內沒有 .gguf")
        return os.path.join(d, cands[0])

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        with self._lock:
            if self._llm is not None:
                return
            # B：import 前設定 backend（wheel 自帶 llama.cpp 載不了 qwen4exp；
            # fork D:\llama\llama.dll + fork_compat 72B→80B 相容層，根因見 fork_compat.py）
            os.environ.setdefault("LLAMA_CPP_LIB_PATH", self.lib_path)
            os.chdir(self.lib_path)  # backend 預設掃描 CWD（CUDA/RPC/haswell）
            import fork_compat

            fork_compat.install()
            from llama_cpp import Llama  # 0.3.x 高層類（小寫 llama 是 C 模組，不能當類用）

            try:
                kwargs = dict(
                    model_path=self._first_shard(),
                    n_ctx=self.cfg.max_ctx,
                    n_batch=self.cfg.n_batch,
                    n_gpu_layers=_resolve_ngl(self.cfg.ngl),
                    verbose=False,
                )
                if self.cfg.n_threads:
                    kwargs["n_threads"] = self.cfg.n_threads
                self._llm = Llama(**kwargs)
            except Exception as e:  # OOM / 找不到檔案 / 編譯缺
                self.last_load_error = str(e)
                raise RuntimeError(
                    f"GGUF 載入失敗：{e} → 見 RUNBOOK_B.md §備案 B（OOM 調小 ngl）"
                ) from e

    def stream_chat(self, messages, max_tokens, temperature) -> Iterator[str]:
        self._ensure_loaded()
        for chunk in self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or 256,
            temperature=temperature,
            stream=True,
        ):
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                yield delta["content"]

    def stream_complete(self, prompt, max_tokens, temperature) -> Iterator[str]:
        self._ensure_loaded()
        for chunk in self._llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens or 256,
            temperature=temperature,
            stream=True,
        ):
            choice = (chunk.get("choices") or [{}])[0]
            if choice.get("text"):
                yield choice["text"]


# --------------------------------------------------------------------------- #
# 請求模型
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = Field(default=256, ge=1)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    stop: Optional[object] = None


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = Field(default=256, ge=1)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    stop: Optional[object] = None


# --------------------------------------------------------------------------- #
# 回應組裝
# --------------------------------------------------------------------------- #
def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def _chat_response(model: str, text: str, prompt_tokens: int, completion_tokens: int, finish: str) -> dict:
    return {
        "id": _new_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _completion_response(model: str, text: str, prompt_tokens: int, completion_tokens: int, finish: str) -> dict:
    return {
        "id": _new_id("cmpl"),
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_chat(engine: BaseEngine, messages: list[dict], model: str, max_tokens: int, temperature: float):
    cid = _new_id("chatcmpl")
    created = int(time.time())

    def gen():
        parts: list[str] = []
        n_tok = 0
        for piece in engine.stream_chat(messages, max_tokens, temperature):
            parts.append(piece)
            n_tok += len(stub_tokenize(piece))
            yield _sse(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }
            )
        text = "".join(parts)
        prompt_tokens = sum(len(stub_tokenize((m.get("content") or ""))) for m in messages)
        yield _sse(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": n_tok,
                    "total_tokens": prompt_tokens + n_tok,
                },
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse_complete(engine: BaseEngine, prompt: str, model: str, max_tokens: int, temperature: float):
    cid = _new_id("cmpl")
    created = int(time.time())
    prompt_tokens = len(stub_tokenize(prompt))

    def gen():
        parts: list[str] = []
        n_tok = 0
        for piece in engine.stream_complete(prompt, max_tokens, temperature):
            parts.append(piece)
            n_tok += len(stub_tokenize(piece))
            yield _sse(
                {
                    "id": cid,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": piece, "finish_reason": None}],
                }
            )
        yield _sse(
            {
                "id": cid,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": n_tok,
                    "total_tokens": prompt_tokens + n_tok,
                },
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def _default_engine(scfg: interfaces.ServerConfig) -> BaseEngine:
    mode = (os.environ.get("DSH_ENGINE") or scfg.engine or "stub").lower()
    if mode == "llama":
        raw = cfgmod.load_config()
        lib_path = str((raw.get("engine") or {}).get("lib_path") or r"D:\llama")
        return LlamaEngine(cfgmod.engine_config(raw), model_name=scfg.model_name, lib_path=lib_path)
    return StubEngine(model_name=scfg.model_name)


def create_app(
    engine: BaseEngine | None = None,
    raw_cfg: dict | None = None,
    cfg_path: str | None = None,
) -> FastAPI:
    raw = raw_cfg if raw_cfg is not None else cfgmod.load_config(cfg_path)
    scfg = cfgmod.server_config(raw)
    if engine is None:
        engine = _default_engine(scfg)
    valid_keys = auth.collect_keys(raw.get("server"))

    app = FastAPI(title="推理工具 /v1", version="0.1a")
    app.state.engine = engine
    app.state.raw_cfg = raw
    app.state.valid_keys = valid_keys
    dep = Depends(auth.make_auth_dependency(valid_keys))

    @app.exception_handler(auth.HTTPError)
    async def _http_error_handler(request, exc: auth.HTTPError):
        return auth.error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request, exc):  # OpenAI 風格 400
        msgs = "; ".join(
            f"{'.'.join(map(str, e.get('loc', [])))}: {e.get('msg')}" for e in exc.errors()[:3]
        )
        return auth.error_response(400, "invalid_request_error", f"invalid request: {msgs}")

    @app.get("/health")
    def health():
        eng = app.state.engine
        return {
            "status": "ok",
            "engine": {
                "kind": eng.kind,
                "loaded": eng.is_loaded(),
                "model": eng.model_name,
            },
        }

    @app.get("/v1/models", dependencies=[dep])
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": app.state.engine.model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[dep])
    def chat_completions(req: ChatRequest):
        if not req.messages:
            auth.raise_error(400, "invalid_request_error", "messages must be a non-empty array")
        eng: BaseEngine = app.state.engine
        model = req.model or eng.model_name
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        if req.stream:
            return _sse_chat(eng, messages, model, req.max_tokens or 256, req.temperature if req.temperature is not None else 0.7)
        parts = list(eng.stream_chat(messages, req.max_tokens or 256, req.temperature if req.temperature is not None else 0.7))
        text = "".join(parts)
        prompt_tokens = sum(len(stub_tokenize((m.get("content") or ""))) for m in messages)
        completion_tokens = len(stub_tokenize(text))
        return _chat_response(model, text, prompt_tokens, completion_tokens, "stop")

    @app.post("/v1/completions", dependencies=[dep])
    def completions(req: CompletionRequest):
        eng: BaseEngine = app.state.engine
        model = req.model or eng.model_name
        if req.stream:
            return _sse_complete(eng, req.prompt, model, req.max_tokens or 256, req.temperature if req.temperature is not None else 0.7)
        text = "".join(eng.stream_complete(req.prompt, req.max_tokens or 256, req.temperature if req.temperature is not None else 0.7))
        return _completion_response(
            model, text, len(stub_tokenize(req.prompt)), len(stub_tokenize(text)), "stop"
        )

    return app


# `uvicorn server:app` 入口（預設 stub，不需 GPU）
app = create_app()
