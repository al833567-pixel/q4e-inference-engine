# -*- coding: utf-8 -*-
"""`/v1` API 合約參考實作（v2：合約 oracle）。

> v2 定位：生產引擎為 C++ `q4e-server`（engine/ 目錄，P0–P2）。
> 本檔＋tests/test_server.py＝`/v1` API 的**合約 oracle**（200/401/400 形狀、
> SSE 格式、[DONE]、usage 字段）— q4e-server 驗收對照此。
> v1 wheel 路線（llama-cpp-python）已退役 → 見 legacy_v1/。

端點：
  GET  /health                免驗 key
  GET  /v1/models             列模型
  POST /v1/chat/completions   stream=true → SSE；false → JSON
  POST /v1/completions        純文字補全

引擎：
  - StubEngine  **不載模型、不需 GPU**（確定性、可重放；合約測試用）。

啟動：
  py -3.14 -m uvicorn server:app --host 127.0.0.1 --port 8080
"""
from __future__ import annotations

import json
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
    """v2：只有合約 stub；生產引擎＝q4e-server（C++，engine/ 目錄）。"""
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
