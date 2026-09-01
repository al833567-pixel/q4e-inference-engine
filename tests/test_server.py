# -*- coding: utf-8 -*-
import json

import pytest
from fastapi.testclient import TestClient

from server import StubEngine, create_app, stub_tokenize

KEY = "sk-test"
CFG = {"server": {"api_keys": [KEY], "model_name": "m1"}}


@pytest.fixture()
def client():
    app = create_app(engine=StubEngine("m1"), raw_cfg=CFG)
    return TestClient(app)


def test_stub_tokenize_deterministic():
    toks = stub_tokenize("你好 world 123")
    assert toks == ["你", "好", "world", "123"]


def test_health_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "ok"
    assert b["engine"]["kind"] == "stub"
    assert b["engine"]["loaded"] is True


def test_models(client):
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "m1"


def test_chat_no_key_401(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_chat_wrong_key_401(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer sk-wrong"},
    )
    assert r.status_code == 401


def test_chat_nonstream_shape(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "你好"}], "max_tokens": 64},
        headers={"Authorization": f"Bearer {KEY}"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["id"].startswith("chatcmpl-")
    assert b["object"] == "chat.completion"
    assert b["model"] == "m1"
    ch = b["choices"][0]
    assert ch["message"]["role"] == "assistant"
    assert "你好" in ch["message"]["content"]  # stub 回顯
    assert ch["finish_reason"] == "stop"
    u = b["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    assert u["prompt_tokens"] >= 1 and u["completion_tokens"] >= 1


def _parse_sse(text: str):
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                events.append("[DONE]")
            else:
                events.append(json.loads(payload))
    return events


def test_chat_stream_sse(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "你好"}], "stream": True},
        headers={"Authorization": f"Bearer {KEY}"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    assert events[-1] == "[DONE]"
    deltas = [e["choices"][0]["delta"].get("content", "") for e in events if e != "[DONE]" and e["choices"][0]["delta"].get("content")]
    content = "".join(deltas)
    # 串流拼接 == 非串流內容
    r2 = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "你好"}], "stream": False},
        headers={"Authorization": f"Bearer {KEY}"},
    )
    assert content == r2.json()["choices"][0]["message"]["content"]
    # 最後一個 chunk finish_reason=stop
    last_chunk = events[-2]
    assert last_chunk["choices"][0]["finish_reason"] == "stop"


def test_completions_nonstream(client):
    r = client.post(
        "/v1/completions",
        json={"prompt": "hello", "max_tokens": 32},
        headers={"Authorization": f"Bearer {KEY}"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["object"] == "text_completion"
    assert "hello" in b["choices"][0]["text"]
    assert b["usage"]["prompt_tokens"] >= 1


def test_empty_messages_400(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers={"Authorization": f"Bearer {KEY}"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request_error"


def test_no_keys_configured_rejects(monkeypatch):
    monkeypatch.delenv("DSH_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = create_app(engine=StubEngine("m1"), raw_cfg={"server": {}})
    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}, headers={"Authorization": "Bearer x"})
    assert r.status_code == 401
    assert "no API keys" in r.json()["error"]["message"]
    # /health 仍免驗
    assert c.get("/health").status_code == 200
