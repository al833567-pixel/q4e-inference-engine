# -*- coding: utf-8 -*-
"""API key 驗證（A-1）。

- key 來源（合併、去重、保序）：
  1. `config.yaml` 的 `server.api_keys`（list[str]）
  2. 環境變數 `DSH_API_KEY` / `OPENAI_API_KEY`（空白或逗號分隔，可多 key）
- 驗證 `Authorization: Bearer <key>`；**常數時間比較**（hmac.compare_digest，防 timing 側洩漏）。
- 未設定任何 key → 拒絕所有 /v1 請求（fail closed，401 附說明）。
- `/health` 免驗（server.py 不掛 dependency）。

錯誤格式走 OpenAI 風格：
  {"error": {"message": "...", "type": "invalid_request_error", "code": "invalid_api_key", "status": 401}}
"""
from __future__ import annotations

import hmac
import os

from fastapi import Request

ENV_KEYS = ("DSH_API_KEY", "OPENAI_API_KEY")


class ApiKeyError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def keys_from_env() -> list[str]:
    out: list[str] = []
    for name in ENV_KEYS:
        raw = os.environ.get(name, "")
        out.extend(p for p in raw.replace(",", " ").split() if p)
    return out


def collect_keys(cfg_server: dict | None = None) -> list[str]:
    """config + env 合併去重（config 優先）。"""
    keys: list[str] = []
    if cfg_server:
        keys.extend(k for k in cfg_server.get("api_keys", []) if isinstance(k, str) and k)
    keys.extend(keys_from_env())
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def check_auth(authorization: str | None, valid_keys: list[str]) -> None:
    """驗證 Authorization header；失敗丟 ApiKeyError。"""
    if not valid_keys:
        raise ApiKeyError(
            "server has no API keys configured "
            "(set config.yaml server.api_keys or env DSH_API_KEY)"
        )
    if not authorization or not authorization.strip().lower().startswith("bearer "):
        raise ApiKeyError('missing "Authorization: Bearer <key>" header')
    token = authorization.strip()[7:].strip()
    if not token:
        raise ApiKeyError("empty API key")
    for k in valid_keys:
        if hmac.compare_digest(token.encode("utf-8"), k.encode("utf-8")):
            return
    raise ApiKeyError("invalid API key")


class HTTPError(Exception):
    """可 raise 的 OpenAI 風格錯誤（server.py 註冊 handler 轉 JSONResponse）。"""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def raise_error(status: int, code: str, message: str) -> None:
    raise HTTPError(status, code, message)


def error_response(status: int, code: str, message: str):
    """OpenAI 風格錯誤 JSONResponse（handler / 直接回傳用）。"""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code,
                "status": status,
            }
        },
    )


def make_auth_dependency(valid_keys: list[str]):
    """回 FastAPI dependency：通過→None；失敗→丟 401 JSONResponse。"""
    async def dep(request: Request) -> None:
        try:
            check_auth(request.headers.get("authorization"), valid_keys)
        except ApiKeyError as e:
            raise HTTPError(401, "invalid_api_key", e.message) from None

    return dep
