# -*- coding: utf-8 -*-
import pytest

import auth


def test_valid_key():
    auth.check_auth("Bearer sk-abc", ["sk-abc", "sk-def"])


def test_case_insensitive_scheme():
    auth.check_auth("bearer sk-abc", ["sk-abc"])


def test_missing_header():
    with pytest.raises(auth.ApiKeyError):
        auth.check_auth(None, ["sk-abc"])
    with pytest.raises(auth.ApiKeyError):
        auth.check_auth("", ["sk-abc"])
    with pytest.raises(auth.ApiKeyError):
        auth.check_auth("Token sk-abc", ["sk-abc"])


def test_empty_token():
    with pytest.raises(auth.ApiKeyError):
        auth.check_auth("Bearer   ", ["sk-abc"])


def test_wrong_key():
    with pytest.raises(auth.ApiKeyError, match="invalid API key"):
        auth.check_auth("Bearer sk-wrong", ["sk-abc"])


def test_no_keys_configured_fails_closed():
    with pytest.raises(auth.ApiKeyError, match="no API keys configured"):
        auth.check_auth("Bearer anything", [])


def test_collect_keys_config_and_env(monkeypatch):
    monkeypatch.setenv("DSH_API_KEY", "env1 env2,env3")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    keys = auth.collect_keys({"api_keys": ["cfg1", "env1"]})
    assert keys == ["cfg1", "env1", "env2", "env3"]  # config 優先、去重、保序


def test_collect_keys_empty(monkeypatch):
    monkeypatch.delenv("DSH_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert auth.collect_keys({"api_keys": []}) == []
