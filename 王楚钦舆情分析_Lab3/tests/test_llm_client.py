from __future__ import annotations

import importlib
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest


def _llm_client():
    return importlib.import_module("lab3.llm_client")


class FakeCompletions:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response),
                )
            ]
        )


class FakeSDKClient:
    def __init__(self, responses: list[Any]) -> None:
        self.chat = SimpleNamespace(
            completions=FakeCompletions(responses),
        )


def test_resolve_api_key_prefers_explicit_then_environment_then_api_txt(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _llm_client()
    (tmp_path / "api.txt").write_text(
        "note: ignored\ndeepseek-api: file-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", " env-key ")

    assert module.resolve_api_key(tmp_path, " explicit-key ") == "explicit-key"
    assert module.resolve_api_key(tmp_path) == "env-key"

    monkeypatch.delenv("DEEPSEEK_API_KEY")
    assert module.resolve_api_key(tmp_path) == "file-key"


def test_resolve_api_key_returns_none_for_missing_or_empty_sources(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _llm_client()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (tmp_path / "api.txt").write_text(
        "deepseek-api:   \nother: value\n",
        encoding="utf-8",
    )

    assert module.resolve_api_key(tmp_path) is None


def test_client_reads_lab2_model_config_and_allows_environment_override(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _llm_client()
    taxonomy = (
        tmp_path
        / "王楚钦舆情分析_Lab2"
        / "03_说明与配置"
        / "sentiment_taxonomy.json"
    )
    taxonomy.parent.mkdir(parents=True)
    taxonomy.write_text(
        json.dumps(
            {
                "model": {
                    "name": "taxonomy-model",
                    "base_url": "https://taxonomy.invalid",
                }
            }
        ),
        encoding="utf-8",
    )
    fake = FakeSDKClient(['{"ok": true}'])

    configured = module.DeepSeekClient(
        tmp_path,
        api_key="test-key",
        sdk_client=fake,
    )
    assert configured.model_name == "taxonomy-model"
    assert configured.base_url == "https://taxonomy.invalid"

    monkeypatch.setenv("DEEPSEEK_MODEL", "env-model")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.invalid")
    overridden = module.DeepSeekClient(
        tmp_path,
        api_key="test-key",
        sdk_client=fake,
    )
    assert overridden.model_name == "env-model"
    assert overridden.base_url == "https://env.invalid"


def test_client_uses_safe_defaults_when_taxonomy_is_unavailable(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _llm_client()
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    client = module.DeepSeekClient(
        tmp_path,
        api_key="test-key",
        sdk_client=FakeSDKClient(['{"ok": true}']),
    )

    assert client.model_name == "deepseek-v4-pro"
    assert client.base_url == "https://api.deepseek.com"


def test_valid_json_succeeds_once_with_constrained_request_and_no_key(
    tmp_path: Any,
) -> None:
    module = _llm_client()
    secret = "sk-secret-never-send"
    fake = FakeSDKClient(
        ['{"title": "简报", "citation_ids": ["post-1"]}']
    )
    client = module.DeepSeekClient(
        tmp_path,
        api_key=secret,
        sdk_client=fake,
        timeout=7,
    )
    evidence = {
        "facts": ["评论样本共 12 条。"],
        "citation_ids": ["post-1"],
    }

    result = client.generate(
        "生成简报 JSON",
        evidence,
        lambda payload: payload.get("title") == "简报",
    )

    assert result.ok is True
    assert result.reason is None
    assert result.payload["title"] == "简报"
    assert len(fake.chat.completions.calls) == 1
    request = fake.chat.completions.calls[0]
    assert request["model"] == "deepseek-v4-pro"
    assert request["response_format"] == {"type": "json_object"}
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert request["stream"] is False
    assert request["timeout"] == 7
    assert json.loads(request["messages"][1]["content"]) == evidence
    system_prompt = request["messages"][0]["content"]
    assert all(
        phrase in system_prompt
        for phrase in ("仅使用", "数字", "因果", "总体", "人工")
    )
    assert secret not in json.dumps(request, ensure_ascii=False)
    assert "api_key" not in request


def test_json_code_fence_is_accepted(tmp_path: Any) -> None:
    module = _llm_client()
    fake = FakeSDKClient(
        ['```json\n{"answerable": false}\n```']
    )
    client = module.DeepSeekClient(
        tmp_path,
        api_key="test-key",
        sdk_client=fake,
    )

    result = client.generate(
        "回答问题",
        {"facts": []},
        lambda payload: payload.get("answerable") is False,
    )

    assert result.ok is True
    assert result.payload["answerable"] is False


def test_invalid_json_is_retried_once_then_fails(tmp_path: Any) -> None:
    module = _llm_client()
    fake = FakeSDKClient(["not json", "still not json"])
    client = module.DeepSeekClient(
        tmp_path,
        api_key="test-key",
        sdk_client=fake,
    )

    result = client.generate("生成", {"facts": []}, lambda payload: True)

    assert result.ok is False
    assert result.payload is None
    assert "JSON" in result.reason
    assert len(fake.chat.completions.calls) == 2


def test_validator_failure_is_retried_once_then_fails(tmp_path: Any) -> None:
    module = _llm_client()
    fake = FakeSDKClient(['{"value": 1}', '{"value": 2}'])
    client = module.DeepSeekClient(
        tmp_path,
        api_key="test-key",
        sdk_client=fake,
    )

    result = client.generate(
        "生成",
        {"facts": []},
        lambda payload: payload.get("value") == 3,
    )

    assert result.ok is False
    assert result.payload is None
    assert "校验" in result.reason
    assert len(fake.chat.completions.calls) == 2


def test_exception_reason_does_not_leak_key_payload_or_exception_text(
    tmp_path: Any,
) -> None:
    module = _llm_client()
    secret = "sk-secret-value"
    full_payload = '{"private_evidence":"完整敏感文本"}'
    fake = FakeSDKClient(
        [RuntimeError(f"provider failure {secret} {full_payload}")]
    )
    client = module.DeepSeekClient(
        tmp_path,
        api_key=secret,
        sdk_client=fake,
    )

    result = client.generate(
        "生成",
        {"private_evidence": "完整敏感文本"},
        lambda payload: True,
    )

    assert result.ok is False
    assert result.payload is None
    assert result.reason
    assert secret not in result.reason
    assert "完整敏感文本" not in result.reason
    assert "provider failure" not in result.reason


def test_missing_key_returns_unavailable_without_touching_sdk(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _llm_client()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    fake = FakeSDKClient(['{"should_not": "run"}'])
    client = module.DeepSeekClient(tmp_path, sdk_client=fake)

    result = client.generate("生成", {"facts": []}, lambda payload: True)

    assert result.ok is False
    assert "API key" in result.reason
    assert fake.chat.completions.calls == []


def test_llm_call_result_is_frozen_and_defensively_copies_payload() -> None:
    module = _llm_client()
    payload = {"nested": {"values": [1, 2]}}
    result = module.LLMCallResult(ok=True, payload=payload, reason=None)

    payload["nested"]["values"].append(3)
    payload["nested"]["extra"] = "mutated"

    assert result.payload["nested"]["values"] == (1, 2)
    assert "extra" not in result.payload["nested"]
    with pytest.raises(TypeError):
        result.payload["nested"]["new"] = "blocked"
    with pytest.raises(FrozenInstanceError):
        result.ok = False
