"""Constrained, optional DeepSeek access for Lab 3 generation."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


_DEFAULT_MODEL = "deepseek-v4-pro"
_DEFAULT_BASE_URL = "https://api.deepseek.com"
_TAXONOMY_PARTS = (
    "王楚钦舆情分析_Lab2",
    "03_说明与配置",
    "sentiment_taxonomy.json",
)
_API_KEY_LINE = re.compile(r"^\s*deepseek-api\s*:\s*(.*?)\s*$", re.IGNORECASE)
_JSON_FENCE = re.compile(
    r"\A\s*```json\s*(.*?)\s*```\s*\Z",
    re.IGNORECASE | re.DOTALL,
)
_SYSTEM_PROMPT = """你是证据约束型舆情决策助手。
仅使用用户消息所给的 JSON 证据，不得补充外部事实或常识性推断。
不得更改、重算、补写或夸大任何数字；不得把相关或先后关系写成因果。
不得把案例样本外推为微博总体舆情，必须保留证据边界与局限。
输出只能是一个 JSON 对象，不得附带解释或代码块。
结果仅供辅助判断，最终结论与行动必须由人工复核决定。"""


@dataclass(frozen=True)
class LLMCallResult:
    """A safe immutable result returned by :class:`DeepSeekClient`."""

    ok: bool
    payload: Mapping[str, Any] | None
    reason: str | None

    def __post_init__(self) -> None:
        if self.payload is not None:
            if not isinstance(self.payload, Mapping):
                raise TypeError("payload must be a mapping or None")
            object.__setattr__(self, "payload", _freeze(self.payload))


def resolve_api_key(
    repo_root: str | os.PathLike[str],
    explicit: str | None = None,
) -> str | None:
    """Resolve a key without raising, logging, or exposing its value."""

    explicit_key = _nonempty(explicit)
    if explicit_key is not None:
        return explicit_key

    environment_key = _nonempty(os.environ.get("DEEPSEEK_API_KEY"))
    if environment_key is not None:
        return environment_key

    try:
        lines = (Path(repo_root) / "api.txt").read_text(
            encoding="utf-8"
        ).splitlines()
    except (OSError, UnicodeError, TypeError, ValueError):
        return None

    for line in lines:
        match = _API_KEY_LINE.match(line)
        if match is None:
            continue
        file_key = _nonempty(match.group(1))
        if file_key is not None:
            return file_key
    return None


class DeepSeekClient:
    """Small OpenAI-compatible client with strict JSON validation."""

    def __init__(
        self,
        repo_root: str | os.PathLike[str],
        api_key: str | None = None,
        sdk_client: Any | None = None,
        timeout: float = 20,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.api_key = resolve_api_key(self.repo_root, api_key)
        self.model_name, self.base_url = _resolve_model_config(self.repo_root)
        self.timeout = timeout
        self._sdk_client = sdk_client
        self._sdk_unavailable = False

    def generate(
        self,
        task: str,
        evidence: Mapping[str, Any],
        validator: Callable[[Mapping[str, Any]], bool],
    ) -> LLMCallResult:
        """Generate and validate one JSON object, with at most two calls."""

        if self.api_key is None:
            return LLMCallResult(
                ok=False,
                payload=None,
                reason="未找到 DeepSeek API key，在线生成功能不可用。",
            )

        client = self._get_sdk_client()
        if client is None:
            return LLMCallResult(
                ok=False,
                payload=None,
                reason="OpenAI SDK 不可用，无法调用在线模型。",
            )

        try:
            user_content = json.dumps(
                evidence,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError, OverflowError):
            return LLMCallResult(
                ok=False,
                payload=None,
                reason="证据无法安全序列化为 JSON，在线生成已停止。",
            )

        messages = (
            {
                "role": "system",
                "content": f"{_SYSTEM_PROMPT}\n具体任务：{task}",
            },
            {"role": "user", "content": user_content},
        )
        last_failure = "JSON"
        for _attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                    stream=False,
                    timeout=self.timeout,
                )
            except Exception:
                return LLMCallResult(
                    ok=False,
                    payload=None,
                    reason="在线模型调用发生异常；敏感错误详情已隐藏。",
                )

            payload = _parse_json_object(_response_content(response))
            if payload is None:
                last_failure = "JSON"
                continue

            try:
                is_valid = validator(payload) is True
            except Exception:
                is_valid = False
            if is_valid:
                return LLMCallResult(ok=True, payload=payload, reason=None)
            last_failure = "validation"

        if last_failure == "validation":
            reason = "模型输出连续两次未通过安全校验。"
        else:
            reason = "模型输出连续两次不是有效 JSON 对象。"
        return LLMCallResult(ok=False, payload=None, reason=reason)

    def _get_sdk_client(self) -> Any | None:
        if self._sdk_client is not None:
            return self._sdk_client
        if self._sdk_unavailable:
            return None

        try:
            from openai import OpenAI

            self._sdk_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        except Exception:
            self._sdk_unavailable = True
            return None
        return self._sdk_client


def _resolve_model_config(repo_root: Path) -> tuple[str, str]:
    model_name = _DEFAULT_MODEL
    base_url = _DEFAULT_BASE_URL
    taxonomy_path = repo_root.joinpath(*_TAXONOMY_PARTS)
    try:
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        model = taxonomy.get("model")
        if isinstance(model, Mapping):
            model_name = _nonempty(model.get("name")) or model_name
            base_url = _nonempty(model.get("base_url")) or base_url
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        pass

    model_name = _nonempty(os.environ.get("DEEPSEEK_MODEL")) or model_name
    base_url = _nonempty(os.environ.get("DEEPSEEK_BASE_URL")) or base_url
    return model_name, base_url


def _parse_json_object(content: str | None) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    candidate = content.strip()
    fence = _JSON_FENCE.fullmatch(candidate)
    if fence is not None:
        candidate = fence.group(1).strip()
    try:
        payload = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _response_content(response: Any) -> str | None:
    try:
        content = response.choices[0].message.content
    except Exception:
        return None
    return content if isinstance(content, str) else None


def _nonempty(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value
