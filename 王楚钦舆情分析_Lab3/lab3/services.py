"""Online-first Lab 3 services with deterministic offline degradation."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .llm_client import DeepSeekClient
from .models import EvidencePacket, GeneratedResult
from .offline import (
    PRESET_QUESTIONS,
    answer_offline,
    brief_offline,
    strategies_offline,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRIEF_TASK = (
    "生成舆情简报 JSON，必须包含 title、facts、observations、"
    "decision_focus、limitations、citation_ids。observations 与 "
    "decision_focus 必须是非空字符串数组；citation_ids 只能取自证据。"
)
_QA_TASK = (
    "回答证据内问题并输出 JSON，必须包含 question、answerable、facts、"
    "interpretation、limitations、citation_ids。facts 只能逐字选择证据包"
    "中的 facts，citation_ids 只能取自证据；证据不足时明确不可回答。"
)
_STRATEGY_TASK = (
    "生成定性策略比较 JSON，必须包含 goal、audience、恰好三个 options 和"
    " disclaimer。每个 option 必须包含 name、action、timing、benefits、"
    "risks、checks、evidence_ids；引用只能取自证据。disclaimer 必须明确"
    "非预测且最终由人工决定。"
)


class BriefService:
    def __init__(
        self,
        client: Any | None = None,
        *,
        repo_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.client = _coerce_client(client, repo_root)

    def generate(self, packet: EvidencePacket) -> GeneratedResult:
        offline = brief_offline(packet)
        try:
            evidence = packet.as_prompt_dict()
        except Exception:
            return _degraded(offline, "证据无法安全序列化")

        allowed_ids = _packet_citation_ids(packet)
        validator = lambda payload: _validate_brief(payload, allowed_ids)
        payload, reason = _request_online(
            self.client,
            _BRIEF_TASK,
            evidence,
            validator,
        )
        if payload is None:
            return _degraded(offline, reason)

        return GeneratedResult(
            mode="online",
            payload={
                "title": payload["title"].strip(),
                "facts": packet.facts,
                "observations": tuple(payload["observations"]),
                "decision_focus": tuple(payload["decision_focus"]),
                "limitations": _merge_limitations(
                    tuple(payload["limitations"]),
                    offline.payload["limitations"],
                ),
                "citation_ids": _deduplicate(payload["citation_ids"]),
            },
        )


class QAService:
    def __init__(
        self,
        client: Any | None = None,
        *,
        repo_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.client = _coerce_client(client, repo_root)

    def answer(
        self,
        question: str,
        packet: EvidencePacket,
        preset: bool = False,
    ) -> GeneratedResult:
        offline = _offline_answer(question, packet, preset)
        try:
            evidence = {
                "question": question,
                "preset": preset,
                "packet": packet.as_prompt_dict(),
            }
        except Exception:
            return _degraded(offline, "证据无法安全序列化")

        allowed_ids = _packet_citation_ids(packet)
        allowed_facts = frozenset(packet.facts)
        validator = lambda payload: _validate_qa(
            payload,
            allowed_facts,
            allowed_ids,
        )
        payload, reason = _request_online(
            self.client,
            _QA_TASK,
            evidence,
            validator,
        )
        if payload is None:
            return _degraded(offline, reason)

        displayed_question = (
            PRESET_QUESTIONS.get(question, question) if preset else question
        )
        authoritative_limits = brief_offline(packet).payload["limitations"]
        return GeneratedResult(
            mode="online",
            payload={
                "question": displayed_question,
                "answerable": payload["answerable"],
                "facts": _deduplicate(payload["facts"]),
                "interpretation": payload["interpretation"].strip(),
                "limitations": _merge_limitations(
                    tuple(payload["limitations"]),
                    authoritative_limits,
                ),
                "citation_ids": _deduplicate(payload["citation_ids"]),
            },
        )


class StrategyService:
    def __init__(
        self,
        client: Any | None = None,
        *,
        repo_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.client = _coerce_client(client, repo_root)

    def generate(
        self,
        packet: EvidencePacket,
        goal: str,
        audience: str,
    ) -> GeneratedResult:
        offline = strategies_offline(packet, goal=goal, audience=audience)
        try:
            evidence = {
                "goal": goal,
                "audience": audience,
                "packet": packet.as_prompt_dict(),
            }
        except Exception:
            return _degraded(offline, "证据无法安全序列化")

        allowed_ids = _packet_citation_ids(packet)
        validator = lambda payload: _validate_strategy(payload, allowed_ids)
        payload, reason = _request_online(
            self.client,
            _STRATEGY_TASK,
            evidence,
            validator,
        )
        if payload is None:
            return _degraded(offline, reason)

        options = tuple(_clean_option(option) for option in payload["options"])
        return GeneratedResult(
            mode="online",
            payload={
                "goal": goal,
                "audience": audience,
                "options": options,
                "disclaimer": payload["disclaimer"].strip(),
            },
        )


def _coerce_client(
    client: Any | None,
    repo_root: str | os.PathLike[str] | None,
) -> Any:
    if client is not None and isinstance(client, (str, os.PathLike)):
        return DeepSeekClient(client)
    if client is not None:
        return client
    return DeepSeekClient(repo_root or _REPO_ROOT)


def _request_online(
    client: Any,
    task: str,
    evidence: Mapping[str, Any],
    validator: Callable[[Mapping[str, Any]], bool],
) -> tuple[Mapping[str, Any] | None, str]:
    try:
        result = client.generate(task, evidence, validator)
    except Exception:
        return None, "在线模型调用异常，敏感详情已隐藏"

    if getattr(result, "ok", False) is not True:
        return None, _safe_reason(getattr(result, "reason", None))
    payload = getattr(result, "payload", None)
    if not isinstance(payload, Mapping):
        return None, "模型输出缺少有效 JSON 对象"
    try:
        valid = validator(payload) is True
    except Exception:
        valid = False
    if not valid:
        return None, "模型输出未通过服务层安全校验"
    return payload, ""


def _validate_brief(
    payload: Mapping[str, Any],
    allowed_ids: frozenset[str],
) -> bool:
    required = {
        "title",
        "facts",
        "observations",
        "decision_focus",
        "limitations",
        "citation_ids",
    }
    return (
        required <= set(payload)
        and _is_nonempty_string(payload.get("title"))
        and _is_string_sequence(payload.get("facts"))
        and _is_string_sequence(payload.get("observations"), nonempty=True)
        and _is_string_sequence(payload.get("decision_focus"), nonempty=True)
        and _is_string_sequence(payload.get("limitations"))
        and _references_are_known(payload.get("citation_ids"), allowed_ids)
    )


def _validate_qa(
    payload: Mapping[str, Any],
    allowed_facts: frozenset[str],
    allowed_ids: frozenset[str],
) -> bool:
    required = {
        "question",
        "answerable",
        "facts",
        "interpretation",
        "limitations",
        "citation_ids",
    }
    facts = payload.get("facts")
    return (
        required <= set(payload)
        and _is_nonempty_string(payload.get("question"))
        and isinstance(payload.get("answerable"), bool)
        and _is_string_sequence(facts)
        and all(fact in allowed_facts for fact in facts)
        and _is_nonempty_string(payload.get("interpretation"))
        and _is_string_sequence(payload.get("limitations"))
        and _references_are_known(payload.get("citation_ids"), allowed_ids)
    )


def _validate_strategy(
    payload: Mapping[str, Any],
    allowed_ids: frozenset[str],
) -> bool:
    required = {"goal", "audience", "options", "disclaimer"}
    if not required <= set(payload):
        return False
    if not _is_nonempty_string(payload.get("goal")):
        return False
    if not _is_nonempty_string(payload.get("audience")):
        return False
    disclaimer = payload.get("disclaimer")
    if not (
        _is_nonempty_string(disclaimer)
        and "人工" in disclaimer
        and "非预测" in disclaimer
    ):
        return False

    options = payload.get("options")
    if not isinstance(options, (list, tuple)) or len(options) != 3:
        return False
    if not all(_validate_option(option, allowed_ids) for option in options):
        return False
    names = [option["name"].strip() for option in options]
    return len(set(names)) == 3


def _validate_option(
    option: Any,
    allowed_ids: frozenset[str],
) -> bool:
    if not isinstance(option, Mapping):
        return False
    required = {
        "name",
        "action",
        "timing",
        "benefits",
        "risks",
        "checks",
        "evidence_ids",
    }
    return (
        required <= set(option)
        and _is_nonempty_string(option.get("name"))
        and _is_nonempty_string(option.get("action"))
        and _is_nonempty_string(option.get("timing"))
        and _is_string_sequence(option.get("benefits"), nonempty=True)
        and _is_string_sequence(option.get("risks"), nonempty=True)
        and _is_string_sequence(option.get("checks"), nonempty=True)
        and _references_are_known(option.get("evidence_ids"), allowed_ids)
    )


def _clean_option(option: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = {
        "name": option["name"].strip(),
        "action": option["action"].strip(),
        "timing": option["timing"].strip(),
        "evidence_ids": _deduplicate(option["evidence_ids"]),
        "benefits": tuple(option["benefits"]),
        "risks": tuple(option["risks"]),
        "checks": tuple(option["checks"]),
    }
    if _is_nonempty_string(option.get("intensity")):
        cleaned["intensity"] = option["intensity"].strip()
    return cleaned


def _references_are_known(value: Any, allowed: frozenset[str]) -> bool:
    return _is_string_sequence(value) and all(item in allowed for item in value)


def _is_string_sequence(value: Any, *, nonempty: bool = False) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if nonempty and not value:
        return False
    return all(_is_nonempty_string(item) for item in value)


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _packet_citation_ids(packet: EvidencePacket) -> frozenset[str]:
    return frozenset(citation.record_id for citation in packet.citations)


def _merge_limitations(
    model_limitations: tuple[str, ...],
    authoritative_limitations: Any,
) -> tuple[str, ...]:
    return _deduplicate((*model_limitations, *authoritative_limitations))


def _deduplicate(items: Any) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(items))


def _offline_answer(
    question: str,
    packet: EvidencePacket,
    preset: bool,
) -> GeneratedResult:
    if preset or question not in PRESET_QUESTIONS:
        return answer_offline(question, packet)

    # A value that looks like a preset key is still a free-form question when
    # ``preset`` is false, so keep the offline answer conservative.
    result = answer_offline(f"自由问题：{question}", packet)
    payload = dict(result.payload)
    payload["question"] = question
    return GeneratedResult(mode="offline", payload=payload)


def _degraded(offline: GeneratedResult, reason: str) -> GeneratedResult:
    return GeneratedResult(
        mode="offline",
        payload=offline.payload,
        warning=f"在线生成已降级为离线模式：{_safe_reason(reason)}。",
    )


def _safe_reason(reason: Any) -> str:
    if not isinstance(reason, str):
        return "在线模型不可用"
    lowered = reason.casefold()
    if "api key" in lowered or "apikey" in lowered:
        return "未找到 API key"
    if "sdk" in lowered:
        return "在线模型 SDK 不可用"
    if "json" in lowered:
        return "模型输出不是有效 JSON 对象"
    if "校验" in reason or "validation" in lowered or "schema" in lowered:
        return "模型输出未通过安全校验"
    if "序列化" in reason:
        return "证据无法安全序列化"
    if "异常" in reason:
        return "在线模型调用异常，敏感详情已隐藏"
    return "在线模型不可用"
