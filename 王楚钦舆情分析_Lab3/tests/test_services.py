from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import pytest

from lab3.llm_client import DeepSeekClient, LLMCallResult
from lab3.offline import answer_offline, brief_offline, strategies_offline


def _services():
    return importlib.import_module("lab3.services")


class StubLLM:
    """Return scripted results without performing any network operation."""

    def __init__(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        reason: str = "在线模型不可用",
        exception: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.reason = reason
        self.exception = exception
        self.calls: list[dict[str, Any]] = []

    def generate(self, task: str, evidence: Mapping[str, Any], validator: Any):
        self.calls.append(
            {"task": task, "evidence": evidence, "validator": validator}
        )
        if self.exception is not None:
            raise self.exception
        if self.payload is None:
            return LLMCallResult(False, None, self.reason)
        # Deliberately bypass validator: services must enforce the schema too.
        return LLMCallResult(True, self.payload, None)


def _brief_payload(citation_id: str) -> dict[str, Any]:
    return {
        "title": "在线舆情简报",
        "facts": ["模型声称共有 999 条评论。"],
        "observations": ["当前样本同时存在多类情绪表达。"],
        "decision_focus": ["先核验当前证据，再由人工决定。"],
        "limitations": ["仅依据当前证据。"],
        "citation_ids": [citation_id],
    }


def _qa_payload(packet: Any, citation_id: str) -> dict[str, Any]:
    return {
        "question": "模型改写的问题",
        "answerable": True,
        "facts": [packet.facts[0]],
        "interpretation": "当前证据仅支持样本内的描述性回答。",
        "limitations": ["不能外推。"],
        "citation_ids": [citation_id],
    }


def _strategy_payload(citation_id: str) -> dict[str, Any]:
    options = []
    for index in range(3):
        options.append(
            {
                "name": f"在线方案 {index + 1}",
                "action": "只依据当前证据进行说明。",
                "timing": "完成人工核验后。",
                "benefits": ["保留事实边界。"],
                "risks": ["样本覆盖有限。"],
                "checks": ["逐项核验引用。"],
                "evidence_ids": [citation_id],
            }
        )
    return {
        "goal": "模型擅自修改的目标",
        "audience": "模型擅自修改的受众",
        "options": options,
        "disclaimer": "方案非预测，最终必须由人工决定。",
    }


def test_missing_key_brief_service_falls_back_without_network(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    loss_packet: Any,
) -> None:
    module = _services()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = DeepSeekClient(tmp_path)

    result = module.BriefService(client).generate(loss_packet)

    assert result.mode == "offline"
    assert result.payload == brief_offline(loss_packet).payload
    assert "降级" in result.warning
    assert "API key" in result.warning


def test_online_brief_uses_model_wording_but_forces_packet_facts_and_limits(
    loss_packet: Any,
) -> None:
    module = _services()
    citation_id = loss_packet.citations[0].record_id
    fake = StubLLM(_brief_payload(citation_id))

    result = module.BriefService(fake).generate(loss_packet)

    assert result.mode == "online"
    assert result.warning is None
    assert result.payload["title"] == "在线舆情简报"
    assert result.payload["facts"] == loss_packet.facts
    assert "999" not in "".join(result.payload["facts"])
    assert all(
        warning in result.payload["limitations"]
        for warning in loss_packet.warnings
    )
    assert any(
        "不能代表微博总体舆情" in limitation
        for limitation in result.payload["limitations"]
    )
    assert result.payload["citation_ids"] == (citation_id,)
    assert fake.calls[0]["validator"](_brief_payload(citation_id)) is True


def test_brief_unknown_citation_or_missing_field_is_rejected_and_downgraded(
    loss_packet: Any,
) -> None:
    module = _services()
    unknown = StubLLM(_brief_payload("unknown-citation"))
    missing = _brief_payload(loss_packet.citations[0].record_id)
    missing.pop("decision_focus")

    unknown_result = module.BriefService(unknown).generate(loss_packet)
    missing_result = module.BriefService(StubLLM(missing)).generate(loss_packet)

    assert unknown_result.mode == "offline"
    assert missing_result.mode == "offline"
    assert "降级" in unknown_result.warning
    assert "降级" in missing_result.warning
    assert unknown_result.payload == brief_offline(loss_packet).payload
    assert missing_result.payload == brief_offline(loss_packet).payload


def test_service_exception_downgrade_hides_sensitive_exception_text(
    loss_packet: Any,
) -> None:
    module = _services()
    fake = StubLLM(
        exception=RuntimeError(
            "provider failed sk-secret-value 完整敏感证据"
        )
    )

    result = module.BriefService(fake).generate(loss_packet)

    assert result.mode == "offline"
    assert "降级" in result.warning
    assert "sk-secret-value" not in result.warning
    assert "完整敏感证据" not in result.warning
    assert "provider failed" not in result.warning


def test_online_qa_keeps_only_packet_fact_subset_and_known_citations(
    loss_packet: Any,
) -> None:
    module = _services()
    question = "当前样本能说明什么？"
    citation_id = loss_packet.citations[0].record_id
    fake = StubLLM(_qa_payload(loss_packet, citation_id))

    result = module.QAService(fake).answer(
        question,
        loss_packet,
        preset=False,
    )

    assert result.mode == "online"
    assert result.payload["question"] == question
    assert set(result.payload["facts"]) <= set(loss_packet.facts)
    assert set(result.payload["citation_ids"]) <= {
        citation.record_id for citation in loss_packet.citations
    }
    assert fake.calls[0]["evidence"]["question"] == question
    assert fake.calls[0]["evidence"]["preset"] is False


def test_qa_fact_outside_packet_is_rejected_and_downgraded(
    loss_packet: Any,
) -> None:
    module = _services()
    citation_id = loss_packet.citations[0].record_id
    payload = _qa_payload(loss_packet, citation_id)
    payload["facts"] = ["证据包中不存在的事实。"]

    result = module.QAService(StubLLM(payload)).answer(
        "这个结论可靠吗？",
        loss_packet,
    )

    assert result.mode == "offline"
    assert result.payload["answerable"] is False
    assert "降级" in result.warning


def test_qa_preset_fallback_passes_the_key_to_offline_answer(
    loss_packet: Any,
) -> None:
    module = _services()
    fake = StubLLM(reason="API key unavailable")

    result = module.QAService(fake).answer(
        "loss_all_negative",
        loss_packet,
        preset=True,
    )

    expected = answer_offline("loss_all_negative", loss_packet)
    assert result.mode == "offline"
    assert result.payload == expected.payload
    assert result.payload["answerable"] is True
    assert fake.calls[0]["evidence"]["question"] == "loss_all_negative"
    assert fake.calls[0]["evidence"]["preset"] is True
    assert "降级" in result.warning


def test_qa_free_question_fallback_is_a_conservative_refusal(
    loss_packet: Any,
) -> None:
    module = _services()
    question = "请预测下一场比赛比分"

    result = module.QAService(StubLLM()).answer(
        question,
        loss_packet,
        preset=False,
    )

    assert result.mode == "offline"
    assert result.payload["question"] == question
    assert result.payload["answerable"] is False
    assert result.payload["citation_ids"] == ()
    assert "无法可靠回答" in result.payload["interpretation"]
    assert "降级" in result.warning


def test_online_strategy_requires_three_options_and_forces_user_inputs(
    loss_packet: Any,
) -> None:
    module = _services()
    citation_id = loss_packet.citations[0].record_id
    fake = StubLLM(_strategy_payload(citation_id))

    result = module.StrategyService(fake).generate(
        loss_packet,
        goal="回应争议",
        audience="球迷",
    )

    assert result.mode == "online"
    assert result.payload["goal"] == "回应争议"
    assert result.payload["audience"] == "球迷"
    assert len(result.payload["options"]) == 3
    assert all(option["benefits"] for option in result.payload["options"])
    assert "人工" in result.payload["disclaimer"]
    assert "非预测" in result.payload["disclaimer"]
    assert fake.calls[0]["evidence"]["goal"] == "回应争议"
    assert fake.calls[0]["evidence"]["audience"] == "球迷"


def test_strategy_bad_schema_downgrades_to_offline(loss_packet: Any) -> None:
    module = _services()
    citation_id = loss_packet.citations[0].record_id
    payload = _strategy_payload(citation_id)
    payload["options"] = payload["options"][:2]

    result = module.StrategyService(StubLLM(payload)).generate(
        loss_packet,
        goal="回应争议",
        audience="球迷",
    )

    assert result.mode == "offline"
    assert result.payload == strategies_offline(
        loss_packet,
        goal="回应争议",
        audience="球迷",
    ).payload
    assert "降级" in result.warning


def test_strategy_unknown_evidence_id_downgrades_to_offline(
    loss_packet: Any,
) -> None:
    module = _services()
    payload = _strategy_payload("unknown-evidence-id")

    result = module.StrategyService(StubLLM(payload)).generate(
        loss_packet,
        goal="稳定球迷情绪",
        audience="球迷",
    )

    assert result.mode == "offline"
    assert result.payload == strategies_offline(
        loss_packet,
        goal="稳定球迷情绪",
        audience="球迷",
    ).payload
    assert "降级" in result.warning
