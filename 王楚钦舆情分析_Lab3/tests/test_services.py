from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import replace
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
        self.call_count = 0

    def generate(self, task: str, evidence: Mapping[str, Any], validator: Any):
        self.call_count += 1
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


@pytest.mark.parametrize("service_name", ["brief", "qa", "strategy"])
def test_services_skip_online_generation_when_packet_has_no_citations(
    service_name: str,
    zero_comment_packet: Any,
) -> None:
    module = _services()
    packet = replace(zero_comment_packet, citations=())
    spy = StubLLM({})
    assert packet.comments is not None
    assert packet.comments.n == 0
    assert packet.citations == ()

    if service_name == "brief":
        result = module.BriefService(spy).generate(packet)
    elif service_name == "qa":
        result = module.QAService(spy).answer(
            "当前样本能说明什么？",
            packet,
        )
    else:
        result = module.StrategyService(spy).generate(
            packet,
            goal="回应争议",
            audience="球迷",
        )

    assert result.mode == "offline"
    assert "降级" in result.warning
    assert "证据" in result.warning
    assert "无可用引文" in result.warning or "为空" in result.warning
    assert spy.call_count == 0


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


def test_brief_empty_citation_ids_are_rejected_when_packet_has_evidence(
    loss_packet: Any,
) -> None:
    module = _services()
    payload = _brief_payload(loss_packet.citations[0].record_id)
    payload["citation_ids"] = []

    result = module.BriefService(StubLLM(payload)).generate(loss_packet)

    assert result.mode == "offline"
    assert result.payload == brief_offline(loss_packet).payload
    assert "降级" in result.warning


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


@pytest.mark.parametrize("empty_field", ["facts", "citation_ids"])
def test_answerable_qa_requires_nonempty_facts_and_citations(
    empty_field: str,
    loss_packet: Any,
) -> None:
    module = _services()
    citation_id = loss_packet.citations[0].record_id
    payload = _qa_payload(loss_packet, citation_id)
    payload[empty_field] = []

    result = module.QAService(StubLLM(payload)).answer(
        "当前样本能说明什么？",
        loss_packet,
    )

    assert result.mode == "offline"
    assert result.payload["answerable"] is False
    assert "降级" in result.warning


def test_unanswerable_online_qa_may_have_empty_facts_and_citations(
    loss_packet: Any,
) -> None:
    module = _services()
    payload = _qa_payload(
        loss_packet,
        loss_packet.citations[0].record_id,
    )
    payload["answerable"] = False
    payload["facts"] = []
    payload["citation_ids"] = []

    result = module.QAService(StubLLM(payload)).answer(
        "证据外的问题",
        loss_packet,
    )

    assert result.mode == "online"
    assert result.payload["answerable"] is False
    assert result.payload["facts"] == ()
    assert result.payload["citation_ids"] == ()


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


def test_online_strategy_forces_authoritative_packet_limitations(
    loss_packet: Any,
) -> None:
    module = _services()
    citation_id = loss_packet.citations[0].record_id

    result = module.StrategyService(
        StubLLM(_strategy_payload(citation_id))
    ).generate(
        loss_packet,
        goal="回应争议",
        audience="球迷",
    )

    assert result.mode == "online"
    limitations = result.payload["limitations"]
    assert all(warning in limitations for warning in loss_packet.warnings)
    assert any("n=1" in warning for warning in limitations)
    assert any(
        "不能代表微博总体舆情" in limitation
        for limitation in limitations
    )
    assert len(limitations) == len(tuple(dict.fromkeys(limitations)))


def test_offline_strategy_exposes_the_same_authoritative_limitations(
    loss_packet: Any,
) -> None:
    result = strategies_offline(
        loss_packet,
        goal="回应争议",
        audience="球迷",
    )

    limitations = result.payload["limitations"]
    assert all(warning in limitations for warning in loss_packet.warnings)
    assert any("n=1" in warning for warning in limitations)
    assert any(
        "不能代表微博总体舆情" in limitation
        for limitation in limitations
    )
    assert len(limitations) == len(tuple(dict.fromkeys(limitations)))


def test_strategy_limitations_deduplicate_repeated_packet_warnings(
    loss_packet: Any,
) -> None:
    repeated_warning = "重复的样本边界警告。"
    packet = replace(
        loss_packet,
        warnings=(repeated_warning, repeated_warning),
    )

    result = strategies_offline(
        packet,
        goal="回应争议",
        audience="球迷",
    )

    assert result.payload["limitations"].count(repeated_warning) == 1
    assert any(
        "不能代表微博总体舆情" in limitation
        for limitation in result.payload["limitations"]
    )


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


def test_strategy_option_requires_nonempty_evidence_ids(
    loss_packet: Any,
) -> None:
    module = _services()
    citation_id = loss_packet.citations[0].record_id
    payload = _strategy_payload(citation_id)
    payload["options"][1]["evidence_ids"] = []

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
