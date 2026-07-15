from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from lab3.models import (
    AnalysisScope,
    Citation,
    EvidencePacket,
    GeneratedResult,
    MetricSummary,
)


def _assert_plain_json_structure(value: Any) -> None:
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value)
        for item in value.values():
            _assert_plain_json_structure(item)
        return

    if isinstance(value, list):
        for item in value:
            _assert_plain_json_structure(item)
        return

    assert value is None or isinstance(value, (str, int, float, bool))


def test_single_event_scope_requires_event_id() -> None:
    with pytest.raises(ValueError, match="event_id"):
        AnalysisScope(
            kind="single_event",
            source="both",
            audience="球迷",
        )


@pytest.mark.parametrize(
    "kind",
    ["win_group", "loss_group", "win_loss_comparison"],
)
def test_non_single_event_scope_rejects_event_id(kind: str) -> None:
    with pytest.raises(ValueError):
        AnalysisScope(
            kind=kind,
            source="post",
            audience="球迷",
            event_id="event-001",
        )


def test_evidence_packet_is_frozen() -> None:
    packet = EvidencePacket(
        label="胜场舆情",
        scope=AnalysisScope(
            kind="win_group",
            source="both",
            audience="球迷",
        ),
        posts=None,
        comments=None,
        citations=(),
        warnings=(),
        facts=(),
    )

    with pytest.raises(FrozenInstanceError):
        packet.label = "新标签"


def test_as_prompt_dict_recursively_returns_json_safe_plain_structure() -> None:
    summary = MetricSummary(
        n=4,
        mean_score=0.25,
        polarity_pct={"positive": 50.0, "neutral": 25.0, "negative": 25.0},
        top_emotions=(("喜悦", 2), ("期待", 1)),
        top_topics=(("技战术", 3),),
    )
    citation = Citation(
        record_id="post-001",
        content_type="post",
        event_id="event-001",
        event_name="巴黎奥运会男单",
        text="关键分处理稳定。",
        polarity="positive",
        emotion="喜悦",
        topics=("技战术", "心态"),
        confidence=0.93,
        likes=88,
    )
    packet = EvidencePacket(
        label="胜场舆情",
        scope=AnalysisScope(
            kind="win_group",
            source="both",
            audience="球迷",
        ),
        posts=summary,
        comments=None,
        citations=(citation,),
        warnings=("评论样本较少",),
        facts=("关键分处理稳定", "正向占比为 50%"),
    )

    prompt_data = packet.as_prompt_dict()

    assert isinstance(prompt_data, dict)
    assert prompt_data["posts"]["top_emotions"] == [["喜悦", 2], ["期待", 1]]
    assert prompt_data["citations"][0]["topics"] == ["技战术", "心态"]
    assert prompt_data["warnings"] == ["评论样本较少"]
    assert prompt_data["facts"] == ["关键分处理稳定", "正向占比为 50%"]
    _assert_plain_json_structure(prompt_data)
    json.dumps(prompt_data, ensure_ascii=False, allow_nan=False)


def test_metric_summary_isolated_from_mutable_input_aliases() -> None:
    polarity_pct = {"positive": 100.0}
    top_emotions = [["喜悦", 2]]
    top_topics = [["技战术", 3]]
    summary = MetricSummary(
        n=2,
        mean_score=0.8,
        polarity_pct=polarity_pct,
        top_emotions=top_emotions,
        top_topics=top_topics,
    )

    polarity_pct["positive"] = 0.0
    top_emotions[0][1] = 99
    top_emotions.append(["期待", 1])
    top_topics.clear()

    assert dict(summary.polarity_pct) == {"positive": 100.0}
    assert summary.top_emotions == (("喜悦", 2),)
    assert summary.top_topics == (("技战术", 3),)


def test_metric_summary_mapping_is_read_only() -> None:
    summary = MetricSummary(
        n=1,
        mean_score=0.5,
        polarity_pct={"positive": 100.0},
        top_emotions=(),
        top_topics=(),
    )

    with pytest.raises(TypeError):
        summary.polarity_pct["positive"] = 0.0


def test_generated_result_recursively_copies_mutable_payload() -> None:
    payload = {"nested": {"scores": [0.2, 0.8]}}
    result = GeneratedResult(payload=payload, mode="offline")

    payload["nested"]["scores"].append(1.0)
    payload["nested"]["status"] = "mutated"

    assert result.payload["nested"]["scores"] == (0.2, 0.8)
    assert "status" not in result.payload["nested"]


def test_generated_result_nested_mapping_is_read_only() -> None:
    result = GeneratedResult(
        payload={"nested": {"score": 0.8}},
        mode="online",
    )

    with pytest.raises(TypeError):
        result.payload["nested"]["score"] = 0.0


def test_citation_and_evidence_packet_tupleize_sequence_inputs() -> None:
    topics = ["技战术"]
    citation = Citation(
        record_id="post-001",
        content_type="post",
        event_id="event-001",
        event_name="巴黎奥运会男单",
        text="关键分处理稳定。",
        polarity="positive",
        emotion="喜悦",
        topics=topics,
        confidence=0.93,
        likes=88,
    )
    citations = [citation]
    warnings = ["评论样本较少"]
    facts = ["关键分处理稳定"]
    packet = EvidencePacket(
        label="胜场舆情",
        scope=AnalysisScope(
            kind="win_group",
            source="both",
            audience="球迷",
        ),
        posts=None,
        comments=None,
        citations=citations,
        warnings=warnings,
        facts=facts,
    )

    topics.append("心态")
    citations.clear()
    warnings.append("新增警告")
    facts.append("新增事实")

    assert citation.topics == ("技战术",)
    assert packet.citations == (citation,)
    assert packet.warnings == ("评论样本较少",)
    assert packet.facts == ("关键分处理稳定",)


@pytest.mark.parametrize(
    "non_finite",
    [math.nan, math.inf, -math.inf],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_as_prompt_dict_rejects_non_finite_metric_values(
    non_finite: float,
) -> None:
    packet = EvidencePacket(
        label="异常指标",
        scope=AnalysisScope(
            kind="win_group",
            source="post",
            audience="球迷",
        ),
        posts=MetricSummary(
            n=1,
            mean_score=non_finite,
            polarity_pct={"positive": 100.0},
            top_emotions=(),
            top_topics=(),
        ),
        comments=None,
        citations=(),
        warnings=(),
        facts=(),
    )

    with pytest.raises(ValueError, match="finite"):
        packet.as_prompt_dict()


@pytest.mark.parametrize(
    "non_finite",
    [math.nan, math.inf, -math.inf],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_generated_result_rejects_non_finite_nested_payload(
    non_finite: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        GeneratedResult(
            payload={"score": non_finite},
            mode="offline",
        )
