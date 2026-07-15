from __future__ import annotations

import importlib
import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest


def _models() -> Any:
    return importlib.import_module("lab3.models")


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
    models = _models()

    with pytest.raises(ValueError, match="event_id"):
        models.AnalysisScope(
            kind="single_event",
            source="both",
            audience="球迷",
        )


@pytest.mark.parametrize(
    "kind",
    ["win_group", "loss_group", "win_loss_comparison"],
)
def test_non_single_event_scope_rejects_event_id(kind: str) -> None:
    models = _models()

    with pytest.raises(ValueError):
        models.AnalysisScope(
            kind=kind,
            source="post",
            audience="球迷",
            event_id="event-001",
        )


def test_evidence_packet_is_frozen() -> None:
    models = _models()
    packet = models.EvidencePacket(
        label="胜场舆情",
        scope=models.AnalysisScope(
            kind="win_group",
            source="both",
            audience="球迷",
        ),
        posts=None,
        comments=None,
        citations=(),
        warnings=(),
        facts={},
    )

    with pytest.raises(FrozenInstanceError):
        packet.label = "新标签"


def test_as_prompt_dict_recursively_returns_json_safe_plain_structure() -> None:
    models = _models()
    summary = models.MetricSummary(
        n=4,
        mean_score=0.25,
        polarity_pct={"positive": 50.0, "neutral": 25.0, "negative": 25.0},
        top_emotions=(("喜悦", 2), ("期待", 1)),
        top_topics=(("技战术", 3),),
    )
    citation = models.Citation(
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
    packet = models.EvidencePacket(
        label="胜场舆情",
        scope=models.AnalysisScope(
            kind="win_group",
            source="both",
            audience="球迷",
        ),
        posts=summary,
        comments=None,
        citations=(citation,),
        warnings=("评论样本较少",),
        facts={
            "nested": ("第一层", {"scores": (0.2, 0.8)}),
        },
    )

    prompt_data = packet.as_prompt_dict()

    assert isinstance(prompt_data, dict)
    assert prompt_data["posts"]["top_emotions"] == [["喜悦", 2], ["期待", 1]]
    assert prompt_data["citations"][0]["topics"] == ["技战术", "心态"]
    assert prompt_data["warnings"] == ["评论样本较少"]
    assert prompt_data["facts"]["nested"] == [
        "第一层",
        {"scores": [0.2, 0.8]},
    ]
    _assert_plain_json_structure(prompt_data)
    json.dumps(prompt_data, ensure_ascii=False)
