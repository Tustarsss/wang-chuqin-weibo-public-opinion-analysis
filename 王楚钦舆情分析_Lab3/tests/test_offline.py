from __future__ import annotations

import importlib
from typing import Any


def _offline():
    return importlib.import_module("lab3.offline")


def _citation_ids(packet: Any) -> set[str]:
    return {citation.record_id for citation in packet.citations}


def test_offline_exposes_required_questions_and_strategy_goals() -> None:
    offline = _offline()

    assert {
        "loss_all_negative",
        "source_difference",
        "top_topics",
        "representative_views",
        "coverage_limits",
    } <= set(offline.PRESET_QUESTIONS)
    assert {
        "回应争议",
        "稳定球迷情绪",
        "准备媒体简报",
        "内部复盘",
    } <= set(offline.STRATEGY_GOALS)


def test_brief_preserves_facts_and_only_uses_packet_citations(
    loss_packet: Any,
) -> None:
    offline = _offline()

    result = offline.brief_offline(loss_packet)

    assert result.mode == "offline"
    assert {
        "title",
        "facts",
        "observations",
        "decision_focus",
        "limitations",
        "citation_ids",
    } <= set(result.payload)
    assert result.payload["facts"] == loss_packet.facts
    assert set(result.payload["citation_ids"]) <= _citation_ids(loss_packet)
    assert all(
        warning in result.payload["limitations"]
        for warning in loss_packet.warnings
    )
    assert any(
        "不能代表微博总体舆情" in limitation
        for limitation in result.payload["limitations"]
    )


def test_loss_all_negative_uses_comment_mix_and_current_evidence(
    loss_packet: Any,
) -> None:
    offline = _offline()

    result = offline.answer_offline("loss_all_negative", loss_packet)

    assert result.mode == "offline"
    assert result.payload["answerable"] is True
    assert result.payload["facts"] == loss_packet.facts
    assert "输球不等于全部负面" in result.payload["interpretation"]
    assert all(
        label in result.payload["interpretation"]
        for label in ("评论", "正面", "中性", "负面")
    )
    assert set(result.payload["citation_ids"]) <= _citation_ids(loss_packet)
    assert result.payload["citation_ids"]


def test_other_known_questions_are_deterministically_answered_from_packet(
    loss_packet: Any,
) -> None:
    offline = _offline()

    answers = {
        key: offline.answer_offline(key, loss_packet).payload
        for key in (
            "source_difference",
            "top_topics",
            "representative_views",
            "coverage_limits",
        )
    }

    assert all(answer["answerable"] is True for answer in answers.values())
    assert all(
        set(answer["citation_ids"]) <= _citation_ids(loss_packet)
        for answer in answers.values()
    )
    assert "正文" in answers["source_difference"]["interpretation"]
    assert "评论" in answers["source_difference"]["interpretation"]
    assert loss_packet.posts.top_topics[0][0] in answers["top_topics"][
        "interpretation"
    ]
    assert loss_packet.citations[0].text in answers["representative_views"][
        "interpretation"
    ]
    assert loss_packet.warnings[0] in answers["coverage_limits"][
        "limitations"
    ]


def test_unknown_question_is_refused_without_fabricating_an_answer(
    loss_packet: Any,
) -> None:
    offline = _offline()

    result = offline.answer_offline("future_score_prediction", loss_packet)

    assert result.payload["answerable"] is False
    assert "离线" in result.payload["interpretation"]
    assert "无法可靠回答" in result.payload["interpretation"]
    assert set(result.payload["available_questions"]) == set(
        offline.PRESET_QUESTIONS
    )
    assert result.payload["citation_ids"] == ()


def test_strategies_return_three_complete_distinct_human_decision_options(
    loss_packet: Any,
) -> None:
    offline = _offline()

    result = offline.strategies_offline(
        loss_packet,
        goal="回应争议",
        audience="球迷",
    )

    assert result.mode == "offline"
    assert result.payload["goal"] == "回应争议"
    assert result.payload["audience"] == "球迷"
    options = result.payload["options"]
    assert len(options) == 3
    assert len({option["name"] for option in options}) == 3
    assert {option["name"] for option in options} == {
        "及时情绪回应",
        "事实说明与复盘",
        "持续监测",
    }
    for option in options:
        assert {
            "name",
            "action",
            "timing",
            "evidence_ids",
            "benefits",
            "risks",
            "checks",
        } <= set(option)
        assert set(option["evidence_ids"]) <= _citation_ids(loss_packet)
        assert option["benefits"]
        assert option["risks"]
        assert option["checks"]
    assert "人工" in result.payload["disclaimer"]
    assert "非预测" in result.payload["disclaimer"]


def test_zero_comment_packet_has_no_invented_evidence_and_is_repeatable(
    zero_comment_packet: Any,
) -> None:
    offline = _offline()

    brief_first = offline.brief_offline(zero_comment_packet)
    brief_second = offline.brief_offline(zero_comment_packet)
    answer_first = offline.answer_offline(
        "representative_views", zero_comment_packet
    )
    answer_second = offline.answer_offline(
        "representative_views", zero_comment_packet
    )
    strategy_first = offline.strategies_offline(
        zero_comment_packet,
        goal="准备媒体简报",
        audience="媒体",
    )
    strategy_second = offline.strategies_offline(
        zero_comment_packet,
        goal="准备媒体简报",
        audience="媒体",
    )

    assert brief_first == brief_second
    assert brief_first.payload["facts"] == zero_comment_packet.facts
    assert brief_first.payload["citation_ids"] == ()
    assert answer_first == answer_second
    assert answer_first.payload["answerable"] is False
    assert answer_first.payload["citation_ids"] == ()
    assert strategy_first == strategy_second
    for option in strategy_first.payload["options"]:
        assert option["evidence_ids"] == ()
        assert any("证据不足" in check for check in option["checks"])
