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


def test_loss_all_negative_uses_comparison_loss_comments(
    comparison_packet: Any,
) -> None:
    offline = _offline()

    result = offline.answer_offline(
        "loss_all_negative",
        comparison_packet,
    )

    assert result.payload["answerable"] is True
    interpretation = result.payload["interpretation"]
    loss_comments = comparison_packet.comment_comparison.loss
    assert "输球不等于全部负面" in interpretation
    assert f"{loss_comments.polarity_pct['positive']:.4f}%" in interpretation
    assert f"{loss_comments.polarity_pct['neutral']:.4f}%" in interpretation
    assert f"{loss_comments.polarity_pct['negative']:.4f}%" in interpretation
    expected_ids = tuple(
        citation.record_id
        for citation in comparison_packet.citations
        if citation.content_type == "comment"
        and citation.event_id.startswith("loss_")
    )
    assert result.payload["citation_ids"] == expected_ids


def test_loss_all_negative_reports_all_negative_without_contradiction() -> None:
    from lab3.models import (
        AnalysisScope,
        Citation,
        EvidencePacket,
        MetricSummary,
    )

    offline = _offline()
    comments = MetricSummary(
        n=1,
        mean_score=-1.0,
        polarity_pct={
            "positive": 0.0,
            "neutral": 0.0,
            "negative": 100.0,
        },
        top_emotions=(("失望", 100.0),),
        top_topics=(("赛果", 100.0),),
    )
    citation = Citation(
        record_id="negative-only-comment",
        content_type="comment",
        event_id="loss_synthetic",
        event_name="合成负场",
        text="当前样本中的负面评论。",
        polarity="negative",
        emotion="失望",
        topics=("赛果",),
        confidence=1.0,
        likes=0,
    )
    packet = EvidencePacket(
        label="全负面评论样本",
        scope=AnalysisScope(
            kind="single_event",
            source="comment",
            audience="内部团队",
            event_id="loss_synthetic",
        ),
        posts=None,
        comments=comments,
        citations=(citation,),
        warnings=("合成案例样本。",),
        facts=("评论负面占比为 100.0000%。",),
    )

    result = offline.answer_offline("loss_all_negative", packet)

    assert result.payload["answerable"] is True
    interpretation = result.payload["interpretation"]
    assert "100.0000%" in interpretation
    assert "当前样本" in interpretation
    assert "全部为负面" in interpretation
    assert "不能外推" in interpretation
    assert "不等于" not in interpretation
    assert result.payload["citation_ids"] == (citation.record_id,)


def test_loss_all_negative_accepts_only_loss_single_events(
    project_data: Any,
) -> None:
    from lab3.evidence import build_evidence
    from lab3.models import AnalysisScope

    offline = _offline()
    loss_event_id = next(
        event_id
        for event_id, event in project_data.events.items()
        if event["result"] == "loss"
        and any(
            row["event_id"] == event_id
            for row in project_data.comments
        )
    )
    win_event_id = next(
        event_id
        for event_id, event in project_data.events.items()
        if event["result"] == "win"
        and any(
            row["event_id"] == event_id
            for row in project_data.comments
        )
    )

    def packet_for(event_id: str):
        return build_evidence(
            project_data,
            AnalysisScope(
                kind="single_event",
                source="comment",
                audience="球迷",
                event_id=event_id,
            ),
        )

    loss_packet = packet_for(loss_event_id)
    win_packet = packet_for(win_event_id)
    assert loss_packet.comments.n > 0
    assert win_packet.comments.n > 0

    assert offline.answer_offline(
        "loss_all_negative", loss_packet
    ).payload["answerable"] is True
    assert offline.answer_offline(
        "loss_all_negative", win_packet
    ).payload["answerable"] is False


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


def test_source_difference_uses_win_loss_comparison_metrics(
    comparison_packet: Any,
) -> None:
    offline = _offline()

    result = offline.answer_offline(
        "source_difference",
        comparison_packet,
    )

    assert result.payload["answerable"] is True
    interpretation = result.payload["interpretation"]
    assert all(
        label in interpretation
        for label in ("正文胜组", "正文负组", "评论胜组", "评论负组")
    )
    assert (
        f"n={comparison_packet.post_comparison.win.n}"
        in interpretation
    )
    assert (
        f"n={comparison_packet.comment_comparison.loss.n}"
        in interpretation
    )
    assert set(result.payload["citation_ids"]) <= _citation_ids(
        comparison_packet
    )


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
        assert "分开呈现正文与评论" not in option["action"]

    options = {
        option["name"]: option
        for option in strategy_first.payload["options"]
    }
    emotion_action = options["及时情绪回应"]["action"]
    assert "评论零记录" in emotion_action
    assert "证据不足" in emotion_action
    assert "克制表达" in emotion_action or "暂缓判断" in emotion_action
    facts_action = options["事实说明与复盘"]["action"]
    assert "评论零记录" in facts_action
    assert "只陈述" in facts_action
    assert "已核实事实" in facts_action
    monitoring_action = options["持续监测"]["action"]
    assert "评论零记录" in monitoring_action
    assert "持续监测评论来源" in monitoring_action


def test_strategies_for_post_only_scope_do_not_assume_comments(
    project_data: Any,
) -> None:
    from lab3.evidence import build_evidence
    from lab3.models import AnalysisScope

    offline = _offline()
    packet = build_evidence(
        project_data,
        AnalysisScope(
            kind="win_group",
            source="post",
            audience="媒体",
        ),
    )

    result = offline.strategies_offline(
        packet,
        goal="准备媒体简报",
        audience="媒体",
    )
    options = {option["name"]: option for option in result.payload["options"]}

    assert "当前仅有正文证据" in options["及时情绪回应"]["action"]
    assert "仅依据正文样本" in options["事实说明与复盘"]["action"]
    assert "持续监测正文来源" in options["持续监测"]["action"]
    assert all(
        "分开呈现正文与评论" not in option["action"]
        for option in options.values()
    )


def test_strategies_for_comment_only_scope_do_not_assume_posts(
    project_data: Any,
) -> None:
    from lab3.evidence import build_evidence
    from lab3.models import AnalysisScope

    offline = _offline()
    packet = build_evidence(
        project_data,
        AnalysisScope(
            kind="loss_group",
            source="comment",
            audience="球迷",
        ),
    )

    result = offline.strategies_offline(
        packet,
        goal="稳定球迷情绪",
        audience="球迷",
    )
    options = {option["name"]: option for option in result.payload["options"]}

    assert "当前仅有评论证据" in options["及时情绪回应"]["action"]
    assert "仅依据评论样本" in options["事实说明与复盘"]["action"]
    assert "持续监测评论来源" in options["持续监测"]["action"]
    assert all(
        "分开呈现正文与评论" not in option["action"]
        for option in options.values()
    )


def test_duplicate_citation_ids_are_emitted_once_in_first_seen_order(
    loss_packet: Any,
) -> None:
    from dataclasses import replace

    offline = _offline()
    first, second = loss_packet.citations[:2]
    packet = replace(
        loss_packet,
        citations=(first, second, first, second),
    )

    brief = offline.brief_offline(packet)
    strategies = offline.strategies_offline(
        packet,
        goal="回应争议",
        audience="球迷",
    )

    expected = (first.record_id, second.record_id)
    assert brief.payload["citation_ids"] == expected
    assert all(
        option["evidence_ids"] == expected
        for option in strategies.payload["options"]
    )
