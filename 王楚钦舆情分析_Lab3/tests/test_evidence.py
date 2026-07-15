from __future__ import annotations

import json
from typing import Any

import pytest

from lab3.evidence import build_evidence, scope_event_ids
from lab3.models import AnalysisScope


SINGAPORE_LOSS = "loss_20250208_singapore_liang"
CHINA_SMASH_WIN = "win_20251005_china_smash_lebrun"


def _scope(
    kind: str,
    source: str = "both",
    event_id: str | None = None,
) -> AnalysisScope:
    return AnalysisScope(
        kind=kind,
        source=source,
        audience="球迷",
        event_id=event_id,
    )


def _row_lookup(project_data: Any) -> dict[tuple[str, str], Any]:
    rows: dict[tuple[str, str], Any] = {}
    for row in project_data.posts:
        rows[("post", row["post_id"])] = row
    for row in project_data.comments:
        rows[("comment", row["comment_id"])] = row
    return rows


def test_scope_event_ids_selects_exact_event_and_result_groups(
    project_data: Any,
) -> None:
    single = scope_event_ids(
        project_data,
        _scope("single_event", event_id=SINGAPORE_LOSS),
    )
    wins = scope_event_ids(project_data, _scope("win_group"))
    losses = scope_event_ids(project_data, _scope("loss_group"))
    comparison = scope_event_ids(
        project_data,
        _scope("win_loss_comparison"),
    )

    assert single == (SINGAPORE_LOSS,)
    assert len(wins) == len(losses) == 4
    assert all(project_data.events[event_id]["result"] == "win" for event_id in wins)
    assert all(project_data.events[event_id]["result"] == "loss" for event_id in losses)
    assert comparison == tuple(project_data.events)


def test_loss_group_keeps_post_and_comment_metrics_separate(
    project_data: Any,
) -> None:
    packet = build_evidence(project_data, _scope("loss_group"))
    loss_ids = {
        event_id
        for event_id, event in project_data.events.items()
        if event["result"] == "loss"
    }

    assert packet.posts is not None and packet.posts.n == 22
    assert packet.comments is not None and packet.comments.n == 31
    assert packet.post_comparison is None
    assert packet.comment_comparison is None
    assert any("正文" in fact and "n=22" in fact for fact in packet.facts)
    assert any("评论" in fact and "n=31" in fact for fact in packet.facts)
    assert packet.citations
    assert {citation.event_id for citation in packet.citations} <= loss_ids


def test_single_singapore_comment_reports_thin_coverage(
    project_data: Any,
) -> None:
    packet = build_evidence(
        project_data,
        _scope("single_event", "comment", SINGAPORE_LOSS),
    )

    assert packet.posts is None
    assert packet.comments is not None and packet.comments.n == 1
    assert any("少于 3 条" in warning for warning in packet.warnings)
    assert all(citation.content_type == "comment" for citation in packet.citations)
    assert all(citation.event_id == SINGAPORE_LOSS for citation in packet.citations)


def test_single_china_smash_comment_is_explicit_zero_not_fabricated(
    project_data: Any,
) -> None:
    packet = build_evidence(
        project_data,
        _scope("single_event", "comment", CHINA_SMASH_WIN),
    )

    assert packet.posts is None
    assert packet.comments is not None
    assert packet.comments.n == 0
    assert packet.comments.mean_score is None
    assert dict(packet.comments.polarity_pct) == {
        "positive": 0.0,
        "neutral": 0.0,
        "negative": 0.0,
    }
    assert packet.comments.top_emotions == ()
    assert packet.comments.top_topics == ()
    assert packet.citations == ()
    assert any("零记录" in warning for warning in packet.warnings)
    assert any("无可用记录" in fact for fact in packet.facts)


@pytest.mark.parametrize(
    ("source", "present_attr", "absent_attr", "content_type", "absent_word"),
    [
        ("post", "posts", "comments", "post", "评论"),
        ("comment", "comments", "posts", "comment", "正文"),
    ],
)
def test_source_filter_never_leaks_the_other_source(
    project_data: Any,
    source: str,
    present_attr: str,
    absent_attr: str,
    content_type: str,
    absent_word: str,
) -> None:
    packet = build_evidence(project_data, _scope("win_group", source))

    assert getattr(packet, present_attr) is not None
    assert getattr(packet, absent_attr) is None
    assert packet.citations
    assert {citation.content_type for citation in packet.citations} == {
        content_type
    }
    assert all(absent_word not in warning for warning in packet.warnings)
    assert all(absent_word not in fact for fact in packet.facts)


def test_win_loss_comparison_exposes_both_results_without_mixed_metric(
    project_data: Any,
) -> None:
    packet = build_evidence(
        project_data,
        _scope("win_loss_comparison"),
    )

    assert packet.posts is None
    assert packet.comments is None
    assert packet.post_comparison is not None
    assert packet.comment_comparison is not None
    assert packet.post_comparison.win.n == 23
    assert packet.post_comparison.loss.n == 22
    assert packet.comment_comparison.win.n == 30
    assert packet.comment_comparison.loss.n == 31
    assert any("胜负" in fact and "差异" in fact for fact in packet.facts)
    assert all("导致" not in fact and "因为" not in fact for fact in packet.facts)


def test_citations_are_scoped_unique_bounded_sorted_and_repeatable(
    project_data: Any,
) -> None:
    scope = _scope("loss_group")
    first = build_evidence(project_data, scope)
    second = build_evidence(project_data, scope)
    loss_ids = set(scope_event_ids(project_data, scope))
    rows = _row_lookup(project_data)
    identities = [
        (citation.content_type, citation.record_id)
        for citation in first.citations
    ]
    sort_keys = [
        (
            -citation.likes,
            -rows[(citation.content_type, citation.record_id)][
                "sentiment_intensity"
            ],
            citation.record_id,
        )
        for citation in first.citations
    ]

    assert 0 < len(first.citations) <= 8
    assert len(identities) == len(set(identities))
    assert all(citation.event_id in loss_ids for citation in first.citations)
    assert sort_keys == sorted(sort_keys)
    assert first.citations == second.citations


def test_citations_cover_polarities_high_interaction_and_top_topic(
    project_data: Any,
) -> None:
    packet = build_evidence(project_data, _scope("loss_group", "post"))
    rows = _row_lookup(project_data)
    selected_rows = [
        rows[(citation.content_type, citation.record_id)]
        for citation in packet.citations
    ]
    loss_post_rows = [
        row
        for row in project_data.posts
        if row["match_result"] == "loss"
    ]
    highest_likes = max(row["likes"] for row in loss_post_rows)
    top_topic = packet.posts.top_topics[0][0]

    assert {row["sentiment_polarity"] for row in selected_rows} >= {
        "positive",
        "negative",
    }
    assert any(row["likes"] == highest_likes for row in selected_rows)
    assert any(top_topic in row["topic_tags"] for row in selected_rows)


def test_top_metrics_are_value_sorted_and_converted_to_float_percentages(
    project_data: Any,
) -> None:
    packet = build_evidence(project_data, _scope("loss_group", "post"))

    assert packet.posts is not None
    assert packet.posts.top_emotions == (
        ("中性陈述", 44.1675),
        ("理性分析", 18.335),
        ("支持鼓励", 14.1675),
    )
    assert packet.posts.top_topics == (
        ("赛果", 95.83),
        ("心理心态", 37.5),
        ("对手", 25.83),
    )
    assert all(
        isinstance(value, float)
        for _, value in packet.posts.top_emotions + packet.posts.top_topics
    )


def test_evidence_prompt_dict_is_strictly_json_safe(project_data: Any) -> None:
    packet = build_evidence(
        project_data,
        _scope("win_loss_comparison"),
    )

    json.dumps(
        packet.as_prompt_dict(),
        ensure_ascii=False,
        allow_nan=False,
    )


def test_every_packet_includes_case_sample_limitation(project_data: Any) -> None:
    packet = build_evidence(project_data, _scope("win_group", "post"))

    assert any(
        "案例样本，不能代表微博总体舆情" in warning
        for warning in packet.warnings
    )
