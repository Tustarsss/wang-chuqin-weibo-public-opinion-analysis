from __future__ import annotations

from lab3.models import AnalysisScope
from lab3.ui_helpers import (
    AUDIENCE_LABELS,
    AUDIENCE_VALUES,
    SCOPE_LABELS,
    SCOPE_VALUES,
    SOURCE_LABELS,
    SOURCE_VALUES,
    citation_lookup,
    context_key,
    event_options,
    metric_chart_rows,
    metric_rows,
    synchronize_context,
)


def test_selection_constants_have_chinese_labels_and_reverse_values():
    assert SCOPE_LABELS == {
        "single_event": "单场",
        "win_group": "胜组",
        "loss_group": "负组",
        "win_loss_comparison": "胜负对比",
    }
    assert SOURCE_LABELS == {
        "both": "正文与评论",
        "post": "正文",
        "comment": "评论",
    }
    assert tuple(AUDIENCE_LABELS.values()) == ("球迷", "媒体", "运动队内部")
    assert SCOPE_VALUES == {label: value for value, label in SCOPE_LABELS.items()}
    assert SOURCE_VALUES == {label: value for value, label in SOURCE_LABELS.items()}
    assert AUDIENCE_VALUES == {
        label: label for label in AUDIENCE_LABELS.values()
    }


def test_context_key_is_stable_and_covers_every_scope_dimension():
    base = AnalysisScope(
        kind="single_event",
        source="both",
        audience="球迷",
        event_id="event-a",
    )

    assert context_key(base) == context_key(base)
    variants = (
        AnalysisScope(kind="loss_group", source="both", audience="球迷"),
        AnalysisScope(
            kind="single_event",
            source="post",
            audience="球迷",
            event_id="event-a",
        ),
        AnalysisScope(
            kind="single_event",
            source="both",
            audience="媒体",
            event_id="event-a",
        ),
        AnalysisScope(
            kind="single_event",
            source="both",
            audience="球迷",
            event_id="event-b",
        ),
    )
    assert all(context_key(base) != context_key(scope) for scope in variants)


def test_synchronize_context_clears_only_context_values_when_key_changes():
    state = {
        "context_key": "old",
        "messages": [{"role": "user", "content": "旧问题"}],
        "brief": object(),
        "strategies": object(),
        "human_choice": "方案一",
        "human_note": "旧备注",
        "loaded_data": object(),
        "sidebar_choice": "保留",
    }

    assert synchronize_context(state, "new") is True
    assert state["context_key"] == "new"
    assert state["sidebar_choice"] == "保留"
    assert "loaded_data" in state
    for key in (
        "messages",
        "brief",
        "strategies",
        "human_choice",
        "human_note",
    ):
        assert key not in state


def test_synchronize_context_does_nothing_for_same_key():
    state = {
        "context_key": "same",
        "messages": ["keep"],
        "brief": "keep",
        "strategies": "keep",
        "human_choice": "keep",
        "human_note": "keep",
    }
    before = state.copy()

    assert synchronize_context(state, "same") is False
    assert state == before


def test_event_options_are_all_active_events_in_date_order(project_data):
    options = event_options(project_data)

    assert len(options) == 8
    assert [event_id for _label, event_id in options] == [
        event_id
        for event_id, _event in sorted(
            project_data.events.items(),
            key=lambda item: (item[1]["match_date"], item[0]),
        )
    ]
    for label, event_id in options:
        event = project_data.events[event_id]
        assert event["match_date"] in label
        assert event["event_name"] in label
        assert ("胜" if event["result"] == "win" else "负") in label


def test_metric_rows_keep_loss_posts_and_comments_separate(loss_packet):
    rows = metric_rows(loss_packet)

    assert [(row["来源"], row["样本量"]) for row in rows] == [
        ("正文", 22),
        ("评论", 31),
    ]
    assert {row["分组"] for row in rows} == {"负组"}
    assert all(row["来源"] != "正文与评论" for row in rows)
    required = {
        "分组",
        "来源",
        "样本量",
        "平均情感分",
        "正面%",
        "中性%",
        "负面%",
    }
    assert all(required <= set(row) for row in rows)


def test_metric_chart_rows_use_compact_source_labels_for_single_event(
    project_data,
):
    from lab3.evidence import build_evidence

    packet = build_evidence(
        project_data,
        AnalysisScope(
            kind="single_event",
            source="both",
            audience="球迷",
            event_id="loss_20240731_paris_moregard",
        ),
    )
    rows = metric_chart_rows(packet)

    assert [row["样本"] for row in rows] == ["正文（n=5）", "评论（n=10）"]
    assert all(packet.label not in row["样本"] for row in rows)
    assert rows[0]["正面%"] + rows[0]["中性%"] + rows[0]["负面%"] == 100


def test_metric_chart_rows_keep_groups_visible_for_comparison(
    comparison_packet,
):
    rows = metric_chart_rows(comparison_packet)

    assert [row["样本"] for row in rows] == [
        "胜组｜正文（n=23）",
        "负组｜正文（n=22）",
        "胜组｜评论（n=30）",
        "负组｜评论（n=31）",
    ]


def test_metric_rows_keep_comparison_groups_and_sources_separate(
    comparison_packet,
):
    rows = metric_rows(comparison_packet)

    assert [
        (row["来源"], row["分组"], row["样本量"]) for row in rows
    ] == [
        ("正文", "胜组", 23),
        ("正文", "负组", 22),
        ("评论", "胜组", 30),
        ("评论", "负组", 31),
    ]
    assert all(row["来源"] != "正文与评论" for row in rows)


def test_citation_lookup_is_limited_to_packet_citations(loss_packet):
    expected = loss_packet.citations[0]

    assert citation_lookup(loss_packet, expected.record_id) is expected
    assert citation_lookup(loss_packet, "not-in-this-packet") is None
