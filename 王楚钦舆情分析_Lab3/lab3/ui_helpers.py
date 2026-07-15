"""Pure presentation helpers for the Lab 3 Streamlit interface."""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from types import MappingProxyType
from typing import Any

from .models import AnalysisScope, Citation, EvidencePacket, MetricSummary


SCOPE_LABELS = MappingProxyType(
    {
        "single_event": "单场",
        "win_group": "胜组",
        "loss_group": "负组",
        "win_loss_comparison": "胜负对比",
    }
)
SOURCE_LABELS = MappingProxyType(
    {
        "both": "正文与评论",
        "post": "正文",
        "comment": "评论",
    }
)
AUDIENCE_LABELS = MappingProxyType(
    {
        "fans": "球迷",
        "media": "媒体",
        "team": "运动队内部",
    }
)

SCOPE_VALUES = MappingProxyType(
    {label: value for value, label in SCOPE_LABELS.items()}
)
SOURCE_VALUES = MappingProxyType(
    {label: value for value, label in SOURCE_LABELS.items()}
)
AUDIENCE_VALUES = MappingProxyType(
    {label: label for label in AUDIENCE_LABELS.values()}
)

# Readable aliases for callers that prefer "by label" terminology.
SCOPE_BY_LABEL = SCOPE_VALUES
SOURCE_BY_LABEL = SOURCE_VALUES
AUDIENCE_BY_LABEL = AUDIENCE_VALUES

_CONTEXT_VALUES = (
    "messages",
    "brief",
    "strategies",
    "human_choice",
    "human_note",
)


def context_key(scope: AnalysisScope) -> str:
    """Return a deterministic key covering every user-selectable scope field."""

    return json.dumps(
        {
            "kind": scope.kind,
            "source": scope.source,
            "audience": scope.audience,
            "event_id": scope.event_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def synchronize_context(
    state: MutableMapping[str, Any],
    new_key: str,
) -> bool:
    """Clear generated UI state exactly when the analysis context changes."""

    if state.get("context_key") == new_key:
        return False
    for key in _CONTEXT_VALUES:
        state.pop(key, None)
    state["context_key"] = new_key
    return True


def event_options(data: Any) -> tuple[tuple[str, str], ...]:
    """Return active events as ``(Chinese display label, event_id)`` pairs."""

    rows = sorted(
        data.events.items(),
        key=lambda item: (str(item[1]["match_date"]), item[0]),
    )
    result_labels = {"win": "胜", "loss": "负"}
    return tuple(
        (
            (
                f"{event['match_date']}｜{event['event_name']}｜"
                f"{result_labels.get(event['result'], event['result'])}"
            ),
            event_id,
        )
        for event_id, event in rows
        if event.get("active", True)
    )


def metric_rows(packet: EvidencePacket) -> list[dict[str, Any]]:
    """Build source-separated rows without aggregating posts and comments."""

    rows: list[dict[str, Any]] = []
    if packet.scope.kind == "win_loss_comparison":
        if packet.post_comparison is not None:
            rows.append(
                _metric_row("胜组", "正文", packet.post_comparison.win)
            )
            rows.append(
                _metric_row("负组", "正文", packet.post_comparison.loss)
            )
        if packet.comment_comparison is not None:
            rows.append(
                _metric_row("胜组", "评论", packet.comment_comparison.win)
            )
            rows.append(
                _metric_row("负组", "评论", packet.comment_comparison.loss)
            )
        return rows

    group = {
        "win_group": "胜组",
        "loss_group": "负组",
        "single_event": packet.label,
    }[packet.scope.kind]
    if packet.posts is not None:
        rows.append(_metric_row(group, "正文", packet.posts))
    if packet.comments is not None:
        rows.append(_metric_row(group, "评论", packet.comments))
    return rows


def citation_lookup(
    packet: EvidencePacket,
    citation_id: str,
) -> Citation | None:
    """Find the first matching citation within the current packet only."""

    return next(
        (
            citation
            for citation in packet.citations
            if citation.record_id == citation_id
        ),
        None,
    )


def _metric_row(
    group: str,
    source: str,
    summary: MetricSummary,
) -> dict[str, Any]:
    polarity = summary.polarity_pct
    return {
        "分组": group,
        "来源": source,
        "样本量": summary.n,
        "平均情感分": summary.mean_score,
        "正面%": polarity["positive"],
        "中性%": polarity["neutral"],
        "负面%": polarity["negative"],
    }
