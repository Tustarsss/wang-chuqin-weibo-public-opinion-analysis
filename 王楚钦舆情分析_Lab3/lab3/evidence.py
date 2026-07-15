"""Build deterministic, traceable evidence packets from validated artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .data_loader import ProjectData
from .models import (
    AnalysisScope,
    Citation,
    EvidencePacket,
    MetricSummary,
    ResultComparison,
)


_Source = Literal["post", "comment"]
_SOURCE_DETAILS: dict[_Source, tuple[str, str, str]] = {
    "post": ("posts", "post_id", "正文"),
    "comment": ("comments", "comment_id", "评论"),
}
_LIMITATION = "局限：案例样本，不能代表微博总体舆情。"


def scope_event_ids(
    data: ProjectData,
    scope: AnalysisScope,
) -> tuple[str, ...]:
    """Return active event IDs covered by ``scope`` in source order."""

    if scope.kind == "single_event":
        event_id = scope.event_id
        if event_id not in data.events:
            raise ValueError(f"unknown active event_id: {event_id!r}")
        return (event_id,)

    if scope.kind == "win_loss_comparison":
        return tuple(data.events)

    result_by_kind = {
        "win_group": "win",
        "loss_group": "loss",
    }
    result = result_by_kind.get(scope.kind)
    if result is None:
        raise ValueError(f"unsupported scope kind: {scope.kind!r}")
    return tuple(
        event_id
        for event_id, event in data.events.items()
        if event["result"] == result
    )


def build_evidence(
    data: ProjectData,
    scope: AnalysisScope,
) -> EvidencePacket:
    """Build a deterministic packet without mixing posts and comments."""

    event_ids = scope_event_ids(data, scope)
    sources = _selected_sources(scope.source)
    summaries: dict[_Source, tuple[MetricSummary, ...]] = {}
    facts: list[str] = []

    posts: MetricSummary | None = None
    comments: MetricSummary | None = None
    post_comparison: ResultComparison | None = None
    comment_comparison: ResultComparison | None = None

    if scope.kind == "win_loss_comparison":
        for source in sources:
            comparison = _build_comparison(data, source)
            summaries[source] = (comparison.win, comparison.loss)
            facts.extend(_comparison_facts(source, comparison))
            if source == "post":
                post_comparison = comparison
            else:
                comment_comparison = comparison
    else:
        for source in sources:
            summary = _build_summary(data, scope, source)
            summaries[source] = (summary,)
            facts.append(_summary_fact(source, summary))
            if source == "post":
                posts = summary
            else:
                comments = summary

    citations = select_citations(
        data,
        event_ids,
        sources,
        summaries,
    )
    warnings = _coverage_warnings(data, event_ids, sources) + (_LIMITATION,)

    return EvidencePacket(
        label=_scope_label(data, scope, event_ids),
        scope=scope,
        posts=posts,
        comments=comments,
        citations=citations,
        warnings=warnings,
        facts=tuple(facts),
        post_comparison=post_comparison,
        comment_comparison=comment_comparison,
    )


def select_citations(
    data: ProjectData,
    event_ids: Sequence[str],
    sources: Sequence[_Source],
    summaries: Mapping[_Source, Sequence[MetricSummary]],
) -> tuple[Citation, ...]:
    """Select at most eight scoped citations with deterministic coverage."""

    selected_rows: list[Mapping[str, Any]] = []
    selected_ids: set[tuple[str, str]] = set()
    per_source_limit = 4 if len(sources) == 2 else 8
    event_id_set = set(event_ids)

    for source in sources:
        rows = [
            row
            for row in _source_rows(data, source)
            if row["event_id"] in event_id_set
        ]
        ranked = sorted(rows, key=_selection_key)
        source_selected: list[Mapping[str, Any]] = []

        def add_best(predicate: Any) -> None:
            if len(source_selected) >= per_source_limit:
                return
            for row in ranked:
                identity = _row_identity(row)
                if identity in selected_ids or not predicate(row):
                    continue
                source_selected.append(row)
                selected_ids.add(identity)
                return

        add_best(lambda row: row["sentiment_polarity"] == "positive")
        add_best(lambda row: row["sentiment_polarity"] == "negative")
        add_best(lambda row: True)

        for topic in _top_topic_names(summaries.get(source, ())):
            add_best(lambda row, topic=topic: topic in row["topic_tags"])

        for row in ranked:
            if len(source_selected) >= per_source_limit:
                break
            identity = _row_identity(row)
            if identity not in selected_ids:
                source_selected.append(row)
                selected_ids.add(identity)

        selected_rows.extend(source_selected)

    selected_rows.sort(key=_display_key)
    return tuple(_citation_from_row(row) for row in selected_rows[:8])


def _selected_sources(source: str) -> tuple[_Source, ...]:
    if source == "both":
        return ("post", "comment")
    if source == "post":
        return ("post",)
    if source == "comment":
        return ("comment",)
    raise ValueError(f"unsupported source: {source!r}")


def _build_summary(
    data: ProjectData,
    scope: AnalysisScope,
    source: _Source,
) -> MetricSummary:
    report_name = _SOURCE_DETAILS[source][0]
    report_source = data.report[report_name]
    if scope.kind == "single_event":
        metric = report_source["by_event"].get(scope.event_id)
        return _metric_summary(metric, rollup=False)

    result = "win" if scope.kind == "win_group" else "loss"
    metric = report_source["by_result_rollup"][result]
    return _metric_summary(metric, rollup=True)


def _build_comparison(
    data: ProjectData,
    source: _Source,
) -> ResultComparison:
    report_name = _SOURCE_DETAILS[source][0]
    rollups = data.report[report_name]["by_result_rollup"]
    return ResultComparison(
        win=_metric_summary(rollups["win"], rollup=True),
        loss=_metric_summary(rollups["loss"], rollup=True),
    )


def _metric_summary(
    metric: Mapping[str, Any] | None,
    *,
    rollup: bool,
) -> MetricSummary:
    if metric is None:
        return _empty_summary()

    count_field = "n_total" if rollup else "n"
    n = int(metric[count_field])
    if n == 0:
        return _empty_summary()

    if rollup:
        polarity = {
            "positive": float(metric["positive_pct"]),
            "neutral": float(metric["neutral_pct"]),
            "negative": float(metric["negative_pct"]),
        }
    else:
        event_polarity = metric["polarity_pct"]
        polarity = {
            "positive": float(event_polarity["positive"]),
            "neutral": float(event_polarity["neutral"]),
            "negative": float(event_polarity["negative"]),
        }

    return MetricSummary(
        n=n,
        mean_score=float(metric["mean_score"]),
        polarity_pct=polarity,
        top_emotions=_rank_percentages(metric["emotion_dist_pct"]),
        top_topics=_rank_percentages(
            metric["topic_mention_rate"],
            multiplier=100.0,
        ),
    )


def _empty_summary() -> MetricSummary:
    return MetricSummary(
        n=0,
        mean_score=None,
        polarity_pct={
            "positive": 0.0,
            "neutral": 0.0,
            "negative": 0.0,
        },
        top_emotions=(),
        top_topics=(),
    )


def _rank_percentages(
    values: Mapping[str, Any],
    *,
    multiplier: float = 1.0,
) -> tuple[tuple[str, float], ...]:
    ranked = [
        (name, float(round(float(value) * multiplier, 4)))
        for name, value in values.items()
        if float(value) > 0.0
    ]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return tuple(ranked[:3])


def _coverage_warnings(
    data: ProjectData,
    event_ids: Sequence[str],
    sources: Sequence[_Source],
) -> tuple[str, ...]:
    warnings: list[str] = []
    for source in sources:
        rows = _source_rows(data, source)
        counts: dict[str, int] = {event_id: 0 for event_id in event_ids}
        for row in rows:
            event_id = row["event_id"]
            if event_id in counts:
                counts[event_id] += 1

        source_label = _SOURCE_DETAILS[source][2]
        for event_id in event_ids:
            n = counts[event_id]
            event_name = data.events[event_id]["event_name"]
            if n == 0:
                warnings.append(
                    f"{event_name}的{source_label}零记录，无法形成该来源指标或引文。"
                )
            elif n < 3:
                warnings.append(
                    f"{event_name}的{source_label}少于 3 条（n={n}），结论仅供案例参考。"
                )
    return tuple(warnings)


def _summary_fact(source: _Source, summary: MetricSummary) -> str:
    source_label = _SOURCE_DETAILS[source][2]
    polarity = summary.polarity_pct
    if summary.n == 0:
        return (
            f"{source_label}：无可用记录；n=0，平均分=None，"
            "极性为正面 0.0000%、中性 0.0000%、负面 0.0000%。"
        )
    return (
        f"{source_label}：n={summary.n}，平均分={summary.mean_score:.4f}，"
        f"极性为正面 {polarity['positive']:.4f}%、"
        f"中性 {polarity['neutral']:.4f}%、"
        f"负面 {polarity['negative']:.4f}%。"
    )


def _comparison_facts(
    source: _Source,
    comparison: ResultComparison,
) -> tuple[str, ...]:
    source_label = _SOURCE_DETAILS[source][2]
    win = comparison.win
    loss = comparison.loss
    differences = {
        name: win.polarity_pct[name] - loss.polarity_pct[name]
        for name in ("positive", "neutral", "negative")
    }
    mean_difference = _safe_difference(win.mean_score, loss.mean_score)
    return (
        _group_fact(source_label, "胜组", win),
        _group_fact(source_label, "负组", loss),
        (
            f"{source_label}胜负描述性差异（胜组减负组）："
            f"平均分 {mean_difference}；"
            f"正面 {differences['positive']:+.4f} 个百分点、"
            f"中性 {differences['neutral']:+.4f} 个百分点、"
            f"负面 {differences['negative']:+.4f} 个百分点；"
            "仅描述样本差异，不作因果解释。"
        ),
    )


def _group_fact(
    source_label: str,
    group_label: str,
    summary: MetricSummary,
) -> str:
    if summary.n == 0:
        return f"{source_label}{group_label}：无可用记录；n=0，平均分=None。"
    polarity = summary.polarity_pct
    return (
        f"{source_label}{group_label}：n={summary.n}，"
        f"平均分={summary.mean_score:.4f}，"
        f"极性为正面 {polarity['positive']:.4f}%、"
        f"中性 {polarity['neutral']:.4f}%、"
        f"负面 {polarity['negative']:.4f}%。"
    )


def _safe_difference(win: float | None, loss: float | None) -> str:
    if win is None or loss is None:
        return "无可用值"
    return f"{win - loss:+.4f}"


def _top_topic_names(
    summaries: Sequence[MetricSummary],
) -> tuple[str, ...]:
    names: list[str] = []
    for summary in summaries:
        for name, _ in summary.top_topics:
            if name not in names:
                names.append(name)
    return tuple(names)


def _source_rows(
    data: ProjectData,
    source: _Source,
) -> tuple[Mapping[str, Any], ...]:
    return data.posts if source == "post" else data.comments


def _selection_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    confidence_bucket = 0 if float(row["confidence"]) >= 0.6 else 1
    return (
        confidence_bucket,
        -int(row["likes"]),
        -int(row["sentiment_intensity"]),
        _record_id(row),
    )


def _display_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        -int(row["likes"]),
        -int(row["sentiment_intensity"]),
        _record_id(row),
    )


def _row_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["content_type"]), _record_id(row)


def _record_id(row: Mapping[str, Any]) -> str:
    id_field = "post_id" if row["content_type"] == "post" else "comment_id"
    return str(row[id_field])


def _citation_from_row(row: Mapping[str, Any]) -> Citation:
    return Citation(
        record_id=_record_id(row),
        content_type=row["content_type"],
        event_id=row["event_id"],
        event_name=row["event_name"],
        text=row["text_clean"],
        polarity=row["sentiment_polarity"],
        emotion=row["sentiment_category"],
        topics=tuple(row["topic_tags"]),
        confidence=float(row["confidence"]),
        likes=int(row["likes"]),
    )


def _scope_label(
    data: ProjectData,
    scope: AnalysisScope,
    event_ids: Sequence[str],
) -> str:
    if scope.kind == "single_event":
        return str(data.events[event_ids[0]]["event_name"])
    return {
        "win_group": "胜场组舆情",
        "loss_group": "负场组舆情",
        "win_loss_comparison": "胜负舆情对比",
    }[scope.kind]
