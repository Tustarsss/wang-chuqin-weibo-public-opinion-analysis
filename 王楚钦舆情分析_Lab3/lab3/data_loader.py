"""Read and validate the fixed Lab 1/Lab 2 artifacts consumed by Lab 3."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


POLARITIES = frozenset({"positive", "negative", "neutral"})
REPORT_SECTIONS = ("meta", "posts", "comments", "domain_comparison")


class DataContractError(ValueError):
    """Raised when an upstream artifact violates the Lab 3 data contract."""


@dataclass(frozen=True)
class ProjectData:
    events: Mapping[str, Mapping[str, Any]]
    report: Mapping[str, Any]
    posts: tuple[Mapping[str, Any], ...]
    comments: tuple[Mapping[str, Any], ...]
    ingestion: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", _freeze(self.events))
        object.__setattr__(self, "report", _freeze(self.report))
        object.__setattr__(self, "posts", _freeze(self.posts))
        object.__setattr__(self, "comments", _freeze(self.comments))
        object.__setattr__(self, "ingestion", _freeze(self.ingestion))


def artifact_paths(repo_root: str | Path) -> dict[str, Path]:
    root = Path(repo_root)
    lab2_output = root / "王楚钦舆情分析_Lab2" / "01_输出"
    return {
        "events": root
        / "王楚钦舆情分析_Lab1"
        / "03_说明与配置"
        / "events.json",
        "posts": lab2_output / "posts_sentiment.jsonl",
        "comments": lab2_output / "comments_sentiment.jsonl",
        "report": lab2_output / "sentiment_report.json",
        "ingestion": root
        / "王楚钦舆情分析_Lab2"
        / "02_质量报告"
        / "lab3_ingestion_check.json",
    }


def load_project_data(repo_root: str | Path) -> ProjectData:
    paths = artifact_paths(repo_root)
    for name, path in paths.items():
        if not path.is_file():
            raise DataContractError(
                f"missing required artifact {name}: {path.name}"
            )

    event_rows = _load_json(paths["events"])
    posts = _load_jsonl(paths["posts"])
    comments = _load_jsonl(paths["comments"])
    report = _load_json(paths["report"])
    ingestion = _load_json(paths["ingestion"])

    events = _active_events(event_rows)
    emotions, topics = _report_enums(report)
    validate_ingestion(ingestion)
    validate_rows(
        posts,
        label="posts",
        id_field="post_id",
        expected_content_type="post",
        events=events,
        emotions=emotions,
        topics=topics,
    )
    validate_rows(
        comments,
        label="comments",
        id_field="comment_id",
        expected_content_type="comment",
        events=events,
        emotions=emotions,
        topics=topics,
    )
    validate_report(report, posts, comments, set(events))

    return ProjectData(
        events=events,
        report=report,
        posts=posts,
        comments=comments,
        ingestion=ingestion,
    )


def validate_rows(
    rows: Any,
    label: str,
    id_field: str,
    expected_content_type: str,
    events: Mapping[str, Mapping[str, Any]],
    emotions: frozenset[str],
    topics: frozenset[str],
) -> None:
    """Validate one post/comment collection against active Lab 1 events."""

    if not isinstance(rows, (list, tuple)):
        raise DataContractError(f"{label}: rows must be a JSON array")

    seen_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise DataContractError(
                f"{label} record at index {index}: record must be an object"
            )

        record_id = row.get(id_field)
        where = (
            f"{label} record {record_id!r}"
            if _nonempty_string(record_id)
            else f"{label} record at index {index}"
        )
        if not _nonempty_string(record_id):
            _raise_field(where, id_field, "must be a non-empty string")
        if record_id in seen_ids:
            _raise_field(where, id_field, "must be unique (duplicate ID)")
        seen_ids.add(record_id)

        event_id = row.get("event_id")
        if not _nonempty_string(event_id):
            _raise_field(where, "event_id", "must be a non-empty string")
        if event_id not in events:
            _raise_field(
                where,
                "event_id",
                f"references unknown or inactive event {event_id!r}",
            )
        event = events[event_id]
        expected_result = event["result"]
        if row.get("match_result") != expected_result:
            _raise_field(
                where,
                "match_result",
                f"must match event result {expected_result!r}",
            )
        expected_event_name = event["event_name"]
        if row.get("event_name") != expected_event_name:
            _raise_field(
                where,
                "event_name",
                f"must match active event name {expected_event_name!r}",
            )

        if row.get("content_type") != expected_content_type:
            _raise_field(
                where,
                "content_type",
                f"must be {expected_content_type!r}",
            )
        if not _nonempty_string(row.get("text_clean")):
            _raise_field(where, "text_clean", "must be a non-empty string")

        likes = row.get("likes")
        if (
            not isinstance(likes, int)
            or isinstance(likes, bool)
            or likes < 0
        ):
            _raise_field(where, "likes", "must be a non-negative integer")

        hours = row.get("hours_after_event")
        if not _number(hours) or not 0 <= float(hours) < 24:
            _raise_field(where, "hours_after_event", "must be in [0, 24)")

        if row.get("sentiment_polarity") not in POLARITIES:
            _raise_field(
                where,
                "sentiment_polarity",
                "must be positive, negative, or neutral",
            )
        category = row.get("sentiment_category")
        if not _nonempty_string(category) or category not in emotions:
            _raise_field(
                where,
                "sentiment_category",
                "must be declared in report.meta.emotions",
            )

        intensity = row.get("sentiment_intensity")
        if (
            not isinstance(intensity, int)
            or isinstance(intensity, bool)
            or not 1 <= intensity <= 5
        ):
            _raise_field(
                where,
                "sentiment_intensity",
                "must be an integer in [1, 5]",
            )

        tags = row.get("topic_tags")
        if (
            not isinstance(tags, (list, tuple))
            or not 1 <= len(tags) <= 3
            or any(not _nonempty_string(tag) for tag in tags)
            or any(tag not in topics for tag in tags)
        ):
            _raise_field(
                where,
                "topic_tags",
                "must contain 1-3 values declared in report.meta.topics",
            )

        confidence = row.get("confidence")
        if not _number(confidence) or not 0 <= float(confidence) <= 1:
            _raise_field(where, "confidence", "must be in [0, 1]")


def validate_report(
    report: Any,
    posts: list[Any] | tuple[Any, ...],
    comments: list[Any] | tuple[Any, ...],
    active_event_ids: set[str],
) -> tuple[frozenset[str], frozenset[str]]:
    """Validate report fields consumed by the Lab 3 evidence builder."""

    emotions, topics = _report_enums(report)
    meta = report["meta"]
    expected_counts = {
        "n_posts": len(posts),
        "n_comments": len(comments),
    }
    for field, expected in expected_counts.items():
        if meta.get(field) != expected:
            _report_error(
                f"report.meta.{field}",
                f"must equal {expected}",
            )

    for source_name, rows in (("posts", posts), ("comments", comments)):
        source = report[source_name]
        source_path = f"report.{source_name}"
        n_total = source.get("n_total")
        if (
            not isinstance(n_total, int)
            or isinstance(n_total, bool)
            or n_total < 0
            or n_total != len(rows)
        ):
            _report_error(
                f"{source_path}.n_total",
                f"must be the non-negative integer {len(rows)}",
            )

        by_event = _report_mapping(
            source.get("by_event"), f"{source_path}.by_event"
        )
        actual_event_ids = set(by_event)
        unknown_event_ids = actual_event_ids - active_event_ids
        if unknown_event_ids:
            bad_event_id = sorted(unknown_event_ids, key=str)[0]
            _report_error(
                f"{source_path}.by_event.{bad_event_id}",
                "must reference an active event",
            )
        expected_event_ids = {row["event_id"] for row in rows}
        if actual_event_ids != expected_event_ids:
            _report_error(
                f"{source_path}.by_event",
                "keys must exactly match events present in source rows",
            )
        for event_id, metric in by_event.items():
            _validate_event_metric(
                metric,
                f"{source_path}.by_event.{event_id}",
                emotions,
                topics,
            )

        rollups = _report_mapping(
            source.get("by_result_rollup"),
            f"{source_path}.by_result_rollup",
        )
        for result in ("win", "loss"):
            result_path = f"{source_path}.by_result_rollup.{result}"
            if result not in rollups:
                _report_error(result_path, "is required")
            _validate_rollup_metric(
                rollups[result], result_path, emotions, topics
            )

    return emotions, topics


def _report_enums(
    report: Any,
) -> tuple[frozenset[str], frozenset[str]]:
    if not isinstance(report, Mapping):
        _report_error("report", "must be an object")
    for section in REPORT_SECTIONS:
        if section not in report:
            _report_error(f"report.{section}", "is required")
        _report_mapping(report[section], f"report.{section}")

    meta = report["meta"]
    emotions = _report_string_enum(meta.get("emotions"), "emotions")
    topics = _report_string_enum(meta.get("topics"), "topics")
    return emotions, topics


def _report_string_enum(value: Any, field: str) -> frozenset[str]:
    path = f"report.meta.{field}"
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not _nonempty_string(item) for item in value)
        or len(set(value)) != len(value)
    ):
        _report_error(path, "must be a non-empty unique string enum")
    return frozenset(value)


def _validate_event_metric(
    value: Any,
    path: str,
    emotions: frozenset[str],
    topics: frozenset[str],
) -> None:
    metric = _report_mapping(value, path)
    _validate_metric_count(metric, "n", path)
    _validate_metric_number(metric, "mean_score", path)

    polarity = _report_mapping(
        metric.get("polarity_pct"), f"{path}.polarity_pct"
    )
    for polarity_name in POLARITIES:
        _validate_bounded_number(
            polarity,
            polarity_name,
            f"{path}.polarity_pct",
            0,
            100,
        )
    _validate_metric_distributions(metric, path, emotions, topics)


def _validate_rollup_metric(
    value: Any,
    path: str,
    emotions: frozenset[str],
    topics: frozenset[str],
) -> None:
    metric = _report_mapping(value, path)
    _validate_metric_count(metric, "n_total", path)
    _validate_metric_number(metric, "mean_score", path)
    for field in ("positive_pct", "negative_pct", "neutral_pct"):
        _validate_bounded_number(metric, field, path, 0, 100)
    _validate_metric_distributions(metric, path, emotions, topics)


def _validate_metric_distributions(
    metric: Mapping[str, Any],
    path: str,
    emotions: frozenset[str],
    topics: frozenset[str],
) -> None:
    emotion_dist = _report_mapping(
        metric.get("emotion_dist_pct"), f"{path}.emotion_dist_pct"
    )
    for emotion, value in emotion_dist.items():
        field_path = f"{path}.emotion_dist_pct.{emotion}"
        if emotion not in emotions:
            _report_error(field_path, "is not declared in report.meta.emotions")
        if not _number(value) or float(value) < 0:
            _report_error(field_path, "must be a finite non-negative number")

    topic_rates = _report_mapping(
        metric.get("topic_mention_rate"), f"{path}.topic_mention_rate"
    )
    for topic, value in topic_rates.items():
        field_path = f"{path}.topic_mention_rate.{topic}"
        if topic not in topics:
            _report_error(field_path, "is not declared in report.meta.topics")
        if not _number(value) or not 0 <= float(value) <= 1:
            _report_error(field_path, "must be a finite number in [0, 1]")


def _validate_metric_count(
    metric: Mapping[str, Any], field: str, path: str
) -> None:
    value = metric.get(field)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        _report_error(
            f"{path}.{field}", "must be a non-negative integer"
        )


def _validate_metric_number(
    metric: Mapping[str, Any], field: str, path: str
) -> None:
    if not _number(metric.get(field)):
        _report_error(f"{path}.{field}", "must be a finite number")


def _validate_bounded_number(
    metric: Mapping[str, Any],
    field: str,
    path: str,
    minimum: float,
    maximum: float,
) -> None:
    value = metric.get(field)
    if not _number(value) or not minimum <= float(value) <= maximum:
        _report_error(
            f"{path}.{field}",
            f"must be a finite number in [{minimum}, {maximum}]",
        )


def _report_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _report_error(path, "must be an object")
    return value


def _report_error(path: str, reason: str) -> None:
    raise DataContractError(f"{path}: {reason}")


def validate_ingestion(ingestion: Any) -> None:
    """Require the Lab 2 hand-off check to be ready and error-free."""

    if not isinstance(ingestion, Mapping):
        raise DataContractError(
            "lab3_ingestion_check.json: ingestion must be an object"
        )
    if ingestion.get("lab3_ready") is not True:
        raise DataContractError(
            "lab3_ingestion_check.json: field lab3_ready must be true"
        )
    errors = ingestion.get("errors")
    if not isinstance(errors, (list, tuple)) or errors:
        raise DataContractError(
            "lab3_ingestion_check.json: field errors must be an empty list"
        )


def _active_events(value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        raise DataContractError("events.json: top-level value must be an array")

    events: dict[str, Mapping[str, Any]] = {}
    seen_ids: set[str] = set()
    for index, event in enumerate(value, 1):
        if not isinstance(event, Mapping):
            raise DataContractError(
                f"events.json record at index {index}: record must be an object"
            )
        event_id = event.get("event_id")
        where = (
            f"events.json record {event_id!r}"
            if _nonempty_string(event_id)
            else f"events.json record at index {index}"
        )
        if not _nonempty_string(event_id):
            _raise_field(where, "event_id", "must be a non-empty string")
        if event_id in seen_ids:
            _raise_field(where, "event_id", "must be unique (duplicate ID)")
        seen_ids.add(event_id)

        active = event.get("active", True)
        if not isinstance(active, bool):
            _raise_field(where, "active", "must be a boolean")
        result = event.get("result")
        if result not in {"win", "loss"}:
            _raise_field(where, "result", "must be 'win' or 'loss'")
        if not _nonempty_string(event.get("event_name")):
            _raise_field(where, "event_name", "must be a non-empty string")
        if active:
            events[event_id] = event
    return events


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(
            f"{path.name}: could not read valid JSON: {exc}"
        ) from exc


def _load_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DataContractError(
                        f"{path.name}:{line_number}: invalid JSON: {exc}"
                    ) from exc
    except (OSError, UnicodeError) as exc:
        raise DataContractError(f"{path.name}: could not be read: {exc}") from exc
    return rows


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _raise_field(where: str, field: str, reason: str) -> None:
    raise DataContractError(f"{where}: field {field} {reason}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value
