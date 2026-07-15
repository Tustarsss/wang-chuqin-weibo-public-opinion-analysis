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
    events: tuple[Mapping[str, Any], ...]
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

    events, event_results = _active_events(event_rows)
    validate_ingestion(ingestion)
    validate_rows(
        posts,
        label="posts",
        id_field="post_id",
        expected_content_type="post",
        event_results=event_results,
    )
    validate_rows(
        comments,
        label="comments",
        id_field="comment_id",
        expected_content_type="comment",
        event_results=event_results,
    )
    validate_report(report, len(posts), len(comments))

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
    event_results: Mapping[str, str],
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
        if event_id not in event_results:
            _raise_field(
                where,
                "event_id",
                f"references unknown or inactive event {event_id!r}",
            )
        expected_result = event_results[event_id]
        if row.get("match_result") != expected_result:
            _raise_field(
                where,
                "match_result",
                f"must match event result {expected_result!r}",
            )

        if row.get("content_type") != expected_content_type:
            _raise_field(
                where,
                "content_type",
                f"must be {expected_content_type!r}",
            )
        if not _nonempty_string(row.get("text_clean")):
            _raise_field(where, "text_clean", "must be a non-empty string")

        hours = row.get("hours_after_event")
        if not _number(hours) or not 0 <= float(hours) < 24:
            _raise_field(where, "hours_after_event", "must be in [0, 24)")

        if row.get("sentiment_polarity") not in POLARITIES:
            _raise_field(
                where,
                "sentiment_polarity",
                "must be positive, negative, or neutral",
            )
        if not _nonempty_string(row.get("sentiment_category")):
            _raise_field(
                where,
                "sentiment_category",
                "must be a non-empty string",
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
        ):
            _raise_field(
                where,
                "topic_tags",
                "must contain 1-3 non-empty strings",
            )

        confidence = row.get("confidence")
        if not _number(confidence) or not 0 <= float(confidence) <= 1:
            _raise_field(where, "confidence", "must be in [0, 1]")


def validate_report(
    report: Any,
    post_count: int,
    comment_count: int,
) -> None:
    """Validate the aggregate report sections and source row counts."""

    if not isinstance(report, Mapping):
        raise DataContractError("sentiment_report.json: report must be an object")
    for section in REPORT_SECTIONS:
        if section not in report:
            raise DataContractError(
                "sentiment_report.json: "
                f"missing top-level field {section}"
            )
        if not isinstance(report[section], Mapping):
            raise DataContractError(
                "sentiment_report.json: "
                f"field {section} must be an object"
            )

    meta = report["meta"]
    expected_counts = {
        "n_posts": post_count,
        "n_comments": comment_count,
    }
    for field, expected in expected_counts.items():
        if meta.get(field) != expected:
            raise DataContractError(
                "sentiment_report.json: "
                f"field meta.{field} must equal {expected}"
            )


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


def _active_events(value: Any) -> tuple[list[Mapping[str, Any]], dict[str, str]]:
    if not isinstance(value, list):
        raise DataContractError("events.json: top-level value must be an array")

    events: list[Mapping[str, Any]] = []
    event_results: dict[str, str] = {}
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
        if active:
            events.append(event)
            event_results[event_id] = result
    return events, event_results


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
