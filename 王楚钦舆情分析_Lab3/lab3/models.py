from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal


ScopeKind = Literal[
    "single_event",
    "win_group",
    "loss_group",
    "win_loss_comparison",
]
SourceKind = Literal["both", "post", "comment"]


@dataclass(frozen=True)
class AnalysisScope:
    kind: ScopeKind
    source: SourceKind
    audience: str
    event_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "single_event" and not self.event_id:
            raise ValueError("event_id is required for a single_event scope")
        if self.kind != "single_event" and self.event_id is not None:
            raise ValueError("event_id is only valid for a single_event scope")


@dataclass(frozen=True)
class MetricSummary:
    n: int
    mean_score: float | None
    polarity_pct: Mapping[str, float]
    top_emotions: tuple[tuple[str, int], ...]
    top_topics: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Citation:
    record_id: str
    content_type: Literal["post", "comment"]
    event_id: str
    event_name: str
    text: str
    polarity: str
    emotion: str
    topics: tuple[str, ...]
    confidence: float
    likes: int


@dataclass(frozen=True)
class EvidencePacket:
    label: str
    scope: AnalysisScope
    posts: MetricSummary | None
    comments: MetricSummary | None
    citations: tuple[Citation, ...]
    warnings: tuple[str, ...]
    facts: Mapping[str, Any]

    def as_prompt_dict(self) -> dict[str, Any]:
        prompt_data = _as_json_safe(self)
        if not isinstance(prompt_data, dict):
            raise TypeError("EvidencePacket must convert to a dictionary")
        return prompt_data


@dataclass(frozen=True)
class GeneratedResult:
    payload: Mapping[str, Any]
    mode: Literal["online", "offline"]
    warning: str | None = None


def _as_json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _as_json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _as_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON-safe")
