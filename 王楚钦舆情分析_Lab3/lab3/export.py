"""Render offline decision-support results as inert Markdown text."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import EvidencePacket, GeneratedResult


_SOURCE_LABELS = {
    "both": "正文与评论",
    "post": "正文",
    "comment": "评论",
}


def export_markdown(
    packet: EvidencePacket,
    brief: GeneratedResult,
    strategies: GeneratedResult,
    selected_option: str | None = "",
    human_note: str | None = "",
) -> str:
    """Return a complete Markdown report without executing embedded HTML."""

    limitations = tuple(
        dict.fromkeys(
            packet.warnings
            + _as_items(brief.payload.get("limitations"))
        )
    )
    lines = [
        f"# {_md_text(brief.payload.get('title') or packet.label)}",
        "",
        "## 范围与来源",
        "",
        f"- 分析范围：{_md_text(packet.label)}",
        f"- 范围类型：{_md_text(packet.scope.kind)}",
        f"- 来源：{_md_text(_SOURCE_LABELS.get(packet.scope.source, packet.scope.source))}",
        f"- 受众：{_md_text(packet.scope.audience)}",
        "",
        "## 模式",
        "",
        f"- 简报生成模式：{_md_text(brief.mode)}",
        f"- 策略生成模式：{_md_text(strategies.mode)}",
        "",
        "## 事实",
        "",
    ]
    _append_items(lines, packet.facts)

    lines.extend(("", "## 观察", ""))
    _append_items(lines, brief.payload.get("observations"))

    lines.extend(("", "## 决策关注", ""))
    _append_items(lines, brief.payload.get("decision_focus"))

    lines.extend(("", "## 局限", ""))
    _append_items(lines, limitations)

    lines.extend(("", "## 证据", ""))
    if not packet.citations:
        lines.append("- 无可用证据。")
    for citation in packet.citations:
        topics = "、".join(citation.topics) if citation.topics else "未标注"
        lines.extend(
            (
                f"### 证据 {_md_text(citation.record_id)}",
                "",
                f"- ID：{_md_text(citation.record_id)}",
                f"- 来源：{_md_text(_SOURCE_LABELS[citation.content_type])}",
                (
                    f"- 事件：{_md_text(citation.event_name)}"
                    f"（{_md_text(citation.event_id)}）"
                ),
                f"- 文本：{_md_text(citation.text)}",
                (
                    f"- 情绪：{_md_text(citation.emotion)}；"
                    f"极性：{_md_text(citation.polarity)}"
                ),
                f"- 主题：{_md_text(topics)}",
                f"- 置信度：{_md_text(citation.confidence)}",
                f"- 点赞：{_md_text(citation.likes)}",
                "",
            )
        )

    lines.extend(
        (
            "## 三方案",
            "",
            f"- 目标：{_md_text(strategies.payload.get('goal'))}",
            f"- 受众：{_md_text(strategies.payload.get('audience'))}",
            "",
        )
    )
    options = strategies.payload.get("options", ())
    for option in _mapping_items(options):
        lines.extend(
            (
                f"### {_md_text(option.get('name'))}",
                "",
                f"- 力度：{_md_text(option.get('intensity'))}",
                f"- 行动：{_md_text(option.get('action'))}",
                f"- 时机：{_md_text(option.get('timing'))}",
                (
                    "- 证据 ID："
                    + _joined_values(option.get("evidence_ids"), "无可用证据")
                ),
                "- 收益：",
            )
        )
        _append_items(lines, option.get("benefits"), indent="  ")
        lines.append("- 风险：")
        _append_items(lines, option.get("risks"), indent="  ")
        lines.append("- 核验：")
        _append_items(lines, option.get("checks"), indent="  ")
        lines.append("")

    lines.extend(
        (
            "- 方案声明："
            + _md_text(strategies.payload.get("disclaimer")),
            "",
            "## 人工选择与备注",
            "",
            f"- 人工选择：{_human_value(selected_option)}",
            f"- 人工备注：{_human_value(human_note)}",
            "",
            "## 声明",
            "",
            "- 本报告基于案例样本，不能代表微博总体舆情。",
            "- 所列策略仅作定性情景比较，非预测；最终由人工决定并复核。",
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def _append_items(
    lines: list[str],
    value: Any,
    *,
    indent: str = "",
) -> None:
    items = _as_items(value)
    if not items:
        lines.append(f"{indent}- 无可用内容。")
        return
    lines.extend(f"{indent}- {_md_text(item)}" for item in items)


def _as_items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Sequence):
        return tuple(item for item in value if item is not None)
    return (value,)


def _mapping_items(value: Any) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        item
        for item in _as_items(value)
        if isinstance(item, Mapping)
    )


def _joined_values(value: Any, empty: str) -> str:
    items = _as_items(value)
    if not items:
        return empty
    return "、".join(_md_text(item) for item in items)


def _human_value(value: Any) -> str:
    if value is None or not str(value).strip():
        return "尚未填写"
    return _md_text(value)


def _md_text(value: Any) -> str:
    """Collapse newlines and neutralize raw HTML/active Markdown links."""

    if value is None:
        return "无可用内容"
    text = " ".join(str(value).split())
    replacements = (
        ("&", "&amp;"),
        ("<", "&lt;"),
        (">", "&gt;"),
        ("`", "&#96;"),
        ("[", "&#91;"),
        ("]", "&#93;"),
        ("(", "&#40;"),
        (")", "&#41;"),
    )
    for raw, escaped in replacements:
        text = text.replace(raw, escaped)
    return text
