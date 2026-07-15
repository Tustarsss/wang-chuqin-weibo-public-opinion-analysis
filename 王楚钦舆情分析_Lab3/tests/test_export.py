from __future__ import annotations

import importlib
from typing import Any


def _modules():
    return (
        importlib.import_module("lab3.offline"),
        importlib.import_module("lab3.export"),
    )


def test_export_contains_scope_evidence_strategies_and_human_note(
    loss_packet: Any,
) -> None:
    offline, export = _modules()
    brief = offline.brief_offline(loss_packet)
    strategies = offline.strategies_offline(
        loss_packet,
        goal="回应争议",
        audience="球迷",
    )

    markdown = export.export_markdown(
        loss_packet,
        brief,
        strategies,
        selected_option="事实说明与复盘",
        human_note="发布前交由教练团队核验。",
    )

    assert loss_packet.label in markdown
    assert "## 范围与来源" in markdown
    assert "## 模式" in markdown
    assert "## 事实" in markdown
    assert "## 观察" in markdown
    assert "## 决策关注" in markdown
    assert "## 局限" in markdown
    assert "## 证据" in markdown
    assert loss_packet.citations[0].record_id in markdown
    assert all(
        field in markdown
        for field in ("来源", "事件", "文本", "情绪", "主题", "置信度", "点赞")
    )
    for option in strategies.payload["options"]:
        assert option["name"] in markdown
        assert option["action"] in markdown
        assert option["timing"] in markdown
        assert all(item in markdown for item in option["benefits"])
        assert all(item in markdown for item in option["risks"])
        assert all(item in markdown for item in option["checks"])
    assert "事实说明与复盘" in markdown
    assert "发布前交由教练团队核验。" in markdown
    assert "不能代表微博总体舆情" in markdown
    assert "非预测" in markdown
    assert "<script" not in markdown.lower()


def test_export_handles_empty_evidence_and_empty_human_fields(
    zero_comment_packet: Any,
) -> None:
    offline, export = _modules()
    brief = offline.brief_offline(zero_comment_packet)
    strategies = offline.strategies_offline(
        zero_comment_packet,
        goal="准备媒体简报",
        audience="媒体",
    )

    markdown = export.export_markdown(
        zero_comment_packet,
        brief,
        strategies,
        selected_option=None,
        human_note=None,
    )

    assert "## 证据" in markdown
    assert "无可用证据" in markdown
    assert markdown.count("尚未填写") >= 2
    assert "不能代表微博总体舆情" in markdown
    assert "非预测" in markdown


def test_export_neutralizes_executable_html(loss_packet: Any) -> None:
    offline, export = _modules()
    brief = offline.brief_offline(loss_packet)
    strategies = offline.strategies_offline(
        loss_packet,
        goal="内部复盘",
        audience="团队",
    )

    markdown = export.export_markdown(
        loss_packet,
        brief,
        strategies,
        selected_option="<script>alert(1)</script>",
        human_note="<SCRIPT src=x></SCRIPT>",
    )

    assert "<script" not in markdown.lower()
    assert "&lt;script&gt;" in markdown.lower()
