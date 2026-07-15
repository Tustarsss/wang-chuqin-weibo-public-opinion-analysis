from __future__ import annotations

from pathlib import Path

from lab3.data_loader import load_project_data
from lab3.evidence import build_evidence
from lab3.export import export_markdown
from lab3.llm_client import DeepSeekClient
from lab3.models import AnalysisScope
from lab3.services import BriefService, QAService, StrategyService


def _client_without_key(
    tmp_path: Path,
    monkeypatch,
) -> DeepSeekClient:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = DeepSeekClient(repo_root=tmp_path)
    assert client.api_key is None
    return client


def test_real_loss_group_pipeline_falls_back_offline_and_exports(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    data = load_project_data(repo_root)
    packet = build_evidence(
        data,
        AnalysisScope(
            kind="loss_group",
            source="both",
            audience="球迷",
        ),
    )
    client = _client_without_key(tmp_path, monkeypatch)

    brief = BriefService(client).generate(packet)
    answer = QAService(client).answer(
        "loss_all_negative",
        packet,
        preset=True,
    )
    strategies = StrategyService(client).generate(
        packet,
        goal="回应争议",
        audience="球迷",
    )

    assert brief.mode == answer.mode == strategies.mode == "offline"
    assert answer.payload["answerable"] is True
    assert "输球不等于全部负面" in answer.payload["interpretation"]
    assert len(strategies.payload["options"]) == 3
    for result in (brief, answer, strategies):
        limitations = result.payload["limitations"]
        assert any("不能代表微博总体舆情" in item for item in limitations)
        assert any("少于 3 条" in item and "n=1" in item for item in limitations)

    human_note = "发布前交由教练团队复核。"
    markdown = export_markdown(
        packet,
        brief,
        strategies,
        selected_option="事实说明与复盘",
        human_note=human_note,
    )

    assert human_note in markdown
    assert "## 证据" in markdown
    assert packet.citations[0].record_id in markdown


def test_real_win_loss_comparison_supports_offline_source_difference(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    data = load_project_data(repo_root)
    packet = build_evidence(
        data,
        AnalysisScope(
            kind="win_loss_comparison",
            source="both",
            audience="媒体",
        ),
    )

    assert packet.post_comparison is not None
    assert packet.comment_comparison is not None
    assert packet.post_comparison.win.n == 23
    assert packet.post_comparison.loss.n == 22
    assert packet.comment_comparison.win.n == 30
    assert packet.comment_comparison.loss.n == 31

    answer = QAService(_client_without_key(tmp_path, monkeypatch)).answer(
        "source_difference",
        packet,
        preset=True,
    )

    assert answer.mode == "offline"
    assert answer.payload["answerable"] is True
    assert all(
        label in answer.payload["interpretation"]
        for label in ("正文胜组", "正文负组", "评论胜组", "评论负组")
    )
