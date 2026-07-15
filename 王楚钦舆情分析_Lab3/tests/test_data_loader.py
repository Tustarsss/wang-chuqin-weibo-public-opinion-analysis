from __future__ import annotations

import importlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Callable

import pytest


JsonMutator = Callable[[Any], None]


def _copy_artifacts(module: Any, repo_root: Path, target_root: Path) -> dict[str, Path]:
    source_paths = module.artifact_paths(repo_root)
    target_paths = module.artifact_paths(target_root)
    for name, source in source_paths.items():
        target = target_paths[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return target_paths


def _mutate_json(path: Path, mutator: JsonMutator) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _mutate_jsonl(path: Path, mutator: JsonMutator) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mutator(rows)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_data_loader_exposes_contract_api() -> None:
    module = importlib.import_module("lab3.data_loader")

    assert issubclass(module.DataContractError, ValueError)
    assert module.ProjectData.__dataclass_params__.frozen is True
    assert callable(module.artifact_paths)
    assert callable(module.load_project_data)


def test_artifact_paths_are_fixed(repo_root: Path) -> None:
    module = importlib.import_module("lab3.data_loader")

    assert module.artifact_paths(repo_root) == {
        "events": repo_root
        / "王楚钦舆情分析_Lab1"
        / "03_说明与配置"
        / "events.json",
        "posts": repo_root
        / "王楚钦舆情分析_Lab2"
        / "01_输出"
        / "posts_sentiment.jsonl",
        "comments": repo_root
        / "王楚钦舆情分析_Lab2"
        / "01_输出"
        / "comments_sentiment.jsonl",
        "report": repo_root
        / "王楚钦舆情分析_Lab2"
        / "01_输出"
        / "sentiment_report.json",
        "ingestion": repo_root
        / "王楚钦舆情分析_Lab2"
        / "02_质量报告"
        / "lab3_ingestion_check.json",
    }


def test_project_data_defensively_freezes_nested_inputs() -> None:
    module = importlib.import_module("lab3.data_loader")
    events = {
        "event-1": {"event_id": "event-1", "keywords": ["原关键词"]}
    }
    report = {"meta": {"n_posts": 1}}
    posts = [{"post_id": "post-1", "topic_tags": ["赛果"]}]
    comments = [{"comment_id": "comment-1"}]
    ingestion = {"lab3_ready": True, "errors": []}

    data = module.ProjectData(events, report, posts, comments, ingestion)
    events["event-1"]["keywords"].append("污染")
    events["event-2"] = {"event_id": "event-2"}
    report["meta"]["n_posts"] = 99
    posts[0]["topic_tags"][0] = "污染"
    comments.clear()
    ingestion["errors"].append("污染")

    assert data.events["event-1"]["keywords"] == ("原关键词",)
    assert len(data.events) == 1
    assert data.report["meta"]["n_posts"] == 1
    assert data.posts[0]["topic_tags"] == ("赛果",)
    assert data.comments[0]["comment_id"] == "comment-1"
    assert data.ingestion["errors"] == ()
    with pytest.raises(TypeError):
        data.report["meta"]["n_posts"] = 2
    with pytest.raises(TypeError):
        data.events["event-2"] = {"event_id": "event-2"}
    with pytest.raises(FrozenInstanceError):
        data.events = {}


def test_loads_real_project_counts_and_excludes_inactive_events(
    project_data: Any,
) -> None:
    assert isinstance(project_data.events, Mapping)
    event_ids = set(project_data.events)

    assert len(project_data.events) == 8
    assert len(project_data.posts) == 45
    assert len(project_data.comments) == 61
    assert project_data.report["meta"]["n_posts"] == 45
    assert project_data.report["meta"]["n_comments"] == 61
    assert project_data.ingestion["lab3_ready"] is True
    assert project_data.ingestion["errors"] == ()
    assert "win_20240317_singapore_liang" not in event_ids
    assert "win_20250524_doha_moregard" not in event_ids


def test_report_rejects_empty_posts_section(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    module = importlib.import_module("lab3.data_loader")
    paths = _copy_artifacts(module, repo_root, tmp_path)
    _mutate_json(
        paths["report"],
        lambda report: report.__setitem__("posts", {}),
    )

    with pytest.raises(
        module.DataContractError,
        match=r"report\.posts\.n_total",
    ):
        module.load_project_data(tmp_path)


def test_report_rejects_missing_nested_metric_field(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    module = importlib.import_module("lab3.data_loader")
    paths = _copy_artifacts(module, repo_root, tmp_path)

    def remove_mean_score(report: dict[str, Any]) -> None:
        event_id = next(iter(report["posts"]["by_event"]))
        del report["posts"]["by_event"][event_id]["mean_score"]

    _mutate_json(paths["report"], remove_mean_score)

    with pytest.raises(
        module.DataContractError,
        match=r"report\.posts\.by_event\..*\.mean_score",
    ):
        module.load_project_data(tmp_path)


def test_report_rejects_out_of_range_metric_polarity(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    module = importlib.import_module("lab3.data_loader")
    paths = _copy_artifacts(module, repo_root, tmp_path)

    def corrupt_polarity(report: dict[str, Any]) -> None:
        event_id = next(iter(report["posts"]["by_event"]))
        report["posts"]["by_event"][event_id]["polarity_pct"][
            "positive"
        ] = 101

    _mutate_json(paths["report"], corrupt_polarity)

    with pytest.raises(
        module.DataContractError,
        match=r"report\.posts\.by_event\..*\.polarity_pct\.positive",
    ):
        module.load_project_data(tmp_path)


def test_report_by_event_keys_match_source_rows(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    module = importlib.import_module("lab3.data_loader")
    paths = _copy_artifacts(module, repo_root, tmp_path)

    def remove_event_metric(report: dict[str, Any]) -> None:
        event_id = next(iter(report["posts"]["by_event"]))
        del report["posts"]["by_event"][event_id]

    _mutate_json(paths["report"], remove_event_metric)

    with pytest.raises(
        module.DataContractError,
        match=r"report\.posts\.by_event",
    ):
        module.load_project_data(tmp_path)


@pytest.mark.parametrize(
    ("case", "expected_path"),
    [
        ("empty-emotions", r"report\.meta\.emotions"),
        ("bad-source-count", r"report\.posts\.n_total"),
        ("missing-rollup", r"report\.posts\.by_result_rollup\.win"),
        ("bool-event-count", r"report\.posts\.by_event\..*\.n"),
        ("infinite-rollup-mean", r"report\.posts\.by_result_rollup\.win\.mean_score"),
        ("unknown-emotion", r"emotion_dist_pct\.未知情绪"),
        ("bad-topic-rate", r"topic_mention_rate\.赛果"),
    ],
)
def test_report_rejects_invalid_evidence_metric(
    repo_root: Path,
    tmp_path: Path,
    case: str,
    expected_path: str,
) -> None:
    module = importlib.import_module("lab3.data_loader")
    paths = _copy_artifacts(module, repo_root, tmp_path)

    def corrupt_report(report: dict[str, Any]) -> None:
        event_id = next(iter(report["posts"]["by_event"]))
        event_metric = report["posts"]["by_event"][event_id]
        rollups = report["posts"]["by_result_rollup"]
        if case == "empty-emotions":
            report["meta"]["emotions"] = []
        elif case == "bad-source-count":
            report["posts"]["n_total"] = -1
        elif case == "missing-rollup":
            del rollups["win"]
        elif case == "bool-event-count":
            event_metric["n"] = True
        elif case == "infinite-rollup-mean":
            rollups["win"]["mean_score"] = float("inf")
        elif case == "unknown-emotion":
            event_metric["emotion_dist_pct"]["未知情绪"] = 1.0
        elif case == "bad-topic-rate":
            event_metric["topic_mention_rate"]["赛果"] = 1.01

    _mutate_json(paths["report"], corrupt_report)

    with pytest.raises(module.DataContractError, match=expected_path):
        module.load_project_data(tmp_path)


def test_missing_events_artifact_names_the_file(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    module = importlib.import_module("lab3.data_loader")
    paths = _copy_artifacts(module, repo_root, tmp_path)
    paths["events"].unlink()

    with pytest.raises(module.DataContractError, match=r"events\.json"):
        module.load_project_data(tmp_path)


def test_duplicate_post_id_names_the_record_and_field(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    module = importlib.import_module("lab3.data_loader")
    paths = _copy_artifacts(module, repo_root, tmp_path)
    rows = [
        json.loads(line)
        for line in paths["posts"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    duplicate_id = rows[0]["post_id"]
    _mutate_jsonl(
        paths["posts"],
        lambda values: values[1].__setitem__("post_id", duplicate_id),
    )

    with pytest.raises(
        module.DataContractError,
        match=rf"{duplicate_id}.*post_id",
    ):
        module.load_project_data(tmp_path)


@pytest.mark.parametrize(
    "bad_event_id",
    ["event-does-not-exist", "win_20240317_singapore_liang"],
    ids=["unknown", "inactive"],
)
def test_post_rejects_unknown_or_inactive_event(
    repo_root: Path,
    tmp_path: Path,
    bad_event_id: str,
) -> None:
    module = importlib.import_module("lab3.data_loader")
    paths = _copy_artifacts(module, repo_root, tmp_path)
    rows = [
        json.loads(line)
        for line in paths["posts"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record_id = rows[0]["post_id"]
    _mutate_jsonl(
        paths["posts"],
        lambda values: values[0].__setitem__("event_id", bad_event_id),
    )

    with pytest.raises(
        module.DataContractError,
        match=rf"{record_id}.*event_id.*{bad_event_id}",
    ):
        module.load_project_data(tmp_path)


@pytest.mark.parametrize("bad_hours", [-0.001, 24], ids=["negative", "at-24"])
def test_post_rejects_hours_outside_24_hour_window(
    repo_root: Path,
    tmp_path: Path,
    bad_hours: float,
) -> None:
    module = importlib.import_module("lab3.data_loader")
    paths = _copy_artifacts(module, repo_root, tmp_path)
    rows = [
        json.loads(line)
        for line in paths["posts"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record_id = rows[0]["post_id"]
    _mutate_jsonl(
        paths["posts"],
        lambda values: values[0].__setitem__("hours_after_event", bad_hours),
    )

    with pytest.raises(
        module.DataContractError,
        match=rf"{record_id}.*hours_after_event",
    ):
        module.load_project_data(tmp_path)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("post_id", ""),
        ("match_result", "win"),
        ("content_type", "comment"),
        ("text_clean", "   "),
        ("sentiment_polarity", "mixed"),
        ("sentiment_category", ""),
        ("sentiment_intensity", True),
        ("topic_tags", []),
        ("confidence", True),
    ],
)
def test_post_rejects_invalid_contract_field(
    repo_root: Path,
    tmp_path: Path,
    field: str,
    bad_value: Any,
) -> None:
    module = importlib.import_module("lab3.data_loader")
    paths = _copy_artifacts(module, repo_root, tmp_path)
    _mutate_jsonl(
        paths["posts"],
        lambda values: values[0].__setitem__(field, bad_value),
    )

    with pytest.raises(module.DataContractError, match=field):
        module.load_project_data(tmp_path)


@pytest.mark.parametrize(
    ("case", "field"),
    [
        ("missing-likes", "likes"),
        ("negative-likes", "likes"),
        ("unknown-category", "sentiment_category"),
        ("unknown-topic", "topic_tags"),
        ("event-name-mismatch", "event_name"),
    ],
)
def test_post_rejects_invalid_evidence_row_field(
    repo_root: Path,
    tmp_path: Path,
    case: str,
    field: str,
) -> None:
    module = importlib.import_module("lab3.data_loader")
    paths = _copy_artifacts(module, repo_root, tmp_path)
    rows = [
        json.loads(line)
        for line in paths["posts"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record_id = rows[0]["post_id"]

    def corrupt_row(values: list[dict[str, Any]]) -> None:
        row = values[0]
        if case == "missing-likes":
            del row["likes"]
        elif case == "negative-likes":
            row["likes"] = -1
        elif case == "unknown-category":
            row["sentiment_category"] = "未知情绪"
        elif case == "unknown-topic":
            row["topic_tags"] = ["未知议题"]
        elif case == "event-name-mismatch":
            row["event_name"] = "错误赛事"

    _mutate_jsonl(paths["posts"], corrupt_row)

    with pytest.raises(
        module.DataContractError,
        match=rf"{record_id}.*{field}",
    ):
        module.load_project_data(tmp_path)


def test_report_count_mismatch_names_meta_field(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    module = importlib.import_module("lab3.data_loader")
    paths = _copy_artifacts(module, repo_root, tmp_path)
    _mutate_json(
        paths["report"],
        lambda value: value["meta"].__setitem__("n_posts", 44),
    )

    with pytest.raises(module.DataContractError, match=r"meta\.n_posts"):
        module.load_project_data(tmp_path)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("lab3_ready", False), ("errors", ["upstream failure"])],
)
def test_ingestion_must_be_ready_and_error_free(
    repo_root: Path,
    tmp_path: Path,
    field: str,
    bad_value: Any,
) -> None:
    module = importlib.import_module("lab3.data_loader")
    paths = _copy_artifacts(module, repo_root, tmp_path)
    _mutate_json(
        paths["ingestion"],
        lambda value: value.__setitem__(field, bad_value),
    )

    with pytest.raises(module.DataContractError, match=field):
        module.load_project_data(tmp_path)
