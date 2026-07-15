"""Lab 2 -> Lab 3 数据契约干跑与固定人工抽检回归。

本脚本不调用 API，只读取 Lab 1 事件配置、Lab 2 逐条输出和聚合报告。
退出码 0 表示结构/类型/取值约束全部通过；已知的稀疏或空事件只记 warning。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
ROOT = LAB2_DIR.parent
DEFAULT_OUTPUT = LAB2_DIR / "01_输出"
DEFAULT_QUALITY = LAB2_DIR / "02_质量报告" / "lab3_ingestion_check.json"
DEFAULT_EVENTS = ROOT / "王楚钦舆情分析_Lab1" / "03_说明与配置" / "events.json"
DEFAULT_TAXONOMY = LAB2_DIR / "03_说明与配置" / "sentiment_taxonomy.json"

SIGN = {"positive": 1, "negative": -1, "neutral": 0}
MIN_EVENT_COVERAGE = 3

# 人工阅读原文后确定的 5 个预期。脚本负责在输出重生成后防止这些标签静默漂移。
SPOT_CHECKS: list[dict[str, Any]] = [
    {
        "content_type": "comment",
        "record_id": "5132082504798861",
        "case": "输球后仍支持，验证赛果不决定极性",
        "expected": {"sentiment_polarity": "positive", "sentiment_category": "支持鼓励"},
    },
    {
        "content_type": "post",
        "record_id": "5062247656721642",
        "case": "球拍被踩断与心疼输球的混合表达",
        "expected": {
            "sentiment_polarity": "negative",
            "sentiment_category": "失望惋惜",
            "is_mixed": True,
            "topic_contains": "器材场地",
        },
    },
    {
        "content_type": "comment",
        "record_id": "5062460966962097",
        "case": "“世一也就是出勤率”的贬低语义",
        "expected": {"sentiment_polarity": "negative", "sentiment_category": "批评质疑"},
    },
    {
        "content_type": "comment",
        "record_id": "5198540999494928",
        "case": "“骰子”黑称的调侃与负向语义",
        "expected": {"sentiment_polarity": "negative", "sentiment_category": "调侃戏谑"},
    },
    {
        "content_type": "comment",
        "record_id": "5198554912003467",
        "case": "乱码应降置信度而非伪造情绪",
        "expected": {
            "sentiment_polarity": "neutral",
            "sentiment_category": "中性陈述",
            "confidence_max": 0.3,
        },
    },
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_no} 不是 JSON object")
            rows.append(value)
    return rows


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_rows(
    rows: list[dict[str, Any]],
    label: str,
    id_field: str,
    expected_content_type: str,
    emotions: set[str],
    topics: set[str],
    event_results: dict[str, str],
) -> tuple[list[str], list[str], dict[str, int]]:
    errors: list[str] = []
    warnings: list[str] = []
    ids: list[str] = []
    event_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, 1):
        rid = row.get(id_field)
        where = f"{label}[{index}]({rid or 'missing-id'})"
        if not isinstance(rid, str) or not rid:
            errors.append(f"{where}: {id_field} 必须为非空字符串")
        else:
            ids.append(rid)

        required_strings = [
            "event_id", "match_result", "content_type", "text_clean",
            "sentiment_polarity", "sentiment_category", "viewpoint", "rationale", "model",
        ]
        for field in required_strings:
            if not isinstance(row.get(field), str) or not row[field]:
                errors.append(f"{where}: {field} 必须为非空字符串")

        event_id = row.get("event_id")
        if isinstance(event_id, str):
            event_counts[event_id] += 1
            if event_id not in event_results:
                errors.append(f"{where}: event_id 不在 Lab 1 active 事件中")
            elif row.get("match_result") != event_results[event_id]:
                errors.append(f"{where}: match_result 与 Lab 1 事件配置不一致")
        if row.get("content_type") != expected_content_type:
            errors.append(f"{where}: content_type 应为 {expected_content_type}")
        if row.get("match_result") not in {"win", "loss"}:
            errors.append(f"{where}: match_result 只能为 win/loss")
        if row.get("sentiment_polarity") not in SIGN:
            errors.append(f"{where}: sentiment_polarity 越界")
        if row.get("sentiment_category") not in emotions:
            errors.append(f"{where}: sentiment_category 不在 taxonomy")
        secondary = row.get("sentiment_category_secondary")
        if secondary is not None and secondary not in emotions:
            errors.append(f"{where}: sentiment_category_secondary 不在 taxonomy")

        intensity = row.get("sentiment_intensity")
        if not isinstance(intensity, int) or isinstance(intensity, bool) or not 1 <= intensity <= 5:
            errors.append(f"{where}: sentiment_intensity 必须为 1-5 整数")
        if not isinstance(row.get("is_mixed"), bool):
            errors.append(f"{where}: is_mixed 必须为 bool")
        tags = row.get("topic_tags")
        if not isinstance(tags, list) or not 1 <= len(tags) <= 3 or any(t not in topics for t in tags):
            errors.append(f"{where}: topic_tags 必须含 1-3 个 taxonomy 议题")
        confidence = row.get("confidence")
        if not is_number(confidence) or not 0 <= float(confidence) <= 1:
            errors.append(f"{where}: confidence 必须在 [0,1]")
        score = row.get("sentiment_score")
        if not is_number(score):
            errors.append(f"{where}: sentiment_score 必须为数值")
        elif isinstance(intensity, int) and row.get("sentiment_polarity") in SIGN:
            expected_score = round(SIGN[row["sentiment_polarity"]] * intensity / 5, 3)
            if abs(float(score) - expected_score) > 1e-9:
                errors.append(f"{where}: sentiment_score 应为 {expected_score}")
        hours = row.get("hours_after_event")
        if not is_number(hours) or not 0 <= float(hours) < 24:
            errors.append(f"{where}: hours_after_event 应在 [0,24)")
        if "_status" in row:
            errors.append(f"{where}: 内部字段 _status 不应出现在正式输出")

    duplicates = [rid for rid, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"{label}: {id_field} 重复 {duplicates}")

    missing_events = sorted(set(event_results) - set(event_counts))
    thin_events = sorted(eid for eid, count in event_counts.items() if count < MIN_EVENT_COVERAGE)
    if missing_events:
        warnings.append(f"{label}: 零记录事件 {missing_events}")
    if thin_events:
        warnings.append(f"{label}: 少于 {MIN_EVENT_COVERAGE} 条的稀疏事件 {thin_events}")
    return errors, warnings, dict(sorted(event_counts.items()))


def run_spot_checks(posts: list[dict[str, Any]], comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexes = {
        "post": {row["post_id"]: row for row in posts},
        "comment": {row["comment_id"]: row for row in comments},
    }
    results: list[dict[str, Any]] = []
    for spec in SPOT_CHECKS:
        row = indexes[spec["content_type"]].get(spec["record_id"])
        checks: dict[str, bool] = {}
        expected = spec["expected"]
        if row is not None:
            for key, value in expected.items():
                if key == "topic_contains":
                    checks[key] = value in row.get("topic_tags", [])
                elif key == "confidence_max":
                    checks[key] = float(row.get("confidence", 1)) <= value
                else:
                    checks[key] = row.get(key) == value
        results.append({
            **spec,
            "text_clean": row.get("text_clean") if row else None,
            "actual": ({
                "sentiment_polarity": row.get("sentiment_polarity"),
                "sentiment_category": row.get("sentiment_category"),
                "is_mixed": row.get("is_mixed"),
                "topic_tags": row.get("topic_tags"),
                "confidence": row.get("confidence"),
            } if row else None),
            "checks": checks,
            "passed": row is not None and all(checks.values()),
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 2 -> Lab 3 数据契约干跑")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_QUALITY)
    args = parser.parse_args()

    events = [event for event in load_json(args.events) if event.get("active", True)]
    event_results = {event["event_id"]: event["result"] for event in events}
    taxonomy = load_json(args.taxonomy)
    emotions = {item["key"] for item in taxonomy["emotions"]}
    topics = {item["key"] for item in taxonomy["topics"]}
    posts = load_jsonl(args.data_dir / "posts_sentiment.jsonl")
    comments = load_jsonl(args.data_dir / "comments_sentiment.jsonl")
    report = load_json(args.data_dir / "sentiment_report.json")

    post_errors, post_warnings, post_counts = validate_rows(
        posts, "posts", "post_id", "post", emotions, topics, event_results
    )
    comment_errors, comment_warnings, comment_counts = validate_rows(
        comments, "comments", "comment_id", "comment", emotions, topics, event_results
    )
    errors = post_errors + comment_errors
    warnings = post_warnings + comment_warnings

    for section in ("meta", "posts", "comments", "domain_comparison"):
        if section not in report:
            errors.append(f"sentiment_report.json 缺少顶层字段 {section}")
    meta = report.get("meta", {})
    if meta.get("n_posts") != len(posts):
        errors.append("sentiment_report.meta.n_posts 与逐条输出不一致")
    if meta.get("n_comments") != len(comments):
        errors.append("sentiment_report.meta.n_comments 与逐条输出不一致")
    for side in ("posts", "comments"):
        rollups = report.get(side, {}).get("by_result_rollup", {})
        if not {"win", "loss"}.issubset(rollups):
            errors.append(f"sentiment_report.{side} 缺少 win/loss 汇总")
        reps = report.get(side, {}).get("representative_viewpoints_by_result", {})
        if not {"win", "loss"}.issubset(reps):
            errors.append(f"sentiment_report.{side} 缺少按胜负组组织的代表性观点")
        else:
            for result in ("win", "loss"):
                if not reps[result].get("headline") or not reps[result].get("by_topic"):
                    errors.append(f"sentiment_report.{side}.{result} 代表性观点不完整")

    spot_checks = run_spot_checks(posts, comments)
    failed_spots = [item["record_id"] for item in spot_checks if not item["passed"]]
    if failed_spots:
        errors.append(f"固定人工抽检回归失败: {failed_spots}")

    status = "pass" if not errors and not warnings else "pass_with_warnings" if not errors else "fail"
    result = {
        "status": status,
        "schema_assertions_passed": not errors,
        "lab3_ready": not errors,
        "summary": {
            "active_events": len(events),
            "posts": len(posts),
            "comments": len(comments),
            "spot_checks_passed": sum(item["passed"] for item in spot_checks),
            "spot_checks_total": len(spot_checks),
            "errors": len(errors),
            "warnings": len(warnings),
        },
        "event_counts": {"posts": post_counts, "comments": comment_counts},
        "errors": errors,
        "warnings": warnings,
        "manual_spot_check_regression": spot_checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"] | {"status": status, "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
