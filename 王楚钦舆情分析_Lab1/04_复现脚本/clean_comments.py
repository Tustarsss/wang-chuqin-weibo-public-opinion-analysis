"""Clean first-level comments collected from the 45 balanced Weibo anchors."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


HTML_TAG = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\s+")
URL = re.compile(r"(?:https?://|www\.)\S+|网页链接")
GENERIC_INTERACTION = re.compile(
    r"感谢.{0,4}分享|内容.{0,3}精彩|(?:早上|上午|中午|下午|晚上)好|"
    r"周末快乐|来了解一下|支持一下|转发微博|欣赏佳作|辛苦整理|拜读|好文|互动|打卡"
)
TOPIC_SIGNAL = re.compile(
    r"王楚钦|大头|比赛|乒乓|球|赢|输|胜|败|冠军|决赛|半决赛|比分|"
    r"加油|发挥|状态|实力|心态|对手|莫雷加德|梁靖崑|卡尔德拉诺|雨果|"
    r"张本智和|林诗栋|勒布伦"
)


def clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = HTML_TAG.sub(" ", value)
    value = URL.sub(" ", value)
    return SPACE.sub(" ", value).strip()


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in paths:
        candidates = [root] if root.is_file() else list(root.rglob("detail_comments_*.jsonl"))
        for path in candidates:
            with path.open(encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"WARN invalid JSON: {path}:{line_no}")
    return rows


def write_rows(rows: list[dict[str, Any]], jsonl_path: Path, csv_path: Path) -> None:
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument(
        "--anchors", type=Path, default=here / "output" / "wang_chuqin_weibo_balanced.jsonl"
    )
    parser.add_argument("--output", type=Path, default=here / "output")
    args = parser.parse_args()

    anchors = {
        str(row["post_id"]): row
        for row in load_jsonl([args.anchors])
    }
    raw_rows = load_jsonl(args.raw)
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    seen_ids: set[str] = set()
    window_rows: list[dict[str, Any]] = []

    for row in raw_rows:
        post_id = str(row.get("note_id", ""))
        anchor = anchors.get(post_id)
        if not anchor:
            continue
        event_id = anchor["event_id"]
        stats[event_id]["raw"] += 1
        comment_id = str(row.get("comment_id", ""))
        if not comment_id or comment_id in seen_ids:
            stats[event_id]["duplicate_or_missing_id"] += 1
            continue
        seen_ids.add(comment_id)
        try:
            published = datetime.fromisoformat(row["create_date_time"])
        except (KeyError, TypeError, ValueError):
            stats[event_id]["invalid_time"] += 1
            continue
        start = datetime.fromisoformat(anchor["window_start"])
        end = datetime.fromisoformat(anchor["window_end"])
        if not start <= published < end:
            stats[event_id]["outside_window"] += 1
            continue
        text = clean_text(row.get("content", ""))
        if not text:
            stats[event_id]["empty"] += 1
            continue

        generic = bool(
            (GENERIC_INTERACTION.search(text) and not TOPIC_SIGNAL.search(text))
            or re.fullmatch(r"[\d\W_]+", text)
        )
        stats[event_id]["within_window"] += 1
        if generic:
            stats[event_id]["possible_generic_interaction"] += 1
        else:
            stats[event_id]["analysis_ready"] += 1
        window_rows.append(
            {
                "comment_id": comment_id,
                "parent_post_id": post_id,
                "event_id": event_id,
                "athlete": anchor["athlete"],
                "match_result": anchor["match_result"],
                "event_name": anchor["event_name"],
                "event_level": anchor["event_level"],
                "round": anchor["round"],
                "opponent": anchor["opponent"],
                "window_start": anchor["window_start"],
                "window_end": anchor["window_end"],
                "publish_time": row["create_date_time"],
                "hours_after_event": round((published - start).total_seconds() / 3600, 3),
                "content_type": "comment",
                "comment_level": 1,
                "text_raw": html.unescape(row.get("content", "")),
                "text_clean": text,
                "likes": as_int(row.get("comment_like_count")),
                "reply_count": as_int(row.get("sub_comment_count")),
                "creator_hash": row.get("creator_hash", ""),
                "is_possible_generic_interaction": generic,
            }
        )

    window_rows.sort(key=lambda row: (row["event_id"], row["parent_post_id"], row["publish_time"]))
    analysis_rows = [row for row in window_rows if not row["is_possible_generic_interaction"]]
    balanced_rows: list[dict[str, Any]] = []
    analysis_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in analysis_rows:
        analysis_by_event[row["event_id"]].append(row)
    # Round-robin across anchor posts so an event's first high-volume comment
    # section does not consume the whole event cap.
    for event_id in sorted(analysis_by_event):
        by_post: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in analysis_by_event[event_id]:
            by_post[row["parent_post_id"]].append(row)
        post_ids = sorted(by_post)
        index = 0
        while len([row for row in balanced_rows if row["event_id"] == event_id]) < 10:
            added = False
            for post_id in post_ids:
                if index < len(by_post[post_id]):
                    balanced_rows.append(by_post[post_id][index])
                    added = True
                    if len([row for row in balanced_rows if row["event_id"] == event_id]) == 10:
                        break
            if not added:
                break
            index += 1
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(
        window_rows,
        args.output / "wang_chuqin_weibo_comments_24h.jsonl",
        args.output / "wang_chuqin_weibo_comments_24h.csv",
    )
    write_rows(
        analysis_rows,
        args.output / "wang_chuqin_weibo_comments_analysis_ready.jsonl",
        args.output / "wang_chuqin_weibo_comments_analysis_ready.csv",
    )
    write_rows(
        balanced_rows,
        args.output / "wang_chuqin_weibo_comments_balanced.jsonl",
        args.output / "wang_chuqin_weibo_comments_balanced.csv",
    )

    report = {
        "anchor_posts": len(anchors),
        "raw_comments": len(raw_rows),
        "unique_comment_ids_observed": len(seen_ids),
        "duplicate_or_missing_id_rows": sum(
            counts["duplicate_or_missing_id"] for counts in stats.values()
        ),
        "comments_within_24h": len(window_rows),
        "analysis_ready_comments": len(analysis_rows),
        "balanced_comments": len(balanced_rows),
        "within_24h_by_result": dict(Counter(row["match_result"] for row in window_rows)),
        "analysis_ready_by_result": dict(Counter(row["match_result"] for row in analysis_rows)),
        "balanced_by_result": dict(Counter(row["match_result"] for row in balanced_rows)),
        "within_24h_by_event": dict(Counter(row["event_id"] for row in window_rows)),
        "analysis_ready_by_event": dict(Counter(row["event_id"] for row in analysis_rows)),
        "balanced_by_event": dict(Counter(row["event_id"] for row in balanced_rows)),
        "anchors_with_raw_comments": len({str(row.get("note_id", "")) for row in raw_rows} & anchors.keys()),
        "anchors_with_24h_comments": len({row["parent_post_id"] for row in window_rows}),
        "by_event": {event_id: dict(counts) for event_id, counts in sorted(stats.items())},
        "sampling_note": "The API returns hot-flow comments, not a complete chronological archive.",
    }
    (args.output / "comments_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
