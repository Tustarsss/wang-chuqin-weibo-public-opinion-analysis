"""Clean MediaCrawler Weibo JSONL files into the shared Lab 1 dataset."""

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
AD_PATTERN = re.compile(r"抽奖|带货|优惠券|下单|购买链接|私信领取|加微|代购")


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


def load_raw(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in paths:
        for path in root.rglob("search_contents_*.jsonl"):
            with path.open(encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"WARN invalid JSON: {path}:{line_no}")
                        continue
                    item["_raw_file"] = str(path)
                    rows.append(item)
    return rows


def is_relevant(text: str, event: dict[str, Any]) -> bool:
    if "王楚钦" not in text:
        return False
    opponent_hit = any(alias in text for alias in event["opponent_aliases"])
    event_terms = [
        term
        for term in re.split(r"WTT|20\d{2}|男单|决赛|半决赛", event["event_name"])
        if len(term) >= 2
    ]
    return opponent_hit or any(term in text for term in event_terms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=Path(__file__).with_name("events.json"))
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("output"))
    args = parser.parse_args()

    all_events = json.loads(args.events.read_text(encoding="utf-8"))
    events = [event for event in all_events if event.get("active", True)]
    keyword_to_event = {
        keyword: event for event in events for keyword in event["keywords"]
    }
    raw_rows = load_raw(args.raw)
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    selected: list[dict[str, Any]] = []
    seen_event_ids: set[tuple[str, str]] = set()
    seen_event_texts: set[tuple[str, str]] = set()

    for row in raw_rows:
        keyword = row.get("source_keyword", "")
        event = keyword_to_event.get(keyword)
        if not event:
            continue
        event_id = event["event_id"]
        stats[event_id]["raw"] += 1
        try:
            published = datetime.fromisoformat(row["create_date_time"])
        except (KeyError, TypeError, ValueError):
            stats[event_id]["invalid_time"] += 1
            continue
        start = datetime.fromisoformat(event["window_start"])
        end = datetime.fromisoformat(event["window_end"])
        if not start <= published < end:
            stats[event_id]["outside_window"] += 1
            continue
        text = clean_text(row.get("content", ""))
        if not text:
            stats[event_id]["empty"] += 1
            continue
        if not is_relevant(text, event):
            stats[event_id]["irrelevant"] += 1
            continue
        dedupe_key = (event_id, str(row.get("note_id", "")))
        if dedupe_key in seen_event_ids:
            stats[event_id]["duplicate"] += 1
            continue
        seen_event_ids.add(dedupe_key)
        text_fingerprint = re.sub(r"[\W_]+", "", text).lower()
        text_key = (event_id, text_fingerprint)
        if text_key in seen_event_texts:
            stats[event_id]["text_duplicate"] += 1
            continue
        seen_event_texts.add(text_key)
        stats[event_id]["kept"] += 1
        selected.append(
            {
                "post_id": str(row.get("note_id", "")),
                "event_id": event_id,
                "athlete": "王楚钦",
                "match_result": event["result"],
                "event_name": event["event_name"],
                "event_level": event["event_level"],
                "round": event["round"],
                "opponent": event["opponent"],
                "window_start": event["window_start"],
                "window_end": event["window_end"],
                "publish_time": row["create_date_time"],
                "hours_after_event": round((published - start).total_seconds() / 3600, 3),
                "source_keyword": keyword,
                "content_type": "post",
                "text_raw": html.unescape(row.get("content", "")),
                "text_clean": text,
                "likes": as_int(row.get("liked_count")),
                "comments": as_int(row.get("comments_count")),
                "shares": as_int(row.get("shared_count")),
                "post_url": row.get("note_url", ""),
                "creator_hash": row.get("creator_hash", ""),
                "is_possible_ad": bool(AD_PATTERN.search(text)),
            }
        )

    selected.sort(key=lambda row: (row["event_id"], row["publish_time"], row["post_id"]))
    args.output.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output / "wang_chuqin_weibo_clean.jsonl"
    csv_path = args.output / "wang_chuqin_weibo_clean.csv"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if selected:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=selected[0].keys())
            writer.writeheader()
            writer.writerows(selected)

    # A small comparison-ready subset. Taking the earliest posts is
    # deterministic and keeps high-volume events from dominating Lab 2.
    balanced: list[dict[str, Any]] = []
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_event[row["event_id"]].append(row)
    for event in events:
        balanced.extend(by_event[event["event_id"]][:6])
    balanced_jsonl = args.output / "wang_chuqin_weibo_balanced.jsonl"
    balanced_csv = args.output / "wang_chuqin_weibo_balanced.csv"
    with balanced_jsonl.open("w", encoding="utf-8") as stream:
        for row in balanced:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if balanced:
        with balanced_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=balanced[0].keys())
            writer.writeheader()
            writer.writerows(balanced)

    report = {
        "raw_records": len(raw_rows),
        "clean_records": len(selected),
        "by_result": dict(Counter(row["match_result"] for row in selected)),
        "balanced_records": len(balanced),
        "balanced_by_result": dict(Counter(row["match_result"] for row in balanced)),
        "balanced_by_event": dict(Counter(row["event_id"] for row in balanced)),
        "by_event": {event["event_id"]: dict(stats[event["event_id"]]) for event in events},
    }
    (args.output / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
