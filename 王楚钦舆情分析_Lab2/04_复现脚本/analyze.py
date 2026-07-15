"""Lab 2 聚合分析：把逐条情感标注聚合成胜负对比、议题分布、代表性观点与领域差异，输出机器可读 JSON 与中文分析报告。

严守 Lab 1 README：正文/评论分开（content_type），先按 event_id 分场统计，再以各场指标的均值汇总到胜负组，
避免单场热点主导结论。稀疏覆盖（每场 <3 条）的事件不参与胜负汇总。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
DEFAULT_POSTS = LAB2_DIR / "01_输出" / "posts_sentiment.jsonl"
DEFAULT_COMMENTS = LAB2_DIR / "01_输出" / "comments_sentiment.jsonl"
DEFAULT_OUTPUT = LAB2_DIR / "01_输出"

MIN_COVERAGE = 3
POLARITIES = ["positive", "negative", "neutral"]
POLARITY_CN = {"positive": "正面", "negative": "负面", "neutral": "中性"}
TIMELINE_BINS = [(0, 2, "0-2h"), (2, 6, "2-6h"), (6, 12, "6-12h"), (12, 24, "12-24h")]
REP_CONF_THRESHOLD = 0.6


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


def is_failed(row: dict[str, Any]) -> bool:
    return str(row.get("rationale", "")).startswith("API_")


def likes_of(row: dict[str, Any]) -> int:
    try:
        return int(row.get("likes", 0) or 0)
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# 单事件（event_id）指标
# --------------------------------------------------------------------------- #
def timeline_bins(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lo, hi, label in TIMELINE_BINS:
        bucket = [r for r in records if lo <= float(r.get("hours_after_event", 0) or 0) < hi]
        out.append({
            "bin": label,
            "n": len(bucket),
            "mean_score": mean([r["sentiment_score"] for r in bucket]) if bucket else 0.0,
        })
    return out


def event_metrics(event_id: str, records: list[dict[str, Any]], emotions: list[str], topics: list[str]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"event_id": event_id, "n": 0, "coverage_flag": "empty"}
    pol = Counter(r["sentiment_polarity"] for r in records)
    emo = Counter(r["sentiment_category"] for r in records)
    topic_counter: Counter[str] = Counter()
    for r in records:
        for t in r.get("topic_tags", []):
            topic_counter[t] += 1
    scores = [r["sentiment_score"] for r in records]
    intensities = [r["sentiment_intensity"] for r in records]
    total_likes = sum(likes_of(r) for r in records)
    weighted = sum(likes_of(r) * r["sentiment_score"] for r in records)
    eng = round(weighted / total_likes, 4) if total_likes > 0 else mean(scores)
    return {
        "event_id": event_id,
        "n": n,
        "coverage_flag": "thin" if n < MIN_COVERAGE else "ok",
        "polarity": {
            p: pol.get(p, 0) for p in POLARITIES
        },
        "polarity_pct": {p: pct(pol.get(p, 0), n) for p in POLARITIES},
        "mean_score": mean(scores),
        "engagement_weighted_score": eng,
        "emotion_dist": {e: emo.get(e, 0) for e in emotions},
        "emotion_dist_pct": {e: pct(emo.get(e, 0), n) for e in emotions},
        "dominant_emotion": (emo.most_common(1)[0][0] if emo else None),
        "topic_counts": {t: topic_counter.get(t, 0) for t in topics},
        "topic_mention_rate": {t: round(topic_counter.get(t, 0) / n, 4) for t in topics},
        "mean_intensity": mean([float(i) for i in intensities]),
        "mixed_rate": round(sum(1 for r in records if r.get("is_mixed")) / n, 4),
        "timeline": timeline_bins(records),
        "n_failed": sum(1 for r in records if is_failed(r)),
        "n_low_confidence": sum(1 for r in records if float(r.get("confidence", 1)) < 0.4),
    }


# --------------------------------------------------------------------------- #
# 胜负汇总（各 event 指标的均值，排除稀疏/空事件）
# --------------------------------------------------------------------------- #
def rollup(events: list[dict[str, Any]], pooled_records: list[dict[str, Any]], emotions: list[str], topics: list[str]) -> dict[str, Any]:
    valid = [e for e in events if e["n"] >= MIN_COVERAGE]
    excluded = [
        {"event_id": e["event_id"], "n": e["n"], "reason": "empty" if e["n"] == 0 else "thin"}
        for e in events if e["n"] < MIN_COVERAGE
    ]
    n_total = sum(e["n"] for e in events)
    if not valid:
        return {
            "n_total": n_total, "n_events_valid": 0, "excluded_from_rollup": excluded,
            "coverage_flag": "no_valid_events",
        }
    return {
        "n_total": n_total,
        "n_events_valid": len(valid),
        "excluded_from_rollup": excluded,
        "positive_pct": mean([e["polarity_pct"]["positive"] for e in valid]),
        "negative_pct": mean([e["polarity_pct"]["negative"] for e in valid]),
        "neutral_pct": mean([e["polarity_pct"]["neutral"] for e in valid]),
        "mean_score": mean([e["mean_score"] for e in valid]),
        "engagement_weighted_score": mean([e["engagement_weighted_score"] for e in valid]),
        "mean_intensity": mean([e["mean_intensity"] for e in valid]),
        "mixed_rate": mean([e["mixed_rate"] for e in valid]),
        "emotion_dist_pct": {k: mean([e["emotion_dist_pct"][k] for e in valid]) for k in emotions},
        "topic_mention_rate": {k: mean([e["topic_mention_rate"][k] for e in valid]) for k in topics},
        "timeline": timeline_bins(pooled_records),  # 时间线在胜负组内池化（跨事件的时间演化）
    }


# --------------------------------------------------------------------------- #
# 代表性观点
# --------------------------------------------------------------------------- #
def _rep(row: dict[str, Any], id_field: str) -> dict[str, Any]:
    return {
        "record_id": row.get(id_field),
        "text_clean": (row.get("text_clean") or "").strip()[:80].rstrip(),
        "viewpoint": row.get("viewpoint"),
        "sentiment_category": row.get("sentiment_category"),
        "sentiment_polarity": row.get("sentiment_polarity"),
        "likes": likes_of(row),
        "event_id": row.get("event_id"),
    }


def representative_viewpoints(records: list[dict[str, Any]], id_field: str, top_topics: list[str]) -> dict[str, Any]:
    """每组选头条 + 各 top 议题 + 正面/负面声音代表；优先 confidence>=0.6，否则放宽。"""

    def pick(cands: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not cands:
            return None
        cands.sort(key=lambda r: (-likes_of(r), -float(r.get("sentiment_intensity", 0)), str(r.get(id_field, ""))))
        return _rep(cands[0], id_field)

    hi = [r for r in records if float(r.get("confidence", 0)) >= REP_CONF_THRESHOLD]
    pool = hi or records
    pos = [r for r in pool if r.get("sentiment_polarity") == "positive"]
    neg = [r for r in pool if r.get("sentiment_polarity") == "negative"]
    out: dict[str, Any] = {
        "headline": pick(pool),
        "positive_voice": pick(pos),
        "negative_voice": pick(neg),
        "by_topic": {},
    }
    for topic in top_topics:
        cands = [r for r in hi if topic in r.get("topic_tags", [])] or [r for r in records if topic in r.get("topic_tags", [])]
        out["by_topic"][topic] = pick(cands)
    return out


def top_topics_for(records: list[dict[str, Any]], topics: list[str], k: int = 3) -> list[str]:
    counter: Counter[str] = Counter()
    for r in records:
        for t in r.get("topic_tags", []):
            counter[t] += 1
    return [t for t, _ in counter.most_common(k)]


# --------------------------------------------------------------------------- #
# 领域对比（体育舆情 vs 商品评论）
# --------------------------------------------------------------------------- #
def build_domain_comparison(posts_roll: dict[str, Any], comments_roll: dict[str, Any], emotions: list[str]) -> dict[str, Any]:
    comparison_table = [
        {"维度": "情感对象", "商品评论（通用基线）": "产品/服务（物）", "体育舆情（本领域）": "运动员本人（人，存在拟社会关系与身份投射）"},
        {"维度": "正向核心", "商品评论（通用基线）": "满意、好评、推荐、复购意愿", "体育舆情（本领域）": "支持鼓励、自豪骄傲（指向人，非物）"},
        {"维度": "负向核心", "商品评论（通用基线）": "不满、差评、退货", "体育舆情（本领域）": "失望惋惜、批评质疑（多针对心态/技战术，非退货）"},
        {"维度": "领域特有类别", "商品评论（通用基线）": "满意度、购买意愿", "体育舆情（本领域）": "幸灾乐祸（调侃对手）、爱国民族情感、对运动员心态的评价、对未来展望的焦虑"},
        {"维度": "通用特有（本领域无）", "商品评论（通用基线）": "复购意愿、性价比、物流评价", "体育舆情（本领域）": "—"},
        {"维度": "评价维度", "商品评论（通用基线）": "质量、价格、服务", "体育舆情（本领域）": "技战术、心理心态、赛果、对手强弱、舆论环境"},
        {"维度": "情绪强度来源", "商品评论（通用基线）": "体验落差", "体育舆情（本领域）": "身份认同（球迷/国人）+ 竞技悬念与不可预期性"},
        {"维度": "中性占比", "商品评论（通用基线）": "低（评论多为表态）", "体育舆情（本领域）": "较高（新闻性正文与赛果转述）"},
        {"维度": "混合情感", "商品评论（通用基线）": "少见", "体育舆情（本领域）": "常见（心疼+鼓励、赢球+挑剔、惜败+释然）"},
    ]
    observations: list[dict[str, str]] = []

    def loss_support_rate(roll: dict[str, Any]) -> float:
        return roll.get("emotion_dist_pct", {}).get("支持鼓励", 0.0)

    posts_loss = posts_roll.get("loss", {})
    comments_loss = comments_roll.get("loss", {})
    if comments_loss and comments_loss.get("n_events_valid", 0) > 0:
        sup = comments_loss.get("emotion_dist_pct", {}).get("支持鼓励", 0.0)
        crit = comments_loss.get("emotion_dist_pct", {}).get("批评质疑", 0.0)
        disa = comments_loss.get("emotion_dist_pct", {}).get("失望惋惜", 0.0)
        observations.append({
            "finding": "评论侧输球仍含较高比例的支持鼓励，体现体育舆情'败亦支持'的特征",
            "evidence": f"评论侧输球组中支持鼓励占 {sup:.1f}%，而批评质疑 {crit:.1f}%、失望惋惜 {disa:.1f}%——对运动员本人的支持在输球后依然显著，这在商品差评（不满即差评/退货）中几乎不存在。",
        })
    observations.append({
        "finding": "本模型采用 9 类领域情感 taxonomy 而非通用正/负/中三元分类",
        "evidence": "幸灾乐祸（调侃对手）、爱国民族、对运动员心态的评价等类别在商品评论中没有对应；反之商品评论的'复购意愿/性价比'在本领域无意义。故保留极性字段以便与通用模型对比，同时用领域类别刻画更丰富的情绪。",
    })
    if posts_roll.get("win", {}) and posts_roll.get("loss", {}):
        observations.append({
            "finding": "正文侧中性占比较高，区别于商品评论以表态为主",
            "evidence": f"正文胜/负组中性占比分别约 {posts_roll['win'].get('neutral_pct',0):.1f}% / {posts_roll['loss'].get('neutral_pct',0):.1f}%，因正文多为赛果转述与新闻性陈述；商品评论中性占比通常很低。",
        })
    return {"comparison_table": comparison_table, "data_observations": observations}


# --------------------------------------------------------------------------- #
# 单侧（content_type）聚合
# --------------------------------------------------------------------------- #
def aggregate_side(records: list[dict[str, Any]], id_field: str, emotions: list[str], topics: list[str]) -> dict[str, Any]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_event[r.get("event_id", "")].append(r)

    event_metrics_map = {eid: event_metrics(eid, recs, emotions, topics) for eid, recs in by_event.items()}
    by_result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_result[r.get("match_result", "")].append(r)

    rollups: dict[str, Any] = {}
    for result, res_recs in by_result.items():
        res_events = [eid for eid in by_event if any(r.get("match_result") == result for r in by_event[eid])]
        ev_list = [event_metrics_map[eid] for eid in res_events]
        pooled = res_recs
        rollups[result] = rollup(ev_list, pooled, emotions, topics)

    top_topics = top_topics_for(records, topics, k=3)
    reps = representative_viewpoints(records, id_field, top_topics)
    top_topics_by_result: dict[str, list[str]] = {}
    reps_by_result: dict[str, dict[str, Any]] = {}
    for result, result_records in by_result.items():
        result_topics = top_topics_for(result_records, topics, k=3)
        top_topics_by_result[result] = result_topics
        reps_by_result[result] = representative_viewpoints(result_records, id_field, result_topics)

    return {
        "n_total": len(records),
        "by_event": event_metrics_map,
        "by_result_rollup": rollups,
        "top_topics": top_topics,
        "representative_viewpoints": reps,
        "top_topics_by_result": top_topics_by_result,
        "representative_viewpoints_by_result": reps_by_result,
    }


# --------------------------------------------------------------------------- #
# 中文 Markdown 报告
# --------------------------------------------------------------------------- #
def fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    posts = report["posts"]
    comments = report["comments"]
    pr = posts["by_result_rollup"]
    cr = comments["by_result_rollup"]
    emotions = report["meta"]["emotions"]

    lines: list[str] = []
    lines.append("# 王楚钦赛后微博舆情情感/观点分析报告（Lab 2）\n")
    lines.append(f"模型：{report['meta']['model']}（关闭思考，JSON mode）。"
                 f"标注样本：正文 {posts['n_total']} 条、评论 {comments['n_total']} 条。"
                 "正文与评论分开统计，先按 event_id 分场，再以各场指标均值汇总到胜负组（每场 <3 条的事件不参与汇总）。\n")

    lines.append("## 一、概述\n")
    if "win" in pr and "loss" in pr:
        lines.append(f"- **正文**：胜组均分 {pr['win']['mean_score']:+.3f}、负组 {pr['loss']['mean_score']:+.3f}；"
                     f"胜组正面占比 {fmt_pct(pr['win']['positive_pct'])}，负组正面占比 {fmt_pct(pr['loss']['positive_pct'])}。")
    if "win" in cr and "loss" in cr:
        lines.append(f"- **评论**：胜组均分 {cr['win']['mean_score']:+.3f}、负组 {cr['loss']['mean_score']:+.3f}；"
                     f"胜组正面占比 {fmt_pct(cr['win']['positive_pct'])}，负组正面占比 {fmt_pct(cr['loss']['positive_pct'])}。")
    lines.append("- 关键观察：比赛胜负 ≠ 情感极性。输球文本中仍有支持鼓励与理性分析；模型按文本表达判断，不把赛果直接映射为极性。\n")

    lines.append("## 二、方法\n")
    lines.append("1. **情感 agent**：DeepSeek v4-pro（关闭思考），逐条输出极性、9 类领域情绪、强度、是否混合、12 类议题、代表性观点与置信度。")
    lines.append("2. **派生分数**：`sentiment_score = 极性符号(±1/0) × 强度/5`，范围 [-1, 1]，确定性可复现。")
    lines.append("3. **聚合**：正文/评论分开；先按 event_id 分场统计，再以各场指标均值汇总到胜负组，避免单场热点主导。每场 <3 条的事件标记稀疏并排除出胜负汇总。\n")

    # 胜负对比表
    lines.append("## 三、正文（posts）胜负对比\n")
    headers = ["指标", "胜（win）", "负（loss）"]
    def side_row(roll: dict[str, Any]) -> list[str]:
        if roll.get("coverage_flag") == "no_valid_events":
            return ["—（无有效事件）"] * 3
        return [
            fmt_pct(roll.get("positive_pct", 0)), fmt_pct(roll.get("negative_pct", 0)),
            fmt_pct(roll.get("neutral_pct", 0)), f"{roll.get('mean_score', 0):+.3f}",
            f"{roll.get('engagement_weighted_score', 0):+.3f}", f"{roll.get('mean_intensity', 0):.2f}",
            fmt_pct(roll.get("mixed_rate", 0) * 100), str(roll.get("n_total", 0)),
        ]
    rows = [
        ["正面占比", fmt_pct(pr.get("win", {}).get("positive_pct", 0)), fmt_pct(pr.get("loss", {}).get("positive_pct", 0))],
        ["负面占比", fmt_pct(pr.get("win", {}).get("negative_pct", 0)), fmt_pct(pr.get("loss", {}).get("negative_pct", 0))],
        ["中性占比", fmt_pct(pr.get("win", {}).get("neutral_pct", 0)), fmt_pct(pr.get("loss", {}).get("neutral_pct", 0))],
        ["情感均分", f"{pr.get('win', {}).get('mean_score', 0):+.3f}", f"{pr.get('loss', {}).get('mean_score', 0):+.3f}"],
        ["互动加权分", f"{pr.get('win', {}).get('engagement_weighted_score', 0):+.3f}", f"{pr.get('loss', {}).get('engagement_weighted_score', 0):+.3f}"],
        ["平均强度", f"{pr.get('win', {}).get('mean_intensity', 0):.2f}", f"{pr.get('loss', {}).get('mean_intensity', 0):.2f}"],
        ["混合情感率", fmt_pct(pr.get("win", {}).get("mixed_rate", 0) * 100), fmt_pct(pr.get("loss", {}).get("mixed_rate", 0) * 100)],
        ["样本数(全)", str(pr.get("win", {}).get("n_total", 0)), str(pr.get("loss", {}).get("n_total", 0))],
    ]
    lines.append(md_table(headers, rows))
    lines.append("")

    lines.append("## 四、评论（comments）胜负对比\n")
    rows = [
        ["正面占比", fmt_pct(cr.get("win", {}).get("positive_pct", 0)), fmt_pct(cr.get("loss", {}).get("positive_pct", 0))],
        ["负面占比", fmt_pct(cr.get("win", {}).get("negative_pct", 0)), fmt_pct(cr.get("loss", {}).get("negative_pct", 0))],
        ["中性占比", fmt_pct(cr.get("win", {}).get("neutral_pct", 0)), fmt_pct(cr.get("loss", {}).get("neutral_pct", 0))],
        ["情感均分", f"{cr.get('win', {}).get('mean_score', 0):+.3f}", f"{cr.get('loss', {}).get('mean_score', 0):+.3f}"],
        ["互动加权分", f"{cr.get('win', {}).get('engagement_weighted_score', 0):+.3f}", f"{cr.get('loss', {}).get('engagement_weighted_score', 0):+.3f}"],
        ["平均强度", f"{cr.get('win', {}).get('mean_intensity', 0):.2f}", f"{cr.get('loss', {}).get('mean_intensity', 0):.2f}"],
        ["混合情感率", fmt_pct(cr.get("win", {}).get("mixed_rate", 0) * 100), fmt_pct(cr.get("loss", {}).get("mixed_rate", 0) * 100)],
        ["样本数(全)", str(cr.get("win", {}).get("n_total", 0)), str(cr.get("loss", {}).get("n_total", 0))],
    ]
    lines.append(md_table(headers, rows))
    # 稀疏覆盖说明
    excluded = []
    for result in ("win", "loss"):
        for e in cr.get(result, {}).get("excluded_from_rollup", []):
            excluded.append(f"{result} 侧 {e['event_id']}（{e['n']} 条，{e['reason']}）")
    if excluded:
        lines.append(f"\n> 稀疏覆盖：{chr(12289).join(excluded)} 不参与评论侧胜负汇总。\n")
    else:
        lines.append("")

    # 情绪类别分布
    lines.append("## 五、领域情绪类别分布（事件均值占比）\n")
    for label, side in [("正文", pr), ("评论", cr)]:
        lines.append(f"**{label}**\n")
        headers2 = ["情绪类别", "胜(%)", "负(%)"]
        rows2 = []
        for e in emotions:
            w = side.get("win", {}).get("emotion_dist_pct", {}).get(e, 0.0)
            l = side.get("loss", {}).get("emotion_dist_pct", {}).get(e, 0.0)
            rows2.append([e, fmt_pct(w), fmt_pct(l)])
        lines.append(md_table(headers2, rows2))
        lines.append("")

    # 议题分布
    lines.append("## 六、议题分布（事件均值提及率 Top）\n")
    for label, side in [("正文", pr), ("评论", cr)]:
        lines.append(f"**{label}**\n")
        headers3 = ["议题", "胜(%)", "负(%)"]
        all_topics = set()
        for result in ("win", "loss"):
            all_topics.update(side.get(result, {}).get("topic_mention_rate", {}).keys())
        ranked = sorted(all_topics, key=lambda t: -(side.get("win", {}).get("topic_mention_rate", {}).get(t, 0) + side.get("loss", {}).get("topic_mention_rate", {}).get(t, 0)))
        rows3 = []
        for t in ranked[:8]:
            w = side.get("win", {}).get("topic_mention_rate", {}).get(t, 0.0) * 100
            l = side.get("loss", {}).get("topic_mention_rate", {}).get(t, 0.0) * 100
            rows3.append([t, fmt_pct(w), fmt_pct(l)])
        lines.append(md_table(headers3, rows3))
        lines.append("")

    # 代表性观点
    lines.append("## 七、代表性观点\n")
    slot_cn = {"headline": "头条", "positive_voice": "正面声音", "negative_voice": "负面声音"}
    for label, side in [("正文", posts), ("评论", comments)]:
        for result, result_cn in (("win", "胜组"), ("loss", "负组")):
            reps = side.get("representative_viewpoints_by_result", {}).get(result, {})
            lines.append(f"**{label} · {result_cn}**\n")
            for slot in ("headline", "positive_voice", "negative_voice"):
                r = reps.get(slot)
                if r:
                    lines.append(f"- [{slot_cn[slot]}] {r['viewpoint']}（{r['sentiment_category']}，{r['sentiment_polarity']}，{r['likes']} 赞）")
                    lines.append(f"  > {r['text_clean']}")
            for topic, r in reps.get("by_topic", {}).items():
                if r:
                    lines.append(f"- [议题：{topic}] {r['viewpoint']}（{r['sentiment_category']}，{r['likes']} 赞）")
                    lines.append(f"  > {r['text_clean']}")
            lines.append("")

    # 时间线
    lines.append("## 八、赛后 24 小时情感时间线（胜负组均分）\n")
    headers4 = ["时段", "正文-胜", "正文-负", "评论-胜", "评论-负"]
    rows4 = []
    for i, (_, _, bl) in enumerate(TIMELINE_BINS):
        row = [bl]
        for side in (pr, cr):
            for result in ("win", "loss"):
                tl = side.get(result, {}).get("timeline", [])
                v = tl[i]["mean_score"] if i < len(tl) else 0.0
                row.append(f"{v:+.3f}")
        rows4.append(row)
    lines.append(md_table(headers4, rows4))
    lines.append("")

    # 领域对比
    lines.append("## 九、体育舆情与通用场景（商品评论）情感表达差异分析\n")
    dc = report["domain_comparison"]
    lines.append(md_table(["维度", "商品评论（通用基线）", "体育舆情（本领域）"],
                          [[r["维度"], r["商品评论（通用基线）"], r["体育舆情（本领域）"]] for r in dc["comparison_table"]]))
    lines.append("")
    for obs in dc["data_observations"]:
        lines.append(f"- **{obs['finding']}**：{obs['evidence']}")
    lines.append("")

    # 局限
    lines.append("## 十、局限\n")
    lines.append("- 样本为微博移动端热门流，非完整档案，仅适合小规模案例比较，不能代表微博总体舆情比例。")
    lines.append("- 情感标签由 LLM 判定，9 类情绪存在主观边界（如'辛苦了'可归中性陈述或支持鼓励），已用 confidence 与人工抽检缓解。")
    lines.append("- 新加坡输球事件仅 1 条评论、中国大满贯赢球事件无可用评论，二者不参与评论侧胜负汇总。")
    lines.append("- 乱码/纯表情评论（如 dhhdjen）保留为低置信中性记录，未删除以维持样本完整。\n")

    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 2 聚合分析：胜负对比 + 议题 + 代表性观点 + 领域差异")
    parser.add_argument("--posts", type=Path, default=DEFAULT_POSTS)
    parser.add_argument("--comments", type=Path, default=DEFAULT_COMMENTS)
    parser.add_argument("--taxonomy", type=Path, default=LAB2_DIR / "03_说明与配置" / "sentiment_taxonomy.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    emotions = [e["key"] for e in taxonomy["emotions"]]
    topics = [t["key"] for t in taxonomy["topics"]]
    model = taxonomy["model"]["name"]

    posts_rows = load_rows(args.posts)
    comments_rows = load_rows(args.comments)

    posts = aggregate_side(posts_rows, "post_id", emotions, topics)
    comments = aggregate_side(comments_rows, "comment_id", emotions, topics)

    domain = build_domain_comparison(posts["by_result_rollup"], comments["by_result_rollup"], emotions)

    report = {
        "meta": {
            "model": model,
            "emotions": emotions,
            "topics": topics,
            "n_posts": len(posts_rows),
            "n_comments": len(comments_rows),
            "aggregation": "event_id 先分场统计，胜负组取各场指标均值；每场 <3 条排除",
        },
        "posts": posts,
        "comments": comments,
        "domain_comparison": domain,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sentiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(report, args.output / "舆情分析报告.md")
    print(json.dumps({
        "report": str(args.output / "sentiment_report.json"),
        "markdown": str(args.output / "舆情分析报告.md"),
        "n_posts": len(posts_rows),
        "n_comments": len(comments_rows),
        "posts_results": list(posts["by_result_rollup"].keys()),
        "comments_results": list(comments["by_result_rollup"].keys()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
