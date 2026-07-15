"""Lab 2 可视化：从 sentiment_report.json 生成 4 张图表（胜负情感分布、领域情绪类别、议题频次、赛后时间线）。

供第四天汇报演示用。中文字体使用微软雅黑/黑体。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
DEFAULT_REPORT = LAB2_DIR / "01_输出" / "sentiment_report.json"
DEFAULT_FIGDIR = LAB2_DIR / "05_图表"

# Windows 中文字体
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

WIN_COLOR = "#4C9F70"   # 胜——绿
LOSS_COLOR = "#D9534F"  # 负——红
SIDE_COLORS = {"win": WIN_COLOR, "loss": LOSS_COLOR}
POLARITY_ORDER = ["positive", "negative", "neutral"]
POLARITY_CN = {"positive": "正面", "negative": "负面", "neutral": "中性"}


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _roll(report: dict[str, Any], side: str, result: str) -> dict[str, Any]:
    return report[side]["by_result_rollup"].get(result, {})


# --------------------------------------------------------------------------- #
# 图1：情感极性分布（正文 | 评论，各一个子图，胜负分组柱）
# --------------------------------------------------------------------------- #
def fig_polarity(report: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, side, title in zip(axes, ("posts", "comments"), ("微博正文", "一级评论")):
        x = range(len(POLARITY_ORDER))
        width = 0.38
        win_vals = [_roll(report, side, "win").get(f"{p}_pct", 0) for p in POLARITY_ORDER]
        loss_vals = [_roll(report, side, "loss").get(f"{p}_pct", 0) for p in POLARITY_ORDER]
        ax.bar([i - width / 2 for i in x], win_vals, width, label="胜", color=WIN_COLOR)
        ax.bar([i + width / 2 for i in x], loss_vals, width, label="负", color=LOSS_COLOR)
        ax.set_xticks(list(x))
        ax.set_xticklabels([POLARITY_CN[p] for p in POLARITY_ORDER])
        ax.set_title(title)
        ax.set_ylabel("占比 (%)")
        ax.legend()
        ax.set_ylim(0, 100)
        for i, v in enumerate(win_vals):
            ax.text(i - width / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
        for i, v in enumerate(loss_vals):
            ax.text(i + width / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    fig.suptitle("图1 情感极性分布：胜 vs 负（事件均值占比）")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 图2：领域情绪类别分布（正文 | 评论，9 类横向分组柱）
# --------------------------------------------------------------------------- #
def fig_emotion(report: dict[str, Any], out: Path) -> None:
    emotions = report["meta"]["emotions"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, side, title in zip(axes, ("posts", "comments"), ("微博正文", "一级评论")):
        y = range(len(emotions))
        height = 0.38
        win_vals = [_roll(report, side, "win").get("emotion_dist_pct", {}).get(e, 0) for e in emotions]
        loss_vals = [_roll(report, side, "loss").get("emotion_dist_pct", {}).get(e, 0) for e in emotions]
        ax.barh([i - height / 2 for i in y], win_vals, height, label="胜", color=WIN_COLOR)
        ax.barh([i + height / 2 for i in y], loss_vals, height, label="负", color=LOSS_COLOR)
        ax.set_yticks(list(y))
        ax.set_yticklabels(emotions)
        ax.set_title(title)
        ax.set_xlabel("占比 (%)")
        ax.legend()
        ax.invert_yaxis()
    fig.suptitle("图2 领域情绪类别分布：胜 vs 负（事件均值占比）")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 图3：议题频次（正文 | 评论，Top8 分组柱）
# --------------------------------------------------------------------------- #
def fig_topic(report: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, side, title in zip(axes, ("posts", "comments"), ("微博正文", "一级评论")):
        win_rate = _roll(report, side, "win").get("topic_mention_rate", {})
        loss_rate = _roll(report, side, "loss").get("topic_mention_rate", {})
        all_topics = set(win_rate) | set(loss_rate)
        ranked = sorted(all_topics, key=lambda t: -(win_rate.get(t, 0) + loss_rate.get(t, 0)))[:8]
        x = range(len(ranked))
        width = 0.38
        win_vals = [win_rate.get(t, 0) * 100 for t in ranked]
        loss_vals = [loss_rate.get(t, 0) * 100 for t in ranked]
        ax.bar([i - width / 2 for i in x], win_vals, width, label="胜", color=WIN_COLOR)
        ax.bar([i + width / 2 for i in x], loss_vals, width, label="负", color=LOSS_COLOR)
        ax.set_xticks(list(x))
        ax.set_xticklabels(ranked, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel("事件均值提及率 (%)")
        ax.legend()
    fig.suptitle("图3 议题分布 Top8：胜 vs 负")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 图4：赛后 24h 情感时间线（4 条线）
# --------------------------------------------------------------------------- #
def fig_timeline(report: dict[str, Any], out: Path) -> None:
    bins = ["0-2h", "2-6h", "6-12h", "12-24h"]
    fig, ax = plt.subplots(figsize=(9, 5))
    series = [
        ("正文-胜", "posts", "win", WIN_COLOR, "o"),
        ("正文-负", "posts", "loss", LOSS_COLOR, "o"),
        ("评论-胜", "comments", "win", WIN_COLOR, "s"),
        ("评论-负", "comments", "loss", LOSS_COLOR, "s"),
    ]
    for label, side, result, color, marker in series:
        tl = _roll(report, side, result).get("timeline", [])
        vals = [b["mean_score"] for b in tl]
        linestyle = "-" if side == "posts" else "--"
        ax.plot(bins[: len(vals)], vals, marker=marker, linestyle=linestyle, color=color, label=label)
    ax.axhline(0, color="#999", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("情感均分 (sentiment_score)")
    ax.set_xlabel("赛后时长")
    ax.set_title("图4 赛后 24 小时情感时间线（胜负组均分）")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 2 可视化：生成 4 张图表")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR)
    args = parser.parse_args()

    report = load_report(args.report)
    args.figdir.mkdir(parents=True, exist_ok=True)
    fig_polarity(report, args.figdir / "fig_polarity_distribution.png")
    fig_emotion(report, args.figdir / "fig_emotion_category.png")
    fig_topic(report, args.figdir / "fig_topic_frequency.png")
    fig_timeline(report, args.figdir / "fig_sentiment_timeline.png")
    print(json.dumps({
        "figdir": str(args.figdir),
        "charts": [
            "fig_polarity_distribution.png",
            "fig_emotion_category.png",
            "fig_topic_frequency.png",
            "fig_sentiment_timeline.png",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
