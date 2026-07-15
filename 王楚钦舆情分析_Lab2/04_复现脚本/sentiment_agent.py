"""Lab 2 情感/观点分析 agent：逐条调用 DeepSeek v4-pro（关闭思考）对 Lab 1 的微博正文与一级评论做情感与议题标注。

读取 Lab 1 的两份均衡 JSONL，逐条分类（极性 + 9 类领域情绪 + 12 类议题 + 代表性观点），
保留全部原始列并追加情感字段，输出 JSONL/CSV，供 Lab 3 生成式决策支持消费。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)


# --- 路径常量（脚本位于 王楚钦舆情分析_Lab2/04_复现脚本/，parents[2] = 仓库根） ---
SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
REPO_ROOT = LAB2_DIR.parent
LAB1_DATA = REPO_ROOT / "王楚钦舆情分析_Lab1" / "01_数据"
DEFAULT_TAXONOMY = LAB2_DIR / "03_说明与配置" / "sentiment_taxonomy.json"
DEFAULT_OUTPUT = LAB2_DIR / "01_输出"
DEFAULT_QUALITY = LAB2_DIR / "02_质量报告"
DEFAULT_POSTS = LAB1_DATA / "wang_chuqin_weibo_balanced.jsonl"
DEFAULT_COMMENTS = LAB1_DATA / "wang_chuqin_weibo_comments_balanced.jsonl"

API_KEY_PATTERN = re.compile(r"deepseek-api:\s*(sk-[A-Za-z0-9]+)")
MAX_WORKERS = 4
MAX_VALIDATION_RETRIES = 3
RATELIMIT_EXTRA_SLEEP = 2.0
SIGN = {"positive": 1, "negative": -1, "neutral": 0}

# 关思考参数的回退阶梯：probe 时按序尝试，取第一个 200 的配置。
PROBE_CONFIGS = [
    {"json_mode": True, "thinking": {"type": "disabled"}, "label": "json_mode+thinking_disabled"},
    {"json_mode": True, "thinking": None, "label": "json_mode+no_thinking_param"},
    {"json_mode": False, "thinking": {"type": "disabled"}, "label": "no_json_mode+thinking_disabled"},
    {"json_mode": False, "thinking": None, "label": "no_json_mode+no_thinking_param"},
]


# --------------------------------------------------------------------------- #
# 配置加载与 prompt 构建
# --------------------------------------------------------------------------- #
def load_taxonomy(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_system_prompt(taxonomy: dict[str, Any]) -> str:
    """把情绪/议题类别与极性定义注入 system prompt 模板。

    用 .replace() 而非 .format()：模板末尾有 JSON 示例，花括号会被 .format() 误当作占位符。
    """
    polar = taxonomy["polarities"]
    emotions_block = "\n".join(
        f"- {e['key']}：{e['definition']}" for e in taxonomy["emotions"]
    )
    topics_block = "、".join(t["key"] for t in taxonomy["topics"])
    prompt = taxonomy["system_prompt_template"]
    replacements = {
        "{positive_def}": polar["positive"],
        "{negative_def}": polar["negative"],
        "{neutral_def}": polar["neutral"],
        "{intensity_scale}": taxonomy["intensity_scale"],
        "{n_emotions}": str(len(taxonomy["emotions"])),
        "{n_topics}": str(len(taxonomy["topics"])),
        "{emotions_block}": emotions_block,
        "{topics_block}": topics_block,
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def build_user_prompt(record: dict[str, Any], taxonomy: dict[str, Any]) -> str:
    text = (record.get("text_clean") or "").strip()
    max_chars = taxonomy["model"].get("max_text_chars", 4000)
    if len(text) > max_chars:
        text = text[:max_chars]
    return taxonomy["user_prompt_template"].format(
        content_type=record.get("content_type", ""),
        match_result=record.get("match_result", ""),
        event_name=record.get("event_name", ""),
        opponent=record.get("opponent", ""),
        text_clean=text,
    )


def build_messages(record: dict[str, Any], taxonomy: dict[str, Any], system_prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(record, taxonomy)},
    ]


# --------------------------------------------------------------------------- #
# API key 与客户端
# --------------------------------------------------------------------------- #
def load_api_key() -> str:
    """优先环境变量，回退解析仓库根 api.txt；绝不硬编码。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    api_txt = REPO_ROOT / "api.txt"
    if api_txt.exists():
        for line in api_txt.read_text(encoding="utf-8").splitlines():
            m = API_KEY_PATTERN.search(line)
            if m:
                return m.group(1)
    raise RuntimeError("未找到 DEEPSEEK_API_KEY：请设置环境变量或在仓库根目录放置 api.txt")


def make_client(api_key: str, taxonomy: dict[str, Any]) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=taxonomy["model"]["base_url"],
        max_retries=taxonomy["model"].get("max_retries", 4),
    )


# --------------------------------------------------------------------------- #
# API 调用（隔离在单函数，便于替换为 httpx）
# --------------------------------------------------------------------------- #
def _build_kwargs(model: str, messages: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if cfg.get("json_mode"):
        kwargs["response_format"] = {"type": "json_object"}
    if cfg.get("thinking"):
        kwargs["extra_body"] = {"thinking": cfg["thinking"]}
    return kwargs


def call_deepseek(client: OpenAI, model: str, messages: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
    """返回 assistant 消息内容字符串；失败时抛异常由调用方处理。"""
    resp = client.chat.completions.create(**_build_kwargs(model, messages, cfg))
    return resp.choices[0].message.content or ""


def probe(client: OpenAI, taxonomy: dict[str, Any]) -> dict[str, Any]:
    """按回退阶梯探测可用请求配置：确认 key 有效、v4-pro 应答、关思考、JSON mode。"""
    model = taxonomy["model"]["name"]
    probe_messages = [
        {"role": "system", "content": "仅输出一个JSON对象 {\"ok\":1}"},
        {"role": "user", "content": "ping"},
    ]
    for cfg in PROBE_CONFIGS:
        try:
            resp = client.chat.completions.create(**_build_kwargs(model, probe_messages, cfg))
            content = resp.choices[0].message.content or ""
            reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
            obj = parse_json_content(content)
            if obj is not None:
                return {
                    "ok": True,
                    "cfg": cfg,
                    "content": content,
                    "reasoning_content": reasoning,
                    "usage": _usage(resp),
                }
        except BadRequestError:
            continue
        except (APITimeoutError, APIConnectionError, RateLimitError):
            continue
    return {"ok": False, "cfg": None, "content": "", "reasoning_content": None, "usage": {}}


def _usage(resp: Any) -> dict[str, Any]:
    try:
        u = resp.usage
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None),
        }
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# JSON 解析与校验
# --------------------------------------------------------------------------- #
def parse_json_content(content: str) -> dict[str, Any] | None:
    """容错解析：去掉代码块围栏与前后缀，再 json.loads。"""
    if not content:
        return None
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def validate_sentiment(obj: dict[str, Any], taxonomy: dict[str, Any]) -> list[str]:
    """返回错误字段列表；空列表表示通过。viewpoint/rationale 长度为软限制，不在此失败。"""
    emotions = {e["key"] for e in taxonomy["emotions"]}
    topics = {t["key"] for t in taxonomy["topics"]}
    errs: list[str] = []
    if obj.get("sentiment_polarity") not in SIGN:
        errs.append("sentiment_polarity")
    cat = obj.get("sentiment_category")
    if cat not in emotions:
        errs.append("sentiment_category")
    sec = obj.get("sentiment_category_secondary")
    if sec is not None and sec not in emotions:
        errs.append("sentiment_category_secondary")
    intensity = obj.get("sentiment_intensity")
    if not isinstance(intensity, int) or isinstance(intensity, bool) or not 1 <= intensity <= 5:
        errs.append("sentiment_intensity")
    if not isinstance(obj.get("is_mixed"), bool):
        errs.append("is_mixed")
    tags = obj.get("topic_tags")
    if not isinstance(tags, list) or not 1 <= len(tags) <= 3 or any(t not in topics for t in tags):
        errs.append("topic_tags")
    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0.0 <= float(conf) <= 1.0:
        errs.append("confidence")
    if not isinstance(obj.get("viewpoint"), str) or not obj["viewpoint"].strip():
        errs.append("viewpoint")
    if not isinstance(obj.get("rationale"), str):
        errs.append("rationale")
    return errs


# --------------------------------------------------------------------------- #
# 输出行构建
# --------------------------------------------------------------------------- #
def _to_float(v: Any) -> float:
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return 0.0


def success_row(record: dict[str, Any], obj: dict[str, Any], model: str) -> dict[str, Any]:
    polarity = obj["sentiment_polarity"]
    intensity = int(obj["sentiment_intensity"])
    score = round(SIGN[polarity] * (intensity / 5.0), 3)
    return {
        **record,
        "sentiment_polarity": polarity,
        "sentiment_category": obj["sentiment_category"],
        "sentiment_category_secondary": obj.get("sentiment_category_secondary"),
        "sentiment_intensity": intensity,
        "is_mixed": bool(obj["is_mixed"]),
        "sentiment_score": score,
        "topic_tags": list(obj["topic_tags"]),
        "viewpoint": obj["viewpoint"].strip(),
        "confidence": _to_float(obj["confidence"]),
        "rationale": obj["rationale"].strip(),
        "model": model,
        "_status": "ok",
    }


def failure_row(record: dict[str, Any], status: str, model: str, detail: str = "") -> dict[str, Any]:
    rationale = status
    if detail:
        rationale = f"{status}: {detail[:80]}"
    return {
        **record,
        "sentiment_polarity": "neutral",
        "sentiment_category": "中性陈述",
        "sentiment_category_secondary": None,
        "sentiment_intensity": 1,
        "is_mixed": False,
        "sentiment_score": 0.0,
        "topic_tags": ["其他"],
        "viewpoint": "",
        "confidence": 0.0,
        "rationale": rationale,
        "model": model,
        "_status": status,
    }


# --------------------------------------------------------------------------- #
# 单条分类（含两层重试）
# --------------------------------------------------------------------------- #
class Stats:
    """线程安全的运行统计。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._d: Counter[str] = Counter()

    def bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._d[key] += n

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return dict(self._d)


def classify_one(
    client: OpenAI,
    taxonomy: dict[str, Any],
    system_prompt: str,
    record: dict[str, Any],
    cfg: dict[str, Any],
    stats: Stats,
) -> dict[str, Any]:
    """对单条记录做情感标注；绝不抛异常中断整批，失败返回 failure_row。"""
    model = taxonomy["model"]["name"]
    base_messages = build_messages(record, taxonomy, system_prompt)
    cur_messages = list(base_messages)
    for attempt in range(1, MAX_VALIDATION_RETRIES + 1):
        try:
            content = call_deepseek(client, model, cur_messages, cfg)
        except RateLimitError:
            stats.bump("retries_total")
            time.sleep(RATELIMIT_EXTRA_SLEEP * attempt)
            continue
        except (APITimeoutError, APIConnectionError):
            stats.bump("retries_total")
            time.sleep(2.0 * attempt)
            continue
        except BadRequestError as e:
            # probe 已选定可用配置，这里一般不应触发；直接记失败。
            return failure_row(record, "API_BAD_REQUEST", model, str(e))
        except Exception as e:  # noqa: BLE001 — 兜底，绝不中断整批
            stats.bump("retries_total")
            if attempt == MAX_VALIDATION_RETRIES:
                return failure_row(record, "API_FAILED", model, str(e))
            time.sleep(1.0 * attempt)
            continue

        obj = parse_json_content(content)
        if obj is None:
            stats.bump("retries_total")
            cur_messages = base_messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": "上一次输出不是合法JSON。请仅重新输出修正后的JSON对象。"},
            ]
            continue
        errs = validate_sentiment(obj, taxonomy)
        if errs:
            stats.bump("retries_total")
            cur_messages = base_messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"上一次输出不符合要求，错误字段：{errs}。请仅重新输出修正后的JSON对象。"},
            ]
            continue
        return success_row(record, obj, model)

    return failure_row(record, "API_PARSE_FAILED", model)


# --------------------------------------------------------------------------- #
# checkpoint 与 IO
# --------------------------------------------------------------------------- #
def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_checkpoint(path: Path, id_field: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[str(row.get(id_field, ""))] = row
    return done


def append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _csv_value(v: Any) -> Any:
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


def write_outputs(rows: list[dict[str, Any]], jsonl_path: Path, csv_path: Path) -> None:
    # _status 是内部审计字段，不写入最终交付（checkpoint 保留以支持断点续跑）。
    clean = [{k: v for k, v in row.items() if k != "_status"} for row in rows]
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in clean:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if clean:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(clean[0].keys()))
            writer.writeheader()
            for row in clean:
                writer.writerow({k: _csv_value(v) for k, v in row.items()})


# --------------------------------------------------------------------------- #
# 单数据集处理
# --------------------------------------------------------------------------- #
def run_dataset(
    client: OpenAI,
    taxonomy: dict[str, Any],
    system_prompt: str,
    cfg: dict[str, Any],
    records: list[dict[str, Any]],
    id_field: str,
    checkpoint_path: Path,
    jsonl_path: Path,
    csv_path: Path,
    label: str,
    limit: int | None,
    redo: bool,
    stats: Stats,
) -> list[dict[str, Any]]:
    if redo and checkpoint_path.exists():
        checkpoint_path.unlink()

    done = load_checkpoint(checkpoint_path, id_field)
    todo = [r for r in records if str(r.get(id_field, "")) not in done]
    if limit is not None:
        todo = todo[:limit]

    total = len(records)
    stats.bump(f"{label}_total", total)
    stats.bump(f"{label}_already_done", len(done))
    stats.bump(f"{label}_todo", len(todo))
    print(f"[{label}] 共 {total} 条，已完成 {len(done)}，待标注 {len(todo)}")

    results: dict[str, dict[str, Any]] = {k: v for k, v in done.items()}
    if todo:
        completed = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            future_map = {
                ex.submit(classify_one, client, taxonomy, system_prompt, rec, cfg, stats): rec
                for rec in todo
            }
            for fut in as_completed(future_map):
                rec = future_map[fut]
                rid = str(rec.get(id_field, ""))
                try:
                    row = fut.result()
                except Exception as e:  # noqa: BLE001
                    row = failure_row(rec, "UNEXPECTED", taxonomy["model"]["name"], str(e))
                    traceback.print_exc()
                results[rid] = row
                append_checkpoint(checkpoint_path, row)
                completed += 1
                status = row.get("_status", "ok")
                if status != "ok":
                    stats.bump(f"{label}_failed")
                if completed % 10 == 0 or completed == len(todo):
                    print(f"[{label}] 进度 {completed}/{len(todo)}")

    # 确定性排序，写正式输出
    ordered = list(results.values())
    ordered.sort(key=lambda r: (r.get("event_id", ""), r.get("publish_time", ""), str(r.get(id_field, ""))))
    write_outputs(ordered, jsonl_path, csv_path)
    return ordered


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 2 情感/观点分析 agent（DeepSeek v4-pro，关闭思考）")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--posts", type=Path, default=DEFAULT_POSTS)
    parser.add_argument("--comments", type=Path, default=DEFAULT_COMMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--limit", type=int, default=None, help="每个数据集只处理前 N 条（测试用）")
    parser.add_argument("--redo", action="store_true", help="清空 checkpoint 重新标注")
    parser.add_argument("--probe", action="store_true", help="仅探测 API 配置并打印，不跑标注")
    parser.add_argument("--posts-only", action="store_true")
    parser.add_argument("--comments-only", action="store_true")
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    system_prompt = build_system_prompt(taxonomy)
    api_key = load_api_key()
    client = make_client(api_key, taxonomy)

    args.output.mkdir(parents=True, exist_ok=True)
    args.quality.mkdir(parents=True, exist_ok=True)

    # --- probe 模式 ---
    if args.probe:
        print("=== probe DeepSeek API ===")
        result = probe(client, taxonomy)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["ok"]:
            print("probe 失败：所有请求配置均不可用，请检查 key/网络/模型名。")
        else:
            print(f"probe 成功，采用配置：{result['cfg']['label']}")
            rc = result.get("reasoning_content")
            print(f"reasoning_content={'(空，思考已关闭)' if not rc else '(非空！思考可能未关闭)'}")
        return

    # --- 先 probe 选定配置 ---
    print("=== 探测可用请求配置 ===")
    probe_result = probe(client, taxonomy)
    if not probe_result["ok"]:
        raise RuntimeError("probe 失败：所有请求配置均不可用，请用 --probe 排查。")
    cfg = probe_result["cfg"]
    print(f"采用配置：{cfg['label']}")
    rc = probe_result.get("reasoning_content")
    print(f"reasoning_content={'(空，思考已关闭)' if not rc else '(非空！思考可能未关闭)'}")

    stats = Stats()
    run_posts = not args.comments_only
    run_comments = not args.posts_only

    all_rows: list[dict[str, Any]] = []

    if run_posts and args.posts.exists():
        posts = load_records(args.posts)
        all_rows += run_dataset(
            client, taxonomy, system_prompt, cfg, posts, "post_id",
            args.output / ".posts_sentiment_checkpoint.jsonl",
            args.output / "posts_sentiment.jsonl",
            args.output / "posts_sentiment.csv",
            "posts", args.limit, args.redo, stats,
        )
    elif run_posts:
        print(f"[posts] 输入不存在：{args.posts}")

    if run_comments and args.comments.exists():
        comments = load_records(args.comments)
        all_rows += run_dataset(
            client, taxonomy, system_prompt, cfg, comments, "comment_id",
            args.output / ".comments_sentiment_checkpoint.jsonl",
            args.output / "comments_sentiment.jsonl",
            args.output / "comments_sentiment.csv",
            "comments", args.limit, args.redo, stats,
        )
    elif run_comments:
        print(f"[comments] 输入不存在：{args.comments}")

    # --- 标签审计 ---
    audit = build_label_audit(all_rows, taxonomy)
    (args.quality / "label_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- api_stats ---
    api_stats = {
        "model": taxonomy["model"]["name"],
        "request_config": cfg["label"],
        "thinking_disabled_reasoning_content_empty": not bool(probe_result.get("reasoning_content")),
        "json_mode": cfg["json_mode"],
        **stats.as_dict(),
    }
    (args.quality / "api_stats.json").write_text(
        json.dumps(api_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== api_stats ===")
    print(json.dumps(api_stats, ensure_ascii=False, indent=2))


def build_label_audit(rows: list[dict[str, Any]], taxonomy: dict[str, Any]) -> dict[str, Any]:
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        ctype = row.get("content_type", "unknown")
        by_type[ctype]["n"] += 1
        by_type[ctype][f"polarity_{row.get('sentiment_polarity', '?')}"] += 1
        by_type[ctype][f"category_{row.get('sentiment_category', '?')}"] += 1
        by_type[ctype]["low_confidence"] += 1 if float(row.get("confidence", 1.0)) < 0.4 else 0
        by_type[ctype]["failed"] += 1 if row.get("_status", "ok") != "ok" else 0
    return {
        "total_rows": len(rows),
        "by_content_type": {k: dict(v) for k, v in by_type.items()},
    }


if __name__ == "__main__":
    main()
