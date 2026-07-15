"""Deterministic offline decision support built only from evidence packets."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

from .models import EvidencePacket, GeneratedResult, MetricSummary


PRESET_QUESTIONS = MappingProxyType(
    {
        "loss_all_negative": "输球是否意味着舆情全部负面？",
        "source_difference": "正文与评论呈现出哪些差异？",
        "top_topics": "当前样本主要讨论哪些主题？",
        "representative_views": "当前证据中有哪些代表性观点？",
        "coverage_limits": "当前分析覆盖了什么，又有哪些边界？",
    }
)

STRATEGY_GOALS = (
    "回应争议",
    "稳定球迷情绪",
    "准备媒体简报",
    "内部复盘",
)

_CASE_LIMITATION = "案例样本，不能代表微博总体舆情。"
_POLARITY_LABELS = {
    "positive": "正面",
    "neutral": "中性",
    "negative": "负面",
}


def brief_offline(packet: EvidencePacket) -> GeneratedResult:
    """Create a deterministic offline brief without adding outside facts."""

    return GeneratedResult(
        mode="offline",
        payload={
            "title": f"{packet.label}离线舆情简报",
            "facts": packet.facts,
            "observations": _observations(packet),
            "decision_focus": (
                f"围绕“{packet.label}”核对事实、来源差异与可引用证据。",
                (
                    f"面向{packet.scope.audience}使用时，应把样本观察与"
                    "人工决策分开。"
                ),
            ),
            "limitations": _limitations(packet),
            "citation_ids": _citation_ids(packet),
        },
    )


def answer_offline(
    question_key: str,
    packet: EvidencePacket,
) -> GeneratedResult:
    """Answer a supported preset solely from fields in ``packet``."""

    if question_key not in PRESET_QUESTIONS:
        return GeneratedResult(
            mode="offline",
            payload={
                "question": question_key,
                "answerable": False,
                "facts": (),
                "interpretation": (
                    "离线模式无法可靠回答该问题；请改用下列可用预设，"
                    "或由人工补充证据。"
                ),
                "limitations": _limitations(packet),
                "citation_ids": (),
                "available_questions": tuple(PRESET_QUESTIONS),
            },
        )

    handlers = {
        "loss_all_negative": _answer_loss_all_negative,
        "source_difference": _answer_source_difference,
        "top_topics": _answer_top_topics,
        "representative_views": _answer_representative_views,
        "coverage_limits": _answer_coverage_limits,
    }
    answerable, interpretation, citation_ids = handlers[question_key](packet)
    return GeneratedResult(
        mode="offline",
        payload={
            "question": PRESET_QUESTIONS[question_key],
            "answerable": answerable,
            "facts": packet.facts,
            "interpretation": interpretation,
            "limitations": _limitations(packet),
            "citation_ids": citation_ids,
        },
    )


def strategies_offline(
    packet: EvidencePacket,
    goal: str,
    audience: str,
) -> GeneratedResult:
    """Return three qualitative options for a human decision maker."""

    evidence_ids = _citation_ids(packet)
    evidence_check = (
        ("当前证据充足性仍须由人工复核。",)
        if evidence_ids
        else ("证据不足：当前证据包无可用引文，执行前须补充人工核验。",)
    )
    options = (
        {
            "name": "及时情绪回应",
            "intensity": "轻量快速",
            "action": (
                f"面向{audience}回应可确认的情绪关切，重申当前事实与"
                "证据边界，不扩展未核验判断。"
            ),
            "timing": "争议出现且事实边界已确认时及时启动。",
            "evidence_ids": evidence_ids,
            "benefits": (
                "减少信息真空，先回应受众的直接关切。",
                "为后续事实说明保留空间。",
            ),
            "risks": (
                "回应过快可能遗漏必要背景。",
                "安抚措辞可能被理解为回避事实。",
            ),
            "checks": (
                "逐项核对回应中的事实与引文是否对应。",
                "由人工确认语气适合目标受众。",
            )
            + evidence_check,
        },
        {
            "name": "事实说明与复盘",
            "intensity": "完整说明",
            "action": (
                f"围绕“{goal}”整理已确认事实，分开呈现正文与评论的"
                "样本观察，并说明仍待核验之处。"
            ),
            "timing": "事实与表述完成交叉核验后发布或用于内部复盘。",
            "evidence_ids": evidence_ids,
            "benefits": (
                "信息结构完整，便于解释证据与判断的边界。",
                "可沉淀为后续沟通和复盘材料。",
            ),
            "risks": (
                "准备周期较长，可能错过早期回应窗口。",
                "细节过多可能放大次要争议。",
            ),
            "checks": (
                "确认正文指标与评论指标没有混用。",
                "确认每项判断均可追溯到当前数据包。",
            )
            + evidence_check,
        },
        {
            "name": "持续监测",
            "intensity": "持续观察",
            "action": (
                "暂不扩大结论，持续观察同一范围内的新证据，并记录"
                "主题、情绪和来源是否出现描述性变化。"
            ),
            "timing": "在后续观察期持续进行，并在证据变化时人工复核。",
            "evidence_ids": evidence_ids,
            "benefits": (
                "降低基于薄弱样本仓促表态的风险。",
                "便于发现需要进一步核验的变化。",
            ),
            "risks": (
                "暂缓公开回应可能延长信息真空。",
                "持续投入监测会占用人工复核资源。",
            ),
            "checks": (
                "保持事件范围与来源口径一致。",
                "由人工决定何时升级为公开回应或正式复盘。",
            )
            + evidence_check,
        },
    )
    return GeneratedResult(
        mode="offline",
        payload={
            "goal": goal,
            "audience": audience,
            "options": options,
            "disclaimer": (
                "以上仅为基于当前案例证据的定性情景比较，非预测；"
                "最终方案必须由人工决定并在执行前复核。"
            ),
        },
    )


def _answer_loss_all_negative(
    packet: EvidencePacket,
) -> tuple[bool, str, tuple[str, ...]]:
    comments = packet.comments
    if packet.scope.kind != "loss_group":
        return (
            False,
            "当前证据范围不是负场组，无法可靠判断该预设问题。",
            (),
        )
    if comments is None or comments.n == 0:
        return (
            False,
            "当前没有可用评论指标，无法判断负场评论的正中负构成。",
            (),
        )

    polarity = comments.polarity_pct
    comment_ids = tuple(
        citation.record_id
        for citation in packet.citations
        if citation.content_type == "comment"
    )
    interpretation = (
        "输球不等于全部负面。当前评论样本的极性构成为："
        f"正面 {polarity['positive']:.4f}%、"
        f"中性 {polarity['neutral']:.4f}%、"
        f"负面 {polarity['negative']:.4f}%。"
        "该结论只描述当前案例评论，并以当前评论引文为核验入口。"
    )
    return True, interpretation, comment_ids


def _answer_source_difference(
    packet: EvidencePacket,
) -> tuple[bool, str, tuple[str, ...]]:
    posts = packet.posts
    comments = packet.comments
    if (
        posts is None
        or comments is None
        or posts.n == 0
        or comments.n == 0
    ):
        return (
            False,
            "当前证据未同时提供可用的正文与评论指标，无法可靠比较来源。",
            (),
        )

    interpretation = (
        f"正文样本：{_polarity_mix(posts)}；"
        f"评论样本：{_polarity_mix(comments)}。"
        "两类来源按各自口径分开呈现，只作描述性比较，不作因果解释。"
    )
    return True, interpretation, _citation_ids(packet)


def _answer_top_topics(
    packet: EvidencePacket,
) -> tuple[bool, str, tuple[str, ...]]:
    topic_lines = []
    for label, summary in _labeled_summaries(packet):
        if not summary.top_topics:
            continue
        topics = "、".join(
            f"{name}（{value:.4f}%）"
            for name, value in summary.top_topics
        )
        topic_lines.append(f"{label}的主要主题依次为{topics}")
    if not topic_lines:
        return False, "当前证据没有可用主题指标，无法归纳主要主题。", ()
    return (
        True,
        "；".join(topic_lines) + "。以上仅描述当前样本的主题提及情况。",
        _citation_ids(packet),
    )


def _answer_representative_views(
    packet: EvidencePacket,
) -> tuple[bool, str, tuple[str, ...]]:
    if not packet.citations:
        return False, "当前证据包无可用引文，无法列示代表性观点。", ()
    views = "；".join(
        f"[{citation.record_id}] {citation.text}"
        for citation in packet.citations
    )
    return (
        True,
        (
            f"当前代表性引文按证据包顺序列示：{views}。"
            "这些引文仅呈现样本内观点，不代表总体分布。"
        ),
        _citation_ids(packet),
    )


def _answer_coverage_limits(
    packet: EvidencePacket,
) -> tuple[bool, str, tuple[str, ...]]:
    limitations = _limitations(packet)
    return (
        True,
        "当前证据的覆盖边界为：" + "；".join(limitations),
        (),
    )


def _observations(packet: EvidencePacket) -> tuple[str, ...]:
    observations = tuple(
        _summary_observation(label, summary)
        for label, summary in _labeled_summaries(packet)
    )
    if observations:
        return observations
    return ("当前范围没有可供描述性归纳的来源指标。",)


def _summary_observation(label: str, summary: MetricSummary) -> str:
    if summary.n == 0:
        return f"{label}无可用记录，无法归纳该来源的情绪或主题倾向。"

    polarity_name, polarity_value = max(
        summary.polarity_pct.items(),
        key=lambda item: item[1],
    )
    parts = [
        (
            f"{label}共 {summary.n} 条，占比最高的极性为"
            f"{_POLARITY_LABELS.get(polarity_name, polarity_name)}"
            f"（{polarity_value:.4f}%）"
        )
    ]
    if summary.top_emotions:
        emotion, value = summary.top_emotions[0]
        parts.append(f"最高情绪标签为{emotion}（{value:.4f}%）")
    if summary.top_topics:
        topic, value = summary.top_topics[0]
        parts.append(f"最高主题提及为{topic}（{value:.4f}%）")
    return "，".join(parts) + "；仅作当前样本的描述性归纳。"


def _labeled_summaries(
    packet: EvidencePacket,
) -> Iterable[tuple[str, MetricSummary]]:
    if packet.posts is not None:
        yield "正文", packet.posts
    if packet.comments is not None:
        yield "评论", packet.comments
    if packet.post_comparison is not None:
        yield "正文胜组", packet.post_comparison.win
        yield "正文负组", packet.post_comparison.loss
    if packet.comment_comparison is not None:
        yield "评论胜组", packet.comment_comparison.win
        yield "评论负组", packet.comment_comparison.loss


def _polarity_mix(summary: MetricSummary) -> str:
    polarity = summary.polarity_pct
    return (
        f"n={summary.n}，正面 {polarity['positive']:.4f}%、"
        f"中性 {polarity['neutral']:.4f}%、"
        f"负面 {polarity['negative']:.4f}%"
    )


def _limitations(packet: EvidencePacket) -> tuple[str, ...]:
    limitations = list(packet.warnings)
    if not any("不能代表微博总体舆情" in item for item in limitations):
        limitations.append(_CASE_LIMITATION)
    return tuple(limitations)


def _citation_ids(packet: EvidencePacket) -> tuple[str, ...]:
    return tuple(citation.record_id for citation in packet.citations)
