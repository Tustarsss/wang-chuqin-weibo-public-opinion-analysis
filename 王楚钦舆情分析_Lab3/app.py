"""Streamlit interface for evidence-constrained Lab 3 decision support."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import streamlit as st

from lab3.data_loader import DataContractError, ProjectData, load_project_data
from lab3.evidence import build_evidence
from lab3.export import export_markdown
from lab3.llm_client import resolve_api_key
from lab3.models import (
    AnalysisScope,
    EvidencePacket,
    GeneratedResult,
    MetricSummary,
)
from lab3.offline import (
    PRESET_QUESTIONS,
    STRATEGY_GOALS,
    brief_offline,
    strategies_offline,
)
from lab3.services import BriefService, QAService, StrategyService
from lab3.ui_helpers import (
    AUDIENCE_VALUES,
    SCOPE_VALUES,
    SOURCE_VALUES,
    citation_lookup,
    context_key,
    event_options,
    metric_chart_rows,
    metric_rows,
    synchronize_context,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
_DOWNLOAD_NAME = "wang_chuqin_lab3_brief.md"
_CASE_NOTICE = "案例样本声明：当前 8 场赛事样本不能代表微博总体舆情。"
_CSS = """
<style>
    :root { --lab-blue: #1557a0; --lab-pale: #eef6ff; }
    .stApp { background: #f7faff; }
    [data-testid="stSidebar"] { background: #edf5ff; }
    h1, h2, h3 { color: var(--lab-blue); }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #d8e8fb;
        border-radius: 0.65rem;
        padding: 0.45rem 0.65rem;
    }
    div[data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #c9ddf5;
    }
</style>
"""


st.set_page_config(
    page_title="王楚钦舆情三合一决策台",
    page_icon="📊",
    layout="wide",
)
st.markdown(_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _cached_project_data(repo_root: str) -> ProjectData:
    return load_project_data(repo_root)


def _load_data_or_stop() -> ProjectData:
    try:
        return _cached_project_data(str(REPO_ROOT))
    except DataContractError as exc:
        st.error(f"数据契约校验失败：{exc}")
        st.stop()
        raise RuntimeError("unreachable after st.stop") from exc


def _sidebar_scope(data: ProjectData) -> AnalysisScope:
    with st.sidebar:
        st.header("分析设置")
        scope_label = st.selectbox(
            "分析范围",
            tuple(SCOPE_VALUES),
            key="scope_selector",
        )
        kind = SCOPE_VALUES[scope_label]

        selected_event_id: str | None = None
        if kind == "single_event":
            options = event_options(data)
            labels = {event_id: label for label, event_id in options}
            selected_event_id = st.selectbox(
                "单场赛事",
                tuple(labels),
                format_func=labels.__getitem__,
                key="event_selector",
            )

        source_label = st.selectbox(
            "来源",
            tuple(SOURCE_VALUES),
            key="source_selector",
        )
        audience_label = st.selectbox(
            "受众",
            tuple(AUDIENCE_VALUES),
            key="audience_selector",
        )

        st.divider()
        st.subheader("案例数据状态")
        first, second, third = st.columns(3)
        first.metric("赛事", len(data.events))
        second.metric("正文", len(data.posts))
        third.metric("评论", len(data.comments))
        st.write(f"ingestion 状态：{data.ingestion.get('status', '未知')}")
        key_status = (
            "已配置" if resolve_api_key(REPO_ROOT) is not None else "未配置"
        )
        st.write(f"DeepSeek key：{key_status}")

    return AnalysisScope(
        kind=kind,
        source=SOURCE_VALUES[source_label],
        audience=AUDIENCE_VALUES[audience_label],
        event_id=selected_event_id,
    )


def _write_items(value: Any, empty: str = "无可用内容。") -> None:
    items = _as_items(value)
    if not items:
        st.write(empty)
        return
    for item in items:
        st.write(f"• {item}")


def _as_items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _render_status(result: GeneratedResult) -> None:
    st.caption(f"mode: {result.mode}")
    if result.warning:
        st.warning(result.warning)


def _render_citation_ids(
    packet: EvidencePacket,
    citation_ids: Any,
) -> None:
    ids = tuple(str(item) for item in _as_items(citation_ids))
    if not ids:
        st.write("证据 ID：无可用引文。")
        return
    st.write("证据 ID：" + "、".join(ids))
    for citation_id in ids:
        citation = citation_lookup(packet, citation_id)
        if citation is None:
            continue
        source = "正文" if citation.content_type == "post" else "评论"
        with st.expander(f"证据 {citation.record_id}｜{source}"):
            st.write(f"事件：{citation.event_name}（{citation.event_id}）")
            st.write(citation.text)
            st.write(
                f"极性：{citation.polarity}；情绪：{citation.emotion}；"
                f"主题：{'、'.join(citation.topics) or '未标注'}"
            )
            st.write(
                f"置信度：{citation.confidence:.2f}；点赞：{citation.likes}"
            )


def _packet_summaries(
    packet: EvidencePacket,
) -> tuple[tuple[str, MetricSummary], ...]:
    summaries: list[tuple[str, MetricSummary]] = []
    if packet.scope.kind == "win_loss_comparison":
        if packet.post_comparison is not None:
            summaries.extend(
                (
                    ("正文｜胜组", packet.post_comparison.win),
                    ("正文｜负组", packet.post_comparison.loss),
                )
            )
        if packet.comment_comparison is not None:
            summaries.extend(
                (
                    ("评论｜胜组", packet.comment_comparison.win),
                    ("评论｜负组", packet.comment_comparison.loss),
                )
            )
        return tuple(summaries)

    if packet.posts is not None:
        summaries.append(("正文", packet.posts))
    if packet.comments is not None:
        summaries.append(("评论", packet.comments))
    return tuple(summaries)


def _rank_text(values: Sequence[tuple[str, float]]) -> str:
    if not values:
        return "无可用数据"
    return "、".join(f"{name} {value:.2f}%" for name, value in values)


def _render_metrics(packet: EvidencePacket) -> None:
    rows = metric_rows(packet)
    st.subheader("样本指标")
    st.dataframe(rows, hide_index=True, width="stretch")
    chart_rows = metric_chart_rows(packet)
    if chart_rows:
        st.bar_chart(
            chart_rows,
            x="样本",
            y=["正面%", "中性%", "负面%"],
            stack=True,
            color=["#2e8b57", "#8fa6b8", "#d95f5f"],
            height=340,
        )

    st.subheader("主要情绪与议题")
    for label, summary in _packet_summaries(packet):
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.write(f"Top 情绪：{_rank_text(summary.top_emotions)}")
            st.write(f"Top 议题：{_rank_text(summary.top_topics)}")


def _render_brief(packet: EvidencePacket) -> None:
    _render_metrics(packet)
    if "brief" not in st.session_state:
        st.session_state["brief"] = brief_offline(packet)

    if st.button("生成/刷新简报", type="primary", key="generate_brief"):
        with st.spinner("正在基于当前证据生成简报……"):
            st.session_state["brief"] = BriefService(
                repo_root=REPO_ROOT
            ).generate(packet)

    brief: GeneratedResult = st.session_state["brief"]
    st.subheader(str(brief.payload.get("title", packet.label)))
    _render_status(brief)
    for heading, field in (
        ("事实", "facts"),
        ("观察", "observations"),
        ("决策关注", "decision_focus"),
        ("局限", "limitations"),
    ):
        st.markdown(f"#### {heading}")
        _write_items(brief.payload.get(field))
    st.markdown("#### 引用")
    _render_citation_ids(packet, brief.payload.get("citation_ids"))


def _append_qa_message(
    question: str,
    result: GeneratedResult,
) -> None:
    messages = st.session_state.setdefault("messages", [])
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "result": result})


def _render_qa_result(
    packet: EvidencePacket,
    result: GeneratedResult,
) -> None:
    _render_status(result)
    answerable = "是" if result.payload.get("answerable") else "否"
    st.write(f"可回答：{answerable}")
    st.markdown("**事实**")
    _write_items(result.payload.get("facts"))
    st.markdown("**解释**")
    st.write(result.payload.get("interpretation", "无可用解释。"))
    st.markdown("**局限**")
    _write_items(result.payload.get("limitations"))
    _render_citation_ids(packet, result.payload.get("citation_ids"))


def _render_qa(packet: EvidencePacket) -> None:
    st.subheader("证据约束问答")
    preset_key = st.selectbox(
        "预设问题（5 个）",
        tuple(PRESET_QUESTIONS),
        format_func=PRESET_QUESTIONS.__getitem__,
        key="preset_question",
    )
    if st.button("回答预设问题", type="primary", key="answer_preset"):
        with st.spinner("正在核对当前证据……"):
            result = QAService(repo_root=REPO_ROOT).answer(
                preset_key,
                packet,
                preset=True,
            )
        _append_qa_message(PRESET_QUESTIONS[preset_key], result)

    for message in st.session_state.setdefault("messages", []):
        if message.get("role") == "user":
            with st.chat_message("user"):
                st.write(message.get("content", ""))
            continue
        result = message.get("result")
        if isinstance(result, GeneratedResult):
            with st.chat_message("assistant"):
                _render_qa_result(packet, result)

    question = st.chat_input("输入自由问题；证据不足时系统会明确说明")
    if question and question.strip():
        with st.spinner("正在核对当前证据……"):
            result = QAService(repo_root=REPO_ROOT).answer(
                question.strip(),
                packet,
                preset=False,
            )
        _append_qa_message(question.strip(), result)
        st.rerun()


def _render_option(
    packet: EvidencePacket,
    option: Mapping[str, Any],
) -> None:
    with st.container(border=True):
        st.subheader(str(option.get("name", "未命名方案")))
        st.caption(f"力度：{option.get('intensity', '未标注')}")
        st.markdown("**行动**")
        st.write(option.get("action", "无可用内容。"))
        st.markdown("**时机**")
        st.write(option.get("timing", "无可用内容。"))
        left, right = st.columns(2)
        with left:
            st.markdown("**收益**")
            _write_items(option.get("benefits"))
        with right:
            st.markdown("**风险**")
            _write_items(option.get("risks"))
        st.markdown("**核验**")
        _write_items(option.get("checks"))
        _render_citation_ids(packet, option.get("evidence_ids"))


def _render_download(
    packet: EvidencePacket,
    brief: GeneratedResult,
    strategies: GeneratedResult,
) -> None:
    selected = st.session_state.get("human_choice", "")
    if selected == "尚未选择":
        selected = ""
    report = export_markdown(
        packet,
        brief,
        strategies,
        selected_option=selected,
        human_note=st.session_state.get("human_note", ""),
    )
    st.download_button(
        "下载当前决策简报",
        data=report,
        file_name=_DOWNLOAD_NAME,
        mime="text/markdown; charset=utf-8",
        key="download_report",
    )


def _render_strategies(packet: EvidencePacket) -> None:
    st.subheader("三方案定性比较")
    goal = st.selectbox("沟通目标", STRATEGY_GOALS, key="strategy_goal")
    current = st.session_state.get("strategies")
    if not isinstance(current, GeneratedResult) or current.payload.get(
        "goal"
    ) != goal:
        st.session_state["strategies"] = strategies_offline(
            packet,
            goal=goal,
            audience=packet.scope.audience,
        )

    if st.button("生成/刷新方案", type="primary", key="generate_strategies"):
        with st.spinner("正在生成三方案定性比较……"):
            st.session_state["strategies"] = StrategyService(
                repo_root=REPO_ROOT
            ).generate(packet, goal=goal, audience=packet.scope.audience)
        st.session_state.pop("human_choice", None)
        st.session_state.pop("human_note", None)

    strategies: GeneratedResult = st.session_state["strategies"]
    _render_status(strategies)
    for option in _as_items(strategies.payload.get("options")):
        if isinstance(option, Mapping):
            _render_option(packet, option)

    st.markdown("#### 局限")
    _write_items(strategies.payload.get("limitations"))
    st.markdown("#### 方案声明")
    st.write(strategies.payload.get("disclaimer", "无可用声明。"))

    option_names = tuple(
        str(option.get("name"))
        for option in _as_items(strategies.payload.get("options"))
        if isinstance(option, Mapping) and option.get("name")
    )
    st.divider()
    st.subheader("人工决定")
    st.selectbox(
        "人工选择",
        ("尚未选择", *option_names),
        key="human_choice",
    )
    st.text_area(
        "人工备注",
        key="human_note",
        placeholder="记录复核结论、责任人或下一步；不会自动发布。",
    )
    _render_download(packet, st.session_state["brief"], strategies)


data = _load_data_or_stop()
scope = _sidebar_scope(data)
synchronize_context(st.session_state, context_key(scope))
packet = build_evidence(data, scope)

st.title("王楚钦微博舆情三合一决策台")
st.caption("证据约束 · 离线可用 · 人工最终决策")
st.header(packet.label)
st.info(_CASE_NOTICE)
for coverage_warning in packet.warnings:
    st.warning(coverage_warning)

if "brief" not in st.session_state:
    st.session_state["brief"] = brief_offline(packet)

brief_tab, qa_tab, strategy_tab = st.tabs(
    ("自动简报", "交互问答", "方案建议")
)
with brief_tab:
    _render_brief(packet)
with qa_tab:
    _render_qa(packet)
with strategy_tab:
    _render_strategies(packet)

st.divider()
st.caption("本页不自动发布、不上传文件，也不恢复或推断任何用户身份。")
