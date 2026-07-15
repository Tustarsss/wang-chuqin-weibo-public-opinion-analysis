# Lab 3 Three-in-One Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a locally runnable Streamlit assistant that turns the existing eight-event Lab 2 analysis into evidence-grounded briefs, question answering, and communication strategy options with automatic offline fallback.

**Architecture:** A pure-Python core loads and validates the fixed Lab 1/2 artifacts, builds immutable evidence packets for a selected scope, and feeds three services. Deterministic offline renderers always work; an isolated DeepSeek client may improve wording but cannot calculate facts. A thin Streamlit layer shares scope across three tabs and exports the reviewed result.

**Tech Stack:** Python 3.11, dataclasses, standard-library JSON, OpenAI Python SDK, Streamlit, pytest.

**Design:** `docs/superpowers/specs/2026-07-15-lab3-three-in-one-assistant-design.md`

---

## File map

- `王楚钦舆情分析_Lab3/requirements.txt`: runtime dependencies.
- `王楚钦舆情分析_Lab3/requirements-dev.txt`: test dependency.
- `王楚钦舆情分析_Lab3/app.py`: Streamlit state and rendering only.
- `王楚钦舆情分析_Lab3/lab3/models.py`: immutable domain and output types.
- `王楚钦舆情分析_Lab3/lab3/data_loader.py`: fixed artifact paths, parsing, validation and indexes.
- `王楚钦舆情分析_Lab3/lab3/evidence.py`: scope filtering, metrics, citations and warnings.
- `王楚钦舆情分析_Lab3/lab3/offline.py`: deterministic brief, preset answers and three strategies.
- `王楚钦舆情分析_Lab3/lab3/llm_client.py`: DeepSeek JSON calls and safe failure results.
- `王楚钦舆情分析_Lab3/lab3/services.py`: online-first, offline-fallback orchestration.
- `王楚钦舆情分析_Lab3/lab3/export.py`: Markdown decision-record export.
- `王楚钦舆情分析_Lab3/tests/`: pure-Python unit and integration tests.
- `王楚钦舆情分析_Lab3/README.md`: setup, run, demo and limitations.
- `README.md`: top-level Lab 1–3 project navigation.

### Task 1: Scaffold the package and domain types

**Files:**
- Create: `王楚钦舆情分析_Lab3/requirements.txt`
- Create: `王楚钦舆情分析_Lab3/requirements-dev.txt`
- Create: `王楚钦舆情分析_Lab3/lab3/__init__.py`
- Create: `王楚钦舆情分析_Lab3/lab3/models.py`
- Create: `王楚钦舆情分析_Lab3/tests/test_models.py`

- [ ] **Step 1: Write failing model tests**

```python
from dataclasses import FrozenInstanceError
import pytest

from lab3.models import AnalysisScope, Citation, EvidencePacket, MetricSummary


def test_single_event_scope_requires_event_id():
    with pytest.raises(ValueError, match="event_id"):
        AnalysisScope(kind="single_event", source="both", audience="球迷")


def test_evidence_packet_is_immutable():
    metric = MetricSummary(1, 0.4, {"positive": 100.0, "neutral": 0.0, "negative": 0.0}, (), ())
    citation = Citation("p1", "post", "e1", "赛事", "原文", "positive", "支持鼓励", ("赛果",), 0.9, 1)
    packet = EvidencePacket("赛事", AnalysisScope("single_event", "both", "球迷", "e1"), metric, None, (citation,), (), ("事实",))
    with pytest.raises(FrozenInstanceError):
        packet.label = "changed"
```

- [ ] **Step 2: Run tests and verify the import fails**

Run: `python -m pytest 王楚钦舆情分析_Lab3/tests/test_models.py -v`

Expected: FAIL because `lab3.models` does not exist.

- [ ] **Step 3: Add dependencies and implement the domain types**

`requirements.txt`:

```text
openai>=1.0.0
streamlit>=1.36
```

`requirements-dev.txt`:

```text
-r requirements.txt
pytest>=8.0
```

`models.py` must define these frozen dataclasses and validation:

```python
from dataclasses import dataclass, field
from typing import Any, Literal

ScopeKind = Literal["single_event", "win_group", "loss_group", "win_loss_comparison"]
SourceKind = Literal["both", "post", "comment"]


@dataclass(frozen=True)
class AnalysisScope:
    kind: ScopeKind
    source: SourceKind
    audience: str
    event_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "single_event" and not self.event_id:
            raise ValueError("single_event scope requires event_id")
        if self.kind != "single_event" and self.event_id is not None:
            raise ValueError("event_id is only valid for single_event scope")


@dataclass(frozen=True)
class MetricSummary:
    n: int
    mean_score: float
    polarity_pct: dict[str, float]
    top_emotions: tuple[tuple[str, float], ...]
    top_topics: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Citation:
    record_id: str
    content_type: str
    event_id: str
    event_name: str
    text: str
    polarity: str
    emotion: str
    topics: tuple[str, ...]
    confidence: float
    likes: int


@dataclass(frozen=True)
class EvidencePacket:
    label: str
    scope: AnalysisScope
    posts: MetricSummary | None
    comments: MetricSummary | None
    citations: tuple[Citation, ...]
    warnings: tuple[str, ...]
    facts: tuple[str, ...]

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "scope": {"kind": self.scope.kind, "source": self.scope.source, "audience": self.scope.audience, "event_id": self.scope.event_id},
            "posts": vars(self.posts) if self.posts else None,
            "comments": vars(self.comments) if self.comments else None,
            "citations": [vars(item) for item in self.citations],
            "warnings": list(self.warnings),
            "facts": list(self.facts),
        }


@dataclass(frozen=True)
class GeneratedResult:
    payload: dict[str, Any]
    mode: Literal["online", "offline"]
    warning: str | None = None
```

- [ ] **Step 4: Run the model tests**

Run: `python -m pytest 王楚钦舆情分析_Lab3/tests/test_models.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add 王楚钦舆情分析_Lab3
git commit -m "feat(lab3): define assistant domain models"
```

### Task 2: Load and validate the fixed Lab 1/2 artifacts

**Files:**
- Create: `王楚钦舆情分析_Lab3/lab3/data_loader.py`
- Create: `王楚钦舆情分析_Lab3/tests/conftest.py`
- Create: `王楚钦舆情分析_Lab3/tests/test_data_loader.py`

- [ ] **Step 1: Write failing loader tests against the real course artifacts**

```python
from pathlib import Path
import pytest

from lab3.data_loader import DataContractError, load_project_data


def test_loads_lab2_contract(repo_root: Path):
    data = load_project_data(repo_root)
    assert len(data.events) == 8
    assert len(data.posts) == 45
    assert len(data.comments) == 61
    assert data.ingestion["lab3_ready"] is True


def test_missing_artifacts_raise_named_error(tmp_path: Path):
    with pytest.raises(DataContractError, match="events.json"):
        load_project_data(tmp_path)
```

`conftest.py` adds the Lab 3 directory to `sys.path` and defines the shared fixtures used by later tasks:

```python
from pathlib import Path
import sys
import pytest

LAB3_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB3_DIR.parent
sys.path.insert(0, str(LAB3_DIR))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def project_data(repo_root):
    from lab3.data_loader import load_project_data
    return load_project_data(repo_root)


@pytest.fixture
def loss_packet(project_data):
    from lab3.evidence import build_evidence
    from lab3.models import AnalysisScope
    return build_evidence(project_data, AnalysisScope("loss_group", "both", "媒体"))


@pytest.fixture
def offline_brief(loss_packet):
    from lab3.offline import brief_offline
    return brief_offline(loss_packet)


@pytest.fixture
def offline_strategies(loss_packet):
    from lab3.offline import strategies_offline
    return strategies_offline(loss_packet, "回应争议", "媒体")
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest 王楚钦舆情分析_Lab3/tests/test_data_loader.py -v`

Expected: FAIL because `load_project_data` is undefined.

- [ ] **Step 3: Implement strict loading and indexes**

```python
@dataclass(frozen=True)
class ProjectData:
    events: dict[str, dict[str, Any]]
    report: dict[str, Any]
    posts: tuple[dict[str, Any], ...]
    comments: tuple[dict[str, Any], ...]
    ingestion: dict[str, Any]


def load_project_data(repo_root: Path) -> ProjectData:
    paths = artifact_paths(repo_root)
    for label, path in paths.items():
        if not path.is_file():
            raise DataContractError(f"missing {label}: {path}")
    active = {item["event_id"]: item for item in read_json(paths["events.json"]) if item.get("active", True)}
    posts = tuple(read_jsonl(paths["posts_sentiment.jsonl"]))
    comments = tuple(read_jsonl(paths["comments_sentiment.jsonl"]))
    ingestion = read_json(paths["lab3_ingestion_check.json"])
    if ingestion.get("lab3_ready") is not True or ingestion.get("errors"):
        raise DataContractError("Lab 3 ingestion contract is not ready")
    validate_rows(posts, "post_id", "post", active)
    validate_rows(comments, "comment_id", "comment", active)
    return ProjectData(active, read_json(paths["sentiment_report.json"]), posts, comments, ingestion)
```

`validate_rows` must reject duplicate IDs, unknown events, mismatched `content_type`/`match_result`, out-of-window hours, invalid polarity, missing text and invalid confidence. Error messages include the record ID and field.

- [ ] **Step 4: Run loader tests and the upstream contract check**

Run: `python -m pytest 王楚钦舆情分析_Lab3/tests/test_data_loader.py -v`

Expected: 2 passed.

Run: `python 王楚钦舆情分析_Lab2/04_复现脚本/validate_lab3.py`

Expected: `errors: 0`, `warnings: 2`, `status: pass_with_warnings`.

- [ ] **Step 5: Commit**

```bash
git add 王楚钦舆情分析_Lab3/lab3/data_loader.py 王楚钦舆情分析_Lab3/tests
git commit -m "feat(lab3): validate Lab 1 and Lab 2 inputs"
```

### Task 3: Build deterministic evidence packets

**Files:**
- Create: `王楚钦舆情分析_Lab3/lab3/evidence.py`
- Create: `王楚钦舆情分析_Lab3/tests/test_evidence.py`

- [ ] **Step 1: Write failing scope and warning tests**

```python
from lab3.evidence import build_evidence
from lab3.models import AnalysisScope


def test_loss_group_keeps_posts_and_comments_separate(project_data):
    packet = build_evidence(project_data, AnalysisScope("loss_group", "both", "球迷"))
    assert packet.posts and packet.posts.n == 22
    assert packet.comments and packet.comments.n == 31
    assert "正文" in packet.facts[0]
    assert all(c.content_type in {"post", "comment"} for c in packet.citations)


def test_sparse_single_event_has_coverage_warning(project_data):
    scope = AnalysisScope("single_event", "comment", "媒体", "loss_20250208_singapore_liang")
    packet = build_evidence(project_data, scope)
    assert packet.comments and packet.comments.n == 1
    assert any("少于 3 条" in warning for warning in packet.warnings)


def test_zero_comment_event_does_not_invent_metrics(project_data):
    scope = AnalysisScope("single_event", "comment", "媒体", "win_20251005_china_smash_lebrun")
    packet = build_evidence(project_data, scope)
    assert packet.comments and packet.comments.n == 0
    assert any("零记录" in warning for warning in packet.warnings)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest 王楚钦舆情分析_Lab3/tests/test_evidence.py -v`

Expected: FAIL because `build_evidence` is undefined.

- [ ] **Step 3: Implement evidence selection and summaries**

```python
def build_evidence(data: ProjectData, scope: AnalysisScope) -> EvidencePacket:
    event_ids = scope_event_ids(data, scope)
    posts = tuple(row for row in data.posts if row["event_id"] in event_ids)
    comments = tuple(row for row in data.comments if row["event_id"] in event_ids)
    post_metric = None if scope.source == "comment" else summarize(data, posts, "posts", scope)
    comment_metric = None if scope.source == "post" else summarize(data, comments, "comments", scope)
    warnings = coverage_warnings(data, event_ids, posts, comments, scope.source)
    citations = select_citations(posts if scope.source != "comment" else (), comments if scope.source != "post" else ())
    facts = fact_lines(post_metric, comment_metric)
    return EvidencePacket(scope_label(data, scope), scope, post_metric, comment_metric, citations, tuple(warnings), tuple(facts))
```

For a group, `summarize` uses `report[side]["by_result_rollup"][result]`; for a single event it uses `report[side]["by_event"][event_id]`. It maps counts, mean score, three polarity percentages, top three nonzero emotions and top three topics. `select_citations` deterministically prefers confidence `>=0.6`, then likes, intensity and record ID, and includes positive, negative, high-engagement and top-topic evidence without duplicate IDs.

- [ ] **Step 4: Run evidence tests**

Run: `python -m pytest 王楚钦舆情分析_Lab3/tests/test_evidence.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add 王楚钦舆情分析_Lab3/lab3/evidence.py 王楚钦舆情分析_Lab3/tests/test_evidence.py
git commit -m "feat(lab3): build traceable evidence packets"
```

### Task 4: Implement complete offline behavior and Markdown export

**Files:**
- Create: `王楚钦舆情分析_Lab3/lab3/offline.py`
- Create: `王楚钦舆情分析_Lab3/lab3/export.py`
- Create: `王楚钦舆情分析_Lab3/tests/test_offline.py`
- Create: `王楚钦舆情分析_Lab3/tests/test_export.py`

- [ ] **Step 1: Write failing offline-result tests**

```python
from lab3.offline import answer_offline, brief_offline, strategies_offline


def test_offline_brief_contains_required_sections(loss_packet):
    result = brief_offline(loss_packet)
    assert result.mode == "offline"
    assert set(result.payload) >= {"facts", "observations", "decision_focus", "limitations", "citation_ids"}


def test_preset_question_rejects_loss_equals_negative(loss_packet):
    result = answer_offline("loss_all_negative", loss_packet)
    assert result.payload["answerable"] is True
    assert "不等于" in result.payload["interpretation"]
    assert result.payload["citation_ids"]


def test_strategy_always_returns_three_human_reviewed_options(loss_packet):
    result = strategies_offline(loss_packet, "回应争议", "媒体")
    assert len(result.payload["options"]) == 3
    assert all({"action", "timing", "evidence_ids", "benefits", "risks", "checks"} <= set(item) for item in result.payload["options"])
    assert "人工" in result.payload["disclaimer"]
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest 王楚钦舆情分析_Lab3/tests/test_offline.py -v`

Expected: FAIL because offline functions do not exist.

- [ ] **Step 3: Implement deterministic renderers**

```python
PRESET_QUESTIONS = {
    "loss_all_negative": "输球后的评论是否都是负面的？",
    "source_difference": "微博正文和一级评论有什么差异？",
    "top_topics": "当前范围最受关注的议题是什么？",
    "representative_views": "有哪些代表性观点？",
    "coverage_limits": "哪些结论受到样本覆盖限制？",
}


def brief_offline(packet: EvidencePacket) -> GeneratedResult:
    payload = {
        "facts": list(packet.facts),
        "observations": observation_lines(packet),
        "decision_focus": decision_focus_lines(packet),
        "limitations": list(packet.warnings) or ["这是 8 场比赛的案例样本，不能代表微博总体舆情。"],
        "citation_ids": [item.record_id for item in packet.citations[:6]],
    }
    return GeneratedResult(payload, "offline", "使用确定性离线模板")


def answer_offline(question_key: str, packet: EvidencePacket) -> GeneratedResult:
    if question_key not in PRESET_QUESTIONS:
        return GeneratedResult({"answerable": False, "facts": [], "interpretation": "离线模式无法可靠回答该自由问题。", "limitations": list(packet.warnings), "citation_ids": []}, "offline", "请使用预设问题或恢复在线模式")
    return GeneratedResult(build_preset_answer(question_key, packet), "offline", "使用确定性离线问答")
```

`strategies_offline` returns three named options: timely emotional acknowledgement, factual clarification/review, and continued monitoring. Wording varies by goal/audience, but all evidence IDs come from the packet and every result states that it is qualitative scenario comparison, not prediction.

- [ ] **Step 4: Write the failing export test and implement export**

```python
def test_markdown_export_contains_scope_evidence_and_human_note(loss_packet, offline_brief, offline_strategies):
    markdown = export_markdown(loss_packet, offline_brief, offline_strategies, "选择继续监测", "等待更多评论")
    assert loss_packet.label in markdown
    assert "## 证据" in markdown
    assert loss_packet.citations[0].record_id in markdown
    assert "等待更多评论" in markdown
    assert "不能代表微博总体舆情" in markdown
```

`export_markdown` uses only passed structured values and escapes no content into executable HTML.

- [ ] **Step 5: Run offline and export tests**

Run: `python -m pytest 王楚钦舆情分析_Lab3/tests/test_offline.py 王楚钦舆情分析_Lab3/tests/test_export.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add 王楚钦舆情分析_Lab3/lab3/offline.py 王楚钦舆情分析_Lab3/lab3/export.py 王楚钦舆情分析_Lab3/tests
git commit -m "feat(lab3): add complete offline decision support"
```

### Task 5: Add the constrained DeepSeek client and online-first services

**Files:**
- Create: `王楚钦舆情分析_Lab3/lab3/llm_client.py`
- Create: `王楚钦舆情分析_Lab3/lab3/services.py`
- Create: `王楚钦舆情分析_Lab3/tests/test_llm_client.py`
- Create: `王楚钦舆情分析_Lab3/tests/test_services.py`

- [ ] **Step 1: Write failing client failure-mode tests using a fake SDK client**

```python
def test_missing_key_returns_safe_failure(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = DeepSeekClient(repo_root=None).generate("brief", {"facts": ["x"]}, lambda value: "facts" in value)
    assert result.ok is False
    assert "API key" in result.reason


def test_invalid_json_retries_once_then_fails(fake_client):
    fake_client.responses = ["not-json", "still-not-json"]
    result = DeepSeekClient(api_key="sk-test", sdk_client=fake_client).generate("brief", {}, lambda value: True)
    assert result.ok is False
    assert fake_client.call_count == 2
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest 王楚钦舆情分析_Lab3/tests/test_llm_client.py -v`

Expected: FAIL because `DeepSeekClient` does not exist.

The test file defines a minimal `FakeSDKClient` whose `chat.completions.create` method pops strings from `responses`, returns them as `choices[0].message.content`, and increments `call_count`; its pytest fixture returns a new instance. Service tests define `FailingLLM.generate` returning `LLMCallResult(False, None, "模拟超时")` and `SuccessfulLLM.generate` returning schema-valid payloads whose citation IDs are provided by the test.

- [ ] **Step 3: Implement the client boundary**

```python
@dataclass(frozen=True)
class LLMCallResult:
    ok: bool
    payload: dict[str, Any] | None
    reason: str | None


class DeepSeekClient:
    def __init__(self, *, repo_root: Path | None, api_key: str | None = None, sdk_client: Any = None, timeout: float = 20.0):
        self.api_key = api_key or resolve_key(repo_root)
        self.client = sdk_client or create_sdk_client(self.api_key, timeout)

    def generate(self, task: str, evidence: dict[str, Any], validator: Callable[[dict[str, Any]], bool]) -> LLMCallResult:
        if not self.api_key:
            return LLMCallResult(False, None, "未找到 API key")
        for _ in range(2):
            try:
                raw = request_json(self.client, task, evidence)
                parsed = json.loads(raw)
                if validator(parsed):
                    return LLMCallResult(True, parsed, None)
            except Exception as exc:
                reason = safe_reason(exc)
        return LLMCallResult(False, None, reason if "reason" in locals() else "模型返回无效 JSON")
```

`resolve_key` checks `DEEPSEEK_API_KEY` and then the ignored root `api.txt` using Lab 2's `deepseek-api: sk-...` pattern. `request_json` uses base URL/model from Lab 2 taxonomy with environment overrides, JSON mode, disabled thinking and no unrelated raw data. `safe_reason` never returns credentials or complete request bodies.

- [ ] **Step 4: Write service fallback tests and implement orchestration**

```python
def test_brief_service_falls_back_when_client_fails(loss_packet, failing_llm):
    result = BriefService(failing_llm).generate(loss_packet)
    assert result.mode == "offline"
    assert "降级" in (result.warning or "")


def test_answer_service_keeps_online_citation_ids_within_packet(loss_packet, successful_llm):
    result = QAService(successful_llm).answer("为什么？", loss_packet)
    allowed = {item.record_id for item in loss_packet.citations}
    assert set(result.payload["citation_ids"]) <= allowed
```

`BriefService`, `QAService`, and `StrategyService` each build a task-specific JSON schema, call the client, reject unknown evidence IDs or missing required fields, and call the corresponding offline function on any failure.

- [ ] **Step 5: Run online-boundary tests**

Run: `python -m pytest 王楚钦舆情分析_Lab3/tests/test_llm_client.py 王楚钦舆情分析_Lab3/tests/test_services.py -v`

Expected: all tests pass without network access.

- [ ] **Step 6: Commit**

```bash
git add 王楚钦舆情分析_Lab3/lab3/llm_client.py 王楚钦舆情分析_Lab3/lab3/services.py 王楚钦舆情分析_Lab3/tests
git commit -m "feat(lab3): add constrained DeepSeek generation"
```

### Task 6: Build the Streamlit three-tab application

**Files:**
- Create: `王楚钦舆情分析_Lab3/app.py`
- Create: `王楚钦舆情分析_Lab3/tests/test_app_helpers.py`

- [ ] **Step 1: Extract and test UI-independent state helpers first**

```python
def test_context_key_changes_with_scope():
    first = context_key(AnalysisScope("loss_group", "both", "球迷"))
    second = context_key(AnalysisScope("win_group", "both", "球迷"))
    assert first != second


def test_source_labels_are_not_merged(loss_packet):
    rows = metric_rows(loss_packet)
    assert [row["来源"] for row in rows] == ["微博正文", "一级评论"]
```

- [ ] **Step 2: Run helper tests and verify failure**

Run: `python -m pytest 王楚钦舆情分析_Lab3/tests/test_app_helpers.py -v`

Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Implement a thin Streamlit page**

`app.py` must:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent
data = load_project_data(REPO_ROOT)
scope = render_sidebar(data)
packet = build_evidence(data, scope)
if st.session_state.get("context_key") != context_key(scope):
    st.session_state["context_key"] = context_key(scope)
    st.session_state["messages"] = []
    st.session_state.pop("brief", None)
    st.session_state.pop("strategies", None)

brief_tab, qa_tab, strategy_tab = st.tabs(["自动简报", "交互问答", "方案建议"])
with brief_tab:
    render_brief(packet, services)
with qa_tab:
    render_qa(packet, services)
with strategy_tab:
    render_strategies(packet, services)
```

The sidebar renders scope, event, source and audience controls plus data/model status. Brief renders separate post/comment metric cards, warnings, top topics/emotions, generated sections, evidence expanders and download. QA renders preset buttons, chat input and answer sections with citation expanders. Strategy renders goal selection, three option cards, qualitative non-prediction warning, human choice/note fields and combined Markdown download. No API call occurs until the user presses a generate/ask button.

- [ ] **Step 4: Run tests and syntax/import smoke checks**

Run: `python -m pytest 王楚钦舆情分析_Lab3/tests/test_app_helpers.py -v`

Expected: all tests pass.

Run: `python -m compileall -q 王楚钦舆情分析_Lab3`

Expected: exit code 0.

- [ ] **Step 5: Commit**

```bash
git add 王楚钦舆情分析_Lab3/app.py 王楚钦舆情分析_Lab3/tests/test_app_helpers.py
git commit -m "feat(lab3): add Streamlit three-in-one interface"
```

### Task 7: Document, integrate and verify the complete course pipeline

**Files:**
- Create: `王楚钦舆情分析_Lab3/README.md`
- Modify: `README.md`
- Create: `王楚钦舆情分析_Lab3/tests/test_integration.py`

- [ ] **Step 1: Add a no-network integration test**

```python
def test_complete_offline_pipeline(repo_root, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    data = load_project_data(repo_root)
    packet = build_evidence(data, AnalysisScope("loss_group", "both", "媒体"))
    brief = BriefService(DeepSeekClient(repo_root=None)).generate(packet)
    answer = QAService(DeepSeekClient(repo_root=None)).answer("loss_all_negative", packet, preset=True)
    strategies = StrategyService(DeepSeekClient(repo_root=None)).generate(packet, "回应争议", "媒体")
    markdown = export_markdown(packet, brief, strategies, "持续监测", "课堂演示")
    assert brief.mode == answer.mode == strategies.mode == "offline"
    assert len(strategies.payload["options"]) == 3
    assert "课堂演示" in markdown
```

- [ ] **Step 2: Run the integration test and fix only contract mismatches**

Run: `python -m pytest 王楚钦舆情分析_Lab3/tests/test_integration.py -v`

Expected: 1 passed.

- [ ] **Step 3: Write operator documentation**

`王楚钦舆情分析_Lab3/README.md` includes:

- the decision-support boundary and the three capabilities;
- exact Python 3.11 environment and install commands;
- `DEEPSEEK_API_KEY` and ignored `api.txt` configuration;
- exact run command `streamlit run 王楚钦舆情分析_Lab3/app.py`;
- a two-minute Paris-loss demo script;
- offline-mode behavior;
- input/output file map;
- known zero/thin-comment warnings;
- test commands and privacy limitations.

Update root `README.md` to describe the full Lab 1 → Lab 2 → Lab 3 pipeline and link each lab README.

- [ ] **Step 4: Run complete verification**

Run: `python -m pytest 王楚钦舆情分析_Lab3/tests -q`

Expected: all tests pass.

Run: `python 王楚钦舆情分析_Lab2/04_复现脚本/validate_lab3.py`

Expected: `errors: 0`, `warnings: 2`, `status: pass_with_warnings`.

Run: `python -m compileall -q 王楚钦舆情分析_Lab3`

Expected: exit code 0.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add README.md 王楚钦舆情分析_Lab3
git commit -m "docs(lab3): add runbook and pipeline handoff"
```
