from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_starts_offline_with_complete_three_tab_interface(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "自动简报",
        "交互问答",
        "方案建议",
    ]
    assert any("王楚钦" in title.value for title in app.title)
    assert app.warning
    assert app.get("download_button")
