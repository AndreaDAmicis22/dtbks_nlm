"""Smoke test of the Streamlit UI, with every Databricks call faked."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import databricks.sdk
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from nlm import history, library, llm, retrieval, studio
from nlm.passages import Passage

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")
NOTEBOOK = "Ricerca"
PASSAGE = Passage(
    n=1,
    chunk_id="d1-00000",
    doc_id="d1",
    file_name="bilancio.pdf",
    page_start=3,
    page_end=3,
    section="Ricavi",
    content="Il fatturato è cresciuto del 12%.",
)
SOURCES = [
    library.Source(path="/Volumes/c/s/v/Ricerca/bilancio.pdf", file_name="bilancio.pdf", status="ok",
                   doc_id="d1", num_pages=3, num_chunks=5),
    library.Source(path="/Volumes/c/s/v/Ricerca/nota.pdf", file_name="nota.pdf", status="pending"),
]


class FakeWorkspaceClient:
    def __init__(self, **kwargs):
        self.config = SimpleNamespace(
            host="https://example.cloud.databricks.com", authenticate=lambda: {"Authorization": "Bearer x"}
        )
        self.current_user = SimpleNamespace(me=lambda: SimpleNamespace(user_name="mario.rossi@example.com"))


@pytest.fixture
def app(monkeypatch):
    env = {
        "NLM_CATALOG": "c",
        "NLM_SCHEMA": "s",
        "NLM_VOLUME": "v",
        "DATABRICKS_WAREHOUSE_ID": "wh",
        "NLM_LLM_ENDPOINT": "databricks-claude-opus-5-5",
        "NLM_INGEST_JOB_ID": "42",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", FakeWorkspaceClient)
    monkeypatch.setattr(library, "list_notebooks", lambda ws, cfg: [NOTEBOOK])
    monkeypatch.setattr(library, "list_sources", lambda ws, cfg, notebook: SOURCES)
    monkeypatch.setattr(library, "ingest_running", lambda ws, cfg: True)
    monkeypatch.setattr(history, "load", lambda *args, **kwargs: [])
    saved = []
    monkeypatch.setattr(history, "append_turn", lambda *args: saved.append(args))
    monkeypatch.setattr(retrieval, "search", lambda *args, **kwargs: [PASSAGE])
    monkeypatch.setattr(llm, "client_for", lambda ws: nullcontext(object()))
    monkeypatch.setattr(
        llm, "stream", lambda *args, **kwargs: iter(["Il fatturato è cresciuto del 12% [1]", " (circa $5M)."])
    )
    monkeypatch.setattr(
        studio, "load_documents", lambda ws, cfg, ids: ([{"file_name": "bilancio.pdf", "text": "..."}], True)
    )
    st.cache_data.clear()
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=30), saved


def test_sidebar_lists_sources(app):
    at, _ = app
    at.run()
    assert not at.exception
    assert at.sidebar.selectbox[0].value == NOTEBOOK
    assert [c.label for c in at.sidebar.checkbox] == ["bilancio.pdf"]
    captions = " ".join(c.value for c in at.sidebar.caption)
    assert "nota.pdf" in captions and "Indicizzazione in corso" in captions


def test_chat_answers_with_citations_and_saves_history(app):
    at, saved = app
    at.run()
    at.chat_input[0].set_value("Quanto è cresciuto il fatturato?").run()
    assert not at.exception

    answer = " ".join(m.value for m in at.chat_message[1].markdown)
    assert "cresciuto del 12% [1]" in answer
    assert "\\$5M" in answer  # escaped, or Streamlit would render LaTeX
    assert any(e.label == "Fonti citate (1)" for e in at.expander)
    (_, _, notebook, user, question, raw_answer, passages), = saved
    assert (notebook, user, question) == (NOTEBOOK, "mario.rossi@example.com", "Quanto è cresciuto il fatturato?")
    assert raw_answer.endswith("(circa $5M).")  # stored unescaped
    assert passages == [PASSAGE]


def test_studio_generates_and_warns_on_truncation(app):
    at, _ = app
    at.run()
    next(b for b in at.button if b.label == "Genera").click().run()
    assert not at.exception
    assert at.warning, "truncated sources should be reported"
    outputs = at.session_state[f"studio:{NOTEBOOK}"]
    assert outputs[0]["title"] == "Riassunto"
    assert outputs[0]["content"].startswith("Il fatturato")


def test_missing_tables_show_an_error_not_a_traceback(app, monkeypatch):
    at, _ = app

    def fail(*args):
        raise RuntimeError("TABLE_OR_VIEW_NOT_FOUND")

    monkeypatch.setattr(library, "list_sources", fail)
    at.run()
    assert not at.exception
    assert "nlm-setup" in at.sidebar.error[0].value
    assert at.chat_input[0].disabled


def test_no_notebooks_shows_onboarding(app, monkeypatch):
    at, _ = app
    monkeypatch.setattr(library, "list_notebooks", lambda ws, cfg: [])
    at.run()
    assert not at.exception
    assert "primo notebook" in at.sidebar.info[0].value
