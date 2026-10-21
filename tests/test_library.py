from types import SimpleNamespace

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.files import DirectoryEntry

from nlm import library
from nlm.config import DEFAULT_NOTEBOOK, Settings

SETTINGS = Settings(
    catalog="c", schema="s", volume="v", warehouse_id="wh", llm_endpoint="ep", ingest_job_id=1
)
ROOT = SETTINGS.volume_root


def _entry(path, is_directory=False):
    return DirectoryEntry(path=path, name=path.rsplit("/", 1)[-1], is_directory=is_directory)


class FakeFiles:
    def __init__(self, tree):
        self.tree = tree

    def list_directory_contents(self, path):
        if path not in self.tree:
            raise NotFound("missing")
        return iter(self.tree[path])


def test_clean_name():
    assert library.clean_name("  ../report [v2]{x}.pdf ") == "report v2x.pdf"
    assert library.clean_name("C:\\tmp\\_bozza.docx") == "bozza.docx"
    assert library.clean_name("Relazione  2025\nfinale") == "Relazione 2025 finale"
    with pytest.raises(ValueError):
        library.clean_name("._")


def test_list_notebooks_skips_hidden_and_maps_root_files():
    files = FakeFiles(
        {
            ROOT: [
                _entry(f"{ROOT}/Bilanci", True),
                _entry(f"{ROOT}/_images", True),
                _entry(f"{ROOT}/.DS_Store"),
                _entry(f"{ROOT}/appunti.txt"),
                _entry(f"{ROOT}/contratti", True),
            ]
        }
    )
    ws = SimpleNamespace(files=files)
    assert library.list_notebooks(ws, SETTINGS) == ["Bilanci", "contratti", DEFAULT_NOTEBOOK]


def test_list_sources_merges_table_and_pending_files(monkeypatch):
    rows = [
        {"doc_id": "d1", "path": f"{ROOT}/Bilanci/b.pdf", "file_name": "b.pdf", "status": "ok",
         "error": None, "num_pages": "12", "num_chunks": "30"},
    ]
    monkeypatch.setattr(library.sql, "query", lambda *args, **kwargs: rows)
    files = FakeFiles({f"{ROOT}/Bilanci": [_entry(f"{ROOT}/Bilanci/b.pdf"), _entry(f"{ROOT}/Bilanci/a.pdf")]})

    sources = library.list_sources(SimpleNamespace(files=files), SETTINGS, "Bilanci")

    assert [(s.file_name, s.status, s.num_pages) for s in sources] == [("a.pdf", "pending", None), ("b.pdf", "ok", 12)]


def test_list_sources_of_missing_folder(monkeypatch):
    monkeypatch.setattr(library.sql, "query", lambda *args, **kwargs: [])
    assert library.list_sources(SimpleNamespace(files=FakeFiles({})), SETTINGS, "Nuovo") == []
