from types import SimpleNamespace

import pytest
from databricks.sdk.service.sql import ResultData, StatementResponse

from nlm import sql


class FakeStatements:
    def __init__(self, responses, chunks=None):
        self.responses = list(responses)
        self.chunks = chunks or {}
        self.executed = []

    def execute_statement(self, **kwargs):
        self.executed.append(kwargs)
        return self.responses.pop(0)

    def get_statement(self, statement_id):
        return self.responses.pop(0)

    def get_statement_result_chunk_n(self, statement_id, chunk_index):
        return self.chunks[chunk_index]


def _response(state, rows=None, next_chunk=None, error=None):
    data = {"statement_id": "s1", "status": {"state": state}}
    if error:
        data["status"]["error"] = {"message": error}
    if rows is not None:
        data["manifest"] = {"schema": {"columns": [{"name": "a"}, {"name": "b"}]}}
        data["result"] = {"data_array": rows, "next_chunk_index": next_chunk}
    return StatementResponse.from_dict(data)


def test_query_polls_paginates_and_types_parameters(monkeypatch):
    monkeypatch.setattr(sql.time, "sleep", lambda _: None)
    statements = FakeStatements(
        [_response("PENDING"), _response("SUCCEEDED", [["1", "x"]], next_chunk=1)],
        chunks={1: ResultData.from_dict({"data_array": [["2", None]]})},
    )
    ws = SimpleNamespace(statement_execution=statements)

    rows = sql.query(ws, "wh", "SELECT ...", small=5, big=2**40, name="nb", flag=True, missing=None)

    assert rows == [{"a": "1", "b": "x"}, {"a": "2", "b": None}]
    params = {p.name: (p.type, p.value) for p in statements.executed[0]["parameters"]}
    assert params == {
        "small": ("INT", "5"),
        "big": ("BIGINT", str(2**40)),
        "name": ("STRING", "nb"),
        "flag": ("BOOLEAN", "true"),
        "missing": ("STRING", None),
    }


def test_query_without_result_returns_empty_list():
    statements = FakeStatements([_response("SUCCEEDED")])
    assert sql.query(SimpleNamespace(statement_execution=statements), "wh", "DELETE ...") == []
    assert statements.executed[0]["parameters"] is None


def test_query_raises_with_server_message():
    statements = FakeStatements([_response("FAILED", error="TABLE_OR_VIEW_NOT_FOUND")])
    with pytest.raises(RuntimeError, match="TABLE_OR_VIEW_NOT_FOUND"):
        sql.query(SimpleNamespace(statement_execution=statements), "wh", "SELECT 1")


def test_to_int():
    assert sql.to_int(None) is None
    assert sql.to_int("") is None
    assert sql.to_int("3") == 3
    assert sql.to_int(4.0) == 4
