import json
from types import SimpleNamespace

from databricks.sdk.service.vectorsearch import QueryVectorIndexResponse

from nlm import retrieval


class FakeIndexes:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def query_index(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _ws(response):
    return SimpleNamespace(vector_search_indexes=FakeIndexes(response))


def test_search_maps_rows_and_builds_filters():
    response = QueryVectorIndexResponse.from_dict(
        {
            "manifest": {"column_count": 8, "columns": [{"name": c} for c in [*retrieval.COLUMNS, "score"]]},
            "result": {
                "row_count": 2,
                "data_array": [
                    ["d1-00000", "d1", "a.pdf", 3.0, 4.0, "Intro", "testo uno", 0.031],
                    ["d2-00001", "d2", "b.md", None, None, None, "testo due", 0.016],
                ],
            },
        }
    )
    ws = _ws(response)

    passages = retrieval.search(ws, "c.s.chunks_index", "domanda", "nb", ["d1", "d2"], k=5)

    assert [(p.n, p.chunk_id, p.page_start, p.page_end, p.section) for p in passages] == [
        (1, "d1-00000", 3, 4, "Intro"),
        (2, "d2-00001", None, None, None),
    ]
    assert passages[0].score == 0.031
    call = ws.vector_search_indexes.calls[0]
    assert json.loads(call["filters_json"]) == {"notebook": "nb", "doc_id": ["d1", "d2"]}
    assert (call["query_type"], call["num_results"], call["columns"]) == ("HYBRID", 5, retrieval.COLUMNS)


def test_search_whole_notebook_and_empty_result():
    ws = _ws(QueryVectorIndexResponse.from_dict({"manifest": {"columns": []}, "result": {"row_count": 0}}))
    assert retrieval.search(ws, "c.s.chunks_index", "domanda", "nb") == []
    assert json.loads(ws.vector_search_indexes.calls[0]["filters_json"]) == {"notebook": "nb"}
