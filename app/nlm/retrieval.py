"""Hybrid (vector + keyword) search on the chunks index, scoped to a notebook."""

from __future__ import annotations

import json

from databricks.sdk import WorkspaceClient

from .passages import Passage
from .sql import to_int

COLUMNS = ["chunk_id", "doc_id", "file_name", "page_start", "page_end", "section", "content"]


def search(
    ws: WorkspaceClient,
    index_name: str,
    query: str,
    notebook: str,
    doc_ids: list[str] | None = None,
    k: int = 8,
) -> list[Passage]:
    """Top-k passages for `query`. `doc_ids=None` searches the whole notebook."""
    filters: dict = {"notebook": notebook}
    if doc_ids is not None:
        filters["doc_id"] = list(doc_ids)
    response = ws.vector_search_indexes.query_index(
        index_name=index_name,
        columns=COLUMNS,
        query_text=query,
        query_type="HYBRID",
        filters_json=json.dumps(filters),
        num_results=k,
    )
    if not response.result or not response.result.data_array:
        return []
    names = [column.name for column in response.manifest.columns]
    passages = []
    for n, values in enumerate(response.result.data_array, start=1):
        row = dict(zip(names, values))
        passages.append(
            Passage(
                n=n,
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                file_name=row["file_name"],
                page_start=to_int(row.get("page_start")),
                page_end=to_int(row.get("page_end")),
                section=row.get("section"),
                content=row["content"],
                score=float(row["score"]) if row.get("score") is not None else None,
            )
        )
    return passages
