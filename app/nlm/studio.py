"""Studio: generations over whole documents (summary, FAQ, study guide...)."""

from __future__ import annotations

import json

from databricks.sdk import WorkspaceClient

from . import prompts, sql
from .config import Settings


def load_documents(ws: WorkspaceClient, settings: Settings, doc_ids: list[str]) -> tuple[list[dict], bool]:
    """Text of the selected documents within the character budget, and whether any was cut."""
    per_doc = max(settings.studio_max_chars // max(len(doc_ids), 1), 1000)
    rows = sql.query(
        ws,
        settings.warehouse_id,
        f"SELECT file_name, left(text, :max_chars) AS text, length(text) AS full_length "
        f"FROM {settings.table('documents')} "
        "WHERE array_contains(from_json(:doc_ids, 'array<string>'), doc_id) "
        "ORDER BY file_name",
        max_chars=per_doc,
        doc_ids=json.dumps(doc_ids),
    )
    truncated = any((sql.to_int(row["full_length"]) or 0) > per_doc for row in rows)
    return [{"file_name": row["file_name"], "text": row["text"] or ""} for row in rows], truncated


def messages(documents: list[dict], instruction: str) -> list[dict]:
    return [
        {"role": "system", "content": prompts.STUDIO_SYSTEM},
        {"role": "user", "content": prompts.studio_user_message(documents, instruction)},
    ]
