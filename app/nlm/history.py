"""Chat history per user and notebook, stored in a Delta table.

Always written with the app identity, so users need no write grants.
"""

from __future__ import annotations

import json
import time
import uuid

from databricks.sdk import WorkspaceClient

from . import sql
from .config import Settings
from .passages import Passage


def load(ws: WorkspaceClient, settings: Settings, notebook: str, user: str, limit: int = 200) -> list[dict]:
    rows = sql.query(
        ws,
        settings.warehouse_id,
        f"SELECT role, content, passages FROM {settings.table('chat_messages')} "
        "WHERE notebook = :notebook AND user_email = :user "
        f"ORDER BY created_at DESC LIMIT {int(limit)}",
        notebook=notebook,
        user=user,
    )
    messages = []
    for row in reversed(rows):
        passages = [Passage.from_dict(p) for p in json.loads(row["passages"] or "[]")]
        messages.append({"role": row["role"], "content": row["content"], "passages": passages})
    return messages


def append_turn(
    ws: WorkspaceClient,
    settings: Settings,
    notebook: str,
    user: str,
    question: str,
    answer: str,
    passages: list[Passage],
) -> None:
    now_ms = int(time.time() * 1000)
    sql.query(
        ws,
        settings.warehouse_id,
        f"INSERT INTO {settings.table('chat_messages')} "
        "(message_id, notebook, user_email, role, content, passages, created_at) VALUES "
        "(:question_id, :notebook, :user, 'user', :question, NULL, timestamp_millis(:asked_at)), "
        "(:answer_id, :notebook, :user, 'assistant', :answer, :passages, timestamp_millis(:answered_at))",
        question_id=str(uuid.uuid4()),
        answer_id=str(uuid.uuid4()),
        notebook=notebook,
        user=user,
        question=question,
        answer=answer,
        passages=json.dumps([p.to_dict() for p in passages], ensure_ascii=False),
        asked_at=now_ms,
        answered_at=now_ms + 1,
    )


def clear(ws: WorkspaceClient, settings: Settings, notebook: str, user: str) -> None:
    sql.query(
        ws,
        settings.warehouse_id,
        f"DELETE FROM {settings.table('chat_messages')} WHERE notebook = :notebook AND user_email = :user",
        notebook=notebook,
        user=user,
    )
