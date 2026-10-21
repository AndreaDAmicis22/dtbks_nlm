"""Minimal SQL access through the Statement Execution API (no extra driver needed)."""

from __future__ import annotations

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementParameterListItem, StatementState

_RUNNING = (StatementState.PENDING, StatementState.RUNNING)


def query(ws: WorkspaceClient, warehouse_id: str, statement: str, **params) -> list[dict]:
    """Run `statement` with named `:params` and return rows as dicts of strings."""
    response = ws.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=warehouse_id,
        parameters=[_parameter(name, value) for name, value in params.items()] or None,
        wait_timeout="30s",
    )
    while response.status.state in _RUNNING:
        time.sleep(1)
        response = ws.statement_execution.get_statement(response.statement_id)
    if response.status.state != StatementState.SUCCEEDED:
        error = response.status.error
        raise RuntimeError(error.message if error else f"Statement {response.status.state.value}")

    if not response.manifest or not response.result:
        return []
    columns = [column.name for column in response.manifest.schema.columns]
    rows = list(response.result.data_array or [])
    next_chunk = response.result.next_chunk_index
    while next_chunk is not None:
        chunk = ws.statement_execution.get_statement_result_chunk_n(response.statement_id, next_chunk)
        rows.extend(chunk.data_array or [])
        next_chunk = chunk.next_chunk_index
    return [dict(zip(columns, row)) for row in rows]


def _parameter(name: str, value) -> StatementParameterListItem:
    if isinstance(value, bool):
        sql_type = "BOOLEAN"
    elif isinstance(value, int):
        # INT where it fits: with ANSI mode a BIGINT is not narrowed implicitly (e.g. left(s, n)).
        sql_type = "INT" if -(2**31) <= value < 2**31 else "BIGINT"
    elif isinstance(value, float):
        sql_type = "DOUBLE"
    else:
        sql_type = "STRING"
    if value is not None:
        value = str(value).lower() if isinstance(value, bool) else str(value)
    return StatementParameterListItem(name=name, value=value, type=sql_type)


def to_int(value) -> int | None:
    return None if value in (None, "") else int(float(value))
