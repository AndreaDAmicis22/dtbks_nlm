"""Notebooks and their sources.

A notebook is a top-level folder of the volume and its sources are the files
inside it; ingestion status comes from the `documents` table written by the
ingest job.
"""

from __future__ import annotations

import io
import posixpath
import re
from dataclasses import dataclass

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

from . import sql
from .config import DEFAULT_NOTEBOOK, Settings

_UNSAFE_CHARS = re.compile(r"[^\w .\-()&+,']")


@dataclass(frozen=True)
class Source:
    path: str
    file_name: str
    status: str  # ok | error | unsupported | pending
    doc_id: str | None = None
    num_pages: int | None = None
    num_chunks: int | None = None
    error: str | None = None


def clean_name(name: str) -> str:
    """Safe file/folder name. Leading '.' or '_' would make Spark skip the file."""
    clean = " ".join(posixpath.basename(name.replace("\\", "/")).split())
    clean = _UNSAFE_CHARS.sub("", clean).lstrip("._ ").strip()
    if not clean:
        raise ValueError("Nome non valido")
    return clean[:120]


def notebook_dir(settings: Settings, notebook: str) -> str:
    return f"{settings.volume_root}/{notebook}"


def list_notebooks(ws: WorkspaceClient, settings: Settings) -> list[str]:
    names = set()
    for entry in ws.files.list_directory_contents(settings.volume_root):
        if entry.name.startswith((".", "_")):  # skipped by the ingest job as well
            continue
        names.add(entry.name if entry.is_directory else DEFAULT_NOTEBOOK)
    return sorted(names, key=str.casefold)


def create_notebook(ws: WorkspaceClient, settings: Settings, name: str) -> str:
    name = clean_name(name)
    ws.files.create_directory(notebook_dir(settings, name))
    return name


def list_sources(ws: WorkspaceClient, settings: Settings, notebook: str) -> list[Source]:
    rows = sql.query(
        ws,
        settings.warehouse_id,
        f"SELECT doc_id, path, file_name, status, error, num_pages, num_chunks "
        f"FROM {settings.table('documents')} WHERE notebook = :notebook",
        notebook=notebook,
    )
    sources = {
        row["path"]: Source(
            path=row["path"],
            file_name=row["file_name"],
            status=row["status"],
            doc_id=row["doc_id"],
            num_pages=sql.to_int(row["num_pages"]),
            num_chunks=sql.to_int(row["num_chunks"]),
            error=row["error"],
        )
        for row in rows
    }
    # Files already in the volume but not processed yet by the ingest job.
    for path, name in _volume_files(ws, settings, notebook):
        sources.setdefault(path, Source(path=path, file_name=name, status="pending"))
    return sorted(sources.values(), key=lambda s: s.file_name.casefold())


def _volume_files(ws: WorkspaceClient, settings: Settings, notebook: str):
    folders = [notebook_dir(settings, notebook)]
    if notebook == DEFAULT_NOTEBOOK:
        folders.append(settings.volume_root)
    for folder in folders:
        try:
            for entry in ws.files.list_directory_contents(folder):
                if not entry.is_directory and not entry.name.startswith((".", "_")):
                    yield entry.path, entry.name
        except NotFound:
            continue


def upload(ws: WorkspaceClient, settings: Settings, notebook: str, file_name: str, data: bytes) -> str:
    path = f"{notebook_dir(settings, notebook)}/{clean_name(file_name)}"
    ws.files.upload(path, io.BytesIO(data), overwrite=True)
    return path


def delete(ws: WorkspaceClient, path: str) -> None:
    ws.files.delete(path)


def trigger_ingest(app_ws: WorkspaceClient, settings: Settings) -> None:
    """Start the ingest job now (the file-arrival trigger would also start it, but later)."""
    if settings.ingest_job_id is not None:
        app_ws.jobs.run_now(job_id=settings.ingest_job_id)


def ingest_running(app_ws: WorkspaceClient, settings: Settings) -> bool:
    if settings.ingest_job_id is None:
        return False
    runs = app_ws.jobs.list_runs(job_id=settings.ingest_job_id, active_only=True, limit=1)
    return next(iter(runs), None) is not None
