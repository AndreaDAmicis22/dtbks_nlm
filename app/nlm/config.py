"""App settings, read from the environment (see app.yaml)."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Files placed directly in the volume root (not in a notebook folder) end up here.
DEFAULT_NOTEBOOK = "Generale"


@dataclass(frozen=True)
class Settings:
    catalog: str
    schema: str
    volume: str
    warehouse_id: str
    llm_endpoint: str
    ingest_job_id: int | None
    # "app": everything runs as the app service principal.
    # "user": data access runs on behalf of the signed-in user (needs user authorization scopes).
    auth_mode: str = "app"
    num_passages: int = 8
    history_messages: int = 6
    studio_max_chars: int = 300_000

    @classmethod
    def from_env(cls) -> "Settings":
        job_id = os.getenv("NLM_INGEST_JOB_ID")
        return cls(
            catalog=_required("NLM_CATALOG"),
            schema=_required("NLM_SCHEMA"),
            volume=_required("NLM_VOLUME"),
            warehouse_id=_required("DATABRICKS_WAREHOUSE_ID"),
            llm_endpoint=_required("NLM_LLM_ENDPOINT"),
            ingest_job_id=int(job_id) if job_id else None,
            auth_mode=os.getenv("NLM_AUTH_MODE", "app").lower(),
            num_passages=int(os.getenv("NLM_NUM_PASSAGES", "8")),
            history_messages=int(os.getenv("NLM_HISTORY_MESSAGES", "6")),
            studio_max_chars=int(os.getenv("NLM_STUDIO_MAX_CHARS", "300000")),
        )

    @property
    def volume_root(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

    @property
    def index_name(self) -> str:
        return f"{self.catalog}.{self.schema}.chunks_index"

    def table(self, name: str) -> str:
        return f"`{self.catalog}`.`{self.schema}`.`{name}`"


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Variabile d'ambiente mancante: {name}")
    return value
