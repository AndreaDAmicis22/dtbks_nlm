# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest delle fonti
# MAGIC
# MAGIC Allinea l'indice al contenuto del volume:
# MAGIC 1. confronta i file del volume con la tabella `documents` (nuovi, modificati, rimossi);
# MAGIC 2. fa il parsing di PDF, Office e immagini con `ai_parse_document` (output salvato in `parsed_documents`);
# MAGIC 3. divide il testo in chunk con pagine e sezione, aggiorna `chunks` e `documents`;
# MAGIC 4. avvia il sync dell'indice di Vector Search.
# MAGIC
# MAGIC Ogni cartella di primo livello del volume è un notebook; i file nella radice finiscono nel notebook "Generale".
# MAGIC
# MAGIC `reprocess=true` rifà il chunking di tutti i file riusando il parsing già salvato (utile se cambi i
# MAGIC parametri del chunking) e riprova il parsing dei file andati in errore.

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "notebooklm")
dbutils.widgets.text("volume", "sources")
dbutils.widgets.dropdown("reprocess", "false", ["false", "true"])

# COMMAND ----------

import json
import os
import time
from datetime import datetime, timedelta, timezone

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from pyspark.sql import functions as F

from chunking import (
    TEXT_EXTENSIONS,
    build_chunk_rows,
    doc_id_for,
    elements_from_parsed,
    elements_from_text,
    render_document_text,
)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
REPROCESS = dbutils.widgets.get("reprocess") == "true"
assert CATALOG, "Il parametro 'catalog' è obbligatorio"

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
DEFAULT_NOTEBOOK = "Generale"  # same value as app/nlm/config.py
PARSE_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".doc", ".docx", ".ppt", ".pptx"}
BATCH_SIZE = 20


def fq(name: str) -> str:
    return f"`{CATALOG}`.`{SCHEMA}`.`{name}`"


DOCUMENTS, PARSED, CHUNKS = fq("documents"), fq("parsed_documents"), fq("chunks")
INDEX_NAME = f"{CATALOG}.{SCHEMA}.chunks_index"

CHUNK_SCHEMA = (
    "chunk_id STRING, doc_id STRING, notebook STRING, path STRING, file_name STRING, chunk_index INT, "
    "page_start INT, page_end INT, section STRING, content STRING, embed_text STRING"
)
DOCUMENT_SCHEMA = (
    "doc_id STRING, notebook STRING, path STRING, file_name STRING, file_size BIGINT, mtime_ms BIGINT, "
    "status STRING, error STRING, num_pages INT, num_chunks INT, text STRING"
)

# COMMAND ----------

# MAGIC %md ## 1. Cosa è cambiato
# MAGIC
# MAGIC I tempi di modifica si confrontano in millisecondi interi: un confronto tra `datetime` Python
# MAGIC dipende dal fuso orario della sessione e potrebbe far rielaborare tutto a ogni run.

# COMMAND ----------


def volume_files():
    # Spark reads the `content` column only when it is selected: listing metadata is cheap.
    return (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load(VOLUME_ROOT)
        .withColumn("path", F.regexp_replace("path", "^dbfs:", ""))
    )


inventory = {
    row.path: row
    for row in volume_files()
    .select("path", F.unix_millis("modificationTime").alias("mtime_ms"), F.col("length").alias("file_size"))
    .collect()
}
existing = {
    row.path: row.mtime_ms
    for row in spark.table(DOCUMENTS)
    .select("path", F.unix_millis("modification_time").alias("mtime_ms"))
    .collect()
}
parsed_keys = {
    (row.path, row.mtime_ms)
    for row in spark.table(PARSED)
    .select("path", F.unix_millis("modification_time").alias("mtime_ms"))
    .collect()
}

changed = sorted(path for path, file in inventory.items() if REPROCESS or existing.get(path) != file.mtime_ms)
removed = sorted(path for path in existing if path not in inventory)
print(f"{len(inventory)} file nel volume: {len(changed)} da elaborare, {len(removed)} rimossi")

# COMMAND ----------

# MAGIC %md ## 2-3. Parsing, chunking e scrittura (a lotti)

# COMMAND ----------


def extension(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def locate(path: str) -> tuple[str, str]:
    """(notebook, file name): the notebook is the first folder under the volume root."""
    parts = path[len(VOLUME_ROOT) + 1 :].split("/")
    return (parts[0] if len(parts) > 1 else DEFAULT_NOTEBOOK), parts[-1]


def paths_df(paths):
    return F.broadcast(spark.createDataFrame([(path,) for path in paths], "path STRING"))


def to_df(rows: list[dict], schema: str):
    columns = [field.strip().split()[0] for field in schema.split(",")]
    return spark.createDataFrame([tuple(row[c] for c in columns) for row in rows], schema)


def read_contents(paths):
    """Content of the given files only. Paths are joined rather than passed to `load()`,
    where characters like [ ] { } would be read as glob patterns."""
    oldest = min(inventory[path].mtime_ms for path in paths)
    since = datetime.fromtimestamp(oldest / 1000, tz=timezone.utc) - timedelta(days=1)
    return (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .option("modifiedAfter", since.strftime("%Y-%m-%dT%H:%M:%S"))
        .load(VOLUME_ROOT)
        .withColumn("path", F.regexp_replace("path", "^dbfs:", ""))
        .join(paths_df(paths), "path")
    )


def delete_where(table: str, column: str, values) -> None:
    values = sorted(set(values))
    if not values:
        return
    spark.createDataFrame([(v,) for v in values], "value STRING").createOrReplaceTempView("nlm_delete_keys")
    spark.sql(f"MERGE INTO {table} t USING nlm_delete_keys s ON t.{column} = s.value WHEN MATCHED THEN DELETE")


parse_errors: dict[str, str] = {}


def parse(paths) -> None:
    """ai_parse_document on `paths`, appended to parsed_documents.

    Written with a plain append (not MERGE) so that the function runs exactly once per file.
    If the batch fails, files are retried one by one and failures are recorded per file.
    """
    try:
        parsed = (
            read_contents(paths)
            .repartition(len(paths))  # one file per task, so files are parsed in parallel
            .selectExpr(
                "path",
                "modificationTime AS modification_time",
                "ai_parse_document(content, map('version', '2.0')) AS parsed",
                "current_timestamp() AS parsed_at",
            )
        )
        delete_where(PARSED, "path", paths)
        parsed.write.mode("append").saveAsTable(PARSED)
    except Exception as error:
        if len(paths) == 1:
            parse_errors[paths[0]] = f"ai_parse_document: {error}"[:2000]
            return
        for path in paths:
            parse([path])
        if all(path in parse_errors for path in paths):
            # Every file failing points at a configuration problem, not at the files.
            raise RuntimeError(f"Parsing fallito per tutti i file del lotto: {parse_errors[paths[0]]}")


def load_parsed(paths) -> dict:
    if not paths:
        return {}
    rows = (
        spark.table(PARSED)
        .join(paths_df(paths), "path")
        .select("path", F.unix_millis("modification_time").alias("mtime_ms"), F.to_json("parsed").alias("parsed"))
        .collect()
    )
    return {row.path: json.loads(row.parsed) for row in rows if row.mtime_ms == inventory[row.path].mtime_ms}


def load_texts(paths) -> dict:
    """Plain text files are small: read them directly from the /Volumes path on the driver."""
    texts = {}
    for path in paths:
        try:
            with open(path, "rb") as file:
                data = file.read()
        except OSError as error:  # e.g. deleted after the inventory: next run cleans it up
            print(f"Lettura non riuscita per {path}: {error}")
            continue
        try:
            texts[path] = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            texts[path] = data.decode("cp1252", errors="replace")
    return texts


def process(path: str, parsed: dict | None, text: str | None) -> tuple[dict, list[dict]]:
    notebook, file_name = locate(path)
    doc_id = doc_id_for(path)
    ext = extension(path)
    status, error, elements, num_pages = "ok", None, [], None

    if ext in TEXT_EXTENSIONS:
        elements = elements_from_text(text or "")
    elif ext not in PARSE_EXTENSIONS:
        status, error = "unsupported", f"Formato '{ext or '?'}' non supportato"
    elif path in parse_errors:
        status, error = "error", parse_errors[path]
    elif parsed is None:
        status, error = "error", "Output di ai_parse_document non trovato"
    else:
        elements = elements_from_parsed(parsed)
        num_pages = len((parsed.get("document") or {}).get("pages") or []) or None
        if page_errors := parsed.get("error_status") or []:
            messages = "; ".join(sorted({e.get("error_message") or "?" for e in page_errors}))
            error = f"{len(page_errors)} errori di parsing: {messages}"[:2000]

    if status == "ok" and not elements:
        status, error = "error", error or "Nessun testo estratto"
    chunks = build_chunk_rows(doc_id, notebook, path, file_name, elements) if status == "ok" else []
    document = {
        "doc_id": doc_id,
        "notebook": notebook,
        "path": path,
        "file_name": file_name,
        "file_size": inventory[path].file_size,
        "mtime_ms": inventory[path].mtime_ms,
        "status": status,
        "error": error,
        "num_pages": num_pages,
        "num_chunks": len(chunks),
        "text": render_document_text(elements) if chunks else None,
    }
    return document, chunks


def write(documents: list[dict], chunks: list[dict]) -> None:
    delete_where(CHUNKS, "doc_id", [d["doc_id"] for d in documents])
    if chunks:
        to_df(chunks, CHUNK_SCHEMA).write.mode("append").saveAsTable(CHUNKS)
    (
        to_df(documents, DOCUMENT_SCHEMA)
        .withColumn("modification_time", F.expr("timestamp_millis(mtime_ms)"))
        .drop("mtime_ms")
        .withColumn("ingested_at", F.current_timestamp())
        .createOrReplaceTempView("nlm_documents_batch")
    )
    spark.sql(
        f"MERGE INTO {DOCUMENTS} t USING nlm_documents_batch s ON t.path = s.path "
        "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *"
    )


for start in range(0, len(changed), BATCH_SIZE):
    batch = changed[start : start + BATCH_SIZE]
    to_parse = [
        path
        for path in batch
        if extension(path) in PARSE_EXTENSIONS and (path, inventory[path].mtime_ms) not in parsed_keys
    ]
    if to_parse:
        parse(to_parse)
    parsed = load_parsed([path for path in batch if extension(path) in PARSE_EXTENSIONS])
    texts = load_texts([path for path in batch if extension(path) in TEXT_EXTENSIONS])

    documents, chunks = [], []
    for path in batch:
        document, doc_chunks = process(path, parsed.get(path), texts.get(path))
        documents.append(document)
        chunks.extend(doc_chunks)
    write(documents, chunks)
    statuses = [d["status"] for d in documents]
    print(f"Lotto {start // BATCH_SIZE + 1}: {statuses.count('ok')} ok, {len(chunks)} chunk, "
          f"{statuses.count('error')} errori, {statuses.count('unsupported')} non supportati")

if removed:
    delete_where(CHUNKS, "doc_id", [doc_id_for(path) for path in removed])
    delete_where(PARSED, "path", removed)
    delete_where(DOCUMENTS, "path", removed)
    print(f"Rimossi {len(removed)} documenti")

# COMMAND ----------

# MAGIC %md ## 4. Sync dell'indice
# MAGIC
# MAGIC Si avvia sempre, anche senza modifiche: così un run interrotto prima del sync si recupera al run successivo.

# COMMAND ----------

w = WorkspaceClient()


def sync_index(timeout_minutes: int = 30) -> None:
    deadline = time.monotonic() + timeout_minutes * 60
    while True:
        try:
            status = w.vector_search_indexes.get_index(index_name=INDEX_NAME).status
        except NotFound:
            raise RuntimeError(f"Indice {INDEX_NAME} non trovato: esegui prima il job di setup")
        last_error = None
        if status and status.ready:
            try:
                w.vector_search_indexes.sync_index(index_name=INDEX_NAME)
                print("Sync dell'indice avviato")
                return
            except Exception as error:  # usually a sync already in progress: retry later
                last_error = error
        if time.monotonic() > deadline:
            raise TimeoutError(f"Sync non avviato entro {timeout_minutes} minuti: {last_error or status}")
        time.sleep(30)


sync_index()
