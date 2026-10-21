# Databricks notebook source
# MAGIC %md
# MAGIC # Setup
# MAGIC
# MAGIC Crea, se mancano, schema, volume, tabelle, endpoint e indice di Vector Search (AI Search),
# MAGIC poi concede al service principal dell'app i permessi Unity Catalog che le servono.
# MAGIC È idempotente: si può rieseguire senza effetti collaterali.
# MAGIC
# MAGIC La prima creazione dell'endpoint di Vector Search può richiedere 10-20 minuti.

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "notebooklm")
dbutils.widgets.text("volume", "sources")
dbutils.widgets.text("vs_endpoint", "nlm-vector-search")
dbutils.widgets.text("embedding_endpoint", "databricks-qwen3-embedding-0-6b")
dbutils.widgets.text("app_principal", "", "Application ID del service principal dell'app")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
VS_ENDPOINT = dbutils.widgets.get("vs_endpoint")
EMBEDDING_ENDPOINT = dbutils.widgets.get("embedding_endpoint")
APP_PRINCIPAL = dbutils.widgets.get("app_principal").strip()
assert CATALOG, "Il parametro 'catalog' è obbligatorio"


def fq(name: str) -> str:
    return f"`{CATALOG}`.`{SCHEMA}`.`{name}`"


DOCUMENTS, PARSED, CHUNKS, CHAT = fq("documents"), fq("parsed_documents"), fq("chunks"), fq("chat_messages")
CHUNKS_NAME = f"{CATALOG}.{SCHEMA}.chunks"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.chunks_index"

# COMMAND ----------

# MAGIC %md ## Schema, volume e tabelle

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {fq(VOLUME)} COMMENT 'Fonti: una cartella per notebook'")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {DOCUMENTS} (
  doc_id STRING NOT NULL,
  notebook STRING NOT NULL,
  path STRING NOT NULL,
  file_name STRING,
  file_size BIGINT,
  modification_time TIMESTAMP,
  status STRING COMMENT 'ok | error | unsupported',
  error STRING,
  num_pages INT,
  num_chunks INT,
  text STRING COMMENT 'Testo completo con marcatori [Pagina N], usato dallo Studio',
  ingested_at TIMESTAMP
) COMMENT 'Una riga per file del volume, scritta dal job di ingest'
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {PARSED} (
  path STRING NOT NULL,
  modification_time TIMESTAMP,
  parsed VARIANT,
  parsed_at TIMESTAMP
) COMMENT 'Output grezzo di ai_parse_document: se cambia il chunking non si ripaga il parsing'
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CHUNKS} (
  chunk_id STRING NOT NULL,
  doc_id STRING NOT NULL,
  notebook STRING NOT NULL,
  path STRING,
  file_name STRING,
  chunk_index INT,
  page_start INT,
  page_end INT,
  section STRING,
  content STRING COMMENT 'Testo mostrato nelle citazioni',
  embed_text STRING COMMENT 'Testo indicizzato: intestazione documento/sezione + content'
) COMMENT 'Sorgente del Delta Sync index'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CHAT} (
  message_id STRING NOT NULL,
  notebook STRING NOT NULL,
  user_email STRING NOT NULL,
  role STRING NOT NULL,
  content STRING,
  passages STRING COMMENT 'JSON dei passaggi recuperati per la risposta',
  created_at TIMESTAMP
) COMMENT 'Storico delle chat dell app, per utente e notebook'
""")

# COMMAND ----------

# MAGIC %md ## Vector Search: endpoint e indice

# COMMAND ----------

from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    EndpointType,
    PipelineType,
    VectorIndexType,
)

w = WorkspaceClient()

try:
    w.vector_search_endpoints.get_endpoint(endpoint_name=VS_ENDPOINT)
    print(f"Endpoint {VS_ENDPOINT} già presente")
except NotFound:
    print(f"Creo l'endpoint {VS_ENDPOINT} (può richiedere diversi minuti)...")
    w.vector_search_endpoints.create_endpoint_and_wait(
        name=VS_ENDPOINT, endpoint_type=EndpointType.STANDARD, timeout=timedelta(minutes=45)
    )

try:
    w.vector_search_indexes.get_index(index_name=INDEX_NAME)
    print(f"Indice {INDEX_NAME} già presente")
except NotFound:
    print(f"Creo l'indice {INDEX_NAME} con embedding {EMBEDDING_ENDPOINT}")
    w.vector_search_indexes.create_index(
        name=INDEX_NAME,
        endpoint_name=VS_ENDPOINT,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=CHUNKS_NAME,
            pipeline_type=PipelineType.TRIGGERED,
            embedding_source_columns=[
                EmbeddingSourceColumn(name="embed_text", embedding_model_endpoint_name=EMBEDDING_ENDPOINT)
            ],
        ),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Permessi per l'app
# MAGIC
# MAGIC Con `NLM_AUTH_MODE=app` l'app legge e scrive come il proprio service principal.
# MAGIC Con `NLM_AUTH_MODE=user` legge come l'utente collegato: in quel caso concedi `READ VOLUME`,
# MAGIC `WRITE VOLUME`, `SELECT` su `documents` e sull'indice anche agli utenti (o al loro gruppo).
# MAGIC Lo storico chat è sempre scritto dal service principal.

# COMMAND ----------

if not APP_PRINCIPAL:
    print("Nessun service principal indicato: salto i GRANT")
else:
    principal = APP_PRINCIPAL
    grants = [
        f"GRANT USE CATALOG ON CATALOG `{CATALOG}` TO `{principal}`",
        f"GRANT USE SCHEMA ON SCHEMA `{CATALOG}`.`{SCHEMA}` TO `{principal}`",
        f"GRANT READ VOLUME, WRITE VOLUME ON VOLUME {fq(VOLUME)} TO `{principal}`",
        f"GRANT SELECT ON TABLE {DOCUMENTS} TO `{principal}`",
        f"GRANT SELECT, MODIFY ON TABLE {CHAT} TO `{principal}`",
        f"GRANT SELECT ON TABLE {fq('chunks_index')} TO `{principal}`",
    ]
    failed = []
    for statement in grants:
        try:
            spark.sql(statement)
            print("OK  ", statement)
        except Exception as error:  # e.g. not the owner of the catalog: report and go on
            failed.append(statement)
            print("FAIL", statement, "->", str(error).splitlines()[0])
    if failed:
        raise RuntimeError(
            f"{len(failed)} GRANT non riusciti: falli eseguire a chi possiede gli oggetti (vedi output sopra)"
        )
