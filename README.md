# NotebookLM su Databricks

Chat in stile NotebookLM sui documenti di un volume Unity Catalog: risposte basate solo sulle fonti,
con citazioni a documento e pagina, una cartella per notebook e uno "Studio" per riassunti, FAQ e guide.
Tutto resta dentro Databricks: permessi Unity Catalog, Model Serving, Vector Search (AI Search),
serverless jobs e Databricks Apps.

## Architettura

```
/Volumes/<catalog>/<schema>/sources/<notebook>/<file>      ← upload dall'app o copia nel volume
        │  trigger all'arrivo dei file, oppure avvio dall'app
        ▼
job nlm-ingest (serverless)
  ai_parse_document ───────────► parsed_documents   (parsing salvato: non si ripaga)
  chunking con pagine e sezioni ► chunks (Change Data Feed) + documents
        │  sync
        ▼
Vector Search: chunks_index  (Delta Sync, embedding Qwen3 multilingue, ricerca ibrida)
        │
        ▼
Databricks App (Streamlit)
  notebook = cartella · fonti selezionabili · chat con citazioni [n] · Studio
  LLM via Model Serving (default databricks-claude-opus-5-5) · storico in chat_messages
```

**Cosa fa**

- **Notebook**: ogni cartella di primo livello del volume è un notebook. I file nella radice finiscono in "Generale".
- **Fonti**:
  - PDF, DOC/DOCX, PPT/PPTX e immagini (PNG, JPG, TIFF) passano da `ai_parse_document` (massimo 100 MB e 500 pagine per file).
  - TXT e MD si leggono direttamente.
  - Nella sidebar ogni fonte si può includere o escludere dalla chat.
- **Chat**:
  - Una domanda di follow-up ("e il secondo punto?") viene riscritta in una query autonoma.
  - La ricerca è ibrida (vettoriale + keyword) e filtrata su notebook e fonti selezionate.
  - Il modello risponde solo con i passaggi recuperati e li cita come `[n]`; ogni citazione si espande in file, pagina, sezione e testo.
- **Studio**: genera riassunto, briefing, guida allo studio, FAQ, cronologia o istruzioni libere. Lavora sul testo completo delle fonti selezionate (con un limite configurabile e un avviso se il testo viene troncato).
- **Ingest incrementale**: gestisce file nuovi, modificati e rimossi. Il risultato del parsing viene salvato, così cambiare il chunking non costa un nuovo parsing.

## Struttura

| Percorso | Contenuto |
|---|---|
| `databricks.yml` | Bundle: variabili e target `dev` / `prod` |
| `resources/jobs.yml` | Job `nlm-setup` e `nlm-ingest` (serverless) |
| `resources/app.yml` | App, risorse (warehouse, endpoint LLM, job) e variabili d'ambiente |
| `src/ingest/00_setup.py` | Notebook: tabelle, endpoint e indice di Vector Search, GRANT all'app |
| `src/ingest/01_ingest.py` | Notebook: inventario, parsing, chunking, scrittura, sync dell'indice |
| `src/ingest/chunking.py` | Chunking (Python puro, testato) |
| `app/app.py` | Interfaccia Streamlit |
| `app/nlm/` | Logica dell'app: retrieval, prompt, citazioni, LLM, fonti, storico, Studio |
| `tests/` | Test unitari e smoke test della UI (tutte le chiamate Databricks simulate) |

## Prerequisiti

- Workspace con Unity Catalog, serverless (job e SQL warehouse), Databricks Apps e Vector Search.
- AI Functions (`ai_parse_document`) disponibili nella region del workspace, altrimenti serve il cross-geo processing.
- Endpoint di Model Serving `databricks-claude-opus-5-5` (o un altro LLM a scelta) e `databricks-qwen3-embedding-0-6b` (Public Preview).
  - Gli embedding `gte`/`bge` sono solo in inglese: per documenti in italiano usa Qwen3.
- Un catalog su cui hai `CREATE SCHEMA`, oppure uno schema esistente su cui hai `CREATE TABLE` e `CREATE VOLUME`.
- Il permesso di creare un endpoint di Vector Search.
- Databricks CLI recente: i bundle sono stati validati contro lo schema della v1.21.

## Deploy

1. **Crea schema e volume** (una volta sola: il trigger del job di ingest punta al volume).

   ```sql
   CREATE SCHEMA IF NOT EXISTS <catalog>.notebooklm;
   CREATE VOLUME IF NOT EXISTS <catalog>.notebooklm.sources;
   ```

2. **Configura il target** in `databricks.yml`: `host`, `catalog` e `warehouse_id`. In alternativa passali a ogni comando con `--var="catalog=...,warehouse_id=..."`.

3. **Deploy di job e app.**

   ```bash
   databricks bundle deploy -t dev
   ```

4. **Setup**: crea le tabelle, l'endpoint e l'indice di Vector Search e concede i permessi al service principal dell'app.

   ```bash
   databricks bundle run nlm_setup -t dev
   ```

   - La prima volta l'endpoint di Vector Search richiede 10-20 minuti.
   - Se un `GRANT` fallisce perché non possiedi il catalog, il job lo segnala: fallo eseguire a chi lo possiede.

5. **Avvia l'app.** Il comando stampa l'URL; ripetilo dopo ogni modifica al codice dell'app.

   ```bash
   databricks bundle run nlm_app -t dev
   ```

6. **Usa l'app**: crea un notebook e carica i documenti. In alternativa copiali nella cartella del volume:
   - in `prod` il job parte da solo all'arrivo dei file;
   - in `dev` i trigger sono in pausa, quindi lancia `databricks bundle run nlm_ingest -t dev`.

Dopo l'ingest l'indice impiega uno o due minuti a sincronizzarsi. Una fonte può quindi risultare indicizzata qualche istante prima di comparire nelle risposte.

## Configurazione

| Variabile del bundle | Default | Note |
|---|---|---|
| `catalog` | — | Obbligatoria |
| `warehouse_id` | — | Obbligatoria; usata dall'app per documenti, Studio e storico |
| `schema` / `volume` | `notebooklm` / `sources` | |
| `llm_endpoint` | `databricks-claude-opus-5-5` | Qualsiasi endpoint chat di Model Serving |
| `embedding_endpoint` | `databricks-qwen3-embedding-0-6b` | Se lo cambi va ricreato l'indice: elimina `chunks_index` e rilancia il setup |
| `vs_endpoint` | `nlm-vector-search` | |
| `app_name` | `notebooklm` (`notebooklm-dev` in dev) | Unico nel workspace: la modalità development non rinomina le app |
| `auth_mode` | `app` | Vedi [Identità e permessi](#identità-e-permessi) |

L'app legge anche altre variabili d'ambiente, che puoi aggiungere in `resources/app.yml`:

- `NLM_NUM_PASSAGES`: passaggi per risposta (default 8);
- `NLM_HISTORY_MESSAGES`: messaggi di contesto (default 6);
- `NLM_STUDIO_MAX_CHARS`: limite di testo dello Studio (default 300.000 caratteri, circa 75-100k token).

Il chunking usa 2000 caratteri con 200 di overlap (`chunk_elements` in `src/ingest/chunking.py`). Dopo averlo modificato, lancia l'ingest con `reprocess=true`:

```bash
databricks bundle run nlm_ingest -t dev --params reprocess=true
```

`reprocess=true` rifà il chunking riusando il parsing salvato e riprova i file andati in errore.

## Identità e permessi

- **`auth_mode: app`** (default): l'app legge, cerca e chiama il modello come il proprio service principal, con i permessi concessi dal job di setup. Chiunque possa usare l'app vede tutte le fonti del volume.
- **`auth_mode: user`** (on-behalf-of-user, GA): lettura delle fonti, ricerca e chiamate al modello avvengono con i permessi dell'utente collegato. Per attivarla:
  1. imposta `auth_mode: user`;
  2. scommenta `user_api_scopes` in `resources/app.yml`;
  3. concedi agli utenti (o al loro gruppo) `READ VOLUME`, `WRITE VOLUME`, `SELECT` su `documents` e `SELECT` sull'indice.

  Anche in questa modalità lo storico chat viene scritto dal service principal, e il job di ingest viene avviato dal service principal, perché non esiste uno scope OAuth per i job.
- **Condividere l'app**: aggiungi `permissions` in `resources/app.yml` (c'è un esempio commentato).

## Modello e API

L'app chiama il modello tramite le API OpenAI-compatibili di Model Serving (`/serving-endpoints`).

- Puoi passare a qualsiasi endpoint chat cambiando solo `llm_endpoint`. Le chiamate passano comunque dall'AI Gateway di Databricks, con i suoi limiti e il suo tracciamento d'uso.
- Il codice non invia parametri di sampling come `temperature`, che i modelli Claude più recenti rifiutano.
- Se in futuro vuoi usare solo Claude con funzioni native come le citazioni dei documenti, Databricks espone anche la Messages API di Anthropic (`/ai-gateway/anthropic`).

## Sviluppo locale

```bash
pip install -r requirements-dev.txt
pytest
```

Per far girare l'app in locale contro il workspace (autenticazione dal profilo della CLI), esempio PowerShell:

```powershell
$env:DATABRICKS_CONFIG_PROFILE = "<profilo>"
$env:NLM_CATALOG = "<catalog>"; $env:NLM_SCHEMA = "notebooklm"; $env:NLM_VOLUME = "sources"
$env:DATABRICKS_WAREHOUSE_ID = "<warehouse-id>"; $env:NLM_LLM_ENDPOINT = "databricks-claude-opus-5-5"
$env:NLM_INGEST_JOB_ID = "<job-id di nlm-ingest>"
cd app; streamlit run app.py
```

## Costi da tenere d'occhio

- **Parsing**: `ai_parse_document` è fatturato a pagina, ma si paga una sola volta per versione del file grazie a `parsed_documents`.
- **Vector Search**: l'endpoint standard ha un costo orario anche quando è inattivo. Eliminalo dagli ambienti che non usi.
- **LLM**: si paga a token. Lo Studio invia il testo completo delle fonti selezionate, fino a `NLM_STUDIO_MAX_CHARS`.

## Possibili evoluzioni

- **Reranker** di Vector Search (`reranker=` in `query_index`) per risposte più precise.
- **`ai_prep_search`** (Beta, DBR 18.2+) al posto del chunking custom: produce chunk con contesto e pagine.
- **Tracing e valutazione** con MLflow (Agent Evaluation) per misurare la qualità delle risposte.
- **Storico su Lakebase** invece che sul SQL warehouse, per latenze più basse.
- **Unity Gateway**: aggiungere l'LLM come model service (`uc_securable` di tipo `MODEL_SERVICE`). Nel selettore delle risorse delle app gli endpoint `databricks-*` sono marcati come legacy.
- **Audio overview** in stile NotebookLM, con un TTS esterno.

## Alternativa senza codice

Se basta una chat con citazioni su un volume, senza notebook, Studio o interfaccia su misura, valuta prima i Genie Agents con volumi allegati ("Analyze files in volumes", Beta). Mantengono l'indice sincronizzato da soli. Knowledge Assistant ora è marcato come legacy.
