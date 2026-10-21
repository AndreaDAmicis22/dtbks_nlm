"""NotebookLM-style app: chat with citations and Studio over the sources of a notebook."""

from __future__ import annotations

import logging

import streamlit as st
from databricks.sdk import WorkspaceClient

from nlm import history, library, llm, rag, retrieval, studio
from nlm.config import Settings
from nlm.passages import Passage, cited_passages
from nlm.prompts import STUDIO_PRESETS

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Notebook", page_icon="📓", layout="wide")

CUSTOM_PRESET = "Personalizzato"
UPLOAD_TYPES = ["pdf", "docx", "doc", "pptx", "ppt", "png", "jpg", "jpeg", "tif", "tiff", "txt", "md"]
STATUS_LABELS = {
    "pending": ("⏳", "in attesa di indicizzazione"),
    "error": ("⚠️", "errore di indicizzazione"),
    "unsupported": ("🚫", "formato non supportato"),
}


# --------------------------------------------------------------------------
# Identities and cached reads
# --------------------------------------------------------------------------


@st.cache_resource
def settings() -> Settings:
    return Settings.from_env()


@st.cache_resource
def app_client() -> WorkspaceClient:
    """The app's service principal: Databricks Apps injects its credentials."""
    return WorkspaceClient()


def data_client(cfg: Settings) -> WorkspaceClient:
    """Identity used for sources, search and models: the signed-in user with OBO, otherwise the app."""
    if cfg.auth_mode != "user":
        return app_client()
    token = st.context.headers.get("x-forwarded-access-token")
    if not token:
        st.error("Token utente assente: abilita la user authorization dell'app oppure usa NLM_AUTH_MODE=app.")
        st.stop()
    return WorkspaceClient(host=app_client().config.host, token=token, auth_type="pat")


def current_user() -> str:
    if "user" not in st.session_state:
        # The header is missing only when running locally.
        st.session_state.user = (
            st.context.headers.get("x-forwarded-email") or app_client().current_user.me().user_name
        )
    return st.session_state.user


# The user is part of each cache key, so with OBO nobody sees another user's listing.
@st.cache_data(ttl=60, show_spinner=False)
def cached_notebooks(_ws: WorkspaceClient, user: str) -> list[str]:
    return library.list_notebooks(_ws, settings())


@st.cache_data(ttl=30, show_spinner=False)
def cached_sources(_ws: WorkspaceClient, user: str, notebook: str) -> list[library.Source]:
    return library.list_sources(_ws, settings(), notebook)


@st.cache_data(ttl=15, show_spinner=False)
def cached_ingest_running() -> bool:
    try:
        return library.ingest_running(app_client(), settings())
    except Exception:
        # Only an indicator: never block the page for it.
        logging.exception("Cannot read the ingest job runs")
        return False


def refresh() -> None:
    cached_notebooks.clear()
    cached_sources.clear()
    cached_ingest_running.clear()


def flash(message: str) -> None:
    """Toast shown after the next rerun."""
    st.session_state.flash = message


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def md(text: str) -> str:
    """Streamlit renders $...$ as LaTeX, and amounts in euro/dollars are common in documents."""
    return text.replace("$", "\\$")


def escaped_stream(chunks, sink: list[str]):
    """Keep the raw text in `sink` while streaming the escaped text."""
    for chunk in chunks:
        sink.append(chunk)
        yield md(chunk)


def show_citations(answer: str, passages: list[Passage]) -> None:
    cited = cited_passages(answer, passages)
    if not cited:
        return
    with st.expander(f"Fonti citate ({len(cited)})"):
        for passage in cited:
            title = f"**[{passage.n}] {md(passage.location)}**"
            if passage.section:
                title += f" · _{md(passage.section)}_"
            st.markdown(title)
            st.caption(md(passage.content))


def answer_question(
    ws: WorkspaceClient, cfg: Settings, notebook: str, question: str, recent: list[dict], doc_ids
) -> tuple[str, list[Passage]]:
    """Retrieve passages, then stream a grounded answer into the current container."""
    with llm.client_for(ws) as client:
        with st.spinner("Cerco nelle fonti…"):
            query = rag.search_query(client, cfg.llm_endpoint, recent, question)
            passages = retrieval.search(ws, cfg.index_name, query, notebook, doc_ids, cfg.num_passages)
        if query != question:
            st.caption(f"Ricerca: {md(query)}")
        if not passages:
            answer = "Non ho trovato passaggi pertinenti nelle fonti selezionate."
            st.markdown(answer)
            return answer, []
        sink: list[str] = []
        messages = rag.answer_messages(recent, question, passages)
        st.write_stream(escaped_stream(llm.stream(client, cfg.llm_endpoint, messages, max_tokens=16000), sink))
    answer = "".join(sink)
    show_citations(answer, passages)
    return answer, passages


def generate(ws: WorkspaceClient, cfg: Settings, documents: list[dict], instruction: str) -> str:
    sink: list[str] = []
    with llm.client_for(ws) as client, st.container(border=True):
        messages = studio.messages(documents, instruction)
        st.write_stream(escaped_stream(llm.stream(client, cfg.llm_endpoint, messages, max_tokens=32000), sink))
    return "".join(sink)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

cfg = settings()
ws = data_client(cfg)
user = current_user()

if message := st.session_state.pop("flash", None):
    st.toast(message)
# A widget's state can only be set before the widget is created.
if "next_notebook" in st.session_state:
    st.session_state.notebook = st.session_state.pop("next_notebook")

with st.sidebar:
    st.title("📓 Notebook")
    try:
        notebooks = cached_notebooks(ws, user)
    except Exception as error:
        st.error(f"Volume {cfg.volume_root} non accessibile: {error}")
        st.stop()

    with st.popover("➕ Nuovo notebook"):
        with st.form("new_notebook", clear_on_submit=True, border=False):
            name = st.text_input("Nome del notebook")
            if st.form_submit_button("Crea") and name.strip():
                try:
                    created = library.create_notebook(ws, cfg, name)
                except ValueError as error:
                    st.error(str(error))
                else:
                    refresh()
                    st.session_state.next_notebook = created
                    st.rerun()

    if not notebooks:
        st.info("Crea il primo notebook per iniziare.")
        st.stop()
    notebook = st.selectbox("Notebook", notebooks, key="notebook")

    st.subheader("Fonti")
    if cached_ingest_running():
        st.caption("🔄 Indicizzazione in corso…")
    try:
        sources = cached_sources(ws, user, notebook)
    except Exception as error:
        logging.exception("Cannot list sources")
        st.error(f"Fonti non leggibili (il job nlm-setup è stato eseguito?): {error}")
        sources = []
    ready = [s for s in sources if s.status == "ok"]
    selected: list[str] = []
    for source in sources:
        if source.status == "ok":
            details = f"{source.num_pages} pagine · " if source.num_pages else ""
            details += f"{source.num_chunks} passaggi"
            if source.error:
                details += f"\n\n⚠️ {source.error}"
            if st.checkbox(source.file_name, value=True, key=f"use:{notebook}:{source.doc_id}", help=details):
                selected.append(source.doc_id)
        else:
            icon, label = STATUS_LABELS.get(source.status, ("❔", source.status))
            st.caption(f"{icon} {source.file_name} — {label}", help=source.error)
    if not sources:
        st.caption("Nessuna fonte: aggiungine una qui sotto.")

    with st.expander("Aggiungi fonti"):
        with st.form("upload", clear_on_submit=True, border=False):
            files = st.file_uploader(
                "Documenti", type=UPLOAD_TYPES, accept_multiple_files=True, label_visibility="collapsed"
            )
            if st.form_submit_button("Carica e indicizza", type="primary") and files:
                try:
                    with st.spinner("Caricamento…"):
                        for file in files:
                            library.upload(ws, cfg, notebook, file.name, file.getvalue())
                        library.trigger_ingest(app_client(), cfg)
                except Exception as error:
                    logging.exception("Upload failed")
                    st.error(f"Caricamento non riuscito: {error}")
                else:
                    refresh()
                    flash(f"{len(files)} file caricati: indicizzazione avviata")
                    st.rerun()

    if sources:
        with st.expander("Gestisci fonti"):
            target = st.selectbox(
                "Fonte", sources, format_func=lambda s: s.file_name, key=f"manage:{notebook}"
            )
            if st.button("Elimina fonte"):
                try:
                    library.delete(ws, target.path)
                    library.trigger_ingest(app_client(), cfg)
                except Exception as error:
                    logging.exception("Delete failed")
                    st.error(f"Eliminazione non riuscita: {error}")
                else:
                    refresh()
                    flash(f"{target.file_name} eliminato")
                    st.rerun()

    if st.button("↻ Aggiorna"):
        refresh()
        st.rerun()

st.header(notebook)
tab_chat, tab_studio = st.tabs(["💬 Chat", "✨ Studio"])

with tab_chat:
    chat_key = f"chat:{notebook}"
    if chat_key not in st.session_state:
        try:
            st.session_state[chat_key] = history.load(app_client(), cfg, notebook, user)
        except Exception as error:
            logging.exception("History load failed")
            st.warning(f"Storico chat non disponibile: {error}")
            st.session_state[chat_key] = []
    messages: list[dict] = st.session_state[chat_key]

    if messages and st.button("🗑 Nuova conversazione"):
        history.clear(app_client(), cfg, notebook, user)
        st.session_state[chat_key] = []
        st.rerun()

    conversation = st.container()
    with conversation:
        if not messages:
            st.caption("Le risposte usano solo le fonti selezionate e citano i passaggi da cui provengono.")
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(md(message["content"]))
                if message["role"] == "assistant":
                    show_citations(message["content"], message["passages"])

    question = st.chat_input(
        "Chiedi qualcosa sulle fonti…" if selected else "Seleziona almeno una fonte",
        disabled=not selected,
    )
    if question:
        recent = messages[-cfg.history_messages :]
        # No doc filter when everything is selected: a smaller, faster query.
        doc_ids = None if len(selected) == len(ready) else selected
        with conversation:
            with st.chat_message("user"):
                st.markdown(md(question))
            with st.chat_message("assistant"):
                try:
                    answer, passages = answer_question(ws, cfg, notebook, question, recent, doc_ids)
                except Exception as error:
                    logging.exception("Chat failed")
                    st.error(f"Errore durante la risposta: {error}")
                    answer = None
        if answer is not None:
            messages += [
                {"role": "user", "content": question, "passages": []},
                {"role": "assistant", "content": answer, "passages": passages},
            ]
            try:
                history.append_turn(app_client(), cfg, notebook, user, question, answer, passages)
            except Exception as error:
                logging.exception("History save failed")
                st.warning(f"Risposta non salvata nello storico: {error}")

with tab_studio:
    st.caption(
        "Genera un contenuto dal testo completo delle fonti selezionate, "
        "non solo dai passaggi più pertinenti."
    )
    preset = st.selectbox("Cosa vuoi generare?", [*STUDIO_PRESETS, CUSTOM_PRESET])
    extra = st.text_area(
        "Istruzioni" if preset == CUSTOM_PRESET else "Istruzioni aggiuntive (facoltative)",
        placeholder="Es. concentrati sugli aspetti economici, massimo una pagina",
    ).strip()
    outputs: list[dict] = st.session_state.setdefault(f"studio:{notebook}", [])

    can_generate = bool(selected) and (preset != CUSTOM_PRESET or bool(extra))
    if st.button("Genera", type="primary", disabled=not can_generate):
        instruction = extra if preset == CUSTOM_PRESET else f"{STUDIO_PRESETS[preset]}\n{extra}".strip()
        try:
            with st.spinner("Leggo i documenti…"):
                documents, truncated = studio.load_documents(ws, cfg, selected)
            if truncated:
                st.warning(
                    "Le fonti superano il limite di testo dello Studio e sono state troncate: "
                    "seleziona meno fonti per un risultato completo."
                )
            outputs.insert(0, {"title": preset, "content": generate(ws, cfg, documents, instruction)})
            st.download_button("Scarica .md", outputs[0]["content"], file_name=f"{preset}.md", key="studio_new")
            previous = outputs[1:]
        except Exception as error:
            logging.exception("Studio failed")
            st.error(f"Errore durante la generazione: {error}")
            previous = outputs
    else:
        previous = outputs

    for i, output in enumerate(previous):
        with st.expander(output["title"], expanded=i == 0 and previous is outputs):
            st.markdown(md(output["content"]))
            st.download_button(
                "Scarica .md", output["content"], file_name=f"{output['title']}.md", key=f"studio_dl:{i}"
            )
