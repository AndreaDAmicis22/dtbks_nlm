"""Chat over a notebook's sources: a retrieval query, then a grounded answer with [n] citations."""

from __future__ import annotations

import logging

from . import llm, prompts
from .passages import Passage, format_passages, strip_citations

log = logging.getLogger(__name__)


def search_query(client, model: str, history: list[dict], question: str) -> str:
    """Standalone retrieval query: a follow-up like "e il secondo punto?" needs the history."""
    if not history:
        return question
    messages = [
        {"role": "system", "content": prompts.CONDENSE_SYSTEM},
        {"role": "user", "content": prompts.condense_user_message(history, question)},
    ]
    try:
        return llm.complete(client, model, messages, max_tokens=300) or question
    except Exception:
        # The rewrite only improves recall: fall back to the raw question.
        log.exception("Query rewrite failed")
        return question


def answer_messages(history: list[dict], question: str, passages: list[Passage]) -> list[dict]:
    messages = [{"role": "system", "content": prompts.CHAT_SYSTEM}]
    # Past answers cite passages that are no longer in context: drop their markers.
    messages += [{"role": m["role"], "content": strip_citations(m["content"])} for m in history]
    messages.append(
        {"role": "user", "content": prompts.chat_user_message(question, format_passages(passages))}
    )
    return messages
