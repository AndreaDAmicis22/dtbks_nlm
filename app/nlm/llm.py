"""Chat completions against Databricks Model Serving (OpenAI-compatible API).

Going through Model Serving keeps the model swappable (Claude or any other
endpoint) and governed by AI Gateway: change NLM_LLM_ENDPOINT, not the code.
"""

from __future__ import annotations

from collections.abc import Iterator

from databricks.sdk import WorkspaceClient
from openai import OpenAI


def client_for(ws: WorkspaceClient) -> OpenAI:
    """OpenAI client for Model Serving, acting as the identity of `ws`.

    Build one per request: the token is read now, and the app's OAuth tokens expire.
    (`ws.serving_endpoints.get_open_ai_client()` is deprecated.)
    """
    token = ws.config.authenticate()["Authorization"].split(" ", 1)[1]
    return OpenAI(api_key=token, base_url=f"{ws.config.host}/serving-endpoints")


def complete(client, model: str, messages: list[dict], max_tokens: int = 512) -> str:
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
    return _text(response.choices[0].message.content).strip()


def stream(client, model: str, messages: list[dict], max_tokens: int = 4096) -> Iterator[str]:
    events = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, stream=True
    )
    for event in events:
        if event.choices:
            if delta := _text(event.choices[0].delta.content):
                yield delta


def _text(content) -> str:
    """Content is a string, or a list of typed parts for reasoning models: keep the text parts."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    texts = []
    for part in content:
        part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
        if part_type == "text":
            texts.append(part.get("text", "") if isinstance(part, dict) else part.text)
    return "".join(texts)
