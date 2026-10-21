"""llm.py against a local fake of the Model Serving chat completions API, through the real openai SDK."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from nlm import llm

STREAM_CHUNKS = [
    {"role": "assistant", "content": ""},
    {"content": "Ciao"},
    # Reasoning models return a list of typed parts instead of a string.
    {"content": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "..."}]}]},
    {"content": [{"type": "text", "text": " mondo [1]"}]},
]


class _Handler(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.requests.append({"path": self.path, "auth": self.headers["Authorization"], "body": body})
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta in STREAM_CHUNKS:
                event = {
                    "id": "1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": body["model"],
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            payload = {
                "id": "1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "  Domanda riscritta?  "},
                        "finish_reason": "stop",
                    }
                ],
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def serving():
    _Handler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ws = SimpleNamespace(
        config=SimpleNamespace(
            host=f"http://127.0.0.1:{server.server_port}",
            authenticate=lambda: {"Authorization": "Bearer token-123"},
        )
    )
    yield ws, _Handler.requests
    server.shutdown()
    server.server_close()


def test_stream_keeps_only_text(serving):
    ws, requests = serving
    with llm.client_for(ws) as client:
        text = "".join(llm.stream(client, "databricks-claude-opus-5-5", [{"role": "user", "content": "x"}]))
    assert text == "Ciao mondo [1]"
    assert requests[0]["path"] == "/serving-endpoints/chat/completions"
    assert requests[0]["auth"] == "Bearer token-123"
    assert requests[0]["body"]["model"] == "databricks-claude-opus-5-5"
    # No sampling parameters: recent Claude models reject them.
    assert "temperature" not in requests[0]["body"]


def test_complete_strips(serving):
    ws, _ = serving
    with llm.client_for(ws) as client:
        assert llm.complete(client, "ep", [{"role": "user", "content": "x"}]) == "Domanda riscritta?"
